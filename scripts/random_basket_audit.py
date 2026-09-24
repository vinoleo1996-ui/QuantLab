#!/usr/bin/env python3
"""Qualified-data random research entry. cached_proxy is an archived forensic helper only."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantlab.config import CONFIG, OUTPUT_DIR
from quantlab.data.store import DataStore
from quantlab.data.contracts import MarketDataset, fingerprint
from quantlab.report.runs import provenance


def cached_proxy():
    """Counterfactual units explicitly retain survivor selection and incomplete data."""
    store = DataStore()
    saved = pickle.loads((OUTPUT_DIR / "results.pkl").read_bytes())["result"]
    codes = list(saved["codes"])
    spot = store.load("a", "candidates_a")
    panels, records, hashes = {}, [], {}
    for c in codes:
        raw = store.load("a", f"daily_{c}")
        hashes[c] = hashlib.sha256(store._path("a", f"daily_{c}").read_bytes()).hexdigest()
        p = raw.copy()
        unit = 100. / p.close.dropna().iloc[0]
        for k in ("open", "close", "high", "low"): p[k] *= unit
        p["available_at"] = [str(d.date()) + " 18:00:00" for d in p.index]
        p["source"] = "counterfactual_adjusted_unit_proxy:" + raw.attrs.get("source", "unknown")
        p["revision"], p["quality_status"] = "diagnostic", "ok"
        # This is an optimistic execution assumption, explicitly not observed auction status.
        p["open_status"] = np.where(p.open.gt(0), "trading", "unknown")
        p["open_status_available_at"] = [str(d.date()) + " 09:25:00" for d in p.index]
        p["reference_price"] = p.close.shift(1)
        panels[c] = p
        sector = spot.loc[c, "sector"] if spot is not None and c in spot.index and "sector" in spot else "UNKNOWN"
        records.append(dict(instrument_id=c, available_at="1900-01-01", effective_at="1900-01-01",
                            source="CURRENT_SNAPSHOT_BACKCAST_DIAGNOSTIC_ONLY", revision="diagnostic", quality_status="ok",
                            active=True, is_st=False, listing_date=str(p.index[0].date()), sector=sector,
                            asset_type="stock", buy_lot=1, sell_lot=1, sell_delay=1, price_tick=.0001,
                            commission=.00025, min_commission=5., limit_ratio=.2 if c.startswith(("300", "688")) else .1))
    calendar = pd.DatetimeIndex(sorted(set().union(*(set(p.index) for p in panels.values()))))
    # Extra calendar day is only a scheduling placeholder, never a fabricated price row.
    calendar = calendar.append(pd.DatetimeIndex([calendar[-1]+pd.offsets.BDay()]))
    close = pd.DataFrame({c: p.close for c, p in panels.items()})
    bench = (close.div(close.iloc[0]).mean(axis=1)).to_frame("close")
    manifest = dict(source="legacy_cache_diagnostic", revision="1", universe_coverage="snapshot_research",
                    actions_coverage="adjusted_proxy_only", price_basis="adjusted_proxy", as_of=str(close.index[-1]),
                    benchmark_type="diagnostic_pool_proxy", input_hashes=hashes)
    return MarketDataset(panels, pd.DataFrame(records), calendar, benchmark=bench, manifest=manifest)


def main():
    """Current entry requires qualified data; legacy proxy results remain archived."""
    from quantlab.data.qualification import audit_manifest
    from quantlab.data.contracts import load_dataset
    from quantlab.pipeline import run_suite,DEFAULT_PROTOCOL
    from quantlab.report.runs import atomic_json
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='data/research/manifest.json')
    parser.add_argument('--trials',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20260924)
    parser.add_argument('--output',default='output/random_basket_v3')
    args=parser.parse_args()
    report=audit_manifest(args.manifest)
    if report['status']!='qualified_for_daily_research':
        verdict=dict(status='blocked_data',observed=None,percentile=None,mc_p_upper=None,
            model_beats_random_at_10pct=None,trials=0,next_stage='qualified_PIT_dataset_required',
            reasons=report['reasons'],promotion_approved=False,data_version=report.get('data_version'),
            model_version=None,as_of=None)
        verdict['run_id']=fingerprint(dict(report=report,code=provenance()['code_sha256']))[:24]
        atomic_json(Path(args.output)/verdict['run_id']/'summary.json',verdict)
        print(json.dumps(verdict,ensure_ascii=False,indent=2));return
    protocol=deepcopy(DEFAULT_PROTOCOL);protocol.update(random_trials=args.trials,seed=args.seed)
    print(json.dumps(run_suite(load_dataset(args.manifest),args.output,protocol),default=str,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
