#!/usr/bin/env python3
"""Conditional random-basket diagnostic on frozen legacy caches. Never a live/PIT result."""
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
from quantlab.backtest.engine import Engine, prepare
from quantlab.report.metrics import performance_table
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.trials < 200: raise ValueError("at least 200 trials required")
    out = OUTPUT_DIR / "random_basket_20260924"
    out.mkdir(exist_ok=True)
    ds = cached_proxy()
    cfg = deepcopy(CONFIG)
    cfg.capital.rebalance_rule = "monthly"
    cfg.capital.a_share_positions = 27
    cfg.capital.target_positions = 30
    cfg.portfolio.max_sector_weight = .30
    cfg.research.max_stock_weight = 1/27+.015
    cfg.research.universe_mode = "pit_liquidity"
    cfg.factors.use_experts = True
    cfg.seed = args.seed
    contract = {"null": "repaired selection no better than fixed random priority basket under same allocator and risk policy",
                "trials": args.trials, "seed": args.seed, "basket_size": 27, "nominal_pool_size": len(ds.panels),
                "primary_metric": "calendar_cagr", "exploratory_percentile": .90,
                "one_sided_mc_p": "(1 + count(null >= observed)) / (1 + trials)",
                "gate": "cannot promote on diagnostic data; p>0.10 means this model has not beaten conditional random selection",
                "does_not_test": "absence of all possible alpha or superiority of another universe",
                "assumptions": ["survivor snapshot pool", "adjusted prices normalized to counterfactual units; not real share lots",
                                "opening tradability approximated; no PIT corporate actions", "same prior ADV eligibility for both arms",
                                "same minimum variance allocator, commissions, slippage and frozen monthly risk policy",
                                "actual eligible pool/count may be below 40/27 due missing liquidity", "retrospective exploratory test"],
                "config": asdict(cfg), "dataset": ds.manifest, "data_version": ds.version,
                "provenance": provenance()}
    target = out / "preregistered_contract.json"
    if target.exists():
        prior = json.loads(target.read_text())
        # Preserve every changed pre-run specification in a new file; do not overwrite an old hypothesis.
        if fingerprint(prior) != fingerprint(json.loads(json.dumps(contract, default=str))):
            target = out / ("preregistered_contract_" + fingerprint(contract)[:12] + ".json")
    if not target.exists(): target.write_text(json.dumps(contract, default=str, ensure_ascii=False, indent=2))
    print("contract saved before evaluating:", target, flush=True)
    p = prepare(ds, cfg)
    print("prepared", len(p["close"]), "sessions", flush=True)
    # Restrict random choices to the same factor-valid/crowding-filtered opportunity set each month.
    eligible = {}
    decisions = {}
    for d in p["close"].index:
        dec = p["service"].score(d, p["eligible"][d]) if d in p["service"].months else None
        if dec is not None and dec.status == "ok": decisions[d] = dec
        eligible[d] = list(dec.scores.index) if dec is not None and dec.status == "ok" else p["eligible"][d]
    p["eligible"] = eligible
    p["decisions"] = decisions
    p["review_override"] = list(decisions)
    engine = Engine(cfg)
    kwargs = dict(diagnostic=True, risk_mode="legacy_monthly", prepared=p)
    observed = engine.run_dataset(ds, **kwargs)
    metric = performance_table(observed["nav"], observed["daily_return"])
    print("observed diagnostic", metric, flush=True)
    (out / "observed.pkl").write_bytes(pickle.dumps(observed))
    ds.benchmark.to_parquet(out / "benchmark_proxy.parquet")
    rng = np.random.default_rng(args.seed)
    trials, daily, seen = [], {}, set()
    while len(trials)<args.trials:
        order = list(rng.permutation(sorted(ds.panels)))
        key = tuple(order)
        if key in seen: continue
        seen.add(key)
        result = engine.run_dataset(ds, strategy="random", random_order=order, basket_size=27, **kwargs)
        stats = performance_table(result["nav"], result["daily_return"])
        trials.append(dict(trial=len(trials), order=order, **stats, fills=len(result["fills"]), status=result["status"]))
        daily[str(len(trials))] = result["daily_return"]
        if len(trials)%10 == 0:
            print("completed", len(trials), flush=True)
            (out / "trials.partial.json").write_text(json.dumps(trials, indent=2))
    null = np.array([x["annual_return"] for x in trials])
    obs = metric["annual_return"]
    verdict = {"status": "diagnostic_only", "promotion_approved": False, "observed": metric,
               "trials": len(trials), "percentile": float(np.mean(null<obs)),
               "mc_p_upper": float((1+(null>=obs).sum())/(len(null)+1)),
               "null_quantiles": {str(q): float(np.quantile(null,q)) for q in (.1,.5,.9,.95)},
               "model_beats_random_at_10pct": bool(obs>np.quantile(null,.9)),
               "next_stage": "blocked_pending_PIT_and_predeclared_OOS_evidence",
               "interpretation": "conditional test of this model only; no claim that the pool lacks alpha",
               "contract": str(target), "actual_eligible_counts": pd.Series([len(x) for x in eligible.values()]).describe().to_dict()}
    (out / "summary.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
    (out / "trials.json").write_text(json.dumps(trials, indent=2))
    pd.DataFrame(daily).to_parquet(out / "null_daily_returns.parquet")
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
