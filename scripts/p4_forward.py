#!/usr/bin/env python3
"""P4 paper forward record (starts only after docs/prereg/P4_ALLOC.md passes; never trades).

  .venv2/bin/python scripts/p4_forward.py snapshot --asof 2026-10-30
  .venv2/bin/python scripts/p4_forward.py settle --to 2026-11-27

``snapshot`` freezes the 50-name equity basket (the P4_alpha equity book, refit on every label known
by ``asof``) and the ETF sleeve weights (P4_ALLOC rules) at a month end. ``settle`` books the period
from the latest snapshot to ``--to`` at closing prices with estimated costs; periods the panel does
not cover stay ``pending_data``. Output: output/p4_forward/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alloc as ALLOC_SCRIPT
from quantlab import p4_alloc as A
from quantlab.alpha import features as F
from quantlab.alpha.labels import HORIZON, forward_open_returns
from quantlab.alpha.models import GBDTEnsemble, LinearICIR, percentile_frame
from quantlab.alpha.portfolio import random_order, select
from quantlab.cloud import load_snapshot
from quantlab.p4_mandate import MANDATE
from quantlab.smallcap import universe
from quantlab.smallcap.panel import Panel, latest_version

OUT = ROOT / 'output/p4_forward'
ALPHA = ROOT / 'output/p4_alpha_20261012'
ALLOC = ROOT / 'output/p4_alloc_20261012/report.json'
EQUITY_TRADE_COST, ETF_TRADE_COST = 0.004, 0.0005
FIELDS = ('open', 'high', 'low', 'close', 'adj_close', 'amount', 'turnover', 'float_mktcap', 'tradestatus',
          'limit_up', 'is_st')


def require_pass():
    if not ALLOC.exists() or not json.loads(ALLOC.read_text()).get('passed'):
        raise SystemExit(json.dumps(dict(status='not_started', reason='P4_ALLOC has not passed')))
    return json.loads(ALLOC.read_text())


def snapshots():
    return sorted(OUT.glob('snapshot_*.json'))


def stage_snapshot(asof):
    alloc = require_pass()
    book = alloc['equity_book']
    panel = Panel(latest_version())
    asof = pd.Timestamp(asof)
    if asof not in panel.sessions:
        raise SystemExit(json.dumps(dict(status='pending_data', reason=f'panel ends {panel.sessions[-1].date()}')))
    fields = {name: panel.field(name) for name in FIELDS}
    pools = universe.rolling_features(panel, with_st=True)
    pool = [c for c in universe.pool_on(asof, pools, 'adv', band='relative', exclude_st=True, min_history=252)
            if not c.startswith('bj.')]
    sectors = F.Industries(load_snapshot('sw_l1_members'))
    industries = sectors.on(asof).reindex(pool).fillna('unclassified').to_dict()
    previous = json.loads(snapshots()[-1].read_text()) if snapshots() else None
    held = previous['equity']['names'] if previous else []
    if book == 'POOL50':
        ordered = random_order(pool, 1)
        model_note = dict(model='POOL50', seed=1)
    else:
        price = F.PriceVolume(fields)
        today = F.cross_section(asof, pool, price, F.Fundamentals(load_snapshot('financials_pit')), sectors,
                                fields['close'])
        samples = pd.read_parquet(ALPHA / 'samples.parquet')
        present = json.loads((ALPHA / 'samples_meta.json').read_text())['features']
        position = pd.Series(np.arange(len(panel.sessions)), index=panel.sessions)
        train = samples[(samples.label_known_pos <= position[asof]) & samples.label.notna()]
        model = (GBDTEnsemble() if book == 'GBDT' else LinearICIR()).fit(train, present)
        scored = percentile_frame(today, present)
        scored['score'] = model.predict(scored)
        ordered = list(scored.sort_values('score', ascending=False, kind='mergesort').code)
        model_note = dict(model=book, train_rows=len(train), last_label_known=str(panel.sessions[int(train.label_known_pos.max())].date()))
    names = select(ordered, industries, held)
    etf = ALLOC_SCRIPT.etf_returns(panel.sessions[panel.sessions <= asof])
    nav_rows = _nav_rows()
    peak = max([1.0] + [r['nav'] for r in nav_rows])
    current = nav_rows[-1]['nav'] if nav_rows else 1.0
    stressed = bool(previous and previous['stressed'])
    drawdown = 1 - current / peak
    stressed = (drawdown >= A.DISASTER_DD) if not stressed else (drawdown > A.RECOVER_DD)
    weights = A.target_weights(etf, A.DISASTER_EQUITY if stressed else 0.60)
    snapshot = dict(asof=str(asof.date()), created_at=datetime.now(timezone.utc).isoformat(), panel=latest_version(),
                    equity=dict(book=book, names=names, weight_each=1 / len(names) if names else 0.0, **model_note),
                    sleeves=dict(zip(['equity'] + list(A.ETF_CODES), [float(w) for w in weights])),
                    stressed=stressed, mandate=MANDATE.version, capital_cny=MANDATE.capital_cny,
                    promotion_approved=False, forward_decision=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'snapshot_{asof.date()}.json').write_text(json.dumps(snapshot, indent=1, ensure_ascii=False))
    print(json.dumps(dict(asof=snapshot['asof'], names=len(names), sleeves=snapshot['sleeves'], stressed=stressed)))


def _nav_rows():
    path = OUT / 'nav.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stage_settle(to):
    require_pass()
    if not snapshots():
        raise SystemExit(json.dumps(dict(status='no_snapshot')))
    snap = json.loads(snapshots()[-1].read_text())
    start, to = pd.Timestamp(snap['asof']), pd.Timestamp(to)
    panel = Panel(latest_version())
    if panel.sessions[-1] < to:
        row = dict(period=[str(start.date()), str(to.date())], status='pending_data', panel_end=str(panel.sessions[-1].date()))
        print(json.dumps(row))
        return
    window = panel.sessions[(panel.sessions > start) & (panel.sessions <= to)]
    adj = panel.field('adj_close')[snap['equity']['names']].loc[[start, *window]]
    basket = float(adj.iloc[-1].div(adj.iloc[0]).fillna(1.0).mean() - 1) if snap['equity']['names'] else 0.0
    etf = ALLOC_SCRIPT.etf_returns(pd.DatetimeIndex([start, *window]))
    etf_growth = (1 + etf.iloc[1:]).prod() - 1
    sleeves = snap['sleeves']
    previous = json.loads(snapshots()[-2].read_text()) if len(snapshots()) > 1 else None
    turnover = 1.0 if previous is None else len(set(snap['equity']['names']) - set(previous['equity']['names'])) / 50
    gross = sleeves['equity'] * basket + sum(sleeves[k] * float(etf_growth[k]) for k in A.ETF_CODES)
    cost = sleeves['equity'] * turnover * 2 * EQUITY_TRADE_COST
    rows = _nav_rows()
    nav = (rows[-1]['nav'] if rows else 1.0) * (1 + gross - cost)
    row = dict(period=[str(start.date()), str(to.date())], status='settled', equity_return=basket,
               etf_returns={k: float(etf_growth[k]) for k in A.ETF_CODES}, gross=gross, cost=cost, nav=nav,
               settled_at=datetime.now(timezone.utc).isoformat(), approximation='close_to_close_estimated_costs')
    with (OUT / 'nav.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    print(json.dumps(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('snapshot', 'settle'))
    parser.add_argument('--asof')
    parser.add_argument('--to')
    args = parser.parse_args()
    stage_snapshot(args.asof) if args.stage == 'snapshot' else stage_settle(args.to)
