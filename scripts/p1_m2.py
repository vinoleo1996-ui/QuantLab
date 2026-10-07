#!/usr/bin/env python3
"""P1-M2 small-cap control basket (docs/prereg/P1_M2_SMALLCAP_CONTROL.md).

Stages:
  prep    snapshots under the M2 pool (relative band, ST removed before ranking, 252 history)
  sim     fast simulator, 200 seeds x 5 phases (report only)
  ledger  stable_book.replay_account on phases {0,4,8,12,16} x seeds {1,2}, cost 1.0 and 2.0
Use --window confirm for the one-time 2015-07..2017-12 run.
"""
from __future__ import annotations

import argparse
import functools
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import smallcap_audit as AU
import smallcap_layers as L
from quantlab.research_statistics import win_rates
from quantlab.smallcap import universe
from quantlab.smallcap.sim import metrics, simulate

OUT = ROOT / 'output/p1_m2_20261008'
WINDOWS = {'is': ('2018-01-02', '2026-09-22'), 'confirm': ('2015-07-01', '2017-12-29')}
LEDGER_PATHS = [(phase, seed) for phase in L.PHASES for seed in (1, 2)]

_rolling = universe.rolling_features
M2_POOL = functools.partial(universe.pool_on, band='relative', exclude_st=True, min_history=252)


def use(window):
    universe.pool_on = M2_POOL
    universe.rolling_features = functools.partial(_rolling, with_st=True)
    L.START, L.END = WINDOWS[window]
    if window == 'confirm':
        L.SEGMENTS = AU.OOS_SEGMENTS
    L.OUT = OUT / window
    L.OUT.mkdir(parents=True, exist_ok=True)
    return L.OUT


def schedule_for(phase, seed):
    snapshot = L.snapshots_for(L._STATE['snaps'], phase, 'adv', False, None)
    return universe.build_schedule(snapshot, seed, key_names=('tier',))


def to_targets(schedule):
    """``sell`` lists every name ever targeted that is not targeted now: a name that failed to sell
    (limit-down, suspension) is otherwise share-locked by ``execution_weights`` forever."""
    rows, ever = [], set()
    for day in sorted(schedule):
        weights = {AU_code(code): w for code, w in schedule[day].items()}
        rows.append({'signal_date': str(pd.Timestamp(day).date()), 'weights': weights, 'active': list(weights),
                     'sell': sorted(ever - set(weights)), 'sleeve': 0})
        ever |= set(weights)
    return rows


def AU_code(code):
    market, number = code.split('.')
    return f'{number}.{market.upper()}'


def benchmark(index):
    level = AU.bench_series('932000CNY010').reindex(index).ffill()
    return level.pct_change(fill_method=None).fillna(0.0)


def ledger(window, cost_mult):
    import stable_book as book
    from quantlab.data.contracts import MarketDataset
    folder = use(window)
    L.load_state()
    start, end = WINDOWS[window]
    plans = {f'{p}_{s}': to_targets(schedule_for(p, s)) for p, s in LEDGER_PATHS}
    root = book.MANIFEST.parent
    manifest = json.loads(book.MANIFEST.read_text())
    codes = sorted({c for rows in plans.values() for r in rows for c in list(r['weights']) + r['sell']})
    read = lambda entry: pd.read_parquet(io.BytesIO(book._read_checked(root, entry)))
    calendar = pd.DatetimeIndex(book._dated(read(manifest['calendar'])).session)
    panels = {book.canon(c): book._dated(read(manifest['panels'][book.canon(c)]))
              for c in codes if book.canon(c) in manifest['panels']}
    dataset = MarketDataset(panels, read(manifest['instruments']), calendar, read(manifest['actions']),
                            book._dated(read(manifest['benchmark'])), manifest=manifest)
    rows = []
    for name, targets in plans.items():
        result = book.replay_account(dataset, targets, start=start, end=end, cost_mult=cost_mult)
        nav = result['nav']
        sim = simulate(L._STATE['market'], schedule_for(*map(int, name.split('_'))))['nav']
        s = result['summary']
        rates = win_rates(result['daily_return'].fillna(0.0), benchmark(nav.index))
        rows.append({'path': name, 'cost_mult': cost_mult, 'ledger_cagr': s['cagr'], 'ledger_mdd': s['max_drawdown'],
                     'sim_cagr': metrics(sim)['cagr'], 'sim_mdd': metrics(sim)['max_drawdown'],
                     'fees': s['fees'], 'slippage': s['slippage'], 'fills': s['fills'],
                     'checks_passed': s['account_checks_passed'], 'missing_panels': len(s['missing_panels']),
                     **rates})
        nav.to_frame('nav').to_csv(folder / f'ledger_nav_{name}_cost{cost_mult:g}.csv')
        print(json.dumps(rows[-1]), flush=True)
    frame = pd.DataFrame(rows)
    frame['gap_pp'] = (frame.ledger_cagr - frame.sim_cagr) * 100
    frame.to_csv(folder / f'ledger_cost{cost_mult:g}.csv', index=False)
    summary = {'window': window, 'cost_mult': cost_mult, 'paths': len(frame),
               'median': frame.drop(columns='path').median(numeric_only=True).to_dict(), 'promotion_approved': False}
    (folder / f'ledger_cost{cost_mult:g}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prep', 'sim', 'ledger'))
    parser.add_argument('--window', default='is', choices=tuple(WINDOWS))
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    if args.stage == 'prep':
        use(args.window)
        L.prep()
    elif args.stage == 'sim':
        folder = use(args.window)
        L.load_state()
        AU.run_family('A_adv', range(1, 201), {}, folder / 'equity' / 'M2')
    else:
        ledger(args.window, args.cost)
