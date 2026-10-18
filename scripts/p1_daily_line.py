#!/usr/bin/env python3
"""P1-1 diagnostic: daily vs rebalance-date 23% disaster-line checks on frozen G97_65 paths.

No new strategy trial: this only asks whether checking the same 23%/8% rule every
session changes triggers, turnover, CAGR or MDD relative to the pre-registered
quarterly check. Output: output/p1_daily_line_20261017/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import p4_alloc as R
import p5_g0_alloc as G0
from quantlab import p4_alloc as A

START, END = '2018-01-02', '2026-09-22'
SHARE, LINE, COST = 0.65, 0.23, (1, 2)
LEDGER = ROOT / 'output/p4_gross_20261013/ledger'
OUT = ROOT / 'output/p1_daily_line_20261017'


def one(equity, mode, cost):
    etf = G0.etf_g0(equity.index)
    premium = R.qdii_premium_frame(equity.index, basis='legacy')
    A.DISASTER_DD, A.RECOVER_DD, A.DISASTER_EQUITY = LINE, 0.08, 0.50
    nav, trace = A.allocate(equity, etf, SHARE, cost_mult=cost, premium=premium,
                            rebalance='quarter', drawdown_check=mode,
                            overflow={'gold': 0.5, 'treasury': 0.5}, target_fn=G0.g0_target)
    stats = A.stats(nav)
    weights = trace['weights']
    turnover = float(weights.diff().abs().sum(axis=1).fillna(0.0).sum())
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return dict(cagr=stats['cagr'], mdd=stats['mdd'], triggers=trace['disaster_triggers'],
                trigger_dates=trace['trigger_dates'], recovery_dates=trace['recovery_dates'],
                qdii_pauses=trace['qdii_pauses'], turnover=turnover,
                turnover_per_year=turnover / years, nav=nav)


def main():
    frames = {}
    rows = []
    for cost in COST:
        files = sorted(LEDGER.glob(f'nav_POOL50_G97_p*_cost{cost}.csv'))
        for path in files:
            equity = pd.read_csv(path, index_col=0, parse_dates=True)['nav'].loc[START:END]
            result = {}
            for mode in ('rebalance', 'daily'):
                result[mode] = one(equity, mode, cost)
                rows.append(dict(path=path.stem, cost=cost, mode=mode,
                                 cagr=result[mode]['cagr'], mdd=result[mode]['mdd'],
                                 triggers=result[mode]['triggers'],
                                 trigger_dates=';'.join(result[mode]['trigger_dates']),
                                 recovery_dates=';'.join(result[mode]['recovery_dates']),
                                 qdii_pauses=result[mode]['qdii_pauses'],
                                 turnover=result[mode]['turnover'],
                                 turnover_per_year=result[mode]['turnover_per_year']))
            frames[(path.stem, cost)] = result
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'paths.csv', index=False)

    summary = {}
    for cost in COST:
        for mode in ('rebalance', 'daily'):
            sub = frame[(frame.cost == cost) & (frame['mode'] == mode)]
            summary[f'cost{cost}_{mode}'] = {
                'paths': int(len(sub)),
                'cagr_median': float(sub.cagr.median()),
                'mdd_median': float(sub.mdd.median()),
                'paths_with_trigger': int((sub.triggers > 0).sum()),
                'trigger_total': int(sub.triggers.sum()),
                'trigger_dates': sorted({d for text in sub.trigger_dates if text for d in text.split(';')}),
                'turnover_per_year_median': float(sub.turnover_per_year.median()),
            }
    pairs = []
    for (name, cost), result in frames.items():
        delta = result['daily']['cagr'] - result['rebalance']['cagr']
        pairs.append(dict(path=name, cost=cost,
                          cagr_delta=float(delta),
                          mdd_delta=float(result['daily']['mdd'] - result['rebalance']['mdd']),
                          turnover_delta=float(result['daily']['turnover'] - result['rebalance']['turnover']),
                          trigger_delta=int(result['daily']['triggers'] - result['rebalance']['triggers'])))
    paired = pd.DataFrame(pairs)
    paired.to_csv(OUT / 'paired.csv', index=False)
    report = {
        'diagnostic': 'P1-1_daily_disaster_line',
        'window': [START, END], 'equity_share': SHARE, 'line': LINE,
        'rebalance': 'quarter', 'premium_basis': 'legacy',
        'note': 'Diagnostic only; the pre-registered decision remains the quarterly check. '
                'No new trial is registered and no rule is changed.',
        'summary': summary,
        'paired_median': {column: float(paired[column].median())
                          for column in ('cagr_delta', 'mdd_delta', 'turnover_delta', 'trigger_delta')},
        'promotion_approved': False,
    }
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
