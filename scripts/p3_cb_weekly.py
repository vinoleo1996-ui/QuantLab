#!/usr/bin/env python3
"""Weekly-rebalanced convertible double-low (docs/prereg/P3_CB_WEEKLY.md).

  python scripts/p3_cb_weekly.py

Same rule as scripts/p2_cb_double_low.py with week-end signals. Writes output/p3_cb_weekly_20261008/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd

import p2_cb_double_low as CB
import p2_portfolio as P
from quantlab import research_trials as T
from quantlab.broad import month_ends
from quantlab.etf_account import replay, stats

PREREG = 'docs/prereg/P3_CB_WEEKLY.md'
OUT = ROOT / 'output/p3_cb_weekly_20261008'
MONTHLY = {'cagr': 0.10929079077634829, 'mdd': -0.29626022416496156, 'cost2_cagr': 0.1010085415177977}


def week_ends(sessions, start, end):
    s = pd.Series(sessions, index=sessions).loc[start:end]
    return list(s.groupby([s.index.isocalendar().year, s.index.isocalendar().week]).max())


def main():
    CB.month_ends = week_ends
    terms, bars, value, index, last, kind = CB.load()
    sessions = pd.DatetimeIndex(sorted(bars.date.unique()))
    sessions = sessions[sessions <= CB.DATA_END]
    tg, counts = CB.targets(bars, value, terms, last, kind, sessions)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, cost, recovery in (('main', 1.0, 0.3), ('cost2', 2.0, 0.3), ('recovery0', 1.0, 0.0), ('recovery100', 1.0, 1.0)):
        result = replay(CB.account_bars(bars, sessions, last, kind, recovery), tg, CB.START, CB.END,
                        cost_mult=cost, lot=10)
        result['nav'].to_csv(OUT / f'nav_{name}.csv')
        runs[name] = result
    nav = runs['main']['nav']
    m, c2 = stats(nav), stats(runs['cost2']['nav'])
    bench = index.loc[CB.START:CB.END]

    psessions = P.nav_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{P.PATHS[0]}_cost1.csv').loc[P.START:P.END].index
    signal_days = set(month_ends(psessions, P.START, P.END))
    cb_index = pd.read_parquet(ROOT / 'data/cb/index_000832.parquet').set_index('date').close
    c_ret = 0.5 * P.align(cb_index, psessions).pct_change() + 0.5 * P.align(P.AU.bench_series('H20269'), psessions).pct_change()
    c_leg = (1 + c_ret.fillna(0.0)).cumprod()
    rows = []
    for path in P.PATHS:
        m2 = P.align(P.nav_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{path}_cost1.csv'), psessions)
        out, _ = P.risk_parity(pd.DataFrame({'m2': m2, 'cb': P.align(nav, psessions), 'c': c_leg}), signal_days)
        rows.append(P.stats(out))
    port = pd.DataFrame(rows, columns=['cagr', 'mdd']).median()
    gate = {'cagr>=monthly+2pp': m['cagr'] >= MONTHLY['cagr'] + 0.02,
            'mdd_not_worse': m['max_drawdown'] >= MONTHLY['mdd'],
            'cost2>=monthly_cost2+1pp': c2['cagr'] >= MONTHLY['cost2_cagr'] + 0.01}
    gate['pass'] = all(gate.values())
    report = {'main': m, 'cost2': c2, 'monthly_reference': MONTHLY, 'gate': gate,
              'benchmark_000832': stats(bench / bench.iloc[0]), 'yearly': CB.yearly(nav),
              'trades': runs['main']['trades'], 'fees': runs['main']['fees'],
              'sensitivity': {k: stats(runs[k]['nav']) for k in ('recovery0', 'recovery100')},
              'rp3_with_weekly_median': port.to_dict(), 'signals': len(tg),
              'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG), 'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ('main', 'cost2', 'gate', 'rp3_with_weekly_median', 'trades', 'fees')},
                     default=float))


if __name__ == '__main__':
    main()
