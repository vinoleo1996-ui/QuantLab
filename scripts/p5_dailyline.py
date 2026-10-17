#!/usr/bin/env python3
"""P5_DAILYLINE: does a daily drawdown check beat the month-end one?

The 23%/8% disaster line is evaluated at month ends only, so it is an ex-post label,
not risk control: 2018's intra-year drawdown reached -22.8% while the monthly check
never fired. This diagnostic runs the same rule on daily NAVs on the frozen G97 books
(no new stock replay, alloc layer only) and reports the cost, the MDD improvement and
the trigger counts, against the no-line baseline.

Output: output/p5_dailyline_20261017/report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alloc as R
from quantlab import p4_alloc as A

START, END = '2018-01-02', '2026-09-22'
GROSS = ROOT / 'output/p4_gross_20261013/ledger'
OUT = ROOT / 'output/p5_dailyline_20261017'
SHARE = 0.65


def g0_target(etf_returns_upto, equity_share):
    sleeve = 1.0 - equity_share
    return np.array([equity_share, 0.375 * sleeve, 0.375 * sleeve, 0.25 * sleeve])


def etf_g0(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide[['sh518880', 'sh511260', 'sh513100']].rename(
        columns={'sh518880': 'gold', 'sh511260': 'treasury', 'sh513100': 'nasdaq'})
    wide = wide.reindex(wide.index.union(sessions)).ffill().reindex(sessions)
    return wide.pct_change(fill_method=None).fillna(0.0)


def run(cost, line, check):
    rows = []
    for path in sorted(GROSS.glob(f'nav_POOL50_G97_p*_cost{cost}.csv')):
        equity = pd.read_csv(path, index_col=0, parse_dates=True)['nav'].loc[START:END]
        etf = etf_g0(equity.index)
        premium = R.qdii_premium_frame(equity.index)
        A.DISASTER_DD, A.RECOVER_DD, A.DISASTER_EQUITY = line, 0.08, 0.50
        nav, trace = A.allocate(equity, etf, SHARE, cost_mult=cost, premium=premium,
                                rebalance='quarter', drawdown_check=check,
                                overflow={'gold': 0.5, 'treasury': 0.5}, target_fn=g0_target)
        s = A.stats(nav)
        rows.append(dict(path=path.stem, cagr=float(s['cagr']), mdd=float(s['mdd']),
                         triggers=trace['disaster_triggers']))
    frame = pd.DataFrame(rows)
    return dict(n=len(frame), cagr=float(frame.cagr.median()), mdd=float(frame.mdd.median()),
                cagr_p10=float(frame.cagr.quantile(0.1)), cagr_p90=float(frame.cagr.quantile(0.9)),
                mdd_worst=float(frame.mdd.min()),
                triggers=[int(x) for x in frame.triggers], paths=rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = {}
    for cost in (1, 2):
        cells[f'no_line_c{cost}'] = run(cost, 1.0, 'rebalance')
        cells[f'monthly_23_c{cost}'] = run(cost, 0.23, 'rebalance')
        cells[f'daily_23_c{cost}'] = run(cost, 0.23, 'daily')
        for key in (f'no_line_c{cost}', f'monthly_23_c{cost}', f'daily_23_c{cost}'):
            c = cells[key]
            print(f'{key:16s} cagr {c["cagr"]*100:.2f}% mdd {c["mdd"]*100:.2f}% '
                  f'worst {c["mdd_worst"]*100:.2f}% triggers {sum(c["triggers"])}', flush=True)
    (OUT / 'report.json').write_text(json.dumps(cells, ensure_ascii=False, indent=1))
    print('wrote', OUT / 'report.json')


if __name__ == '__main__':
    main()
