#!/usr/bin/env python3
"""Diagnostic: CB double-low weekly/monthly returns and excess by segment.

Benchmarks: CSI convertible index 000832 and an equal-weight eligible-CB universe (daily rebalanced,
no cost, defaults not marked down after the last bar, so it is a generous benchmark).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import p2_cb_double_low as CB

OUT = ROOT / 'output/p3_cb_segments_20261008'
SEGMENTS = ([('2018H1', '2018-01-02', '2018-06-29'), ('2018H2', '2018-07-02', '2018-12-28')]
            + [(str(y), f'{y}-01-01', f'{y}-12-31') for y in range(2019, 2022)]
            + [('2022-01..07', '2022-01-01', '2022-07-29'), ('2022-08..12', '2022-08-01', '2022-12-31')]
            + [(str(y), f'{y}-01-01', f'{y}-12-31') for y in range(2023, 2027)]
            + [('before_rule 2018-01..2022-07', '2018-01-02', '2022-07-29'),
               ('after_rule 2022-08..2026-09', '2022-08-01', '2026-09-30')])


def equal_weight(bars):
    close = bars.pivot(index='date', columns='code', values='close').sort_index()
    amount = (bars.close * bars.volume).groupby([bars.date, bars.code]).sum().unstack().reindex(close.index)
    eligible = close.notna() & (close.notna().cumsum() > 20) & (amount.rolling(20, min_periods=20).mean() >= 5e6)
    ret = close.pct_change(fill_method=None).where(eligible.shift(1, fill_value=False))
    return (1 + ret.mean(axis=1).fillna(0)).cumprod()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, bars, _, index, _, _ = CB.load()
    series = {'weekly': pd.read_csv(ROOT / 'output/p3_cb_weekly_20261008/nav_main.csv', index_col=0, parse_dates=True).nav,
              'monthly': pd.read_csv(ROOT / 'output/p2_cb_double_low_20261008/nav_main.csv', index_col=0, parse_dates=True).nav,
              'index_000832': index, 'universe_ew': equal_weight(bars)}
    rows = []
    for name, a, b in SEGMENTS:
        row = {'segment': name}
        for key, s in series.items():
            s = s.loc[a:b].dropna()
            years = (s.index[-1] - s.index[0]).days / 365.25
            total = s.iloc[-1] / s.iloc[0] - 1
            row[key] = round(((1 + total) ** (1 / years) - 1 if years > 1.5 else total) * 100, 1)
        row['weekly_minus_index'] = round(row['weekly'] - row['index_000832'], 1)
        row['weekly_minus_ew'] = round(row['weekly'] - row['universe_ew'], 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'segments.csv', index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
