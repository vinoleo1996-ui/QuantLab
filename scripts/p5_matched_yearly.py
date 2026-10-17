#!/usr/bin/env python3
"""Yearly decomposition of the P5_MATCHED difference (GBDT_HYST minus random-lag twin).

The +1.85pp / NW t=2.23 headline is a pooled monthly mean; the known failure mode of this
repo's alpha claims is single-year dependence (2018 carried 63% of GBDT_HYST's excess over
POOL50). This prints, per phase and year, the log-return difference, and the share of the
total log difference contributed by the largest year.

Reads: output/p5_matched_20261017/nav_RAND_HYST_p*_c1.csv and the frozen HYST books.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/Users/agiuser/Desktop/QuantLab')
RAND = ROOT / 'output/p5_matched_20261017'
HYST = ROOT / 'output/p4_lowturn_20261013/ledger'


def yearly(nav):
    nav = nav / nav.iloc[0]
    out = {}
    for year, block in nav.groupby(nav.index.year):
        if len(block) < 20:
            continue
        out[int(year)] = float(block.iloc[-1] / block.iloc[0] - 1)
    return out


rows = []
for path in sorted(RAND.glob('nav_RAND_HYST_p*_c1.csv')):
    name = path.stem.replace('nav_', '').replace('_c1', '')
    phase = name.split('_p')[1].split('_')[0]
    hyst_path = HYST / f'nav_GBDT_HYST_p{phase}_cost1.csv'
    if not hyst_path.exists():
        continue
    r = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    h = pd.read_csv(hyst_path, index_col=0, parse_dates=True)['nav']
    ry, hy = yearly(r), yearly(h)
    for year in sorted(set(ry) & set(hy)):
        rows.append(dict(path=name, year=year, rand=ry[year], hyst=hy[year],
                         diff=hy[year] - ry[year]))
frame = pd.DataFrame(rows)
frame.to_csv(RAND / 'yearly_diff_c1.csv', index=False)
pivot = frame.pivot(index='year', columns='path', values='diff')
print(pivot.round(3).to_string())
mean_by_year = pivot.mean(axis=1)
print('\nmean diff by year:')
print(mean_by_year.round(3).to_string())
total = mean_by_year.sum()
print(f'\nsum of yearly mean diffs: {total*100:.2f}pp')
if total != 0:
    print(f'largest year share: {mean_by_year.abs().max()/mean_by_year.abs().sum()*100:.1f}% '
          f'({mean_by_year.abs().idxmax()})')
print('positive years:', int((mean_by_year > 0).sum()), '/', len(mean_by_year))
