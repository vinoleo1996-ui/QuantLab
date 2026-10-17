#!/usr/bin/env python3
"""Correct NW t for P5_MATCHED: average the 8 paths per month first, then test.

The scripted comparison pooled 8 paths x 104 months = 832 observations, but the paths are
the same months with different baskets, so the pooled SE is understated. This computes the
monthly mean difference across paths (one series of 104 points) and runs NW t on it, plus
the per-phase t to show the spread.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/Users/agiuser/Desktop/QuantLab')
RAND = ROOT / 'output/p5_matched_20261017'
HYST = ROOT / 'output/p4_lowturn_20261013/ledger'


def nw_t(x, lag=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float('nan')
    mean = x.mean()
    var = np.var(x, ddof=0)
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        cov = np.cov(x[k:], x[:-k])[0, 1]
        var += 2 * w * cov
    return float(mean / np.sqrt(max(var, 1e-18) / n))


def main():
    rows = {}
    for path in sorted(RAND.glob('nav_RAND_HYST_p*_c1.csv')):
        name = path.stem.replace('nav_', '').replace('_c1', '')
        phase = name.split('_p')[1].split('_')[0]
        h = pd.read_csv(HYST / f'nav_GBDT_HYST_p{phase}_cost1.csv', index_col=0, parse_dates=True)['nav']
        r = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        hm = (h / h.iloc[0]).resample('ME').last().pct_change()
        rm = (r / r.iloc[0]).resample('ME').last().pct_change()
        rows[name] = (hm - rm).dropna()

    frame = pd.DataFrame(rows)
    per_path = {c: (frame[c].mean() * 12, nw_t(frame[c])) for c in frame}
    avg = frame.mean(axis=1)
    print('per-path annualised diff / NW t:')
    for c, (a, t) in sorted(per_path.items()):
        print(f'  {c:20s} {a*100:+.2f}pp  t={t:.2f}')
    print(f'\nmonthly cross-path mean: {avg.mean()*12*100:+.2f}pp/yr  NW t={nw_t(avg):.2f}  n={len(avg)}')
    print(f'path-level mean of annualised diff: {np.mean([v[0] for v in per_path.values()])*100:+.2f}pp')
    print(f'path-level t range: {min(v[1] for v in per_path.values()):.2f} .. {max(v[1] for v in per_path.values()):.2f}')
    print(f'positive years (cross-path mean): ', end='')
    yearly = avg.groupby(avg.index.year).apply(lambda s: (1 + s).prod() - 1)
    print(int((yearly > 0).sum()), '/', len(yearly), '->', (yearly * 100).round(1).to_dict())


if __name__ == '__main__':
    main()
