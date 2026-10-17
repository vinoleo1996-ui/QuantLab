#!/usr/bin/env python3
"""P5_G0: restore the fixed 3:3:2 sleeve on the existing P4 stock books. Diagnostic, no stock replay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import os
ROOT = Path(os.environ.get('QUANTLAB_ROOT', '/Users/agiuser/Desktop/QuantLab'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alloc as R
from quantlab import p4_alloc as A

START, END = '2018-01-02', '2026-09-22'
ALPHA = ROOT / 'output/p4_alpha_20261012/ledger'
GROSS = ROOT / 'output/p4_gross_20261013/ledger'


def etf_g0(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide[['sh518880', 'sh511260', 'sh513100']].rename(
        columns={'sh518880': 'gold', 'sh511260': 'treasury', 'sh513100': 'nasdaq'})
    wide = wide.reindex(wide.index.union(sessions)).ffill().reindex(sessions)
    return wide.pct_change(fill_method=None).fillna(0.0)


ORIG_TARGET = A.target_weights


def g0_target(etf_returns_upto, equity_share):
    sleeve = 1.0 - equity_share
    return np.array([equity_share, 0.375 * sleeve, 0.375 * sleeve, 0.25 * sleeve])


def run(folder, prefix, share, line, cost, sleeve):
    files = sorted(folder.glob(f'nav_{prefix}p*_cost{cost}.csv'))
    A.target_weights = g0_target if sleeve == 'g0' else ORIG_TARGET
    out = []
    for path in files:
        equity = pd.read_csv(path, index_col=0, parse_dates=True)['nav'].loc[START:END]
        if sleeve == 'g0':
            etf = etf_g0(equity.index)
        else:
            etf = R.etf_returns(equity.index)
        premium = R.qdii_premium_frame(equity.index)
        A.DISASTER_DD, A.RECOVER_DD, A.DISASTER_EQUITY = line, 0.08, 0.50
        nav, trace = A.allocate(equity, etf, share, cost_mult=cost, premium=premium)
        s = A.stats(nav)
        out.append(dict(path=path.stem, cagr=s['cagr'], mdd=s['mdd'],
                        triggers=trace['disaster_triggers']))
    frame = pd.DataFrame(out)
    return dict(n=len(frame), cagr=float(frame.cagr.median()), mdd=float(frame.mdd.median()),
                triggers=frame.triggers.tolist())


def main():
    rows = []
    # validation: reproduce ALLOC60 with the existing inverse-vol rule and 18% line
    rows.append(('ALLOC60_invvol_60_18', run(ALPHA, 'POOL50_', 0.60, 0.18, 1, 'g1')))
    # G0
    for share in (0.60, 0.70):
        for line in (0.23, 1.0):
            for cost in (1, 2):
                rows.append((f'G0_{int(share*100)}_{"noline" if line == 1.0 else "23"}_c{cost}',
                             run(ALPHA, 'POOL50_', share, line, cost, 'g0')))
    # G97 with G0
    for share in (0.65, 0.70):
        for line in (0.23, 1.0):
            for cost in (1, 2):
                rows.append((f'G97_{int(share*100)}_{"noline" if line == 1.0 else "23"}_c{cost}',
                             run(GROSS, 'POOL50_G97_', share, line, cost, 'g0')))
    for name, r in rows:
        print(f'{name:28s} n={r["n"]:2d} cagr={r["cagr"]*100:6.2f}% +ipo={r["cagr"]*100+1.5:6.2f}% '
              f'mdd={r["mdd"]*100:6.2f}% triggers={r["triggers"]}')


if __name__ == '__main__':
    main()
