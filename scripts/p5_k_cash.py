#!/usr/bin/env python3
"""P5_K: 70/30 allocation with a cash seat; equity goes to 0 at a 23% drawdown.

Phase 0 seeds, cost1 and cost2, 2018-01-02..2026-09-22. Equity leg is the existing
POOL50_G97 ledger NAV; ETF sleeve is G0 (gold 11.25 / 10y treasury 11.25 / nasdaq 7.5).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('QUANTLAB_ROOT', '/Users/agiuser/Desktop/QuantLab'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alloc as R
from quantlab import p4_alloc as A

START, END = '2018-01-02', '2026-09-22'
EQ_SHARE, SLEEVE_SHARE = 0.70, 0.30
DD_CUT, DD_RECOVER = 0.23, 0.08


def quarter_ends(sessions):
    s = pd.Series(sessions, index=sessions)
    return set(s.groupby([sessions.year, (sessions.month - 1) // 3]).max())


def etf_sleeve(sessions):
    b = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    w = b.pivot(index='date', columns='code', values='close_hfq').sort_index()
    w.index = pd.to_datetime(w.index)
    w = w[['sh518880', 'sh511260', 'sh513100']].rename(
        columns={'sh518880': 'gold', 'sh511260': 'treasury', 'sh513100': 'nasdaq'})
    return w.reindex(w.index.union(sessions)).ffill().reindex(sessions).pct_change(fill_method=None).fillna(0.0)


def run(cost):
    out = []
    for p in sorted((ROOT / 'output/p4_gross_20261013/ledger').glob('nav_POOL50_G97_p*_cost{cost}.csv'.format(cost=cost))):
        eq = pd.read_csv(p, index_col=0, parse_dates=True)['nav'].loc[START:END]
        etf = etf_sleeve(eq.index)
        eqr = eq.pct_change(fill_method=None).fillna(0.0)
        blocks = {'gold': np.zeros(len(eq.index)), 'treasury': np.zeros(len(eq.index)),
                  'nasdaq': np.zeros(len(eq.index))}
        sessions = eq.index
        month_ends = A.month_ends(sessions)
        q_ends = quarter_ends(sessions)
        w_eq, w_gold, w_t10, w_nas, w_cash = EQ_SHARE, 0.1125, 0.1125, 0.075, 0.0
        nav, peak, stressed = 1.0, 1.0, False
        rows = []
        for t, day in enumerate(sessions):
            if t > 0:
                growth = np.array([w_eq, w_gold, w_t10, w_nas, w_cash]) * \
                         (1 + np.array([eqr.iloc[t], etf.gold.iloc[t], etf.treasury.iloc[t], etf.nasdaq.iloc[t], 0.0]))
                nav *= float(growth.sum())
                w_eq, w_gold, w_t10, w_nas, w_cash = (growth / growth.sum()).tolist()
            peak = max(peak, nav)
            dd = 1 - nav / peak
            rows.append(nav)
            if day in month_ends:
                if not stressed and dd >= DD_CUT:
                    stressed = True
                elif stressed and dd <= DD_RECOVER:
                    stressed = False
                share = 0.0 if stressed else EQ_SHARE
                cost_arr = np.array([A.EQUITY_COST, A.ETF_COST, A.ETF_COST, A.ETF_COST, 0.0]) * cost
                old = np.array([w_eq, w_gold, w_t10, w_nas, w_cash])
                if stressed:
                    new = np.array([0.0] + [0.3 * f for f in (0.375, 0.375, 0.25)] + [0.7])
                elif day in q_ends:
                    new = np.array([share] + [SLEEVE_SHARE * f for f in (0.375, 0.375, 0.25)] + [0.0])
                else:
                    new = old.copy()
                    new[0] = share
                    rest = new[1:4].sum()
                    if rest > 0:
                        new[1:4] = new[1:4] / rest * SLEEVE_SHARE
                    new[4] = max(0.0, 1.0 - new[:4].sum())
                nav *= 1 - float((np.abs(new - old) * cost_arr).sum())
                w_eq, w_gold, w_t10, w_nas, w_cash = new.tolist()
        navs = pd.Series(rows, index=sessions)
        years = (navs.index[-1] - navs.index[0]).days / 365.25
        out.append(dict(path=p.stem, cagr=float((navs.iloc[-1]) ** (1 / years) - 1),
                        mdd=float((navs / navs.cummax() - 1).min()), stressed=bool(stressed)))
    df = pd.DataFrame(out)
    return df


def main():
    rows = []
    for cost in (1, 2):
        df = run(cost)
        print(cost, 'median cagr', round(df.cagr.median() * 100, 2), 'mdd', round(df.mdd.median() * 100, 2))
        rows.append(df)
    roll = None
    navs = None
    (ROOT / 'output/p5_k_20261017').mkdir(parents=True, exist_ok=True)
    rows[0].to_csv(ROOT / 'output/p5_k_20261017/results_cost1.csv', index=False)
    rows[1].to_csv(ROOT / 'output/p5_k_20261017/results_cost2.csv', index=False)
    print(json.dumps({'cost1_cagr': round(rows[0].cagr.median() * 100, 2),
                      'cost1_mdd': round(rows[0].mdd.median() * 100, 2),
                      'cost2_cagr': round(rows[1].cagr.median() * 100, 2),
                      'cost2_mdd': round(rows[1].mdd.median() * 100, 2)}, indent=1))


if __name__ == '__main__':
    main()
