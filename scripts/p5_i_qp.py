#!/usr/bin/env python3
"""P5_I diagnostic: convex rebalance on the existing POOL50 pool (phase 0, seed 1)."""
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
from scipy.optimize import minimize

import p4_alpha as P
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

OUT = ROOT / 'output/p5_i_20261017'
START, END = '2018-01-02', '2026-09-22'
GROSS, MAXW, INDCAP = 0.97, 0.04, 0.24


def solve_qp(mu, sigma, w0, cost, lam, industry):
    n = len(mu)
    idx = np.arange(n)

    def obj(w):
        d = w - w0
        return -(mu @ w - lam * (w @ sigma @ w) - cost * np.sum(d * d))

    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - GROSS}]
    inds = {}
    for i, g in enumerate(industry):
        inds.setdefault(g, []).append(i)
    for g, ix in inds.items():
        if len(ix) > 1:
            cons.append({'type': 'ineq', 'fun': lambda w, ix=np.array(ix): INDCAP - w[ix].sum()})
    x0 = w0.copy() if abs(w0.sum() - GROSS) < 1e-6 else np.full(n, GROSS / n)
    res = minimize(obj, x0, method='SLSQP', bounds=[(0, MAXW)] * n, constraints=cons,
                   options={'maxiter': 300, 'ftol': 1e-9})
    w = np.clip(res.x, 0, MAXW)
    return w / w.sum() * GROSS if w.sum() > 0 else x0


def main():
    panel = Panel(P.PANEL)
    f = U.rolling_features(panel, with_st=True)
    ret = panel.field('adj_close').pct_change(fill_method=None)
    scores = pd.read_parquet(P.OUT / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(START)]
    sessions = panel.sessions
    grid = U.decision_grid(sessions, START, END, 0, 20)
    nav, peak, rows = 1.0, 1.0, []
    w_prev = pd.Series(dtype=float)
    for i, d in enumerate(grid):
        a = f['adv'].loc[d]; h = f['history'].loc[d]; st = f['st'].loc[d]
        keep = (h >= 252) & a.notna() & (a >= 5e6) & ~st.eq(1)
        pool = [c for c in a[keep].index if not c.startswith('bj.')]
        sc = scores[scores.date == d].set_index('code')
        sc = sc.reindex(pool).dropna(subset=['GBDT'])
        cand = list(sc.nlargest(120, 'GBDT').index)
        for c in w_prev.index:
            if c not in cand and c in ret.columns:
                cand.append(c)
        sc = sc.reindex(cand)
        industry = list(sc.industry.fillna('x'))
        mu = sc.GBDT.to_numpy(dtype=float)
        mu = (mu - mu.mean()) / (mu.std() + 1e-12) * 0.02
        hist = ret.loc[:d, cand].tail(120)
        sigma = np.cov(hist.fillna(0).to_numpy().T) * 20
        sigma = 0.5 * sigma + 0.5 * np.diag(np.diag(sigma))
        w0 = w_prev.reindex(cand).fillna(0.0).to_numpy()
        w = solve_qp(mu, sigma, w0, cost=0.01, lam=1.0, industry=industry)
        w_prev = pd.Series(w, index=cand)
        nxt = grid[i + 1] if i + 1 < len(grid) else pd.Timestamp(END)
        window = sessions[(sessions > d) & (sessions <= nxt)]
        if len(window) == 0:
            continue
        block = ret.loc[window, cand].fillna(0.0)
        growth = (1 + block).cumprod().shift(1).fillna(1.0)
        nav *= float(((growth * block).sum(axis=1) / growth.sum(axis=1) + 1).prod())
        peak = max(peak, nav)
        rows.append(dict(date=str(d.date()), nav=nav, names=int((w > 1e-6).sum()),
                         turnover=float(np.abs(w - w0).sum())))
    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    cagr = nav ** (1 / years) - 1
    mdd = min(r['nav'] / max(x['nav'] for x in rows[:k + 1]) - 1 for k, r in enumerate(rows))
    out = dict(cagr=float(cagr), mdd=float(mdd), years=years,
               mean_names=float(np.mean([r['names'] for r in rows])),
               mean_turnover=float(np.mean([r['turnover'] for r in rows])),
               note='phase 0, seed 1, 20-day rebalance, frictionless')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'diag.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    pd.DataFrame(rows).to_csv(OUT / 'path.csv', index=False)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
