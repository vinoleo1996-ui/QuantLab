#!/usr/bin/env python3
"""P5_G: Amihud top-20% pool, 50-name 97% book, 4 phases x 2 seeds x 2 costs."""
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

import p4_alpha as P
from quantlab import p4_ledger
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel
from quantlab.alpha.portfolio import schedule, random_order

OUT = ROOT / 'output/p5_g_20261017'
START, END = '2018-01-02', '2026-09-22'


def main():
    panel = Panel(P.PANEL)
    f = U.rolling_features(panel, with_st=True)
    ret = panel.field('adj_close').pct_change(fill_method=None)
    amount = panel.field('amount')
    amihud = (ret.abs() / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    samples = pd.read_parquet(P.OUT / 'samples.parquet', columns=['code', 'industry'])
    ind_map = samples.groupby('code').industry.last().to_dict()
    sessions = panel.sessions
    rows = []
    for phase in (0, 5, 10, 15):
        grid = U.decision_grid(sessions, START, END, phase, 20)
        pools = {}
        for d in grid:
            a = f['adv'].loc[d]; h = f['history'].loc[d]; st = f['st'].loc[d]
            keep = (h >= 252) & a.notna() & (a >= 5e6) & ~st.eq(1)
            codes = [c for c in a[keep].index if not c.startswith('bj.')]
            v = amihud.loc[d].reindex(codes).dropna()
            k = max(50, int(0.2 * len(v)))
            pools[d] = list(v.nlargest(k).index)
        for seed in (1, 2):
            order = lambda day, s=seed: random_order(pools[day], s)
            ind = lambda day: {c: ind_map.get(c, c) for c in pools[day]}
            plan = schedule(list(grid), order, ind, gross=0.97)
            targets = p4_ledger.to_targets(plan)
            for cost in (1, 2):
                result = p4_ledger.replay(targets, start=START, end=END, cost_mult=cost, gross=0.97)
                nav = result['nav']
                years = (nav.index[-1] - nav.index[0]).days / 365.25
                cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
                mdd = float((nav / nav.cummax() - 1).min())
                name = f'AMIHUD20_p{phase}_s{seed}_cost{cost}'
                (OUT / 'ledger').mkdir(parents=True, exist_ok=True)
                nav.rename('nav').to_csv(OUT / 'ledger' / f'nav_{name}.csv')
                rows.append(dict(path=name, cost=cost, cagr=cagr, mdd=mdd,
                                 fills=len(result['fills']), checks=result['summary']['account_checks_passed']))
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    df = pd.DataFrame(rows)
    med = df.groupby('cost')[['cagr', 'mdd']].median()
    summary = dict(median=med.round(4).to_dict(), n=len(df),
                   gates=dict(cost1_ge_13_86=bool(med.loc[1, 'cagr'] >= 0.1386),
                              cost2_ge_12_59=bool(med.loc[2, 'cagr'] >= 0.1259),
                              mdd_ge_minus_36_6=bool(med.loc[1, 'mdd'] >= -0.366)))
    (OUT / 'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
