#!/usr/bin/env python3
"""Ranking check inside the excluded high-turnover 60 percent (docs/prereg/P4_HOT_SLEEVE.md). No account."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantlab import research_trials as T
from quantlab.alpha.labels import forward_open_returns
from quantlab.research_statistics import newey_west
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

PREREG = 'docs/prereg/P4_HOT_SLEEVE.md'
PANEL = '20261006T152412Z'
OUT = ROOT / 'output/p4_hot_sleeve_20261013'
START, END, SPLIT = '2018-01-02', '2026-09-22', '2022-01-01'
TOP, ROUND_TRIP, PERIODS_PER_YEAR = 50, 0.007, 12.6


def groups(day, features):
    adv, history = features['adv'].loc[day], features['history'].loc[day]
    st = features['st'].loc[day].reindex(adv.index)
    eligible = adv[(history >= 252) & adv.notna() & ~st.eq(1)]
    eligible = eligible[~eligible.index.str.startswith('bj.')]
    ordered = eligible.sort_values(ascending=False, kind='mergesort')
    cut = math.ceil(0.6 * len(ordered))
    pool = [c for c in ordered.index[cut:] if ordered[c] >= U.MIN_ADV]
    return list(ordered.index[:cut]), pool


def t_stat(series):
    _, t = newey_west(series.to_numpy(), lags=2)
    return float(t[0])


def main():
    panel = Panel(PANEL)
    fields = {n: panel.field(n) for n in ('open', 'close', 'adj_close', 'tradestatus', 'limit_up')}
    features = U.rolling_features(panel, with_st=True)
    adj = fields['adj_close']
    ret = adj.pct_change(fill_method=None)
    mom = adj.shift(20) / adj.shift(120) - 1
    high = adj / adj.rolling(250, min_periods=120).max()
    vol = ret.rolling(60, min_periods=45).std()
    label = forward_open_returns(fields)
    sessions = panel.sessions
    grid = U.decision_grid(sessions, START, END, 0, 20)
    grid = [d for d in grid if sessions.get_loc(d) + 21 < len(sessions)]
    rows, previous = [], set()
    for day in grid:
        hot, pool = groups(day, features)
        frame = pd.DataFrame({'mom': mom.loc[day].reindex(hot), 'high': high.loc[day].reindex(hot),
                              'vol': vol.loc[day].reindex(hot), 'y': label.loc[day].reindex(hot)})
        score = (frame['mom'].rank(pct=True) + frame['high'].rank(pct=True) + (-frame['vol']).rank(pct=True)) / 3
        frame['score'] = score
        labeled = frame.dropna(subset=['score', 'y'])
        top = list(frame['score'].dropna().sort_values(ascending=False, kind='mergesort').index[:TOP])
        top_y = labeled.loc[labeled.index.intersection(top), 'y']
        turnover = 1.0 if not previous else len(set(top) - previous) / TOP
        previous = set(top)
        pool_y = label.loc[day].reindex(pool).dropna()
        rows.append(dict(date=day, ic=labeled['score'].corr(labeled['y'], method='spearman'),
                         top=top_y.mean(), group=labeled['y'].mean(), pool=pool_y.mean(), turnover=turnover,
                         n_hot=len(hot), n_top_labeled=len(top_y)))
    frame = pd.DataFrame(rows).set_index('date')
    frame['excess'] = frame['top'] - frame['group']
    frame['net_excess'] = frame['excess'] - frame['turnover'] * ROUND_TRIP
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'periods.csv')
    segments = {'screen': frame.loc[:pd.Timestamp(SPLIT) - pd.Timedelta(days=1)], 'holdout': frame.loc[SPLIT:]}
    stats = {}
    for name, seg in segments.items():
        stats[name] = dict(periods=len(seg), mean_ic=float(seg['ic'].mean()), ic_t=t_stat(seg['ic'].dropna()),
                           gross_excess_ann=float(seg['excess'].mean() * PERIODS_PER_YEAR),
                           net_excess_ann=float(seg['net_excess'].mean() * PERIODS_PER_YEAR),
                           mean_turnover=float(seg['turnover'].mean()),
                           corr_top_vs_pool=float(seg['top'].corr(seg['pool'])))
    screen, hold = stats['screen'], stats['holdout']
    sign = np.sign(screen['mean_ic'])
    gates = {'screen_ic_t_ge_3': abs(screen['ic_t']) >= 3,
             'holdout_same_sign_t_ge_2': np.sign(hold['mean_ic']) == sign and sign * hold['ic_t'] >= 2,
             'net_excess_ge_5pp_both': screen['net_excess_ann'] >= 0.05 and hold['net_excess_ann'] >= 0.05,
             'corr_with_pool_lt_0_6': float(frame['top'].corr(frame['pool'])) < 0.6}
    report = dict(stats=stats, corr_top_vs_pool_full=float(frame['top'].corr(frame['pool'])), gates=gates,
                  passed=all(gates.values()), prereg=PREREG, prereg_sha256=T.sha256_file(ROOT / PREREG),
                  promotion_approved=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
