#!/usr/bin/env python3
"""S1 zero-account attribution (plan 完备验证方案 v2, 2026-10-18).

Diagnostic only, no ledger, no trial: decides advance/kill before any 3.5h account work.
  S1.1 ADV-bottom-10% vs Amihud-top-20% / ADV-bottom-40% (current band): pool overlap + path correlation.
  S1.2 Amihud20 volatility tilt: period |ret| exposure vs eligible market, intercept after tilt regression.
  S1.3 turnover-ranked bottom-40% band vs the current ADV relative band, pool-level paths.
Universe-scan method (RESEARCH_CONTINUITY 2026-10-17 18:50): non-overlapping 20-session open-to-open
equal-weight, no cost. Eligibility as in scripts/p5_g_amihud.py (history>=252, ADV>=5m, minus ST, minus bj.).
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

import p4_alpha as P
from quantlab.research_statistics import newey_west
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

START, END = '2018-01-02', '2026-09-22'
OUT = ROOT / 'output/attribution_s1_20261018'


def eligible_day(f, day):
    adv = f['adv'].loc[day]
    hist = f['history'].loc[day]
    st = f['st'].loc[day]
    keep = (hist >= 252) & adv.notna() & (adv >= 5e6) & ~st.eq(1)
    return [c for c in adv[keep].index if not c.startswith('bj.')]


def build_pools(panel, f, amihud, turn20, grid):
    pools = {name: {} for name in ('adv10', 'amihud20', 'turn20low40')}
    for d in list(grid) + [grid[-1]]:
        codes = eligible_day(f, day=d)
        adv = f['adv'].loc[d].reindex(codes)
        ordered = adv.sort_values(ascending=False, kind='mergesort')
        n = len(ordered)
        # 'ADV 后 10%' = the least-liquid tail of the eligible list (ordered is descending).
        pools['adv10'][d] = list(ordered.index[n - max(50, int(np.ceil(0.10 * n))):])
        am = amihud.loc[d].reindex(codes).dropna()
        pools['amihud20'][d] = list(am.nlargest(max(50, int(np.ceil(0.20 * len(am))))).index)
        tr = turn20.loc[d].reindex(codes).dropna()
        tr_ordered = tr.sort_values(ascending=False, kind='mergesort')
        pools['turn20low40'][d] = list(
            tr_ordered.index[max(0, len(tr_ordered) - max(50, int(np.ceil(0.40 * len(tr_ordered))))):])
    return pools


def period_paths(panel, pools, grid):
    """Non-overlapping 20-session open-to-open EW return per pool per grid day.

    Open is adjusted with the labels.py convention: adj_open = open * adj_close / close."""
    open_ = panel.field('open') * panel.field('adj_close') / panel.field('close')
    pos = pd.DatetimeIndex(panel.sessions)
    paths = {name: pd.Series(dtype=float) for name in pools}
    tilt = {name: pd.Series(dtype=float) for name in pools}
    ret_d1 = panel.field('adj_close').pct_change(fill_method=None)
    for i, d in enumerate(grid):
        j = i + 1
        if j >= len(grid):
            break
        o0, o1 = open_.loc[d], open_.loc[grid[j]]
        window = ret_d1.loc[(ret_d1.index > d) & (ret_d1.index <= grid[j])]
        for name, byday in pools.items():
            codes = byday[d]
            nxt = byday.get(grid[j])
            if nxt is None:
                continue
            codes = [c for c in codes if c in nxt]  # held across the whole period
            if len(codes) < 10:
                continue
            r = (o1.reindex(codes) / o0.reindex(codes) - 1).dropna()
            if len(r) >= 10:
                paths[name][grid[j]] = float(r.mean())
            a = window.reindex(columns=codes)
            tilt[name][grid[j]] = float(np.nanmean(np.abs(a.to_numpy())))
    return paths, tilt


def cagr(series):
    if series.empty:
        return None
    total = float(np.prod(1.0 + series.to_numpy()))
    years = (series.index[-1] - series.index[0]).days / 365.25
    return total ** (1 / years) - 1 if years > 0 and total > 0 else None


def overlap(a, b, grid):
    out = []
    for d in grid:
        sa, sb = set(a[d]), set(b[d])
        if sa:
            out.append(len(sa & sb) / len(sa))
    return float(np.median(out)) if out else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = Panel(P.PANEL)
    f = U.rolling_features(panel, with_st=True)
    ret = panel.field('adj_close').pct_change(fill_method=None)
    amount = panel.field('amount').replace(0, np.nan)
    amihud = (ret.abs() / amount).rolling(20, min_periods=10).mean()
    turn20 = panel.field('turnover').rolling(20, min_periods=15).mean()
    sessions = pd.DatetimeIndex(panel.sessions)
    grid = sessions[(sessions >= pd.Timestamp(START)) & (sessions <= pd.Timestamp(END))][0::20]
    pools = build_pools(panel, f, amihud, turn20, grid)

    # market tilt baseline: eligible universe average |ret| per period
    market = {}
    open_ = panel.field('open')
    for i, d in enumerate(grid[:-1]):
        codes = eligible_day(f, day=d)
        window = ret.loc[(ret.index > d) & (ret.index <= grid[i + 1])]
        market[grid[i + 1]] = float(np.nanmean(np.abs(window.reindex(columns=codes).to_numpy())))
    mkt = pd.Series(market)

    pools_ew = {'adv40low': {d: U.pool_on(d, f, 'adv', band='relative') for d in list(grid) + [grid[-1]]}}
    pools_all = dict(pools)
    pools_all.update(pools_ew)
    # no-floor variants: the phantom scan's likely universe (all listed names, no ADV/ST filter,
    # entry-day membership held to period end) -- tests whether 22.74%/21.89% is reproducible at all.
    adv_row = f['adv']
    hist_row = f['history']
    nofloor = {'adv10_nofloor': {}, 'amihud20_nofloor': {}}
    for d_i in range(len(grid) - 1):
        d = grid[d_i]
        adv = adv_row.loc[d]
        hist = hist_row.loc[d]
        base_keep = (hist >= 252) & adv.notna()
        names = [c for c in adv[base_keep].index if not c.startswith('bj.')]
        ordered = adv.loc[names].sort_values(ascending=False, kind='mergesort')
        n = len(ordered)
        tail10 = list(ordered.index[n - max(50, int(np.ceil(0.10 * n))):])
        keep_nb = (hist >= 252) & adv.notna() & (adv >= 5e6)
        am = amihud.loc[d].reindex(adv[keep_nb].dropna().index).dropna()
        top_am = list(am.nlargest(max(50, int(np.ceil(0.20 * len(am))))).index)
        # refreshed every grid day, like the ledger pools; the last grid day reuses the prior draw
        nofloor['adv10_nofloor'][d] = tail10
        nofloor['amihud20_nofloor'][d] = top_am
    last = grid[-1]
    nofloor['adv10_nofloor'][last] = nofloor['adv10_nofloor'][grid[-2]]
    nofloor['amihud20_nofloor'][last] = nofloor['amihud20_nofloor'][grid[-2]]
    pools_all.update(nofloor)
    paths, tilts = period_paths(panel, pools_all, grid)
    base = pd.concat([paths['adv10'], paths['amihud20']], axis=1).dropna()

    result = {
        'method': 'non-overlapping 20-session open-to-open EW, no cost, eligible-minus-ST/bj, 2018-01-02..2026-09-22',
        'cagr': {name: cagr(s) for name, s in paths.items()},
        's1_1': {
            'overlap_adv10_vs_amihud20': overlap(pools['adv10'], pools['amihud20'], grid),
            'overlap_adv10_vs_adv40low': overlap(pools['adv10'], pools_ew['adv40low'], grid),
            'corr_adv10_amihud20_period_ret': float(np.corrcoef(base.iloc[:, 0], base.iloc[:, 1])[0, 1]),
        },
        's1_2': {},
        'mean_abs_ret_tilt': {name: float(np.nanmean(t.to_numpy())) for name, t in tilts.items()},
        'market_abs_ret': float(np.nanmean(mkt.to_numpy())),
        'n_periods': int(len(base)),
    }

    # volatility-tilt regression: excess over eligible-market period mean replaced by pairwise
    # vs adv40low path; tilt regressor = pool mean|ret| minus market mean|ret| for the same periods
    for name in ('amihud20', 'adv10'):
        df = pd.concat([paths[name], paths['adv40low'], tilts[name], pd.Series(market)], axis=1,
                       keys=('p', 'b', 'tp', 'tm')).dropna()
        excess = df['p'] - df['b']
        tilt_x = df['tp'] - df['tm']
        diag = {'n': int(len(df)), 'sd_excess': float(excess.std()), 'sd_tilt': float(tilt_x.std())}
        if diag['n'] >= 24 and diag['sd_tilt'] > 1e-12:
            beta, tstat = newey_west(excess.to_numpy(), tilt_x.to_numpy(), lags=1)
            diag.update(alpha_period=float(beta[0]), alpha_nw_t=float(tstat[0]),
                        beta_tilt=float(beta[1]), beta_nw_t=float(tstat[1]))
        else:
            diag['reason'] = 'degenerate_tilt_or_too_few_periods'
        result['s1_2'][name] = diag

    (OUT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float))
    for name, s in paths.items():
        s.rename(name).to_csv(OUT / f'path_{name}.csv')
    print(json.dumps(result, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
