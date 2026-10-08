#!/usr/bin/env python3
"""P2 portfolio layer: monthly inverse-volatility risk parity (docs/prereg/P2_PORTFOLIO.md).

  python scripts/p2_portfolio.py

Writes output/p2_portfolio_20261008/{results.csv, report.json}.
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

import smallcap_audit as AU
import smallcap_e4 as E4
from quantlab import research_trials as T
from quantlab.broad import month_ends

PREREG = 'docs/prereg/P2_PORTFOLIO.md'
OUT = ROOT / 'output/p2_portfolio_20261008'
START, END = '2018-01-02', '2026-09-22'
PATHS = [f'{p}_{s}' for p in (0, 4, 8, 12, 16) for s in (1, 2)]
VOL = 120
COST = {'m2': 0.004, 'cb': 0.001, 'c': 0.0005}
TILT_GATE = (0.5, 3.0)


def nav_csv(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)['nav']


def align(series, sessions):
    return series.reindex(series.index.union(sessions)).ffill().reindex(sessions)


def risk_parity(navs, signal_days, cost_mult=1.0, k=None):
    """navs: DataFrame of leg NAVs (column 'm2' is the tilted leg); returns daily NAV and mean weights."""
    names = list(navs.columns)
    costs = np.array([COST.get(n, COST['m2']) for n in names])
    r = navs.pct_change().iloc[1:].fillna(0.0)
    vol = r.rolling(VOL, min_periods=VOL).std()
    sleeves = np.full(len(names), 1 / len(names))
    nav, values, weights, pending = 1.0, [1.0], [], None
    for day, row in r.iterrows():
        if pending is not None:
            nav *= 1 - (np.abs(pending - sleeves) * costs).sum() * cost_mult
            sleeves, pending = pending, None
        growth = sleeves * (1 + row.to_numpy())
        nav *= growth.sum()
        sleeves = growth / growth.sum()
        values.append(nav)
        weights.append(sleeves)
        if day in signal_days:
            v = vol.loc[day].to_numpy()
            base = np.full(len(names), 1 / len(names)) if not np.isfinite(v).all() else (1 / v) / (1 / v).sum()
            if k is not None and 'm2' in names:
                i = names.index('m2')
                w = min(base[i] * k.get(day, 1.0), 1.0)
                rest = np.delete(base, i)
                base = np.insert(rest / rest.sum() * (1 - w), i, w)
            pending = base
    return pd.Series(values, index=navs.index), dict(zip(names, np.mean(weights, axis=0)))


def stats(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1), float((nav / nav.cummax() - 1).min())


def main():
    sessions = nav_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{PATHS[0]}_cost1.csv').loc[START:END].index
    signal_days = set(month_ends(sessions, START, END))
    cb_index = pd.read_parquet(ROOT / 'data/cb/index_000832.parquet').set_index('date').close
    h20269 = AU.bench_series('H20269')
    c_ret = 0.5 * align(cb_index, sessions).pct_change() + 0.5 * align(h20269, sessions).pct_change()
    c_leg = (1 + c_ret.fillna(0.0)).cumprod()
    cb = {1.0: align(nav_csv(ROOT / 'output/p2_cb_double_low_20261008/nav_main.csv'), sessions),
          2.0: align(nav_csv(ROOT / 'output/p2_cb_double_low_20261008/nav_cost2.csv'), sessions)}
    k = pd.read_csv(ROOT / 'output/p2_m2_tilt_20261008/indicators.csv', index_col=0, parse_dates=True).k_T3.to_dict()
    etf = E4.etf_returns(sessions)
    rows, legs = [], {}
    for path in PATHS:
        for cost_mult in (1.0, 2.0):
            m2 = align(nav_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{path}_cost{int(cost_mult)}.csv'), sessions)
            e4, _ = E4.combine(m2, etf)
            variants = {
                'fixed': (pd.DataFrame({'m2': m2, 'cb': cb[cost_mult], 'c': c_leg}), None),
                'tilt_T3': (pd.DataFrame({'m2': m2, 'cb': cb[cost_mult], 'c': c_leg}), k),
                'no_c_fixed': (pd.DataFrame({'m2': m2, 'cb': cb[cost_mult]}), None),
                'no_c_tilt_T3': (pd.DataFrame({'m2': m2, 'cb': cb[cost_mult]}), k),
                'e4_variant': (pd.DataFrame({'e4': e4, 'cb': cb[cost_mult], 'c': c_leg}), None),
            }
            for name, (frame, kk) in variants.items():
                nav, w = risk_parity(frame, signal_days, cost_mult, kk)
                c, d = stats(nav)
                rows.append({'path': path, 'cost_mult': cost_mult, 'variant': name, 'cagr': c, 'mdd': d,
                             **{f'w_{n}': x for n, x in w.items()}})
            if cost_mult == 1.0:
                legs[path] = pd.DataFrame({'m2': m2, 'cb': cb[1.0], 'c': c_leg, 'e4': e4})
    results = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / 'results.csv', index=False)
    med = results.groupby(['cost_mult', 'variant'])[['cagr', 'mdd']].median()
    fixed, tilt = med.loc[(1.0, 'fixed')], med.loc[(1.0, 'tilt_T3')]
    d_cagr, d_mdd = (tilt.cagr - fixed.cagr) * 100, (tilt.mdd - fixed.mdd) * 100
    recheck = bool((d_cagr >= TILT_GATE[0] and d_mdd >= 0) or (d_mdd >= TILT_GATE[1] and d_cagr >= -0.5))
    main_rule = 'tilt_T3' if recheck else 'fixed'
    main = med.loc[(1.0, main_rule)]
    sub = results[(results.cost_mult == 1.0) & (results.variant == main_rule)]
    leg0 = legs[PATHS[0]]
    monthly = leg0.resample('ME').last().pct_change()
    report = {
        't3_recheck': {'delta_cagr_pp': d_cagr, 'delta_mdd_pp': d_mdd, 'pass': recheck},
        'main_rule': main_rule,
        'main': {'cagr': main.cagr, 'mdd': main.mdd},
        'gate': {'cagr>=15%': bool(main.cagr >= 0.15), 'mdd>=-25%': bool(main.mdd >= -0.25)},
        'share_paths_cagr15_dd25': float(((sub.cagr >= 0.15) & (sub.mdd >= -0.25)).mean()),
        'median_by_variant': {f'{v}@cost{int(c)}': {'cagr': r.cagr, 'mdd': r.mdd} for (c, v), r in med.iterrows()},
        'mean_weights_main': sub.filter(like='w_').median().to_dict(),
        'legs_standalone_path0_1': {n: dict(zip(('cagr', 'mdd'), stats(leg0[n].dropna()))) for n in leg0},
        'legs_standalone_m2_median': dict(zip(('cagr', 'mdd'), np.median(
            [stats(legs[p]['m2']) for p in PATHS], axis=0))),
        'monthly_corr_path0_1': monthly.corr().round(3).to_dict(),
        'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG), 'promotion_approved': False,
    }
    report['gate']['pass'] = all(report['gate'].values())
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ('t3_recheck', 'main_rule', 'main', 'gate', 'share_paths_cagr15_dd25')},
                     default=float))


if __name__ == '__main__':
    main()
