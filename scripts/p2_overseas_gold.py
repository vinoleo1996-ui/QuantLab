#!/usr/bin/env python3
"""Overseas equity index QDII + gold passive leg, then 4-leg risk parity (docs/prereg/P2_OVERSEAS_GOLD.md).

  python scripts/p2_overseas_gold.py

Writes output/p2_overseas_gold_20261008/{nav_*.csv, results.csv, report.json}.
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

import p2_portfolio as P
from quantlab import research_trials as T
from quantlab.broad import month_ends
from quantlab.etf_account import replay, stats

PREREG = 'docs/prereg/P2_OVERSEAS_GOLD.md'
OUT = ROOT / 'output/p2_overseas_gold_20261008'
ASSETS = ['sh513100', 'sh513500', 'sz159920', 'sh510900', 'sh513030', 'sh518880']
DISCOVERY, CONFIRM = ('2018-01-02', '2026-09-22'), ('2014-09-05', '2017-12-29')
VOL, INVESTED = 120, 0.98
P.COST['og'] = 0.0005


def leg_targets(bars, assets, start, end):
    close = bars[bars.code.isin(assets)].pivot(index='date', columns='code', values='close_hfq').sort_index().ffill()
    returns = close.pct_change()
    sessions = close.loc[start:end].index
    out = {}
    for day in month_ends(sessions, start, end):
        window = returns.loc[:day].tail(VOL)
        vol = window.std()
        ok = window.count() >= VOL
        if ok.all():
            w = (1 / vol) / (1 / vol).sum()
        else:
            w = pd.Series(1 / len(assets), index=assets)
        out[day] = (w * INVESTED).to_dict()
    return out


def leg(bars, assets, window, cost_mult=1.0):
    start, end = window
    targets = leg_targets(bars, assets, start, end)
    first = min(targets)
    if first > pd.Timestamp(start):
        targets[pd.Timestamp(start)] = {a: INVESTED / len(assets) for a in assets}
    return replay(bars, targets, start, end, cost_mult=cost_mult)['nav']


def main():
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    OUT.mkdir(parents=True, exist_ok=True)
    navs = {'discovery': leg(bars, ASSETS, DISCOVERY), 'discovery_cost2': leg(bars, ASSETS, DISCOVERY, 2.0),
            'confirm': leg(bars, ASSETS, CONFIRM),
            'discovery_no_gold': leg(bars, [a for a in ASSETS if a != 'sh518880'], DISCOVERY)}
    for name, nav in navs.items():
        nav.to_csv(OUT / f'nav_{name}.csv')
    bond = bars[bars.code == 'sh511010'].set_index('date').close_hfq.sort_index().loc[CONFIRM[0]:CONFIRM[1]]
    confirm_leg, confirm_bond = stats(navs['confirm']), stats(bond)

    sessions = P.nav_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{P.PATHS[0]}_cost1.csv').loc[DISCOVERY[0]:DISCOVERY[1]].index
    signal_days = set(month_ends(sessions, *DISCOVERY))
    cb_index = pd.read_parquet(ROOT / 'data/cb/index_000832.parquet').set_index('date').close
    c_ret = 0.5 * P.align(cb_index, sessions).pct_change() + 0.5 * P.align(P.AU.bench_series('H20269'), sessions).pct_change()
    c_leg = (1 + c_ret.fillna(0.0)).cumprod()
    cb = {1.0: P.align(P.nav_csv(ROOT / 'output/p2_cb_double_low_20261008/nav_main.csv'), sessions),
          2.0: P.align(P.nav_csv(ROOT / 'output/p2_cb_double_low_20261008/nav_cost2.csv'), sessions)}
    og = {1.0: P.align(navs['discovery'], sessions), 2.0: P.align(navs['discovery_cost2'], sessions)}
    rows = []
    for path in P.PATHS:
        for cost_mult in (1.0, 2.0):
            m2 = P.align(P.nav_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{path}_cost{int(cost_mult)}.csv'), sessions)
            for name, frame in (('rp4', pd.DataFrame({'m2': m2, 'cb': cb[cost_mult], 'c': c_leg, 'og': og[cost_mult]})),
                                ('rp3', pd.DataFrame({'m2': m2, 'cb': cb[cost_mult], 'c': c_leg}))):
                nav, w = P.risk_parity(frame, signal_days, cost_mult)
                c, d = P.stats(nav)
                rows.append({'path': path, 'cost_mult': cost_mult, 'variant': name, 'cagr': c, 'mdd': d,
                             **{f'w_{k}': v for k, v in w.items()}})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'results.csv', index=False)
    med = results.groupby(['cost_mult', 'variant'])[['cagr', 'mdd']].median()
    rp4, rp3 = med.loc[(1.0, 'rp4')], med.loc[(1.0, 'rp3')]
    m2_mean = pd.concat([P.nav_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{p}_cost1.csv') for p in P.PATHS],
                        axis=1).mean(axis=1)
    monthly = pd.DataFrame({'og': og[1.0], 'm2': P.align(m2_mean, sessions), 'cb': cb[1.0]}).resample('ME').last().pct_change()
    report = {
        'gate': {'rp4_cagr>=15%': bool(rp4.cagr >= 0.15), 'rp4_mdd>=-25%': bool(rp4.mdd >= -0.25),
                 'confirm_leg>bond': bool(confirm_leg['cagr'] > confirm_bond['cagr'])},
        'rp4': {'cagr': rp4.cagr, 'mdd': rp4.mdd}, 'rp3': {'cagr': rp3.cagr, 'mdd': rp3.mdd},
        'rp4_cost2': {'cagr': float(med.loc[(2.0, 'rp4')].cagr), 'mdd': float(med.loc[(2.0, 'rp4')].mdd)},
        'share_paths_cagr15_dd25': float(((results.cost_mult == 1.0) & (results.variant == 'rp4') &
                                          (results.cagr >= 0.15) & (results.mdd >= -0.25)).sum() / len(P.PATHS)),
        'rp4_mean_weights': results[(results.cost_mult == 1.0) & (results.variant == 'rp4')].filter(like='w_').median().to_dict(),
        'leg_discovery': stats(navs['discovery']), 'leg_discovery_no_gold': stats(navs['discovery_no_gold']),
        'leg_confirm': confirm_leg, 'bond_511010_confirm': confirm_bond,
        'monthly_corr': monthly.corr().round(3).to_dict(),
        'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG), 'promotion_approved': False,
    }
    report['gate']['pass'] = all(report['gate'].values())
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ('gate', 'rp4', 'rp3', 'rp4_cost2', 'leg_confirm', 'bond_511010_confirm')},
                     default=float))


if __name__ == '__main__':
    main()
