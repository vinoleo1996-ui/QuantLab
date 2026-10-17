#!/usr/bin/env python3
"""P5_REALIZABLE: what P5_BASE actually buys under A-share lot constraints.

The P5_G0/P5_BASE numbers (13.19% / -22.78%) treat the equity leg as a fungible
series weighted at 65% x 97% gross = 63.05% of capital. A real account buys whole
lots: at 65% x 800k the per-name budget is 10,088 yuan and the four most expensive
target names (122-298 yuan) cannot be filled even with one lot, so the real equity
exposure lands near 53% with ~12% idle cash.

This diagnostic re-runs the production share ledger at the true sub-account scale
(520k = 65% x 800k, 390k = 65% x 600k), then packs it with the same G0 sleeve at
65% through quantlab/p4_alloc.py. Read-only w.r.t. the frozen books; new output
under output/p5_realizable_20261017/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alloc as R
from quantlab import p4_alloc as A
from quantlab import p4_ledger
from quantlab.alpha.portfolio import random_order, schedule

START, END = '2018-01-02', '2026-09-22'
ALPHA = ROOT / 'output/p4_alpha_20261012'
OUT = ROOT / 'output/p5_realizable_20261017'
DELISTINGS = ROOT / 'data/smallcap_v4/20261006T152412Z/delistings.parquet'
PHASES = (0, 5, 10, 15)
SEEDS = (1, 2)
GROSS = 0.97
SHARE = 0.65


def g0_target(etf_returns_upto, equity_share):
    sleeve = 1.0 - equity_share
    return np.array([equity_share, 0.375 * sleeve, 0.375 * sleeve, 0.25 * sleeve])


def etf_g0(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide[['sh518880', 'sh511260', 'sh513100']].rename(
        columns={'sh518880': 'gold', 'sh511260': 'treasury', 'sh513100': 'nasdaq'})
    wide = wide.reindex(wide.index.union(sessions)).ffill().reindex(sessions)
    return wide.pct_change(fill_method=None).fillna(0.0)


def plans():
    scores = pd.read_parquet(ALPHA / 'scores.parquet', columns=['date', 'code', 'industry'])
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    out = {}
    for phase in PHASES:
        rebalance = days[phase // 5::4]
        industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
        for seed in SEEDS:
            plan = schedule(rebalance, lambda day, s=seed: random_order(list(by_day[day].code), s),
                            industries)
            out[(phase, seed)] = p4_ledger.to_targets(plan)
    return out


def equity_stats(result):
    snaps = pd.DataFrame([
        dict(date=state['date'], nav=state['nav'], cash=state['cash'],
             stock=sum(sh * state['marks'].get(c, 0) for c, sh in state['quantities'].items()),
             n=len(state['quantities']))
        for state in result['account']])
    snaps['exposure'] = snaps.stock / snaps.nav
    snaps['cash_ratio'] = snaps.cash / snaps.nav
    return snaps


def combo(equity_nav, cost, rebalance='quarter'):
    etf = etf_g0(equity_nav.index)
    premium = R.qdii_premium_frame(equity_nav.index)
    nav, trace = A.allocate(equity_nav, etf, SHARE, cost_mult=cost, premium=premium,
                            rebalance=rebalance, overflow={'gold': 0.5, 'treasury': 0.5},
                            target_fn=g0_target)
    s = A.stats(nav)
    return dict(cagr=float(s['cagr']), mdd=float(s['mdd']), nav=nav)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capitals', default='520000,390000')
    parser.add_argument('--costs', default='1,2')
    parser.add_argument('--quick', action='store_true', help='one path only, for timing')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_plans = plans()
    if args.quick:
        all_plans = {(0, 1): all_plans[(0, 1)]}
    codes = sorted({c for rows in all_plans.values() for row in rows
                    for c in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    delist = pd.read_parquet(DELISTINGS).set_index('code') if DELISTINGS.exists() else pd.DataFrame()

    report = {'window': [START, END], 'share': SHARE, 'gross': GROSS,
              'prereg': 'docs/prereg/P5_REALIZABLE.md',
              'cells': {}, 'promotion_approved': False}
    for capital in [float(x) for x in args.capitals.split(',')]:
        for cost in [float(x) for x in args.costs.split(',')]:
            rows = []
            for (phase, seed), targets in sorted(all_plans.items()):
                result = p4_ledger.replay(targets, start=START, end=END, cost_mult=cost,
                                           dataset=dataset, capital_cny=capital, gross=GROSS)
                nav = pd.Series([s['nav'] for s in result['account']],
                                index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in result['account']]))
                nav = nav / nav.iloc[0]
                snaps = equity_stats(result)
                c = combo(nav, cost)
                end_hold = result['account'][-1]['quantities']
                delisted_end = [code for code in end_hold if code in delist.index]
                cell = dict(path=f'p{phase}_s{seed}', cagr=c['cagr'], mdd=c['mdd'],
                            equity_cagr=float((nav.iloc[-1] ** (365.25 / (nav.index[-1] - nav.index[0]).days) - 1)),
                            mean_exposure=float(snaps.exposure.mean()),
                            mean_cash=float(snaps.cash_ratio.mean()),
                            mean_names=float(snaps.n.mean()),
                            orders=len(result['orders']),
                            data_issues=len(result['data_issues']),
                            delisted_end=delisted_end)
                rows.append(cell)
                report['cells'][f'cap{int(capital)}_cost{cost:g}_{cell["path"]}'] = cell
                pd.Series(c['nav']).to_csv(OUT / f'nav_cap{int(capital)}_cost{cost:g}_{cell["path"]}.csv')
                print(json.dumps(cell, ensure_ascii=False), flush=True)
            frame = pd.DataFrame(rows)
            report['cells'][f'summary_cap{int(capital)}_cost{cost:g}'] = dict(
                n=len(frame), cagr=float(frame.cagr.median()), mdd=float(frame.mdd.median()),
                cagr_p10=float(frame.cagr.quantile(0.1)), cagr_p90=float(frame.cagr.quantile(0.9)),
                mean_exposure=float(frame.mean_exposure.mean()),
                mean_cash=float(frame.mean_cash.mean()), mean_names=float(frame.mean_names.mean()))
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print('wrote', OUT / 'report.json')


if __name__ == '__main__':
    main()
