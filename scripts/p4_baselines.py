#!/usr/bin/env python3
"""P4 baselines at the 800k mandate (no new selection rule; report only).

Stages:
  m2      frozen M2 basket (scripts/p1_m2.py) replayed on the share ledger at 800k, 10 paths, --cost 1|2
  pool    equal-weight index of the M2 pool (relative band, ST removed, 252 history), frictionless,
          re-formed every 20 sessions from the decision-day close, next-session returns
  report  M2@800k, pool EW, CSI 2000 total return, RP3 reference (contains convertible bonds,
          not admissible under the P4 mandate), written to output/p4_baselines_20261012/
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

from quantlab import p4_ledger
from quantlab.p4_mandate import MANDATE

OUT = ROOT / 'output/p4_baselines_20261012'
START, END = '2018-01-02', '2026-09-22'


def stats(nav):
    nav = nav.dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return dict(cagr=float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
                mdd=float((nav / nav.cummax() - 1).min()), first=str(nav.index[0].date()), last=str(nav.index[-1].date()))


def stage_m2(cost_mult):
    import p1_m2 as M2
    L = M2.L
    M2.use('is')
    L.load_state()
    folder = OUT / 'm2'
    folder.mkdir(parents=True, exist_ok=True)
    plans = {f'{p}_{s}': p4_ledger.to_targets(M2.schedule_for(p, s)) for p, s in M2.LEDGER_PATHS}
    codes = sorted({c for rows in plans.values() for r in rows for c in list(r['weights']) + r['sell']})
    dataset = p4_ledger.load_dataset(codes)
    rows = []
    for name, targets in plans.items():
        result = p4_ledger.replay(targets, start=START, end=END, cost_mult=cost_mult, dataset=dataset)
        s = result['summary']
        result['nav'].to_frame('nav').to_csv(folder / f'nav_{name}_cost{cost_mult:g}.csv')
        rows.append(dict(path=name, cost_mult=cost_mult, cagr=s['cagr'], mdd=s['max_drawdown'],
                         initial_cash=s['initial_cash'], final_equity=s['final_equity'], fees=s['fees'],
                         slippage=s['slippage'], fills=s['fills'], checks_passed=s['account_checks_passed']))
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(folder / f'summary_cost{cost_mult:g}.csv', index=False)


def stage_pool():
    import p1_m2 as M2
    from quantlab.smallcap import universe
    from quantlab.smallcap.panel import Panel, latest_version
    M2.use('is')
    panel = Panel(latest_version())
    features = universe.rolling_features(panel, with_st=True)
    adj = panel.field('adj_close')
    returns = adj.pct_change(fill_method=None)
    sessions = adj.index
    grid = universe.decision_grid(sessions, START, END, phase=0, step=20)
    daily, members = [], {}
    for i, day in enumerate(grid):
        pool = universe.pool_on(day, features, 'adv', band='relative', exclude_st=True, min_history=252)
        members[str(day.date())] = len(pool)
        stop = grid[i + 1] if i + 1 < len(grid) else sessions[sessions <= pd.Timestamp(END)][-1]
        window = sessions[(sessions > day) & (sessions <= stop)]
        if not pool:
            daily.append(pd.Series(0.0, index=window))
            continue
        block = returns.loc[window, pool].fillna(0.0)
        weights = (1 + block).cumprod().shift(1).fillna(1.0)
        daily.append((weights * block).sum(axis=1) / weights.sum(axis=1))
    series = pd.concat(daily).sort_index()
    nav = (1 + series).cumprod()
    OUT.mkdir(parents=True, exist_ok=True)
    nav.to_frame('nav').to_csv(OUT / 'pool_ew_nav.csv')
    (OUT / 'pool_ew_members.json').write_text(json.dumps(members))
    print(json.dumps(dict(pool_ew=stats(nav), decision_days=len(grid))), flush=True)


def stage_report():
    import smallcap_audit as AU
    import p2_portfolio as RP
    from quantlab.broad import month_ends
    summary = {'mandate': MANDATE.version, 'capital_cny': MANDATE.capital_cny, 'window': [START, END]}
    for cost in (1, 2):
        path = OUT / f'm2/summary_cost{cost}.csv'
        if path.exists():
            frame = pd.read_csv(path)
            summary[f'm2_800k_cost{cost}'] = dict(median_cagr=float(frame.cagr.median()),
                                                  median_mdd=float(frame.mdd.median()),
                                                  range_cagr=[float(frame.cagr.min()), float(frame.cagr.max())],
                                                  paths=len(frame), all_checks_passed=bool(frame.checks_passed.all()))
    pool = pd.read_csv(OUT / 'pool_ew_nav.csv', index_col=0, parse_dates=True)['nav']
    summary['pool_ew_frictionless'] = stats(pool)
    csi2000 = AU.bench_series('932000CNY010').loc[START:END]
    summary['csi2000_total_return'] = stats(csi2000)
    m2_paths = sorted((OUT / 'm2').glob('nav_*_cost1.csv'))
    if m2_paths:
        sessions = pd.read_csv(m2_paths[0], index_col=0, parse_dates=True).loc[START:END].index
        signal_days = set(month_ends(sessions, START, END))
        cb_index = pd.read_parquet(ROOT / 'data/cb/index_000832.parquet').set_index('date').close
        c_ret = 0.5 * RP.align(cb_index, sessions).pct_change() + 0.5 * RP.align(AU.bench_series('H20269'), sessions).pct_change()
        c_leg = (1 + c_ret.fillna(0.0)).cumprod()
        cb = RP.align(RP.nav_csv(ROOT / 'output/p2_cb_double_low_20261008/nav_main.csv'), sessions)
        results = []
        for path in m2_paths:
            m2 = RP.align(RP.nav_csv(path), sessions)
            nav, _ = RP.risk_parity(pd.DataFrame({'m2': m2, 'cb': cb, 'c': c_leg}), signal_days)
            results.append(RP.stats(nav))
        summary['rp3_reference_not_admissible'] = dict(median_cagr=float(np.median([r[0] for r in results])),
                                                       median_mdd=float(np.median([r[1] for r in results])),
                                                       note='contains convertible-bond legs excluded by P4 mandate')
    summary['promotion_approved'] = False
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('m2', 'pool', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    {'m2': lambda: stage_m2(args.cost), 'pool': stage_pool, 'report': stage_report}[args.stage]()
