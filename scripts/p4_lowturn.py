#!/usr/bin/env python3
"""P4 plan A: GBDT scores with entry/exit hysteresis (docs/prereg/P4_LOWTURN.md).

  ledger --cost 1|2   GBDT_HYST x phases {0,5,10,15} on the 800k ledger (scores reused from P4_alpha)
  report              gates against the P4_alpha POOL50 ledgers
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

import p4_alpha as P
from quantlab import p4_ledger
from quantlab import research_trials as T
from quantlab.alpha.portfolio import schedule, select_hysteresis
from quantlab.research_statistics import deflated_sharpe, newey_west

PREREG = 'docs/prereg/P4_LOWTURN.md'
OUT = ROOT / 'output/p4_lowturn_20261013'
ALPHA = ROOT / 'output/p4_alpha_20261012'


def plans():
    scores = pd.read_parquet(ALPHA / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(P.ANCHOR)]
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    out = {}
    for phase in P.PHASES:
        rebalance = days[phase // P.STEP::P.REBALANCE // P.STEP]
        industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
        order = lambda day: list(by_day[day].sort_values('GBDT', ascending=False, kind='mergesort').code)
        out[f'GBDT_HYST_p{phase}'] = schedule(rebalance, order, industries, selector=select_hysteresis)
    return out


def stage_ledger(cost_mult):
    folder = OUT / 'ledger'
    folder.mkdir(parents=True, exist_ok=True)
    targets = {name: p4_ledger.to_targets(s) for name, s in plans().items()}
    codes = sorted({c for rows in targets.values() for r in rows for c in list(r['weights']) + r['sell']})
    dataset = p4_ledger.load_dataset(codes)
    rows = []
    for name, target in targets.items():
        result = p4_ledger.replay(target, start=P.ANCHOR, end=P.END, cost_mult=cost_mult, dataset=dataset)
        s = result['summary']
        result['nav'].to_frame('nav').to_csv(folder / f'nav_{name}_cost{cost_mult:g}.csv')
        rows.append(dict(path=name, cost_mult=cost_mult, cagr=s['cagr'], mdd=s['max_drawdown'], fees=s['fees'],
                         slippage=s['slippage'], fills=s['fills'], checks_passed=s['account_checks_passed']))
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(folder / f'summary_cost{cost_mult:g}.csv', index=False)


def stage_report():
    nav = lambda folder, name, cost: pd.read_csv(folder / f'nav_{name}_cost{cost}.csv', index_col=0, parse_dates=True)['nav']
    mine1, mine2 = pd.read_csv(OUT / 'ledger/summary_cost1.csv'), pd.read_csv(OUT / 'ledger/summary_cost2.csv')
    pool1, pool2 = pd.read_csv(ALPHA / 'ledger/summary_cost1.csv'), pd.read_csv(ALPHA / 'ledger/summary_cost2.csv')
    pool1, pool2 = pool1[pool1.path.str.startswith('POOL50_')], pool2[pool2.path.str.startswith('POOL50_')]
    sub = lambda folder, names: float(np.median([P._cagr(nav(folder, n, 1).loc[P.SUB_START:]) for n in names]))
    control = nav(ALPHA / 'ledger', 'POOL50_p0_s1', 1).pct_change().dropna()
    excess = (nav(OUT / 'ledger', 'GBDT_HYST_p0', 1).pct_change().dropna() - control).dropna()
    _, t_stat = newey_west(excess.to_numpy(), lags=20)
    plan_turnover = []
    for schedule_ in plans().values():
        days, previous = sorted(schedule_), set()
        for day in days:
            names = set(schedule_[day])
            if previous:
                plan_turnover.append(len(names - previous) / max(len(names), 1))
            previous = names
    gates = {'cagr_minus_pool_ge_2pp': mine1.cagr.median() - pool1.cagr.median() >= 0.02,
             'nw_t_ge_2': float(t_stat[0]) >= 2.0,
             'cost2_above_pool': mine2.cagr.median() > pool2.cagr.median(),
             'sub_2022_above_pool': sub(OUT / 'ledger', mine1.path) > sub(ALPHA / 'ledger', pool1.path)}
    report = dict(gbdt_hyst=dict(cagr_cost1=float(mine1.cagr.median()), cagr_cost2=float(mine2.cagr.median()),
                                 mdd_cost1=float(mine1.mdd.median()), cagr_2022=sub(OUT / 'ledger', mine1.path),
                                 by_phase=dict(zip(mine1.path, mine1.cagr.round(4))), nw_t=float(t_stat[0]),
                                 mean_replaced=float(np.mean(plan_turnover)),
                                 yearly_one_way=float(np.mean(plan_turnover) * 252 / 20),
                                 dsr_global=deflated_sharpe(excess, trials=T.count(include_diagnostics=False)).get('value')),
                  pool50=dict(cagr_cost1=float(pool1.cagr.median()), cagr_cost2=float(pool2.cagr.median()),
                              mdd_cost1=float(pool1.mdd.median()), cagr_2022=sub(ALPHA / 'ledger', pool1.path)),
                  gates=gates, passed=all(gates.values()), prereg=PREREG, prereg_sha256=T.sha256_file(ROOT / PREREG),
                  promotion_approved=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('ledger', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    stage_ledger(args.cost) if args.stage == 'ledger' else stage_report()
