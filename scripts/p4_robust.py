#!/usr/bin/env python3
"""Robustness of plan A (docs/prereg/P4_LOWTURN.md), diagnostics only: nothing here selects a rule.

  neighborhood  fast simulator, entry {5,10,20}% x exit {40,50,60}% x phases, against POOL50 in the same simulator;
                passes when at least 7 of 9 combos beat POOL50 by 1.5 points (median over phases)
  seeds         GBDT retrained with seeds 6..10 (same walk-forward), then plan A on the 800k ledger, phase 0, cost 1
  stats         daily excess of plan A over POOL50 (same phase, mean of seeds) from the ledgers: by year,
                moving-block bootstrap interval, deflated Sharpe at the global trial count
Output: output/p4_robust_20261013/.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alpha as P
import p4_lowturn as LT
from quantlab import p4_ledger
from quantlab import research_trials as T
from quantlab.alpha.labels import HORIZON
from quantlab.alpha.models import GBDTEnsemble
from quantlab.alpha.portfolio import random_order, schedule, select_hysteresis
from quantlab.p4_mandate import MANDATE
from quantlab.research_statistics import deflated_sharpe, simultaneous_block_ci
from quantlab.smallcap.panel import Panel
from quantlab.smallcap.sim import Market, metrics, simulate

OUT = ROOT / 'output/p4_robust_20261013'
ALPHA, LOWTURN = ROOT / 'output/p4_alpha_20261012', ROOT / 'output/p4_lowturn_20261013'
ENTRY, EXIT = (0.05, 0.10, 0.20), (0.40, 0.50, 0.60)


def score_days(scores_path, column):
    scores = pd.read_parquet(scores_path)
    scores = scores[scores.date >= pd.Timestamp(P.ANCHOR)]
    by_day = {day: group for day, group in scores.groupby('date')}
    return by_day, sorted(by_day)


def book(by_day, days, phase, column=None, selector=None, seed=None):
    rebalance = days[phase // P.STEP::P.REBALANCE // P.STEP]
    industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
    if seed is not None:
        order = lambda day: random_order(list(by_day[day].code), seed)
    else:
        order = lambda day: list(by_day[day].sort_values(column, ascending=False, kind='mergesort').code)
    return schedule(rebalance, order, industries, selector=selector)


def market():
    panel = Panel(P.PANEL)
    delist = panel.table('delistings')
    return Market.from_panel(panel, P.ANCHOR, P.END, delist=dict(zip(delist.code, delist.last_bar)))


def sim_cagr(mk, plan):
    return metrics(simulate(mk, plan, initial_cash=MANDATE.capital_cny)['nav'], MANDATE.capital_cny)['cagr']


def stage_neighborhood():
    mk = market()
    by_day, days = score_days(ALPHA / 'scores.parquet', 'GBDT')
    pool = [sim_cagr(mk, book(by_day, days, p, seed=s)) for p in P.PHASES for s in P.SEEDS]
    rows = []
    for entry in ENTRY:
        for exit_ in EXIT:
            selector = functools.partial(select_hysteresis, entry_frac=entry, exit_frac=exit_)
            values = [sim_cagr(mk, book(by_day, days, p, 'GBDT', selector)) for p in P.PHASES]
            rows.append(dict(entry=entry, exit=exit_, median_cagr=float(np.median(values)),
                             phases=[round(v, 4) for v in values]))
            print(json.dumps(rows[-1]), flush=True)
    frame = pd.DataFrame(rows)
    frame['excess_vs_pool'] = frame.median_cagr - float(np.median(pool))
    passed = int((frame.excess_vs_pool >= 0.015).sum())
    out = dict(pool50_sim_median=float(np.median(pool)), combos=frame.to_dict('records'),
               combos_beating_by_1_5pp=passed, robust=passed >= 7,
               center=frame[(frame.entry == 0.10) & (frame.exit == 0.50)].to_dict('records'))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'neighborhood.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('pool50_sim_median', 'combos_beating_by_1_5pp', 'robust')}, default=float))


def stage_seeds():
    samples = pd.read_parquet(ALPHA / 'samples.parquet')
    present = json.loads((ALPHA / 'samples_meta.json').read_text())['features']
    sessions = Panel(P.PANEL).sessions
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    refits = P.refit_days(samples.date.unique(), P.ANCHOR)
    parts = []
    for i, refit in enumerate(refits):
        stop = refits[i + 1] if i + 1 < len(refits) else samples.date.max() + pd.Timedelta(days=1)
        train = samples[(samples.label_known_pos <= position[refit] - HORIZON - 1) & samples.label.notna()]
        test = samples[(samples.date >= refit) & (samples.date < stop)].copy()
        test['GBDT'] = GBDTEnsemble().fit(train, present, seeds=(6, 7, 8, 9, 10)).predict(test)
        parts.append(test[['date', 'code', 'industry', 'label', 'GBDT']])
        print(json.dumps(dict(refit=str(refit.date()))), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(OUT / 'scores_seed6_10.parquet', index=False)
    by_day, days = score_days(OUT / 'scores_seed6_10.parquet', 'GBDT')
    plan = book(by_day, days, 0, 'GBDT', select_hysteresis)
    targets = p4_ledger.to_targets(plan)
    result = p4_ledger.replay(targets, start=P.ANCHOR, end=P.END, cost_mult=1.0)
    result['nav'].to_frame('nav').to_csv(OUT / 'nav_GBDT_HYST_seed6_10_p0_cost1.csv')
    original = pd.read_csv(LOWTURN / 'ledger/summary_cost1.csv').set_index('path').loc['GBDT_HYST_p0', 'cagr']
    out = dict(seed6_10_cagr=result['summary']['cagr'], seed1_5_cagr=float(original),
               difference_pp=(result['summary']['cagr'] - float(original)) * 100,
               within_1pp=abs(result['summary']['cagr'] - float(original)) <= 0.01,
               checks_passed=result['summary']['account_checks_passed'])
    (OUT / 'seeds.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float))


def stage_stats():
    nav = lambda folder, name: pd.read_csv(folder / f'nav_{name}_cost1.csv', index_col=0, parse_dates=True)['nav']
    family = {}
    for phase in P.PHASES:
        mine = nav(LOWTURN / 'ledger', f'GBDT_HYST_p{phase}').pct_change()
        pool = pd.concat([nav(ALPHA / 'ledger', f'POOL50_p{phase}_s{s}').pct_change() for s in P.SEEDS], axis=1).mean(axis=1)
        family[f'p{phase}'] = (mine - pool).dropna()
    frame = pd.DataFrame(family).dropna()
    frame['mean'] = frame.mean(axis=1)
    yearly = frame['mean'].groupby(frame.index.year).sum()
    ci = simultaneous_block_ci(frame, block=20)
    dsr = deflated_sharpe(frame['mean'], trials=T.count(include_diagnostics=False))
    out = dict(yearly_excess_pp={int(y): round(v * 100, 2) for y, v in yearly.items()},
               positive_years=int((yearly > 0).sum()), years=len(yearly),
               bootstrap=ci['comparisons'], dsr_global=dsr.get('value'), dsr_trials=dsr.get('trials'),
               robust=bool(ci['comparisons']['mean']['lower'] > 0 and (yearly > 0).sum() >= 6))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stats.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('neighborhood', 'seeds', 'stats'))
    args = parser.parse_args()
    {'neighborhood': stage_neighborhood, 'seeds': stage_seeds, 'stats': stage_stats}[args.stage]()
