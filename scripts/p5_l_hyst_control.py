#!/usr/bin/env python3
"""P5_L matched-turnover control for GBDT_HYST (docs/prereg/P5_L_HYST_CONTROL.md).

  calibrate            pick AR(1) rho for the random scores from plan replacement only
  ledger --cost 1|2    treatment (GBDT_HYST x 4 phases) + control (random x 4 phases x 2 seeds)
  report               NW t / block bootstrap on the phase-averaged daily excess

The treatment and control share universe, grid, hysteresis rule, industry cap, capital, and ledger;
only the ordering score differs. No result here changes the strategy baseline.
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
import p4_lowturn as L
from quantlab import p4_ledger
from quantlab import research_trials as T
from quantlab.alpha.portfolio import schedule, select_hysteresis
from quantlab.research_statistics import newey_west

PREREG = 'docs/prereg/P5_L_HYST_CONTROL.md'
OUT = ROOT / 'output/p5_l_hyst_control_20261018'
RHO_GRID = (0.90, 0.95, 0.98, 0.99)
SEEDS = (1, 2)
TARGET_REPLACED = 0.2339048
BOOT_BLOCK, BOOT_DRAWS, BOOT_SEED = 20, 10_000, 20261018


def scores_by_day():
    scores = pd.read_parquet(P.OUT / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(P.ANCHOR)]
    by_day = {day: group for day, group in scores.groupby('date')}
    return by_day, sorted(by_day)


def grid(days, phase):
    return days[phase // P.STEP::P.REBALANCE // P.STEP]


def random_orders(by_day, days, rho, seed):
    """AR(1) ordering scores over the rebalance grid; only the ordering is used."""
    rng = np.random.default_rng(seed)
    previous, out = {}, {}
    for day in days:
        codes = list(by_day[day].code)
        eps = rng.standard_normal(len(codes))
        current = {}
        for code, shock in zip(codes, eps):
            old = previous.get(code)
            current[code] = float(shock if old is None else rho * old + np.sqrt(1 - rho * rho) * shock)
        previous = current
        out[day] = sorted(codes, key=lambda code: (-current[code], code))
    return out


def industries_on(by_day):
    return lambda day: dict(zip(by_day[day].code, by_day[day].industry))


def replacement_rates(plan):
    rates, previous = [], set()
    for day in sorted(plan):
        names = set(plan[day])
        if previous:
            rates.append(len(names - previous) / max(len(names), 1))
        previous = names
    return rates


def treatment_plans():
    return L.plans()


def control_plans(rho):
    by_day, days = scores_by_day()
    industries = industries_on(by_day)
    out = {}
    for phase in P.PHASES:
        days_phase = grid(days, phase)
        for seed in SEEDS:
            orders = random_orders(by_day, days_phase, rho, seed)
            out[f'RAND_p{phase}_s{seed}'] = schedule(days_phase, lambda day, o=orders: o[day],
                                                     industries, selector=select_hysteresis)
    return out


def stage_calibrate():
    by_day, days = scores_by_day()
    industries = industries_on(by_day)
    days_phase = grid(days, 0)
    table = {}
    for rho in RHO_GRID:
        per_seed = {}
        pooled = []
        for seed in SEEDS:
            orders = random_orders(by_day, days_phase, rho, seed)
            plan = schedule(days_phase, lambda day, o=orders: o[day], industries,
                            selector=select_hysteresis)
            rates = replacement_rates(plan)
            pooled.extend(rates)
            per_seed[str(seed)] = float(np.median(rates))
        table[str(rho)] = {'median': float(np.median(pooled)), 'per_seed_median': per_seed}
    chosen = min(RHO_GRID, key=lambda rho: (abs(table[str(rho)]['median'] - TARGET_REPLACED), -rho))
    result = {'target_replaced': TARGET_REPLACED, 'rho_grid': table, 'chosen_rho': chosen,
              'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG),
              'promotion_approved': False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'calibration.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def stage_ledger(cost_mult):
    rho = json.loads((OUT / 'calibration.json').read_text())['chosen_rho']
    plans = treatment_plans()
    plans.update(control_plans(rho))
    targets = {name: p4_ledger.to_targets(plan) for name, plan in plans.items()}
    codes = sorted({code for rows in targets.values() for row in rows
                    for code in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    folder = OUT / 'ledger'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, target in targets.items():
        result = p4_ledger.replay(target, start=P.ANCHOR, end=P.END, cost_mult=cost_mult,
                                  dataset=dataset)
        summary = result['summary']
        result['nav'].to_frame('nav').to_csv(folder / f'nav_{name}_cost{cost_mult:g}.csv')
        plan_rates = replacement_rates(plans[name])
        row = dict(path=name, cost_mult=cost_mult, cagr=summary['cagr'],
                   mdd=summary['max_drawdown'], fees=summary['fees'], slippage=summary['slippage'],
                   fills=summary['fills'], mean_exposure=summary['mean_exposure'],
                   mean_cash_weight=summary['mean_cash_weight'],
                   data_quality_ok=summary['data_quality_ok'],
                   stale_held_max_days=summary['stale_held_max_days'],
                   delisting_liquidations=summary['delisting_liquidations'],
                   account_checks_passed=summary['account_checks_passed'],
                   plan_replaced=float(np.mean(plan_rates)))
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    pd.DataFrame(rows).to_csv(folder / f'summary_cost{cost_mult:g}.csv', index=False)


def circular_block_bootstrap_mean(values, block=BOOT_BLOCK, draws=BOOT_DRAWS, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    samples = values[index.reshape(draws, -1)[:, :n]]
    return samples.mean(axis=1)


def stage_report():
    folder = OUT / 'ledger'
    summaries = {cost: pd.read_csv(folder / f'summary_cost{cost:g}.csv') for cost in (1, 2)}
    nav = lambda name, cost: pd.read_csv(folder / f'nav_{name}_cost{cost:g}.csv', index_col=0,
                                         parse_dates=True)['nav']
    by_phase = {}
    phase_excess = {}
    for phase in P.PHASES:
        name = f'GBDT_HYST_p{phase}'
        treat1 = nav(name, 1).pct_change().dropna()
        controls = [nav(f'RAND_p{phase}_s{seed}', 1).pct_change().dropna() for seed in SEEDS]
        control = pd.concat(controls, axis=1).median(axis=1)
        excess = (treat1 - control).dropna()
        phase_excess[phase] = excess
        _, t_stat = newey_west(excess.to_numpy(), lags=20)
        by_phase[str(phase)] = {'mean_excess': float(excess.mean()), 'nw_t': float(t_stat[0]),
                                'days': len(excess)}
    averaged = pd.concat(phase_excess.values(), axis=1).mean(axis=1).dropna()
    _, t_stat = newey_west(averaged.to_numpy(), lags=20)
    bootstrap = circular_block_bootstrap_mean(averaged.to_numpy())
    p_value = float(min(1.0, 2 * min((bootstrap <= 0).mean(), (bootstrap >= 0).mean())))

    def group(prefix, cost):
        frame = summaries[cost]
        return frame[frame.path.str.startswith(prefix)]

    def stats(prefix):
        out = {}
        for cost in (1, 2):
            frame = group(prefix, cost)
            out[f'cagr_cost{cost}_median'] = float(frame.cagr.median())
            out[f'mdd_cost{cost}_median'] = float(frame.mdd.median())
            out[f'plan_replaced_cost{cost}_median'] = float(frame.plan_replaced.median())
            out[f'fees_cost{cost}_median'] = float(frame.fees.median())
            out[f'slippage_cost{cost}_median'] = float(frame.slippage.median())
            out[f'data_quality_ok_cost{cost}'] = bool(frame.data_quality_ok.all())
            out[f'account_checks_ok_cost{cost}'] = bool(frame.account_checks_passed.all())
            out[f'stale_held_max_days_cost{cost}'] = int(frame.stale_held_max_days.max())
        return out

    treatment, control = stats('GBDT_HYST_'), stats('RAND_')
    data_ok = all(treatment[f'data_quality_ok_cost{cost}'] and control[f'data_quality_ok_cost{cost}']
                  and treatment[f'account_checks_ok_cost{cost}'] and control[f'account_checks_ok_cost{cost}']
                  for cost in (1, 2))
    significant = bool(data_ok and abs(float(t_stat[0])) >= 1.96 and p_value <= 0.05)
    verdict = ('insufficient_data' if not data_ok else
               'significant' if significant else 'not_significant')
    treat_turn = treatment['plan_replaced_cost1_median']
    control_turn = control['plan_replaced_cost1_median']
    turnover_matched = bool(control_turn >= 0.8 * treat_turn)
    report = {
        'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG),
        'chosen_rho': json.loads((OUT / 'calibration.json').read_text())['chosen_rho'],
        'treatment': treatment, 'control': control,
        'turnover_match': {'treatment_plan_replaced': treat_turn,
                           'control_plan_replaced': control_turn,
                           'within_20pct': turnover_matched},
        'design_caveat': (None if turnover_matched else
                          'the registered AR(1) grid could not reproduce GBDT_HYST replacement; '
                          'the control trades less and pays less cost, so the excess test is '
                          'conservative for GBDT but is not a true matched-turnover test'),
        'nw_t_averaged': float(t_stat[0]), 'bootstrap_p': p_value,
        'bootstrap_block': BOOT_BLOCK, 'bootstrap_draws': BOOT_DRAWS,
        'by_phase': by_phase, 'data_quality_ok': data_ok,
        'verdict': verdict, 'significant': significant,
        'caveat': ('diagnostic only; not an independent confirmation and not usable to approve a strategy'
                   if data_ok else
                   'data-quality gate failed (stale held quotes >5 sessions); statistics are descriptive only'),
        'promotion_approved': False,
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('calibrate', 'ledger', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    if args.stage == 'calibrate':
        stage_calibrate()
    elif args.stage == 'ledger':
        stage_ledger(args.cost)
    else:
        stage_report()
