#!/usr/bin/env python3
"""P4 plan B: POOL50 at 97 percent gross (docs/prereg/P4_GROSS.md).

  ledger --cost 1|2   POOL50_G97 x phases {0,5,10,15} x seeds {1,2}
  report              gates against the P4_alpha POOL50 (90 percent) ledgers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd

import p4_alpha as P
from quantlab import p4_ledger
from quantlab import research_trials as T
from quantlab.alpha.portfolio import random_order, schedule

PREREG = 'docs/prereg/P4_GROSS.md'
OUT = ROOT / 'output/p4_gross_20261013'
ALPHA = ROOT / 'output/p4_alpha_20261012'
GROSS = 0.97


def plans():
    scores = pd.read_parquet(ALPHA / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(P.ANCHOR)]
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    out = {}
    for phase in P.PHASES:
        rebalance = days[phase // P.STEP::P.REBALANCE // P.STEP]
        industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
        for seed in P.SEEDS:
            order = lambda day, s=seed: random_order(list(by_day[day].code), s)
            out[f'POOL50_G97_p{phase}_s{seed}'] = schedule(rebalance, order, industries, gross=GROSS)
    return out


def stage_ledger(cost_mult):
    folder = OUT / 'ledger'
    folder.mkdir(parents=True, exist_ok=True)
    targets = {name: p4_ledger.to_targets(s) for name, s in plans().items()}
    codes = sorted({c for rows in targets.values() for r in rows for c in list(r['weights']) + r['sell']})
    dataset = p4_ledger.load_dataset(codes)
    rows = []
    for name, target in targets.items():
        result = p4_ledger.replay(target, start=P.ANCHOR, end=P.END, cost_mult=cost_mult, dataset=dataset, gross=GROSS)
        s = result['summary']
        result['nav'].to_frame('nav').to_csv(folder / f'nav_{name}_cost{cost_mult:g}.csv')
        rows.append(dict(path=name, cost_mult=cost_mult, cagr=s['cagr'], mdd=s['max_drawdown'], fees=s['fees'],
                         slippage=s['slippage'], fills=s['fills'], checks_passed=s['account_checks_passed']))
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(folder / f'summary_cost{cost_mult:g}.csv', index=False)


def stage_report():
    mine1, mine2 = pd.read_csv(OUT / 'ledger/summary_cost1.csv'), pd.read_csv(OUT / 'ledger/summary_cost2.csv')
    base1, base2 = (pd.read_csv(ALPHA / f'ledger/summary_cost{c}.csv') for c in (1, 2))
    base1, base2 = base1[base1.path.str.startswith('POOL50_')], base2[base2.path.str.startswith('POOL50_')]
    gates = {'cagr_plus_1pp': mine1.cagr.median() - base1.cagr.median() >= 0.01,
             'mdd_worse_at_most_2pp': mine1.mdd.median() - base1.mdd.median() >= -0.02,
             'cost2_above': mine2.cagr.median() > base2.cagr.median()}
    report = dict(g97=dict(cagr_cost1=float(mine1.cagr.median()), cagr_cost2=float(mine2.cagr.median()),
                           mdd_cost1=float(mine1.mdd.median()), checks_passed=bool(mine1.checks_passed.all() and mine2.checks_passed.all())),
                  g90=dict(cagr_cost1=float(base1.cagr.median()), cagr_cost2=float(base2.cagr.median()),
                           mdd_cost1=float(base1.mdd.median())),
                  gates=gates, passed=all(gates.values()), equity_book='POOL50_G97' if all(gates.values()) else 'POOL50',
                  prereg=PREREG, prereg_sha256=T.sha256_file(ROOT / PREREG), promotion_approved=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('ledger', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    stage_ledger(args.cost) if args.stage == 'ledger' else stage_report()
