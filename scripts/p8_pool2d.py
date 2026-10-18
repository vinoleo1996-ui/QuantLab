#!/usr/bin/env python3
"""P8_POOL2D: production-ledger confirmation of the 2-D ADV x turnover pool cells.

The zero-cost diagnostic ranked nine cells; two passed its CAGR/MDD screen. This script runs
the primary cell on the real ledger (520k stock sleeve, 8 paths, cost 1/2, board lots, T+1,
suspension evidence, delisting settlement). Gates are against the current stock sleeve
(13.48%/-34.93% cost1, 12.03%/-35.49% cost2), not the nominal 15.1%.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab import p4_ledger
from quantlab.alpha.portfolio import random_order
from quantlab.data.suspensions import SuspensionEvidence
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

MANIFEST = '20261013T023232Z'
PREREG = ROOT / 'docs/prereg/P8_POOL2D.md'
BASELINE = ROOT / 'output/p5_base_lot_20261018_r1/report.json'
OUT = ROOT / 'output/p8_pool2d_20261018'
START, END = '2018-01-02', '2026-09-22'
CAPITAL, GROSS = 520_000.0, 0.97
PHASES, SEEDS = (0, 5, 10, 15), (1, 2)
CELLS = {'bottom20_mid': (0.80, 'mid'), 'bottom30_mid': (0.70, 'mid')}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context():
    panel = Panel(MANIFEST)
    sessions = panel.sessions[(panel.sessions >= pd.Timestamp(START)) &
                              (panel.sessions <= pd.Timestamp(END))]
    features = U.rolling_features(panel, with_st=True)
    adv = features['adv'].reindex(sessions)
    codes = [c for c in adv.columns if not str(c).startswith('bj.')]
    adv = adv.reindex(columns=codes)
    eligible = (features['history'].reindex(sessions).ge(253) & adv.notna() & adv.ge(5e6)
                & ~features['st'].reindex(sessions).eq(1)).reindex(columns=codes)
    turnover = panel.field('turnover').reindex(sessions).reindex(columns=codes)
    turnover20 = turnover.rolling(20, min_periods=10).mean()
    return sessions, adv, eligible, turnover20


def cell_codes(day, adv, eligible, turnover20, share, tercile):
    band = adv.loc[day][eligible.loc[day]]
    if len(band) < 100:
        return []
    desc = band.sort_values(ascending=False, kind='mergesort')
    members = desc.index[int(np.ceil(share * len(desc))):]
    values = turnover20.loc[day].reindex(members).dropna()
    if len(values) < 50:
        return list(members)
    low, high = values.quantile([1 / 3, 2 / 3])
    if tercile == 'low':
        keep = values.index[values <= low]
    elif tercile == 'high':
        keep = values.index[values > high]
    else:
        keep = values.index[(values > low) & (values <= high)]
    return list(keep)


def plans(sessions, adv, eligible, turnover20, share, tercile):
    out = {}
    for phase in PHASES:
        grid = U.decision_grid(sessions, START, END, phase, 20)
        for seed in SEEDS:
            schedule = {}
            for day in grid:
                codes = cell_codes(day, adv, eligible, turnover20, share, tercile)
                chosen = random_order(codes, seed)[:U.HOLDINGS] if codes else []
                schedule[day] = {code: GROSS / len(chosen) for code in chosen} if chosen else {}
            out[f'p{phase}_s{seed}'] = p4_ledger.to_targets(schedule)
    return out


def stats(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
    mdd = float((nav / nav.cummax() - 1).min())
    yearly = (nav.pct_change().fillna(0) + 1).groupby(nav.index.year).prod() - 1
    return cagr, mdd, {int(k): float(v) for k, v in yearly.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=(*CELLS, 'both'), default='bottom20_mid')
    parser.add_argument('--cost', choices=('1', '2', 'both'), default='1')
    args = parser.parse_args()
    sessions, adv, eligible, turnover20 = context()
    base = json.loads(BASELINE.read_text())['cells']['800000']
    baseline = {'1': {'cagr': base['1']['stock_cagr'], 'mdd': base['1']['stock_mdd']},
                '2': {'cagr': base['2']['stock_cagr'], 'mdd': base['2']['stock_mdd']}}
    cells = list(CELLS) if args.cell == 'both' else [args.cell]
    costs = (1.0, 2.0) if args.cost == 'both' else (float(args.cost),)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'strategy': 'P8_POOL2D', 'stage': 'B', 'capital': CAPITAL, 'gross': GROSS,
              'prereg': str(PREREG.relative_to(ROOT)), 'prereg_sha256': sha256(PREREG),
              'manifest': MANIFEST, 'baseline': baseline, 'cells': {}}
    for cell in cells:
        share, tercile = CELLS[cell]
        plan_set = plans(sessions, adv, eligible, turnover20, share, tercile)
        cell_rows = {}
        for cost in costs:
            rows = []
            for name, targets in plan_set.items():
                result = p4_ledger.replay(targets, start=START, end=END, cost_mult=cost,
                                          dataset=None, capital_cny=CAPITAL, gross=GROSS,
                                          suspension_evidence=SuspensionEvidence())
                cagr, mdd, yearly = stats(result['nav'])
                rows.append({'path': name, 'cagr': cagr, 'mdd': mdd,
                             'fills': len(result['fills']),
                             'checks': result['summary']['account_checks_passed'],
                             'yearly': yearly})
                print(json.dumps({'cell': cell, 'cost': int(cost), **rows[-1]}, ensure_ascii=False),
                      flush=True)
            frame = pd.DataFrame(rows)
            key = str(int(cost))
            cell_rows[key] = {
                'cagr_median': float(frame.cagr.median()), 'mdd_median': float(frame.mdd.median()),
                'cagr_min': float(frame.cagr.min()), 'mdd_worst': float(frame.mdd.min()),
                'paths': int(len(frame)),
                'paths_above_minus_36_6': int((frame.mdd >= -0.366).sum()),
                'gates': {
                    'cagr_ge_baseline_plus_1pp': bool(frame.cagr.median() >= baseline[key]['cagr'] + 0.01),
                    'cost2_rule': bool(frame.cagr.median() >= baseline[key]['cagr'] + 0.01),
                    'mdd_not_worse_than_minus_36_6': bool(frame.mdd.median() >= -0.366),
                },
            }
        report['cells'][cell] = cell_rows
    report['promotion_approved'] = False
    (OUT / f'report_{args.cell}_cost{args.cost}.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
