#!/usr/bin/env python3
"""P8_G0_60: lot-aware 60/40 re-run under the Calmar ranking.

The stock sleeve and the G0 rules are unchanged; only the capital split moves from 65/35 to
60/40. This answers whether the higher nominal Calmar of the 60/40 configuration survives
board lots, T+1, suspensions and costs at 800k.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import p4_gross as G
import p5_base_lot_replay as B
from quantlab import p4_ledger

OUT = ROOT / 'output/p8_g0_60_20261018'
PREREG = ROOT / 'docs/prereg/P8_G0_60.md'
BASELINE = ROOT / 'output/p5_base_lot_20261018_r1/report.json'
CAPITAL, EQUITY_SHARE = 800_000.0, 0.60


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(frame):
    out = {}
    for cost, group in frame.groupby('cost'):
        cagr = float(group.blended_cagr.median())
        mdd = float(group.blended_mdd.median())
        worst = float(group.blended_mdd.min())
        out[str(int(cost))] = {'cagr': cagr, 'mdd': mdd,
                               'calmar': cagr / abs(mdd) if mdd < 0 else float('inf'),
                               'worst_path_mdd': worst, 'paths': int(len(group))}
    return out


def main():
    plans = G.plans()
    targets = {name: p4_ledger.to_targets(plan) for name, plan in plans.items()}
    codes = sorted({code for rows in targets.values() for row in rows
                    for code in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    sessions = dataset.calendar[(dataset.calendar >= pd.Timestamp(B.START)) &
                                (dataset.calendar <= pd.Timestamp(B.END))]
    OUT.mkdir(parents=True, exist_ok=True)
    for cost in (1.0, 2.0):
        B.run_config(CAPITAL, cost, plans, targets, dataset, sessions, OUT, 'calendar', EQUITY_SHARE)
    frame = pd.concat([pd.read_csv(OUT / 'runs' / f'cap{int(CAPITAL)}_cost{int(cost)}.csv')
                       for cost in (1, 2)], ignore_index=True)
    frame.to_csv(OUT / 'summary.csv', index=False)
    cells = summarize(frame)
    base = json.loads(BASELINE.read_text())['cells']['800000']
    comparison = {}
    for cost in ('1', '2'):
        base_cagr = base[cost]['blended_cagr']
        base_mdd = base[cost]['blended_mdd']
        comparison[cost] = {
            'p5_base': {'cagr': base_cagr, 'mdd': base_mdd,
                        'calmar': base_cagr / abs(base_mdd)},
            'g0_60': cells[cost],
            'delta_cagr_pp': (cells[cost]['cagr'] - base_cagr) * 100,
            'delta_mdd_pp': (cells[cost]['mdd'] - base_mdd) * 100,
            'calmar_better': cells[cost]['calmar'] > base_cagr / abs(base_mdd)}
    gates = {'cost1_calmar_better': comparison['1']['calmar_better'],
             'cost2_calmar_better': comparison['2']['calmar_better'],
             'cost1_cagr_ge_13_0': cells['1']['cagr'] >= 0.13,
             'cost2_cagr_ge_11_0': cells['2']['cagr'] >= 0.11}
    gates['passed'] = all(gates.values())
    report = {'strategy': 'P8_G0_60', 'stage': 'A', 'capital': CAPITAL,
              'equity_share': EQUITY_SHARE, 'quarter_basis': 'calendar',
              'prereg': str(PREREG.relative_to(ROOT)), 'prereg_sha256': sha256(PREREG),
              'window': [B.START, B.END], 'cells': cells, 'comparison': comparison,
              'gates': gates, 'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
