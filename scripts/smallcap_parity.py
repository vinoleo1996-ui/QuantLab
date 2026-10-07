#!/usr/bin/env python3
"""Parity check: replay the frozen 2018-extended broad-basket targets in the fast simulator.

The research Ledger reported 14.63 percent CAGR and -22.45 percent drawdown for
these exact targets. The simulator is accepted for the seed/phase experiments only
if the CAGR gap is within 1.0 percentage point and drawdown within 2.0 points
(tolerance registered before running). Delisting haircut 0 matches the Ledger;
haircut 0.5 is the v4 convention and is reported separately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from quantlab.smallcap.panel import Panel, latest_version
from quantlab.smallcap.sim import Costs, Market, metrics, simulate

FROZEN = ROOT / 'output/continuous_band_2018ext_20260928'
OUT = ROOT / 'output/smallcap_rebuild_20261006/parity'
TOLERANCE = {'cagr_pp': 1.0, 'drawdown_pp': 2.0}


def to_panel(code):
    number, market = code.split('.')
    return f'{market.lower()}.{number}'


def load_schedule(path):
    rows = json.loads(Path(path).read_text())['targets']
    return {row['signal_date']: {to_panel(code): weight for code, weight in row['weights'].items()} for row in rows}


def delist_map(panel):
    table = panel.table('delistings')
    return dict(zip(table.code, table.last_bar))


def main():
    panel = Panel(latest_version())
    schedule = load_schedule(FROZEN / 'targets.json')
    market = Market.from_panel(panel, '2018-01-02', '2026-09-22', delist=delist_map(panel))
    reference = json.loads((FROZEN / 'cost1/summary.json').read_text())
    results = {}
    for haircut in (0.0, 0.5):
        run = simulate(market, schedule, costs=Costs(delist_haircut=haircut))
        stats = metrics(run['nav'])
        stats.update(fees=run['fees'], slippage=run['slippage'], fills=run['fills'],
                     delist_events=len(run['events']))
        results[f'haircut_{haircut}'] = stats
    ledger_nav = pd.read_csv(FROZEN / 'cost1/daily_returns.csv', index_col=0, parse_dates=True)['daily_return']
    states = OUT / 'ledger_states.csv'
    ledger_gross = None
    if states.exists():
        frame = pd.read_csv(states)
        ledger_gross = float((frame.held / frame.nav).loc[frame.held > 0].median())
        scale = ledger_gross / 0.90
        matched = {day: {code: weight * scale for code, weight in weights.items()} for day, weights in schedule.items()}
        run = simulate(market, matched, costs=Costs(delist_haircut=0.0))
        stats = metrics(run['nav'])
        stats.update(median_gross=float(run['gross'][run['gross'] > 0].median()))
        results['exposure_matched_haircut_0.0'] = stats
    key = 'exposure_matched_haircut_0.0' if 'exposure_matched_haircut_0.0' in results else 'haircut_0.0'
    gap_cagr = (results[key]['cagr'] - reference['cagr']) * 100
    gap_dd = (results[key]['max_drawdown'] - reference['max_drawdown']) * 100
    passed = abs(gap_cagr) <= TOLERANCE['cagr_pp'] and abs(gap_dd) <= TOLERANCE['drawdown_pp']
    out = {'reference': {k: reference[k] for k in ('cagr', 'max_drawdown', 'final_equity', 'fees', 'slippage', 'fills')},
           'ledger_median_gross': ledger_gross,
           'ledger_underinvestment_cause': ('buys execute in code order after sells; later codes hit the 10 percent '
                                            'cash reserve and stay unfilled until the next signal'),
           'parity_basis': key,
           'simulator': results, 'gap_cagr_pp': gap_cagr, 'gap_drawdown_pp': gap_dd,
           'tolerance': TOLERANCE, 'passed': passed, 'panel_version': panel.meta['version'],
           'ledger_sessions': int(len(ledger_nav)), 'promotion_approved': False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'parity.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
