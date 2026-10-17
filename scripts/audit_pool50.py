#!/usr/bin/env python3
"""POOL50 execution audit: exposure, cash, holdings, industry cap, order coverage, data issues.

Reads the frozen score book and replays the 20-day random-50 plan through the share
ledger. Industry weights use the signal-day classification (no full-sample ``.last()``).
Writes snapshots.csv, coverage.csv and summary.json under --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab import p4_ledger
from quantlab.alpha.portfolio import random_order, schedule

SCORES = ROOT / 'output/p4_alpha_20261012/scores.parquet'


def industry_at(by_day, days, day):
    prior = [d for d in days if d <= day]
    if not prior:
        return {}
    return dict(zip(by_day[prior[-1]].code, by_day[prior[-1]].industry))


def stale_runs(account, max_stale_days=5):
    """Longest consecutive held-stale run per code, with dates, straight from the replay."""
    running, best = {}, {}
    for state in account:
        stale = set(state.get('stale') or [])
        for code in list(running):
            if code not in stale:
                running.pop(code)
        for code in stale:
            start = running[code][0] if code in running else state['date']
            days = running[code][2] + 1 if code in running else 1
            running[code] = (start, state['date'], days)
            if code not in best or days > best[code][2]:
                best[code] = running[code]
    return {code: {'start': str(pd.Timestamp(start).date()), 'end': str(pd.Timestamp(end).date()),
                   'days': int(days)}
            for code, (start, end, days) in best.items() if days > max_stale_days}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2018-01-02')
    parser.add_argument('--end', default='2026-09-22')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--capital', type=float, default=800_000.0)
    parser.add_argument('--out', default=str(ROOT / 'output/audit_20261017'))
    args = parser.parse_args()

    scores = pd.read_parquet(SCORES, columns=['date', 'code', 'industry'])
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    rebalance = days[0::4]
    plan = schedule(rebalance, lambda day: random_order(list(by_day[day].code), args.seed),
                    lambda day: dict(zip(by_day[day].code, by_day[day].industry)))
    targets = p4_ledger.to_targets(plan)
    codes = sorted({c for row in targets for c in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    result = p4_ledger.replay(targets, start=args.start, end=args.end, cost_mult=1,
                              dataset=dataset, capital_cny=args.capital, gross=0.97)

    snapshots = []
    for state in result['account']:
        quantities, marks, nav = state['quantities'], state['marks'], state['nav']
        stock_value = sum(shares * marks.get(code, 0.0) for code, shares in quantities.items())
        industry = industry_at(by_day, days, pd.Timestamp(state['date']))
        weights = Counter()
        for code, shares in quantities.items():
            weights[industry.get(code, 'unclassified')] += shares * marks.get(code, 0.0) / nav
        snapshots.append(dict(date=state['date'], nav=nav, cash=state['cash'],
                              stock_value=stock_value, n=len(quantities),
                              cash_ratio=state['cash'] / nav if nav else np.nan,
                              max_ind_weight=max(weights.values()) if weights else 0.0,
                              max_ind=max(weights, key=weights.get) if weights else ''))
    snapshots = pd.DataFrame(snapshots)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(out / 'pool50_snapshots.csv', index=False)

    orders_by_date = Counter(order.get('date') for order in result['orders'])
    fills_by_date = Counter(fill.get('date') for fill in result['fills'])
    coverage = pd.DataFrame([dict(date=str(pd.Timestamp(day).date()), targets=len(plan.get(day, {})),
                                  orders=orders_by_date.get(str(pd.Timestamp(day).date()), 0),
                                  fills=fills_by_date.get(str(pd.Timestamp(day).date()), 0))
                             for day in rebalance])
    coverage.to_csv(out / 'pool50_coverage.csv', index=False)

    delistings = dataset.delistings.set_index('code') if len(dataset.delistings) else pd.DataFrame()
    end_holdings = result['account'][-1]['quantities']
    summary = {
        'window': [args.start, args.end], 'capital': args.capital, 'seed': args.seed,
        'source': str(SCORES.relative_to(ROOT)),
        'industry_basis': 'signal_day',
        'stock_exposure': {'mean': float(snapshots.stock_value.div(snapshots.nav).mean()),
                           'median': float(snapshots.stock_value.div(snapshots.nav).median()),
                           'min': float(snapshots.stock_value.div(snapshots.nav).min()),
                           'max': float(snapshots.stock_value.div(snapshots.nav).max())},
        'cash_ratio': {'mean': float(snapshots.cash_ratio.mean()),
                       'median': float(snapshots.cash_ratio.median())},
        'holdings': {'mean': float(snapshots.n.mean()), 'min': int(snapshots.n.min()),
                     'max': int(snapshots.n.max())},
        'industry_weight': {'mean': float(snapshots.max_ind_weight.mean()),
                            'median': float(snapshots.max_ind_weight.median()),
                            'max': float(snapshots.max_ind_weight.max()),
                            'days_over_25pct': int((snapshots.max_ind_weight > 0.25).sum()),
                            'days': int(len(snapshots))},
        'orders': len(result['orders']),
        'order_status': dict(Counter(order.get('status') for order in result['orders'])),
        'fills': len(result['fills']),
        'data_issues': len(result['data_issues']),
        'data_issue_reasons': dict(Counter(str(issue.get('reason')) for issue in result['data_issues'])),
        'missing_panels': result['summary']['missing_panels'],
        'delisting_liquidations': result['summary']['delisting_liquidations'],
        'data_quality_ok': result['summary']['data_quality_ok'],
        'stale_held_max_days': result['summary']['stale_held_max_days'],
        'stale_held_codes': result['summary'].get('stale_held_codes', {}),
        'stale_held_runs': stale_runs(result['account']),
        'end_holdings': len(end_holdings),
        'delisted_end_holdings': [[code, str(delistings.loc[code, 'delist_date'])]
                                  for code in end_holdings if code in delistings.index],
        'promotion_approved': False,
    }
    (out / 'pool50_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
