#!/usr/bin/env python3
"""Post-close reconciliation for the T+1 flow (landing v1, second piece).

  .venv/bin/python scripts/reconcile.py --orders output/p5_base_orders/<date>/orders.csv \
      --fills fills.csv [--capital 800000]

fills.csv: code,side,shares,price (one row per actual fill, written after the close).
Outputs output/p5_base_orders/<date>/reconcile.json: price deviation vs the order limit,
realised slippage vs the 30bp assumption, unfilled orders, cash left, position drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STOCK_SLIPPAGE_BP = 30.0
ETF_SLIPPAGE_BP = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orders', required=True)
    ap.add_argument('--fills', required=True)
    ap.add_argument('--capital', type=float, default=800000.0)
    ap.add_argument('--cash-before', type=float, default=None)
    args = ap.parse_args()

    orders = pd.read_csv(args.orders)
    fills = pd.read_csv(args.fills)
    rows = []
    for _, o in orders.iterrows():
        f = fills[fills.code == o.code]
        if f.empty:
            rows.append(dict(code=o.code, leg=o.leg, status='unfilled', order_shares=o.shares,
                             fill_shares=0, price_dev=None, slip_bp=None))
            continue
        f = f.iloc[0]
        dev = float(f.price) / float(o.limit) - 1
        basis = STOCK_SLIPPAGE_BP if o.leg != 'etf' else ETF_SLIPPAGE_BP
        slip = (abs(dev) - basis / 1e4) * 1e4
        rows.append(dict(code=o.code, leg=o.leg, status='filled', order_shares=int(o.shares),
                         fill_shares=int(f.shares), price_dev=round(dev, 5), slip_bp=round(slip, 1)))
    rec = pd.DataFrame(rows)
    filled = rec[rec.status == 'filled']
    spent = float((filled.fill_shares * pd.read_csv(args.fills).set_index('code').price.reindex(filled.code)).sum())
    cash_left = (args.cash_before if args.cash_before is not None else args.capital) - spent
    skipped = orders[~orders.code.isin(fills.code)]
    out = {
        'orders': len(orders), 'filled': int((rec.status == 'filled').sum()),
        'unfilled': int((rec.status == 'unfilled').sum()),
        'skipped_by_precheck': int(len(skipped)),
        'cash_left': round(cash_left, 2),
        'median_price_dev_bp': round(float(filled.price_dev.abs().median() * 1e4), 1) if len(filled) else None,
        'worst_price_dev_bp': round(float(filled.price_dev.abs().max() * 1e4), 1) if len(filled) else None,
        'share_of_orders_worse_than_assumed_bp': round(float((filled.slip_bp > 0).mean()), 3) if len(filled) else None,
        'lines': rec.to_dict('records'),
    }
    out_path = Path(args.orders).parent / 'reconcile.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'lines'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
