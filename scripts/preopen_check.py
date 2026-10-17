#!/usr/bin/env python3
"""Pre-open check and order list for the T+1 semi-automatic flow.

Builds the target holdings of the P5_BASE stock leg (POOL50_G97 rules) at the last
rebalance on or before SIGNAL_DATE, then checks every name against the next session's
market data (ST, suspension, price limits, lot size, participation, cash) and writes
an order list. It never trades; it produces what a human or an OMS would submit.

  .venv/bin/python scripts/preopen_check.py --data-root /path/to/QuantLab \
      --signal-date 2026-09-18 [--capital 800000] [--cash 800000] [--seed 1] [--gross 0.97]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_targets(data_root: Path, signal_date: str, seed: int, gross: float):
    sys.path.insert(0, str(data_root))
    sys.path.insert(0, str(data_root / 'scripts'))
    import p4_alpha as P
    from quantlab.smallcap import universe as U
    from quantlab.smallcap.panel import Panel
    from quantlab.alpha.portfolio import schedule, random_order

    panel = Panel(P.PANEL)
    f = U.rolling_features(panel, with_st=True)
    sessions = panel.sessions
    grid = [d for d in U.decision_grid(sessions, '2018-01-02', '2026-09-22', 0, 20)
            if d <= pd.Timestamp(signal_date)]
    day = grid[-1]
    a = f['adv'].loc[day]; h = f['history'].loc[day]; st = f['st'].loc[day]
    keep = (h >= 252) & a.notna() & (a >= 5e6) & ~st.eq(1)
    pool = [c for c in a[keep].index if not c.startswith('bj.')]
    order = lambda d: random_order(pool, seed)
    scores = pd.read_parquet(data_root / 'output' / 'p4_alpha_20261012' / 'scores.parquet',
                             columns=['code', 'industry'])
    ind_map = scores.groupby('code').industry.last().to_dict()
    ind = {c: ind_map.get(c, c) for c in pool}
    plan = schedule([day], order, lambda d: ind, gross=gross)[day]
    return plan, day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/Users/agiuser/Desktop/QuantLab')
    ap.add_argument('--signal-date', required=True)
    ap.add_argument('--capital', type=float, default=800000.0)
    ap.add_argument('--cash', type=float, default=800000.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gross', type=float, default=0.97)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    root = Path(args.data_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'scripts'))
    from quantlab import p4_ledger
    from quantlab.p4_mandate import MANDATE

    plan, day = build_targets(root, args.signal_date, args.seed, args.gross)
    codes = sorted(plan)
    targets = [{'signal_date': day, 'weights': dict(plan), 'active': codes, 'sell': [], 'sleeve': 0}]
    dataset = p4_ledger.load_dataset(codes)
    sessions = dataset.calendar
    nxt = sessions[sessions > day]
    if len(nxt) == 0:
        raise SystemExit('no session after signal date')
    t = nxt[0]

    def px(code, field):
        frame = dataset.panels[code]
        try:
            v = frame.loc[t, field]
        except KeyError:
            return float('nan')
        return float(v) if v is not None else float('nan')

    def adv60(code):
        frame = dataset.panels[code]
        amt = frame.loc[:t, 'amount'].dropna()
        return float(amt.tail(60).mean())

    nav = args.cash if args.cash else args.capital
    orders, rejects, skipped = [], [], []
    cash_needed = 0.0
    for code in codes:
        w = plan[code]
        price = px(code, 'open')
        st = px(code, 'is_st_at_open')
        upper = px(code, 'upper_limit')
        lower = px(code, 'lower_limit')
        volume = px(code, 'volume')
        amount = px(code, 'amount')
        if not np.isfinite(price) or price <= 0:
            rejected = 'no_open_price_or_suspended'
        elif st == 1:
            rejected = 'st_at_open'
        elif np.isfinite(upper) and price >= upper * 0.995:
            rejected = 'limit_up_cannot_buy'
        elif np.isfinite(lower) and price <= lower * 1.005:
            rejected = 'limit_down_check_manually'
        else:
            rejected = None
        if rejected:
            skipped.append({'code': code, 'reason': rejected})
            continue
        value = w * nav
        lots = int(np.floor(value / (price * 100)))
        if lots <= 0:
            skipped.append({'code': code, 'reason': 'lot_too_small'})
            continue
        shares = lots * 100
        cost = shares * price
        part = cost / adv60(code) if adv60(code) > 0 else 1.0
        if part > MANDATE.max_stock_weight * 0 + 0.01:
            skipped.append({'code': code, 'reason': f'participation_{part:.3f}_gt_1pct_adv'})
            continue
        cash_needed += cost
        orders.append({'code': code, 'side': 'buy', 'shares': shares, 'limit': round(price, 2),
                       'value': round(cost, 2), 'weight': w, 'adv60_m': round(adv60(code) / 1e6, 1),
                       'participation': round(part, 4)})
    summary = {
        'signal_date': str(day.date()), 'execution_date': str(t.date()),
        'capital': args.capital, 'gross': args.gross, 'seed': args.seed,
        'target_names': len(codes), 'orders': len(orders), 'cash_needed': round(cash_needed, 2),
        'cash_after': round(args.cash - cash_needed, 2), 'skipped': len(skipped),
        'skipped_detail': skipped,
        'gates': {'cash_ok': cash_needed <= args.cash,
                  'coverage': round(len(orders) / len(codes), 3)},
    }
    out = Path(args.out) if args.out else root / 'output' / f'preopen_{day.date()}'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(orders).to_csv(out / 'orders.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
