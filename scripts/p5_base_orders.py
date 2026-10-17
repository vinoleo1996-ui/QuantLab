#!/usr/bin/env python3
"""P5_BASE landing v1: end-to-end order list for the T+1 flow (no orders are sent).

  .venv/bin/python scripts/p5_base_orders.py --data-root <root> --signal-date 2026-09-17 \
      [--capital 800000] [--holdings current.csv]

Stock leg: 65% x POOL50_G97 rules (50 names, 97% gross, equal weight, SW L1 <= 12).
Sleeve:    35% G0 fixed 3:3:2 (gold 13.125 / 10y treasury 13.125 / nasdaq 8.75),
           QDII premium pause overflow split 1:1 between gold and treasury.
Cash:      0%. Outputs orders.csv + summary.json under output/p5_base_orders/<date>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('QUANTLAB_ROOT', '/Users/agiuser/Desktop/QuantLab'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def build_stock_targets(data_root: Path, signal_date: str, seed: int, gross: float):
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
    scores = pd.read_parquet(P.OUT / 'scores.parquet', columns=['code', 'industry'])
    ind_map = scores.groupby('code').industry.last().to_dict()
    ind = {c: ind_map.get(c, c) for c in pool}
    order = lambda d: random_order(pool, seed)
    plan = schedule([day], order, lambda d: ind, gross=gross)[day]
    return plan, day, panel, sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(ROOT))
    ap.add_argument('--signal-date', required=True)
    ap.add_argument('--capital', type=float, default=800000.0)
    ap.add_argument('--cash', type=float, default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gross', type=float, default=0.97)
    ap.add_argument('--equity-share', type=float, default=0.65)
    args = ap.parse_args()
    root = Path(args.data_root)
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'scripts'))
    from quantlab import p4_alloc as A
    import p4_alloc as R

    plan, day, panel, sessions = build_stock_targets(root, args.signal_date, args.seed, args.gross)
    codes = sorted(plan)
    targets = [{'signal_date': day, 'weights': dict(plan), 'active': codes, 'sell': [], 'sleeve': 0}]
    from quantlab import p4_ledger
    dataset = p4_ledger.load_dataset(codes)
    nxt = dataset.calendar[dataset.calendar > day]
    t = nxt[0]
    eq_share = args.equity_share
    sleeve = {'gold': 0.375 * (1 - eq_share), 'treasury': 0.375 * (1 - eq_share), 'nasdaq': 0.25 * (1 - eq_share)}

    def stock_px(code, field):
        fr = dataset.panels[code]
        try:
            v = fr.loc[t, field]
        except KeyError:
            return float('nan')
        return float(v) if v is not None else float('nan')

    def adv60(code):
        fr = dataset.panels[code]
        return float(fr.loc[:t, 'amount'].dropna().tail(60).mean())

    # QDII premium pause: overflow 1:1 to gold/treasury
    etf_bars = pd.read_parquet(root / 'data/smallcap_assets/etf_bars.parquet')
    wide = etf_bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    prem = R.qdii_premium_frame(sessions[(sessions >= day) & (sessions <= t)])
    nasdaq_prem = float(prem.nasdaq.dropna().iloc[-1]) if len(prem.nasdaq.dropna()) else 0.0
    paused = nasdaq_prem > 0.03
    if paused:
        delta = sleeve['nasdaq']
        sleeve = {'gold': sleeve['gold'] + delta / 2, 'treasury': sleeve['treasury'] + delta / 2, 'nasdaq': 0.0}

    def etf_px(code):
        s = wide[code].loc[:t].dropna()
        return float(s.loc[t]) if t in s.index else float('nan')

    orders, skipped = [], []
    cash_needed = 0.0
    for code in codes:
        w = plan[code] * eq_share
        price = stock_px(code, 'open')
        st = stock_px(code, 'is_st_at_open')
        upper = stock_px(code, 'upper_limit')
        lower = stock_px(code, 'lower_limit')
        if not np.isfinite(price) or price <= 0:
            skipped.append({'code': code, 'reason': 'no_open_or_suspended'}); continue
        if st == 1:
            skipped.append({'code': code, 'reason': 'st_at_open'}); continue
        if np.isfinite(upper) and price >= upper * 0.995:
            skipped.append({'code': code, 'reason': 'limit_up_cannot_buy'}); continue
        if np.isfinite(lower) and price <= lower * 1.005:
            skipped.append({'code': code, 'reason': 'limit_down_check_manually'}); continue
        lots = int(np.floor(w * args.capital / (price * 100)))
        if lots <= 0:
            skipped.append({'code': code, 'reason': 'lot_too_small', 'price': round(price, 2)}); continue
        shares = lots * 100
        cost = shares * price
        part = cost / adv60(code) if adv60(code) > 0 else 1.0
        if part > 0.01:
            skipped.append({'code': code, 'reason': f'participation_{part:.3f}'}); continue
        cash_needed += cost
        orders.append({'code': code, 'leg': 'stock', 'side': 'buy', 'shares': shares,
                       'limit': round(price, 2), 'value': round(cost, 2),
                       'weight': round(w, 4), 'participation': round(part, 4)})
    for name, code in (('gold', 'sh518880'), ('treasury', 'sh511260'), ('nasdaq', 'sh513100')):
        w = sleeve[name]
        if w <= 0:
            skipped.append({'code': code, 'reason': 'qdii_paused_zero_weight'}); continue
        price = etf_px(code)
        lots = int(np.floor(w * args.capital / (price * 100)))
        if lots <= 0:
            skipped.append({'code': code, 'reason': 'lot_too_small'}); continue
        shares = lots * 100
        cost = shares * price
        cash_needed += cost
        orders.append({'code': code, 'leg': 'etf', 'side': 'buy', 'shares': shares,
                       'limit': round(price, 3), 'value': round(cost, 2), 'weight': round(w, 4),
                       'participation': None})
    summary = {
        'signal_date': str(day.date()), 'execution_date': str(t.date()),
        'capital': args.capital, 'equity_share': eq_share, 'gross': args.gross, 'seed': args.seed,
        'stock_names': len(codes), 'stock_orders': sum(1 for o in orders if o['leg'] == 'stock'),
        'etf_orders': sum(1 for o in orders if o['leg'] == 'etf'),
        'qdii_paused': paused, 'nasdaq_premium': round(nasdaq_prem, 4),
        'cash_needed': round(cash_needed, 2), 'skipped': len(skipped), 'skipped_detail': skipped,
        'gates': {'cash_ok': cash_needed <= args.capital,
                  'stock_coverage': round(sum(1 for o in orders if o['leg'] == 'stock') / len(codes), 3)},
    }
    out = root / 'output' / 'p5_base_orders' / str(day.date())
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(orders).to_csv(out / 'orders.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
