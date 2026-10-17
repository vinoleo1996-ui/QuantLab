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


def industry_map_at(scores, day):
    """Industry labels usable at ``day``: latest snapshot on or before the signal date."""
    scores = scores.copy()
    scores['date'] = pd.to_datetime(scores['date'])
    available = scores[scores['date'] <= pd.Timestamp(day)]
    if available.empty:
        return {}
    latest = available['date'].max()
    snapshot = available[available['date'] == latest].drop_duplicates('code', keep='last')
    return snapshot.set_index('code')['industry'].to_dict()


def panel_value(dataset, code, day, field):
    frame = dataset.panels.get(code)
    if frame is None or day not in frame.index or field not in frame.columns:
        return float('nan')
    try:
        value = float(frame.at[day, field])
    except (TypeError, ValueError):
        return float('nan')
    return value if np.isfinite(value) else float('nan')


def execution_check(dataset, code, day, side='buy'):
    """Next-open execution gate shared by the landing order list and the forward snapshot."""
    price = panel_value(dataset, code, day, 'open')
    if not np.isfinite(price) or price <= 0:
        return price, 'no_open_or_suspended'
    if side == 'sell':
        lower = panel_value(dataset, code, day, 'lower_limit')
        if np.isfinite(lower) and price <= lower * 1.005:
            return price, 'limit_down_cannot_sell'
        return price, None
    if panel_value(dataset, code, day, 'is_st_at_open') == 1:
        return price, 'st_at_open'
    upper = panel_value(dataset, code, day, 'upper_limit')
    lower = panel_value(dataset, code, day, 'lower_limit')
    if np.isfinite(upper) and price >= upper * 0.995:
        return price, 'limit_up_cannot_buy'
    if np.isfinite(lower) and price <= lower * 1.005:
        return price, 'limit_down_check_manually'
    return price, None


def adv60(dataset, code, day):
    frame = dataset.panels.get(code)
    if frame is None or 'amount' not in frame.columns:
        return 0.0
    value = float(frame.loc[:day, 'amount'].dropna().tail(60).mean())
    return value if np.isfinite(value) else 0.0


def lot_shares(value, price):
    if not np.isfinite(value) or not np.isfinite(price) or price <= 0:
        return 0
    return int(np.floor(value / (price * 100))) * 100


def limit_price(open_price, side, upper=None, lower=None, buffer=0.0, tick=0.01):
    """Buffered limit around the next open, capped inside the daily price-limit band."""
    if not np.isfinite(open_price) or open_price <= 0:
        return float('nan'), 'no_open'
    raw = open_price * (1 + buffer) if side == 'buy' else open_price * (1 - buffer)
    capped = False
    if side == 'buy' and upper is not None and np.isfinite(upper) and raw > upper:
        raw, capped = upper, True
    if side == 'sell' and lower is not None and np.isfinite(lower) and raw < lower:
        raw, capped = lower, True
    return round(np.floor(raw / tick) * tick, 3 if tick < 0.01 else 2), ('rule_band' if capped else None)


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
    end = min(pd.Timestamp(signal_date), sessions[-1])
    grid = [d for d in U.decision_grid(sessions, '2018-01-02', end, 0, 20)
            if d <= pd.Timestamp(signal_date)]
    day = grid[-1]
    a = f['adv'].loc[day]; h = f['history'].loc[day]; st = f['st'].loc[day]
    keep = (h >= 252) & a.notna() & (a >= 5e6) & ~st.eq(1)
    pool = [c for c in a[keep].index if not c.startswith('bj.')]
    scores = pd.read_parquet(P.OUT / 'scores.parquet', columns=['date', 'code', 'industry'])
    ind_map = industry_map_at(scores, day)
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
    ap.add_argument('--limit-buffer', type=float, default=0.01,
                    help='limit offset from next open; buy = open*(1+b), sell = open*(1-b)')
    ap.add_argument('--sweep', action='store_true',
                    help='diagnostic only: sweep residual cash into 511010 (not in P5_BASE prereg)')
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
    limit_counts = {'rule_band': 0, 'open_buffer': 0}
    for code in codes:
        w = plan[code] * eq_share
        price, reason = execution_check(dataset, code, t)
        if reason:
            skipped.append({'code': code, 'reason': reason, 'fill_status': 'skipped'}); continue
        shares = lot_shares(w * args.capital, price)
        if shares <= 0:
            skipped.append({'code': code, 'reason': 'lot_too_small', 'price': round(price, 2),
                            'fill_status': 'skipped'}); continue
        cost = shares * price
        part = cost / adv60(dataset, code, t) if adv60(dataset, code, t) > 0 else 1.0
        if part > 0.01:
            skipped.append({'code': code, 'reason': f'participation_{part:.3f}',
                            'fill_status': 'skipped'}); continue
        limit, limit_reason = limit_price(
            price, 'buy', panel_value(dataset, code, t, 'upper_limit'),
            panel_value(dataset, code, t, 'lower_limit'), args.limit_buffer)
        limit_counts['rule_band' if limit_reason == 'rule_band' else 'open_buffer'] += 1
        cash_needed += cost
        orders.append({'code': code, 'leg': 'stock', 'side': 'buy', 'shares': shares,
                       'open': round(price, 2), 'limit': limit, 'limit_basis': limit_reason or 'open_buffer',
                       'value': round(cost, 2), 'weight': round(w, 4), 'participation': round(part, 4),
                       'fill_status': 'pending_next_open', 'fill_reason': None})
    for name, code in (('gold', 'sh518880'), ('treasury', 'sh511260'), ('nasdaq', 'sh513100')):
        w = sleeve[name]
        if w <= 0:
            skipped.append({'code': code, 'reason': 'qdii_paused_zero_weight',
                            'fill_status': 'skipped'}); continue
        price = etf_px(code)
        shares = lot_shares(w * args.capital, price)
        if shares <= 0:
            skipped.append({'code': code, 'reason': 'lot_too_small', 'fill_status': 'skipped'}); continue
        limit, limit_reason = limit_price(price, 'buy', buffer=args.limit_buffer, tick=0.001)
        limit_counts['rule_band' if limit_reason == 'rule_band' else 'open_buffer'] += 1
        cost = shares * price
        cash_needed += cost
        orders.append({'code': code, 'leg': 'etf', 'side': 'buy', 'shares': shares,
                       'open': round(price, 3), 'limit': limit, 'limit_basis': limit_reason or 'open_buffer',
                       'value': round(cost, 2), 'weight': round(w, 4), 'participation': None,
                       'fill_status': 'pending_next_open', 'fill_reason': None})
    # Residual sweep into 511010 is not part of the P5_BASE prereg; keep it an explicit diagnostic.
    residual = args.capital - cash_needed
    sweep = None
    if args.sweep and residual > 5000:
        price = etf_px('sh511010')
        if np.isfinite(price) and price > 0:
            shares = lot_shares(residual, price)
            if shares > 0:
                cash_needed += shares * price
                orders.append({'code': 'sh511010', 'leg': 'sweep', 'side': 'buy', 'shares': shares,
                               'open': round(price, 3), 'limit': round(price, 3),
                               'limit_basis': 'open_buffer', 'value': round(shares * price, 2),
                               'weight': round(shares * price / args.capital, 4), 'participation': None,
                               'fill_status': 'pending_next_open', 'fill_reason': None})
                sweep = {'code': 'sh511010', 'shares': shares, 'value': round(shares * price, 2)}

    summary = {
        'signal_date': str(day.date()), 'execution_date': str(t.date()),
        'capital': args.capital, 'equity_share': eq_share, 'gross': args.gross, 'seed': args.seed,
        'stock_names': len(codes), 'stock_orders': sum(1 for o in orders if o['leg'] == 'stock'),
        'etf_orders': sum(1 for o in orders if o['leg'] == 'etf'),
        'qdii_paused': paused, 'nasdaq_premium': round(nasdaq_prem, 4),
        'limit_buffer': args.limit_buffer, 'limit_basis': limit_counts,
        'fill_status': {status: sum(1 for o in orders if o['fill_status'] == status)
                        for status in sorted({o['fill_status'] for o in orders})},
        'cash_needed': round(cash_needed, 2), 'cash_sweep': sweep, 'skipped': len(skipped), 'skipped_detail': skipped,
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
