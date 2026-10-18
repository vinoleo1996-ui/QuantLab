"""Pre-replay snapshot helpers, kept only so their arithmetic stays under test.

``scripts/p5_snapshot.py`` prices the book by replaying both sleeves from 2018 (P5_FORWARD
amendment 3). These helpers instead marked and settled an *incremental* state file, and
``etf_books``/``settle_orders`` fill at a forward-filled open, which Phase 2 ruled out as an
execution price. Nothing on the production path may import them; they live here so the unit tests
that pin lot rounding, fee parity and cash conservation keep running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import p5_base_orders as O

ALPHA = ROOT / 'output/p4_alpha_20261012'
EQUITY_SHARE = 0.65
ETF_WEIGHTS = {'gold': 0.375 * 0.35, 'treasury': 0.375 * 0.35, 'nasdaq': 0.25 * 0.35}
ETF_CODES = {'gold': 'sh518880', 'treasury': 'sh511260', 'nasdaq': 'sh513100'}
ETF_NAMES = {code: name for name, code in ETF_CODES.items()}
SLIPPAGE, COMMISSION, MIN_COMMISSION, STAMP = 0.003, 0.0003, 5.0, 0.0005
MIN_TRADE, PARTICIPATION = 1000.0, 0.01


def etf_books():
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    close = bars.pivot(index='date', columns='code', values='close_raw').sort_index()
    open_ = bars.pivot(index='date', columns='code', values='open_raw').sort_index()
    close.index, open_.index = pd.to_datetime(close.index), pd.to_datetime(open_.index)
    return close.ffill(), open_.ffill()


def load_scores():
    return pd.read_parquet(ALPHA / 'scores.parquet', columns=['date', 'code', 'industry'])


def settlement_fee(value, side):
    commission = max(MIN_COMMISSION, value * COMMISSION)
    stamp = value * STAMP if side == 'sell' else 0.0
    return commission + stamp


def settle_orders(orders, open_prices, cash, holdings):
    """Fill at open plus slippage; whole-lot buys, any-size sells, no negative cash."""
    cash, holdings = float(cash), {k: int(v) for k, v in holdings.items()}
    fills, unfilled = [], []
    for order in orders:
        code, side, shares = order['code'], order['side'], int(order['shares'])
        raw = open_prices.get(code)
        if not np.isfinite(raw) or raw <= 0:
            unfilled.append(dict(order, reason='no_open_or_suspended'))
            continue
        price = raw * (1 + SLIPPAGE) if side == 'buy' else raw * (1 - SLIPPAGE)
        if side == 'buy':
            shares = min(shares, O.lot_shares(max(0.0, cash), price))
            if shares <= 0:
                unfilled.append(dict(order, reason='cash_limit'))
                continue
        else:
            shares = min(shares, holdings.get(code, 0))
            if shares <= 0:
                unfilled.append(dict(order, reason='no_holding'))
                continue
        value = shares * price
        fee = settlement_fee(value, side)
        if side == 'buy':
            if cash + 1e-9 < value + fee:
                unfilled.append(dict(order, reason='cash_limit'))
                continue
            cash -= value + fee
            holdings[code] = holdings.get(code, 0) + shares
        else:
            cash += value - fee
            holdings[code] = holdings.get(code, 0) - shares
        fills.append(dict(code=code, side=side, shares=shares, price=round(price, 4),
                          value=round(value, 2), fee=round(fee, 2)))
    return cash, {k: v for k, v in holdings.items() if v > 0}, fills, unfilled


def mark_snapshot(holdings, cash, dataset, etf_close, day, scores):
    values, stale = {}, {}
    for code, shares in holdings.items():
        if shares <= 0:
            continue
        if code in ETF_NAMES:
            series = etf_close[code].loc[:day].dropna()
            px = float(series.iloc[-1]) if len(series) else float('nan')
            if day not in etf_close.index or not np.isfinite(etf_close.at[day, code]):
                stale[code] = 'etf_forward_filled'
        else:
            px = O.panel_value(dataset, code, day, 'close')
            if not np.isfinite(px):
                frame = dataset.panels.get(code)
                series = frame.loc[:day, 'close'].dropna() if frame is not None else pd.Series(dtype=float)
                px = float(series.iloc[-1]) if len(series) else float('nan')
                stale[code] = 'stock_last_close'
        if np.isfinite(px):
            values[code] = shares * px
        else:
            values[code] = 0.0
            stale[code] = 'no_mark'
    nav = float(cash + sum(values.values()))
    stock_value = float(sum(v for c, v in values.items() if c not in ETF_NAMES))
    etf_value = float(sum(v for c, v in values.items() if c in ETF_NAMES))
    weights = {c: v / nav for c, v in values.items()} if nav else {}
    industry = O.industry_map_at(scores, day)
    industries = {}
    for code, value in values.items():
        if code in ETF_NAMES:
            continue
        key = industry.get(code, 'unknown')
        industries[key] = industries.get(key, 0.0) + value / nav if nav else 0.0
    top10 = [{'code': c, 'value': round(v, 2), 'weight': round(weights[c], 6)}
             for c, v in sorted(values.items(), key=lambda item: -item[1])[:10]]
    return dict(nav=nav, cash=float(cash), stock_value=stock_value, etf_value=etf_value,
                stock_exposure=stock_value / nav if nav else 0.0,
                etf_exposure=etf_value / nav if nav else 0.0,
                cash_weight=float(cash) / nav if nav else 0.0,
                weights=weights, industry_weights=industries, top10=top10,
                stock_names=sum(1 for c, s in holdings.items() if s > 0 and c not in ETF_NAMES),
                stale_marks=stale)


def stock_targets(plan, nav, stock_marks, holdings):
    budget = nav * EQUITY_SHARE
    targets, skipped = {}, []
    for code in sorted(set(plan) | {c for c in holdings if c not in ETF_NAMES}):
        px = stock_marks.get(code)
        if not np.isfinite(px) or px <= 0:
            skipped.append({'code': code, 'reason': 'no_mark'})
            continue
        shares = O.lot_shares(plan.get(code, 0.0) * budget, px)
        if shares == 0 and plan.get(code, 0.0) > 0:
            skipped.append({'code': code, 'reason': 'below_one_lot', 'price': round(px, 2)})
        targets[code] = shares
    return targets, skipped


def etf_targets(nav, etf_marks, holdings, premium_pause):
    current = {name: holdings.get(ETF_CODES[name], 0) * etf_marks.get(name, 0.0) for name in ETF_WEIGHTS}
    target = {name: weight * nav for name, weight in ETF_WEIGHTS.items()}
    paused = False
    if premium_pause and target['nasdaq'] > current['nasdaq']:
        freed = target['nasdaq'] - current['nasdaq']
        target['nasdaq'] = current['nasdaq']
        target['gold'] += freed / 2
        target['treasury'] += freed / 2
        paused = True
    shares = {name: O.lot_shares(target[name], etf_marks.get(name, float('nan'))) for name in target}
    return shares, {'paused': paused, 'current': current, 'target_value': target}


def apply_pending(state, dataset, etf_open, calendar, as_of):
    pending = state.get('pending_orders') or []
    execute_on = state.get('pending_execute_on')
    if not pending or not execute_on:
        return state, [], []
    fill_days = calendar[(calendar >= pd.Timestamp(execute_on)) & (calendar <= pd.Timestamp(as_of))]
    if not len(fill_days):
        return state, [], []
    fill_day = fill_days[0]
    executable, blocked = [], []
    for order in pending:
        code = order['code']
        if code in ETF_NAMES:
            raw = float(etf_open.at[fill_day, code]) if fill_day in etf_open.index else float('nan')
            reason = None if np.isfinite(raw) else 'no_open_or_suspended'
        else:
            raw, reason = O.execution_check(dataset, code, fill_day,
                                            side=order.get('side', 'buy'))
        if reason:
            blocked.append(dict(order, reason=reason, fill_day=str(fill_day.date())))
        else:
            executable.append(dict(order, _open=raw))
    open_prices = {order['code']: order['_open'] for order in executable}
    cash, holdings, fills, unfilled = settle_orders(executable, open_prices,
                                                   state['cash'], state['holdings'])
    for fill in fills:
        fill['fill_day'] = str(fill_day.date())
    state = dict(state, cash=cash, holdings=holdings, pending_orders=[], pending_execute_on=None)
    return state, fills, blocked + unfilled


def build_rebalance_orders(plan, holdings, nav, cash, stock_marks, etf_marks, adv,
                           rebalance_stocks, rebalance_etf, premium_pause):
    stock_target, skipped = stock_targets(plan, nav, stock_marks, holdings)
    etf_target, etf_meta = etf_targets(nav, etf_marks, holdings, premium_pause)
    if not rebalance_stocks:
        stock_target = {c: holdings.get(c, 0) for c in stock_target}
    if not rebalance_etf:
        etf_target = {name: holdings.get(ETF_CODES[name], 0) for name in etf_target}
    orders, cash_available = [], float(cash)

    def sell(code, side_leg, quantity, px):
        nonlocal cash_available
        value = quantity * px * (1 - SLIPPAGE)
        cash_available += value - settlement_fee(value, 'sell')
        orders.append(dict(code=code, leg=side_leg, side='sell', shares=int(quantity),
                           reference=round(px, 4), value=round(value, 2)))

    for code, target in sorted(stock_target.items()):
        held = holdings.get(code, 0)
        if target < held:
            sell(code, 'stock', held - target, stock_marks[code])
    for name, target in sorted(etf_target.items()):
        code, held = ETF_CODES[name], holdings.get(ETF_CODES[name], 0)
        if target < held:
            sell(code, 'etf', held - target, etf_marks[name])

    def buy(code, leg, quantity, px, adv_value=None):
        nonlocal cash_available
        if adv_value:
            cap = O.lot_shares(PARTICIPATION * adv_value, px)
            quantity = min(quantity, cap)
        quantity = min(quantity, O.lot_shares(max(0.0, cash_available), px * (1 + SLIPPAGE)))
        if quantity <= 0:
            skipped.append({'code': code, 'reason': 'cash_or_participation_limit'})
            return
        value = quantity * px * (1 + SLIPPAGE)
        fee = settlement_fee(value, 'buy')
        if value + fee > cash_available + 1e-9:
            skipped.append({'code': code, 'reason': 'cash_limit'})
            return
        if value < MIN_TRADE:
            skipped.append({'code': code, 'reason': 'below_min_trade_value'})
            return
        cash_available -= value + fee
        orders.append(dict(code=code, leg=leg, side='buy', shares=int(quantity),
                           reference=round(px, 4), value=round(value, 2)))

    for code, target in sorted(stock_target.items(), key=lambda item: -plan.get(item[0], 0.0)):
        held = holdings.get(code, 0)
        if target > held:
            buy(code, 'stock', target - held, stock_marks[code], adv.get(code))
    for name, target in sorted(etf_target.items()):
        held = holdings.get(ETF_CODES[name], 0)
        if target > held:
            buy(ETF_CODES[name], 'etf', target - held, etf_marks[name])
    return orders, skipped, etf_meta, cash_available
