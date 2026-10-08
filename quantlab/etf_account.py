"""Cash account for exchange-traded funds: next-open execution, 100-share lots, fees, distributions.

Shares are bought in 100-share lots at the raw open; positions are then carried in back-adjusted
units, so cash distributions accrue through ``close_hfq``. QDII premium/discount and creation
limits are not modelled.
"""
from __future__ import annotations

import math

import pandas as pd

SLIP, COMMISSION, MIN_FEE, LOT = 0.0005, 0.00025, 5.0, 100


def frames(bars):
    pivot = lambda field: bars.pivot_table(index='date', columns='code', values=field, aggfunc='last').sort_index()
    return {name: pivot(name) for name in ('open_raw', 'open_hfq', 'close_hfq')}


def fee(value, cost_mult):
    return max(MIN_FEE, value * COMMISSION) * cost_mult + value * SLIP * cost_mult if value > 0 else 0.0


def replay(bars, targets, start, end, cash=600_000.0, cost_mult=1.0):
    """``targets``: {signal session: {code: weight}}; trades at the next session's open."""
    f = frames(bars)
    dates = f['close_hfq'].loc[pd.Timestamp(start):pd.Timestamp(end)].index
    schedule = {dates[dates.searchsorted(pd.Timestamp(day), side='right')]: weights
                for day, weights in targets.items()
                if dates.searchsorted(pd.Timestamp(day), side='right') < len(dates)}
    units, navs, trades, fees = {}, [], 0, 0.0
    for day in dates:
        if day in schedule:
            opens = f['open_hfq'].loc[day]
            value = cash + sum(u * opens[c] for c, u in units.items())
            for code in list(units):
                if schedule[day].get(code, 0.0) <= 0 and math.isfinite(opens[code]):
                    proceeds = units.pop(code) * opens[code]
                    cost = fee(proceeds, cost_mult)
                    cash += proceeds - cost
                    fees += cost
                    trades += 1
            orders = []
            for code, weight in schedule[day].items():
                raw, adj = f['open_raw'].loc[day].get(code), opens.get(code)
                if weight > 0 and raw and math.isfinite(raw) and math.isfinite(adj):
                    orders.append((weight * value - units.get(code, 0.0) * adj, code, raw, adj))
            for delta, code, raw, adj in sorted(orders):
                shares = math.floor(abs(delta) / raw / LOT) * LOT
                if shares == 0:
                    continue
                amount = shares * raw
                cost = fee(amount, cost_mult)
                if delta > 0:
                    if amount + cost > cash:
                        shares = math.floor((cash - MIN_FEE) / raw / (1 + COMMISSION + SLIP) / LOT) * LOT
                        if shares <= 0:
                            continue
                        amount, cost = shares * raw, fee(shares * raw, cost_mult)
                    units[code] = units.get(code, 0.0) + amount / adj
                    cash -= amount + cost
                else:
                    units[code] -= amount / adj
                    cash += amount - cost
                fees += cost
                trades += 1
        closes = f['close_hfq'].loc[day]
        navs.append(cash + sum(u * closes[c] for c, u in units.items() if math.isfinite(closes[c])))
    nav = pd.Series(navs, index=dates, name='nav')
    return {'nav': nav, 'trades': trades, 'fees': fees}


def stats(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    return {'cagr': float(cagr), 'max_drawdown': float((nav / nav.cummax() - 1).min()),
            'vol': float(nav.pct_change().std() * math.sqrt(252))}
