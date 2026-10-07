"""Fast daily account simulator for the small-cap research layers.

Same conventions as the research ``Ledger`` used by the broad-basket accounts:
signal at the close, orders at the next open, 30 bp slippage per side,
0.025 percent commission with a 5 yuan minimum, stamp tax on sells (0.1 percent
before 2023-08-28, 0.05 percent after), 100-share lots, trades under 1000 yuan
skipped, one-word limit-up buys and limit-down sells rejected, each name capped
at 1 percent of its prior 20-session average amount per day, unfilled orders
retried until the next signal replaces them.

Differences that are known and reported, not hidden:
- dividends and share changes are folded into the position through the
  adjustment factor (total-return holding), without dividend tax;
- a held name that is delisted is liquidated after its last bar at the last
  close times (1 - ``delist_haircut``) instead of being carried at zero loss.

Non-stock sleeves (ETFs) are passed as extra price columns with no limit rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STAMP_CUT = pd.Timestamp('2023-08-28')


@dataclass
class Costs:
    slippage: float = 0.003
    commission: float = 0.00025
    min_commission: float = 5.0
    stamp_before: float = 0.001
    stamp_after: float = 0.0005
    min_trade_value: float = 1000.0
    participation: float = 0.01
    multiplier: float = 1.0
    delist_haircut: float = 0.5
    cash_buffer: float = 0.10
    gross_cap: float = 0.90
    seal: str = 'open'  # 'day' also rejects buys on a limit-up close and sells on a limit-down close


@dataclass
class Market:
    """Aligned date x code matrices (numpy) plus labels."""
    dates: pd.DatetimeIndex
    codes: pd.Index
    open: np.ndarray
    close: np.ndarray
    factor: np.ndarray
    limit_up: np.ndarray
    limit_down: np.ndarray
    adv20: np.ndarray
    last_bar: dict = field(default_factory=dict)
    no_limits: set = field(default_factory=set)

    @classmethod
    def from_panel(cls, panel, start, end, extra=None, delist=None):
        close = panel.field('close')
        dates = close.index[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
        frames = {name: panel.field(name).loc[dates] for name in ('open', 'close', 'adj_close', 'limit_up', 'limit_down', 'amount')}
        adv = panel.field('amount').rolling(20, min_periods=20).mean().shift(1).loc[dates]
        frames['adv'] = adv
        no_limits = set()
        if extra is not None:
            for name in frames:
                block = extra.get(name)
                if block is None:
                    block = pd.DataFrame(np.nan, index=extra['close'].index, columns=extra['close'].columns)
                frames[name] = frames[name].join(block.reindex(dates), how='left')
            no_limits = set(extra['close'].columns)
        codes = frames['close'].columns
        close_v = frames['close'].to_numpy(dtype=float)
        adj = frames['adj_close'].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = adj / close_v
        factor = pd.DataFrame(factor).ffill().to_numpy()
        last_bar = {}
        if delist is not None:
            index = {code: i for i, code in enumerate(codes)}
            for code, last in delist.items():
                if code in index and last is not None:
                    last_bar[index[code]] = pd.Timestamp(last)
        adv_v = frames['adv'].to_numpy(dtype=float)
        for code in no_limits:
            adv_v[:, codes.get_loc(code)] = np.inf
        return cls(dates, codes, frames['open'].to_numpy(dtype=float), close_v, factor,
                   frames['limit_up'].to_numpy(dtype=float), frames['limit_down'].to_numpy(dtype=float),
                   adv_v, last_bar, no_limits)


def _fee(notional, side, day, costs):
    commission = max(costs.min_commission, notional * costs.commission) * costs.multiplier
    stamp = 0.0
    if side < 0:
        stamp = notional * (costs.stamp_before if day < STAMP_CUT else costs.stamp_after) * costs.multiplier
    return commission + stamp


def simulate(market, schedule, initial_cash=600_000.0, costs=None, record=False):
    """Run one account.

    ``schedule`` maps a signal date to ``{code: weight}`` (weights of NAV at that
    close). Missing codes are sold. Returns NAV series, fills and events.
    """
    costs = costs or Costs()
    col = {code: i for i, code in enumerate(market.codes)}
    free = {col[c] for c in market.no_limits if c in col}
    signals = {pd.Timestamp(day).normalize(): weights for day, weights in schedule.items()}
    shares = {}           # column -> shares
    marks = {}            # column -> last valid close
    cash = initial_cash
    pending = {}          # column -> signed remaining shares
    nav_out = np.empty(len(market.dates))
    gross_out = np.empty(len(market.dates))
    turnover = 0.0
    fees_total = slip_total = 0.0
    fills = 0
    events = []
    for t, day in enumerate(market.dates):
        # corporate actions through the factor, before the open
        if t > 0:
            for j in list(shares):
                f0, f1 = market.factor[t - 1, j], market.factor[t, j]
                if np.isfinite(f0) and np.isfinite(f1) and f0 > 0 and abs(f1 / f0 - 1) > 1e-9:
                    shares[j] *= f1 / f0
                    if j in marks:
                        marks[j] *= f0 / f1
        # delisted names: liquidate after the last bar
        for j in list(shares):
            last = market.last_bar.get(j)
            if last is not None and day > last:
                value = shares[j] * marks.get(j, 0.0) * (1 - costs.delist_haircut)
                cash += value
                events.append({'date': str(day.date()), 'type': 'delisted_liquidation', 'code': market.codes[j],
                               'value': value})
                del shares[j]
                pending.pop(j, None)
        # execute pending orders at the open: sells first
        for side in (-1, 1):
            for j, remaining in list(pending.items()):
                if np.sign(remaining) != side or remaining == 0:
                    continue
                price = market.open[t, j]
                if not np.isfinite(price) or price <= 0:
                    continue
                if j not in free and j < market.limit_up.shape[1]:
                    up, down = market.limit_up[t, j], market.limit_down[t, j]
                    if side > 0 and np.isfinite(up) and price >= up - 0.005:
                        continue
                    if side < 0 and np.isfinite(down) and price <= down + 0.005:
                        continue
                    if costs.seal == 'day':
                        last = market.close[t, j]
                        if side > 0 and np.isfinite(up) and last >= up - 0.005:
                            continue
                        if side < 0 and np.isfinite(down) and last <= down + 0.005:
                            continue
                px = price * (1 + side * costs.slippage * costs.multiplier)
                quantity = abs(remaining)
                cap = market.adv20[t, j] * costs.participation
                if np.isfinite(cap):
                    quantity = min(quantity, np.floor(cap / px / 100) * 100)
                if side > 0:
                    held_value = sum(q * (market.open[t, k] if np.isfinite(market.open[t, k]) else marks.get(k, 0.0))
                                     for k, q in shares.items())
                    nav_open = cash + held_value
                    spendable = min(cash - costs.cash_buffer * nav_open,
                                    costs.gross_cap * nav_open - held_value) - costs.min_commission
                    affordable = np.floor(spendable / (px * (1 + costs.commission)) / 100) * 100
                    quantity = min(quantity, max(affordable, 0))
                else:
                    quantity = min(quantity, shares.get(j, 0.0))
                if quantity <= 0:
                    continue
                notional = quantity * px
                fee = _fee(notional, side, day, costs)
                cash -= side * notional + fee
                shares[j] = shares.get(j, 0.0) + side * quantity
                if shares[j] <= 1e-6:
                    shares.pop(j)
                fees_total += fee
                slip_total += abs(px - price) * quantity
                turnover += notional
                fills += 1
                pending[j] = remaining - side * quantity
                if abs(pending[j]) < 1:
                    pending.pop(j)
        # close marks
        for j in shares:
            value = market.close[t, j]
            if np.isfinite(value) and value > 0:
                marks[j] = value
        held_close = sum(q * marks.get(j, 0.0) for j, q in shares.items())
        nav = cash + held_close
        nav_out[t] = nav
        gross_out[t] = held_close / nav if nav > 0 else 0.0
        # new signal replaces all pending orders
        weights = signals.get(day)
        if weights is None:
            continue
        pending = {}
        targets = {col[c]: w for c, w in weights.items() if c in col and w > 0}
        for j in set(shares) | set(targets):
            price = market.close[t, j] if np.isfinite(market.close[t, j]) else marks.get(j)
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            desired = targets.get(j, 0.0) * nav / price
            delta = desired - shares.get(j, 0.0)
            if j in targets:
                delta = np.floor(delta / 100) * 100 if delta > 0 else np.ceil(delta / 100) * 100
            else:
                delta = -shares.get(j, 0.0)
            if delta != 0 and (abs(delta) * price >= costs.min_trade_value or j not in targets):
                pending[j] = delta
    nav = pd.Series(nav_out, index=market.dates)
    return {'nav': nav, 'gross': pd.Series(gross_out, index=market.dates), 'fees': fees_total, 'slippage': slip_total, 'turnover': turnover,
            'fills': fills, 'events': events}


def metrics(nav, initial_cash=600_000.0):
    nav = nav.dropna()
    elapsed = (nav.index[-1] - nav.index[0]).days
    cagr = (nav.iloc[-1] / initial_cash) ** (365.25 / elapsed) - 1
    drawdown = nav / nav.cummax().clip(lower=initial_cash) - 1
    trough = drawdown.idxmin()
    yearly = nav.groupby(nav.index.year).last()
    base = pd.concat([pd.Series([initial_cash], index=[yearly.index[0] - 1]), yearly])
    years = (base / base.shift(1) - 1).dropna()
    return {'cagr': float(cagr), 'max_drawdown': float(drawdown.min()), 'trough': str(trough.date()),
            'final': float(nav.iloc[-1]), 'yearly': {int(k): float(v) for k, v in years.items()}}


def segment_cagr(nav, start, end, initial_cash=600_000.0):
    """Annualised return of the account between two dates (base = last NAV before start)."""
    before = nav[nav.index < pd.Timestamp(start)]
    base = before.iloc[-1] if len(before) else initial_cash
    window = nav[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
    first = before.index[-1] if len(before) else window.index[0]
    elapsed = (window.index[-1] - first).days
    return float((window.iloc[-1] / base) ** (365.25 / max(elapsed, 1)) - 1)
