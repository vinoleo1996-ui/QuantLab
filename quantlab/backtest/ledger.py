"""Cash/share ledger. Orders are intents; only fills change cash or holdings."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
import math

import numpy as np
import pandas as pd

from quantlab.data.contracts import InstrumentRules, timestamp


def money(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass
class Lot:
    quantity: int
    available_session: int
    cost: float
    acquired_session: int


@dataclass
class Order:
    order_id: str
    instrument_id: str
    side: str
    quantity: int
    remaining: int
    signal_date: str
    reason: str
    status: str = "pending"
    reject_reason: str = ""


def trade_permission(bar, rule: InstrumentRules, side, date):
    if bar is None: return False, "missing_open"
    px = bar.get("open", np.nan)
    if not np.isfinite(px) or px <= 0: return False, "missing_open"
    # Full-day amount/volume is not available at the opening execution time.
    if bar.get("open_status") != "trading": return False, "unknown_or_halted_open"
    when = bar.get("open_status_available_at")
    if when is None or pd.isna(when) or timestamp(when) > timestamp(date) + pd.Timedelta(hours=9, minutes=30):
        return False, "opening_status_not_available"
    upper, lower = bar.get("limit_up", np.nan), bar.get("limit_down", np.nan)
    if rule.limit_ratio is not None and np.isfinite(rule.limit_ratio):
        ref = bar.get("reference_price", np.nan)
        if not (np.isfinite(upper) and np.isfinite(lower)):
            if not np.isfinite(ref) or ref <= 0: return False, "missing_limit_reference"
            upper = float((Decimal(str(ref)) * (1 + Decimal(str(rule.limit_ratio))) /
                           Decimal(str(rule.price_tick))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) * rule.price_tick
            lower = float((Decimal(str(ref)) * (1 - Decimal(str(rule.limit_ratio))) /
                           Decimal(str(rule.price_tick))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) * rule.price_tick
        if side == "buy" and px >= upper - rule.price_tick / 2: return False, "limit_up"
        if side == "sell" and px <= lower + rule.price_tick / 2: return False, "limit_down"
    return True, ""


class Ledger:
    def __init__(self, cash=600_000.0, slippage=0.003, cash_buffer=0.02, participation=0.01, cost_mult=1.0, min_trade_value=1000.):
        self.min_trade_value = min_trade_value
        self.initial_cash = money(cash)
        self.cash = money(cash)
        self.slippage, self.cash_buffer = slippage, cash_buffer
        self.participation, self.cost_mult = participation, cost_mult
        self.lots: dict[str, list[Lot]] = {}
        self.marks: dict[str, float] = {}
        self.orders: list[Order] = []
        self.fills: list[dict] = []
        self.events: list[dict] = []
        self.receivables: list[dict] = []
        self.entitlements: dict[str, dict[str, int]] = {}
        self.applied_actions: set[str] = set()
        self.snapshots: list[dict] = []
        self.data_issues: list[dict] = []

    def quantity(self, c): return sum(l.quantity for l in self.lots.get(c, []))
    def quantities(self): return {c: self.quantity(c) for c in self.lots if self.quantity(c) > 0}
    def available(self, c, session):
        return sum(l.quantity for l in self.lots.get(c, []) if l.available_session <= session)
    def nav(self):
        return self.cash + sum(self.quantity(c) * self.marks[c] for c in self.quantities()) + sum(x["amount"] for x in self.receivables)
    def weights(self):
        n = self.nav()
        return {c: self.quantity(c) * self.marks[c] / n for c in self.quantities()}
    def cost_basis(self, c):
        q = self.quantity(c)
        return sum(l.cost * l.quantity for l in self.lots.get(c, [])) / q if q else 0.0
    def holding_age(self, c, session):
        return session - min((l.acquired_session for l in self.lots.get(c, []) if l.quantity), default=session)

    def mark(self, date, prices, field="close"):
        stale = []
        for c in set(prices) | set(self.quantities()):
            px = prices.get(c, np.nan)
            if np.isfinite(px) and px > 0: self.marks[c] = float(px)
            elif self.quantity(c): stale.append(c)
        if stale:
            self.data_issues.append({"date": str(date), "reason": f"stale_{field}", "instruments": stale})
        return stale

    def close(self, date, session, prices):
        stale = self.mark(date, prices)
        if self.cash < -0.005 or any(l.quantity < 0 for lots in self.lots.values() for l in lots):
            raise AssertionError("cash/share conservation violated")
        state = {"date": pd.Timestamp(date), "session": session, "nav": self.nav(), "cash": self.cash,
                 "receivables": sum(x["amount"] for x in self.receivables), "quantities": self.quantities(),
                 "weights": self.weights(), "stale": stale}
        self.snapshots.append(state)
        self.entitlements[str(pd.Timestamp(date).date())] = self.quantities().copy()
        return state

    def corporate_actions(self, date, session, actions, calendar):
        d = pd.Timestamp(date).normalize()
        if actions is not None and not actions.empty:
            due = actions[pd.to_datetime(actions.ex_date).dt.normalize() == d]
            known = due[due.available_at.map(timestamp) <= timestamp(d) + pd.Timedelta(hours=9)]
            unknown = set(due.action_id) - set(known.action_id)
            if unknown: self.data_issues.append({"date": str(d), "reason": "late_corporate_action", "actions": sorted(unknown)})
            known = known.sort_values("available_at").drop_duplicates("action_id", keep="last")
            for _, a in known.iterrows():
                if pd.Timestamp(a.ex_date).normalize() != d or a.action_id in self.applied_actions: continue
                if timestamp(a.available_at) > timestamp(d) + pd.Timedelta(hours=9):
                    self.data_issues.append({"date": str(d), "reason": "late_corporate_action", "instrument": a.instrument_id})
                    continue
                c = a.instrument_id
                q = self.entitlements.get(str(pd.Timestamp(a.record_date).date()), {}).get(c, 0)
                cash = money(q * float(a.cash_per_share))
                ratio = float(a.share_ratio)
                extra = q * ratio
                if abs(extra - round(extra)) > 1e-8:
                    raise ValueError("fractional action shares require explicit cash-in-lieu action")
                if cash:
                    self.receivables.append({"amount": cash, "pay_date": str(a.pay_date), "action_id": a.action_id})
                if extra:
                    ix = calendar.searchsorted(pd.Timestamp(a.share_date))
                    basis = self.cost_basis(c)
                    # Preserve total cost across bonus shares.
                    for lot in self.lots.get(c, []): lot.cost /= 1 + ratio
                    self.lots.setdefault(c, []).append(Lot(int(round(extra)), int(ix), basis / (1 + ratio), session))
                if c in self.marks:
                    self.marks[c] = (self.marks[c] - float(a.cash_per_share)) / (1 + ratio)
                self.applied_actions.add(a.action_id)
                self.events.append({"date": str(d), "type": "corporate_action", "action_id": a.action_id,
                                    "entitled_shares": q, "cash": cash, "bonus_shares": int(round(extra))})
        for x in list(self.receivables):
            if pd.Timestamp(x["pay_date"]).normalize() <= d:
                self.cash = money(self.cash + x["amount"])
                self.receivables.remove(x)

    def cancel_buys(self, reason):
        for o in self.orders:
            if o.side == "buy" and o.status in ("pending", "partial"):
                o.status, o.reject_reason = "cancelled", reason

    def queue_targets(self, weights, signal_date, reason, rules, replace=True):
        if replace:
            for o in self.orders:
                if o.status in ("pending", "partial"):
                    o.status, o.reject_reason = "cancelled", "superseded"
        if any(not np.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) > 1 + 1e-9:
            raise ValueError("invalid target weights")
        nav = self.nav()
        for c in sorted(set(weights) | set(self.quantities())):
            rule = rules.get(c)
            if rule is None or c not in self.marks: continue
            # Share quantity locked at decision price. No implicit daily rebalancing on retries.
            target = math.floor(nav * weights.get(c, 0.0) / self.marks[c] / rule.buy_lot) * int(rule.buy_lot)
            delta = target - self.quantity(c)
            if delta == 0: continue
            side = "buy" if delta > 0 else "sell"
            qty = abs(int(delta))
            if side == "sell" and target > 0: qty = qty // int(rule.sell_lot) * int(rule.sell_lot)
            if qty and (side == "sell" and (target == 0 or reason == "risk_exit") or qty * self.marks[c] >= self.min_trade_value):
                self.orders.append(Order(f"O{len(self.orders)+1:08d}", c, side, qty, qty,
                                         str(pd.Timestamp(signal_date).date()), reason))

    def fee(self, date, rule, side, notional):
        commission = max(rule.min_commission, notional * rule.commission) * self.cost_mult
        stamp = 0.0
        if rule.asset_type == "stock" and side == "sell":
            stamp = notional * (0.001 if pd.Timestamp(date) < pd.Timestamp("2023-08-28") else 0.0005) * self.cost_mult
        return money(commission), money(stamp)

    def execute(self, date, session, bars, rules, adv, can_buy=None):
        self.mark(date, {c: b.get("open", np.nan) for c, b in bars.items()}, "open")
        for o in sorted(self.orders, key=lambda x: (x.side != "sell", x.order_id)):
            if o.status not in ("pending", "partial") or pd.Timestamp(o.signal_date) >= pd.Timestamp(date): continue
            c, side = o.instrument_id, o.side
            rule = rules.get(c)
            if rule is None:
                o.reject_reason = "unknown_rules"
                continue
            allowed, why = trade_permission(bars.get(c), rule, side, date)
            if not allowed:
                o.reject_reason = why
                if side == "buy": o.status = "expired"
                continue
            a = adv.get(c, np.nan)
            if not np.isfinite(a) or a <= 0:
                o.reject_reason = "missing_prior_liquidity"
                if side == "buy": o.status = "expired"
                continue
            px0 = float(bars[c]["open"])
            px = px0 * (1 + self.slippage * self.cost_mult * (1 if side == "buy" else -1))
            px = round(px / rule.price_tick) * rule.price_tick
            upper, lower = bars[c].get("limit_up", np.inf), bars[c].get("limit_down", -np.inf)
            if np.isfinite(upper): px = min(px, upper)
            if np.isfinite(lower): px = max(px, lower)
            if px <= 0: raise ValueError("invalid slipped price")
            lot = int(rule.buy_lot if side == "buy" else rule.sell_lot)
            qty = min(o.remaining, int(a * self.participation / px))
            if side == "sell": qty = min(qty, self.available(c, session))
            # Odd lots may only be sold as the entire remaining position.
            if side == "buy" or qty < self.quantity(c): qty = qty // lot * lot
            reserve = self.nav() * self.cash_buffer
            if side == "buy":
                qty = min(qty, max(0, int((self.cash - reserve) / px)) // lot * lot)
                while qty > 0:
                    commission, stamp = self.fee(date, rule, side, qty * px)
                    valid_cash = qty * px + commission + stamp <= self.cash - reserve + 0.001
                    if valid_cash and (can_buy is None or can_buy(c, qty, px, commission + stamp, self)): break
                    qty -= lot
            if qty <= 0:
                o.reject_reason = "cash_or_exposure_limit" if side == "buy" else "settlement_or_liquidity_limit"
                if side == "buy": o.status = "expired"
                continue
            notional = money(qty * px)
            commission, stamp = self.fee(date, rule, side, notional)
            fees = money(commission + stamp)
            if side == "buy":
                self.cash = money(self.cash - notional - fees)
                self.lots.setdefault(c, []).append(Lot(qty, session + int(rule.sell_delay), (notional + fees) / qty, session))
            else:
                self.cash = money(self.cash + notional - fees)
                left = qty
                for l in self.lots.get(c, []):
                    if l.available_session > session: continue
                    take = min(left, l.quantity)
                    l.quantity -= take
                    left -= take
                    if not left: break
                if left: raise AssertionError("sold unavailable shares")
            o.remaining -= qty
            o.status, o.reject_reason = ("filled" if o.remaining == 0 else "partial"), ""
            self.fills.append({"order_id": o.order_id, "instrument_id": c, "side": side, "date": str(pd.Timestamp(date).date()),
                               "quantity": qty, "price": px, "notional": notional, "commission": commission,
                               "stamp_tax": stamp, "fees": fees, "slippage_cost": money(abs(px-px0)*qty),
                               "signal_date": o.signal_date, "reason": o.reason, "cash_after": self.cash})
            if side == "buy" and o.remaining: o.status, o.reject_reason = "expired", "partial_day_order"
        if self.cash < -0.005: raise AssertionError("negative cash")

    def export(self):
        return {"orders": [asdict(o) for o in self.orders], "fills": self.fills, "account": self.snapshots,
                "corporate_action_events": self.events, "data_issues": self.data_issues}
