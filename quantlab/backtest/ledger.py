"""Cash/share ledger. Orders are intents; only fills change cash or holdings."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from decimal import Decimal, ROUND_HALF_UP
import math

import numpy as np
import pandas as pd

from quantlab.data.contracts import InstrumentRules, timestamp
from quantlab.data.exchange_rules import RULE_VERSIONS


def money(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass
class Lot:
    quantity: int
    available_session: int
    cost: float
    acquired_session: int
    acquired_date: str = ""
    dividends: list = field(default_factory=list)


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
    industry_policy: str = 'enforced'
    risk_policy: str = 'P0'
    risk_budget: dict = field(default_factory=dict)
    candidate: str = 'baseline'
    estimated_notional: float = 0.
    estimated_fees: float = 0.
    cash_at_submission: float = 0.
    execute_not_before: str | None = None
    execute_not_before_session: int | None = None


def price_bounds(bar,rule):
    upper,lower=bar.get('limit_up',np.nan),bar.get('limit_down',np.nan)
    if rule.limit_ratio is not None and np.isfinite(rule.limit_ratio) and not (np.isfinite(upper) and np.isfinite(lower)):
        ref=bar.get('reference_price',np.nan)
        if np.isfinite(ref) and ref>0:
            def tick(value):return float((value/Decimal(str(rule.price_tick))).quantize(Decimal('1'),rounding=ROUND_HALF_UP))*rule.price_tick
            upper=tick(Decimal(str(ref))*(1+Decimal(str(rule.limit_ratio))))
            lower=tick(Decimal(str(ref))*(1-Decimal(str(rule.limit_ratio))))
    return upper,lower


def trade_permission(bar, rule: InstrumentRules, side, date, *, allow_open_assumption=False):
    if bar is None: return False, "missing_open"
    if side=='sell' and bar.get('stress_sell_blocked',False):return False,'stress_sell_lock'
    px = bar.get("open", np.nan)
    if not np.isfinite(px) or px <= 0: return False, "missing_open"
    # Full-day amount/volume is not available at the opening execution time.
    derived=(allow_open_assumption and bar.get('open_status')=='derived' and
             bar.get('open_status_basis')=='daily_open_assumption' and bar.get('rule_version') in RULE_VERSIONS)
    if derived:
        from quantlab.data.exchange_rules import limit_ratio,limit_prices
        ratio=limit_ratio(rule.instrument_id,date,bar.get('is_st_at_open',np.nan),bar.get('listing_session_lower_bound'))
        ref=bar.get('reference_price',np.nan)
        if ratio is None or not np.isfinite(ref) or ref<=0:return False,'unverified_derived_limit_rule'
        upper,lower=limit_prices(ref,ratio,rule.price_tick)
        if any(not np.isfinite(bar.get(k,np.nan)) or abs(bar[k]-v)>rule.price_tick/10 for k,v in [('limit_up',upper),('limit_down',lower)]):
            return False,'derived_limit_mismatch'
        if px < lower-rule.price_tick/2 or px > upper+rule.price_tick/2:
            return False,'derived_open_outside_rule_bounds'
    if bar.get("open_status") != "trading" and not derived: return False, "unknown_or_halted_open"
    when = bar.get("open_status_available_at")
    if when is None or pd.isna(when) or timestamp(when) > timestamp(date) + pd.Timedelta(hours=9, minutes=30):
        return False, "opening_status_not_available"
    upper, lower = price_bounds(bar,rule)
    if rule.limit_ratio is not None and np.isfinite(rule.limit_ratio):
        if not (np.isfinite(upper) and np.isfinite(lower)):return False,'missing_limit_reference'
        if side == "buy" and px >= upper - rule.price_tick / 2: return False, "limit_up"
        if side == "sell" and px <= lower + rule.price_tick / 2: return False, "limit_down"
    return True, ""


class Ledger:
    def __init__(self, cash=600_000.0, slippage=0.003, cash_buffer=0.02, participation=0.01, cost_mult=1.0, min_trade_value=1000., allow_open_assumption=False):
        self.allow_open_assumption=allow_open_assumption
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
        self.entitlement_lots = {}
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
                 "weights": self.weights(), "stale": stale,
                 "marks": {c:self.marks[c] for c in self.quantities()},
                 "cost_basis": {c:self.cost_basis(c) for c in self.quantities()},
                 "entry_dates": {c:min(l.acquired_date for l in self.lots[c] if l.quantity and l.acquired_date)
                                 for c in self.quantities() if any(l.quantity and l.acquired_date for l in self.lots[c])}}
        self.snapshots.append(state)
        self.entitlements[str(pd.Timestamp(date).date())] = self.quantities().copy()
        self.entitlement_lots[str(pd.Timestamp(date).date())]={c:[(lot,lot.quantity) for lot in self.lots[c] if lot.quantity] for c in self.quantities()}
        return state

    def dividend_tax(self,c,quantity,session,date):
        from quantlab.backtest.tax import dividend_tax_rate
        total=0.;left=quantity
        for lot in self.lots.get(c,[]):
            if lot.available_session>session:continue
            take=min(left,lot.quantity);left-=take
            for claim in lot.dividends:
                if not lot.acquired_date:raise ValueError('dividend_tax_acquisition_date_unknown')
                total+=take*claim['per_share']*dividend_tax_rate(lot.acquired_date,date,claim['ex_date'])
            if not left:break
        return money(total)

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
                if a.get('action_type','distribution')!='distribution':
                    self.data_issues.append(dict(date=str(d),reason='unsupported_corporate_action',action=a.action_id));continue
                policy=a.get('tax_policy','net_fixed')
                if policy not in ('net_fixed','gross_cn_individual','exempt_etf'):
                    self.data_issues.append(dict(date=str(d),reason='unknown_dividend_tax_policy'));continue
                q = self.entitlements.get(str(pd.Timestamp(a.record_date).date()), {}).get(c, 0)
                cash = money(q * float(a.cash_per_share))
                ratio = float(a.share_ratio)
                extra = q * ratio
                entitlement_lots=self.entitlement_lots.get(str(pd.Timestamp(a.record_date).date()),{}).get(c,[])
                if abs(extra-round(extra))>1e-8 or any(abs(entitled*ratio-round(entitled*ratio))>1e-8 for lot,entitled in entitlement_lots):
                    self.data_issues.append(dict(date=str(d),reason='fractional_action_requires_explicit_settlement',action=a.action_id))
                    continue
                if cash:
                    self.receivables.append({"amount": cash, "pay_date": str(a.pay_date), "action_id": a.action_id})
                if policy=='gross_cn_individual' and cash:
                    for lot,entitled in self.entitlement_lots.get(str(pd.Timestamp(a.record_date).date()),{}).get(c,[]):
                        if not lot.acquired_date:
                            self.data_issues.append(dict(date=str(d),reason='dividend_tax_acquisition_date_unknown'))
                            continue
                        if lot.quantity!=entitled:self.data_issues.append(dict(date=str(d),reason='changed_entitlement_lot_requires_tax_reconciliation'))
                        lot.dividends.append(dict(per_share=float(a.cash_per_share),ex_date=str(a.ex_date),action_id=a.action_id))
                if extra:
                    ix = calendar.searchsorted(pd.Timestamp(a.share_date))
                    # Bonus lots inherit acquisition dates and proportionally split tax claims.
                    for lot,entitled in self.entitlement_lots.get(str(pd.Timestamp(a.record_date).date()),{}).get(c,[]):
                        shares=entitled*ratio
                        if abs(shares-round(shares))>1e-8:raise ValueError('fractional_lot_action_requires_settlement')
                        lot.cost/=1+ratio
                        for claim in lot.dividends:claim['per_share']/=1+ratio
                        self.lots[c].append(Lot(int(round(shares)),int(ix),lot.cost,lot.acquired_session,
                                                lot.acquired_date,[dict(x) for x in lot.dividends]))
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

    def queue_targets(self, weights, signal_date, reason, rules, replace=True, target_quantities=None):
        if any(not np.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) > 1 + 1e-9:
            raise ValueError("invalid target weights")
        if target_quantities is not None:
            if reason != 'risk_exit' or any(isinstance(q, bool) or not isinstance(q, (int, np.integer)) or
                                           q < 0 or q > self.quantity(c) for c,q in target_quantities.items()):
                raise ValueError('explicit_risk_targets_must_only_reduce_actual_shares')
        # A model rebalance cannot revoke an unfilled risk exit. Reserve its
        # remaining shares so retries neither buy them back nor duplicate sells.
        reserved = {}
        for o in self.orders:
            if o.side == 'sell' and o.reason == 'risk_exit' and o.status in ('pending', 'partial'):
                reserved[o.instrument_id] = reserved.get(o.instrument_id, 0) + o.remaining
        if replace:
            for o in self.orders:
                if o.status in ("pending", "partial"):
                    if o.side == 'sell' and o.reason == 'risk_exit': continue
                    o.status, o.reject_reason = "cancelled", "superseded"
        nav = self.nav()
        for c in sorted(set(weights) | set(self.quantities())):
            rule = rules.get(c)
            if rule is None or c not in self.marks: continue
            # Share quantity locked at decision price. No implicit daily rebalancing on retries.
            projected = self.quantity(c) - min(self.quantity(c), reserved.get(c, 0))
            raw_quantity = nav * weights.get(c, 0.0) / self.marks[c]
            if target_quantities is not None:
                target = min(projected, target_quantities.get(c, projected))
            elif math.isclose(raw_quantity, projected, rel_tol=1e-12, abs_tol=1e-8):
                # Unchanged holdings (including bonus odd lots) must not become
                # sells when their NAV/weight round trip loses a floating bit.
                target = projected
            else:
                units = raw_quantity / rule.buy_lot
                nearest = round(units)
                if math.isclose(units, nearest, rel_tol=0., abs_tol=1e-10):
                    units = nearest
                target = math.floor(units) * int(rule.buy_lot)
            if reserved.get(c, 0): target = min(target, projected)
            delta = target - projected
            if delta == 0: continue
            side = "buy" if delta > 0 else "sell"
            qty = abs(int(delta))
            legal = rule.order_quantity(qty, side, holding=projected if side == 'sell' else None)
            if qty and not legal:
                self.events.append(dict(date=str(pd.Timestamp(signal_date).date()),type='order_not_submitted',
                    instrument_id=c,side=side,quantity=qty,reason='below_minimum_order_quantity',target_quantity=target))
            qty = legal
            if qty and (side == "sell" and (target == 0 or reason == "risk_exit") or qty * self.marks[c] >= self.min_trade_value):
                self.orders.append(Order(f"O{len(self.orders)+1:08d}", c, side, qty, qty,
                                         str(pd.Timestamp(signal_date).date()), reason))
            elif qty:
                self.events.append(dict(date=str(pd.Timestamp(signal_date).date()),type='order_not_submitted',
                    instrument_id=c,side=side,quantity=qty,reason='below_min_trade_value',target_quantity=target))

    def fee(self, date, rule, side, notional):
        commission = max(rule.min_commission, notional * rule.commission) * self.cost_mult
        stamp = 0.0
        if rule.asset_type == "stock" and side == "sell":
            stamp = notional * (0.001 if pd.Timestamp(date) < pd.Timestamp("2023-08-28") else 0.0005) * self.cost_mult
        return money(commission), money(stamp)

    def _record_execution_shortfall(self, order, date, session, reason, *, filled_quantity=0,
                                    causes=(), context=None):
        # Append a value snapshot: a later fill may clear the order's current
        # reject_reason, but must not erase a previous execution attempt.
        details = {}
        for key, value in (context or {}).items():
            if isinstance(value, np.generic): value = value.item()
            details[key] = None if isinstance(value, float) and not math.isfinite(value) else value
        self.events.append(dict(type='execution_rejected', date=str(pd.Timestamp(date).date()),
            session=int(session), order_id=order.order_id, instrument_id=order.instrument_id,
            side=order.side, remaining=order.remaining, reason=reason,
            outcome='partial' if filled_quantity else 'unfilled', filled_quantity=filled_quantity,
            attempted_quantity=order.remaining+filled_quantity,
            causes=list(dict.fromkeys(causes or (reason,))), execution_context=details))

    def execute(self, date, session, bars, rules, adv, can_buy=None):
        self.mark(date, {c: b.get("open", np.nan) for c, b in bars.items()}, "open")
        for o in sorted(self.orders, key=lambda x: (x.side != "sell", x.order_id)):
            if o.status not in ("pending", "partial") or pd.Timestamp(o.signal_date) >= pd.Timestamp(date): continue
            if o.execute_not_before_session is not None and session<o.execute_not_before_session: continue
            c, side = o.instrument_id, o.side
            source_bar = bars.get(c, {})
            context = {key: source_bar.get(key) for key in ('open', 'reference_price', 'limit_up', 'limit_down')}
            context['raw_open'] = context.pop('open')
            rule = rules.get(c)
            if rule is None:
                o.reject_reason = "unknown_rules"
                self._record_execution_shortfall(o,date,session,o.reject_reason,context=context)
                continue
            allowed, why = trade_permission(bars.get(c), rule, side, date,allow_open_assumption=self.allow_open_assumption)
            if not allowed:
                o.reject_reason = why
                self._record_execution_shortfall(o,date,session,why,context=context)
                if side == "buy": o.status = "expired"
                continue
            a = adv.get(c, np.nan)
            context['prior_average_daily_amount'] = a
            if not np.isfinite(a) or a <= 0:
                o.reject_reason = "missing_prior_liquidity"
                self._record_execution_shortfall(o,date,session,o.reject_reason,context=context)
                if side == "buy": o.status = "expired"
                continue
            px0 = float(bars[c]["open"])
            px = px0 * (1 + self.slippage * self.cost_mult * (1 if side == "buy" else -1))
            px = float((Decimal(str(px))/Decimal(str(rule.price_tick))).quantize(Decimal("1"),rounding=ROUND_HALF_UP))*rule.price_tick
            upper, lower = price_bounds(bars[c],rule)
            context.update(slipped_price=px,limit_up=upper,limit_down=lower)
            if (side=='buy' and np.isfinite(upper) and px>=upper-rule.price_tick/2) or (side=='sell' and np.isfinite(lower) and px<=lower+rule.price_tick/2):
                o.reject_reason='slippage_reaches_price_limit'
                self._record_execution_shortfall(o,date,session,o.reject_reason,context=context)
                if side=='buy':o.status='expired'
                continue
            if np.isfinite(upper): px = min(px, upper)
            if np.isfinite(lower): px = max(px, lower)
            if px <= 0: raise ValueError("invalid slipped price")
            lot = int(rule.buy_lot if side == "buy" else rule.sell_lot)
            used=sum(f["notional"] for f in self.fills if f["instrument_id"]==c and f["date"]==str(pd.Timestamp(date).date()))
            liquidity_quantity = int(max(0.,a * self.participation-used) / px)
            causes = ['liquidity_limit'] if liquidity_quantity < o.remaining else []
            context.update(participation_rate=self.participation,used_notional=used,
                           liquidity_quantity=liquidity_quantity,cash_at_attempt=self.cash)
            qty = min(o.remaining, liquidity_quantity)
            if side == "sell":
                available_quantity = self.available(c, session)
                context['available_quantity'] = available_quantity
                if available_quantity < qty: causes.append('settlement_limit')
                qty = min(qty, available_quantity)
            # Odd lots may only be sold as the entire remaining position.
            if side == "buy" or qty < self.quantity(c):
                if qty % lot: causes.append('lot_rounding')
                qty = qty // lot * lot
            reserve = self.nav() * self.cash_buffer
            context['cash_reserve'] = reserve
            if side == "buy":
                cash_quantity = max(0, int((self.cash - reserve) / px)) // lot * lot
                context['affordable_quantity_before_fees'] = cash_quantity
                if cash_quantity < qty: causes.append('cash_limit')
                qty = min(qty, cash_quantity)
                while qty > 0:
                    commission, stamp = self.fee(date, rule, side, qty * px)
                    valid_cash = qty * px + commission + stamp <= self.cash - reserve + 0.001
                    if valid_cash:
                        if can_buy is None or can_buy(c, qty, px, commission + stamp, self): break
                        # The callback can reject eligibility, risk, or exposure;
                        # a boolean result cannot identify which one failed.
                        causes.append('buy_permission_rejected')
                    else: causes.append('cash_limit')
                    qty -= lot
            if qty <= 0:
                o.reject_reason = "cash_or_exposure_limit" if side == "buy" else "settlement_or_liquidity_limit"
                self._record_execution_shortfall(o,date,session,o.reject_reason,causes=causes,context=context)
                if side == "buy": o.status = "expired"
                continue
            notional = money(qty * px)
            commission, stamp = self.fee(date, rule, side, notional)
            dividend_tax = self.dividend_tax(c,qty,session,date) if side=="sell" else 0.
            fees = money(commission + stamp + dividend_tax)
            if side=="sell" and self.cash+notional-fees<0:
                o.reject_reason="insufficient_cash_for_tax"
                context.update(notional=notional,fees=fees,dividend_tax=dividend_tax)
                self._record_execution_shortfall(o,date,session,o.reject_reason,context=context)
                continue
            if side == "buy":
                self.cash = money(self.cash - notional - fees)
                self.lots.setdefault(c, []).append(Lot(qty, session + int(rule.sell_delay), (notional + fees) / qty, session, str(pd.Timestamp(date).date())))
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
                               "stamp_tax": stamp, "dividend_tax": dividend_tax, "fees": fees, "slippage_cost": money(abs(px-px0)*qty),
                               "signal_date": o.signal_date, "reason": o.reason, "cash_after": self.cash,
                               "risk_policy":o.risk_policy,"risk_budget":o.risk_budget,"candidate":o.candidate,
                               "execution_basis": bars[c].get('open_status_basis','observed_status')})
            if side == "buy" and o.remaining: o.status, o.reject_reason = "expired", "partial_day_order"
            if o.remaining:
                reason = 'partial_day_order' if side == 'buy' else 'settlement_or_liquidity_limit'
                self._record_execution_shortfall(o,date,session,reason,filled_quantity=qty,causes=causes,context=context)
        if self.cash < -0.005: raise AssertionError("negative cash")

    def export(self):
        return {"orders": [asdict(o) for o in self.orders], "fills": self.fills, "account": self.snapshots,
                "corporate_action_events": self.events, "data_issues": self.data_issues}
