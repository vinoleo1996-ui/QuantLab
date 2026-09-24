"""Daily cash/share replay shared by strategies and baselines.

Legacy price panels without execution/PIT contracts are rejected. Diagnostic proxy
runs require an explicit flag and are never eligible for promotion or live signals.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.data.contracts import (MarketDataset, InstrumentRules, asof_records, decision_time,
                                    fingerprint, timestamp)
from quantlab.data.market import panels_for_research, universe_at, exposure_matrix
from quantlab.backtest.ledger import Ledger
from quantlab.strategy.decision import FactorDecisionService, review_dates
from quantlab.strategy.allocation import allocate
from quantlab.strategy.risk import RiskManager, Regime


def _limit_ratio(code):
    if str(code).startswith(("300", "301", "688", "689")): return .20
    if str(code).startswith(("43", "83", "87", "92")): return .30
    return .10


def limit_masks(open_, prev_close, amount=None, volume=None):
    """Legacy diagnostic helper only; opening replay uses dated market status/rules."""
    gap = open_ / prev_close - 1
    known = open_.gt(0) & prev_close.gt(0)
    if amount is not None:
        trading = amount.gt(0)
        if volume is not None: trading |= amount.isna() & volume.gt(0)
        known &= trading
    buys, sells = known.copy(), known.copy()
    for c in gap:
        buys[c] &= gap[c] < _limit_ratio(c) - .005
        sells[c] &= gap[c] > -_limit_ratio(c) + .005
    return buys, sells


def blocked_result(reasons, market="a"):
    return {"nav": pd.Series(dtype=float), "daily_return": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float), "trades": [], "weight_history": [],
            "ic_matrix": pd.DataFrame(), "factor_panels": {}, "codes": [], "market": market,
            "status": "blocked_data", "reasons": reasons, "metrics": None,
            "promotion_approved": False, "model_version": "ledger-v2", "data_version": None,
            "as_of": None, "run_id": fingerprint(reasons)[:20], "orders": [], "fills": []}


def prepare(dataset, cfg):
    close, amount, turnover = panels_for_research(dataset)
    service = FactorDecisionService(close, amount, turnover, dataset.calendar, cfg)
    # Capture raw and processed inputs in every persisted run, including the exact benchmark.
    records, rules, exposures, eligible = {}, {}, {}, {}
    for d in close.index:
        morning = asof_records(dataset.instruments, timestamp(d) + pd.Timedelta(hours=9, minutes=25))
        records[d] = morning
        rules[d] = {}
        for c, row in morning.iterrows():
            try: rules[d][c] = InstrumentRules.from_record(row)
            except ValueError: pass
        exposures[d] = exposure_matrix(dataset, d, morning, list(morning.index))
        eligible[d], _ = universe_at(dataset, d, amount, cfg)
    return {"close": close, "amount": amount, "turnover": turnover,
            "service": service, "version": dataset.version, "preparation_config": fingerprint(asdict(cfg)), "records": records, "rules": rules,
            "exposures": exposures, "eligible": eligible,
            "bars": {d: {c: row.to_dict() for c, panel in dataset.panels.items()
                         if d in panel.index for _, row in panel.loc[[d]].iterrows()} for d in close.index}}



class Engine:
    def __init__(self, cfg=None, market="a"):
        self.cfg, self.market = cfg or CONFIG, market

    def run(self, panels, meta=None, bench=None, **kwargs):
        if isinstance(panels, MarketDataset): return self.run_dataset(panels, **kwargs)
        return blocked_result(["versioned_dataset_manifest_required", "raw_prices_and_corporate_actions_unverified",
                               "historical_universe_and_exchange_calendar_unverified"], self.market)

    def run_dataset(self, dataset, *, strategy="repaired_legacy", random_order=None, basket_size=27,
                    diagnostic=False, risk_mode="daily", prepared=None, target_schedule=None,
                    start=None, end=None, cost_mult=1.0, liquidity_mult=1.0, blocked_sessions=(),
                    execution_delay=0):
        if self.market != "a": return blocked_result(["us_execution_not_validated"], self.market)
        if risk_mode not in ("daily", "legacy_monthly"): raise ValueError("invalid_risk_policy")
        errors = dataset.validate()
        if errors and not diagnostic: return blocked_result(errors)
        cfg = self.cfg
        from quantlab.config_validation import validate_config
        validate_config(cfg)
        if cfg.research.enable_announcements:
            return blocked_result(["announcement_module_requires_PIT_and_incremental_evidence"])
        if strategy not in ("repaired_legacy", "random", "buy_hold", "monthly_equal", "vol_matched_equal", "schedule"):
            raise ValueError("unknown_or_unpromoted_strategy")
        if strategy == "schedule" and target_schedule is None: raise ValueError("target schedule required")
        p = prepared or prepare(dataset, cfg)
        if p["version"] != dataset.version or p.get("preparation_config") != fingerprint(asdict(cfg)):
            raise ValueError("prepared_context_version_mismatch")
        close, amount, svc = p["close"], p["amount"], p["service"]
        start = pd.Timestamp(start or cfg.start_date)
        end = pd.Timestamp(end or close.index[-1])
        dates = close.index[(close.index >= start) & (close.index <= end)]
        if dates.empty: return blocked_result(["empty_replay_window"])
        rule = "monthly" if strategy in ("buy_hold", "monthly_equal", "vol_matched_equal") else cfg.capital.rebalance_rule
        reviews = set(review_dates(dataset.calendar, rule, cfg.capital.rebalance_n_weeks))
        if "review_override" in p: reviews &= set(p["review_override"])
        if strategy in ("buy_hold", "monthly_equal", "vol_matched_equal") and len(dates): reviews.add(dates[0])
        if strategy == "repaired_legacy" and "review_override" not in p:
            first = next((d for d in dates if d in reviews and svc.score(d,p["eligible"][d]).status == "ok"), None)
            if first is None: return blocked_result(["insufficient_factor_history_or_eligible_universe"])
            dates = dates[dates>=first]
        account = Ledger(cfg.capital.init_cash, cfg.cost.a_slippage, cfg.research.cash_buffer,
                         cfg.research.participation, cost_mult, cfg.cost.min_trade_value)
        risk = RiskManager(cfg.risk, cfg.research) if risk_mode == "daily" else LegacyMonthlyRisk(cfg)
        regime = Regime(dataset.benchmark.close) if not dataset.benchmark.empty else None
        states, decisions, traces, allocation_issues = [], [], [], []
        last_risk = dict(gross_scale=1., vol_scale=1., level=0, allow_new_stock=True, reasons=[], drawdown=0.)
        invested = False
        delayed = []
        outside_count = {}
        config_dump = asdict(cfg)
        run_id = fingerprint({"data": p["version"], "config": config_dump, "strategy": strategy,
                              "order": random_order, "start": str(start), "end": str(end), "risk": risk_mode,
                              "cost": cost_mult, "liquidity": liquidity_mult, "blocked": list(blocked_sessions),
                              "delay": execution_delay, "schedule": target_schedule})[:24]
        for d in dates:
            session = int(dataset.calendar.get_loc(d))
            bars = p["bars"].get(d, {})
            if d in blocked_sessions:
                bars = {c: dict(b, open_status="halted") for c, b in bars.items()}
            morning = p["records"][d]
            rules = p["rules"][d].copy()
            account.corporate_actions(d, session, dataset.actions, dataset.calendar)
            adv = amount.loc[amount.index < d].tail(20).mean().to_dict()
            adv = {c: v * liquidity_mult for c, v in adv.items()}
            exp = p["exposures"][d]
            def can_buy(c, q, price, fees, ledger):
                if c not in morning.index or not bool(morning.loc[c].get("active", False)) or (cfg.universe.a_exclude_st and bool(morning.loc[c].get("is_st", True))): return False
                nav = ledger.nav() - fees
                vals = {k: ledger.quantity(k) * ledger.marks[k] for k in ledger.quantities()}
                vals[c] = vals.get(c, 0.) + q * price
                cap = cfg.research.max_etf_weight if rules[c].asset_type == "equity_etf" else cfg.research.max_stock_weight
                if vals[c] / nav > cap + 1e-8: return False
                if sum(vals.values()) / nav > min(last_risk["gross_scale"], 1-cfg.research.cash_buffer) + 1e-8: return False
                count_limit = len(dataset.panels) if strategy in ("buy_hold", "monthly_equal", "vol_matched_equal") else cfg.capital.a_share_positions
                if len([k for k in vals if vals[k]>0 and k in rules and rules[k].asset_type == "stock"]) > count_limit: return False
                if strategy not in ("buy_hold", "monthly_equal", "vol_matched_equal") and len(vals) > cfg.capital.target_positions: return False
                if len([k for k in vals if vals[k]>0 and k in rules and rules[k].asset_type == "equity_etf"]) > cfg.research.max_etfs: return False
                weights = pd.Series(vals).reindex(exp.columns, fill_value=0) / nav
                return bool((exp @ weights <= cfg.portfolio.max_sector_weight + 1e-8).all())
            # Delayed intents preserve original signal dates, and cannot bypass a later halt.
            for intent in list(delayed):
                if intent[0] <= session:
                    if last_risk["gross_scale"] > 0:
                        account.queue_targets(intent[1], intent[2], intent[3], rules)
                    delayed.remove(intent)
            account.execute(d, session, bars, rules, adv, can_buy=can_buy)
            prices = {c: b.get("close", np.nan) for c, b in bars.items()
                      if pd.notna(b.get("available_at")) and timestamp(b["available_at"]) <= decision_time(d)
                      and b.get("quality_status") == "ok"}
            state = account.close(d, session, prices)
            old_level = last_risk["level"]
            if strategy not in ("buy_hold", "monthly_equal") and (risk_mode == "daily" or d in reviews):
                last_risk = risk.evaluate(state["nav"], regime.state(d)["multiplier"] if regime is not None else 1., date=d)
            exposure_now = exp @ pd.Series(state["weights"], dtype=float).reindex(exp.columns,fill_value=0)
            breaches = exposure_now[exposure_now > cfg.portfolio.max_sector_weight + 1e-6].to_dict()
            states.append(dict(date=d, **last_risk, actual_sector_exposure=exposure_now.to_dict(),
                               unresolved_sector_breaches=breaches))
            eligible, records = p["eligible"][d], morning
            # Stop decisions are close-time intents, never fabricated fills.
            stops = [c for c in account.quantities() if account.cost_basis(c)>0 and
                     account.marks[c] / account.cost_basis(c) - 1 < cfg.risk.stop_loss_single]
            if strategy in ("buy_hold", "monthly_equal") or (risk_mode != "daily" and d not in reviews): stops = []
            gross_breach = (risk_mode == "daily" or d in reviews) and last_risk["gross_scale"] < 1 and sum(state["weights"].values()) > last_risk["gross_scale"] + 1e-4
            risk_exit = last_risk["level"] > old_level or bool(stops) or gross_breach
            target = None
            reason = ""
            trace = {}
            if risk_exit:
                weights = account.weights()
                scale = min(1., last_risk["gross_scale"] / max(sum(weights.values()), 1e-12))
                if last_risk["level"] == 1: scale = min(scale, cfg.risk.vol_reduction_factor)
                target = {c: w * scale if c not in stops else 0. for c, w in weights.items()}
                reason = "risk_exit"
                account.cancel_buys(reason); delayed.clear()
            elif strategy == "schedule" and d in target_schedule:
                target, reason = target_schedule[d], "scheduled_target"
            elif d in reviews and not (strategy == "buy_hold" and invested):
                selected = []
                if strategy == "repaired_legacy":
                    decision = p.get("decisions", {}).get(d) or svc.score(d, eligible)
                    trace = decision.trace
                    if decision.status != "ok":
                        traces.append({"date": str(d), "status": decision.status, "reasons": decision.reasons})
                        continue
                    ranks = decision.scores.rank(pct=True)
                    if cfg.capital.rebalance_rule == "monthly":
                        selected = decision.scores.sort_values(ascending=False, kind="stable").index[:cfg.capital.a_share_positions].tolist()
                    else:
                        for c in account.quantities():
                            outside_count[c] = outside_count.get(c, 0)+1 if ranks.get(c, 0) < cfg.research.exit_percentile else 0
                        kept = [c for c in account.quantities() if c in eligible and
                                (account.holding_age(c, session)<cfg.capital.min_holding_period_days or outside_count.get(c, 0)<cfg.research.exit_reviews)]
                        new = ranks[ranks >= cfg.research.entry_percentile].sort_values(ascending=False).index
                        selected = list(dict.fromkeys(kept + list(new)))[:cfg.capital.a_share_positions]
                elif strategy == "random":
                    selected = [c for c in random_order if c in eligible][:basket_size]
                else:
                    selected = eligible
                # Retain held names younger than the ordinary minimum holding interval.
                if strategy in ("repaired_legacy", "random"):
                    young = [c for c in account.quantities() if account.holding_age(c, session)<cfg.capital.min_holding_period_days]
                    selected = list(dict.fromkeys(young + selected))[:cfg.capital.a_share_positions]
                selected = [c for c in selected if c in rules and c in close]
                if not selected: continue
                exposures = exposure_matrix(dataset, d, records, selected)
                ret = close.loc[:d, selected].pct_change(fill_method=None).tail(cfg.portfolio.vol_est_window)
                if strategy in ("buy_hold", "monthly_equal", "vol_matched_equal"):
                    target = {c: min((1-cfg.research.cash_buffer)/len(selected), cfg.research.max_stock_weight) for c in selected}
                    x = pd.Series(target)
                    se = exposures @ x
                    if se.max() > cfg.portfolio.max_sector_weight: x *= cfg.portfolio.max_sector_weight / se.max()
                    if strategy == "vol_matched_equal":
                        clean = ret.dropna()
                        if len(clean)<40: continue
                        vol = float((clean @ x).std()*np.sqrt(252))
                        x *= min(1., cfg.portfolio.target_vol*last_risk["vol_scale"]/max(vol, 1e-12))
                    x *= min(1., last_risk["gross_scale"]/max(x.sum(), 1e-12))
                    target, status = x.to_dict(), "ok"
                else:
                    x, status = allocate(ret, records, exposures, cfg, last_risk["gross_scale"], last_risk["vol_scale"])
                    target = x.to_dict()
                if status != "ok":
                    allocation_issues.append({"date": str(d), "reason": status})
                    account.cancel_buys(status)
                    continue
                if not last_risk.get("allow_new_stock", True):
                    target = {c: min(w, account.weights().get(c, 0.)) for c, w in target.items()}
                if strategy in ("repaired_legacy", "random"):
                    now_w = account.weights()
                    locked = {c: now_w[c] for c in now_w if account.holding_age(c, session)<cfg.capital.min_holding_period_days}
                    for c, w in locked.items(): target[c] = max(target.get(c, 0.), w)
                    free = {c: w for c,w in target.items() if c not in locked}
                    budget = max(0., min(1-cfg.research.cash_buffer,last_risk["gross_scale"]) - sum(target[c] for c in locked))
                    scale = min(1., budget/max(sum(free.values()),1e-12))
                    for c in free: target[c] *= scale
                reason = strategy
            if target is not None:
                next_dates = dataset.calendar[dataset.calendar > d]
                exec_date = next_dates[execution_delay] if len(next_dates)>execution_delay else None
                if execution_delay:
                    delayed.append((session+1+execution_delay, target, d, reason))
                else: account.queue_targets(target, d, reason, rules)
                invested = True
                decisions.append({"signal_date": str(d.date()), "exec_date": str(exec_date.date()) if exec_date is not None else None,
                                  "weights": target, "factor_weights": trace.get("factor_weights", {}),
                                  "experts": trace.get("expert_weights", {}), "risk": dict(last_risk),
                                  "regime": regime.state(d) if regime is not None else {}, "reason": reason})
                traces.append({"date": str(d.date()), "status": "ok", **trace, "target_weights": target,
                               "raw_close": {c: prices.get(c) for c in target}})
        nav = pd.Series({s["date"]: s["nav"] / cfg.capital.init_cash for s in account.snapshots}, name="nav")
        ret = nav.pct_change(fill_method=None)
        if len(ret): ret.iloc[0] = nav.iloc[0]-1
        dd = nav / nav.cummax().clip(lower=1.) - 1
        status = "diagnostic_proxy" if diagnostic else "research_unvalidated"
        if account.data_issues: status = "unverified_data_gaps"
        result = {"nav": nav, "daily_return": ret, "drawdown": dd, "trades": decisions,
                  "weight_history": [pd.Series(x["weights"], name=x["date"], dtype=float) for x in account.snapshots],
                  "factor_panels": {k:v.loc[:end] for k,v in svc.factors.items()}, "ic_matrix": svc.ic.loc[:end], "codes": list(close), "market": self.market,
                  "run_id": run_id, "data_version": p["version"], "model_version": cfg.research.model_version,
                  "as_of": str(end), "status": status, "reasons": errors, "promotion_approved": False,
                  "config": config_dump, "benchmark": dataset.benchmark.copy(), "benchmark_manifest": dataset.manifest,
                  "risk_history": states, "signal_trace": traces, "allocation_issues": allocation_issues,
                  "pending_intents": delayed, "strategy": strategy, "risk_mode": risk_mode,
                  "input_panels": dataset.panels, "input_instruments": dataset.instruments,
                  "input_actions": dataset.actions, "calendar": dataset.calendar, **account.export()}
        return result


def close_ret_only(c2c, top, t, window):
    past = c2c.loc[:t].tail(window)[top].dropna(how="all")
    return past if len(past)>=40 else None


class LegacyMonthlyRisk:
    """Frozen old control policy for attribution only, not a live risk manager."""
    def __init__(self, cfg):
        self.peak = 0.
        self.cfg = cfg
    def evaluate(self, value, regime_mult=1., date=None):
        self.peak = max(self.peak, value)
        dd = value/self.peak-1
        gross = .3 if dd < -.22 or regime_mult <= .55 else .7 if dd < -.15 else 1.
        vol = .75 if dd < -.10 else 1.
        level = 3 if gross <= .3 else 2 if gross < 1 else 1 if vol < 1 else 0
        return dict(drawdown=dd, gross_scale=gross, vol_scale=vol, level=level,
                    allow_new_stock=True, reasons=["frozen_legacy_monthly_policy"])
