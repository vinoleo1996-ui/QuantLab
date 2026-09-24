"""Frozen research risk permissions; forecasts cannot enlarge these budgets.

The 25% drawdown threshold is an exit instruction, not a guaranteed fill or loss
bound. Decisions use information through the stated close and execute later.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantlab.data.contracts import fingerprint
from quantlab.strategy.risk import RiskManager


POLICY_VERSION = "risk-budget-20260925-v2"
PROFILES = {
    "defensive": dict(max_stock_weight=.06, max_sector_weight=.25,
                      max_cluster_weight=.30, cash_buffer=.20, target_vol=.12),
    "normal": dict(max_stock_weight=.08, max_sector_weight=.35,
                   max_cluster_weight=.35, cash_buffer=.02, target_vol=.18),
    "active": dict(max_stock_weight=.10, max_sector_weight=.40,
                   max_cluster_weight=.35, cash_buffer=.01, target_vol=.22),
}
STATES = tuple(PROFILES)


def policy_definition(policy_id):
    if policy_id not in ("P0", "P1", "P2"):
        raise ValueError("unknown_risk_policy")
    return dict(id=policy_id, version=POLICY_VERSION,
                profiles=deepcopy(PROFILES) if policy_id == "P2" else
                deepcopy(PROFILES["normal"]) if policy_id == "P1" else "legacy_config",
                drawdown_levels=[.12, .18, .22, .25] if policy_id != "P0" else "legacy_config",
                promotion_sessions=5, demotion_sessions=2,
                counter_key='same_desired_state_consecutive_benchmark_sessions',
                promotion_transition='one_level_toward_desired',
                demotion_transition='direct_to_confirmed_desired',
                recovery_buffer=.03, recovery_sessions=5,
                stock_buffer=.01, group_buffer=.02,
                unknown_sector=.25, unknown_cluster=.30)


@dataclass(frozen=True)
class RiskBudget:
    policy_id: str
    version: str
    state: str
    as_of: str
    max_stock_weight: float
    max_sector_weight: float
    max_cluster_weight: float
    cash_buffer: float
    target_vol: float
    gross_scale: float
    vol_scale: float
    soft_stock_weight: float
    soft_sector_weight: float
    soft_cluster_weight: float
    unknown_sector_weight: float
    unknown_cluster_weight: float
    risk: dict

    def as_dict(self):
        return asdict(self)


class PolicyController:
    """Causal market state plus the account's irreversible drawdown history.

Only fresh exchange sessions advance counters. Moving from defensive to active
requires two five-session promotions. Duplicate calls cannot accelerate either
market-state or drawdown recovery. Raising permission never creates an order.
"""
    def __init__(self, policy_id, cfg, benchmark_close):
        self.policy_id = policy_id
        definition = policy_definition(policy_id)
        self.cfg = deepcopy(cfg)
        self.benchmark = pd.Series(benchmark_close, dtype=float).sort_index().copy()
        if self.benchmark.index.has_duplicates:
            raise ValueError("duplicate_benchmark_dates")
        self.benchmark.index = pd.to_datetime(self.benchmark.index)
        risk_cfg, research_cfg = deepcopy(cfg.risk), deepcopy(cfg.research)
        if policy_id != "P0":
            (risk_cfg.drawdown_alert, risk_cfg.drawdown_reduce,
             risk_cfg.drawdown_exit, risk_cfg.drawdown_halt) = (.12, .18, .22, .25)
            risk_cfg.vol_reduction_factor = .75
            research_cfg.recovery_buffer, research_cfg.recovery_sessions = .03, 5
        self.risk_manager = RiskManager(risk_cfg, research_cfg, cfg.portfolio.target_vol)
        self.version = fingerprint(dict(definition=definition,
                                        legacy_profile=dict(stock=cfg.research.max_stock_weight,
                                                            sector=cfg.portfolio.max_sector_weight,
                                                            cluster=cfg.research.max_cluster_weight,
                                                            cash=cfg.research.cash_buffer,
                                                            vol=cfg.portfolio.target_vol) if policy_id == 'P0' else None,
                                        risk=asdict(risk_cfg), research_recovery=dict(
                                            buffer=research_cfg.recovery_buffer,
                                            sessions=research_cfg.recovery_sessions)))
        self.state = "normal"
        self.transition_count = 0
        self.transition_direction = 0
        self.transition_target = None
        self.last_date = None

    def approve_resume(self, *, approved_by):
        self.risk_manager.approve_resume(approved_by=approved_by)

    def market_snapshot(self, date):
        """Return the close-time observation, never a full-sample threshold."""
        date = pd.Timestamp(date)
        close = self.benchmark.loc[self.benchmark.index <= date]
        valid = close.notna() & np.isfinite(close) & close.gt(0)
        returns = close.where(valid).pct_change(fill_method=None)
        vol = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
        history = vol.shift(1).tail(252)
        ready = (len(close) >= 273 and close.index[-1] == date and
                 valid.tail(60).all() and history.notna().sum() == 252)
        if not ready:
            return dict(ready=False, desired_state="normal",
                        reason="market_history_incomplete")
        price, ma20, ma60 = float(close.iloc[-1]), float(close.tail(20).mean()), float(close.tail(60).mean())
        realized_vol, threshold = float(vol.iloc[-1]), float(history.quantile(.80))
        if price < ma60 and realized_vol > threshold:
            desired = "defensive"
        elif price > ma60 and ma20 > ma60 and realized_vol <= threshold:
            desired = "active"
        else:
            desired = "normal"
        return dict(ready=True, desired_state=desired, close=price, ma20=ma20,
                    ma60=ma60, realized_vol_20=realized_vol,
                    historical_vol_q80=threshold)

    def evaluate(self, date, nav, data_stale=False, exit_blocked=False):
        date = pd.Timestamp(date)
        if self.last_date is not None and date < self.last_date:
            raise ValueError("risk dates must be monotone")
        session = self.policy_id == 'P0' or date in self.benchmark.index
        fresh = self.last_date != date and session
        risk_date = date if session else self.risk_manager.last_date
        risk = self.risk_manager.evaluate(nav, date=risk_date)
        market = self.market_snapshot(date) if self.policy_id == "P2" else {}
        flags = bool(data_stale or exit_blocked)
        if self.policy_id == "P2" and fresh:
            if self.last_date in self.benchmark.index and self.benchmark.index.get_loc(date)-self.benchmark.index.get_loc(self.last_date)>1:
                self.transition_count = self.transition_direction = 0
                self.transition_target = None
            if flags:
                self.state = "defensive"
                self.transition_count = self.transition_direction = 0
                self.transition_target = None
            else:
                current = STATES.index(self.state)
                wanted = STATES.index(market["desired_state"])
                # Missing history cannot promote a defensive account.
                if not market["ready"]:
                    wanted = min(wanted, current)
                direction = int(np.sign(wanted-current))
                if direction > 0 and risk['level'] > 0:
                    direction = 0
                if direction == 0:
                    self.transition_count = self.transition_direction = 0
                    self.transition_target = None
                else:
                    desired=market['desired_state']
                    self.transition_count = self.transition_count+1 if desired == self.transition_target else 1
                    self.transition_direction = direction
                    self.transition_target = desired
                    if self.transition_count >= (5 if direction > 0 else 2):
                        self.state = STATES[current+1] if direction>0 else STATES[wanted]
                        self.transition_count = self.transition_direction = 0
                        self.transition_target = None
        if self.policy_id == "P0":
            profile = dict(max_stock_weight=self.cfg.research.max_stock_weight,
                           max_sector_weight=self.cfg.portfolio.max_sector_weight,
                           max_cluster_weight=self.cfg.research.max_cluster_weight,
                           cash_buffer=self.cfg.research.cash_buffer,
                           target_vol=self.cfg.portfolio.target_vol)
            state = "legacy"
        else:
            state = "normal" if self.policy_id == "P1" else self.state
            profile = PROFILES[state].copy()
        self.risk_manager.target_vol = profile["target_vol"]
        risk['target_vol'] = profile['target_vol'] * risk['vol_scale']
        if self.policy_id != "P0":
            risk["allow_new_stock"] = risk["allow_new_stock"] and not flags
        reasons = risk["reasons"][:]
        if data_stale: reasons.append("data_stale")
        if exit_blocked: reasons.append("exit_blocked")
        if market and not market["ready"]: reasons.append(market["reason"])
        risk.update(reasons=reasons, policy_id=self.policy_id, policy_version=self.version,
                    policy_state=state, market=market, transition_count=self.transition_count,
                    transition_direction=self.transition_direction,transition_target=self.transition_target)
        stock_buffer = 0. if self.policy_id == "P0" else .01
        group_buffer = 0. if self.policy_id == "P0" else .02
        self.last_date = date
        return RiskBudget(self.policy_id, self.version, state, str(date.date()),
                          **profile, gross_scale=risk["gross_scale"], vol_scale=risk["vol_scale"],
                          soft_stock_weight=profile["max_stock_weight"]-stock_buffer,
                          soft_sector_weight=profile["max_sector_weight"]-group_buffer,
                          soft_cluster_weight=profile["max_cluster_weight"]-group_buffer,
                          unknown_sector_weight=profile["max_sector_weight"] if self.policy_id == "P0" else .25,
                          unknown_cluster_weight=profile["max_cluster_weight"] if self.policy_id == "P0" else .30,
                          risk=risk)
