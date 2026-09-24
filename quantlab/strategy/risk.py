"""Daily drawdown controls: 10/15/18/20 percent, five-session recovery hysteresis.
Regime is an attributed diagnostic only under this policy; frozen legacy control
retains the old regime rule for explicit comparisons. No automatic human resume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import CONFIG, RiskConfig


class Regime:
    """基于宽基指数的市场状态：牛熊趋势 + 波动率分位。"""

    def __init__(self, bench_close: pd.Series, cfg: RiskConfig | None = None):
        self.cfg = cfg or CONFIG.risk
        self.bench = bench_close.sort_index()
        self.ma20 = self.bench.rolling(20).mean()
        self.ma60 = self.bench.rolling(60).mean()
        ret = self.bench.pct_change(fill_method=None)
        self.vol = ret.rolling(120).std() * np.sqrt(252)

    def _thresh(self, date: pd.Timestamp) -> float:
        """滚动(扩张窗口)波动率分位阈值：只用截至 date 的数据，杜绝全样本窥探。"""
        v = self.vol.loc[self.vol.index <= date].dropna()
        if v.size < 120:
            return float(v.median()) if v.size else float("inf")
        return float(v.quantile(self.cfg.regime_vol_quantile))

    def state(self, date: pd.Timestamp) -> dict:
        prev = self.bench.index[self.bench.index <= date]
        if prev.size == 0:
            return {"trend": 0, "vol_high": False, "multiplier": 1.0}
        t = prev[-1]
        trend = 1 if self.ma20.get(t) is not None and self.ma20.get(t) and self.ma20.loc[t] > self.ma60.loc[t] else -1
        thresh = self._thresh(t)
        vol_high = self.vol.loc[t] > thresh if t in self.vol.index and np.isfinite(self.vol.loc[t]) else False
        m = 1.0
        if vol_high and trend < 0:
            m = 0.55          # 深熊：高波动 + 下趋势
        elif trend < 0:
            m = 0.85
        elif vol_high:
            m = 0.90
        return {"trend": trend, "vol_high": vol_high, "multiplier": m}


class RiskManager:
    """Daily close risk state. Hysteresis is counted in exchange sessions, not calls."""
    def __init__(self, cfg=None, research_cfg=None, target_vol=None):
        self.cfg = cfg or CONFIG.risk
        self.research_cfg = research_cfg or CONFIG.research
        self.target_vol = CONFIG.portfolio.target_vol if target_vol is None else target_vol
        self.peak = None
        self.dd = self.max_dd = 0.0
        self.level = 0
        self.vol_scale = self.gross_scale = 1.0
        self.recovery_count = 0
        self.last_date = None
        self.manual_review = False

    def approve_resume(self, *, approved_by: str):
        if not approved_by.strip():
            raise ValueError("named human approval required")
        # Approval unlocks recovery; it cannot erase losses or reset the control peak.
        self.manual_review = False
        self.level = 3
        self.recovery_count = 0
        self.resume_approval = approved_by

    def evaluate(self, portfolio_value: float, regime_mult: float = 1.0, date=None):
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise ValueError("invalid portfolio equity")
        if date is not None and self.last_date is not None and pd.Timestamp(date) < self.last_date:
            raise ValueError("risk dates must be monotone")
        fresh = date is None or self.last_date != pd.Timestamp(date)
        self.peak = max(self.peak or portfolio_value, portfolio_value)
        self.dd = portfolio_value / self.peak - 1
        self.max_dd = min(self.max_dd, self.dd)
        control_dd = self.dd
        thresholds = (self.cfg.drawdown_alert, self.cfg.drawdown_reduce, self.cfg.drawdown_exit, self.cfg.drawdown_halt)
        desired = sum(-control_dd >= x - 1e-12 for x in thresholds)
        if desired > self.level:
            self.level, self.recovery_count = desired, 0
        elif self.level and self.level < 4 and fresh:
            if -control_dd < thresholds[self.level - 1] - self.research_cfg.recovery_buffer:
                self.recovery_count += 1
                if self.recovery_count >= self.research_cfg.recovery_sessions:
                    self.level -= 1
                    self.recovery_count = 0
            else:
                self.recovery_count = 0
        if self.level == 4: self.manual_review = True
        if self.manual_review: self.level = 4
        self.vol_scale = self.cfg.vol_reduction_factor if self.level else 1.0
        self.gross_scale = (1.0, 1.0, 0.60, 0.30, 0.0)[self.level]
        self.last_date = pd.Timestamp(date) if date is not None else None
        reasons = [f"drawdown_level_{self.level}"] if self.level else []
        return {"portfolio_value": portfolio_value, "drawdown": self.dd, "max_drawdown": self.max_dd,
                "control_drawdown": control_dd, "level": self.level, "vol_scale": self.vol_scale,
                "gross_scale": self.gross_scale, "target_vol": self.target_vol * self.vol_scale,
                "manual_review": self.manual_review, "allow_new_stock": self.level < 3,
                "recovery_count": self.recovery_count, "reasons": reasons}

    def single_stock_action(self, symbol, ret_since_buy, score_rank, top_cutoff):
        if ret_since_buy < self.cfg.stop_loss_single: return "sell"
        return "watch" if score_rank > top_cutoff else "hold"
