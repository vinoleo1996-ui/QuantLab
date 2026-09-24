"""Shared, dated legacy-factor decision service for repair attribution and monitoring.

The opportunity challenger is deliberately gated by research evidence. A repaired legacy
model is not relabelled as the proposed new alpha module.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.factors.engine import FactorEngine
from quantlab.strategy.scoring import adaptive_weight, composite_score, decay_penalty, factor_crowding_penalty, weight_row
from quantlab.strategy.experts import experts_score


@dataclass
class ScoreDecision:
    as_of: str
    status: str
    scores: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    reasons: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)


def review_dates(calendar, rule="monthly", every_n_weeks=4):
    """Calendar must include the next exchange session; never infer period end from a truncated sample."""
    idx = pd.DatetimeIndex(calendar)
    if rule not in ("weekly", "monthly", "every_n_weeks"): raise ValueError("unknown rebalance_rule")
    period = "M" if rule == "monthly" else "W-FRI"
    dates = [idx[i] for i in range(len(idx)-1) if idx[i].to_period(period) != idx[i+1].to_period(period)]
    return dates[::every_n_weeks] if rule == "every_n_weeks" else dates


class FactorDecisionService:
    def __init__(self, close, amount, turnover, calendar, cfg=None):
        self.cfg = cfg or CONFIG
        self.close = close
        # All transformations here are rolling or cross-sectional, with no full-sample fit.
        self.factors = FactorEngine("a", cfg=self.cfg.factors).compute(close, amount, turnover)
        self.months = review_dates(calendar, "monthly")
        self.ic = pd.DataFrame(index=self.months, columns=list(self.factors), dtype=float)
        for prev, t in zip(self.months, self.months[1:]):
            if prev not in close.index or t not in close.index: continue
            y = close.loc[t] / close.loc[prev] - 1
            for name, f in self.factors.items():
                x = f.loc[prev].dropna()
                pair = pd.concat([x, y.reindex(x.index)], axis=1).dropna()
                if len(pair) >= 10 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                    self.ic.loc[t, name] = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())

    def score(self, date, eligible=None):
        t = pd.Timestamp(date)
        fc = self.cfg.factors
        if t not in self.close.index: return ScoreDecision(str(t), "invalid", reasons=["missing_signal_bar"])
        full = {k: v.loc[[t]] for k, v in self.factors.items() if t in v.index and v.loc[t].notna().any()}
        if len(full) < 6: return ScoreDecision(str(t), "warmup", reasons=["insufficient_factors"])
        hist = self.ic.loc[self.ic.index <= t, list(full)]
        weights = pd.Series(1., index=list(full))
        for name in full:
            values = hist[name].dropna().tail(fc.ic_window)
            if len(values) >= 24:
                ir = values.mean() / (values.std() + 1e-9)
                weights[name] = adaptive_weight(ir, fc.icir_floor, fc.icir_cap) * decay_penalty(values, fc.factor_decay_lookback)
        pen, _ = factor_crowding_penalty(hist, fc.crowding_corr_window, fc.crowding_corr_alert)
        if not pen.empty:
            pen = pen.reindex(pen.index.union([t])).sort_index().ffill().loc[[t]]
        row = weight_row(weights, t)
        try:
            scores = composite_score(full, row, pen)
            exp = {}
            if fc.use_experts:
                scores, exp = experts_score(full, row, pen, hist, lookback=fc.expert_ic_lookback)
            last = scores.loc[t].dropna() if scores is not None and t in scores.index else pd.Series(dtype=float)
        except ValueError as e:
            return ScoreDecision(str(t), "invalid", reasons=[str(e)])
        if eligible is not None: last = last.reindex(eligible).dropna()
        crowd = full.get("factor_price_crowding")
        if crowd is not None:
            low_crowd = crowd.loc[t].reindex(last.index)
            last = last[low_crowd.rank(pct=True) > (1 - fc.crowding_quantile)]
        status = "ok" if len(last) >= 3 and last.nunique() > 1 else "invalid"
        trace = {"standardized_features": {k: v.loc[t].dropna().to_dict() for k, v in full.items()},
                 "factor_weights": weights.to_dict(), "expert_weights": exp,
                 "scores": last.to_dict(), "last_matured_ic": str(hist.index[-1]) if len(hist) else None}
        return ScoreDecision(str(t), status, last if status == "ok" else pd.Series(dtype=float),
                             [] if status == "ok" else ["constant_or_insufficient_scores"], trace)
