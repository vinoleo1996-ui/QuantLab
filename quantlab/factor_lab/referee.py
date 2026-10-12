"""Frozen referee for LLM-proposed factors (after "Propose, Don't Judge", arXiv 2609.27051).

Two gates, neither of which the proposer can influence:

1. Historical screen: rank IC on non-overlapping H=20 observations (one per 20 sessions) in the screen
   window 2015-2021, re-checked once on the 2022-2026 holdout; low correlation with the existing library.
   Passing it only makes a candidate eligible for the forward stream.
2. Forward admission: from the submission date, each session's signed next-day rank IC ``X`` (score at
   close t-1 against the open-to-open return of t) updates a test supermartingale
   ``W *= 1 + lam (X - delta)`` under ``H0: E[X] <= delta``; ``lam <= 1/(1+delta)`` keeps ``W >= 0``.
   Daily observations are used because one H=20 observation per 20 sessions would need decades of
   evidence; the 20-day horizon is checked by the historical screen. Admission is online e-BH over a
   frozen number of slots; retirement is an e-detector betting the other way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.research_statistics import newey_west

SCREEN = ('2015-01-01', '2021-12-31')
HOLDOUT = ('2022-01-01', '2026-12-31')
MIN_ABS_IC, MIN_SCREEN_T, MIN_HOLDOUT_T, MAX_LIBRARY_CORR = 0.02, 3.0, 2.0, 0.5
DELTA, LAMBDA, ALPHA, SLOTS = 0.005, 0.9, 0.10, 100
RETIRE_DELTA, RETIRE_THRESHOLD = 0.0, 100.0


def daily_ic(values, labels):
    """Spearman correlation per date between two long frames with ``date, code, value``/``label``."""
    frame = values.merge(labels, on=['date', 'code']).dropna(subset=['value', 'label'])
    return frame.groupby('date').apply(lambda g: g['value'].corr(g['label'], method='spearman')
                                       if len(g) > 20 else np.nan).dropna()


def _t(series):
    if len(series) < 8:
        return float('nan')
    _, t = newey_west(series.to_numpy(), lags=2)
    return float(t[0])


def screen(values, labels, library, observation_days):
    """``values``: long frame of the candidate; ``library``: long frame with existing features as columns."""
    observed = values[values['date'].isin(observation_days)]
    ic = daily_ic(observed, labels)
    in_screen = ic.loc[SCREEN[0]:SCREEN[1]]
    holdout = ic.loc[HOLDOUT[0]:HOLDOUT[1]]
    sign = float(np.sign(in_screen.mean())) if len(in_screen) else 0.0
    merged = observed.merge(library, on=['date', 'code'])
    features = [c for c in library.columns if c not in ('date', 'code')]
    corr = {f: float(merged.groupby('date').apply(lambda g: g['value'].corr(g[f], method='spearman')).mean())
            for f in features} if features and len(merged) else {}
    max_corr = max((abs(v) for v in corr.values() if np.isfinite(v)), default=0.0)
    result = dict(screen_mean_ic=float(in_screen.mean()) if len(in_screen) else float('nan'),
                  screen_t=_t(in_screen), screen_n=len(in_screen),
                  holdout_mean_ic=float(holdout.mean()) if len(holdout) else float('nan'),
                  holdout_t=_t(holdout), holdout_n=len(holdout), sign=sign,
                  max_library_corr=max_corr, closest_feature=max(corr, key=lambda k: abs(corr[k])) if corr else None)
    result['passed'] = bool(abs(result['screen_mean_ic']) >= MIN_ABS_IC and abs(result['screen_t']) >= MIN_SCREEN_T
                            and np.sign(result['holdout_mean_ic']) == sign and sign * result['holdout_t'] >= MIN_HOLDOUT_T
                            and max_corr < MAX_LIBRARY_CORR)
    return result


@dataclass
class ForwardStream:
    """State of one candidate after submission. ``sign`` fixes the direction at submission time."""
    sign: float
    submitted_at: str
    wealth: float = 1.0
    observations: list = field(default_factory=list)
    retire_processes: list = field(default_factory=list)
    admitted_at: str | None = None
    retired_at: str | None = None

    def update(self, date, ic):
        x = float(np.clip(self.sign * ic, -1.0, 1.0))
        self.observations.append((str(date), x))
        self.wealth *= 1 + LAMBDA * (x - DELTA)
        if self.admitted_at and not self.retired_at:
            self.retire_processes = [w * (1 + LAMBDA * (RETIRE_DELTA - x)) for w in self.retire_processes] + [1.0]
            if sum(self.retire_processes) >= RETIRE_THRESHOLD:
                self.retired_at = str(date)
        return self.wealth


def online_ebh(wealth, alpha=ALPHA, slots=SLOTS):
    """Candidates admitted by e-BH with equal weights ``1/slots`` over a frozen slot count."""
    values = sorted(wealth.values(), reverse=True)
    k_star = 0
    for k in range(1, len(values) + 1):
        if sum(v >= slots / (k * alpha) for v in values) >= k:
            k_star = k
    if k_star == 0:
        return set()
    bar = slots / (k_star * alpha)
    return {name for name, value in wealth.items() if value >= bar}


def admit(streams, date):
    """Freeze wealth at admission and mark newly admitted candidates; returns the admitted set."""
    live = {name: s.wealth for name, s in streams.items() if not s.retired_at}
    chosen = online_ebh(live)
    for name in chosen:
        if streams[name].admitted_at is None:
            streams[name].admitted_at = str(date)
    return {name for name, s in streams.items() if s.admitted_at and not s.retired_at}
