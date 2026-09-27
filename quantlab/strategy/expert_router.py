"""Research-only conditional mixture of existing experts.

Inspired by Qlib TRA's use of historical errors and context for routing. This
is NOT the neural TRA model: it fits a convex, regularised blend of frozen
expert ranks. Only matured cohorts enter fitting. Context-specific sufficient
statistics shrink toward the global history, and an L2 prior stabilises highly
correlated experts. No account permissions or production defaults are changed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODEL_VERSION = 'conditional-expert-mixture-v1'
STATES = ('active', 'normal', 'defensive', 'unknown')


@dataclass(frozen=True)
class RouterPolicy:
    min_cohorts: int = 12
    max_cohorts: int = 36
    state_prior: float = 12.0
    ridge: float = .01
    min_names: int = 30
    min_coverage: float = .95

    def __post_init__(self):
        if not 1 <= self.min_cohorts <= self.max_cohorts or self.min_names < 3:
            raise ValueError('invalid_router_history')
        if not 0 < self.min_coverage <= 1 or not np.isfinite(self.state_prior) or self.state_prior <= 0:
            raise ValueError('invalid_router_coverage_or_prior')
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('invalid_router_ridge')


@dataclass(frozen=True)
class ExpertCohort:
    signal_date: pd.Timestamp
    available_at: pd.Timestamp
    state: str
    experts: tuple[str, ...]
    gram: np.ndarray
    cross: np.ndarray
    requested: int
    observed: int
    status: str


def cohort_from_outcomes(signal_date, available_at, state, scores, outcomes,
                         policy=RouterPolicy()):
    """Summarise one dated cross-section, keeping unknown labels out of training.

    Outcome ranks are evaluation targets, not promised or tradeable returns.
    Caller must supply the real label endpoint and data availability timestamp.
    """
    signal_date, available_at = pd.Timestamp(signal_date), pd.Timestamp(available_at)
    if pd.isna(signal_date) or pd.isna(available_at) or available_at <= signal_date:
        raise ValueError('label_must_mature_after_signal')
    if state not in STATES:
        raise ValueError('unknown_market_state')
    x = pd.DataFrame(scores, dtype=float).copy()
    if x.empty or x.index.has_duplicates or x.columns.has_duplicates:
        raise ValueError('invalid_expert_cross_section')
    valid_scores = np.isfinite(x).all(axis=1)
    x = x.loc[valid_scores]
    if ((x < 0) | (x > 1)).any().any():
        raise ValueError('expert_predictions_must_be_percentile_ranks')
    y = pd.Series(outcomes, dtype=float).reindex(x.index)
    observed = np.isfinite(y)
    requested, count = len(x), int(observed.sum())
    status = ('insufficient_members' if count < policy.min_names else
              'insufficient_label_coverage' if count / max(requested, 1) < policy.min_coverage else
              'constant_outcomes' if y[observed].nunique() < 2 else 'ok')
    n = len(x.columns)
    gram, cross = np.zeros((n, n)), np.zeros(n)
    if status == 'ok':
        a = x.loc[observed].to_numpy() - .5
        b = y[observed].rank(method='average', pct=True).to_numpy() - .5
        gram, cross = a.T @ a / count, a.T @ b / count
    return ExpertCohort(signal_date, available_at, state, tuple(x.columns),
                        gram, cross, requested, count, status)


@dataclass
class RoutedDecision:
    status: str
    scores: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)


class ExpertRouter:
    def __init__(self, cohorts, policy=RouterPolicy()):
        self.cohorts = tuple(cohorts)
        self.policy = policy
        dates = [row.signal_date for row in self.cohorts]
        if len(set(dates)) != len(dates):
            raise ValueError('duplicate_expert_cohort')

    def decide(self, as_of, state, scores):
        as_of = pd.Timestamp(as_of)
        if pd.isna(as_of) or state not in STATES:
            raise ValueError('invalid_router_decision_context')
        x = pd.DataFrame(scores, dtype=float)
        if x.empty or x.index.has_duplicates or x.columns.has_duplicates:
            return RoutedDecision('invalid_expert_cross_section')
        names = tuple(x.columns)
        # Both timestamps are checked before any label statistic is consulted.
        matured = [r for r in self.cohorts if r.signal_date < as_of and r.available_at <= as_of]
        if any(r.experts != names for r in matured):
            raise ValueError('expert_schema_changed')
        valid = sorted((r for r in matured if r.status == 'ok'),
                       key=lambda r: r.signal_date)[-self.policy.max_cohorts:]
        evidence = dict(model_version=MODEL_VERSION, as_of=str(as_of), state=state,
                        matured=len(matured), used=len(valid),
                        exclusions=dict(Counter(r.status for r in matured if r.status != 'ok')))
        if len(valid) < self.policy.min_cohorts:
            return RoutedDecision('insufficient_matured_history', evidence=evidence)
        g = np.mean([r.gram for r in valid], axis=0)
        b = np.mean([r.cross for r in valid], axis=0)
        matched = [r for r in valid if r.state == state] if state != 'unknown' else []
        alpha = len(matched) / (len(matched) + self.policy.state_prior)
        if matched:
            g = (1 - alpha) * g + alpha * np.mean([r.gram for r in matched], axis=0)
            b = (1 - alpha) * b + alpha * np.mean([r.cross for r in matched], axis=0)
        prior = np.full(len(names), 1 / len(names))
        h = g + self.policy.ridge * np.eye(len(names))
        c = b + self.policy.ridge * prior
        solve = minimize(lambda w: float(w @ h @ w - 2 * c @ w), prior,
                         jac=lambda w: 2 * (h @ w - c), method='SLSQP',
                         bounds=[(0., 1.)] * len(names),
                         constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1,
                                       'jac': lambda w: np.ones(len(w))}],
                         options={'ftol': 1e-12, 'maxiter': 200})
        w = solve.x
        evidence.update(state_cohorts=len(matched), state_history_weight=alpha,
                        latest_label_available_at=str(max(r.available_at for r in valid)),
                        oldest_signal_date=str(valid[0].signal_date),
                        latest_signal_date=str(valid[-1].signal_date),
                        objective='date_balanced_conditional_rank_squared_error_plus_L2')
        if (not solve.success or not np.isfinite(w).all() or abs(w.sum() - 1) > 1e-8
                or w.min() < -1e-8 or w.max() > 1 + 1e-8):
            evidence['solver_message'] = str(solve.message)
            return RoutedDecision('solver_failed', evidence=evidence)
        valid_rows = np.isfinite(x).all(axis=1) & x.ge(0).all(axis=1) & x.le(1).all(axis=1)
        result = x.loc[valid_rows] @ w
        if result.nunique() < 2:
            return RoutedDecision('constant_or_missing_predictions', evidence=evidence)
        return RoutedDecision('ok', result, dict(zip(names, map(float, w))), evidence)


def select_reference_basket(heads, current, industries, *, priorities=None,
                            holdings=50, retention_rank=150, industry_slots=15,
                            selection_policy='reference_retention'):
    """Research policies share financial tiers, missing slots and sector limits.

    Default retains the original reference ordering for held targets. The
    explicit challenger uses one within-tier score order for retention and
    admission. Neither policy converts target holdings into assumed fills.
    """
    if selection_policy not in ('reference_retention', 'score_consistent_retention'):
        raise ValueError('unknown_selection_policy')
    if holdings < 1 or retention_rank < holdings or industry_slots < 1:
        raise ValueError('invalid_reference_basket_limits')
    if any(not np.isfinite(v) for v in heads.values()):
        raise ValueError('invalid_reference_score')
    priorities = priorities or {}
    def reorder(sequence):
        ordered = list(sequence)
        for tier in sorted(set(heads.values()), reverse=True):
            slots = [i for i, c in enumerate(ordered) if heads[c] == tier
                     and c in priorities and np.isfinite(priorities[c])]
            names = sorted((ordered[i] for i in slots), key=lambda c: (-priorities[c], c))
            for i, code in zip(slots, names):
                ordered[i] = code
        return ordered
    ranked = sorted(heads, key=lambda code: (-heads[code], code))
    if selection_policy == 'score_consistent_retention':
        ranked = reorder(ranked)
    retained = [c for c in ranked[:retention_rank] if c in current]
    admitted = [c for c in ranked if c not in retained]
    if selection_policy == 'reference_retention':
        admitted = reorder(admitted)
    chosen, counts = [], Counter()
    for code in retained + admitted:
        industry = industries.get(code, 'unclassified')
        if counts[industry] >= industry_slots:
            continue
        chosen.append(code)
        counts[industry] += 1
        if len(chosen) == holdings:
            break
    return tuple(chosen)
