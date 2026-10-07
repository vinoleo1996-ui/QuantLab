"""Paired, time-block evidence for a finite, declared research family.

All annualised statistics use 252 sessions.  Historical selection uncertainty is
reported separately; these calculations never manufacture an unseen holdout.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd


def _returns(values):
    result = pd.Series(values, dtype=float)
    if result.empty or result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError('invalid_return_calendar')
    if not np.isfinite(result.to_numpy()).all() or (result <= -1).any():
        raise ValueError('invalid_or_missing_returns')
    return result


def paired_returns(runs, contrasts):
    """Refuse silent dropping/filling of mismatched account days."""
    series = {name: _returns(run['daily_return']) for name, run in runs.items()}
    result = {}
    for name, (candidate, control) in contrasts.items():
        a, b = series[candidate], series[control]
        if not a.index.equals(b.index):
            raise ValueError('paired_return_calendar_mismatch:' + name)
        result[name] = a - b
    frame = pd.DataFrame(result)
    if frame.isna().any().any():
        raise ValueError('family_return_calendar_mismatch')
    return frame


def simultaneous_block_ci(excess, *, block=20, samples=2000, seed=20260924,
                          confidence=.95):
    """Moving-block bootstrap with a common draw and max-|t| family interval.

    This corrects this declared family of paired hypotheses, not undocumented
    earlier searches.  Pointwise intervals remain available for comparison.
    """
    frame = pd.DataFrame(excess, dtype=float)
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('invalid_family_returns')
    if block < 1 or samples < 100 or not 0 < confidence < 1:
        raise ValueError('invalid_bootstrap_specification')
    if len(frame) < 3 * block:
        return {'status': 'insufficient_data', 'block': block, 'observations': len(frame),
                'family': list(frame), 'comparisons': {}}
    values = frame.to_numpy()
    n = len(values)
    means = values.mean(axis=0) * 252
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((samples, len(frame.columns)))
    offsets = np.arange(block)
    for i in range(samples):
        starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
        positions = (starts[:, None] + offsets).ravel()[:n]
        bootstrap[i] = values[positions].mean(axis=0) * 252
    standard_errors = bootstrap.std(axis=0, ddof=1)
    variable = standard_errors > 1e-14
    # Center on the bootstrap expectation: overlapping blocks weight the edges
    # differently, which must not inflate the simultaneous critical value.
    centered = bootstrap - bootstrap.mean(axis=0)
    t = np.zeros_like(bootstrap)
    t[:, variable] = centered[:, variable] / standard_errors[variable]
    critical = float(np.quantile(np.max(np.abs(t), axis=1), confidence))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(bootstrap, [alpha, 1 - alpha], axis=0)
    comparisons = {}
    for j, name in enumerate(frame):
        radius = critical * standard_errors[j] if variable[j] else 0.
        comparisons[name] = dict(estimate=float(means[j]), lower=float(low[j]), upper=float(high[j]),
            simultaneous_lower=float(means[j] - radius), simultaneous_upper=float(means[j] + radius),
            standard_error=float(standard_errors[j]))
    return dict(status='estimated', block=block, samples=samples, seed=seed,
                confidence=confidence, observations=n, family=list(frame),
                method='paired_moving_blocks_max_absolute_t', critical_value=critical,
                estimator='252_times_mean_daily_net_return_difference', comparisons=comparisons)


def deflated_sharpe(returns, *, trials, sharpe_std=None, historical_search_complete=False):
    """DSR approximation using observed third/fourth moments, never normal defaults.

    Serial dependence is not removed by this asymptotic statistic; paired block
    evidence is reported alongside it.  DSR is not a posterior truth probability.
    """
    r = _returns(returns).to_numpy()
    if trials < 1:
        raise ValueError('invalid_trial_count')
    n = len(r)
    if n < 3 or np.std(r, ddof=1) < 1e-14:
        return dict(status='insufficient_variation', value=None, observations=n, trials=trials)
    centered = r - r.mean()
    variance = float(np.mean(centered ** 2))
    skew = float(np.mean(centered ** 3) / variance ** 1.5)
    kurtosis = float(np.mean(centered ** 4) / variance ** 2)
    sr = float(r.mean() / np.std(r, ddof=1))
    dispersion = float(sharpe_std) if sharpe_std is not None else 1 / math.sqrt(n)
    if not math.isfinite(dispersion) or dispersion < 0:
        raise ValueError('invalid_sharpe_dispersion')
    normal = NormalDist()
    gamma = .5772156649015329
    maximum = (0. if trials == 1 else (1 - gamma) * normal.inv_cdf(1 - 1 / trials)
               + gamma * normal.inv_cdf(1 - 1 / (math.e * trials)))
    threshold = dispersion * maximum
    variance_term = 1 - skew * sr + (kurtosis - 1) * sr * sr / 4
    if variance_term <= 0:
        return dict(status='invalid_moment_variance', value=None, observations=n, trials=trials)
    value = normal.cdf((sr - threshold) * math.sqrt(n - 1) / math.sqrt(variance_term))
    return dict(status='estimated' if historical_search_complete else 'historical_search_incomplete',
                value=float(value), observations=n, trials=trials, daily_sharpe=sr,
                annualized_sharpe=sr * math.sqrt(252), periods_per_year=252,
                observed_skew=skew, observed_raw_kurtosis=kurtosis,
                trial_sharpe_standard_deviation=dispersion,
                dispersion_source='observed_declared_family' if sharpe_std is not None else 'null_approximation',
                daily_selection_threshold=threshold,
                assumptions=['asymptotic_DSR_does_not_correct_serial_dependence',
                             'historical_search_count_is_a_lower_bound_unless_certified'],
                promotion_approved=False)


def family_sharpe(runs, *, registered_trials, historical_search_complete=False):
    series = {name: _returns(run['daily_return']) for name, run in runs.items()}
    ratios = [float(r.mean() / r.std(ddof=1)) for r in series.values() if r.std(ddof=1) > 1e-14]
    dispersion = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else None
    return {name: deflated_sharpe(r, trials=registered_trials, sharpe_std=dispersion,
                                historical_search_complete=historical_search_complete)
            for name, r in series.items()}


def retrospective_segments(excess):
    """Fixed half-years; neither retraining nor a newly claimed blind test."""
    r = pd.Series(excess, dtype=float)
    rows = []
    for (year, half), group in r.groupby([r.index.year, (r.index.month > 6).astype(int) + 1]):
        rows.append(dict(period=f'{year}H{half}', start=str(group.index[0].date()),
                         end=str(group.index[-1].date()), sessions=len(group),
                         annualized_net_excess=float(group.mean() * 252)))
    return rows


def fixed_two_year_segments(excess):
    """Declared development stages, with 2026H1 explicitly outside the three."""
    r = pd.Series(excess, dtype=float)
    rows = []
    for start, end, name in (('2020-01-01', '2021-12-31', '2020-2021'),
                             ('2022-01-01', '2023-12-31', '2022-2023'),
                             ('2024-01-01', '2025-12-31', '2024-2025')):
        group = r.loc[start:end]
        rows.append(dict(period=name, start=start, end=end, sessions=len(group),
                         complete=bool(len(group) >= 400 and group.index[0].year == int(start[:4])
                                       and group.index[-1].year == int(end[:4])),
                         annualized_net_excess=float(group.mean() * 252) if len(group) else None))
    tail = r.loc['2026-01-01':'2026-06-30']
    return dict(stages=rows, observation_2026H1=dict(sessions=len(tail),
        annualized_net_excess=float(tail.mean() * 252) if len(tail) else None,
        counts_toward_three_stages=False))


def newey_west(y, x=None, lags=20):
    """OLS coefficients and Newey-West t statistics; ``x=None`` tests the mean of ``y``."""
    y = np.asarray(y, dtype=float)
    X = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for lag in range(1, min(lags, len(y) - 1) + 1):
        w = 1 - lag / (lags + 1)
        g = xe[lag:].T @ xe[:-lag]
        S += w * (g + g.T)
    inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(inv @ S @ inv))
    return beta, beta / se


def win_rates(returns, benchmark=None):
    """The three pre-registered win rates from daily simple returns (all reported, none chosen).

    monthly_positive: share of calendar months with a positive return;
    monthly_beat: share of months beating ``benchmark`` (same calendar);
    rolling_12m_positive: share of month-ends whose trailing 12-month return is positive.
    """
    r = _returns(returns)
    monthly = (1 + r).groupby(r.index.to_period('M')).prod() - 1
    out = {'months': int(len(monthly)), 'monthly_positive': float((monthly > 0).mean())}
    trailing = (1 + monthly).rolling(12).apply(np.prod, raw=True) - 1
    trailing = trailing.dropna()
    out['rolling_12m_positive'] = float((trailing > 0).mean()) if len(trailing) else float('nan')
    if benchmark is not None:
        b = pd.Series(benchmark, dtype=float).reindex(r.index)
        if b.isna().any():
            raise ValueError('benchmark_calendar_mismatch')
        bench = (1 + b).groupby(b.index.to_period('M')).prod() - 1
        out['monthly_beat'] = float((monthly > bench).mean())
    return out
