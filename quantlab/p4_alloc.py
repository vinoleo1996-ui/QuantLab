"""P4 cross-asset allocation (docs/prereg/P4_ALLOC.md): fixed equity share, inverse-vol ETF sleeve,
one drawdown line with a recovery threshold, monthly rebalance effective the next session."""
from __future__ import annotations

import numpy as np
import pandas as pd

ETF_CODES = {'gold': 'sh518880', 'nasdaq': 'sh513100', 'sp500': 'sh513500', 'treasury': 'sh511010'}
EQUITY_COST, ETF_COST = 0.004, 0.0005
VOL_WINDOW = 120
DISASTER_DD, RECOVER_DD, DISASTER_EQUITY = 0.18, 0.08, 0.50


def month_ends(sessions):
    frame = pd.Series(sessions, index=sessions)
    return set(frame.groupby([sessions.year, sessions.month]).max())


def quarter_ends(sessions):
    frame = pd.Series(sessions, index=sessions)
    return set(frame.groupby([sessions.year, (sessions.month - 1) // 3]).max())


def rebalance_ends(sessions, rebalance='month'):
    if rebalance == 'month':
        return month_ends(sessions)
    if rebalance == 'quarter':
        return quarter_ends(sessions)
    raise ValueError(f'unknown_rebalance:{rebalance}')


def target_weights(etf_returns_upto, equity_share):
    vol = etf_returns_upto.tail(VOL_WINDOW).std() if len(etf_returns_upto) >= VOL_WINDOW else None
    names = list(etf_returns_upto.columns)
    if vol is None or not np.isfinite(vol.to_numpy()).all() or (vol <= 0).any():
        etf = np.full(len(names), 1 / len(names))
    else:
        inverse = 1 / vol.to_numpy()
        etf = inverse / inverse.sum()
    return np.concatenate([[equity_share], (1 - equity_share) * etf])


QDII_SLEEVES, PREMIUM_PAUSE = ('nasdaq', 'sp500'), 0.03


def pause_qdii(target, current, names, premium_today, overflow=None):
    """A QDII sleeve whose premium is above the pause threshold may not exceed its current weight;
    the freed weight goes to ``overflow`` (P4_ALLOC amendment 1 defaults to treasury)."""
    overflow = {'treasury': 1.0} if overflow is None else dict(overflow)
    if abs(sum(overflow.values()) - 1.0) > 1e-9:
        raise ValueError('qdii_overflow_weights_must_sum_to_one')
    missing = [sleeve for sleeve in overflow if sleeve not in names]
    if missing:
        raise ValueError(f'qdii_overflow_sleeve_missing:{missing}')
    target = target.copy()
    paused = []
    for sleeve in QDII_SLEEVES:
        if sleeve not in names or premium_today is None:
            continue
        value = premium_today.get(sleeve)
        i = names.index(sleeve)
        if value is not None and np.isfinite(value) and value > PREMIUM_PAUSE and target[i] > current[i]:
            freed = target[i] - current[i]
            target[i] = current[i]
            for destination, fraction in overflow.items():
                target[names.index(destination)] += freed * fraction
            paused.append(sleeve)
    return target, paused


def allocate(equity_nav, etf_returns, equity_share=0.60, cost_mult=1.0, premium=None,
             rebalance='month', drawdown_check='rebalance', overflow=None, target_fn=None):
    """Daily NAV (starting at 1) and a trace. ``etf_returns`` columns are the ETF sleeves, same index.
    ``premium``: optional date x sleeve frame of QDII premium known at each close. ``rebalance`` is
    ``month`` or ``quarter``; ``drawdown_check`` is ``rebalance`` (default, legacy) or ``daily``.
    ``overflow`` is the destination split for paused QDII weight. ``target_fn`` replaces the sleeve rule."""
    sessions = equity_nav.index
    eq = equity_nav.pct_change(fill_method=None).fillna(0.0)
    returns = pd.concat([eq.rename('equity'), etf_returns.reindex(sessions).fillna(0.0)], axis=1)
    costs = np.array([EQUITY_COST] + [ETF_COST] * etf_returns.shape[1]) * cost_mult
    ends = rebalance_ends(sessions, rebalance)
    if drawdown_check not in ('rebalance', 'daily'):
        raise ValueError(f'unknown_drawdown_check:{drawdown_check}')
    drawdown_days = set(sessions) if drawdown_check == 'daily' else ends
    target_fn = target_fn or target_weights
    sleeves = target_fn(returns.iloc[:0, 1:], equity_share)
    nav, peak, stressed = 1.0, 1.0, False
    values, weights, pending, triggers, pauses = [], [], None, 0, 0
    trigger_dates, recovery_dates = [], []
    names = list(returns.columns)
    for t, day in enumerate(sessions):
        if pending is not None:
            nav *= 1 - (np.abs(pending - sleeves) * costs).sum()
            sleeves, pending = pending, None
        if t > 0:
            growth = sleeves * (1 + returns.iloc[t].to_numpy())
            nav *= growth.sum()
            sleeves = growth / growth.sum()
        peak = max(peak, nav)
        values.append(nav)
        weights.append(sleeves.copy())
        changed = False
        if day in drawdown_days:
            drawdown = 1 - nav / peak
            if not stressed and drawdown >= DISASTER_DD:
                stressed, triggers, changed = True, triggers + 1, True
                trigger_dates.append(str(pd.Timestamp(day).date()))
            elif stressed and drawdown <= RECOVER_DD:
                stressed, changed = False, True
                recovery_dates.append(str(pd.Timestamp(day).date()))
        if day in ends or changed:
            share = DISASTER_EQUITY if stressed else equity_share
            pending = target_fn(returns.iloc[:t + 1, 1:], share)
            if premium is not None and day in premium.index:
                pending, paused = pause_qdii(pending, sleeves, names, premium.loc[day].to_dict(), overflow=overflow)
                pauses += len(paused)
    trace = dict(mean_weights=dict(zip(returns.columns, np.mean(weights, axis=0).round(4))), disaster_triggers=triggers,
                 trigger_dates=trigger_dates, recovery_dates=recovery_dates,
                 qdii_pauses=pauses, rebalance=rebalance, drawdown_check=drawdown_check,
                 weights=pd.DataFrame(weights, index=sessions, columns=names))
    return pd.Series(values, index=sessions), trace


def stats(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return dict(cagr=float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1), mdd=float((nav / nav.cummax() - 1).min()))
