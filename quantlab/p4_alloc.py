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


def target_weights(etf_returns_upto, equity_share):
    vol = etf_returns_upto.tail(VOL_WINDOW).std() if len(etf_returns_upto) >= VOL_WINDOW else None
    names = list(etf_returns_upto.columns)
    if vol is None or not np.isfinite(vol.to_numpy()).all() or (vol <= 0).any():
        etf = np.full(len(names), 1 / len(names))
    else:
        inverse = 1 / vol.to_numpy()
        etf = inverse / inverse.sum()
    return np.concatenate([[equity_share], (1 - equity_share) * etf])


def allocate(equity_nav, etf_returns, equity_share=0.60, cost_mult=1.0):
    """Daily NAV (starting at 1) and a trace. ``etf_returns`` columns are the ETF sleeves, same index."""
    sessions = equity_nav.index
    eq = equity_nav.pct_change(fill_method=None).fillna(0.0)
    returns = pd.concat([eq.rename('equity'), etf_returns.reindex(sessions).fillna(0.0)], axis=1)
    costs = np.array([EQUITY_COST] + [ETF_COST] * etf_returns.shape[1]) * cost_mult
    ends = month_ends(sessions)
    sleeves = target_weights(returns.iloc[:0, 1:], equity_share)
    nav, peak, stressed = 1.0, 1.0, False
    values, weights, pending, triggers = [], [], None, 0
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
        if day in ends:
            drawdown = 1 - nav / peak
            if not stressed and drawdown >= DISASTER_DD:
                stressed, triggers = True, triggers + 1
            elif stressed and drawdown <= RECOVER_DD:
                stressed = False
            share = DISASTER_EQUITY if stressed else equity_share
            pending = target_weights(returns.iloc[:t + 1, 1:], share)
    trace = dict(mean_weights=dict(zip(returns.columns, np.mean(weights, axis=0).round(4))), disaster_triggers=triggers)
    return pd.Series(values, index=sessions), trace


def stats(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return dict(cagr=float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1), mdd=float((nav / nav.cummax() - 1).min()))
