"""Point-in-time large/mid-cap universe and dividend yield on the v4 panel.

The pool on a decision session uses only that session's close: float market cap rank among
Shanghai/Shenzhen stocks that traded that day, are not ST, and have at least ``min_history``
traded sessions. Dividend yield counts cash distributions whose ex date falls in the trailing
365 calendar days (an ex date is always after its announcement), adjusted for later share
distributions, over the raw close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_ends(sessions, start, end):
    sessions = pd.DatetimeIndex(sessions)
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    return pd.DatetimeIndex(pd.Series(window, index=window).groupby(window.to_period('M')).max().values)


def eligible(panel, min_history=252):
    traded = (panel.field('tradestatus') == 1) & (panel.field('volume') > 0)
    history = traded.cumsum()
    st = panel.field('is_st').fillna(1) > 0
    exchange = np.array([not code.startswith('bj.') for code in traded.columns])
    return traded & (history >= min_history) & ~st & exchange


def top_pool(panel, days, size=1500, min_history=252):
    """{session: list of codes}: the ``size`` largest eligible names by float market cap."""
    ok = eligible(panel, min_history)
    cap = panel.field('float_mktcap')
    pools = {}
    for day in days:
        values = cap.loc[day].where(ok.loc[day]).dropna()
        pools[pd.Timestamp(day)] = values.nlargest(size).index.tolist()
    return pools


def dividend_yield(actions, close, days):
    """Trailing-365-day cash per current share over the raw close; zero for names without cash."""
    actions = pd.DataFrame({'instrument_id': actions.instrument_id.astype(str),
                            'ex_date': pd.to_datetime(actions.ex_date.astype(str)),
                            'cash_per_share': pd.to_numeric(actions.cash_per_share).fillna(0.0).astype(float),
                            'share_ratio': pd.to_numeric(actions.share_ratio).fillna(0.0).astype(float)})
    actions = actions[actions.instrument_id.isin(close.columns)].sort_values('ex_date')
    out = {}
    for day in days:
        day = pd.Timestamp(day)
        recent = actions[(actions.ex_date > day - pd.Timedelta(days=365)) & (actions.ex_date <= day)]
        cash = {}
        for code, group in recent.groupby('instrument_id', observed=True):
            dilution = np.cumprod((1.0 + group.share_ratio.values)[::-1])[::-1]
            cash[code] = float((group.cash_per_share.values / dilution).sum())
        price = close.loc[day]
        out[day] = (pd.Series(cash).reindex(price.index).fillna(0.0) / price).where(price > 0)
    return pd.DataFrame(out).T
