"""Pre-registered layer inputs: PIT financial panels, screens, crowding, PEAD events.

Rules are documented in ``output/smallcap_rebuild_20261006/PREREGISTRATION.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVENTS_RAW = ROOT / 'data/smallcap_events/raw'
POSITIVE_FORECAST = {'预增', '略增', '扭亏', '续盈'}
PEAD_LIFE = 60


def asof_metric(financials, metric, sessions, codes, annual_only=False):
    """Wide date x code panel of the latest report-period value usable on each session."""
    rows = financials[financials.metric == metric].dropna(subset=['available_date', 'value']).copy()
    rows['report_period'] = pd.to_datetime(rows['report_period'])
    if annual_only:
        rows = rows[rows.report_period.dt.month == 12]
    rows['available_date'] = pd.to_datetime(rows['available_date'])
    rows = rows.sort_values(['instrument_id', 'available_date', 'report_period'])
    newest = rows.groupby('instrument_id').report_period.cummax()
    rows = rows[rows.report_period >= newest]
    wide = rows.pivot_table(index='available_date', columns='instrument_id', values='value', aggfunc='last')
    wide = wide.reindex(wide.index.union(sessions)).sort_index().ffill()
    return wide.reindex(index=sessions, columns=codes)


def screen_flags(panel, financials, sessions):
    """True where the pre-registered delisting-risk screen removes the name."""
    codes = panel.codes
    profit = asof_metric(financials, 'net_profit', sessions, codes, annual_only=True)
    revenue = asof_metric(financials, 'revenue', sessions, codes, annual_only=True)
    leverage = asof_metric(financials, 'debt_to_assets', sessions, codes)
    cap20 = panel.field('float_mktcap').rolling(20, min_periods=10).mean().reindex(sessions)
    st = panel.field('is_st').reindex(sessions)
    loss_small = (profit < 0) & (revenue < 3e8)
    flags = (st.fillna(0) > 0) | loss_small | (cap20 < 8e8) | (leverage >= 100)
    return flags, {'st': st.fillna(0) > 0, 'loss_small_revenue': loss_small, 'cap_below_800m': cap20 < 8e8,
                   'negative_equity': leverage >= 100}


def _read_events(kind):
    path = EVENTS_RAW / f'{kind}.jsonl'
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows += record['rows']
    return pd.DataFrame(rows)


def earnings_events(financials):
    """One row per (code, report period): earliest disclosure date and its direction."""
    parts = []
    forecast = _read_events('forecast')
    if len(forecast):
        f = pd.DataFrame({
            'code': forecast['code'],
            'period': pd.to_datetime(forecast['profitForcastExpStatDate'], errors='coerce'),
            'date': pd.to_datetime(forecast['profitForcastExpPubDate'], errors='coerce'),
            'positive': forecast['profitForcastType'].isin(POSITIVE_FORECAST),
            'source': 'forecast'})
        parts.append(f)
    express = _read_events('express')
    if len(express):
        growth = pd.to_numeric(express['performanceExpressOPYOY'], errors='coerce')
        e = pd.DataFrame({
            'code': express['code'],
            'period': pd.to_datetime(express['performanceExpStatDate'], errors='coerce'),
            'date': pd.to_datetime(express['performanceExpPubDate'], errors='coerce'),
            'positive': growth > 0,
            'source': 'express'})
        parts.append(e)
    profit = financials[financials.metric == 'net_profit'].copy()
    profit['report_period'] = pd.to_datetime(profit['report_period'])
    profit['announced_on'] = pd.to_datetime(profit['announced_on'])
    profit = profit.sort_values(['instrument_id', 'report_period'])
    prior = profit.set_index(['instrument_id', 'report_period']).value
    key = list(zip(profit.instrument_id, profit.report_period - pd.DateOffset(years=1)))
    last_year = pd.Series([prior.get(k, np.nan) for k in key], index=profit.index)
    formal = pd.DataFrame({'code': profit.instrument_id, 'period': profit.report_period,
                           'date': profit.announced_on, 'positive': profit.value > last_year,
                           'source': 'formal'})
    parts.append(formal)
    events = pd.concat(parts, ignore_index=True).dropna(subset=['code', 'period', 'date'])
    events = events.sort_values(['code', 'period', 'date'])
    return events.groupby(['code', 'period'], as_index=False).first()


def pead_flags(panel, events, sessions, pool_median):
    """True on sessions where a positive event's reaction-confirmed signal is still live.

    The reaction uses the close of the first session after the announcement, so the
    flag starts at that close and the earliest trade is the following open.
    """
    close = panel.field('adj_close').reindex(sessions)
    codes = list(close.columns)
    index = {c: i for i, c in enumerate(codes)}
    flags = np.zeros(close.shape, dtype=bool)
    values = close.to_numpy()
    position = sessions.searchsorted(pd.to_datetime(events['date']).to_numpy(), side='left')
    for row, start in zip(events.itertuples(index=False), position):
        j = index.get(row.code)
        if j is None or start + 1 >= len(sessions):
            continue
        # announcement-day close is the base; first session strictly after is the reaction day
        base_t = start if sessions[start] == pd.Timestamp(row.date) else start - 1
        react_t = base_t + 1
        if base_t < 0 or react_t >= len(sessions):
            continue
        a, b = values[base_t, j], values[react_t, j]
        if not (np.isfinite(a) and np.isfinite(b) and a > 0):
            continue
        reaction = b / a - 1 - pool_median.iloc[react_t]
        if row.positive and reaction > 0:
            flags[react_t:min(react_t + PEAD_LIFE, len(sessions)), j] = True
    return pd.DataFrame(flags, index=sessions, columns=codes)


def crowding_score(panel, sessions, pool_members):
    """Mean expanding percentile of three crowding measures; NaN until 250 observations."""
    amount = panel.field('amount').reindex(sessions)
    cap = panel.field('float_mktcap').reindex(sessions)
    returns = panel.field('adj_close').reindex(sessions).pct_change(fill_method=None)
    share, ratio, corr = [], [], []
    current = None
    for day in sessions:
        current = pool_members.get(day, current)
        if not current:
            share.append(np.nan); ratio.append(np.nan); corr.append(np.nan)
            continue
        row = amount.loc[day]
        total = row.sum()
        pool_amount = row.reindex(current).sum()
        top = cap.loc[day].dropna().nlargest(300).index
        share.append(pool_amount / total if total > 0 else np.nan)
        big = row.reindex(top).sum()
        ratio.append(pool_amount / big if big > 0 else np.nan)
        corr.append(np.nan)
    frame = pd.DataFrame({'share': share, 'ratio': ratio}, index=sessions)
    # average correlation proxy over 20 sessions: var(equal-weight) / mean(var individual)
    members_returns = []
    current = None
    for day in sessions:
        current = pool_members.get(day, current)
        members_returns.append(returns.loc[day].reindex(current).mean() if current else np.nan)
    ew = pd.Series(members_returns, index=sessions)
    individual_var = []
    current = None
    window = returns.rolling(20, min_periods=15).var()
    for day in sessions:
        current = pool_members.get(day, current)
        individual_var.append(window.loc[day].reindex(current).mean() if current else np.nan)
    frame['corr'] = ew.rolling(20, min_periods=15).var() / pd.Series(individual_var, index=sessions)

    def expanding_pct(series):
        values = series.to_numpy()
        out = np.full(len(values), np.nan)
        seen = []
        for t, v in enumerate(values):
            if np.isfinite(v):
                if len(seen) >= 250:
                    out[t] = np.mean(np.asarray(seen) <= v)
                seen.append(v)
        return pd.Series(out, index=series.index)

    pct = frame.apply(expanding_pct)
    return pct.mean(axis=1, skipna=False), frame


def crowding_states(score, high=0.90, low=0.50):
    """1.0 normal, 0.5 de-risked; hysteresis between ``high`` and ``low``."""
    state, out = 1.0, []
    for value in score:
        if np.isfinite(value):
            if state == 1.0 and value >= high:
                state = 0.5
            elif state == 0.5 and value <= low:
                state = 1.0
        out.append(state)
    return pd.Series(out, index=score.index)
