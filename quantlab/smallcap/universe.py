"""Decision-date candidate pools and basket selection for the small-cap layers.

Everything a decision uses is known at that session's close:
- liquidity history: the last 60 positive-amount sessions and at least 253 of them;
- float cap: 20-session mean of ``float_mktcap`` ending that day;
- financials: rows whose ``available_date`` is on or before that day;
- industry: last record with ``effective_at`` on or before that day.

Pool rules (fixed before running):
- ``adv``: rank by 60-session mean amount, descending, ranks 3001-5000, mean amount >= 5m.
- ``cap``: same eligibility (253 history, mean amount >= 5m), rank by 20-session
  mean float cap, descending, ranks 3001-5000. Names without float cap are not
  ranked (all Beijing-exchange names, which have no turnover upstream).
Selection: financial tier (latest net profit >= 0 and 0 < PB <= 3 scores 1),
seeded random order inside a tier, current holdings kept while inside the top
150, at most 15 names per industry, 50 names, 90 percent gross equal weight.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
import pandas as pd

RANK_START, RANK_END = 3001, 5000
MIN_ADV = 5_000_000.0
MIN_HISTORY = 253
ADV_WINDOW = 60
CAP_WINDOW = 20
HOLDINGS, BUFFER, INDUSTRY_SLOTS, GROSS = 50, 100, 15, 0.90


def decision_grid(sessions, start, end, phase=0, step=20):
    sessions = pd.DatetimeIndex(sessions)
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    return window[phase::step]


def rolling_features(panel):
    """Positive-amount ADV (last 60 positive sessions), history count, 20d mean float cap."""
    amount = panel.field('amount')
    positive = amount.where(amount > 0)
    history = positive.notna().cumsum()
    # mean of the last 60 positive observations: rolling over non-null values per column
    adv = positive.apply(lambda col: col.dropna().rolling(ADV_WINDOW, min_periods=ADV_WINDOW).mean()
                         .reindex(col.index).ffill().where(col.notna()))
    cap = panel.field('float_mktcap').rolling(CAP_WINDOW, min_periods=10).mean()
    return {'adv': adv, 'history': history, 'cap': cap}


def pool_on(day, features, kind):
    adv = features['adv'].loc[day]
    history = features['history'].loc[day]
    eligible = adv[(history >= MIN_HISTORY) & adv.notna()]
    if kind == 'adv':
        ordered = eligible.sort_values(ascending=False, kind='mergesort')
    elif kind == 'cap':
        cap = features['cap'].loc[day].reindex(eligible.index).dropna()
        ordered = cap.sort_values(ascending=False, kind='mergesort')
    else:
        raise ValueError(kind)
    if len(ordered) < RANK_START:
        return []
    band = ordered.iloc[RANK_START - 1:RANK_END]
    return [code for code in band.index if adv[code] >= MIN_ADV]


class Financials:
    """As-of lookups on the v4 financial table."""

    def __init__(self, table):
        table = table.dropna(subset=['available_date']).copy()
        table['available_date'] = pd.to_datetime(table['available_date'])
        table['report_period'] = pd.to_datetime(table['report_period'])
        self.table = table.sort_values(['available_date', 'report_period'])

    def latest(self, metric, day, annual_only=False):
        rows = self.table[(self.table.metric == metric) & (self.table.available_date <= pd.Timestamp(day))]
        if annual_only:
            rows = rows[rows.report_period.dt.month == 12]
        rows = rows.sort_values(['instrument_id', 'report_period', 'available_date'])
        return rows.groupby('instrument_id').value.last()


class Industries:
    def __init__(self, table):
        table = table.copy()
        table['effective_at'] = pd.to_datetime(table['effective_at'])
        self.table = table.sort_values('effective_at')

    def on(self, day):
        rows = self.table[self.table.effective_at <= pd.Timestamp(day)]
        sector = rows.groupby('instrument_id').sector.last().astype(str)
        return sector.where(~sector.isin(['', 'NA', 'UNKNOWN', 'nan']), 'unclassified')


def financial_tier(codes, net_profit, pb):
    profit = net_profit.reindex(codes)
    book = pb.reindex(codes)
    return ((profit >= 0) & (book > 0) & (book <= 3)).astype(float)


def select(ordered, industries, current, holdings=HOLDINGS, buffer=BUFFER, slots=INDUSTRY_SLOTS,
           retain_order=None):
    held = set(current)
    retain_order = ordered if retain_order is None else retain_order
    retained = [code for code in retain_order[:holdings + buffer] if code in held]
    chosen, counts = [], Counter()
    for code in retained + list(ordered):
        industry = industries.get(code, 'unclassified')
        if code in chosen or counts[industry] >= slots:
            continue
        chosen.append(code)
        counts[industry] += 1
        if len(chosen) == holdings:
            break
    return chosen


def priority(seed, code):
    """Uniform [0, 1) tie-break fixed per (seed, code) so it does not churn across dates."""
    digest = hashlib.sha256(f'{seed}:{code}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2 ** 64


def seeded_order(codes, keys, seed):
    """Sort by the tuple of key arrays (descending), ties broken by a seeded per-code draw."""
    draw = np.array([priority(seed, code) for code in codes])
    columns = [np.asarray(k, dtype=float) for k in keys] + [draw]
    order = np.lexsort([-c for c in reversed(columns)])
    return [codes[i] for i in order]


def build_schedule(snapshots, seed, gross=GROSS, key_names=('tier',), retain_key_names=None, holdings=HOLDINGS):
    """``snapshots``: list of (day, codes, {key: array}, industries dict). Returns schedule and baskets.

    ``retain_key_names`` orders the retention check separately, so an entry-only key
    does not change which held names survive.
    """
    basket, schedule = [], {}
    for day, codes, keys, industries in snapshots:
        if not codes:
            basket = []
            schedule[day] = {}
            continue
        ordered = seeded_order(codes, [keys[name] for name in key_names], seed)
        retain = (None if retain_key_names is None
                  else seeded_order(codes, [keys[name] for name in retain_key_names], seed))
        basket = select(ordered, industries, basket, holdings=holdings, retain_order=retain)
        schedule[day] = {code: gross / len(basket) for code in basket} if basket else {}
    return schedule
