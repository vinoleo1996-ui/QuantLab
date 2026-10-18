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
150, one industry at most 25 percent of the book (``industry_slots``), 50 names, 90 percent gross.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np
import pandas as pd

RANK_START, RANK_END = 3001, 5000
MIN_ADV = 5_000_000.0
MIN_HISTORY = 253
ADV_WINDOW = 60
CAP_WINDOW = 20
HOLDINGS, BUFFER, GROSS = 50, 100, 0.90
# ``p4_mandate.max_industry_weight_in_equity``: a fixed name count is only correct at one book size,
# so the per-industry cap is derived from the mandate weight instead of hard-coded.
INDUSTRY_MAX_WEIGHT = 0.25


def industry_slots(holdings=HOLDINGS, max_weight=INDUSTRY_MAX_WEIGHT):
    """Equal-weight name cap that keeps one industry at or below ``max_weight`` of the book."""
    return max(1, int(math.floor(max_weight * holdings + 1e-9)))


def decision_grid(sessions, start, end, phase=0, step=20):
    sessions = pd.DatetimeIndex(sessions)
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    return window[phase::step]


def rolling_features(panel, with_st=False):
    """Positive-amount ADV (last 60 positive sessions), history count, 20d mean float cap."""
    amount = panel.field('amount')
    positive = amount.where(amount > 0)
    history = positive.notna().cumsum()
    # mean of the last 60 positive observations: rolling over non-null values per column
    adv = positive.apply(lambda col: col.dropna().rolling(ADV_WINDOW, min_periods=ADV_WINDOW).mean()
                         .reindex(col.index).ffill().where(col.notna()))
    cap = panel.field('float_mktcap').rolling(CAP_WINDOW, min_periods=10).mean()
    features = {'adv': adv, 'history': history, 'cap': cap}
    if with_st:
        features['st'] = panel.field('is_st')
    return features


def pool_on(day, features, kind, band='absolute', exclude_st=False, min_history=MIN_HISTORY):
    """``band='relative'`` keeps ranks above ceil(0.6 N), the bottom 40 percent of eligible names.

    ``exclude_st`` drops names flagged ST that day before ranking (needs ``features['st']``).
    """
    adv = features['adv'].loc[day]
    history = features['history'].loc[day]
    keep = (history >= min_history) & adv.notna()
    if exclude_st:
        keep &= ~features['st'].loc[day].reindex(adv.index).eq(1)
    # Beijing names have no upstream turnover and were only ever excluded at call sites.
    keep &= ~pd.Index(adv.index.map(str)).str.startswith('bj.')
    eligible = adv[keep]
    if kind == 'adv':
        ordered = eligible.sort_values(ascending=False, kind='mergesort')
    elif kind == 'cap':
        cap = features['cap'].loc[day].reindex(eligible.index).dropna()
        ordered = cap.sort_values(ascending=False, kind='mergesort')
    else:
        raise ValueError(kind)
    if band == 'relative':
        selected = ordered.iloc[math.ceil(0.6 * len(ordered)):]
    elif band == 'absolute':
        if len(ordered) < RANK_START:
            return []
        selected = ordered.iloc[RANK_START - 1:RANK_END]
    else:
        raise ValueError(band)
    return [code for code in selected.index if adv[code] >= MIN_ADV]


# Frozen cuts for the domain trial: shares of the tradable list, most liquid first.
DOMAIN_LIQUID_END, DOMAIN_MID_END = 0.40, 0.70
DOMAINS = ('liquid', 'mid', 'micro')


def domains_on(day, features, kind='adv', exclude_st=False, min_history=MIN_HISTORY, exclude_prefix='bj.'):
    """Split the tradable list into three ADV domains. Ineligible names are removed before the cut.

    ``liquid`` is the most liquid ``ceil(0.40 N)`` names, ``mid`` runs through ``ceil(0.70 N)``,
    and ``micro`` is the remainder. This is not the 3001-5000 band and not the bottom 40 percent.
    """
    if kind != 'adv':
        raise ValueError(kind)
    adv = features['adv'].loc[day]
    history = features['history'].loc[day]
    keep = (history >= min_history) & adv.notna() & (adv >= MIN_ADV)
    if exclude_st:
        keep &= ~features['st'].loc[day].reindex(adv.index).eq(1)
    ordered = adv[keep].sort_values(ascending=False, kind='mergesort')
    if exclude_prefix:
        ordered = ordered[~pd.Index(ordered.index.map(str)).str.startswith(exclude_prefix)]
    empty = {name: [] for name in DOMAINS}
    n = len(ordered)
    if n == 0:
        return empty
    liquid_end = math.ceil(DOMAIN_LIQUID_END * n)
    mid_end = math.ceil(DOMAIN_MID_END * n)
    codes = list(ordered.index)
    return {'liquid': codes[:liquid_end], 'mid': codes[liquid_end:mid_end], 'micro': codes[mid_end:]}


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


def select(ordered, industries, current, holdings=HOLDINGS, buffer=BUFFER, slots=None,
           retain_order=None):
    slots = industry_slots(holdings) if slots is None else slots
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
