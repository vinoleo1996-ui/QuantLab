"""Point-in-time universe and causal feature prices."""
import weakref

import numpy as np
import pandas as pd

from quantlab.data.contracts import asof_records, decision_time, timestamp, timestamps_series
from quantlab.data.industry import resolve_industries


def panels_for_research(dataset):
    dates = dataset.calendar[dataset.calendar <= pd.Timestamp(dataset.manifest["as_of"]).tz_localize(None).normalize()]
    close, amount, turnover = {}, {}, {}
    cutoffs = pd.Series([decision_time(d) for d in dates], index=dates)
    action_groups = {}
    if not dataset.actions.empty:
        actions = dataset.actions.copy()
        actions['_ex'] = pd.to_datetime(actions.ex_date).dt.normalize()
        actions['_available'] = actions.available_at.map(timestamp)
        action_groups = dict(tuple(actions.groupby(['instrument_id', '_ex'])))
    for c, panel in dataset.panels.items():
        p = panel.reindex(dates).copy()
        available = timestamps_series(p.available_at)
        known = available.notna() & available.le(cutoffs) & p.quality_status.eq('ok')
        raw = p.close.where(known)
        if dataset.manifest.get('feature_price_basis') == 'historical_adjustment_snapshot' and 'feature_close' in p:
            # Explicit feature-only snapshot. The account continues to use raw
            # prices and separately observed distributions; no synthetic cash.
            idx = p.feature_close.where(known)
            initial = idx.first_valid_index()
            if initial is not None: idx = idx / idx.loc[initial] * 100.
            close[c] = idx
            amount[c] = p.get('amount', pd.Series(np.nan, index=dates)).where(known)
            turnover[c] = p.get('turnover', pd.Series(np.nan, index=dates)).where(known)
            continue
        # Forward total-return index: actions only affect ex-date and subsequent observations.
        idx = pd.Series(np.nan, index=dates)
        prev = value = None
        pending_cash, pending_ratio = 0., 1.
        for t, px in raw.items():
            acts = action_groups.get((c, t))
            if acts is not None:
                acts = acts[acts._available <= cutoffs.loc[t]].sort_values("_available").drop_duplicates("action_id",keep="last")
                for _, action in acts.iterrows():
                    pending_cash += pending_ratio * float(action.cash_per_share)
                    pending_ratio *= 1 + float(action.share_ratio)
            if not np.isfinite(px): continue
            value = 100. if prev is None else value * (px * pending_ratio + pending_cash) / prev
            idx.loc[t], prev = value, px
            pending_cash, pending_ratio = 0., 1.
        close[c] = idx
        for gap in dataset.manifest.get('data_gaps',[]):
            if gap.get('instrument_id')==c and gap.get('reason')!='delisting_settlement_unverified':
                d=pd.Timestamp(gap['date'])
                if d in idx.index:idx.loc[d]=np.nan
        amount[c] = p.get("amount", pd.Series(np.nan, index=dates)).where(known)
        turnover[c] = p.get("turnover", pd.Series(np.nan, index=dates)).where(known)
    return pd.DataFrame(close), pd.DataFrame(amount), pd.DataFrame(turnover)


def universe_at(dataset, date, amount, cfg):
    records = asof_records(dataset.instruments, decision_time(date))
    industries = resolve_industries(dataset, date)
    if not records.empty:
        records = records.copy()
        records['is_st'] = records['is_st'].astype(object)
        records['sector'] = industries.sector.astype(object).reindex(records.index).fillna('UNKNOWN') if not industries.empty else 'UNKNOWN'
        records['industry_temporal_basis'] = industries.temporal_basis.reindex(records.index).fillna('unknown') if not industries.empty else 'unknown'
    eligible = []
    recent_amount = amount.loc[:date].tail(20)
    adv_count, adv_mean = recent_amount.count(), recent_amount.mean()
    ages = getattr(dataset, '_observation_age_cache', None)
    if ages is None:
        ages = {}
        dataset._observation_age_cache = ages
    for c, r in records.iterrows():
        if c not in dataset.panels or r.get("quality_status") != "ok" or not bool(r.get("active", False)): continue
        panel = dataset.panels[c]
        state = r.get('is_st')
        if r.get('state_basis') == 'daily_observation_required':
            state = panel.at[date, 'is_st_at_open'] if date in panel.index and 'is_st_at_open' in panel else None
            records.loc[c, 'is_st'] = state
        if cfg.universe.a_exclude_st and (pd.isna(state) or bool(state)): continue
        listing = r.get('listing_date')
        if pd.notna(listing):
            sessions = ((dataset.calendar >= pd.Timestamp(listing)) & (dataset.calendar <= date)).sum()
        else:
            # The observed history proves a minimum age, never an exact IPO.
            start = r.get('observation_start')
            if pd.isna(start): continue
            observed = panel.close.notna().to_numpy()
            cached = ages.get(c)
            # Stress replays may replace panels, append/remove rows or mark
            # prices missing in place. A code-only cache would reuse stale ages.
            if (cached is None or cached[0]() is not panel or cached[1] is not panel.index
                    or not np.array_equal(cached[2], observed)):
                cached = (weakref.ref(panel), panel.index, observed.copy(), observed.cumsum(dtype=np.int32))
                ages[c] = cached
            position = panel.index.searchsorted(date,side='right')-1
            sessions = int(cached[3][position]) if position>=0 else 0
        if sessions < cfg.universe.a_min_listing_days: continue
        if adv_count.get(c,0)<20 or adv_mean.get(c,0)<cfg.universe.a_min_avg_amount_wan*1e4:continue
        if cfg.research.universe_mode == "pit_cap":
            cap = r.get("float_mktcap", np.nan)
            if not np.isfinite(cap) or cap < cfg.universe.a_min_mktcap_yi * 1e8: continue
        eligible.append(c)
    return eligible, records


def exposure_matrix(dataset, date, records, codes, *, as_of=None):
    """Latest complete exposure snapshot known at the caller's decision time.

    Each available_at group is a replacement snapshot, including revisions of the
    same effective date. Opening execution must pass its own information cutoff.
    """
    cutoff = timestamp(as_of) if as_of is not None else decision_time(date)
    industries = resolve_industries(dataset, date, as_of=cutoff)
    exposures = {}
    known = pd.DataFrame()
    if not dataset.exposures.empty:
        known = dataset.exposures.copy()
        known['_available'] = known.available_at.map(timestamp)
        known['_effective'] = known.effective_at.map(timestamp)
        known = known[(known._available <= cutoff) & (known._effective <= cutoff)]
        if not known.empty:
            latest = known.groupby("instrument_id")._effective.transform("max")
            known = known[known._effective == latest]
            revision = known.groupby("instrument_id")._available.transform("max")
            known = known[known._available == revision]
    for c in codes:
        r = records.loc[c] if c in records.index else {}
        if r.get("asset_type") == "equity_etf":
            rows = known[known.instrument_id == c] if not known.empty else pd.DataFrame()
            valid = not rows.empty and rows.weight.ge(0).all() and 0.99 <= rows.weight.sum() <= 1.01
            exposures[c] = rows.groupby("sector").weight.sum().to_dict() if valid else {"UNKNOWN_ETF": 1.0}
        else:
            sector = industries.loc[c, 'sector'] if c in industries.index else "UNKNOWN"
            exposures[c] = {str(sector) if pd.notna(sector) and str(sector) not in ("NA", "") else "UNKNOWN": 1.0}
    return pd.DataFrame(exposures).fillna(0.0)
