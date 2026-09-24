"""Point-in-time universe and causal feature prices."""
import numpy as np
import pandas as pd

from quantlab.data.contracts import asof_records, decision_time, timestamp


def panels_for_research(dataset):
    dates = dataset.calendar[dataset.calendar <= pd.Timestamp(dataset.manifest["as_of"]).tz_localize(None).normalize()]
    close, amount, turnover = {}, {}, {}
    for c, panel in dataset.panels.items():
        p = panel.reindex(dates).copy()
        known = pd.Series([pd.notna(row.get("available_at")) and timestamp(row.available_at) <= decision_time(t)
                           and row.get("quality_status") == "ok" for t, row in p.iterrows()], index=dates)
        raw = p.close.where(known)
        # Forward total-return index: actions only affect ex-date and subsequent observations.
        idx = pd.Series(np.nan, index=dates)
        prev = value = None
        pending_cash, pending_ratio = 0., 1.
        for t, px in raw.items():
            if not dataset.actions.empty:
                acts = dataset.actions[(dataset.actions.instrument_id == c) &
                                       (pd.to_datetime(dataset.actions.ex_date).dt.normalize() == t)]
                acts = acts[acts.available_at.map(timestamp) <= decision_time(t)].sort_values("available_at").drop_duplicates("action_id",keep="last")
                for _, action in acts.iterrows():
                    if timestamp(action.available_at) <= decision_time(t):
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
    eligible = []
    for c, r in records.iterrows():
        if c not in dataset.panels or r.get("quality_status") != "ok" or not bool(r.get("active", False)): continue
        if cfg.universe.a_exclude_st and bool(r.get("is_st", True)): continue
        if pd.isna(r.get("listing_date")): continue
        sessions = ((dataset.calendar >= pd.Timestamp(r.listing_date)) & (dataset.calendar <= date)).sum()
        if sessions < cfg.universe.a_min_listing_days: continue
        adv = amount.loc[:date, c].tail(20)
        if len(adv.dropna()) < 20 or adv.mean() < cfg.universe.a_min_avg_amount_wan * 1e4: continue
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
            sector = r.get("sector", "UNKNOWN")
            exposures[c] = {str(sector) if pd.notna(sector) and str(sector) not in ("NA", "") else "UNKNOWN": 1.0}
    return pd.DataFrame(exposures).fillna(0.0)
