"""Industry identity with an explicit separation of publication and effective dates."""
from __future__ import annotations

import pandas as pd

from quantlab.data.contracts import asof_records, decision_time, timestamp, timestamps_series

INDUSTRY_MODES = ("verified_pit", "effective_date_proxy", "disabled")


def resolve_industries(dataset, date, mode=None, *, as_of=None):
    """Return the latest industry and its evidence; never invent an available_at.

    Legacy datasets already store sector in their dated instruments table. Their
    existing contract remains supported. New proxy records must opt in through
    the manifest and cannot enter the verified-PIT path.
    """
    mode = mode or dataset.manifest.get("industry_mode", "verified_pit")
    if mode not in INDUSTRY_MODES:
        raise ValueError("unknown_industry_mode")
    cutoff = timestamp(as_of) if as_of is not None else decision_time(date)
    columns = ["sector", "temporal_basis", "age_days", "available_at", "effective_at"]
    if mode == "disabled":
        return pd.DataFrame(columns=columns)
    history = getattr(dataset, "industry_records", pd.DataFrame())
    if history.empty:
        # Existing manifests carry PIT sector history in their security table.
        rows = asof_records(dataset.instruments, cutoff)
        if rows.empty or "sector" not in rows:
            return pd.DataFrame(columns=columns)
        out = rows[["sector", "available_at", "effective_at"]].copy()
        out["temporal_basis"] = "instrument_history"
    else:
        required = {"instrument_id", "sector", "effective_at", "available_at", "source", "revision"}
        if not required.issubset(history):
            raise ValueError("industry_contract_missing:" + ",".join(sorted(required-set(history))))
        rows = history.copy()
        effective = timestamps_series(rows.effective_at).dt.tz_convert('UTC')
        rows['_effective'] = effective
        rows = rows.loc[effective.notna() & effective.le(cutoff.tz_convert("UTC"))].copy()
        if mode == "verified_pit":
            available = timestamps_series(rows.available_at).dt.tz_convert('UTC')
            rows['_available'] = available
            verified = rows.get("temporal_basis", pd.Series("unknown", index=rows.index)).eq("verified_pit")
            rows = rows.loc[verified & available.notna() & available.le(cutoff.tz_convert("UTC"))]
        if rows.empty:
            return pd.DataFrame(columns=columns)
        # Publication order resolves genuine revisions. In proxy mode a source
        # revision has no asserted historical arrival order; duplicates conflict.
        keys = ["instrument_id", "effective_at"]
        if mode == "effective_date_proxy" and rows.groupby(keys).sector.nunique().gt(1).any():
            raise ValueError("ambiguous_proxy_industry_revision")
        if '_available' not in rows:
            rows['_available'] = timestamps_series(rows.available_at).dt.tz_convert('UTC')
        if mode=='verified_pit' and rows.duplicated(['instrument_id','_effective','_available']).any():
            raise ValueError('ambiguous_verified_industry_revision')
        rows = rows.sort_values(["_effective", "_available", "revision"], na_position="first")
        out = rows.drop_duplicates("instrument_id", keep="last").set_index("instrument_id")
        out = out[["sector", "available_at", "effective_at"]].copy()
        out["temporal_basis"] = "verified_pit" if mode == "verified_pit" else "effective_only"
    out["age_days"] = (cutoff-timestamps_series(out.effective_at)).dt.days
    # Parquet dictionary columns may not contain an UNKNOWN category. Evidence
    # resolution must not fail merely because a requested member lacks history.
    out["sector"] = out.sector.astype(object).replace({"": "UNKNOWN", "NA": "UNKNOWN"}).fillna("UNKNOWN")
    return out[columns]
