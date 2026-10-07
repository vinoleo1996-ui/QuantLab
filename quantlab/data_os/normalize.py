"""把各供应商的行收成同一套银层列。公布时间缺失就保持缺失。"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from quantlab.data_os.pit import publication_available, session_available
from quantlab.data_os.quality import annotate_fact, annotate_market
from quantlab.data_os.schema import (
    ACTION_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    MACRO_COLUMNS,
    MARKET_COLUMNS,
)


def _blank(columns: list[str]) -> dict:
    return {column: pd.NA for column in columns}


def market_frame(rows: list[dict], version: str, ingested_at: datetime | None = None) -> pd.DataFrame:
    ingested = ingested_at or datetime.now(timezone.utc)
    built = []
    for row in rows:
        item = _blank(MARKET_COLUMNS)
        item.update(row)
        item["dataset_version"] = version
        item["ingested_at"] = pd.Timestamp(ingested)
        item["price_basis"] = row.get("price_basis") or "raw"
        item["volume_unit"] = "share"
        amount, volume = item.get("amount"), item.get("volume")
        if pd.notna(amount) and pd.notna(volume) and float(volume) > 0:
            item["vwap"] = float(amount) / float(volume)
        item["available_time"] = session_available(item["trade_date"], row["market"])
        item["availability_rule"] = "session_close"
        item["source_timestamp"] = item["available_time"]
        item["pit_valid"] = True
        item["corporate_action_valid"] = bool(row.get("corporate_action_valid", False))
        item["is_trading"] = bool(row.get("is_trading", True))
        item["is_suspended"] = bool(row.get("is_suspended", False))
        built.append(item)
    frame = pd.DataFrame(built)
    if frame.empty:
        return frame
    return annotate_market(frame)


def fundamental_frame(rows: list[dict], version: str, timezone: str) -> pd.DataFrame:
    built = []
    for row in rows:
        item = _blank(FUNDAMENTAL_COLUMNS)
        item.update({key: row.get(key, pd.NA) for key in FUNDAMENTAL_COLUMNS})
        available, rule, pit = publication_available(row.get("filed_date"), row.get("accepted_datetime"), timezone)
        item["available_time"] = available
        item["availability_rule"] = rule
        item["pit_valid"] = pit
        item["dataset_version"] = version
        item["value"] = row.get("value")
        item["source"] = row.get("source")
        built.append(item)
    frame = pd.DataFrame(built)
    if frame.empty:
        return frame
    return annotate_fact(frame, ["concept", "value", "period_end", "source"])


SAME_DAY_SERIES = {"DGS10", "DGS2", "VIXCLS"}


def macro_frame(rows: list[dict], version: str) -> pd.DataFrame:
    built = []
    for row in rows:
        item = _blank(MACRO_COLUMNS)
        item.update(row)
        item["dataset_version"] = version
        release = row.get("release_date")
        same_day = (release is None or pd.isna(release) or not str(release).strip()) and row.get("series_id") in SAME_DAY_SERIES
        if same_day:
            release = row.get("observation_date")
            item["release_date"] = release
        if release is not None and not pd.isna(release) and str(release).strip():
            available, rule, pit = publication_available(release, row.get("available_time_exact"), "America/New_York")
            item["available_time"] = available
            item["availability_rule"] = "observation_is_print" if same_day else rule
            item["pit_valid"] = pit
        else:
            item["available_time"] = pd.NaT
            item["availability_rule"] = "missing"
            item["pit_valid"] = False
        built.append(item)
    frame = pd.DataFrame(built)
    if frame.empty:
        return frame
    frame["_release_sort"] = pd.to_datetime(frame["release_date"], errors="coerce")
    frame = frame.sort_values(
        ["series_id", "observation_date", "_release_sort"], na_position="first"
    ).drop_duplicates(["series_id", "observation_date"], keep="last")
    return annotate_fact(frame.drop(columns=["_release_sort"]), ["series_id", "observation_date", "value", "source"])


def apply_adjust_factors(rows: list[dict], actions: list[dict]) -> list[dict]:
    """复权因子另存。收盘价保持原始价，不在这里乘进去。"""
    if not rows:
        return rows
    bars = pd.DataFrame(rows)
    bars["price_basis"] = "raw"
    bars["trade_date"] = pd.to_datetime(bars["trade_date"])
    factors = [
        action for action in actions
        if action.get("event_type") == "adjust_factor" and pd.notna(action.get("adjust_factor"))
    ]
    if not factors:
        bars["trade_date"] = bars["trade_date"].dt.strftime("%Y-%m-%d")
        return bars.to_dict("records")
    table = pd.DataFrame(factors)
    table["event_date"] = pd.to_datetime(table["event_date"])
    table["adjust_factor"] = pd.to_numeric(table["adjust_factor"], errors="coerce")
    parts = []
    for instrument_id, group in bars.groupby("instrument_id", sort=False):
        group = group.sort_values("trade_date")
        own = table[table["instrument_id"] == instrument_id].sort_values("event_date")
        if own.empty:
            parts.append(group)
            continue
        merged = pd.merge_asof(
            group,
            own[["event_date", "adjust_factor"]],
            left_on="trade_date",
            right_on="event_date",
            direction="backward",
        )
        if "adjust_factor_x" in merged.columns:
            merged["adjust_factor"] = merged["adjust_factor_y"].combine_first(merged["adjust_factor_x"])
            merged = merged.drop(columns=["adjust_factor_x", "adjust_factor_y"])
        parts.append(merged.drop(columns=["event_date"], errors="ignore"))
    out = pd.concat(parts, ignore_index=True)
    if "adjust_factor" in out.columns:
        out["adj_factor"] = out["adjust_factor"]
        out = out.drop(columns=["adjust_factor"])
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    return out.to_dict("records")


def action_frame(rows: list[dict], version: str, timezone: str) -> pd.DataFrame:
    built = []
    for row in rows:
        item = {column: row.get(column, pd.NA) for column in ACTION_COLUMNS}
        item["dataset_version"] = version
        if pd.isna(item.get("available_time")):
            available, _, pit = publication_available(row.get("announcement_date"), None, timezone)
            item["available_time"] = available if pit else pd.NaT
        built.append(item)
    return pd.DataFrame(built)
