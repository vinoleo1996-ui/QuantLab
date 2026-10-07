"""时点：能用的是当时已经公布的版本，不是报告期本身。"""

from __future__ import annotations

import pandas as pd


UTC = "UTC"


def to_utc(value, timezone: str | None = None) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return pd.NaT
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        if timezone is None:
            raise ValueError(f"无时区时间不能当作公布时间: {value}")
        stamp = stamp.tz_localize(timezone)
    return stamp.tz_convert(UTC)


def session_available(trade_date, market: str) -> pd.Timestamp:
    day = pd.Timestamp(trade_date)
    if day.tzinfo is not None:
        day = day.tz_convert("UTC").tz_localize(None)
    day = day.normalize()
    if market == "CN":
        local = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        rule_tz = "Asia/Shanghai"
    elif market == "US":
        local = day.tz_localize("America/New_York") + pd.Timedelta(hours=16)
        rule_tz = "America/New_York"
    else:
        raise ValueError(f"未知市场: {market}")
    del rule_tz
    return local.tz_convert(UTC)


def publication_available(filed_date, accepted_datetime, timezone: str) -> tuple[pd.Timestamp, str, bool]:
    """有受理时间用受理时间。只有公布日时，用该日当地 23:59:59，并写明精度。"""
    if accepted_datetime is not None and not pd.isna(accepted_datetime) and str(accepted_datetime).strip():
        stamp = pd.Timestamp(accepted_datetime)
        zone = None if stamp.tzinfo is not None else timezone
        return to_utc(stamp, zone), "accepted_datetime", True
    if filed_date is not None and not pd.isna(filed_date) and str(filed_date).strip():
        day = pd.Timestamp(filed_date)
        if day.tzinfo is not None:
            day = day.tz_convert(timezone).tz_localize(None)
        day = day.normalize()
        end = day.tz_localize(timezone) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        return end.tz_convert(UTC), "publication_date_end", True
    return pd.NaT, "missing", False


def visible(frame: pd.DataFrame, as_of, column: str = "available_time") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    moment = to_utc(as_of, "UTC") if pd.Timestamp(as_of).tzinfo is None else pd.Timestamp(as_of).tz_convert(UTC)
    times = pd.to_datetime(frame[column], utc=True)
    return frame.loc[times.notna() & (times <= moment)].copy()
