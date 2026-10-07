"""银层表的列契约。缺列直接失败，不用交易日填公布时间。"""

from __future__ import annotations

import pandas as pd


class SchemaError(ValueError):
    pass


MARKET_COLUMNS = [
    "instrument_id", "symbol", "trade_date", "open", "high", "low", "close", "prev_close",
    "volume", "amount", "turnover", "vwap", "adj_factor", "split_factor", "dividend_factor",
    "is_trading", "is_suspended", "source", "source_timestamp", "available_time",
    "availability_rule", "ingested_at", "dataset_version", "volume_unit", "price_basis",
    "quality_score", "pit_valid", "corporate_action_valid", "research_grade",
    "selected_source", "conflict", "conflict_value",
]

FUNDAMENTAL_COLUMNS = [
    "instrument_id", "concept", "value", "unit", "period_start", "period_end",
    "filed_date", "accepted_datetime", "available_time", "availability_rule",
    "form", "accn", "source", "revision_id", "pit_valid", "research_grade",
    "quality_score", "dataset_version",
]

ACTION_COLUMNS = [
    "instrument_id", "event_date", "event_type", "split_ratio", "cash_dividend",
    "stock_dividend", "rights_ratio", "adjust_factor", "effective_date", "announcement_date",
    "available_time", "source", "dataset_version",
]

UNIVERSE_COLUMNS = [
    "instrument_id", "universe", "effective_start", "effective_end", "as_of",
    "pit_membership", "source", "dataset_version",
]

MACRO_COLUMNS = [
    "series_id", "observation_date", "value", "release_date", "available_time",
    "availability_rule", "source", "pit_valid", "research_grade", "quality_score",
    "dataset_version",
]

INSTRUMENT_COLUMNS = [
    "instrument_id", "symbol", "exchange", "country", "asset_type", "currency", "name",
    "isin", "cik", "list_date", "delist_date", "sector", "industry", "status",
]

ALIAS_COLUMNS = [
    "instrument_id", "symbol", "effective_start", "effective_end", "source",
]


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaError(f"{name} 缺少列: {missing}")
    ordered = frame.loc[:, columns].copy()
    return ordered
