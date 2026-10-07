"""Pure, fail-closed helpers for the read-only portfolio market observatory.

Nothing in this module reads a file, makes a public-data request, or changes a
research account.  Callers supply canonical daily bars, saved HTML, industry
constituents, and already-computed 20-trading-day returns.  That boundary keeps
the UI from quietly turning an incomplete cache or a partial industry list into
a convincing-looking market fact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
import math
import re
from typing import Any

import pandas as pd


CANONICAL_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
VALID_PERIODS = ("day", "week", "month", "year")
_PERIOD_FREQUENCIES = {"week": "W-FRI", "month": "ME", "year": "YE"}
_FORECAST_MESSAGE = "暂无机构预测"


def validate_period(period: object) -> str:
    """Return one supported market-chart period without silently defaulting.

    A bad period is a caller error.  Falling back to daily or monthly bars here
    would make a selected 年K/周K control lie about what the user is viewing.
    """

    if not isinstance(period, str) or period not in VALID_PERIODS:
        raise ValueError(f"unsupported_market_period:{period!r}")
    return period


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CANONICAL_OHLCV_COLUMNS), index=pd.DatetimeIndex([]))


def _canonical_daily_ohlcv(daily: pd.DataFrame) -> pd.DataFrame:
    """Validate a daily OHLCV panel before aggregation.

    The market-observatory contract intentionally rejects missing amount or
    volume rather than treating an unknown value as zero.  It also rejects
    duplicate calendar dates: picking an arbitrary duplicate would change a
    candle's open/close and therefore its visible return.
    """

    if not isinstance(daily, pd.DataFrame):
        raise TypeError("canonical_ohlcv_requires_dataframe")
    missing = [column for column in CANONICAL_OHLCV_COLUMNS if column not in daily.columns]
    if missing:
        raise ValueError("canonical_ohlcv_missing_columns:" + ",".join(missing))
    if daily.empty:
        return _empty_ohlcv()

    if "date" in daily.columns:
        parsed_dates = pd.to_datetime(daily["date"], errors="coerce")
    elif isinstance(daily.index, pd.RangeIndex):
        raise ValueError("canonical_ohlcv_date_index_required")
    else:
        parsed_dates = pd.to_datetime(daily.index, errors="coerce")
    dates = pd.DatetimeIndex(parsed_dates)
    if dates.isna().any():
        raise ValueError("canonical_ohlcv_invalid_date")
    if dates.tz is not None:
        # A daily bar's exchange date is its local calendar date.  Drop the
        # timezone marker without shifting it into another calendar date.
        dates = dates.tz_localize(None)
    if not dates.equals(dates.normalize()):
        raise ValueError("canonical_ohlcv_requires_daily_dates")
    if dates.duplicated().any():
        raise ValueError("canonical_ohlcv_duplicate_date")

    out = daily.loc[:, list(CANONICAL_OHLCV_COLUMNS)].copy()
    out.index = dates
    for column in CANONICAL_OHLCV_COLUMNS:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"canonical_ohlcv_non_numeric:{column}")
        out[column] = values.astype(float)
    if (out[["volume", "amount"]] < 0).any().any():
        raise ValueError("canonical_ohlcv_negative_volume_or_amount")
    if (out["high"] < out[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("canonical_ohlcv_invalid_high")
    if (out["low"] > out[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("canonical_ohlcv_invalid_low")
    out = out.sort_index()
    out.attrs = dict(daily.attrs)
    return out


def resample_ohlcv(daily: pd.DataFrame, period: object) -> pd.DataFrame:
    """Aggregate canonical daily OHLCV to day/week/month/year bars.

    Open is the first daily open, high/low are the extrema, close is the final
    daily close, and volume/amount are sums.  Week labels end on Friday; month
    and year labels use their calendar period end, which makes chart intervals
    stable even when the market closes before that calendar date.
    """

    period_name = validate_period(period)
    frame = _canonical_daily_ohlcv(daily)
    if period_name == "day" or frame.empty:
        return frame
    out = frame.resample(_PERIOD_FREQUENCIES[period_name]).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
    })
    # The source validator disallows holes in a daily bar, but retain the guard
    # so a future aggregation change cannot emit an incomplete candle.
    out = out.dropna(how="any")
    out.attrs = dict(frame.attrs)
    out.attrs["market_period"] = period_name
    return out


def _cutoff_date(cutoff: object) -> pd.Timestamp:
    if cutoff is None:
        raise ValueError("local_panel_cutoff_required")
    value = pd.to_datetime(cutoff, errors="coerce")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("local_panel_invalid_cutoff")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _local_metadata(*, source: str, cutoff: pd.Timestamp, period: str, available: bool,
                    status: str, reason: str | None = None, as_of: str | None = None,
                    excluded_after_cutoff: int = 0, error: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "available": available,
        "status": status,
        "source": source,
        "source_kind": "local_panel",
        "cutoff": cutoff.date().isoformat(),
        "as_of": as_of,
        "period": period,
        "excluded_after_cutoff": excluded_after_cutoff,
        "caveats": [
            "本地面板仅用于行情观察，不能替代公开实时行情或研究账本。",
            "数据严格截断到明确的本地面板截止日。",
        ],
    }
    if reason is not None:
        value["reason"] = reason
    if error is not None:
        value["error"] = error
    return value


def local_panel_fallback(local_daily: pd.DataFrame | None, *, period: object, cutoff: object,
                         source: object) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a bounded local-panel fallback plus explicit provenance metadata.

    This helper never discovers a local file or assumes freshness.  The caller
    must name both its local source and its known cutoff.  Bad/empty input is a
    visible unavailable result, not an empty-but-successful candle series.
    """

    period_name = validate_period(period)
    cutoff_date = _cutoff_date(cutoff)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("local_panel_source_required")
    source_name = source.strip()
    if local_daily is None:
        return _empty_ohlcv(), _local_metadata(
            source=source_name,
            cutoff=cutoff_date,
            period=period_name,
            available=False,
            status="unavailable",
            reason="local_panel_missing",
        )
    try:
        canonical = _canonical_daily_ohlcv(local_daily)
    except (TypeError, ValueError) as exc:
        return _empty_ohlcv(), _local_metadata(
            source=source_name,
            cutoff=cutoff_date,
            period=period_name,
            available=False,
            status="unavailable",
            reason="local_panel_invalid",
            error=str(exc),
        )
    before_cutoff = canonical.loc[canonical.index <= cutoff_date]
    excluded = int((canonical.index > cutoff_date).sum())
    if before_cutoff.empty:
        return _empty_ohlcv(), _local_metadata(
            source=source_name,
            cutoff=cutoff_date,
            period=period_name,
            available=False,
            status="unavailable",
            reason="local_panel_has_no_rows_at_cutoff",
            excluded_after_cutoff=excluded,
        )
    bars = resample_ohlcv(before_cutoff, period_name)
    as_of = before_cutoff.index[-1].date().isoformat()
    metadata = _local_metadata(
        source=source_name,
        cutoff=cutoff_date,
        period=period_name,
        available=not bars.empty,
        status="local_panel_unverified" if not bars.empty else "unavailable",
        reason=None if not bars.empty else "local_panel_has_no_complete_bars",
        as_of=as_of,
        excluded_after_cutoff=excluded,
    )
    bars.attrs.update({"source": source_name, "cutoff": metadata["cutoff"], "as_of": as_of})
    return bars, metadata


class _Cell:
    """Minimal HTML table cell retained without an optional parser dependency."""

    def __init__(self, text: str, tag: str, attrs: Mapping[str, str | None]):
        self.text = re.sub(r"\s+", " ", text).strip()
        self.tag = tag
        self.colspan = _positive_span(attrs.get("colspan"))
        self.rowspan = _positive_span(attrs.get("rowspan"))


def _positive_span(value: object) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return 1
    return parsed if parsed > 0 else 1


class _TablesParser(HTMLParser):
    """Extract enough table structure to parse a saved THS HTML response."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[_Cell]]] = []
        self._table_depth = 0
        self._table: list[list[_Cell]] | None = None
        self._row: list[_Cell] | None = None
        self._cell_tag: str | None = None
        self._cell_attrs: dict[str, str | None] = {}
        self._cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._table = []
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_tag = tag
            self._cell_attrs = dict(attrs)
            self._cell_text = []
        elif tag == "br" and self._cell_tag is not None:
            self._cell_text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            if self._table_depth == 1 and self._table is not None:
                self.tables.append(self._table)
                self._table = None
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth != 1:
            return
        if tag in {"td", "th"} and self._cell_tag == tag and self._row is not None:
            self._row.append(_Cell("".join(self._cell_text), tag, self._cell_attrs))
            self._cell_tag = None
            self._cell_attrs = {}
            self._cell_text = []
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._table_depth == 1 and self._cell_tag is not None:
            self._cell_text.append(data)


def _expand_table(rows: list[list[_Cell]]) -> list[list[_Cell]]:
    """Expand basic colspan/rowspan so a forecast header aligns with values."""

    pending: dict[int, tuple[_Cell, int]] = {}
    expanded: list[list[_Cell]] = []
    for raw_row in rows:
        row: list[_Cell] = []
        column = 0

        def fill_pending() -> None:
            nonlocal column
            while column in pending:
                cell, remaining = pending[column]
                row.append(cell)
                if remaining <= 1:
                    del pending[column]
                else:
                    pending[column] = (cell, remaining - 1)
                column += 1

        fill_pending()
        for cell in raw_row:
            fill_pending()
            for offset in range(cell.colspan):
                row.append(cell)
                if cell.rowspan > 1:
                    pending[column + offset] = (cell, cell.rowspan - 1)
            column += cell.colspan
        expanded.append(row)
    return expanded


def _normalised_label(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).replace("（", "(").replace("）", ")")


def _forecast_year(header: str) -> int | None:
    """Return a year only when the column is explicitly a forecast column."""

    compact = _normalised_label(header)
    match = re.search(r"(?<!\d)(20\d{2})(?:年)?", compact)
    if not match:
        return None
    lowered = compact.lower()
    # Reported/actual headers always win over a generic word elsewhere.
    if any(marker in compact for marker in ("实际", "报告", "历史")) or re.search(r"20\d{2}a\b", lowered):
        return None
    has_e_marker = bool(re.search(r"20\d{2}\s*e\b", lowered))
    has_forecast_word = any(marker in compact for marker in ("预测", "预期", "预计", "一致预期", "估算"))
    if not (has_e_marker or has_forecast_word):
        return None
    return int(match.group(1))


def _revenue_number(value: str) -> float | None:
    """Parse a displayed revenue value, preserving an unavailable cell as None."""

    compact = _normalised_label(value).replace(",", "").replace("，", "")
    if compact in {"", "--", "-", "—", "暂无", "N/A", "NA"}:
        return None
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(万亿|亿元|万元|亿|万|元)?", compact)
    if not match:
        return None
    number = float(match.group(1))
    multiplier = {
        None: 1.0,
        "元": 1.0,
        "万": 10_000.0,
        "万元": 10_000.0,
        "亿": 100_000_000.0,
        "亿元": 100_000_000.0,
        "万亿": 1_000_000_000_000.0,
    }[match.group(2)]
    result = number * multiplier
    return result if math.isfinite(result) else None


def _forecast_unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "status": "unavailable",
        "source": "ths_worth_html",
        "metric": "营业收入(元)",
        "basis": "analyst_consensus_forecast_average",
        "forecasts": [],
        "message": _FORECAST_MESSAGE,
        "reason": reason,
    }


def parse_ths_revenue_forecast(html: object) -> dict[str, Any]:
    """Parse only THS analyst *forecast* revenue-average cells from saved HTML.

    The parser accepts a saved ``worth.html`` payload, not a URL.  It will use
    only a row labelled exactly ``营业收入(元)`` and columns whose headers
    explicitly mark a forecast year (for example ``2026E`` or ``2026年预测``).
    Reported ``A`` years, current revenue tables, and similarly named metrics
    such as ``营业总收入(元)`` are intentionally ignored.
    """

    if not isinstance(html, str) or not html.strip():
        return _forecast_unavailable("forecast_revenue_not_found")
    parser = _TablesParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # HTMLParser is intentionally best-effort for saved pages.
        return _forecast_unavailable("forecast_revenue_html_unreadable")

    values_by_year: dict[int, float] = {}
    for raw_table in parser.tables:
        table = _expand_table(raw_table)
        for row_index, row in enumerate(table):
            if not row or _normalised_label(row[0].text) != "营业收入(元)":
                continue
            header_rows = [
                header_row for header_row in table[:row_index]
                if any(cell.tag == "th" for cell in header_row)
            ]
            if not header_rows:
                continue
            for column, cell in enumerate(row[1:], start=1):
                header = " ".join(
                    header_row[column].text for header_row in header_rows if column < len(header_row)
                )
                year = _forecast_year(header)
                value = _revenue_number(cell.text)
                if year is None or value is None:
                    continue
                prior = values_by_year.get(year)
                if prior is not None and not math.isclose(prior, value, rel_tol=0.0, abs_tol=1e-12):
                    return _forecast_unavailable("ambiguous_forecast_revenue")
                values_by_year[year] = value
    if not values_by_year:
        return _forecast_unavailable("forecast_revenue_not_found")
    forecasts = [{"year": year, "value": values_by_year[year]} for year in sorted(values_by_year)]
    return {
        "available": True,
        "status": "available",
        "source": "ths_worth_html",
        "metric": "营业收入(元)",
        "basis": "analyst_consensus_forecast_average",
        "forecasts": forecasts,
        "message": None,
        "reason": None,
    }


def _public_code(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{1,6}", text):
        return text.zfill(6)
    return text


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _records(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return [dict(row) for row in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        # Constituents are row records; a code->name map is deliberately not
        # accepted because it cannot attest whether membership is complete.
        return []
    try:
        rows = list(value)  # type: ignore[arg-type]
    except TypeError:
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _return_map(values: object) -> dict[str, float]:
    if values is None:
        return {}
    if isinstance(values, pd.DataFrame):
        rows: Iterable[Mapping[str, Any]] = values.to_dict(orient="records")
    elif isinstance(values, pd.Series):
        return {
            code: parsed
            for raw_code, raw_value in values.items()
            if (code := _public_code(raw_code)) is not None and (parsed := _finite_float(raw_value)) is not None
        }
    elif isinstance(values, Mapping):
        return {
            code: parsed
            for raw_code, raw_value in values.items()
            if (code := _public_code(raw_code)) is not None and (parsed := _finite_float(raw_value)) is not None
        }
    else:
        try:
            rows = list(values)  # type: ignore[arg-type]
        except TypeError:
            return {}
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        code = _public_code(row.get("code"))
        value = _finite_float(row.get("return_20d", row.get("month_return")))
        if code is not None and value is not None:
            result[code] = value
    return result


def _ranking_unavailable(reason: str, *, constituent_count: int = 0,
                         missing_return_codes: list[str] | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "status": "unavailable",
        "metric": "20_trading_day_return",
        "window_trading_days": 20,
        "top": [],
        "constituent_count": constituent_count,
        "missing_return_codes": missing_return_codes or [],
        "reason": reason,
    }


def top_month_returns(constituents: object, returns: object, *, limit: int = 3,
                      constituents_complete: bool = True) -> dict[str, Any]:
    """Rank a supplied industry's full constituents by supplied 20-day returns.

    The function does not infer industry membership, download missing price
    history, or rank only the selected portfolio.  Unless the caller explicitly
    attests that its input is a complete industry constituent list *and* every
    member has a finite 20-trading-day return, it returns ``unavailable`` with
    no fabricated Top3 rows.
    """

    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("invalid_top_limit")
    rows = _records(constituents)
    if not rows:
        return _ranking_unavailable("industry_constituents_missing")
    normalized: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = _public_code(row.get("code"))
        if code is None or code in normalized:
            return _ranking_unavailable("industry_constituents_invalid", constituent_count=len(normalized))
        normalized[code] = {"code": code, "name": row.get("name")}
    if not constituents_complete:
        return _ranking_unavailable("industry_constituents_incomplete", constituent_count=len(normalized))
    if len(normalized) < limit:
        return _ranking_unavailable("industry_constituents_insufficient", constituent_count=len(normalized))
    return_values = _return_map(returns)
    missing = sorted(code for code in normalized if code not in return_values)
    if missing:
        return _ranking_unavailable(
            "month_return_coverage_incomplete",
            constituent_count=len(normalized),
            missing_return_codes=missing,
        )
    ranked = sorted(
        normalized,
        key=lambda code: (-return_values[code], code, str(normalized[code].get("name") or "")),
    )[:limit]
    top = [
        {
            "rank": rank,
            "code": code,
            "name": normalized[code]["name"],
            "return_20d": return_values[code],
        }
        for rank, code in enumerate(ranked, start=1)
    ]
    return {
        "available": True,
        "status": "available",
        "metric": "20_trading_day_return",
        "window_trading_days": 20,
        "top": top,
        "constituent_count": len(normalized),
        "missing_return_codes": [],
        "reason": None,
    }


__all__ = [
    "CANONICAL_OHLCV_COLUMNS",
    "VALID_PERIODS",
    "local_panel_fallback",
    "parse_ths_revenue_forecast",
    "resample_ohlcv",
    "top_month_returns",
    "validate_period",
]
