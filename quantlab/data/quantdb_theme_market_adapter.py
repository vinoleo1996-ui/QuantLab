"""QuantDB-only input adapter for ``theme_carrier_graph_v1``.

The adapter does not open a network connection, choose a universe, or place an
order.  A caller passes the exact raw responses it received.  Every response is
stored in the research ledger first.  Limit-up prices are then derived with
``quantlab.data.exchange_rules`` from the previous session's unadjusted close,
the stock-list ST flag, and the trading calendar.  Vendor fields named
``zt_price`` or ``zaf`` are not read.

Two scored sessions need three unadjusted partitions: the decision session,
the previous open session, and the open session before that.  The earliest
close is only the reference price for the middle bar's limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import io
import json
import math
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from quantlab.data.exchange_rules import limit_prices, limit_ratio
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import (
    DECISION_TIMEZONE,
    U0,
    UniverseSnapshot,
    forward_membership_sha256,
    is_exact_decision_cutoff,
)
from quantlab.theme_carrier_graph import ThemeCarrierGraphResult, score_theme_carrier_graph_v1


QUANTDB_THEME_PROVIDER_ID = "quantdb.theme-market"
QUANTDB_THEME_ADAPTER_VERSION = "quantdb-theme-market-adapter-v1"
_HOST = "www.quantdb.cn"
_PATHS = frozenset({
    "/api/v1/data/preview",
    "/api/v1/data/download",
    "/api/v1/data/download/manifest",
    "/api/v1/data/stock-list",
    "/api/v1/data/calendar",
})
_DATASETS = frozenset(
    {"sector_concept", "daily_manifest", "daily_unadjusted", "stock_list", "trade_calendar"}
)
_CONCEPT_TYPE = "概念板块"
_PRICE_TICK = 0.01


class QuantDBThemeMarketError(ValueError):
    """The caller violated the adapter contract before any market parse."""


@dataclass(frozen=True)
class QuantDBRawResponse:
    """One unparsed QuantDB response, including the request that produced it."""

    dataset: str
    response_id: str
    request_parameters: Mapping[str, str]
    canonical_url: str
    raw: bytes
    content_type: str
    published_at: Any
    first_captured_at: Any
    available_at: Any
    availability_basis: str


@dataclass(frozen=True)
class QuantDBThemeGraphInput:
    """Receipt-backed bars and source-native concept edges for one graph call."""

    bars: pd.DataFrame
    theme_edges: pd.DataFrame
    receipt_ids: tuple[str, ...]
    receipt_hashes: tuple[str, ...]


@dataclass(frozen=True)
class QuantDBThemeGraphCapture:
    """A graph score, or one explicit abstention with a cross-section diagnostic."""

    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    decision_session: date
    decision_at: pd.Timestamp
    graph_input: QuantDBThemeGraphInput | None
    graph_result: ThemeCarrierGraphResult | None
    receipt_ids: tuple[str, ...]
    receipt_hashes: tuple[str, ...]
    coverage: Mapping[str, object]
    adapter_version: str = QUANTDB_THEME_ADAPTER_VERSION


@dataclass(frozen=True)
class _RecordedResponse:
    response: QuantDBRawResponse
    receipt_id: str
    receipt_hash: str
    receipt_evidence: Mapping[str, Any]
    parse_error: str | None = None


def _local_time(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_quantdb_" + field) from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("timezone_required_quantdb_" + field)
    return stamp.tz_convert(DECISION_TIMEZONE)


def _session(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_quantdb_" + field) from exc
    if pd.isna(stamp):
        raise ValueError("invalid_quantdb_" + field)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(DECISION_TIMEZONE).tz_localize(None)
    return stamp.normalize()


def _as_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_quantdb_" + field)
    return value.strip()


def _instrument_id(symbol: Any) -> str:
    text = _as_text(str(symbol), "symbol").upper()
    number, separator, market = text.partition(".")
    if separator != "." or len(number) != 6 or not number.isdigit() or market not in {"SZ", "SH", "BJ"}:
        raise ValueError("invalid_quantdb_symbol")
    if number.startswith(("4", "8", "92", "43", "83", "87")):
        prefix = "bj."
    elif number.startswith(("5", "6", "9")):
        prefix = "sh."
    else:
        prefix = "sz."
    return prefix + number


def _calendar(exchange_sessions: Sequence[Any]) -> pd.DatetimeIndex:
    try:
        calendar = pd.DatetimeIndex([_session(value, "exchange_session") for value in exchange_sessions])
    except (TypeError, ValueError) as exc:
        raise QuantDBThemeMarketError("quantdb_exchange_calendar_required") from exc
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise QuantDBThemeMarketError("quantdb_exchange_calendar_invalid")
    return calendar


def _previous_open(calendar: pd.DatetimeIndex, session: pd.Timestamp) -> pd.Timestamp | None:
    if session not in calendar:
        return None
    location = int(calendar.get_loc(session))
    if location == 0:
        return None
    return pd.Timestamp(calendar[location - 1])


def _optional_list_date(value: Any) -> pd.Timestamp:
    """Missing or placeholder listing dates stay unknown instead of failing the file."""
    if value in (None, "", "0", 0):
        return pd.NaT
    try:
        return _session(value, "list_date")
    except ValueError:
        return pd.NaT


def _listing_session(list_date: pd.Timestamp, bar_session: pd.Timestamp, calendar: pd.DatetimeIndex) -> int | None:
    """Return the 1-based listing session, or None when the calendar cannot prove it.

    A list date before the supplied calendar only yields a lower bound.  That
    bound is usable once it already reaches the seasoned threshold of 6.
    Below that threshold the true session count is still unknown.
    """
    if pd.isna(list_date) or bar_session not in calendar:
        return None
    bar_location = int(calendar.get_loc(bar_session))
    if list_date < calendar[0]:
        lower_bound = bar_location + 1
        return lower_bound if lower_bound >= 6 else None
    upcoming = calendar[calendar >= list_date]
    if len(upcoming) == 0:
        return None
    first = pd.Timestamp(upcoming[0])
    if first > bar_session:
        return None
    return bar_location - int(calendar.get_loc(first)) + 1


def _is_declared_url(value: Any) -> bool:
    try:
        parsed = urlparse(_as_text(value, "canonical_url"))
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == _HOST
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and parsed.path in _PATHS
        and not parsed.params
        and not parsed.fragment
    )


def _parameters(response: QuantDBRawResponse) -> dict[str, str]:
    parameters = response.request_parameters
    if not isinstance(parameters, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in parameters.items()
    ):
        raise ValueError("invalid_quantdb_request_parameters")
    return dict(parameters)


def _json_bytes(raw: bytes) -> Any:
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError("quantdb_raw_response_required")
    try:
        return json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_quantdb_json_response") from exc


def _table(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("quantdb_response_table_missing")
    return [dict(row) for row in rows]


def _record_response(ledger: ResearchSignalLedger, response: QuantDBRawResponse) -> _RecordedResponse:
    if not isinstance(response, QuantDBRawResponse):
        raise ValueError("quantdb_raw_response_required")
    dataset = _as_text(response.dataset, "dataset")
    response_id = _as_text(response.response_id, "response_id")
    if dataset not in _DATASETS:
        raise ValueError("unsupported_quantdb_dataset")
    if not _is_declared_url(response.canonical_url):
        raise ValueError("undeclared_quantdb_canonical_url")
    parameters = _parameters(response)
    content_type = _as_text(response.content_type, "content_type").split(";", 1)[0].lower()
    if dataset == "daily_unadjusted":
        if content_type != "application/vnd.apache.parquet":
            raise ValueError("unexpected_quantdb_content_type")
    elif content_type != "application/json":
        raise ValueError("unexpected_quantdb_content_type")
    receipt = ledger.record_receipt(
        source_id=QUANTDB_THEME_PROVIDER_ID,
        source_family="quantdb",
        origin_id="quantdb:" + dataset + ":" + response_id,
        canonical_url=response.canonical_url,
        raw=bytes(response.raw),
        content_type=content_type,
        published_at=response.published_at,
        first_captured_at=response.first_captured_at,
        available_at=response.available_at,
        availability_basis=_as_text(response.availability_basis, "availability_basis"),
        metadata={
            "adapter_version": QUANTDB_THEME_ADAPTER_VERSION,
            "dataset": dataset,
            "response_id": response_id,
            "request_parameters": dict(sorted(parameters.items())),
        },
    )
    receipt_id = str(receipt.get("canonical_receipt_id") or receipt["event_id"])
    evidence = ledger.export_receipt_evidence(receipt_id)
    return _RecordedResponse(
        response=response,
        receipt_id=receipt_id,
        receipt_hash=str(evidence["raw"]["sha256"]),
        receipt_evidence=evidence,
    )


def _receipt_is_on_time(record: _RecordedResponse, cutoff: pd.Timestamp) -> bool:
    try:
        response = record.response
        published = _local_time(response.published_at, "published_at")
        captured = _local_time(response.first_captured_at, "first_captured_at")
        available = _local_time(response.available_at, "available_at")
        recorded = _local_time(record.receipt_evidence["receipt"]["recorded_at"], "recorded_at")
    except (KeyError, TypeError, ValueError):
        return False
    return published <= captured <= available <= cutoff and recorded <= cutoff


def _provenance(record: _RecordedResponse) -> dict[str, str]:
    body = record.receipt_evidence["receipt"]["body"]
    receipt = record.receipt_evidence["receipt"]
    return {
        "published_at": str(body["published_at"]),
        "first_captured_at": str(body["first_captured_at"]),
        "available_at": str(receipt["available_at"]),
        "source_receipt_id": record.receipt_id,
        "content_sha256": record.receipt_hash,
        "capture_mode": "forward_receipt",
    }


def _parse_calendar_rows(record: _RecordedResponse) -> pd.DatetimeIndex:
    rows = _table(_json_bytes(record.response.raw))
    sessions = []
    for row in rows:
        if int(row.get("is_open", 1)) != 1:
            continue
        sessions.append(_session(row.get("trade_date"), "trade_date"))
    calendar = pd.DatetimeIndex(sessions)
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("quantdb_trade_calendar_invalid")
    return calendar


def _parse_stock_list(record: _RecordedResponse) -> pd.DataFrame:
    rows = _table(_json_bytes(record.response.raw))
    parsed = []
    for row in rows:
        symbol = _instrument_id(row.get("symbol"))
        listed = _optional_list_date(row.get("list_date"))
        flag = row.get("is_st")
        if flag not in (0, 1, False, True):
            raise ValueError("quantdb_st_flag_invalid")
        parsed.append({"instrument_id": symbol, "list_date": listed, "is_st": bool(flag)})
    frame = pd.DataFrame(parsed)
    if frame.empty or frame.instrument_id.duplicated().any():
        raise ValueError("quantdb_stock_list_invalid")
    return frame.set_index("instrument_id")


def _parse_concepts(record: _RecordedResponse) -> pd.DataFrame:
    rows = _table(_json_bytes(record.response.raw))
    parsed = []
    seen_anywhere: set[str] = set()
    for row in rows:
        if not {"SectorCode", "SectorType", "Symbol"}.issubset(row):
            raise ValueError("quantdb_sector_columns_missing")
        symbol = _instrument_id(row.get("Symbol"))
        seen_anywhere.add(symbol)
        if str(row.get("SectorType") or "").strip() != _CONCEPT_TYPE:
            continue
        theme = _as_text(str(row.get("SectorCode") or ""), "sector_code")
        parsed.append({"instrument_id": symbol, "theme_id": theme})
    concepts = pd.DataFrame(parsed, columns=["instrument_id", "theme_id"])
    if not concepts.empty and concepts.duplicated(["instrument_id", "theme_id"]).any():
        raise ValueError("quantdb_duplicate_concept_edge")
    concepts.attrs["symbols_in_source"] = seen_anywhere
    return concepts


def _parse_daily(record: _RecordedResponse) -> pd.DataFrame:
    parameters = _parameters(record.response)
    expected = parameters.get("trade_date")
    if expected is None or len(expected) != 8 or not expected.isdigit():
        raise ValueError("quantdb_daily_trade_date_required")
    frame = pd.read_parquet(io.BytesIO(bytes(record.response.raw)))
    required = {"symbol", "time", "open", "high", "low", "close"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("quantdb_daily_columns_missing")
    parsed = frame.loc[:, ["symbol", "time", "open", "high", "low", "close"]].copy()
    parsed["instrument_id"] = parsed.symbol.map(_instrument_id)
    parsed["session"] = parsed.time.map(lambda value: _session(value, "bar_time"))
    expected_session = _session(expected, "trade_date")
    if parsed.session.ne(expected_session).any() or parsed.instrument_id.duplicated().any():
        raise ValueError("quantdb_daily_session_invalid")
    for column in ("open", "high", "low", "close"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    if not parsed.loc[:, ["open", "high", "low", "close"]].map(math.isfinite).all().all():
        raise ValueError("quantdb_daily_price_invalid")
    return parsed


def _limit_for(close: float, code: str, bar_session: pd.Timestamp, listing: int | None, is_st: Any) -> float | None:
    ratio = limit_ratio(code, bar_session, is_st, listing)
    if ratio is None or not math.isfinite(close) or close <= 0:
        return None
    up, _down = limit_prices(close, ratio, tick=_PRICE_TICK)
    return up


def _diagnose(
    *,
    daily: Mapping[pd.Timestamp, pd.DataFrame],
    stocks: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    session: pd.Timestamp,
) -> dict[str, object]:
    previous = _previous_open(calendar, session)
    if previous is None or session not in daily or previous not in daily:
        return {"diagnostic_scope": "cross_section_not_u0_not_a_score", "diagnostic_session": None}
    current = daily[session].set_index("instrument_id")
    prior = daily[previous].set_index("instrument_id")
    defined = touched = sealed = broken = undefined = 0
    for code in current.index.intersection(prior.index):
        if code not in stocks.index or pd.isna(stocks.loc[code, "list_date"]):
            undefined += 1
            continue
        info = stocks.loc[code]
        listing = _listing_session(pd.Timestamp(info.list_date), session, calendar)
        limit = _limit_for(float(prior.loc[code, "close"]), code, session, listing, bool(info.is_st))
        if limit is None:
            undefined += 1
            continue
        defined += 1
        high = float(current.loc[code, "high"])
        close = float(current.loc[code, "close"])
        if high >= limit - _PRICE_TICK / 2:
            touched += 1
            if close >= limit - _PRICE_TICK / 2:
                sealed += 1
            else:
                broken += 1
    return {
        "diagnostic_scope": "cross_section_not_u0_not_a_score",
        "diagnostic_session": str(session.date()),
        "limit_defined_count": defined,
        "limit_undefined_count": undefined,
        "touched_count": touched,
        "sealed_count": sealed,
        "broken_count": broken,
    }


def _u0_reason(u0: Any, session: pd.Timestamp, cutoff: pd.Timestamp) -> str | None:
    if not isinstance(u0, UniverseSnapshot) or u0.status != "ok" or u0.pool_id != U0.pool_id:
        return "quantdb_u0_not_available"
    if _session(u0.session, "u0_session") != session or _local_time(u0.decision_at, "u0_decision_at") != cutoff:
        return "quantdb_u0_decision_identity_mismatch"
    members = tuple(str(item) for item in u0.members)
    if not members or len(members) != len(set(members)):
        return "quantdb_u0_membership_invalid"
    return None


def _outside_listed_limit_rule(code: str) -> bool:
    """Boards with no seasoned cash-equity limit in ``exchange_rules``."""
    number = str(code).split(".")[-1]
    return number.startswith(("4", "8", "92", "43", "83", "87"))


def _scoring_snapshot(u0: UniverseSnapshot, members: tuple[str, ...]) -> UniverseSnapshot:
    """Score only priceable names.  The original U0 membership stays in coverage."""
    membership_hash = forward_membership_sha256(
        spec=U0,
        session=u0.session,
        decision_at=u0.decision_at,
        cadence_anchor=u0.cadence_anchor,
        input_hash=u0.input_hash,
        members=members,
        status="ok",
        abstain_reason=None,
    )
    return UniverseSnapshot(
        contract_version=u0.contract_version,
        pool_id=u0.pool_id,
        session=u0.session,
        decision_at=u0.decision_at,
        cadence_anchor=u0.cadence_anchor,
        members=members,
        rank_by_instrument={code: u0.rank_by_instrument.get(code, 0) for code in members},
        adv60_by_instrument={code: u0.adv60_by_instrument.get(code, 0.0) for code in members},
        eligibility_reason_by_instrument={},
        input_hash=u0.input_hash,
        membership_hash=membership_hash,
        status="ok",
        abstain_reason=None,
    )


def _materialize(
    *,
    records: Sequence[_RecordedResponse],
    concepts: pd.DataFrame,
    stocks: pd.DataFrame,
    daily: Mapping[pd.Timestamp, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    members: Sequence[str],
    session: pd.Timestamp,
    prior: pd.Timestamp,
    reference: pd.Timestamp,
) -> tuple[QuantDBThemeGraphInput | None, str | None, tuple[tuple[str, str], ...]]:
    ordered = tuple(sorted(records, key=lambda record: record.receipt_id))
    receipt_ids = tuple(record.receipt_id for record in ordered)
    receipt_hashes = tuple(record.receipt_hash for record in ordered)
    by_session = {pd.Timestamp(key): value.set_index("instrument_id") for key, value in daily.items()}
    excluded: list[tuple[str, str]] = []
    scored: list[str] = []
    for code in members:
        if _outside_listed_limit_rule(code) or code not in stocks.index or pd.isna(stocks.loc[code, "list_date"]):
            excluded.append((code, "limit_rule_undefined"))
            continue
        unpriceable = False
        for target, reference_session in ((prior, reference), (session, prior)):
            current = by_session[target]
            anchor = by_session[reference_session]
            if code not in current.index or code not in anchor.index:
                return None, "quantdb_bar_or_identity_missing", tuple(excluded)
            info = stocks.loc[code]
            listing = _listing_session(pd.Timestamp(info.list_date), target, calendar)
            if _limit_for(float(anchor.loc[code, "close"]), code, target, listing, bool(info.is_st)) is None:
                unpriceable = True
                break
        if unpriceable:
            excluded.append((code, "limit_rule_undefined"))
        else:
            scored.append(code)
    if not scored:
        return None, "quantdb_no_priceable_members", tuple(excluded)
    bars: list[dict[str, Any]] = []
    for target, reference_session in ((prior, reference), (session, prior)):
        current = by_session[target]
        anchor = by_session[reference_session]
        daily_record = next(
            record for record in records
            if record.response.dataset == "daily_unadjusted"
            and _parameters(record.response)["trade_date"] == target.strftime("%Y%m%d")
        )
        provenance = _provenance(daily_record)
        for code in scored:
            info = stocks.loc[code]
            listing = _listing_session(pd.Timestamp(info.list_date), target, calendar)
            limit = _limit_for(float(anchor.loc[code, "close"]), code, target, listing, bool(info.is_st))
            price = current.loc[code]
            if limit is None or float(price.high) < float(price.close) or float(price.high) <= 0 or float(price.close) <= 0:
                return None, "quantdb_daily_price_invalid", tuple(excluded)
            bars.append({
                "session": target,
                "instrument_id": code,
                "open": float(price.open),
                "high": float(price.high),
                "low": float(price.low),
                "close": float(price.close),
                "reference_price": float(anchor.loc[code, "close"]),
                "limit_up": float(limit),
                "price_tick": _PRICE_TICK,
                "quality_status": "ok",
                "supporting_receipt_ids": receipt_ids,
                "supporting_content_sha256s": receipt_hashes,
                **provenance,
            })
    sector_record = next(record for record in records if record.response.dataset == "sector_concept")
    known_symbols = set(concepts.attrs.get("symbols_in_source", ()))
    edge_rows: list[dict[str, Any]] = []
    provenance = _provenance(sector_record)
    grouped = concepts.groupby("instrument_id").theme_id.apply(tuple).to_dict() if not concepts.empty else {}
    for code in scored:
        if code not in known_symbols:
            return None, "quantdb_theme_membership_unconfirmed", tuple(excluded)
        themes = tuple(sorted(set(grouped.get(code, ()))))
        if themes:
            for theme in themes:
                edge_rows.append({
                    "session": session,
                    "instrument_id": code,
                    "theme_id": theme,
                    "mapping_status": "active",
                    "supporting_receipt_ids": receipt_ids,
                    "supporting_content_sha256s": receipt_hashes,
                    **provenance,
                })
        else:
            edge_rows.append({
                "session": session,
                "instrument_id": code,
                "theme_id": pd.NA,
                "mapping_status": "known_no_eligible_theme",
                "supporting_receipt_ids": receipt_ids,
                "supporting_content_sha256s": receipt_hashes,
                **provenance,
            })
    graph_input = QuantDBThemeGraphInput(
        bars=pd.DataFrame(bars).sort_values(["session", "instrument_id"], kind="stable").reset_index(drop=True),
        theme_edges=pd.DataFrame(edge_rows).sort_values(
            ["instrument_id", "mapping_status", "theme_id"], kind="stable", na_position="last"
        ).reset_index(drop=True),
        receipt_ids=receipt_ids,
        receipt_hashes=receipt_hashes,
    )
    return graph_input, None, tuple(excluded)


def _result(
    *,
    status: Literal["ok", "abstain"],
    reason: str | None,
    session: pd.Timestamp,
    cutoff: pd.Timestamp,
    records: Sequence[_RecordedResponse],
    coverage: Mapping[str, object],
    graph_input: QuantDBThemeGraphInput | None = None,
    graph_result: ThemeCarrierGraphResult | None = None,
) -> QuantDBThemeGraphCapture:
    return QuantDBThemeGraphCapture(
        status=status,
        abstain_reason=reason,
        decision_session=session.date(),
        decision_at=cutoff,
        graph_input=graph_input,
        graph_result=graph_result,
        receipt_ids=tuple(record.receipt_id for record in sorted(records, key=lambda item: item.receipt_id)),
        receipt_hashes=tuple(record.receipt_hash for record in sorted(records, key=lambda item: item.receipt_id)),
        coverage=dict(coverage),
    )


def capture_quantdb_theme_graph_v1(
    *,
    ledger: ResearchSignalLedger,
    u0: UniverseSnapshot | None,
    decision_session: Any,
    prior_session: Any,
    decision_at: Any,
    responses: Sequence[QuantDBRawResponse],
) -> QuantDBThemeGraphCapture:
    """Store QuantDB responses and score the concept graph, or abstain visibly.

    ``u0`` remains mandatory for a score.  This adapter does not manufacture a
    liquidity pool from two or three daily partitions.
    """
    if not isinstance(ledger, ResearchSignalLedger):
        raise QuantDBThemeMarketError("quantdb_research_ledger_required")
    session = _session(decision_session, "decision_session")
    prior = _session(prior_session, "prior_session")
    cutoff = _local_time(decision_at, "decision_at")
    if prior >= session:
        raise QuantDBThemeMarketError("quantdb_prior_session_must_precede_decision_session")
    if not is_exact_decision_cutoff(session, cutoff):
        raise QuantDBThemeMarketError("quantdb_decision_at_must_equal_session_1800_shanghai")
    if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence) or not responses:
        raise QuantDBThemeMarketError("quantdb_responses_required")
    response_ids = []
    for response in responses:
        if not isinstance(response, QuantDBRawResponse):
            return _result(
                status="abstain", reason="quantdb_raw_response_invalid", session=session, cutoff=cutoff,
                records=(), coverage={},
            )
        response_ids.append(response.response_id)
    if len(response_ids) != len(set(response_ids)):
        return _result(
            status="abstain", reason="quantdb_response_id_duplicate", session=session, cutoff=cutoff,
            records=(), coverage={},
        )

    records: list[_RecordedResponse] = []
    try:
        for response in responses:
            records.append(_record_response(ledger, response))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _result(
            status="abstain", reason="quantdb_raw_response_invalid", session=session, cutoff=cutoff,
            records=records, coverage={},
        )

    try:
        calendars = [record for record in records if record.response.dataset == "trade_calendar"]
        lists = [record for record in records if record.response.dataset == "stock_list"]
        sectors = [record for record in records if record.response.dataset == "sector_concept"]
        manifests = [record for record in records if record.response.dataset == "daily_manifest"]
        dailies = [record for record in records if record.response.dataset == "daily_unadjusted"]
        if (
            len(calendars) != 1
            or len(lists) != 1
            or len(sectors) != 1
            or len(manifests) != 1
            or len(dailies) != 3
        ):
            raise ValueError("quantdb_response_set_incomplete")
        calendar = _parse_calendar_rows(calendars[0])
        stocks = _parse_stock_list(lists[0])
        concepts = _parse_concepts(sectors[0])
        daily: dict[pd.Timestamp, pd.DataFrame] = {}
        for record in dailies:
            frame = _parse_daily(record)
            daily[pd.Timestamp(frame.session.iloc[0])] = frame
    except (KeyError, TypeError, ValueError) as exc:
        reason = str(exc) if str(exc).startswith("quantdb_") else "quantdb_raw_response_invalid"
        return _result(
            status="abstain", reason=reason, session=session, cutoff=cutoff, records=records, coverage={},
        )

    expected_prior = _previous_open(calendar, session)
    expected_reference = None if expected_prior is None else _previous_open(calendar, expected_prior)
    if expected_prior != prior or expected_reference is None:
        return _result(
            status="abstain", reason="quantdb_prior_session_not_previous_open_session",
            session=session, cutoff=cutoff, records=records, coverage={},
        )
    coverage = _diagnose(daily=daily, stocks=stocks, calendar=calendar, session=session)
    coverage["concept_edge_count"] = int(len(concepts))
    coverage["symbols_seen_in_sector_file"] = int(len(concepts.attrs.get("symbols_in_source", ())))
    if any(not _receipt_is_on_time(record, cutoff) for record in records):
        return _result(
            status="abstain", reason="quantdb_receipt_late_for_decision", session=session, cutoff=cutoff,
            records=records, coverage=coverage,
        )
    if any(day not in daily for day in (session, prior, expected_reference)):
        return _result(
            status="abstain", reason="quantdb_market_response_missing", session=session, cutoff=cutoff,
            records=records, coverage=coverage,
        )
    reason = _u0_reason(u0, session, cutoff)
    if reason is not None:
        return _result(
            status="abstain", reason=reason, session=session, cutoff=cutoff, records=records, coverage=coverage,
        )
    assert u0 is not None
    members = tuple(str(code) for code in u0.members)
    graph_input, material_reason, excluded = _materialize(
        records=records,
        concepts=concepts,
        stocks=stocks,
        daily=daily,
        calendar=calendar,
        members=members,
        session=session,
        prior=prior,
        reference=expected_reference,
    )
    reason_counts: dict[str, int] = {}
    for _code, reason in excluded:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    coverage = dict(coverage)
    coverage.update(
        {
            "u0_member_count": len(members),
            "excluded_member_count": len(excluded),
            "excluded_reason_counts": reason_counts,
            "scored_member_count": len(members) - len(excluded),
        }
    )
    if material_reason is not None:
        return _result(
            status="abstain", reason=material_reason, session=session, cutoff=cutoff,
            records=records, coverage=coverage,
        )
    assert graph_input is not None and u0 is not None
    scored_members = tuple(dict.fromkeys(graph_input.bars.instrument_id.astype(str)))
    graph_result = score_theme_carrier_graph_v1(
        decision_session=session,
        prior_session=prior,
        decision_at=cutoff,
        u0=_scoring_snapshot(u0, scored_members),
        bars=graph_input.bars,
        theme_edges=graph_input.theme_edges,
        ledger=ledger,
    )
    if graph_result.status != "ok":
        return _result(
            status="abstain", reason=graph_result.abstain_reason, session=session, cutoff=cutoff,
            records=records, coverage=coverage, graph_input=graph_input, graph_result=graph_result,
        )
    return _result(
        status="ok", reason=None, session=session, cutoff=cutoff, records=records, coverage=coverage,
        graph_input=graph_input, graph_result=graph_result,
    )


__all__ = [
    "QUANTDB_THEME_ADAPTER_VERSION",
    "QUANTDB_THEME_PROVIDER_ID",
    "QuantDBRawResponse",
    "QuantDBThemeGraphCapture",
    "QuantDBThemeGraphInput",
    "QuantDBThemeMarketError",
    "capture_quantdb_theme_graph_v1",
]
