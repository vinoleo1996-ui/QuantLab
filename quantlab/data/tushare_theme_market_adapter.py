"""Tushare-only, receipt-backed input adapter for ``theme_carrier_graph_v1``.

This module is deliberately a thin, provider-specific boundary.  It does not
open a network connection, schedule collection, train a model, construct a
portfolio, or call any account code.  A separately authorised collection job
must pass in the exact raw Tushare Pro API responses it received.  The adapter
first persists every supplied response as immutable evidence, then either
materialises a source-native many-to-many graph input or returns one explicit
whole-period abstention.

The adapter is useful only after a real provider trial has been qualified.  Its
synthetic unit-test fixtures prove parsing and provenance rules; they do not
make Tushare (or any other provider) an approved data source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from quantlab.data.provider_trial import (
    ProviderTrialError,
    QualifiedProviderTrial,
    acceptance_requirements,
    qualify_provider_trial,
)
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import (
    DECISION_TIMEZONE,
    U0,
    UniverseSnapshot,
    is_exact_decision_cutoff,
)


TUSHARE_THEME_MARKET_PROVIDER_ID = "tushare.pro.theme-market"
TUSHARE_THEME_MARKET_ADAPTER_VERSION = "tushare-theme-market-adapter-v1"
_TUSHARE_HOST = "api.tushare.pro"
_TUSHARE_CONTENT_TYPE = "application/json"
_MARKET_ENDPOINTS = frozenset({"daily", "stk_limit", "limit_list_d"})
_ALLOWED_ENDPOINTS = frozenset({"ths_index", "ths_member", *_MARKET_ENDPOINTS})
_EOD_CAPTURE_START = (15, 30)
_EOD_CAPTURE_END = (17, 50)


class TushareThemeMarketError(ValueError):
    """A caller attempted to use an unqualified or malformed Tushare trial."""


@dataclass(frozen=True)
class TushareRawResponse:
    """One unparsed API response supplied by an authorised collector.

    The request parameters are evidence too: the thin adapter uses them to
    reject a response for the wrong endpoint/date rather than treating a
    plausible-looking JSON payload as interchangeable source material.
    """

    endpoint: str
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
class TushareThemeGraphInput:
    """Only the graph scorer's receipt-backed, source-native inputs."""

    bars: pd.DataFrame
    theme_edges: pd.DataFrame
    receipt_ids: tuple[str, ...]
    receipt_hashes: tuple[str, ...]


@dataclass(frozen=True)
class TushareThemeMarketCapture:
    """One materialised graph input or a visible whole-period abstention."""

    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    decision_session: date
    decision_at: pd.Timestamp
    graph_input: TushareThemeGraphInput | None
    receipt_ids: tuple[str, ...]
    receipt_hashes: tuple[str, ...]
    adapter_version: str = TUSHARE_THEME_MARKET_ADAPTER_VERSION


@dataclass(frozen=True)
class _RecordedResponse:
    response: TushareRawResponse
    frame: pd.DataFrame | None
    receipt_id: str
    receipt_hash: str
    receipt_evidence: Mapping[str, Any]
    parse_error: str | None = None


def _local_time(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_tushare_" + field) from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("timezone_required_tushare_" + field)
    return stamp.tz_convert(DECISION_TIMEZONE)


def _session(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_tushare_" + field) from exc
    if pd.isna(stamp):
        raise ValueError("invalid_tushare_" + field)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(DECISION_TIMEZONE).tz_localize(None)
    return stamp.normalize()


def _as_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_tushare_" + field)
    return value.strip()


def _is_declared_tushare_url(value: Any) -> bool:
    try:
        parsed = urlparse(_as_text(value, "canonical_url"))
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == _TUSHARE_HOST
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and parsed.path in ("", "/")
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )


def _instrument_id(ts_code: Any) -> str:
    code = _as_text(str(ts_code), "ts_code").upper()
    number, separator, market = code.partition(".")
    if separator != "." or len(number) != 6 or not number.isdigit() or market not in {"SZ", "SH", "BJ"}:
        raise ValueError("invalid_tushare_ts_code")
    return {"SZ": "sz.", "SH": "sh.", "BJ": "bj."}[market] + number


def _numeric(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _parse_response(raw: bytes) -> pd.DataFrame:
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError("tushare_raw_response_required")
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_tushare_json_response") from exc
    if not isinstance(payload, Mapping) or payload.get("code") != 0:
        raise ValueError("tushare_api_response_not_successful")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("tushare_response_data_missing")
    fields = data.get("fields")
    items = data.get("items")
    if not isinstance(fields, list) or not fields or not isinstance(items, list):
        raise ValueError("tushare_response_table_missing")
    normalized = tuple(_as_text(field, "response_field") for field in fields)
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate_tushare_response_field")
    rows: list[list[Any]] = []
    for item in items:
        if not isinstance(item, list) or len(item) != len(normalized):
            raise ValueError("invalid_tushare_response_row")
        rows.append(item)
    return pd.DataFrame(rows, columns=list(normalized))


def _validate_qualified_trial(value: Any) -> QualifiedProviderTrial:
    """Re-qualify instead of trusting a caller-constructed wrapper object."""
    if not isinstance(value, QualifiedProviderTrial):
        raise TushareThemeMarketError("tushare_trial_not_qualified")
    submission = value.submission
    if (
        submission.source_class != "theme_market"
        or submission.provider_id != TUSHARE_THEME_MARKET_PROVIDER_ID
        or not submission.automated_access.granted
        or not submission.raw_retention.granted
    ):
        raise TushareThemeMarketError("tushare_trial_not_qualified")
    try:
        requalified = qualify_provider_trial(submission)
    except (ProviderTrialError, TypeError, ValueError) as exc:
        raise TushareThemeMarketError("tushare_trial_not_qualified") from exc
    expected = tuple(item.requirement_id for item in acceptance_requirements("theme_market"))
    if (
        value.status != "qualified"
        or value.requirement_ids != expected
        or requalified.requirement_ids != expected
        or value.sha256 != requalified.sha256
    ):
        raise TushareThemeMarketError("tushare_trial_not_qualified")
    return requalified


def _response_metadata(response: TushareRawResponse, trial: QualifiedProviderTrial) -> dict[str, Any]:
    parameters = response.request_parameters
    if not isinstance(parameters, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in parameters.items()
    ):
        raise ValueError("invalid_tushare_request_parameters")
    return {
        "adapter_version": TUSHARE_THEME_MARKET_ADAPTER_VERSION,
        "provider_trial_id": trial.submission.trial_id,
        "provider_trial_sha256": trial.sha256,
        "endpoint": response.endpoint,
        "response_id": response.response_id,
        "request_parameters": dict(sorted(parameters.items())),
    }


def _record_response(
    *, ledger: ResearchSignalLedger, trial: QualifiedProviderTrial, response: TushareRawResponse
) -> _RecordedResponse:
    if not isinstance(response, TushareRawResponse):
        raise ValueError("tushare_raw_response_required")
    endpoint = _as_text(response.endpoint, "endpoint")
    response_id = _as_text(response.response_id, "response_id")
    if endpoint not in _ALLOWED_ENDPOINTS:
        raise ValueError("unsupported_tushare_endpoint")
    if not _is_declared_tushare_url(response.canonical_url):
        raise ValueError("undeclared_tushare_canonical_url")
    if _as_text(response.content_type, "content_type").split(";", 1)[0].lower() != _TUSHARE_CONTENT_TYPE:
        raise ValueError("unexpected_tushare_content_type")
    receipt = ledger.record_receipt(
        source_id=TUSHARE_THEME_MARKET_PROVIDER_ID,
        source_family="tushare.pro",
        origin_id="tushare:" + endpoint + ":" + response_id,
        canonical_url=response.canonical_url,
        raw=bytes(response.raw),
        content_type=response.content_type,
        published_at=response.published_at,
        first_captured_at=response.first_captured_at,
        available_at=response.available_at,
        availability_basis=_as_text(response.availability_basis, "availability_basis"),
        metadata=_response_metadata(response, trial),
    )
    receipt_id = str(receipt.get("canonical_receipt_id") or receipt["event_id"])
    evidence = ledger.export_receipt_evidence(receipt_id)
    try:
        frame = _parse_response(response.raw)
        parse_error = None
    except ValueError as exc:
        # The original response is still audit evidence.  A malformed body may
        # never become a signal input, but it must not vanish simply because it
        # could not be parsed into a table.
        frame = None
        parse_error = str(exc)
    return _RecordedResponse(
        response=response,
        frame=frame,
        receipt_id=receipt_id,
        receipt_hash=str(evidence["raw"]["sha256"]),
        receipt_evidence=evidence,
        parse_error=parse_error,
    )


def _receipt_is_on_time(record: _RecordedResponse, cutoff: pd.Timestamp) -> bool:
    try:
        response = record.response
        published = _local_time(response.published_at, "published_at")
        captured = _local_time(response.first_captured_at, "first_captured_at")
        available = _local_time(response.available_at, "available_at")
        receipt = record.receipt_evidence["receipt"]
        recorded = _local_time(receipt["recorded_at"], "recorded_at")
    except (KeyError, TypeError, ValueError):
        return False
    return published <= captured <= available <= cutoff and recorded <= cutoff


def _market_capture_is_on_time(record: _RecordedResponse, cutoff: pd.Timestamp) -> bool:
    captured = _local_time(record.response.first_captured_at, "first_captured_at")
    start = pd.Timestamp(cutoff.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(
        hours=_EOD_CAPTURE_START[0], minutes=_EOD_CAPTURE_START[1]
    )
    end = pd.Timestamp(cutoff.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(
        hours=_EOD_CAPTURE_END[0], minutes=_EOD_CAPTURE_END[1]
    )
    return start <= captured <= end


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


def _by_endpoint(records: Sequence[_RecordedResponse], endpoint: str) -> list[_RecordedResponse]:
    return [record for record in records if record.response.endpoint == endpoint]


def _parameter(record: _RecordedResponse, key: str) -> str | None:
    value = record.response.request_parameters.get(key)
    return value if isinstance(value, str) else None


def _one(records: Sequence[_RecordedResponse]) -> _RecordedResponse | None:
    return records[0] if len(records) == 1 else None


def _u0_reason(u0: Any, session: pd.Timestamp, cutoff: pd.Timestamp) -> str | None:
    if not isinstance(u0, UniverseSnapshot):
        return "tushare_u0_snapshot_required"
    if u0.status != "ok" or u0.pool_id != U0.pool_id:
        return "tushare_u0_not_available"
    if _session(u0.session, "u0_session") != session or pd.Timestamp(u0.decision_at) != cutoff:
        return "tushare_u0_decision_identity_mismatch"
    members = tuple(str(item) for item in u0.members)
    if not members or len(members) != len(set(members)):
        return "tushare_u0_membership_invalid"
    return None


def _date_string(session: pd.Timestamp) -> str:
    return session.strftime("%Y%m%d")


def _selected_daily(
    record: _RecordedResponse, *, session: pd.Timestamp, members: Sequence[str]
) -> tuple[pd.DataFrame | None, str | None]:
    frame = record.frame
    if frame is None:
        return None, "tushare_daily_response_invalid"
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount"}
    if not required.issubset(frame.columns) or _parameter(record, "trade_date") != _date_string(session):
        return None, "tushare_daily_response_invalid"
    if frame.trade_date.astype(str).ne(_date_string(session)).any() or frame.ts_code.duplicated().any():
        return None, "tushare_daily_response_invalid"
    source = frame.copy()
    try:
        source["instrument_id"] = source.ts_code.map(_instrument_id)
    except ValueError:
        return None, "tushare_daily_symbol_invalid"
    selected = source[source.instrument_id.isin(members)].copy()
    if len(selected) != len(members) or selected.instrument_id.duplicated().any():
        return None, "tushare_daily_u0_coverage_incomplete"
    numeric = selected.loc[:, ["open", "high", "low", "close", "pre_close", "amount"]].apply(
        lambda column: column.map(_numeric)
    )
    if not numeric.apply(lambda column: column.map(math.isfinite).all()).all() or (numeric <= 0).any().any():
        return None, "tushare_daily_value_invalid"
    if not ((numeric.high >= numeric.low) & (numeric.high >= numeric.close) & (numeric.low <= numeric.close)).all():
        return None, "tushare_daily_ohlc_invalid"
    selected.loc[:, ["open", "high", "low", "close", "pre_close", "amount"]] = numeric
    return selected.set_index("instrument_id").loc[list(members)].reset_index(), None


def _selected_limits(
    record: _RecordedResponse, *, session: pd.Timestamp, members: Sequence[str]
) -> tuple[pd.DataFrame | None, str | None]:
    frame = record.frame
    if frame is None:
        return None, "tushare_limit_response_invalid"
    required = {"ts_code", "trade_date", "up_limit", "down_limit"}
    if not required.issubset(frame.columns) or _parameter(record, "trade_date") != _date_string(session):
        return None, "tushare_limit_response_invalid"
    if frame.trade_date.astype(str).ne(_date_string(session)).any() or frame.ts_code.duplicated().any():
        return None, "tushare_limit_response_invalid"
    source = frame.copy()
    try:
        source["instrument_id"] = source.ts_code.map(_instrument_id)
    except ValueError:
        return None, "tushare_limit_symbol_invalid"
    selected = source[source.instrument_id.isin(members)].copy()
    if len(selected) != len(members) or selected.instrument_id.duplicated().any():
        return None, "tushare_limit_u0_coverage_incomplete"
    numeric = selected.loc[:, ["up_limit", "down_limit"]].apply(lambda column: column.map(_numeric))
    if not numeric.apply(lambda column: column.map(math.isfinite).all()).all() or (numeric <= 0).any().any():
        return None, "tushare_limit_value_invalid"
    if not (numeric.up_limit > numeric.down_limit).all():
        return None, "tushare_limit_range_invalid"
    selected.loc[:, ["up_limit", "down_limit"]] = numeric
    return selected.set_index("instrument_id").loc[list(members)].reset_index(), None


def _validate_board_events(
    record: _RecordedResponse, *, session: pd.Timestamp, daily: pd.DataFrame, limits: pd.DataFrame
) -> bool:
    frame = record.frame
    if frame is None:
        return False
    required = {"ts_code", "trade_date", "limit", "open_times"}
    if not required.issubset(frame.columns) or _parameter(record, "trade_date") != _date_string(session):
        return False
    if frame.trade_date.astype(str).ne(_date_string(session)).any() or frame.ts_code.duplicated().any():
        return False
    source = frame.copy()
    try:
        source["instrument_id"] = source.ts_code.map(_instrument_id)
    except ValueError:
        return False
    members = set(daily.instrument_id.astype(str))
    source = source[source.instrument_id.isin(members)].copy()
    if not source.empty:
        source["limit"] = source["limit"].astype(str).str.strip().str.upper()
        source["open_times"] = source["open_times"].map(_numeric)
        if not source.open_times.map(math.isfinite).all() or (source.open_times < 0).any():
            return False
    events = source.set_index("instrument_id") if not source.empty else pd.DataFrame()
    merged = daily.set_index("instrument_id").join(limits.set_index("instrument_id"), rsuffix="_limit")
    for code, row in merged.iterrows():
        touched = float(row.high) >= float(row.up_limit) - 0.005
        sealed = float(row.close) >= float(row.up_limit) - 0.005
        expected = "U" if sealed else "Z" if touched else None
        if code not in events.index:
            observed = None
            opens = 0.0
        else:
            event = events.loc[code]
            # D is a different, lower-limit event.  It cannot stand in for an
            # upper-limit or broken-board observation, but need not make a
            # non-touched stock appear to have touched the upper limit.
            observed = str(event["limit"]) if str(event["limit"]) in {"U", "Z"} else None
            opens = float(event["open_times"])
        if expected != observed or (observed == "Z" and opens < 1.0):
            return False
    return True


def _materialize_graph_input(
    *,
    records: Sequence[_RecordedResponse],
    u0: UniverseSnapshot,
    session: pd.Timestamp,
    prior: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[TushareThemeGraphInput | None, str | None]:
    reason = _u0_reason(u0, session, cutoff)
    if reason is not None:
        return None, reason
    members = tuple(str(item) for item in u0.members)
    index = _one(_by_endpoint(records, "ths_index"))
    if index is None or index.frame is None or _parameter(index, "type") != "N":
        return None, "tushare_theme_index_response_missing"
    if not {"ts_code", "name", "type"}.issubset(index.frame.columns):
        return None, "tushare_theme_index_response_invalid"
    themes = index.frame.copy()
    if themes.empty or themes.ts_code.duplicated().any() or themes["type"].astype(str).ne("N").any():
        return None, "tushare_theme_index_response_invalid"
    theme_ids = tuple(themes.ts_code.astype(str))
    if any(not value.strip() for value in theme_ids):
        return None, "tushare_theme_index_response_invalid"

    member_records = _by_endpoint(records, "ths_member")
    by_theme: dict[str, _RecordedResponse] = {}
    for record in member_records:
        theme = _parameter(record, "ts_code")
        if theme is None or theme not in theme_ids or theme in by_theme:
            return None, "tushare_theme_member_response_invalid"
        by_theme[theme] = record
    if set(by_theme) != set(theme_ids):
        return None, "tushare_theme_member_response_missing"

    active: dict[str, list[tuple[str, _RecordedResponse]]] = {code: [] for code in members}
    for theme in theme_ids:
        record = by_theme[theme]
        frame = record.frame
        if frame is None:
            return None, "tushare_theme_member_response_invalid"
        if not {"ts_code", "con_code"}.issubset(frame.columns) or frame.ts_code.astype(str).ne(theme).any():
            return None, "tushare_theme_member_response_invalid"
        if frame.con_code.duplicated().any():
            return None, "tushare_theme_member_response_invalid"
        try:
            instruments = frame.con_code.map(_instrument_id)
        except ValueError:
            return None, "tushare_theme_member_symbol_invalid"
        for code in instruments:
            if code in active:
                active[code].append((theme, record))

    daily_records = _by_endpoint(records, "daily")
    limit_records = _by_endpoint(records, "stk_limit")
    board = _one(_by_endpoint(records, "limit_list_d"))
    daily_by_session = { _parameter(record, "trade_date"): record for record in daily_records }
    limits_by_session = { _parameter(record, "trade_date"): record for record in limit_records }
    if (
        len(daily_records) != 2
        or len(daily_by_session) != 2
        or len(limit_records) != 2
        or len(limits_by_session) != 2
        or board is None
        or set(daily_by_session) != {_date_string(prior), _date_string(session)}
        or set(limits_by_session) != {_date_string(prior), _date_string(session)}
    ):
        return None, "tushare_market_response_missing"
    selected: dict[tuple[str, str], pd.DataFrame] = {}
    for which, target in (("prior", prior), ("current", session)):
        daily, daily_reason = _selected_daily(daily_by_session[_date_string(target)], session=target, members=members)
        limits, limit_reason = _selected_limits(limits_by_session[_date_string(target)], session=target, members=members)
        if daily_reason is not None:
            return None, daily_reason
        if limit_reason is not None:
            return None, limit_reason
        assert daily is not None and limits is not None
        selected[(which, "daily")] = daily
        selected[(which, "limit")] = limits
    if not _validate_board_events(
        board,
        session=session,
        daily=selected[("current", "daily")],
        limits=selected[("current", "limit")],
    ):
        return None, "tushare_board_event_inconsistent"

    # Keep the full raw-response dependency set on every derived input row.
    # ``theme_carrier_graph_v1`` independently rechecks these values against
    # the ledger before it carries them into a score result.  This prevents the
    # limit-price and broken-board receipts from disappearing merely because a
    # bar's direct row pointer is the matching daily response.
    # Keep ids and hashes in the same deterministic order: the graph scorer
    # verifies each pair against the ledger rather than trusting two unrelated
    # sets whose independent lexical order could silently mismatch.
    ordered_records = tuple(sorted(records, key=lambda record: record.receipt_id))
    all_receipt_ids = tuple(record.receipt_id for record in ordered_records)
    all_receipt_hashes = tuple(record.receipt_hash for record in ordered_records)
    bars: list[dict[str, Any]] = []
    for which, target in (("prior", prior), ("current", session)):
        daily_record = daily_by_session[_date_string(target)]
        limit_record = limits_by_session[_date_string(target)]
        daily = selected[(which, "daily")].set_index("instrument_id")
        limits = selected[(which, "limit")].set_index("instrument_id")
        daily_provenance = _provenance(daily_record)
        for code in members:
            price = daily.loc[code]
            bound = limits.loc[code]
            row: dict[str, Any] = {
                "session": target,
                "instrument_id": code,
                "open": float(price.open),
                "high": float(price.high),
                "low": float(price.low),
                "close": float(price.close),
                "reference_price": float(price.pre_close),
                "limit_up": float(bound.up_limit),
                # Tushare's daily and limit endpoints do not carry a tick
                # field.  This is the fixed A-share price-unit conversion used
                # only for equality tolerance, not a model feature.
                "price_tick": 0.01,
                "quality_status": "ok",
                "limit_source_receipt_id": limit_record.receipt_id,
                "limit_content_sha256": limit_record.receipt_hash,
                "supporting_receipt_ids": all_receipt_ids,
                "supporting_content_sha256s": all_receipt_hashes,
                **daily_provenance,
            }
            bars.append(row)

    edge_rows: list[dict[str, Any]] = []
    index_provenance = _provenance(index)
    for code in members:
        mappings = active[code]
        if mappings:
            themes_for_code = [theme for theme, _ in mappings]
            if len(themes_for_code) != len(set(themes_for_code)):
                return None, "tushare_theme_member_response_invalid"
            for theme, record in sorted(mappings, key=lambda item: item[0]):
                edge_rows.append(
                    {
                        "session": session,
                        "instrument_id": code,
                        "theme_id": theme,
                        "mapping_status": "active",
                        "supporting_receipt_ids": all_receipt_ids,
                        "supporting_content_sha256s": all_receipt_hashes,
                        **_provenance(record),
                    }
                )
        else:
            # The complete catalog plus every catalogued member response is
            # what establishes this zero-edge state.  The index receipt is the
            # canonical row anchor; ``receipt_ids`` retains all member receipts
            # so this is never mistaken for a locally invented primary theme.
            edge_rows.append(
                {
                    "session": session,
                    "instrument_id": code,
                    "theme_id": pd.NA,
                    "mapping_status": "known_no_eligible_theme",
                    "supporting_receipt_ids": all_receipt_ids,
                    "supporting_content_sha256s": all_receipt_hashes,
                    **index_provenance,
                }
            )
    graph_input = TushareThemeGraphInput(
        bars=pd.DataFrame(bars).sort_values(["session", "instrument_id"], kind="stable").reset_index(drop=True),
        theme_edges=pd.DataFrame(edge_rows)
        .sort_values(["instrument_id", "mapping_status", "theme_id"], kind="stable", na_position="last")
        .reset_index(drop=True),
        receipt_ids=all_receipt_ids,
        receipt_hashes=all_receipt_hashes,
    )
    return graph_input, None


def _abstain(
    *, session: pd.Timestamp, cutoff: pd.Timestamp, reason: str, records: Sequence[_RecordedResponse]
) -> TushareThemeMarketCapture:
    return TushareThemeMarketCapture(
        status="abstain",
        abstain_reason=reason,
        decision_session=session.date(),
        decision_at=cutoff,
        graph_input=None,
        receipt_ids=tuple(sorted({record.receipt_id for record in records})),
        receipt_hashes=tuple(sorted({record.receipt_hash for record in records})),
    )


def capture_tushare_theme_market_graph_v1(
    *,
    qualified_trial: QualifiedProviderTrial,
    ledger: ResearchSignalLedger,
    u0: UniverseSnapshot,
    decision_session: Any,
    prior_session: Any,
    decision_at: Any,
    responses: Sequence[TushareRawResponse],
) -> TushareThemeMarketCapture:
    """Persist supplied Tushare raw responses and create graph scorer input.

    Failure in any source leg is a whole-period abstention.  Crucially, the
    raw responses already supplied are still retained in the evidence ledger;
    a missing theme-member page is a data-quality result, not a reason to erase
    the pages that did arrive.  This function intentionally has no network,
    account, model, or order side effect.
    """
    trial = _validate_qualified_trial(qualified_trial)
    if not isinstance(ledger, ResearchSignalLedger):
        raise TushareThemeMarketError("tushare_research_ledger_required")
    session = _session(decision_session, "decision_session")
    prior = _session(prior_session, "prior_session")
    cutoff = _local_time(decision_at, "decision_at")
    if prior >= session:
        raise TushareThemeMarketError("tushare_prior_session_must_precede_decision_session")
    if not is_exact_decision_cutoff(session, cutoff):
        raise TushareThemeMarketError("tushare_decision_at_must_equal_session_1800_shanghai")
    if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence):
        raise TushareThemeMarketError("tushare_responses_required")
    response_ids: list[str] = []
    for response in responses:
        if not isinstance(response, TushareRawResponse):
            return _abstain(session=session, cutoff=cutoff, reason="tushare_raw_response_invalid", records=())
        response_ids.append(response.response_id)
    if len(response_ids) != len(set(response_ids)):
        return _abstain(session=session, cutoff=cutoff, reason="tushare_response_id_duplicate", records=())

    records: list[_RecordedResponse] = []
    try:
        for response in responses:
            records.append(_record_response(ledger=ledger, trial=trial, response=response))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _abstain(
            session=session, cutoff=cutoff, reason="tushare_raw_response_invalid", records=records
        )

    if any(record.parse_error is not None for record in records):
        return _abstain(session=session, cutoff=cutoff, reason="tushare_raw_response_invalid", records=records)

    if any(not _receipt_is_on_time(record, cutoff) for record in records):
        return _abstain(session=session, cutoff=cutoff, reason="tushare_receipt_late_for_decision", records=records)
    if any(
        record.response.endpoint in _MARKET_ENDPOINTS and not _market_capture_is_on_time(record, cutoff)
        for record in records
    ):
        return _abstain(
            session=session, cutoff=cutoff, reason="tushare_market_capture_outside_1530_1750", records=records
        )

    graph_input, reason = _materialize_graph_input(
        records=records, u0=u0, session=session, prior=prior, cutoff=cutoff
    )
    if reason is not None:
        return _abstain(session=session, cutoff=cutoff, reason=reason, records=records)
    assert graph_input is not None
    return TushareThemeMarketCapture(
        status="ok",
        abstain_reason=None,
        decision_session=session.date(),
        decision_at=cutoff,
        graph_input=graph_input,
        receipt_ids=graph_input.receipt_ids,
        receipt_hashes=graph_input.receipt_hashes,
    )


__all__ = [
    "TUSHARE_THEME_MARKET_ADAPTER_VERSION",
    "TUSHARE_THEME_MARKET_PROVIDER_ID",
    "TushareRawResponse",
    "TushareThemeGraphInput",
    "TushareThemeMarketCapture",
    "TushareThemeMarketError",
    "capture_tushare_theme_market_graph_v1",
]
