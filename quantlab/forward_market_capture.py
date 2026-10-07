"""Forward-only, receipt-backed public end-of-day market capture.

This is a research sidecar, not a data-feed replacement.  It deliberately
does not call the network or reuse parsed UI/cache output.  A scheduler must
give it the exact raw response bytes obtained from one declared public source.
The sidecar stores those bytes in :class:`ResearchSignalLedger`, derives only
from the captured bytes, and fail-closes on a late, partial, mixed-source, or
ambiguous capture.

The Eastmoney page parser follows the public ``push2 ... clist`` wire shape
already used by ``quantlab.data.eastmoney``.  It is kept here rather than
changing that UI-oriented adapter because the latter may return a parsed frame
without the raw response, receipt clock, or full-page completeness evidence
required by research.  The module imports neither account, router nor order
code and cannot create a target, order, trade, or execution instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.forward_market_transport import FailedRawEastmoneyCollection
from quantlab.forward_source_manifest import (
    EastmoneyEodSourceManifest,
    PreparedRawEastmoneyCollection,
    RawEastmoneyTransportAttempt,
    SourceManifestContractError,
    SourceManifestJournal,
)
from quantlab.research_universe import (
    DECISION_TIMEZONE,
    FORWARD_CONTRACT_VERSION,
    U0,
    UniverseSnapshot,
    build_liquidity_pool,
    canonical_forward_input_sha256,
    is_exact_decision_cutoff,
)
from quantlab.theme_carrier import ThemeCarrierResult, score_theme_carrier_v1


FORWARD_MARKET_CAPTURE_VERSION = "forward-market-capture-v1"
EOD_CAPTURE_OPEN_HOUR = 15
EOD_CAPTURE_OPEN_MINUTE = 30
# 17:50 is the last receipt time for a *same-session close-market snapshot*.
# It is not a transport stop time: late raw responses and failures are still
# ledger evidence, but cannot change that session's frozen 18:00 decision.
EOD_CAPTURE_CLOSE_HOUR = 17
EOD_CAPTURE_CLOSE_MINUTE = 50
EASTMONEY_SOURCE_ID = "eastmoney.push2.clist"
EASTMONEY_SOURCE_FAMILY = "eastmoney.push2"
EASTMONEY_CONTENT_TYPE = "application/json"
# This is a source *profile*, not merely a friendly source label.  Without
# pinning the endpoint and selection parameters, a caller could put arbitrary
# JSON behind an ``eastmoney`` source id and turn a changed cross-section into
# an apparently continuous forward series.
EASTMONEY_EOD_HOST = "push2.eastmoney.com"
EASTMONEY_EOD_PATH = "/api/qt/clist/get"
EASTMONEY_EOD_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81,s:2048"
EASTMONEY_EOD_REQUIRED_QUERY = {
    "po": "1",
    "np": "1",
    "fltt": "2",
    "invt": "2",
    "fid": "f6",
    "fs": EASTMONEY_EOD_FS,
}
EASTMONEY_EOD_REQUIRED_FIELDS = frozenset(
    {"f2", "f5", "f6", "f12", "f14", "f15", "f16", "f17", "f18", "f51"}
)
POOL_INPUT_SCHEMA = "quantlab.forward-pool-input.v1"
DERIVED_POOL_SOURCE_ID = "quantlab.forward-market-derivation"
DERIVED_POOL_SOURCE_FAMILY = "quantlab.forward-market-derivation"
_BAR_COLUMNS = (
    "session",
    "instrument_id",
    "name",
    "open",
    "high",
    "low",
    "close",
    "reference_price",
    "amount",
    "volume",
    "limit_up",
    "price_tick",
    "base_eligible",
    "asof_ok",
    "quality_status",
    "source_quality_status",
    "published_at",
    "first_captured_at",
    "available_at",
    "source_receipt_id",
    "source_receipt_hash",
    "source_receipt_written_at",
    "content_sha256",
    "capture_mode",
)
_POOL_BASE_COLUMNS = (
    "instrument_id",
    "base_eligible",
    "adv60_cny",
    "adv_observation_count",
    "valid_price_session_count",
    "asof_ok",
)
_POOL_COLUMNS = _POOL_BASE_COLUMNS + (
    "source_receipt_id",
    "source_receipt_hash",
    "source_published_at",
    "source_first_captured_at",
    "source_available_at",
    "capture_mode",
)
_MAPPING_COLUMNS = (
    "session",
    "instrument_id",
    "theme_id",
    "mapping_status",
    "published_at",
    "first_captured_at",
    "available_at",
    "source_receipt_id",
    "content_sha256",
    "capture_mode",
)


@dataclass(frozen=True)
class EastmoneyRawPage:
    """One raw page from the declared Eastmoney all-A endpoint.

    ``raw`` is intentionally not a parsed frame.  A transport can supply a
    timeout/error page, but that is evidence of an incomplete capture and
    cannot silently trigger a second provider or cached response.
    """

    page_number: int
    canonical_url: str
    raw: bytes | None
    captured_at: Any
    provider_published_at: Any | None = None
    fetch_error: str | None = None
    content_type: str = EASTMONEY_CONTENT_TYPE


@dataclass(frozen=True)
class MarketSnapshotCapture:
    """A complete all-A EOD snapshot, or a visible capture abstention."""

    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    session: date
    decision_at: pd.Timestamp
    bars: pd.DataFrame
    receipt_ids: tuple[str, ...]
    receipt_hashes: tuple[str, ...]
    capture_version: str = FORWARD_MARKET_CAPTURE_VERSION
    source_manifest_sha256: str | None = None
    source_manifest_event_sha256: str | None = None
    transport_receipt_id: str | None = None
    transport_receipt_hash: str | None = None
    ledger_recorded_at: str | None = None
    decision_eligibility: Literal["eligible_for_decision_at", "not_eligible_for_decision_at"] = (
        "not_eligible_for_decision_at"
    )


@dataclass(frozen=True)
class ForwardPoolRows:
    """Derived 60-ADV/253-price counts before receipt materialisation."""

    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    session: date
    rows: pd.DataFrame
    parent_receipt_ids: tuple[str, ...]


@dataclass(frozen=True)
class ForwardPoolInputPayload:
    """One raw, provider-owned pool-input response or a derived raw bundle."""

    canonical_url: str
    raw: bytes
    captured_at: Any
    source_id: str
    source_family: str
    origin_id: str
    provider_published_at: Any | None = None
    content_type: str = EASTMONEY_CONTENT_TYPE


@dataclass(frozen=True)
class ForwardPoolInputCapture:
    """Canonical pool rows whose single receipt is rechecked by U0 builder."""

    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    session: date
    decision_at: pd.Timestamp
    rows: pd.DataFrame
    receipt_id: str | None
    content_sha256: str | None
    # Unlike a close-market snapshot, a pool-input response has no 17:50
    # source window.  Its eligibility is nevertheless explicit so an
    # interrupted or delayed scheduler cannot pass an ``ok`` object into U0
    # construction without proving the named 18:00 decision boundary.
    decision_eligibility: Literal["eligible_for_decision_at", "not_eligible_for_decision_at"] = (
        "not_eligible_for_decision_at"
    )


@dataclass(frozen=True)
class ForwardPoolBuild:
    """The research-only U0 result plus its declared ledger scope."""

    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    universe: UniverseSnapshot | None
    scope: Mapping[str, Any] | None


@dataclass(frozen=True)
class ThemeMappingPayload:
    """Raw primary-theme mapping; callers must name its source explicitly."""

    canonical_url: str
    raw: bytes
    captured_at: Any
    source_id: str
    source_family: str
    origin_id: str
    provider_published_at: Any | None = None
    content_type: str = EASTMONEY_CONTENT_TYPE


@dataclass(frozen=True)
class ThemeMappingCapture:
    """A receipt-backed, direct primary-theme map or explicit abstention."""

    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    session: date
    decision_at: pd.Timestamp
    mappings: pd.DataFrame
    receipt_id: str | None
    content_sha256: str | None
    # Mapping evidence follows the normal publication/receipt/write cutoff,
    # not the close-market 17:50 collection window.  Keep its decision status
    # on the object so downstream scoring cannot infer eligibility from a
    # non-empty frame alone.
    decision_eligibility: Literal["eligible_for_decision_at", "not_eligible_for_decision_at"] = (
        "not_eligible_for_decision_at"
    )


@dataclass(frozen=True)
class ThemeLedgerRecord:
    """Result of writing the four-object ledger end of the theme pipeline.

    ``status='not_recorded'`` is a deliberate fail-closed state used when the
    declared scope itself cannot be proven to match the supplied U0 snapshot.
    In that case writing an abstention would falsely imply a valid scope.
    """

    status: Literal["observed", "abstained", "not_recorded"]
    reason: str | None
    entity_ids: tuple[str, ...]
    observation: Mapping[str, Any] | None


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _session(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("invalid_capture_session")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(DECISION_TIMEZONE).tz_localize(None)
    return stamp.normalize()


def _local_time(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("invalid_capture_time")
    if stamp.tzinfo is None:
        return stamp.tz_localize(DECISION_TIMEZONE)
    return stamp.tz_convert(DECISION_TIMEZONE)


def _decision_at(session: pd.Timestamp, value: Any) -> pd.Timestamp:
    decision = _local_time(value)
    if not is_exact_decision_cutoff(session, decision):
        raise ValueError("decision_at_must_equal_session_1800_shanghai")
    return decision


def _within_eod_window(captured_at: pd.Timestamp, decision_at: pd.Timestamp) -> bool:
    start = pd.Timestamp(decision_at.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(
        hours=EOD_CAPTURE_OPEN_HOUR, minutes=EOD_CAPTURE_OPEN_MINUTE
    )
    close = pd.Timestamp(decision_at.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(
        hours=EOD_CAPTURE_CLOSE_HOUR, minutes=EOD_CAPTURE_CLOSE_MINUTE
    )
    return start <= captured_at <= close


def _safe_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_" + field)
    return value.strip()


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _instrument_id(value: Any) -> str:
    code = str(value).strip().zfill(6)
    if len(code) != 6 or not code.isdigit():
        raise ValueError("invalid_a_share_code")
    if code.startswith(("4", "8", "92", "43", "83", "87")):
        return "bj." + code
    if code.startswith(("5", "6", "9")):
        return "sh." + code
    return "sz." + code


def _strict_url_query(query: str) -> dict[str, str] | None:
    """Parse a query without treating a literal ``+`` as a blank.

    Eastmoney's ``fs`` selector uses literal plus signs.  The normal form
    decoder is therefore the wrong provenance parser: it makes a raw URL and
    a differently encoded selector look alike.  Repeated keys are also
    rejected rather than allowing an HTTP client/server disagreement to choose
    which source constraint wins.
    """
    if not query:
        return None
    values: dict[str, str] = {}
    for item in query.split("&"):
        key, separator, value = item.partition("=")
        if not separator:
            return None
        decoded_key, decoded_value = unquote(key), unquote(value)
        if not decoded_key or decoded_key in values:
            return None
        values[decoded_key] = decoded_value
    return values


def _is_declared_eastmoney_eod_url(
    canonical_url: Any, *, page_number: int, page_size: int
) -> bool:
    """Whether a page names the one frozen all-A EOD source profile."""
    try:
        parsed = urlparse(_safe_text(canonical_url, "canonical_url"))
        query = _strict_url_query(parsed.query)
        if (
            parsed.scheme != "https"
            or parsed.hostname != EASTMONEY_EOD_HOST
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != EASTMONEY_EOD_PATH
            or parsed.params
            or parsed.fragment
            or query is None
        ):
            return False
        if query.get("pn") != str(page_number) or query.get("pz") != str(page_size):
            return False
        if any(query.get(key) != value for key, value in EASTMONEY_EOD_REQUIRED_QUERY.items()):
            return False
        fields = tuple(value for value in query.get("fields", "").split(",") if value)
        return (
            len(fields) == len(set(fields))
            and EASTMONEY_EOD_REQUIRED_FIELDS.issubset(set(fields))
        )
    except (TypeError, ValueError):
        return False


def _source_times(
    *, captured_at: Any, provider_published_at: Any | None
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Use a provider clock when supplied; otherwise record capture time honestly.

    A public quote response often has no independently verifiable publication
    time.  It is unsafe to backdate it.  The receipt therefore uses its actual
    HTTPS capture time as both the publication and availability lower bound and
    labels that basis so it cannot be misread as an exchange timestamp.
    """
    captured = _local_time(captured_at)
    if provider_published_at is None:
        return captured, captured, "forward_https_capture_no_provider_publish_clock"
    published = _local_time(provider_published_at)
    if published > captured:
        raise ValueError("provider_published_after_capture")
    return published, captured, "provider_published_clock_plus_https_capture"


def _record_raw_receipt(
    *,
    ledger: ResearchSignalLedger,
    source_id: str,
    source_family: str,
    origin_id: str,
    canonical_url: str,
    raw: bytes,
    content_type: str,
    captured_at: Any,
    provider_published_at: Any | None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, Mapping[str, Any]]:
    published, available, basis = _source_times(
        captured_at=captured_at, provider_published_at=provider_published_at
    )
    receipt = ledger.record_receipt(
        source_id=_safe_text(source_id, "source_id"),
        source_family=_safe_text(source_family, "source_family"),
        origin_id=_safe_text(origin_id, "origin_id"),
        canonical_url=_safe_text(canonical_url, "canonical_url"),
        raw=raw,
        content_type=_safe_text(content_type, "content_type"),
        published_at=published,
        first_captured_at=available,
        available_at=available,
        availability_basis=basis,
        metadata=metadata,
        # Do not pass a scheduler/capture timestamp at all.  The ledger's
        # public receipt API obtains its own runtime clock for every append,
        # so no caller can backdate a source response through this helper.
    )
    receipt_id = str(receipt.get("canonical_receipt_id") or receipt["event_id"])
    return receipt_id, ledger.export_receipt_evidence(receipt_id)


def _receipt_provenance(receipt_evidence: Mapping[str, Any]) -> dict[str, str]:
    receipt = receipt_evidence["receipt"]
    body = receipt["body"]
    raw = receipt_evidence["raw"]
    return dict(
        published_at=str(body["published_at"]),
        first_captured_at=str(body["first_captured_at"]),
        available_at=str(receipt["available_at"]),
        source_receipt_id=str(receipt_evidence["canonical_receipt_id"]),
        source_receipt_hash=str(raw["sha256"]),
        source_receipt_written_at=str(receipt["recorded_at"]),
        content_sha256=str(raw["sha256"]),
        capture_mode="forward_receipt",
    )


def _receipt_decision_eligibility_reason(
    receipt_evidence: Mapping[str, Any], cutoff: pd.Timestamp
) -> str | None:
    """Return why a recorded source response cannot support this decision.

    Raw evidence is intentionally recorded first.  Eligibility then checks the
    source's actual publication/receipt/availability clocks *and* the
    ledger-owned write time; a caller cannot use a supplied capture timestamp
    as an earlier stand-in for when the evidence reached the append-only
    ledger.
    """
    try:
        receipt = receipt_evidence["receipt"]
        body = receipt["body"]
        clocks = {
            "published": _local_time(body["published_at"]),
            "received": _local_time(body["first_captured_at"]),
            "available": _local_time(receipt["available_at"]),
            "written": _local_time(receipt["recorded_at"]),
        }
    except (KeyError, TypeError, ValueError):
        return "not_eligible_for_decision_at:receipt_time_unverifiable"
    for name in ("published", "received", "available", "written"):
        if clocks[name] > cutoff:
            return "not_eligible_for_decision_at:receipt_" + name + "_after_decision_cutoff"
    return None


def _parse_eastmoney_page(raw: bytes) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError("missing_raw_page_bytes")
    try:
        document = json.loads(bytes(raw).decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("eastmoney_page_not_json") from exc
    data = document.get("data") if isinstance(document, Mapping) else None
    diff = data.get("diff") if isinstance(data, Mapping) else None
    total = data.get("total") if isinstance(data, Mapping) else None
    if not isinstance(diff, list):
        raise ValueError("eastmoney_page_missing_diff")
    try:
        total_number = int(total)
    except (TypeError, ValueError) as exc:
        raise ValueError("eastmoney_page_invalid_total") from exc
    if total_number < 1:
        raise ValueError("eastmoney_page_invalid_total")
    parsed: list[dict[str, Any]] = []
    for item in diff:
        if not isinstance(item, Mapping):
            raise ValueError("eastmoney_page_invalid_row")
        code = _instrument_id(item.get("f12"))
        name = str(item.get("f14") or "").strip()
        close = _float(item.get("f2"))
        open_ = _float(item.get("f17"))
        high = _float(item.get("f15"))
        low = _float(item.get("f16"))
        reference = _float(item.get("f18"))
        amount = _float(item.get("f6"))
        volume = _float(item.get("f5"))
        # f51 is deliberately optional at parse time.  A pool can accumulate
        # receipts without it, but Theme Carrier V1 will abstain until a
        # declared source delivers verified limit-price evidence.
        limit_up = _float(item.get("f51"))
        valid_bar = all(math.isfinite(value) and value > 0 for value in (close, open_, high, low, reference))
        parsed.append(
            dict(
                instrument_id=code,
                name=name,
                open=open_,
                high=high,
                low=low,
                close=close,
                reference_price=reference,
                amount=amount,
                volume=volume,
                limit_up=limit_up,
                # A-share cash-equity tick rule is deliberately explicit and
                # narrow.  If a later source covers a market with another tick
                # rule, it must get a different capture profile/version.
                price_tick=0.01,
                base_eligible=bool(name) and "ST" not in name.upper() and "退" not in name,
                asof_ok=valid_bar,
            )
        )
    return total_number, parsed


def _snapshot_abstain(
    *, session: pd.Timestamp, decision_at: pd.Timestamp, reason: str,
    receipt_ids: Sequence[str] = (), receipt_hashes: Sequence[str] = (),
    source_manifest_sha256: str | None = None,
    source_manifest_event_sha256: str | None = None,
    transport_receipt_id: str | None = None,
    transport_receipt_hash: str | None = None,
    ledger_recorded_at: str | None = None,
) -> MarketSnapshotCapture:
    return MarketSnapshotCapture(
        status="abstain",
        abstain_reason=reason,
        session=session.date(),
        decision_at=decision_at,
        bars=_empty_frame(_BAR_COLUMNS),
        receipt_ids=tuple(sorted(set(str(value) for value in receipt_ids))),
        receipt_hashes=tuple(sorted(set(str(value) for value in receipt_hashes))),
        source_manifest_sha256=source_manifest_sha256,
        source_manifest_event_sha256=source_manifest_event_sha256,
        transport_receipt_id=transport_receipt_id,
        transport_receipt_hash=transport_receipt_hash,
        ledger_recorded_at=ledger_recorded_at,
        decision_eligibility="not_eligible_for_decision_at",
    )


def _capture_manifest_bound_eastmoney_eod_snapshot(
    *,
    ledger: ResearchSignalLedger,
    session: Any,
    decision_at: Any,
    pages: Sequence[EastmoneyRawPage],
    collection_id: str,
    transport_recorded_at: Any,
    page_size: int = 100,
    source_manifest_sha256: str,
    source_manifest_event_sha256: str,
    transport_receipt_id: str,
    transport_receipt_hash: str,
) -> MarketSnapshotCapture:
    """Materialise one already-registered Eastmoney all-A page set.

    This private helper only receives pages after the public bridge has
    verified an append-only source-manifest binding and recorded the raw
    transport envelope.  Keeping it private prevents callers from putting a
    parsed frame, cache, fallback page, or unregistered raw page directly into
    the research ledger.
    """
    if not isinstance(ledger, ResearchSignalLedger):
        raise TypeError("research_signal_ledger_required")
    snapshot_session = _session(session)
    try:
        cutoff = _decision_at(snapshot_session, decision_at)
    except ValueError as exc:
        return _snapshot_abstain(
            session=snapshot_session,
            decision_at=_local_time(decision_at),
            reason=str(exc),
        )
    try:
        transport_written_at = _local_time(transport_recorded_at)
    except (TypeError, ValueError):
        return _snapshot_abstain(
            session=snapshot_session,
            decision_at=cutoff,
            reason="invalid_transport_ledger_recorded_at",
        )
    if not isinstance(page_size, int) or page_size < 1:
        raise ValueError("invalid_page_size")
    try:
        collection = _safe_text(collection_id, "collection_id")
    except ValueError:
        return _snapshot_abstain(session=snapshot_session, decision_at=cutoff, reason="missing_collection_id")
    bound = dict(
        source_manifest_sha256=source_manifest_sha256,
        source_manifest_event_sha256=source_manifest_event_sha256,
        transport_receipt_id=transport_receipt_id,
        transport_receipt_hash=transport_receipt_hash,
        # This is only the transport-envelope receipt's write clock.  Page
        # receipts deliberately have their own recorded times (also carried
        # row-wise as ``source_receipt_written_at``); never treat this field
        # as a timestamp proxy for the whole page set.
        ledger_recorded_at=transport_written_at.isoformat(),
    )

    def abstain(
        reason: str,
        *,
        receipt_ids: Sequence[str] = (),
        receipt_hashes: Sequence[str] = (),
    ) -> MarketSnapshotCapture:
        return _snapshot_abstain(
            session=snapshot_session,
            decision_at=cutoff,
            reason=reason,
            receipt_ids=receipt_ids,
            receipt_hashes=receipt_hashes,
            **bound,
        )

    if not pages:
        return abstain("missing_page_set")

    # Validate the full source profile before accepting any page into the
    # ledger.  A later page from another endpoint must not leave a partly
    # accepted snapshot that looks like an Eastmoney collection.
    for page in pages:
        if not isinstance(page, EastmoneyRawPage):
            return abstain("invalid_eastmoney_page_object")
        if (
            not isinstance(page.page_number, int)
            or page.page_number < 1
            or not _is_declared_eastmoney_eod_url(
                page.canonical_url, page_number=page.page_number, page_size=page_size
            )
        ):
            return abstain("eastmoney_source_profile_mismatch")

    receipt_ids: list[str] = []
    receipt_hashes: list[str] = []
    parsed_pages: dict[int, tuple[int, list[dict[str, Any]], Mapping[str, Any], pd.Timestamp]] = {}
    had_fetch_error = False
    parse_error = False
    receipt_error = False
    late = False
    page_written_late = False
    duplicate_page_numbers = len({page.page_number for page in pages}) != len(pages)
    for page in sorted(pages, key=lambda item: item.page_number):
        if page.fetch_error or page.raw is None:
            had_fetch_error = True
            continue
        try:
            captured_at = _local_time(page.captured_at)
            # A collection can take long enough to cross the 18:00 decision
            # cutoff while its pages are being persisted.  ``record_receipt``
            # obtains the ledger-owned runtime clock for this individual
            # append; inspect the resulting receipt below rather than sharing
            # one bridge-entry timestamp across every page.
            receipt_id, evidence = _record_raw_receipt(
                ledger=ledger,
                source_id=EASTMONEY_SOURCE_ID,
                source_family=EASTMONEY_SOURCE_FAMILY,
                origin_id=f"{collection}:page:{page.page_number}",
                canonical_url=page.canonical_url,
                raw=bytes(page.raw),
                content_type=page.content_type,
                captured_at=captured_at,
                provider_published_at=page.provider_published_at,
                metadata={
                    "collection_id": collection,
                    "page_number": page.page_number,
                    "source_manifest_sha256": source_manifest_sha256,
                    "source_manifest_event_sha256": source_manifest_event_sha256,
                    "transport_receipt_id": transport_receipt_id,
                    "transport_receipt_hash": transport_receipt_hash,
                    "received_at": captured_at.isoformat(),
                },
            )
            receipt_ids.append(receipt_id)
            receipt_hashes.append(str(evidence["raw"]["sha256"]))
            total, rows = _parse_eastmoney_page(bytes(page.raw))
            parsed_pages[page.page_number] = (total, rows, evidence, captured_at)
            late = late or not _within_eod_window(captured_at, cutoff)
            page_written_late = page_written_late or _local_time(evidence["receipt"]["recorded_at"]) > cutoff
        except (KeyError, TypeError, ValueError, OSError):
            receipt_error = True

    if had_fetch_error:
        return abstain("page_fetch_failed", receipt_ids=receipt_ids, receipt_hashes=receipt_hashes)
    if duplicate_page_numbers:
        return abstain("duplicate_page_number", receipt_ids=receipt_ids, receipt_hashes=receipt_hashes)
    if receipt_error or not parsed_pages:
        return abstain("page_receipt_or_parse_failed", receipt_ids=receipt_ids, receipt_hashes=receipt_hashes)
    totals = {item[0] for item in parsed_pages.values()}
    if len(totals) != 1:
        return abstain("inconsistent_page_total", receipt_ids=receipt_ids, receipt_hashes=receipt_hashes)
    total = next(iter(totals))
    expected_pages = set(range(1, math.ceil(total / page_size) + 1))
    if set(parsed_pages) != expected_pages:
        return abstain("incomplete_page_set", receipt_ids=receipt_ids, receipt_hashes=receipt_hashes)
    if late:
        return abstain(
            "not_eligible_for_decision_at:close_page_received_outside_window",
            receipt_ids=receipt_ids,
            receipt_hashes=receipt_hashes,
        )
    if transport_written_at > cutoff or page_written_late:
        return abstain(
            "not_eligible_for_decision_at:receipt_written_after_decision_cutoff",
            receipt_ids=receipt_ids,
            receipt_hashes=receipt_hashes,
        )

    records: list[dict[str, Any]] = []
    for page_number in sorted(parsed_pages):
        _, page_rows, evidence, _ = parsed_pages[page_number]
        provenance = _receipt_provenance(evidence)
        records.extend(
            # ``ok`` means this row survived the sidecar's exact raw-page,
            # single-source, receipt and cutoff validation.  It does not
            # elevate the public source into a qualified PIT provider; that
            # remains explicit in ``source_quality_status``.
            dict(
                session=snapshot_session,
                quality_status="ok",
                source_quality_status="unverified_public_eod",
                **row,
                **provenance,
            )
            for row in page_rows
        )
    if len(records) != total:
        return abstain("page_row_count_mismatch", receipt_ids=receipt_ids, receipt_hashes=receipt_hashes)
    bars = pd.DataFrame(records)
    if bars.instrument_id.duplicated().any():
        return abstain("duplicate_instrument_id", receipt_ids=receipt_ids, receipt_hashes=receipt_hashes)
    bars = bars.loc[:, _BAR_COLUMNS].sort_values("instrument_id", kind="stable").reset_index(drop=True)
    return MarketSnapshotCapture(
        status="ok",
        abstain_reason=None,
        session=snapshot_session.date(),
        decision_at=cutoff,
        bars=bars,
        receipt_ids=tuple(sorted(set(receipt_ids))),
        receipt_hashes=tuple(sorted(set(receipt_hashes))),
        **bound,
        decision_eligibility="eligible_for_decision_at",
    )


def _transport_attempts_from_collection(
    collection: PreparedRawEastmoneyCollection | FailedRawEastmoneyCollection,
) -> tuple[EastmoneyRawPage, ...]:
    """Return only raw attempt objects carried by a transport result.

    The public bridge deliberately accepts a transport *result*, never an
    arbitrary sequence of pages.  Failed collections retain their successful
    prefix and explicit failed page(s), so they can be persisted before the
    close-market snapshot abstains.
    """
    if isinstance(collection, PreparedRawEastmoneyCollection):
        pages = collection.capture_pages
        if not isinstance(pages, tuple):
            raise ValueError("prepared_transport_pages_not_tuple")
        return tuple(pages)
    if isinstance(collection, FailedRawEastmoneyCollection):
        return tuple(
            EastmoneyRawPage(
                page_number=attempt.page_number,
                canonical_url=attempt.canonical_url,
                raw=attempt.raw,
                captured_at=attempt.captured_at,
                provider_published_at=attempt.provider_published_at,
                fetch_error=attempt.fetch_error,
                content_type=attempt.content_type,
            )
            for attempt in collection.attempts
        )
    raise TypeError("registered_transport_collection_required")


def _transport_attempt_evidence(
    pages: Sequence[EastmoneyRawPage],
    *,
    failure_reported_at: Any | None = None,
) -> tuple[list[dict[str, Any]], pd.Timestamp, pd.Timestamp]:
    """Build immutable transport evidence without treating it as a signal.

    A failed transport can legitimately have no attempts when it detects a
    registration/clock boundary before opening page one.  That state must be
    retained as an empty failure envelope, anchored to the transport's own
    reported clock—not made to look like an HTTP response or discarded.
    """
    if not pages:
        if failure_reported_at is None:
            raise ValueError("transport_attempts_required")
        reported = _local_time(failure_reported_at)
        return [], reported, reported
    result: list[dict[str, Any]] = []
    received: list[pd.Timestamp] = []
    for page in pages:
        if not isinstance(page, EastmoneyRawPage):
            raise ValueError("invalid_eastmoney_page_object")
        received_at = _local_time(page.captured_at)
        received.append(received_at)
        raw = page.raw
        result.append(
            dict(
                page_number=page.page_number,
                canonical_url=page.canonical_url,
                received_at=received_at.isoformat(),
                provider_published_at=(
                    None if page.provider_published_at is None else _local_time(page.provider_published_at).isoformat()
                ),
                raw_sha256=None if raw is None else hashlib.sha256(bytes(raw)).hexdigest(),
                byte_count=0 if raw is None else len(raw),
                fetch_error=page.fetch_error,
                content_type=page.content_type,
            )
        )
    return result, min(received), max(received)


def capture_eastmoney_eod_snapshot(
    *,
    ledger: ResearchSignalLedger,
    journal: SourceManifestJournal,
    collection: PreparedRawEastmoneyCollection | FailedRawEastmoneyCollection,
) -> MarketSnapshotCapture:
    """Bridge one registered transport outcome into raw receipts and a snapshot.

    This is the sole V1 materialisation path.  It first verifies the exact
    source-manifest registration, then persists the transport envelope and any
    successful raw pages using the ledger-owned clock.  A late, failed, or
    partial collection therefore remains auditable while becoming ineligible
    only for its named ``decision_at``; no capture timestamp is passed off as
    the later ledger-write time.
    """
    if not isinstance(ledger, ResearchSignalLedger):
        raise TypeError("research_signal_ledger_required")
    if not isinstance(journal, SourceManifestJournal):
        raise TypeError("source_manifest_journal_required")
    if not isinstance(collection, (PreparedRawEastmoneyCollection, FailedRawEastmoneyCollection)):
        raise TypeError("registered_transport_collection_required")
    try:
        session = _session(collection.session)
        cutoff = _decision_at(session, collection.decision_at)
        collection_id = _safe_text(collection.collection_id, "collection_id")
        manifest_sha = _safe_text(collection.manifest_sha256, "source_manifest_sha256")
        manifest_event_sha = _safe_text(collection.manifest_event_sha256, "source_manifest_event_sha256")
        registration = journal.verify_registration_binding(
            manifest_sha256=manifest_sha,
            manifest_event_sha256=manifest_event_sha,
        )
        manifest = EastmoneyEodSourceManifest.from_mapping(registration.manifest)
        pages = _transport_attempts_from_collection(collection)
        failure_reported_at = (
            collection.reported_at if isinstance(collection, FailedRawEastmoneyCollection) else None
        )
        attempts, first_received_at, available_at = _transport_attempt_evidence(
            pages,
            failure_reported_at=failure_reported_at,
        )
    except (AttributeError, KeyError, TypeError, ValueError, SourceManifestContractError):
        # No unregistered page can become a source receipt.  The caller must
        # hand the bridge a Prepared/Failed result tied to the journal.
        try:
            session = _session(collection.session)
            cutoff = _decision_at(session, collection.decision_at)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("invalid_registered_transport_collection") from None
        return _snapshot_abstain(
            session=session,
            decision_at=cutoff,
            reason="manifest_binding_unverified",
            source_manifest_sha256=getattr(collection, "manifest_sha256", None),
            source_manifest_event_sha256=getattr(collection, "manifest_event_sha256", None),
        )

    transport_status = "failed" if isinstance(collection, FailedRawEastmoneyCollection) else "prepared"
    failure_reason = collection.failure_reason if isinstance(collection, FailedRawEastmoneyCollection) else None
    try:
        transport_event = ledger.record_transport_evidence(
            collection_id=collection_id,
            source_manifest_sha256=manifest_sha,
            source_manifest_event_sha256=manifest_event_sha,
            session=session,
            decision_at=cutoff,
            status=transport_status,
            failure_reason=failure_reason,
            attempts=attempts,
            first_captured_at=first_received_at,
            available_at=available_at,
            reported_at=failure_reported_at,
        )
        transport_evidence = ledger.export_receipt_evidence(transport_event["event_id"])
    except (KeyError, TypeError, ValueError, OSError):
        return _snapshot_abstain(
            session=session,
            decision_at=cutoff,
            reason="transport_evidence_recording_failed",
            source_manifest_sha256=manifest_sha,
            source_manifest_event_sha256=manifest_event_sha,
        )

    transport_receipt_id = str(transport_evidence["canonical_receipt_id"])
    transport_receipt_hash = str(transport_evidence["raw"]["sha256"])
    if transport_status == "failed" and not pages:
        return _snapshot_abstain(
            session=session,
            decision_at=cutoff,
            reason="not_eligible_for_decision_at:transport_failed_before_page_receipt",
            source_manifest_sha256=manifest_sha,
            source_manifest_event_sha256=manifest_event_sha,
            transport_receipt_id=transport_receipt_id,
            transport_receipt_hash=transport_receipt_hash,
            ledger_recorded_at=str(transport_evidence["receipt"]["recorded_at"]),
        )

    return _capture_manifest_bound_eastmoney_eod_snapshot(
        ledger=ledger,
        session=session,
        decision_at=cutoff,
        pages=pages,
        collection_id=collection_id,
        transport_recorded_at=transport_evidence["receipt"]["recorded_at"],
        page_size=manifest.page_size,
        source_manifest_sha256=manifest_sha,
        source_manifest_event_sha256=manifest_event_sha,
        transport_receipt_id=transport_receipt_id,
        transport_receipt_hash=transport_receipt_hash,
    )


def _history_abstain(session: pd.Timestamp, reason: str) -> ForwardPoolRows:
    return ForwardPoolRows(
        status="abstain",
        abstain_reason=reason,
        session=session.date(),
        rows=_empty_frame(_POOL_BASE_COLUMNS),
        parent_receipt_ids=(),
    )


def derive_pool_rows_from_forward_history(
    *, captures: Sequence[MarketSnapshotCapture], session: Any, exchange_sessions: Sequence[Any]
) -> ForwardPoolRows:
    """Derive U0's ADV60 and 253-session price-history counts from receipts.

    No historical panel, cached daily file, or later provider request is
    allowed as a bootstrap.  Until 253 contiguous captured exchange sessions
    exist, this returns an explicit abstention rather than pretending that
    today's provider can prove yesterday's information set.
    """
    target = _session(session)
    try:
        calendar = tuple(_session(value) for value in exchange_sessions)
    except (TypeError, ValueError):
        return _history_abstain(target, "invalid_exchange_session_calendar")
    if not calendar or len(set(calendar)) != len(calendar) or tuple(sorted(calendar)) != calendar or target not in calendar:
        return _history_abstain(target, "invalid_exchange_session_calendar")
    position = calendar.index(target)
    needed = 253
    if position + 1 < needed:
        return _history_abstain(target, "insufficient_forward_price_history")
    expected = calendar[position - needed + 1 : position + 1]
    by_session: dict[pd.Timestamp, MarketSnapshotCapture] = {}
    for capture in captures:
        try:
            captured_session = _session(capture.session)
        except (AttributeError, TypeError, ValueError):
            return _history_abstain(target, "invalid_forward_snapshot_history")
        if captured_session in by_session:
            return _history_abstain(target, "duplicate_forward_snapshot_session")
        by_session[captured_session] = capture
    if any(item not in by_session for item in expected):
        return _history_abstain(target, "insufficient_forward_price_history")
    selected = [by_session[item] for item in expected]
    if any(capture.status != "ok" or capture.bars.empty for capture in selected):
        return _history_abstain(target, "incomplete_forward_snapshot_history")
    if any(set(_BAR_COLUMNS) - set(capture.bars.columns) for capture in selected):
        return _history_abstain(target, "invalid_forward_snapshot_schema")
    if any(capture.bars.instrument_id.duplicated().any() for capture in selected):
        return _history_abstain(target, "duplicate_forward_snapshot_instrument")

    current = selected[-1].bars.set_index("instrument_id")
    frames = [capture.bars.set_index("instrument_id") for capture in selected]
    adv_frames = frames[-60:]
    records: list[dict[str, Any]] = []
    for code, current_row in current.iterrows():
        amounts = np.asarray([_float(frame.at[code, "amount"]) if code in frame.index else np.nan for frame in adv_frames])
        closes = np.asarray([_float(frame.at[code, "close"]) if code in frame.index else np.nan for frame in frames])
        valid_amount = np.isfinite(amounts) & (amounts >= 0)
        valid_close = np.isfinite(closes) & (closes > 0)
        records.append(
            dict(
                instrument_id=str(code),
                base_eligible=bool(current_row.base_eligible),
                adv60_cny=float(np.nanmean(amounts[valid_amount])) if valid_amount.any() else float("nan"),
                adv_observation_count=int(valid_amount.sum()),
                valid_price_session_count=int(valid_close.sum()),
                asof_ok=bool(current_row.asof_ok),
            )
        )
    parent_ids = tuple(sorted({receipt_id for capture in selected for receipt_id in capture.receipt_ids}))
    return ForwardPoolRows(
        status="ok",
        abstain_reason=None,
        session=target.date(),
        rows=pd.DataFrame(records, columns=list(_POOL_BASE_COLUMNS)).sort_values("instrument_id", kind="stable").reset_index(drop=True),
        parent_receipt_ids=parent_ids,
    )


def _pool_input_abstain(
    session: pd.Timestamp,
    decision_at: pd.Timestamp,
    reason: str,
    *,
    receipt_id: str | None = None,
    content_sha256: str | None = None,
) -> ForwardPoolInputCapture:
    """Return an explicit pool-input abstention without discarding raw proof.

    Once a source response has reached the ledger, a late or unparseable
    payload is still material evidence.  Keep its canonical receipt pointer
    on the abstention while marking it unavailable for this decision.
    """
    return ForwardPoolInputCapture(
        status="abstain",
        abstain_reason=reason,
        session=session.date(),
        decision_at=decision_at,
        rows=_empty_frame(_POOL_COLUMNS),
        receipt_id=receipt_id,
        content_sha256=content_sha256,
        decision_eligibility="not_eligible_for_decision_at",
    )


def _parse_pool_input(raw: bytes) -> pd.DataFrame:
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pool_input_not_json") from exc
    if not isinstance(document, Mapping) or document.get("schema") != POOL_INPUT_SCHEMA:
        raise ValueError("pool_input_schema_mismatch")
    source_rows = document.get("rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("pool_input_rows_missing")
    rows: list[dict[str, Any]] = []
    for item in source_rows:
        if not isinstance(item, Mapping):
            raise ValueError("pool_input_invalid_row")
        missing = set(_POOL_BASE_COLUMNS) - set(item)
        if missing:
            raise ValueError("pool_input_missing_columns:" + ",".join(sorted(missing)))
        rows.append({column: item[column] for column in _POOL_BASE_COLUMNS})
    frame = pd.DataFrame(rows, columns=list(_POOL_BASE_COLUMNS))
    frame["instrument_id"] = frame.instrument_id.astype(str)
    if frame.instrument_id.eq("").any() or frame.instrument_id.duplicated().any():
        raise ValueError("pool_input_duplicate_or_empty_instrument")
    return frame.sort_values("instrument_id", kind="stable").reset_index(drop=True)


def capture_forward_pool_input(
    *, ledger: ResearchSignalLedger, session: Any, decision_at: Any, payload: ForwardPoolInputPayload
) -> ForwardPoolInputCapture:
    """Record one raw pool-input response and attach its canonical receipt to rows."""
    if not isinstance(ledger, ResearchSignalLedger):
        raise TypeError("research_signal_ledger_required")
    snapshot_session = _session(session)
    try:
        cutoff = _decision_at(snapshot_session, decision_at)
    except ValueError as exc:
        return _pool_input_abstain(snapshot_session, _local_time(decision_at), str(exc))
    try:
        receipt_id, evidence = _record_raw_receipt(
            ledger=ledger,
            source_id=payload.source_id,
            source_family=payload.source_family,
            origin_id=payload.origin_id,
            canonical_url=payload.canonical_url,
            raw=payload.raw,
            content_type=payload.content_type,
            captured_at=payload.captured_at,
            provider_published_at=payload.provider_published_at,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return _pool_input_abstain(snapshot_session, cutoff, "pool_input_receipt_failed")
    provenance = _receipt_provenance(evidence)
    eligibility_reason = _receipt_decision_eligibility_reason(evidence, cutoff)
    if eligibility_reason is not None:
        return _pool_input_abstain(
            snapshot_session,
            cutoff,
            eligibility_reason,
            receipt_id=receipt_id,
            content_sha256=provenance["content_sha256"],
        )
    try:
        rows = _parse_pool_input(bytes(payload.raw))
    except (TypeError, ValueError):
        return _pool_input_abstain(
            snapshot_session,
            cutoff,
            "pool_input_parse_failed",
            receipt_id=receipt_id,
            content_sha256=provenance["content_sha256"],
        )
    rows["source_receipt_id"] = provenance["source_receipt_id"]
    rows["source_receipt_hash"] = provenance["source_receipt_hash"]
    rows["source_published_at"] = provenance["published_at"]
    rows["source_first_captured_at"] = provenance["first_captured_at"]
    rows["source_available_at"] = provenance["available_at"]
    rows["capture_mode"] = provenance["capture_mode"]
    rows = rows.loc[:, _POOL_COLUMNS]
    return ForwardPoolInputCapture(
        status="ok",
        abstain_reason=None,
        session=snapshot_session.date(),
        decision_at=cutoff,
        rows=rows,
        receipt_id=receipt_id,
        content_sha256=provenance["content_sha256"],
        decision_eligibility="eligible_for_decision_at",
    )


def capture_derived_pool_input_from_history(
    *,
    ledger: ResearchSignalLedger,
    derived: ForwardPoolRows,
    session: Any,
    decision_at: Any,
    captured_at: Any,
) -> ForwardPoolInputCapture:
    """Materialise derived forward history as one clearly labelled raw bundle.

    ``build_liquidity_pool`` has one receipt pointer per input row.  The
    bundle embeds every parent receipt id/hash in its immutable raw JSON, then
    becomes that one pointer.  It is an explicit derivation receipt, never
    relabelled as a provider response.
    """
    snapshot_session = _session(session)
    cutoff = _local_time(decision_at)
    if derived.status != "ok" or derived.session != snapshot_session.date():
        return _pool_input_abstain(snapshot_session, cutoff, derived.abstain_reason or "pool_rows_not_available")
    parents: list[dict[str, str]] = []
    try:
        for receipt_id in derived.parent_receipt_ids:
            evidence = ledger.export_receipt_evidence(receipt_id)
            receipt = evidence["receipt"]
            raw = evidence["raw"]
            if evidence["canonical_receipt_id"] != receipt_id or receipt["status"] != "accepted":
                return _pool_input_abstain(snapshot_session, cutoff, "parent_receipt_not_canonical")
            if _local_time(receipt["recorded_at"]) > cutoff or _local_time(receipt["available_at"]) > cutoff:
                return _pool_input_abstain(snapshot_session, cutoff, "parent_receipt_late")
            parents.append(dict(receipt_id=receipt_id, raw_sha256=str(raw["sha256"])))
    except (KeyError, TypeError, ValueError):
        return _pool_input_abstain(snapshot_session, cutoff, "parent_receipt_unavailable")
    rows = [
        {
            column: (
                bool(row[column]) if column in ("base_eligible", "asof_ok") else
                int(row[column]) if column in ("adv_observation_count", "valid_price_session_count") else
                float(row[column]) if column == "adv60_cny" else str(row[column])
            )
            for column in _POOL_BASE_COLUMNS
        }
        for _, row in derived.rows.iterrows()
    ]
    raw = json.dumps(
        dict(
            schema=POOL_INPUT_SCHEMA,
            derivation_version=FORWARD_MARKET_CAPTURE_VERSION,
            session=snapshot_session.date().isoformat(),
            decision_at=cutoff.isoformat(),
            parent_receipts=parents,
            rows=rows,
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return capture_forward_pool_input(
        ledger=ledger,
        session=snapshot_session,
        decision_at=cutoff,
        payload=ForwardPoolInputPayload(
            canonical_url="urn:quantlab:forward-market-derivation:" + snapshot_session.date().isoformat(),
            raw=raw,
            captured_at=captured_at,
            source_id=DERIVED_POOL_SOURCE_ID,
            source_family=DERIVED_POOL_SOURCE_FAMILY,
            origin_id="forward-history:" + snapshot_session.date().isoformat(),
        ),
    )


def build_forward_u0_pool(
    *,
    ledger: ResearchSignalLedger,
    materialized: ForwardPoolInputCapture,
    session: Any,
    decision_at: Any,
    exchange_sessions: Sequence[Any],
    cadence_anchor: Any,
    scope_id: str,
    now: Any | None = None,
) -> ForwardPoolBuild:
    """Build and declare U0 only; this never registers an account experiment."""
    snapshot_session = _session(session)
    cutoff = _local_time(decision_at)
    if materialized.status != "ok" or materialized.rows.empty:
        return ForwardPoolBuild("abstain", materialized.abstain_reason or "pool_input_not_available", None, None)
    if materialized.decision_eligibility != "eligible_for_decision_at":
        return ForwardPoolBuild("abstain", "pool_input_not_eligible_for_decision_at", None, None)
    if materialized.session != snapshot_session.date() or materialized.decision_at != cutoff:
        return ForwardPoolBuild("abstain", "pool_input_session_or_cutoff_mismatch", None, None)
    try:
        universe = build_liquidity_pool(
            session=snapshot_session,
            decision_at=cutoff,
            rows=materialized.rows,
            spec=U0,
            cadence_anchor=cadence_anchor,
            exchange_sessions=exchange_sessions,
            input_hash=canonical_forward_input_sha256(materialized.rows),
            ledger=ledger,
        )
    except (TypeError, ValueError):
        return ForwardPoolBuild("abstain", "pool_input_not_canonical", None, None)
    if universe.status != "ok":
        return ForwardPoolBuild("abstain", universe.abstain_reason, universe, None)
    try:
        scope = ledger.declare_scope(
            _safe_text(scope_id, "scope_id"),
            cutoff,
            universe.members,
            definition_version=FORWARD_CONTRACT_VERSION,
            source_hash=universe.membership_hash,
            # Scope declaration is a live ledger write.  Do not accept a
            # scheduler-supplied earlier clock as proof that it existed by the
            # decision cutoff; the ledger's injected runtime clock is the
            # only source of that timestamp.
            now=ledger.runtime_now(),
        )
    except (TypeError, ValueError):
        return ForwardPoolBuild("abstain", "scope_not_declarable_by_cutoff", universe, None)
    return ForwardPoolBuild("ok", None, universe, scope)


def _mapping_abstain(
    session: pd.Timestamp,
    decision_at: pd.Timestamp,
    reason: str,
    *,
    receipt_id: str | None = None,
    content_sha256: str | None = None,
) -> ThemeMappingCapture:
    """Return a mapping abstention while retaining any recorded raw receipt."""
    return ThemeMappingCapture(
        status="abstain",
        abstain_reason=reason,
        session=session.date(),
        decision_at=decision_at,
        mappings=_empty_frame(_MAPPING_COLUMNS),
        receipt_id=receipt_id,
        content_sha256=content_sha256,
        decision_eligibility="not_eligible_for_decision_at",
    )


def _parse_primary_theme_map(raw: bytes) -> pd.DataFrame:
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("theme_mapping_not_json") from exc
    source_rows = document.get("rows") if isinstance(document, Mapping) else None
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("theme_mapping_rows_missing")
    rows: list[dict[str, str]] = []
    for item in source_rows:
        if not isinstance(item, Mapping):
            raise ValueError("theme_mapping_invalid_row")
        code = item.get("instrument_id", item.get("code"))
        theme_id = item.get("theme_id")
        status = item.get("mapping_status", "active")
        if not isinstance(theme_id, str) or not theme_id.strip() or status != "active":
            raise ValueError("theme_mapping_ambiguous_or_inactive")
        rows.append(dict(instrument_id=_instrument_id(code), theme_id=theme_id.strip(), mapping_status="active"))
    frame = pd.DataFrame(rows)
    if frame.instrument_id.duplicated().any():
        raise ValueError("theme_mapping_duplicate_instrument")
    return frame.sort_values("instrument_id", kind="stable").reset_index(drop=True)


def capture_primary_theme_mapping(
    *, ledger: ResearchSignalLedger, session: Any, decision_at: Any, payload: ThemeMappingPayload
) -> ThemeMappingCapture:
    """Capture direct primary-theme evidence without an inferred/default map."""
    if not isinstance(ledger, ResearchSignalLedger):
        raise TypeError("research_signal_ledger_required")
    snapshot_session = _session(session)
    try:
        cutoff = _decision_at(snapshot_session, decision_at)
    except ValueError as exc:
        return _mapping_abstain(snapshot_session, _local_time(decision_at), str(exc))
    try:
        receipt_id, evidence = _record_raw_receipt(
            ledger=ledger,
            source_id=payload.source_id,
            source_family=payload.source_family,
            origin_id=payload.origin_id,
            canonical_url=payload.canonical_url,
            raw=payload.raw,
            content_type=payload.content_type,
            captured_at=payload.captured_at,
            provider_published_at=payload.provider_published_at,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return _mapping_abstain(snapshot_session, cutoff, "theme_mapping_receipt_failed")
    provenance = _receipt_provenance(evidence)
    eligibility_reason = _receipt_decision_eligibility_reason(evidence, cutoff)
    if eligibility_reason is not None:
        return _mapping_abstain(
            snapshot_session,
            cutoff,
            eligibility_reason,
            receipt_id=receipt_id,
            content_sha256=provenance["content_sha256"],
        )
    try:
        frame = _parse_primary_theme_map(bytes(payload.raw))
    except (TypeError, ValueError):
        return _mapping_abstain(
            snapshot_session,
            cutoff,
            "theme_mapping_parse_failed",
            receipt_id=receipt_id,
            content_sha256=provenance["content_sha256"],
        )
    frame["session"] = snapshot_session
    for key in ("published_at", "first_captured_at", "available_at", "source_receipt_id", "content_sha256", "capture_mode"):
        frame[key] = provenance[key]
    frame = frame.loc[:, _MAPPING_COLUMNS]
    return ThemeMappingCapture(
        status="ok",
        abstain_reason=None,
        session=snapshot_session.date(),
        decision_at=cutoff,
        mappings=frame,
        receipt_id=receipt_id,
        content_sha256=provenance["content_sha256"],
        decision_eligibility="eligible_for_decision_at",
    )


def _theme_abstain(*, session: Any, decision_at: Any, reason: str) -> ThemeCarrierResult:
    day = _session(session)
    cutoff = _local_time(decision_at)
    return ThemeCarrierResult(
        status="abstain",
        abstain_reason=reason,
        decision_session=day.date(),
        decision_at=cutoff,
        theme_components=pd.DataFrame(),
        stock_scores=pd.Series(dtype=float, name="theme_score_equal_weight_v1"),
        coverage={},
        receipt_hashes=(),
    )


def score_theme_carrier_from_captures(
    *,
    u0: UniverseSnapshot,
    prior_capture: MarketSnapshotCapture,
    current_capture: MarketSnapshotCapture,
    mapping_capture: ThemeMappingCapture,
    ledger: ResearchSignalLedger,
    decision_session: Any,
    prior_session: Any,
    decision_at: Any,
) -> ThemeCarrierResult:
    """Score only the frozen four-component endpoint from captured evidence."""
    today = _session(decision_session)
    yesterday = _session(prior_session)
    cutoff = _local_time(decision_at)
    if (
        prior_capture.status != "ok"
        or current_capture.status != "ok"
        or prior_capture.decision_eligibility != "eligible_for_decision_at"
        or current_capture.decision_eligibility != "eligible_for_decision_at"
    ):
        return _theme_abstain(session=today, decision_at=cutoff, reason="market_capture_abstained")
    if (
        mapping_capture.status != "ok"
        or mapping_capture.decision_eligibility != "eligible_for_decision_at"
    ):
        return _theme_abstain(session=today, decision_at=cutoff, reason="theme_mapping_capture_abstained")
    if prior_capture.session != yesterday.date() or current_capture.session != today.date():
        return _theme_abstain(session=today, decision_at=cutoff, reason="market_capture_session_mismatch")
    if current_capture.decision_at != cutoff or mapping_capture.decision_at != cutoff:
        return _theme_abstain(session=today, decision_at=cutoff, reason="capture_cutoff_mismatch")
    bars = pd.concat([prior_capture.bars, current_capture.bars], ignore_index=True)
    # Theme Carrier intentionally rejects incomplete U0 coverage and missing
    # limit-price evidence itself.  Do not repair or synthesize either here.
    return score_theme_carrier_v1(
        decision_session=today,
        prior_session=yesterday,
        decision_at=cutoff,
        u0=u0,
        bars=bars,
        primary_theme_map=mapping_capture.mappings,
        ledger=ledger,
    )


def _module_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _theme_extractor_identity() -> dict[str, str]:
    # Hash both the pure scorer and the receipt-to-ledger adapter.  This is an
    # identity record, not a model score and not a route to an account.
    import quantlab.theme_carrier as theme_module

    return dict(
        kind="rule",
        version="theme_carrier_minimal_v1",
        theme_carrier_code_sha256=_module_sha256(str(theme_module.__file__)),
        capture_adapter_code_sha256=_module_sha256(__file__),
        capture_contract_version=FORWARD_MARKET_CAPTURE_VERSION,
    )


def _canonical_timed_receipts(
    *, ledger: ResearchSignalLedger, receipt_ids: Sequence[str], cutoff: pd.Timestamp
) -> tuple[tuple[str, ...], bool]:
    """Return canonical receipt ids only if every supplied id is timely proof."""
    canonical: set[str] = set()
    try:
        for receipt_id in receipt_ids:
            evidence = ledger.export_receipt_evidence(str(receipt_id))
            receipt = evidence["receipt"]
            body = receipt["body"]
            actual = str(evidence["canonical_receipt_id"])
            if (
                actual != str(receipt_id)
                or receipt.get("event_id") != actual
                or receipt.get("status") != "accepted"
                or _local_time(receipt["recorded_at"]) > cutoff
                or _local_time(receipt["available_at"]) > cutoff
                or _local_time(body["published_at"]) > cutoff
                or _local_time(body["first_captured_at"]) > cutoff
            ):
                return tuple(sorted(canonical)), False
            canonical.add(actual)
    except (KeyError, TypeError, ValueError, OSError):
        return tuple(sorted(canonical)), False
    return tuple(sorted(canonical)), True


def _scope_matches_u0(
    *, ledger: ResearchSignalLedger, scope_id: str, u0: UniverseSnapshot, cutoff: pd.Timestamp
) -> bool:
    """Verify a declared scope's exact member set without reading private DB state."""
    try:
        coverage = ledger.coverage(scope_id, cutoff, "theme_carrier_minimal_v1")
    except (TypeError, ValueError):
        return False
    scope_members = set(coverage["observed_entity_keys"])
    scope_members.update(coverage["abstained_entity_keys"])
    scope_members.update(coverage["missing"])
    members = {str(code) for code in u0.members}
    return bool(members) and coverage.get("expected") == len(members) and scope_members == members


def _record_theme_abstention(
    *,
    ledger: ResearchSignalLedger,
    scope_id: str,
    u0: UniverseSnapshot,
    cutoff: pd.Timestamp,
    now: Any,
    reason: str,
    receipt_ids: Sequence[str],
    entity_ids: Sequence[str] = (),
) -> ThemeLedgerRecord:
    """Write one all-scope abstention; never replace it with zero scores."""
    try:
        event = ledger.abstain(
            observation_key="theme_carrier_minimal_v1:" + u0.membership_hash,
            scope_id=scope_id,
            decision_at=cutoff,
            feature_id="theme_carrier_minimal_v1",
            target_entity_keys=tuple(str(code) for code in u0.members),
            stage="theme_carrier_capture",
            reason=reason,
            receipt_ids=tuple(receipt_ids),
            entity_ids=tuple(entity_ids),
            extractor=_theme_extractor_identity(),
            now=now,
        )
    except (TypeError, ValueError):
        return ThemeLedgerRecord("not_recorded", "theme_observation_recording_failed", tuple(entity_ids), None)
    return ThemeLedgerRecord("abstained", event["body"].get("reason"), tuple(entity_ids), event)


def _valid_theme_u0(u0: UniverseSnapshot, cutoff: pd.Timestamp) -> bool:
    return bool(
        isinstance(u0, UniverseSnapshot)
        and u0.pool_id == "U0"
        and u0.contract_version == FORWARD_CONTRACT_VERSION
        and u0.status == "ok"
        and len(set(u0.members)) == len(u0.members)
        and u0.decision_at == cutoff
    )


def _is_nonempty_text_tuple(value: Any) -> bool:
    """Return whether an adapter-owned receipt collection has its frozen shape."""
    return isinstance(value, tuple) and all(isinstance(item, str) and item.strip() for item in value)


def _is_well_formed_market_snapshot_capture(value: Any) -> bool:
    """Guard the public feature writer from malformed scheduler arguments.

    The capture function itself only creates canonical tuples and data frames,
    but this public adapter can be invoked by an interrupted scheduler or a
    direct integration.  Validate the small surface used below before reading
    it so a bad object becomes a visible scope-wide abstention rather than an
    exception that leaves the scope silently uncovered.
    """
    if not isinstance(value, MarketSnapshotCapture):
        return False
    if (
        not isinstance(value.status, str)
        or value.status not in {"ok", "abstain"}
        or value.decision_eligibility not in {
            "eligible_for_decision_at",
            "not_eligible_for_decision_at",
        }
        or not isinstance(value.bars, pd.DataFrame)
        or (not _is_nonempty_text_tuple(value.receipt_ids) and value.receipt_ids != ())
        or (not _is_nonempty_text_tuple(value.receipt_hashes) and value.receipt_hashes != ())
        or (value.status == "ok" and (not value.receipt_ids or not value.receipt_hashes))
    ):
        return False
    try:
        _session(value.session)
        _local_time(value.decision_at)
    except (TypeError, ValueError):
        return False
    return True


def _is_well_formed_theme_mapping_capture(value: Any) -> bool:
    """Validate only the mapping-capture fields consumed by the ledger writer."""
    if not isinstance(value, ThemeMappingCapture):
        return False
    if (
        not isinstance(value.status, str)
        or value.status not in {"ok", "abstain"}
        or value.decision_eligibility not in {
            "eligible_for_decision_at",
            "not_eligible_for_decision_at",
        }
        or not isinstance(value.mappings, pd.DataFrame)
        or (
            value.receipt_id is not None
            and (not isinstance(value.receipt_id, str) or not value.receipt_id.strip())
        )
        or (
            value.content_sha256 is not None
            and (not isinstance(value.content_sha256, str) or not value.content_sha256.strip())
        )
        or (value.status == "ok" and (not value.receipt_id or not value.content_sha256))
    ):
        return False
    try:
        _session(value.session)
        _local_time(value.decision_at)
    except (TypeError, ValueError):
        return False
    return True


def record_theme_carrier_ledger_event(
    *,
    ledger: ResearchSignalLedger,
    scope_id: str,
    u0: UniverseSnapshot,
    score: ThemeCarrierResult,
    prior_capture: MarketSnapshotCapture,
    current_capture: MarketSnapshotCapture,
    mapping_capture: ThemeMappingCapture,
    decision_at: Any,
    entity_recorded_at: Any | None = None,
    now: Any | None = None,
) -> ThemeLedgerRecord:
    """Complete the mapping/feature leg of the forward Theme Carrier ledger.

    The pure scorer intentionally writes no state.  This separate adapter
    takes its result plus receipt-backed captures, records one resolved
    stock-to-primary-theme mapping per U0 member, then records exactly one
    all-scope observed feature.  Any missing map, receipt, score or timing
    becomes one all-scope abstention, preserving the baseline entry order.
    """
    if not isinstance(ledger, ResearchSignalLedger):
        raise TypeError("research_signal_ledger_required")
    cutoff = _local_time(decision_at)
    if not _valid_theme_u0(u0, cutoff):
        return ThemeLedgerRecord("not_recorded", "invalid_forward_u0", (), None)
    try:
        scope = _safe_text(scope_id, "scope_id")
    except ValueError:
        return ThemeLedgerRecord("not_recorded", "invalid_scope_id", (), None)
    if not _scope_matches_u0(ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff):
        return ThemeLedgerRecord("not_recorded", "scope_not_declared_or_membership_mismatch", (), None)

    # Both legacy scheduler timestamps remain accepted only for call-site
    # compatibility.  They are intentionally not trusted: each live feature
    # write must use the ledger-owned runtime clock rather than a caller's
    # earlier capture time.
    recorded_at = ledger.runtime_now()

    # A broken scheduler/caller must be visible in the ledger as an
    # all-scope abstention, not as an AttributeError that leaves the scope
    # silently missing.  Do this before accessing receipt fields.
    if not (
        _is_well_formed_market_snapshot_capture(prior_capture)
        and _is_well_formed_market_snapshot_capture(current_capture)
        and _is_well_formed_theme_mapping_capture(mapping_capture)
    ):
        return _record_theme_abstention(
            ledger=ledger,
            scope_id=scope,
            u0=u0,
            cutoff=cutoff,
            now=recorded_at,
            reason="capture_or_mapping_unavailable",
            receipt_ids=(),
        )

    # The validators above establish parseability, but normalize the three
    # clocks once here so malformed external values cannot reach a raw Python
    # comparison further below.
    prior_decision_at = _local_time(prior_capture.decision_at)
    current_decision_at = _local_time(current_capture.decision_at)
    mapping_decision_at = _local_time(mapping_capture.decision_at)

    all_receipt_ids = tuple(
        sorted(
            set(prior_capture.receipt_ids)
            | set(current_capture.receipt_ids)
            | ({mapping_capture.receipt_id} if mapping_capture.receipt_id else set())
        )
    )
    canonical_receipts, receipts_ok = _canonical_timed_receipts(
        ledger=ledger, receipt_ids=all_receipt_ids, cutoff=cutoff
    )
    if not receipts_ok:
        return _record_theme_abstention(
            ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff, now=recorded_at,
            reason="receipt_evidence_incomplete_or_late", receipt_ids=canonical_receipts,
        )
    if (
        prior_capture.status != "ok"
        or current_capture.status != "ok"
        or mapping_capture.status != "ok"
        or prior_capture.decision_eligibility != "eligible_for_decision_at"
        or current_capture.decision_eligibility != "eligible_for_decision_at"
        or mapping_capture.decision_eligibility != "eligible_for_decision_at"
        or not mapping_capture.receipt_id
        or prior_decision_at > cutoff
        or current_decision_at != cutoff
        or mapping_decision_at != cutoff
    ):
        return _record_theme_abstention(
            ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff, now=recorded_at,
            reason="capture_or_mapping_unavailable", receipt_ids=canonical_receipts,
        )
    if mapping_capture.receipt_id not in canonical_receipts:
        return _record_theme_abstention(
            ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff, now=recorded_at,
            reason="theme_mapping_receipt_not_canonical", receipt_ids=canonical_receipts,
        )

    members = tuple(str(code) for code in u0.members)
    try:
        mapping = mapping_capture.mappings.copy()
    except (AttributeError, TypeError, ValueError):
        return _record_theme_abstention(
            ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff, now=recorded_at,
            reason="capture_or_mapping_unavailable", receipt_ids=canonical_receipts,
        )
    if set(_MAPPING_COLUMNS) - set(mapping.columns):
        return _record_theme_abstention(
            ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff, now=recorded_at,
            reason="primary_theme_mapping_schema_incomplete", receipt_ids=canonical_receipts,
        )
    mapping["instrument_id"] = mapping.instrument_id.astype(str)
    map_valid = (
        not mapping.instrument_id.duplicated().any()
        and set(mapping.instrument_id) == set(members)
        and mapping.mapping_status.astype(str).eq("active").all()
        and mapping.theme_id.notna().all()
        and mapping.theme_id.astype(str).ne("").all()
        and mapping.source_receipt_id.astype(str).eq(mapping_capture.receipt_id).all()
    )
    mapped_at = recorded_at
    if not map_valid or mapped_at > cutoff:
        return _record_theme_abstention(
            ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff, now=recorded_at,
            reason="primary_theme_mapping_incomplete_or_late", receipt_ids=canonical_receipts,
        )

    entity_ids: list[str] = []
    try:
        by_code = mapping.set_index("instrument_id")
        for code in members:
            theme_id = str(by_code.at[code, "theme_id"])
            entity = ledger.record_entity(
                mapping_capture.receipt_id,
                entity_type="stock_primary_theme",
                canonical_id=code,
                display_name=code + "::" + theme_id,
                mapping_method="direct_primary_theme_mapping",
                mapping_version="theme_carrier_minimal_v1",
                now=mapped_at,
            )
            entity_ids.append(str(entity["event_id"]))
    except (KeyError, TypeError, ValueError):
        return _record_theme_abstention(
            ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff, now=recorded_at,
            reason="primary_theme_entity_recording_failed", receipt_ids=canonical_receipts,
            entity_ids=entity_ids,
        )

    try:
        score_valid = (
            isinstance(score, ThemeCarrierResult)
            and score.status == "ok"
            and score.decision_session == u0.session
            and score.decision_at == cutoff
            and isinstance(score.stock_scores, pd.Series)
            and set(score.stock_scores.index.astype(str)) == set(members)
            and len(score.stock_scores) == len(members)
            and np.isfinite(score.stock_scores.astype(float).to_numpy()).all()
            and isinstance(score.theme_components, pd.DataFrame)
            and not score.theme_components.empty
            and isinstance(score.receipt_hashes, tuple)
            and set(score.receipt_hashes).issubset(set(prior_capture.receipt_hashes) | set(current_capture.receipt_hashes) | ({mapping_capture.content_sha256} if mapping_capture.content_sha256 else set()))
        )
    except (AttributeError, TypeError, ValueError):
        score_valid = False
    if not score_valid:
        return _record_theme_abstention(
            ledger=ledger, scope_id=scope, u0=u0, cutoff=cutoff, now=recorded_at,
            reason="theme_score_incomplete_or_unlinked", receipt_ids=canonical_receipts,
            entity_ids=entity_ids,
        )

    score_by_instrument = {code: float(score.stock_scores.loc[code]) for code in members}
    value = dict(
        schema="quantlab.theme-carrier-ledger-observation.v1",
        signal_version="theme_carrier_minimal_v1",
        u0_membership_sha256=u0.membership_hash,
        primary_endpoint="theme_score_equal_weight_v1",
        component_results_diagnostic_only=True,
        score_by_instrument=score_by_instrument,
        theme_components=score.theme_components.to_dict(orient="records"),
        receipt_hashes=tuple(sorted(score.receipt_hashes)),
    )
    try:
        event = ledger.record_observation(
            observation_key="theme_carrier_minimal_v1:" + u0.membership_hash,
            scope_id=scope,
            decision_at=cutoff,
            feature_id="theme_carrier_minimal_v1",
            target_entity_keys=members,
            receipt_ids=canonical_receipts,
            entity_ids=entity_ids,
            value=value,
            unit="theme_score_equal_weight_v1",
            extractor=_theme_extractor_identity(),
            now=recorded_at,
        )
    except (TypeError, ValueError):
        return ThemeLedgerRecord("not_recorded", "theme_observation_recording_failed", tuple(entity_ids), None)
    if event["status"] == "abstained":
        return ThemeLedgerRecord("abstained", event["body"].get("reason"), tuple(entity_ids), event)
    return ThemeLedgerRecord("observed", None, tuple(entity_ids), event)
