"""Injected, fail-closed raw HTTPS collection for the forward market sidecar.

The module deliberately contains no HTTP client.  A scheduler may supply a
small opener implementation later, but this layer only accepts its raw bytes
and exact request/response identity.  It registers the frozen source manifest
before the first opener call, fetches page one first, and lets that validated
page total determine the entire remaining request set.  It has no cache,
fallback, account, router, order, or replay dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal, Protocol

import pandas as pd

from quantlab.forward_source_manifest import (
    EastmoneyEodSourceManifest,
    PreparedRawEastmoneyCollection,
    RawEastmoneyTransportAttempt,
    SourceManifestContractError,
    SourceManifestJournal,
)
from quantlab.research_universe import DECISION_TIMEZONE


FORWARD_MARKET_TRANSPORT_VERSION = "forward-market-transport-v1"
# The fixed profile is an all-A endpoint with 100 rows per page.  This bound is
# not a second source or a selection filter: it simply prevents a malformed
# first response from creating an unbounded sequence of opener calls.
MAX_EASTMONEY_EOD_TOTAL_ROWS = 100_000


@dataclass(frozen=True)
class RawEastmoneyHttpsResponse:
    """Exactly one raw response returned by an injected HTTPS opener.

    Both URLs are required.  ``request_url`` catches a buggy opener that issues
    another request than it was handed; ``response_url`` prevents redirects or
    a different endpoint from being treated as the fixed Eastmoney source.
    """

    request_url: str
    response_url: str
    raw: bytes
    content_type: str = "application/json"
    provider_published_at: Any | None = None


class RawEastmoneyHttpsOpener(Protocol):
    """The only capability required to obtain a raw page.

    Production wiring is intentionally outside this module.  Tests inject a
    deterministic fake, so importing this module never performs networking.
    """

    def __call__(self, canonical_url: str) -> RawEastmoneyHttpsResponse:
        """Return raw bytes for exactly ``canonical_url`` or raise an error."""


class CollectionClock(Protocol):
    """An injectable Shanghai-aware clock used for registration and captures."""

    def __call__(self) -> Any:
        """Return the current timestamp."""


@dataclass(frozen=True)
class FailedRawEastmoneyCollection:
    """An explicit, non-recoverable collection outcome.

    ``attempts`` includes each request that was actually attempted.  There is
    deliberately no retry source, cache value, parsed payload, or substitute
    page; a caller must treat this outcome as an abstention input.
    """

    status: Literal["failed"]
    failure_reason: str
    session: object
    decision_at: pd.Timestamp
    collection_id: str
    manifest_sha256: str
    manifest_event_sha256: str
    attempts: tuple[RawEastmoneyTransportAttempt, ...]
    # This is read inside the transport, not supplied by the scheduler.  It
    # lets an early failure (for example, a stale manifest registration before
    # any request is issued) become auditable transport evidence instead of
    # disappearing because there is no page capture timestamp to borrow.
    reported_at: pd.Timestamp
    transport_version: str = FORWARD_MARKET_TRANSPORT_VERSION


def _timestamp(value: Any, *, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SourceManifestContractError(field + "_timestamp_required") from exc
    if pd.isna(stamp):
        raise SourceManifestContractError(field + "_timestamp_required")
    if stamp.tzinfo is None:
        raise SourceManifestContractError(field + "_timezone_required")
    return stamp.tz_convert(DECISION_TIMEZONE)


def _session(value: Any) -> pd.Timestamp:
    try:
        session = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SourceManifestContractError("transport_session_required") from exc
    if pd.isna(session):
        raise SourceManifestContractError("transport_session_required")
    if session.tzinfo is not None:
        session = session.tz_convert(DECISION_TIMEZONE).tz_localize(None)
    return session.normalize()


def _decision_at(session: pd.Timestamp, value: Any) -> pd.Timestamp:
    decision = _timestamp(value, field="transport_decision_at")
    expected = pd.Timestamp(session.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(hours=18)
    if decision != expected:
        raise SourceManifestContractError("transport_decision_at_must_equal_session_1800")
    return decision


def _collection_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestContractError("transport_collection_id_required")
    return value.strip()


def _failure(
    *,
    reason: str,
    session: pd.Timestamp,
    decision_at: pd.Timestamp,
    collection_id: str,
    manifest: EastmoneyEodSourceManifest,
    manifest_event_sha256: str,
    attempts: list[RawEastmoneyTransportAttempt],
    reported_at: pd.Timestamp,
) -> FailedRawEastmoneyCollection:
    return FailedRawEastmoneyCollection(
        status="failed",
        failure_reason=reason,
        session=session.date(),
        decision_at=decision_at,
        collection_id=collection_id,
        manifest_sha256=manifest.manifest_sha256,
        manifest_event_sha256=manifest_event_sha256,
        attempts=tuple(attempts),
        reported_at=reported_at,
    )


def _failure_reported_at(*, clock: CollectionClock, fallback: pd.Timestamp) -> pd.Timestamp:
    """Read the transport clock for a failure report without inventing time.

    A malformed post-response clock cannot make the failure disappear.  In
    that narrow case retain the last valid, internally observed timestamp as
    the conservative lower bound rather than accepting a caller-supplied
    timestamp or using ledger write time as a fictitious request time.
    """
    try:
        return _timestamp(clock(), field="failure_report_clock")
    except SourceManifestContractError:
        return fallback


def _fetch_error_label(error: BaseException) -> str:
    """Preserve a bounded, non-secret failure identity in the failed attempt."""

    name = type(error).__name__
    if not name or len(name) > 120:
        return "opener_error"
    return name


def _failed_attempt(
    *,
    collection_id: str,
    page_number: int,
    canonical_url: str,
    captured_at: pd.Timestamp,
    reason: str,
) -> RawEastmoneyTransportAttempt:
    return RawEastmoneyTransportAttempt(
        collection_id=collection_id,
        page_number=page_number,
        canonical_url=canonical_url,
        raw=None,
        fetch_error=reason,
        captured_at=captured_at,
        content_type="application/json",
        transport_kind="raw_https_get",
        cache_reused=False,
        fallback_attempted=False,
    )


def _validated_page_total(
    raw: bytes,
    *,
    page_number: int,
    page_size: int,
    expected_total: int | None,
) -> int:
    """Validate only enough raw shape to safely determine a page set.

    ``forward_market_capture`` remains responsible for parsing the actual A
    share fields and writing receipts.  Here we verify the page's exact total
    and row count so neither a malformed page one nor a later changing total
    can expand, shrink, or redirect the request sequence.
    """

    if not isinstance(raw, bytes) or not raw:
        raise ValueError("raw_bytes_required")
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("page_not_json") from exc
    if not isinstance(document, dict) or not isinstance(document.get("data"), dict):
        raise ValueError("page_data_object_required")
    data = document["data"]
    total = data.get("total")
    rows = data.get("diff")
    if type(total) is not int or total < 1 or total > MAX_EASTMONEY_EOD_TOTAL_ROWS:
        raise ValueError("page_total_invalid")
    if expected_total is not None and total != expected_total:
        raise ValueError("page_total_changed")
    if not isinstance(rows, list):
        raise ValueError("page_diff_list_required")
    expected_rows = min(page_size, total - (page_number - 1) * page_size)
    if expected_rows < 1 or len(rows) != expected_rows:
        raise ValueError("page_row_count_invalid")
    return total


def _response_attempt_or_failure(
    *,
    opener: RawEastmoneyHttpsOpener,
    clock: CollectionClock,
    collection_id: str,
    page_number: int,
    canonical_url: str,
    session: pd.Timestamp,
    registered_at: pd.Timestamp,
) -> tuple[RawEastmoneyTransportAttempt | None, str | None, pd.Timestamp]:
    """Make one opener call and turn it into raw evidence or a failed attempt."""

    before_fetch = _timestamp(clock(), field="capture_clock")
    if registered_at > before_fetch:
        return None, "registration_after_capture", before_fetch
    try:
        response = opener(canonical_url)
    except Exception as error:  # the failure itself is part of the evidence
        try:
            captured_at = _timestamp(clock(), field="capture_clock")
        except SourceManifestContractError:
            captured_at = before_fetch
        return (
            _failed_attempt(
                collection_id=collection_id,
                page_number=page_number,
                canonical_url=canonical_url,
                captured_at=captured_at,
                reason=_fetch_error_label(error),
            ),
            "page_fetch_failed",
            captured_at,
        )
    try:
        captured_at = _timestamp(clock(), field="capture_clock")
    except SourceManifestContractError:
        return None, "capture_clock_invalid_after_response", before_fetch
    if registered_at > captured_at:
        return (
            _failed_attempt(
                collection_id=collection_id,
                page_number=page_number,
                canonical_url=canonical_url,
                captured_at=captured_at,
                reason="registration_after_capture",
            ),
            "registration_after_capture",
            captured_at,
        )
    if not isinstance(response, RawEastmoneyHttpsResponse):
        return (
            _failed_attempt(
                collection_id=collection_id,
                page_number=page_number,
                canonical_url=canonical_url,
                captured_at=captured_at,
                reason="invalid_opener_response",
            ),
            "invalid_opener_response",
            captured_at,
        )
    if response.request_url != canonical_url or response.response_url != canonical_url:
        return (
            _failed_attempt(
                collection_id=collection_id,
                page_number=page_number,
                canonical_url=canonical_url,
                captured_at=captured_at,
                reason="response_source_identity_mismatch",
            ),
            "response_source_identity_mismatch",
            captured_at,
        )
    if response.content_type != "application/json":
        return (
            _failed_attempt(
                collection_id=collection_id,
                page_number=page_number,
                canonical_url=canonical_url,
                captured_at=captured_at,
                reason="response_content_type_invalid",
            ),
            "response_content_type_invalid",
            captured_at,
        )
    if not isinstance(response.raw, bytes) or not response.raw:
        return (
            _failed_attempt(
                collection_id=collection_id,
                page_number=page_number,
                canonical_url=canonical_url,
                captured_at=captured_at,
                reason="response_raw_bytes_invalid",
            ),
            "response_raw_bytes_invalid",
            captured_at,
        )
    return (
        RawEastmoneyTransportAttempt(
            collection_id=collection_id,
            page_number=page_number,
            canonical_url=canonical_url,
            raw=response.raw,
            captured_at=captured_at,
            provider_published_at=response.provider_published_at,
            content_type=response.content_type,
            transport_kind="raw_https_get",
            cache_reused=False,
            fallback_attempted=False,
        ),
        None,
        captured_at,
    )


def collect_registered_eastmoney_eod_pages(
    *,
    journal: SourceManifestJournal,
    manifest: EastmoneyEodSourceManifest,
    session: Any,
    decision_at: Any,
    collection_id: str,
    opener: RawEastmoneyHttpsOpener,
    clock: CollectionClock,
) -> PreparedRawEastmoneyCollection | FailedRawEastmoneyCollection:
    """Collect exactly the raw pages implied by page one's validated total.

    The source profile is first appended to (or re-opened from) the hash-chain
    journal.  Each later page is requested once, directly from that profile.
    Any clock, opener, response-identity, parse, total, or manifest failure is
    returned as a non-retryable failed collection.  Successful output is ready
    for ``capture_eastmoney_eod_snapshot`` but is not itself a ledger receipt
    or a market signal.
    """

    if not isinstance(journal, SourceManifestJournal):
        raise SourceManifestContractError("source_manifest_journal_required")
    if not isinstance(manifest, EastmoneyEodSourceManifest):
        raise SourceManifestContractError("eastmoney_source_manifest_required")
    if not callable(opener):
        raise SourceManifestContractError("raw_https_opener_required")
    if not callable(clock):
        raise SourceManifestContractError("collection_clock_required")

    day = _session(session)
    cutoff = _decision_at(day, decision_at)
    identifier = _collection_id(collection_id)

    # Registering from the injected clock prevents a caller from supplying an
    # arbitrary historical registration time.  If this exact immutable profile
    # already exists, the journal returns its earlier hash-chain record.
    registered_at = _timestamp(clock(), field="manifest_registration")
    registration = journal.register(manifest, recorded_at=registered_at)
    registered_transport = journal.open_raw_transport(manifest)
    registered_at = _timestamp(registration.recorded_at, field="manifest_registration")
    attempts: list[RawEastmoneyTransportAttempt] = []

    first_url = manifest.canonical_url(1)
    first, failure_reason, failure_reported_at = _response_attempt_or_failure(
        opener=opener,
        clock=clock,
        collection_id=identifier,
        page_number=1,
        canonical_url=first_url,
        session=day,
        registered_at=registered_at,
    )
    if first is not None:
        attempts.append(first)
    if failure_reason is not None:
        return _failure(
            reason=failure_reason,
            session=day,
            decision_at=cutoff,
            collection_id=identifier,
            manifest=manifest,
            manifest_event_sha256=registration.event_sha256,
            attempts=attempts,
            reported_at=failure_reported_at,
        )
    assert first is not None
    try:
        total = _validated_page_total(
            first.raw or b"",
            page_number=1,
            page_size=manifest.page_size,
            expected_total=None,
        )
    except ValueError:
        return _failure(
            reason="first_page_total_parse_failed",
            session=day,
            decision_at=cutoff,
            collection_id=identifier,
            manifest=manifest,
            manifest_event_sha256=registration.event_sha256,
            attempts=attempts,
            reported_at=_failure_reported_at(
                clock=clock,
                fallback=_timestamp(first.captured_at, field="first_page_capture"),
            ),
        )

    page_count = math.ceil(total / manifest.page_size)
    for page_number in range(2, page_count + 1):
        canonical_url = manifest.canonical_url(page_number)
        attempt, failure_reason, failure_reported_at = _response_attempt_or_failure(
            opener=opener,
            clock=clock,
            collection_id=identifier,
            page_number=page_number,
            canonical_url=canonical_url,
            session=day,
            registered_at=registered_at,
        )
        if attempt is not None:
            attempts.append(attempt)
        if failure_reason is not None:
            return _failure(
                reason=failure_reason,
                session=day,
                decision_at=cutoff,
                collection_id=identifier,
                manifest=manifest,
                manifest_event_sha256=registration.event_sha256,
                attempts=attempts,
                reported_at=failure_reported_at,
            )
        assert attempt is not None
        try:
            _validated_page_total(
                attempt.raw or b"",
                page_number=page_number,
                page_size=manifest.page_size,
                expected_total=total,
            )
        except ValueError:
            return _failure(
                reason="page_total_parse_failed",
                session=day,
                decision_at=cutoff,
                collection_id=identifier,
                manifest=manifest,
                manifest_event_sha256=registration.event_sha256,
                attempts=attempts,
                reported_at=_failure_reported_at(
                    clock=clock,
                    fallback=_timestamp(attempt.captured_at, field="later_page_capture"),
                ),
            )

    try:
        return registered_transport.prepare_collection(
            session=day.date(),
            decision_at=cutoff,
            attempts=tuple(attempts),
        )
    except SourceManifestContractError:
        return _failure(
            reason="manifest_validation_failed",
            session=day,
            decision_at=cutoff,
            collection_id=identifier,
            manifest=manifest,
            manifest_event_sha256=registration.event_sha256,
            attempts=attempts,
            reported_at=_failure_reported_at(
                clock=clock,
                fallback=_timestamp(attempts[-1].captured_at, field="last_page_capture"),
            ),
        )
