"""Frozen source manifests for future forward-only market collection.

This module intentionally has no HTTP client, scheduler, cache reader, account
engine, router, or order dependency.  It registers the request contract that a
future raw-byte transport must honour before it hands pages to
``forward_market_capture``.  Registration is an append-only, hash-chained
journal; it is not a market-data sample and cannot create a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quantlab.research_universe import DECISION_TIMEZONE


SOURCE_MANIFEST_SCHEMA = "quantlab.forward-market-source-manifest.v1"
SOURCE_MANIFEST_RECORD_SCHEMA = "quantlab.forward-market-source-manifest-record.v1"
EASTMONEY_SOURCE_ID = "eastmoney.push2.clist"
EASTMONEY_SOURCE_FAMILY = "eastmoney.push2"
# The timing contract changed from a hard 17:55 eligibility cutoff to an
# operations-only target.  That is a source-profile change, not a silent edit
# to a registered V1 contract.
EASTMONEY_PROFILE_VERSION = "eastmoney-push2-clist-eod-v2"
EASTMONEY_HOST = "push2.eastmoney.com"
EASTMONEY_PATH = "/api/qt/clist/get"
EASTMONEY_ALL_A_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81,s:2048"
EASTMONEY_REQUIRED_FIELDS = (
    "f2", "f5", "f6", "f12", "f14", "f15", "f16", "f17", "f18", "f51",
)
EASTMONEY_PAGE_SIZE = 100
_ZERO_SHA256 = "0" * 64


class SourceManifestContractError(ValueError):
    """A source request or its append-only manifest identity is invalid."""


def _canonical_json_bytes(value: object, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourceManifestContractError(name + "_canonical_json_required") from exc


def _sha256(value: object, name: str) -> str:
    digest = sha256(_canonical_json_bytes(value, name)).hexdigest()
    return digest


def _strict_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise SourceManifestContractError(name + "_sha256_required")
    return value.lower()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestContractError(name + "_required")
    return value.strip()


def _timestamp_text(value: object, name: str) -> str:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SourceManifestContractError(name + "_timestamp_required") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise SourceManifestContractError(name + "_timezone_required")
    return stamp.tz_convert(DECISION_TIMEZONE).isoformat()


def _strict_json_object(line: str, name: str) -> dict[str, object]:
    def object_pairs_hook(pairs):
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            line,
            object_pairs_hook=object_pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SourceManifestContractError(name + "_json_required") from exc
    if not isinstance(value, dict):
        raise SourceManifestContractError(name + "_object_required")
    return value


@dataclass(frozen=True)
class EastmoneyEodSourceManifest:
    """The one V1 raw-page source profile, including its non-fallback policy."""

    schema_version: str = SOURCE_MANIFEST_SCHEMA
    source_id: str = EASTMONEY_SOURCE_ID
    source_family: str = EASTMONEY_SOURCE_FAMILY
    source_profile_version: str = EASTMONEY_PROFILE_VERSION
    page_size: int = EASTMONEY_PAGE_SIZE

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_MANIFEST_SCHEMA:
            raise SourceManifestContractError("source_manifest_schema_invalid")
        if self.source_id != EASTMONEY_SOURCE_ID or self.source_family != EASTMONEY_SOURCE_FAMILY:
            raise SourceManifestContractError("source_manifest_source_identity_invalid")
        if self.source_profile_version != EASTMONEY_PROFILE_VERSION:
            raise SourceManifestContractError("source_manifest_profile_version_invalid")
        if type(self.page_size) is not int or self.page_size != EASTMONEY_PAGE_SIZE:
            raise SourceManifestContractError("source_manifest_page_size_invalid")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_profile_version": self.source_profile_version,
            "request_contract": {
                "method": "GET",
                "scheme": "https",
                "host": EASTMONEY_HOST,
                "path": EASTMONEY_PATH,
                "query_template": [
                    ["pn", "{page_number}"],
                    ["pz", "{page_size}"],
                    ["po", "1"],
                    ["np", "1"],
                    ["fltt", "2"],
                    ["invt", "2"],
                    ["fid", "f6"],
                    ["fs", EASTMONEY_ALL_A_FS],
                    ["fields", ",".join(EASTMONEY_REQUIRED_FIELDS)],
                ],
            },
            "page_contract": {
                "page_number_start": 1,
                "page_size": self.page_size,
                "sort_field": "f6",
                "sort_direction": "descending",
                "required_fields": list(EASTMONEY_REQUIRED_FIELDS),
                "require_total": True,
                "require_complete_page_set": True,
            },
            "clock_contract": {
                "timezone": DECISION_TIMEZONE,
                "raw_collection_window": ["15:30", "17:50"],
                # This is an operations target for alerting, never a signal
                # eligibility deadline.  A complete close snapshot written at
                # 17:57 remains eligible if every required receipt is no
                # later than its 18:00 decision cutoff.
                "operations_validation_target": "17:55",
                "decision_at": "18:00",
            },
            "transport_contract": {
                "input_kind": "raw_https_response_bytes",
                "content_type": "application/json",
                "cache_policy": "forbidden",
                "fallback_policy": "forbidden",
                "parsed_payload_policy": "forbidden",
            },
            "availability_contract": {
                "provider_clock_policy": "provider_clock_or_https_capture_lower_bound",
                "late_page_policy": "persist_then_check_named_decision_eligibility",
                "missing_page_policy": "explicit_abstention",
            },
        }

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self._identity_payload(), "source_manifest")

    def to_mapping(self) -> dict[str, object]:
        return {**self._identity_payload(), "manifest_sha256": self.manifest_sha256}

    def canonical_url(self, page_number: int) -> str:
        """Return the sole URL spelling a future V1 transport may request."""
        if type(page_number) is not int or page_number < 1:
            raise SourceManifestContractError("source_manifest_page_number_invalid")
        query = "&".join(
            key + "=" + value.format(page_number=page_number, page_size=self.page_size)
            for key, value in self._identity_payload()["request_contract"]["query_template"]
        )
        return "https://" + EASTMONEY_HOST + EASTMONEY_PATH + "?" + query

    @classmethod
    def from_mapping(cls, value: object) -> "EastmoneyEodSourceManifest":
        if not isinstance(value, Mapping):
            raise SourceManifestContractError("source_manifest_mapping_required")
        expected = cls()
        expected_mapping = expected.to_mapping()
        if set(value) != set(expected_mapping) or dict(value) != expected_mapping:
            raise SourceManifestContractError("source_manifest_payload_mismatch")
        return expected


def fixed_eastmoney_eod_manifest_v1() -> EastmoneyEodSourceManifest:
    """Return the only registered V1 profile; it contains no market payload."""
    return EastmoneyEodSourceManifest()


@dataclass(frozen=True)
class SourceManifestRegistration:
    """One hash-chained journal record for a frozen source profile."""

    sequence: int
    recorded_at: str
    previous_event_sha256: str
    manifest_sha256: str
    event_sha256: str
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class RawEastmoneyTransportAttempt:
    """A supplied raw response or a future raw-transport attempt description.

    This object is deliberately unable to name a backup provider or a parsed
    cache result.  A future scheduler may preserve a failed attempt, but may
    not replace it under this manifest.
    """

    collection_id: str
    page_number: int
    canonical_url: str
    raw: bytes | None
    captured_at: Any
    fetch_error: str | None = None
    provider_published_at: Any | None = None
    content_type: str = "application/json"
    transport_kind: str = "raw_https_get"
    cache_reused: bool = False
    fallback_attempted: bool = False


@dataclass(frozen=True)
class PreparedRawEastmoneyCollection:
    """Validated raw pages plus immutable manifest provenance for later capture."""

    session: object
    decision_at: pd.Timestamp
    collection_id: str
    manifest_sha256: str
    manifest_event_sha256: str
    capture_pages: tuple[Any, ...]


def _local_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SourceManifestContractError(name + "_timestamp_required") from exc
    if pd.isna(stamp):
        raise SourceManifestContractError(name + "_timestamp_required")
    if stamp.tzinfo is None:
        raise SourceManifestContractError(name + "_timezone_required")
    return stamp.tz_convert(DECISION_TIMEZONE)


def _session_timestamp(value: object) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SourceManifestContractError("source_manifest_session_required") from exc
    if pd.isna(stamp):
        raise SourceManifestContractError("source_manifest_session_required")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(DECISION_TIMEZONE).tz_localize(None)
    return stamp.normalize()


def _fixed_decision_at(session: pd.Timestamp, value: object) -> pd.Timestamp:
    decision = _local_timestamp(value, "source_manifest_decision_at")
    expected = pd.Timestamp(session.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(hours=18)
    if decision != expected:
        raise SourceManifestContractError("source_manifest_decision_at_must_equal_session_1800")
    return decision


class RegisteredEastmoneyRawTransport:
    """A no-network validator bound to exactly one registered source manifest."""

    def __init__(
        self,
        manifest: EastmoneyEodSourceManifest,
        registration: SourceManifestRegistration,
    ):
        if registration.manifest_sha256 != manifest.manifest_sha256:
            raise SourceManifestContractError("source_manifest_registration_mismatch")
        self._manifest = manifest
        self._registration = registration

    def prepare_collection(
        self,
        *,
        session: object,
        decision_at: object,
        attempts: Sequence[RawEastmoneyTransportAttempt],
    ) -> PreparedRawEastmoneyCollection:
        day = _session_timestamp(session)
        cutoff = _fixed_decision_at(day, decision_at)
        if isinstance(attempts, (str, bytes)) or not attempts:
            raise SourceManifestContractError("source_manifest_attempts_required")
        registered_at = _local_timestamp(
            self._registration.recorded_at,
            "source_manifest_registration",
        )
        expected_collection_id: str | None = None
        seen_pages: set[int] = set()
        capture_pages: list[Any] = []
        for attempt in attempts:
            if not isinstance(attempt, RawEastmoneyTransportAttempt):
                raise SourceManifestContractError("source_manifest_raw_attempt_required")
            collection_id = _text(attempt.collection_id, "source_manifest_collection_id")
            if expected_collection_id is None:
                expected_collection_id = collection_id
            elif collection_id != expected_collection_id:
                raise SourceManifestContractError("source_manifest_collection_id_mismatch")
            if type(attempt.page_number) is not int or attempt.page_number < 1:
                raise SourceManifestContractError("source_manifest_page_number_invalid")
            if attempt.page_number in seen_pages:
                raise SourceManifestContractError("source_manifest_duplicate_page_number")
            seen_pages.add(attempt.page_number)
            if attempt.canonical_url != self._manifest.canonical_url(attempt.page_number):
                raise SourceManifestContractError("source_manifest_page_url_mismatch")
            if attempt.transport_kind != "raw_https_get":
                raise SourceManifestContractError("source_manifest_transport_kind_invalid")
            if attempt.cache_reused is not False:
                raise SourceManifestContractError("source_manifest_cache_forbidden")
            if attempt.fallback_attempted is not False:
                raise SourceManifestContractError("source_manifest_fallback_forbidden")
            if attempt.content_type != "application/json":
                raise SourceManifestContractError("source_manifest_content_type_invalid")
            if attempt.raw is None:
                if not isinstance(attempt.fetch_error, str) or not attempt.fetch_error.strip():
                    raise SourceManifestContractError("source_manifest_fetch_error_required")
            elif not isinstance(attempt.raw, bytes) or not attempt.raw:
                raise SourceManifestContractError("source_manifest_raw_bytes_required")
            elif attempt.fetch_error is not None:
                raise SourceManifestContractError("source_manifest_received_page_has_fetch_error")
            captured = _local_timestamp(attempt.captured_at, "source_manifest_captured_at")
            if registered_at > captured:
                raise SourceManifestContractError("source_manifest_registration_after_capture")
            provider_published_at = None
            if attempt.provider_published_at is not None:
                provider_published_at = _local_timestamp(
                    attempt.provider_published_at,
                    "source_manifest_provider_published_at",
                )
                if provider_published_at > captured:
                    raise SourceManifestContractError("source_manifest_provider_published_after_capture")
            # Import lazily so this source-contract module never turns the
            # existing capture adapter into an import dependency at startup.
            from quantlab.forward_market_capture import EastmoneyRawPage

            capture_pages.append(
                EastmoneyRawPage(
                    page_number=attempt.page_number,
                    canonical_url=attempt.canonical_url,
                    raw=attempt.raw,
                    captured_at=captured,
                    fetch_error=attempt.fetch_error,
                    provider_published_at=provider_published_at,
                    content_type=attempt.content_type,
                )
            )
        return PreparedRawEastmoneyCollection(
            session=day.date(),
            decision_at=cutoff,
            collection_id=expected_collection_id or "",
            manifest_sha256=self._manifest.manifest_sha256,
            manifest_event_sha256=self._registration.event_sha256,
            capture_pages=tuple(sorted(capture_pages, key=lambda page: page.page_number)),
        )


class SourceManifestJournal:
    """A narrow append-only journal for source contracts, not raw market data."""

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        if self._root.is_symlink() or not self._root.is_dir():
            raise SourceManifestContractError("source_manifest_root_invalid")
        self._path = self._root / "forward-market-source-manifests.v1.jsonl"
        if self._path.exists() and (self._path.is_symlink() or not self._path.is_file()):
            raise SourceManifestContractError("source_manifest_journal_path_invalid")

    @property
    def path(self) -> Path:
        return self._path

    def _records(self) -> tuple[SourceManifestRegistration, ...]:
        if not self._path.exists():
            return ()
        if self._path.is_symlink() or not self._path.is_file():
            raise SourceManifestContractError("source_manifest_journal_path_invalid")
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SourceManifestContractError("source_manifest_journal_unreadable") from exc
        records: list[SourceManifestRegistration] = []
        previous = _ZERO_SHA256
        for sequence, line in enumerate(lines, start=1):
            if not line:
                raise SourceManifestContractError("source_manifest_journal_blank_record")
            document = _strict_json_object(line, "source_manifest_record")
            required = {
                "schema_version",
                "sequence",
                "recorded_at",
                "previous_event_sha256",
                "manifest",
                "manifest_sha256",
                "event_sha256",
            }
            if set(document) != required or document.get("schema_version") != SOURCE_MANIFEST_RECORD_SCHEMA:
                raise SourceManifestContractError("source_manifest_record_schema_invalid")
            if type(document.get("sequence")) is not int or document["sequence"] != sequence:
                raise SourceManifestContractError("source_manifest_record_sequence_invalid")
            recorded_at = _timestamp_text(document.get("recorded_at"), "source_manifest_recorded_at")
            if document.get("recorded_at") != recorded_at:
                raise SourceManifestContractError("source_manifest_recorded_at_not_canonical")
            if _strict_sha256(document.get("previous_event_sha256"), "source_manifest_previous_event") != previous:
                raise SourceManifestContractError("source_manifest_chain_mismatch")
            manifest = EastmoneyEodSourceManifest.from_mapping(document.get("manifest"))
            manifest_sha = _strict_sha256(document.get("manifest_sha256"), "source_manifest")
            if manifest_sha != manifest.manifest_sha256:
                raise SourceManifestContractError("source_manifest_record_hash_mismatch")
            event_payload = {key: document[key] for key in required - {"event_sha256"}}
            event_sha = _strict_sha256(document.get("event_sha256"), "source_manifest_event")
            if event_sha != _sha256(event_payload, "source_manifest_event"):
                raise SourceManifestContractError("source_manifest_event_hash_mismatch")
            records.append(
                SourceManifestRegistration(
                    sequence=sequence,
                    recorded_at=recorded_at,
                    previous_event_sha256=previous,
                    manifest_sha256=manifest_sha,
                    event_sha256=event_sha,
                    manifest=manifest.to_mapping(),
                )
            )
            previous = event_sha
        return tuple(records)

    def verify(self) -> dict[str, object]:
        records = self._records()
        return {
            "verified": True,
            "records": len(records),
            "head_event_sha256": records[-1].event_sha256 if records else _ZERO_SHA256,
        }

    def verify_registration_binding(
        self,
        *,
        manifest_sha256: object,
        manifest_event_sha256: object,
    ) -> SourceManifestRegistration:
        """Return the exact registered profile bound to a prepared collection.

        A page set that merely resembles the public endpoint is not enough:
        downstream materialisation must prove both the immutable manifest
        identity and the specific append-only registration event.
        """
        manifest_sha = _strict_sha256(manifest_sha256, "source_manifest")
        event_sha = _strict_sha256(manifest_event_sha256, "source_manifest_event")
        for record in self._records():
            if record.manifest_sha256 == manifest_sha and record.event_sha256 == event_sha:
                return record
        raise SourceManifestContractError("source_manifest_registration_binding_not_found")

    def register(
        self, manifest: EastmoneyEodSourceManifest, *, recorded_at: object
    ) -> SourceManifestRegistration:
        if not isinstance(manifest, EastmoneyEodSourceManifest):
            raise SourceManifestContractError("eastmoney_source_manifest_required")
        recorded = _timestamp_text(recorded_at, "source_manifest_recorded_at")
        records = self._records()
        for record in records:
            same_identity = (
                record.manifest["source_id"] == manifest.source_id
                and record.manifest["source_profile_version"] == manifest.source_profile_version
            )
            if same_identity and record.manifest_sha256 == manifest.manifest_sha256:
                return record
            if same_identity:
                raise SourceManifestContractError("source_manifest_identity_version_already_registered")
        previous = records[-1].event_sha256 if records else _ZERO_SHA256
        document: dict[str, object] = {
            "schema_version": SOURCE_MANIFEST_RECORD_SCHEMA,
            "sequence": len(records) + 1,
            "recorded_at": recorded,
            "previous_event_sha256": previous,
            "manifest": manifest.to_mapping(),
            "manifest_sha256": manifest.manifest_sha256,
        }
        document["event_sha256"] = _sha256(document, "source_manifest_event")
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical_json_bytes(document, "source_manifest_record").decode("utf-8") + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise SourceManifestContractError("source_manifest_journal_append_failed") from exc
        return SourceManifestRegistration(
            sequence=int(document["sequence"]),
            recorded_at=recorded,
            previous_event_sha256=previous,
            manifest_sha256=manifest.manifest_sha256,
            event_sha256=str(document["event_sha256"]),
            manifest=manifest.to_mapping(),
        )

    def open_raw_transport(
        self, manifest: EastmoneyEodSourceManifest
    ) -> RegisteredEastmoneyRawTransport:
        if not isinstance(manifest, EastmoneyEodSourceManifest):
            raise SourceManifestContractError("eastmoney_source_manifest_required")
        for record in self._records():
            if record.manifest_sha256 == manifest.manifest_sha256:
                return RegisteredEastmoneyRawTransport(manifest, record)
        raise SourceManifestContractError("source_manifest_not_registered")
