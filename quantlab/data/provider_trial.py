"""Fail-closed qualification records for external data-provider trials.

This module deliberately does *not* fetch a provider, persist any source
material, train a model, or invoke an account.  It turns a vendor's written
trial evidence and a small set of already-captured sample-receipt identities
into one deterministic ``qualified`` record, or rejects the submission.

The boundary is intentionally narrow.  A qualified provider trial says only
that the declared source is eligible for the next, separately controlled
forward-collection step.  It does not establish data quality, signal value,
or permission to run an account experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import re
from typing import Any

from quantlab.data.l2_contract import L2_FEATURE_KEYS


PROVIDER_TRIAL_SCHEMA_VERSION = "quantlab.provider-trial-qualification.v1"
SOURCE_CLASSES = ("theme_market", "text", "l2")


class ProviderTrialError(ValueError):
    """A provider trial is incomplete or ambiguous and must not be admitted."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AMBIGUOUS_TEXT = frozenset({"", "-", "n/a", "na", "none", "null", "tbd", "unknown", "unspecified"})


@dataclass(frozen=True)
class AcceptanceRequirement:
    """One stable, human-readable condition attached to a qualification."""

    requirement_id: str
    description: str


_COMMON_REQUIREMENTS = (
    AcceptanceRequirement(
        "provider_trial_identity",
        "Provider, trial, dataset and release identities are explicit and versioned.",
    ),
    AcceptanceRequirement(
        "terms_identity",
        "The governing contract or terms have an immutable identity hash and locator.",
    ),
    AcceptanceRequirement(
        "automated_access_permission",
        "Automated access is explicitly granted with hashed permission evidence.",
    ),
    AcceptanceRequirement(
        "raw_retention_permission",
        "Persistent retention of raw receipts is explicitly granted with hashed permission evidence.",
    ),
    AcceptanceRequirement(
        "sample_receipt_identity",
        "Every sample claim is tied to a unique, hashed raw receipt captured under the trial.",
    ),
)

_SOURCE_REQUIREMENTS = {
    "theme_market": (
        AcceptanceRequirement(
            "theme_capabilities",
            "The source explicitly declares multi-valued stock-theme edges, active-theme semantics, "
            "limit-up, failed-board and daily-bar capability.",
        ),
        AcceptanceRequirement(
            "u0_bar_seed",
            "A 253-session historical-bar seed is identified solely for constructing the current U0 universe.",
        ),
        AcceptanceRequirement(
            "theme_page_reconciliation",
            "All declared market pages and U0 theme coverage reconcile exactly, with no missing or duplicate members.",
        ),
        AcceptanceRequirement(
            "theme_component_receipts",
            "Theme mapping, limit-up, failed-board and market-bar samples each reference a retained raw receipt.",
        ),
    ),
    "text": (
        AcceptanceRequirement(
            "text_capabilities",
            "The source explicitly declares canonical content, original-publisher, timestamp and reprint "
            "identity capability.",
        ),
        AcceptanceRequirement(
            "text_origin_identity",
            "Samples prove publisher and timezone-aware original publication identity for two independent "
            "original sources.",
        ),
        AcceptanceRequirement(
            "text_reprint_identity",
            "A reprint sample explicitly links to a sampled original document and publisher rather than "
            "being counted as independent evidence.",
        ),
    ),
    "l2": (
        AcceptanceRequirement(
            "l2_capabilities",
            "The source explicitly declares a fixed field dictionary, per-field availability, version, raw "
            "receipt and source-as-of capability.",
        ),
        AcceptanceRequirement(
            "l2_fixed_fields",
            "All 14 reviewed QuantMind keys appear exactly once, in the frozen order, with explicit unit, "
            "source field, available-at field, version and receipt hash.",
        ),
        AcceptanceRequirement(
            "l2_dataset_identity",
            "The trial sample names a dataset and release, plus immutable field-dictionary and sample-matrix "
            "identities.",
        ),
    ),
}

_REQUIRED_CAPABILITIES = {
    "theme_market": frozenset(
        {
            "stock_theme_multi_mapping",
            "active_theme_semantics",
            "limit_up_events",
            "failed_board_events",
            "daily_bars",
            "paged_snapshot_reconciliation",
            "u0_membership_coverage",
        }
    ),
    "text": frozenset(
        {
            "canonical_content",
            "original_publisher_identity",
            "published_timestamp",
            "reprint_identity",
        }
    ),
    "l2": frozenset(
        {
            "fixed_14_field_dictionary",
            "per_field_available_at",
            "per_field_source_version",
            "per_field_raw_receipt",
            "source_as_of",
        }
    ),
}


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ProviderTrialError("missing_" + field)
    normalized = value.strip()
    if normalized.lower() in _AMBIGUOUS_TEXT:
        raise ProviderTrialError("missing_or_ambiguous_" + field)
    return normalized


def _sha256(value: object, field: str) -> str:
    digest = _text(value, field)
    if not _SHA256.fullmatch(digest):
        raise ProviderTrialError("invalid_" + field)
    return digest


def _timestamp(value: object, field: str) -> datetime:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderTrialError("invalid_" + field) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProviderTrialError("timezone_required_" + field)
    return parsed


def _date(value: object, field: str) -> date:
    text = _text(value, field)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ProviderTrialError("invalid_" + field) from exc


def _strict_bool(value: object, field: str) -> bool:
    if type(value) is not bool:  # bool subclasses no other native types, keep the contract literal.
        raise ProviderTrialError("boolean_required_" + field)
    return value


def _as_tuple(value: object, field: str) -> tuple:
    if not isinstance(value, (tuple, list)):
        raise ProviderTrialError("sequence_required_" + field)
    return tuple(value)


def _unique(values: tuple[object, ...], field: str) -> None:
    if len(values) != len(set(values)):
        raise ProviderTrialError("duplicate_" + field)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TermsIdentity:
    """Identity of the exact contract or terms used to authorize the trial."""

    contract_id: str
    version: str
    locator: str
    sha256: str

    def validate(self) -> None:
        _text(self.contract_id, "terms_contract_id")
        _text(self.version, "terms_version")
        _text(self.locator, "terms_locator")
        _sha256(self.sha256, "terms_sha256")


@dataclass(frozen=True)
class PermissionDeclaration:
    """One explicit contractual grant, never an inferred permission."""

    permission: str
    granted: bool
    scope: str
    evidence_locator: str
    evidence_sha256: str

    def validate(self, expected_permission: str) -> None:
        if self.permission != expected_permission:
            raise ProviderTrialError("unexpected_permission:" + str(self.permission))
        if not _strict_bool(self.granted, "permission_granted:" + expected_permission):
            raise ProviderTrialError("permission_not_granted:" + expected_permission)
        _text(self.scope, "permission_scope:" + expected_permission)
        _text(self.evidence_locator, "permission_evidence_locator:" + expected_permission)
        _sha256(self.evidence_sha256, "permission_evidence_sha256:" + expected_permission)


@dataclass(frozen=True)
class CapabilityDeclaration:
    """A provider-stated capability with independently retainable evidence."""

    capability: str
    available: bool
    evidence_locator: str
    evidence_sha256: str

    def validate(self) -> None:
        _text(self.capability, "capability")
        if not _strict_bool(self.available, "capability_available:" + str(self.capability)):
            raise ProviderTrialError("capability_not_available:" + str(self.capability))
        _text(self.evidence_locator, "capability_evidence_locator:" + str(self.capability))
        _sha256(self.evidence_sha256, "capability_evidence_sha256:" + str(self.capability))


@dataclass(frozen=True)
class SampleReceipt:
    """Identity of one raw provider response already captured under the trial."""

    receipt_id: str
    raw_sha256: str
    source_locator: str
    captured_at: str

    def validate(self) -> None:
        _text(self.receipt_id, "sample_receipt_id")
        _sha256(self.raw_sha256, "sample_receipt_raw_sha256")
        _text(self.source_locator, "sample_receipt_source_locator")
        _timestamp(self.captured_at, "sample_receipt_captured_at")


@dataclass(frozen=True)
class HistoricalBarSeed:
    """Hashed 253-session seed used only to construct today's U0 membership."""

    seed_id: str
    bar_set_sha256: str
    calendar_sha256: str
    instrument_coverage_sha256: str
    raw_receipt_sha256: str
    session_count: int
    first_session: str
    last_session: str
    purpose: str = "current_u0_only"

    def validate(self) -> None:
        _text(self.seed_id, "historical_bar_seed_id")
        for field in ("bar_set_sha256", "calendar_sha256", "instrument_coverage_sha256", "raw_receipt_sha256"):
            _sha256(getattr(self, field), "historical_bar_seed_" + field)
        if type(self.session_count) is not int or self.session_count != 253:
            raise ProviderTrialError("historical_bar_seed_must_have_exactly_253_sessions")
        first = _date(self.first_session, "historical_bar_seed_first_session")
        last = _date(self.last_session, "historical_bar_seed_last_session")
        if first >= last:
            raise ProviderTrialError("historical_bar_seed_session_range_invalid")
        if self.purpose != "current_u0_only":
            raise ProviderTrialError("historical_bar_seed_purpose_must_be_current_u0_only")

    @property
    def sha256(self) -> str:
        return _canonical_hash(asdict(self))


@dataclass(frozen=True)
class PageReconciliation:
    """A fully reconciled, paginated market snapshot and U0 mapping coverage."""

    snapshot_id: str
    coverage_sha256: str
    expected_page_count: int
    observed_page_numbers: tuple[int, ...]
    page_receipt_sha256s: tuple[str, ...]
    expected_instrument_count: int
    observed_unique_instrument_count: int
    duplicate_instrument_count: int
    u0_membership_sha256: str
    expected_u0_member_count: int
    mapped_u0_member_count: int
    unmapped_u0_member_count: int

    def validate(self) -> None:
        _text(self.snapshot_id, "page_reconciliation_snapshot_id")
        _sha256(self.coverage_sha256, "page_reconciliation_coverage_sha256")
        if type(self.expected_page_count) is not int or self.expected_page_count < 1:
            raise ProviderTrialError("invalid_expected_page_count")
        pages = _as_tuple(self.observed_page_numbers, "observed_page_numbers")
        if any(type(page) is not int for page in pages):
            raise ProviderTrialError("page_number_must_be_int")
        if pages != tuple(range(1, self.expected_page_count + 1)):
            raise ProviderTrialError("page_numbers_not_exactly_reconciled")
        hashes = _as_tuple(self.page_receipt_sha256s, "page_receipt_sha256s")
        if len(hashes) != self.expected_page_count:
            raise ProviderTrialError("page_receipt_count_mismatch")
        for digest in hashes:
            _sha256(digest, "page_receipt_sha256")
        _unique(hashes, "page_receipt_sha256")
        counts = (
            ("expected_instrument_count", self.expected_instrument_count),
            ("observed_unique_instrument_count", self.observed_unique_instrument_count),
            ("duplicate_instrument_count", self.duplicate_instrument_count),
            ("expected_u0_member_count", self.expected_u0_member_count),
            ("mapped_u0_member_count", self.mapped_u0_member_count),
            ("unmapped_u0_member_count", self.unmapped_u0_member_count),
        )
        if any(type(value) is not int or value < 0 for _, value in counts):
            raise ProviderTrialError("invalid_page_reconciliation_count")
        if (
            self.expected_instrument_count < 1
            or self.observed_unique_instrument_count != self.expected_instrument_count
        ):
            raise ProviderTrialError("market_page_instrument_count_not_reconciled")
        if self.duplicate_instrument_count != 0:
            raise ProviderTrialError("market_page_duplicate_instruments")
        _sha256(self.u0_membership_sha256, "u0_membership_sha256")
        if self.expected_u0_member_count < 1 or self.expected_u0_member_count > self.expected_instrument_count:
            raise ProviderTrialError("invalid_expected_u0_member_count")
        if self.mapped_u0_member_count != self.expected_u0_member_count or self.unmapped_u0_member_count != 0:
            raise ProviderTrialError("u0_theme_coverage_not_reconciled")


@dataclass(frozen=True)
class ThemeMarketEvidence:
    """Provider-specific evidence required for the four-component theme signal."""

    historical_bar_seed: HistoricalBarSeed
    page_reconciliation: PageReconciliation
    theme_mapping_receipt_sha256: str
    limit_up_receipt_sha256: str
    failed_board_receipt_sha256: str
    market_bar_receipt_sha256: str

    def validate(self, sample_hashes: frozenset[str]) -> None:
        if not isinstance(self.historical_bar_seed, HistoricalBarSeed):
            raise ProviderTrialError("theme_historical_bar_seed_required")
        if not isinstance(self.page_reconciliation, PageReconciliation):
            raise ProviderTrialError("theme_page_reconciliation_required")
        self.historical_bar_seed.validate()
        self.page_reconciliation.validate()
        hashes = {
            self.historical_bar_seed.raw_receipt_sha256,
            *self.page_reconciliation.page_receipt_sha256s,
        }
        for field in (
            "theme_mapping_receipt_sha256",
            "limit_up_receipt_sha256",
            "failed_board_receipt_sha256",
            "market_bar_receipt_sha256",
        ):
            digest = _sha256(getattr(self, field), "theme_" + field)
            hashes.add(digest)
        absent = sorted(hashes - sample_hashes)
        if absent:
            raise ProviderTrialError("theme_evidence_receipt_not_in_trial_samples:" + ",".join(absent))


@dataclass(frozen=True)
class TextSampleEvidence:
    """One text item whose origin and reprint status can be audited later."""

    document_id: str
    raw_receipt_sha256: str
    content_sha256: str
    canonical_url: str
    publisher_id: str
    original_publisher_id: str
    published_at: str
    original_published_at: str
    is_reprint: bool
    reprint_of_document_id: str | None = None
    reprint_of_canonical_url: str | None = None

    def validate(self, sample_hashes: frozenset[str]) -> None:
        _text(self.document_id, "text_document_id")
        receipt_hash = _sha256(self.raw_receipt_sha256, "text_raw_receipt_sha256")
        if receipt_hash not in sample_hashes:
            raise ProviderTrialError("text_receipt_not_in_trial_samples")
        _sha256(self.content_sha256, "text_content_sha256")
        _text(self.canonical_url, "text_canonical_url")
        publisher = _text(self.publisher_id, "text_publisher_id")
        original_publisher = _text(self.original_publisher_id, "text_original_publisher_id")
        published_at = _timestamp(self.published_at, "text_published_at")
        original_at = _timestamp(self.original_published_at, "text_original_published_at")
        is_reprint = _strict_bool(self.is_reprint, "text_is_reprint")
        if is_reprint:
            _text(self.reprint_of_document_id, "text_reprint_of_document_id")
            _text(self.reprint_of_canonical_url, "text_reprint_of_canonical_url")
            if original_at > published_at:
                raise ProviderTrialError("text_reprint_precedes_original")
        else:
            if publisher != original_publisher:
                raise ProviderTrialError("text_original_publisher_identity_ambiguous")
            if published_at != original_at:
                raise ProviderTrialError("text_original_publication_time_ambiguous")
            if self.reprint_of_document_id is not None or self.reprint_of_canonical_url is not None:
                raise ProviderTrialError("text_original_cannot_have_reprint_link")


@dataclass(frozen=True)
class TextEvidence:
    """A minimum sample that proves independent-origin and reprint handling."""

    samples: tuple[TextSampleEvidence, ...]

    def validate(self, sample_hashes: frozenset[str]) -> None:
        samples = _as_tuple(self.samples, "text_samples")
        if len(samples) < 3:
            raise ProviderTrialError("text_requires_two_originals_and_one_reprint_sample")
        document_ids: list[str] = []
        originals: dict[str, TextSampleEvidence] = {}
        reprints: list[TextSampleEvidence] = []
        for sample in samples:
            if not isinstance(sample, TextSampleEvidence):
                raise ProviderTrialError("invalid_text_sample_evidence")
            sample.validate(sample_hashes)
            document_id = _text(sample.document_id, "text_document_id")
            document_ids.append(document_id)
            if sample.is_reprint:
                reprints.append(sample)
            else:
                originals[document_id] = sample
        _unique(tuple(document_ids), "text_document_id")
        original_publishers = {sample.original_publisher_id for sample in originals.values()}
        if len(originals) < 2 or len(original_publishers) < 2:
            raise ProviderTrialError("text_independent_original_publishers_not_proven")
        # A byte-identical canonical article presented under two publisher
        # labels is a syndication/reprint candidate, not two independent
        # original sources.  The forward ledger repeats this guard for raw
        # captures; qualification must not grant it a pass beforehand merely
        # because the provider used distinct document IDs.
        original_content_hashes = [sample.content_sha256.lower() for sample in originals.values()]
        if len(set(original_content_hashes)) != len(original_content_hashes):
            raise ProviderTrialError("text_independent_original_content_not_proven")
        if not reprints:
            raise ProviderTrialError("text_reprint_sample_required")
        for reprint in reprints:
            original = originals.get(str(reprint.reprint_of_document_id))
            if original is None:
                raise ProviderTrialError("text_reprint_original_not_in_trial_samples")
            if reprint.original_publisher_id != original.original_publisher_id:
                raise ProviderTrialError("text_reprint_original_publisher_mismatch")
            if _timestamp(reprint.original_published_at, "text_original_published_at") != _timestamp(
                original.original_published_at, "text_original_published_at"
            ):
                raise ProviderTrialError("text_reprint_original_time_mismatch")
            if str(reprint.reprint_of_canonical_url) != original.canonical_url:
                raise ProviderTrialError("text_reprint_original_url_mismatch")


@dataclass(frozen=True)
class L2FieldEvidence:
    """Metadata for one reviewed QuantMind field supplied by the trial source."""

    key: str
    unit: str
    source_field: str
    available_at_field: str
    source_version: str
    raw_receipt_sha256: str
    source_as_of_field: str
    missing_policy: str = "abstain"

    def validate(self, sample_hashes: frozenset[str]) -> None:
        _text(self.key, "l2_feature_key")
        _text(self.unit, "l2_feature_unit:" + self.key)
        _text(self.source_field, "l2_feature_source_field:" + self.key)
        if self.available_at_field != "available_at":
            raise ProviderTrialError("l2_feature_available_at_field_must_be_available_at:" + self.key)
        _text(self.source_version, "l2_feature_source_version:" + self.key)
        digest = _sha256(self.raw_receipt_sha256, "l2_feature_raw_receipt_sha256:" + self.key)
        if digest not in sample_hashes:
            raise ProviderTrialError("l2_feature_receipt_not_in_trial_samples:" + self.key)
        if self.source_as_of_field != "source_as_of":
            raise ProviderTrialError("l2_feature_source_as_of_field_must_be_source_as_of:" + self.key)
        if self.missing_policy != "abstain":
            raise ProviderTrialError("l2_feature_missing_policy_must_abstain:" + self.key)


@dataclass(frozen=True)
class L2Evidence:
    """Immutable L2 trial dictionary and sample-matrix identity."""

    dataset_id: str
    source_release_id: str
    field_dictionary_sha256: str
    sample_matrix_sha256: str
    fields: tuple[L2FieldEvidence, ...]

    def validate(self, sample_hashes: frozenset[str]) -> None:
        _text(self.dataset_id, "l2_dataset_id")
        _text(self.source_release_id, "l2_source_release_id")
        _sha256(self.field_dictionary_sha256, "l2_field_dictionary_sha256")
        _sha256(self.sample_matrix_sha256, "l2_sample_matrix_sha256")
        fields = _as_tuple(self.fields, "l2_fields")
        if not all(isinstance(field, L2FieldEvidence) for field in fields):
            raise ProviderTrialError("invalid_l2_field_evidence")
        keys = tuple(field.key for field in fields)
        if keys != L2_FEATURE_KEYS:
            raise ProviderTrialError("l2_fields_must_match_fixed_14_key_order")
        for field in fields:
            field.validate(sample_hashes)


@dataclass(frozen=True)
class ProviderTrialSubmission:
    """All evidence supplied before a source can enter forward collection."""

    source_class: str
    provider_id: str
    trial_id: str
    source_dataset_id: str
    source_release_id: str
    source_release_sha256: str
    terms: TermsIdentity
    automated_access: PermissionDeclaration
    raw_retention: PermissionDeclaration
    capabilities: tuple[CapabilityDeclaration, ...]
    sample_receipts: tuple[SampleReceipt, ...]
    theme_market: ThemeMarketEvidence | None = None
    text: TextEvidence | None = None
    l2: L2Evidence | None = None
    schema_version: str = PROVIDER_TRIAL_SCHEMA_VERSION

    @property
    def sha256(self) -> str:
        return _canonical_hash(asdict(self))


@dataclass(frozen=True)
class QualifiedProviderTrial:
    """The sole successful outcome of ``qualify_provider_trial``."""

    submission: ProviderTrialSubmission
    requirement_ids: tuple[str, ...]
    status: str = "qualified"
    schema_version: str = PROVIDER_TRIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status != "qualified":
            raise ProviderTrialError("qualified_provider_trial_status_must_be_qualified")
        if self.schema_version != PROVIDER_TRIAL_SCHEMA_VERSION:
            raise ProviderTrialError("qualified_provider_trial_schema_version_mismatch")

    @property
    def sha256(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "status": self.status,
                "submission_sha256": self.submission.sha256,
                "requirement_ids": list(self.requirement_ids),
            }
        )

    def to_record(self) -> dict[str, Any]:
        """Return a persistence-ready, deterministic record without doing I/O."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "provider_id": self.submission.provider_id,
            "trial_id": self.submission.trial_id,
            "source_class": self.submission.source_class,
            "submission": asdict(self.submission),
            "submission_sha256": self.submission.sha256,
            "acceptance_requirement_ids": list(self.requirement_ids),
            "qualification_sha256": self.sha256,
        }


def acceptance_requirements(source_class: str) -> tuple[AcceptanceRequirement, ...]:
    """Return the complete, frozen acceptance checklist for one source class."""
    if source_class not in SOURCE_CLASSES:
        raise ProviderTrialError("unknown_source_class:" + str(source_class))
    return _COMMON_REQUIREMENTS + _SOURCE_REQUIREMENTS[source_class]


def _validate_common(submission: ProviderTrialSubmission) -> frozenset[str]:
    if not isinstance(submission, ProviderTrialSubmission):
        raise ProviderTrialError("provider_trial_submission_required")
    if submission.source_class not in SOURCE_CLASSES:
        raise ProviderTrialError("unknown_source_class:" + str(submission.source_class))
    for field in ("provider_id", "trial_id", "source_dataset_id", "source_release_id"):
        _text(getattr(submission, field), field)
    _sha256(submission.source_release_sha256, "source_release_sha256")
    if submission.schema_version != PROVIDER_TRIAL_SCHEMA_VERSION:
        raise ProviderTrialError("provider_trial_schema_version_mismatch")
    if not isinstance(submission.terms, TermsIdentity):
        raise ProviderTrialError("terms_identity_required")
    submission.terms.validate()
    if not isinstance(submission.automated_access, PermissionDeclaration):
        raise ProviderTrialError("automated_access_permission_required")
    if not isinstance(submission.raw_retention, PermissionDeclaration):
        raise ProviderTrialError("raw_retention_permission_required")
    submission.automated_access.validate("automated_access")
    submission.raw_retention.validate("raw_retention")

    capabilities = _as_tuple(submission.capabilities, "capabilities")
    expected_capabilities = _REQUIRED_CAPABILITIES[submission.source_class]
    declared_names: list[str] = []
    for capability in capabilities:
        if not isinstance(capability, CapabilityDeclaration):
            raise ProviderTrialError("invalid_capability_declaration")
        capability.validate()
        declared_names.append(capability.capability)
    _unique(tuple(declared_names), "capability_declaration")
    if frozenset(declared_names) != expected_capabilities:
        missing = sorted(expected_capabilities - frozenset(declared_names))
        extra = sorted(frozenset(declared_names) - expected_capabilities)
        detail = ",".join(("missing=" + ";".join(missing), "extra=" + ";".join(extra)))
        raise ProviderTrialError("capability_declarations_must_match_exactly:" + detail)

    receipts = _as_tuple(submission.sample_receipts, "sample_receipts")
    if not receipts:
        raise ProviderTrialError("sample_receipts_required")
    receipt_ids: list[str] = []
    receipt_hashes: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, SampleReceipt):
            raise ProviderTrialError("invalid_sample_receipt")
        receipt.validate()
        receipt_ids.append(receipt.receipt_id)
        receipt_hashes.append(receipt.raw_sha256)
    _unique(tuple(receipt_ids), "sample_receipt_id")
    _unique(tuple(receipt_hashes), "sample_receipt_raw_sha256")
    return frozenset(receipt_hashes)


def _validate_source_specific(submission: ProviderTrialSubmission, sample_hashes: frozenset[str]) -> None:
    evidence = {
        "theme_market": submission.theme_market,
        "text": submission.text,
        "l2": submission.l2,
    }
    selected = evidence[submission.source_class]
    if selected is None:
        raise ProviderTrialError("source_class_evidence_required:" + submission.source_class)
    supplied_elsewhere = sorted(
        name for name, value in evidence.items() if name != submission.source_class and value is not None
    )
    if supplied_elsewhere:
        raise ProviderTrialError("cross_class_evidence_forbidden:" + ",".join(supplied_elsewhere))
    if submission.source_class == "theme_market":
        if not isinstance(selected, ThemeMarketEvidence):
            raise ProviderTrialError("invalid_theme_market_evidence")
        selected.validate(sample_hashes)
    elif submission.source_class == "text":
        if not isinstance(selected, TextEvidence):
            raise ProviderTrialError("invalid_text_evidence")
        selected.validate(sample_hashes)
    else:
        if not isinstance(selected, L2Evidence):
            raise ProviderTrialError("invalid_l2_evidence")
        selected.validate(sample_hashes)


def qualify_provider_trial(submission: ProviderTrialSubmission) -> QualifiedProviderTrial:
    """Validate a complete trial submission or fail closed with one explicit reason.

    The function has no network, credential, ledger, model, or account side
    effects.  Callers must persist the returned record separately if they want
    it in a research ledger.
    """
    sample_hashes = _validate_common(submission)
    _validate_source_specific(submission, sample_hashes)
    requirements = acceptance_requirements(submission.source_class)
    return QualifiedProviderTrial(submission, tuple(item.requirement_id for item in requirements))


__all__ = [
    "AcceptanceRequirement",
    "CapabilityDeclaration",
    "HistoricalBarSeed",
    "L2Evidence",
    "L2FieldEvidence",
    "PROVIDER_TRIAL_SCHEMA_VERSION",
    "PageReconciliation",
    "PermissionDeclaration",
    "ProviderTrialError",
    "ProviderTrialSubmission",
    "QualifiedProviderTrial",
    "SOURCE_CLASSES",
    "SampleReceipt",
    "TermsIdentity",
    "TextEvidence",
    "TextSampleEvidence",
    "ThemeMarketEvidence",
    "acceptance_requirements",
    "qualify_provider_trial",
]
