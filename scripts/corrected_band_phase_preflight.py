#!/usr/bin/env python3
"""Pure D2-v2 schedule preflight for the corrected broad-basket identity.

This module deliberately has no market-data, target-file, ledger, account, or
network dependencies.  A caller supplies a complete trading calendar, the
original phase-0 decision dates, and one target-reconstruction callback per
phase.  The preflight proves that each decision date was moved over the full
calendar, records tail omissions, and fail-closes when the reconstructed
target identities cannot distinguish the phases.

It is not an account runner.  ``account_replay_started`` is always false.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from typing import Any


DEFAULT_PHASES = (0, 4, 8, 12, 16)
CORRECTED_SELECTION_IDENTITY = ("-financial_score", "liquidity_rank")
# Both the archived target artifact and every reconstructed phase artifact
# must use this exact representation: SHA-256 over UTF-8 canonical JSON made
# with sorted keys, compact separators, and no NaN values. Raw ``targets.json``
# file bytes and an in-memory payload serialization are not interchangeable
# hashes, even when they describe the same targets.
CANONICAL_TARGET_ARTIFACT_HASH_SCHEME = "sha256:canonical-json-target-artifact-v1"


@dataclass(frozen=True)
class DecisionMove:
    """One original decision date and its phase-specific reconstruction date."""

    source_date: date
    scheduled_date: date | None
    reconstruction_date: date | None
    omission_reason: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "source_date": self.source_date.isoformat(),
            "scheduled_date": None if self.scheduled_date is None else self.scheduled_date.isoformat(),
            "reconstruction_date": (
                None if self.reconstruction_date is None else self.reconstruction_date.isoformat()
            ),
            "omission_reason": self.omission_reason,
        }


@dataclass(frozen=True)
class PhaseReconstruction:
    """The minimal evidence a caller must return after rebuilding a phase.

    ``target_artifact`` is required. The preflight calculates its canonical
    hash itself and uses that computed value for every comparison. A supplied
    ``target_hash`` is retained only as an optional audit assertion and must
    match the computed value. ``target_hash_scheme`` must name the canonical
    artifact representation used for both this value and the archived
    comparison artifact. The explicit identity prevents an old input-order /
    ticker-order reconstruction from being accepted as the corrected
    liquidity-order one.
    """

    target_hash: str | None
    selection_identity: tuple[str, ...]
    target_hash_scheme: str | None = None
    target_artifact: Any | None = None


@dataclass(frozen=True)
class PhaseTrace:
    """All scheduling evidence and the reconstruction outcome for one phase."""

    phase: int
    moves: tuple[DecisionMove, ...]
    target_hash: str | None
    selection_identity: tuple[str, ...] | None
    target_hash_scheme: str | None = None
    declared_target_hash: str | None = None
    reconstruction_error: str | None = None

    @property
    def tail_overflow_count(self) -> int:
        return sum(move.omission_reason == "tail_overflow" for move in self.moves)

    @property
    def scheduled_moves(self) -> tuple[DecisionMove, ...]:
        return tuple(move for move in self.moves if move.scheduled_date is not None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "moves": [move.as_dict() for move in self.moves],
            "target_hash": self.target_hash,
            "declared_target_hash": self.declared_target_hash,
            "selection_identity": (
                None if self.selection_identity is None else list(self.selection_identity)
            ),
            "target_hash_scheme": self.target_hash_scheme,
            "tail_overflow_count": self.tail_overflow_count,
            "reconstruction_error": self.reconstruction_error,
        }


@dataclass(frozen=True)
class PreflightVerdict:
    """A no-account result suitable for a later frozen reconstruction step."""

    status: str
    reasons: tuple[str, ...]
    target_hashes: tuple[tuple[int, str | None], ...]
    account_replay_started: bool = False
    required_target_hash_scheme: str = CANONICAL_TARGET_ARTIFACT_HASH_SCHEME
    archived_target_hash: str | None = None
    archived_declared_target_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "target_hashes": {str(phase): target_hash for phase, target_hash in self.target_hashes},
            "account_replay_started": self.account_replay_started,
            "required_target_hash_scheme": self.required_target_hash_scheme,
            "archived_target_hash": self.archived_target_hash,
            "archived_declared_target_hash": self.archived_declared_target_hash,
        }


@dataclass(frozen=True)
class PhasePreflightResult:
    """The reusable trace/verdict object returned by :func:`preflight_phase_schedule`."""

    traces: tuple[PhaseTrace, ...]
    verdict: PreflightVerdict

    def trace_for(self, phase: int) -> PhaseTrace:
        for trace in self.traces:
            if trace.phase == phase:
                return trace
        raise KeyError(f"phase_not_in_preflight:{phase}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace": [trace.as_dict() for trace in self.traces],
            "verdict": self.verdict.as_dict(),
        }


PhaseReconstructor = Callable[[tuple[DecisionMove, ...]], PhaseReconstruction | Mapping[str, Any]]


def preflight_phase_schedule(
    calendar: Sequence[object],
    source_decision_dates: Sequence[object],
    reconstructors: Mapping[int, PhaseReconstructor],
    *,
    archived_target_hash: str | None = None,
    archived_target_hash_scheme: str | None = None,
    archived_target_artifact: Any | None = None,
    phases: Sequence[int] = DEFAULT_PHASES,
    required_selection_identity: Sequence[str] = CORRECTED_SELECTION_IDENTITY,
) -> PhasePreflightResult:
    """Build and validate a D2-v2 phase schedule without replaying an account.

    Each phase maps every original decision date at calendar position ``i`` to
    position ``i + phase``.  That is intentionally different from slicing a
    pre-existing decision schedule (the D2-v1 failure mode).  A shifted date
    past the calendar end is retained as a ``tail_overflow`` trace row and is
    not passed to the reconstruction callback.

    ``reconstructors`` must provide one callback for every requested phase.  A
    callback receives only that phase's valid reconstructed dates and returns a
    target artifact plus the selection identity used to construct it. The
    preflight hashes the artifact itself. The result is blocked when:

    * phase 0 does not use ``(-financial_score, liquidity_rank)``;
    * either side omits, mismatches, or uses a noncanonical target-artifact
      hash scheme;
    * either the archived or reconstructed target artifact is absent or not
      canonical JSON;
    * phase 0 reuses the archived wrong-order target hash under the same
      canonical scheme;
    * any phase uses a different selection identity; or
    * any two phases report the same reconstructed target hash.

    The function itself performs no I/O and never starts an account replay.
    """

    normalized_calendar = _normalise_calendar(calendar)
    normalized_sources = _normalise_source_dates(source_decision_dates, normalized_calendar)
    normalized_phases = _normalise_phases(phases)
    expected_identity = _normalise_identity(required_selection_identity)
    normalized_archived_declared_hash = _normalise_optional_declared_hash(archived_target_hash)
    normalized_archived_hash_scheme = _normalise_optional_hash_scheme(archived_target_hash_scheme)
    _validate_reconstructors(reconstructors, normalized_phases)

    calendar_positions = {day: index for index, day in enumerate(normalized_calendar)}
    traces: list[PhaseTrace] = []
    reasons: list[str] = []
    computed_archived_hash = _compute_archived_target_hash(archived_target_artifact, reasons)
    if (
        computed_archived_hash is not None
        and normalized_archived_declared_hash is not None
        and normalized_archived_declared_hash != computed_archived_hash
    ):
        reasons.append("archived_declared_target_hash_mismatch")

    for phase in normalized_phases:
        moves = tuple(
            _move_source_date(source_day, normalized_calendar, calendar_positions[source_day], phase)
            for source_day in normalized_sources
        )
        scheduled_moves = tuple(move for move in moves if move.scheduled_date is not None)
        if not scheduled_moves:
            traces.append(
                PhaseTrace(
                    phase=phase,
                    moves=moves,
                    target_hash=None,
                    selection_identity=None,
                    target_hash_scheme=None,
                    declared_target_hash=None,
                    reconstruction_error="no_reconstructible_decisions",
                )
            )
            reasons.append(f"phase{phase}_has_no_reconstructible_decisions")
            continue

        try:
            reconstruction = _coerce_reconstruction(reconstructors[phase](scheduled_moves))
        except Exception as error:  # Fail closed while keeping the complete date trace inspectable.
            traces.append(
                PhaseTrace(
                    phase=phase,
                    moves=moves,
                    target_hash=None,
                    selection_identity=None,
                    target_hash_scheme=None,
                    declared_target_hash=None,
                    reconstruction_error=type(error).__name__,
                )
            )
            reasons.append(f"phase{phase}_reconstruction_failed:{type(error).__name__}")
            continue

        computed_target_hash, artifact_error = _compute_phase_target_hash(reconstruction)
        if artifact_error is not None:
            traces.append(
                PhaseTrace(
                    phase=phase,
                    moves=moves,
                    target_hash=None,
                    selection_identity=reconstruction.selection_identity,
                    target_hash_scheme=reconstruction.target_hash_scheme,
                    declared_target_hash=reconstruction.target_hash,
                    reconstruction_error=artifact_error,
                )
            )
            reasons.append(f"phase{phase}_{artifact_error}")
            continue
        assert computed_target_hash is not None
        if reconstruction.target_hash is not None and reconstruction.target_hash != computed_target_hash:
            reasons.append(f"phase{phase}_declared_target_hash_mismatch")
        traces.append(
            PhaseTrace(
                phase=phase,
                moves=moves,
                target_hash=computed_target_hash,
                selection_identity=reconstruction.selection_identity,
                target_hash_scheme=reconstruction.target_hash_scheme,
                declared_target_hash=reconstruction.target_hash,
            )
        )

    _append_identity_reasons(traces, expected_identity, reasons)
    if (
        computed_archived_hash is not None
        and _append_hash_scheme_reasons(traces, normalized_archived_hash_scheme, reasons)
    ):
        _append_hash_reasons(traces, computed_archived_hash, reasons)

    verdict = PreflightVerdict(
        status="ready" if not reasons else "blocked",
        reasons=tuple(reasons),
        target_hashes=tuple((trace.phase, trace.target_hash) for trace in traces),
        account_replay_started=False,
        archived_target_hash=computed_archived_hash,
        archived_declared_target_hash=normalized_archived_declared_hash,
    )
    return PhasePreflightResult(traces=tuple(traces), verdict=verdict)


def _move_source_date(
    source_day: date,
    calendar: tuple[date, ...],
    source_index: int,
    phase: int,
) -> DecisionMove:
    shifted_index = source_index + phase
    if shifted_index >= len(calendar):
        return DecisionMove(
            source_date=source_day,
            scheduled_date=None,
            reconstruction_date=None,
            omission_reason="tail_overflow",
        )
    shifted_day = calendar[shifted_index]
    return DecisionMove(
        source_date=source_day,
        scheduled_date=shifted_day,
        reconstruction_date=shifted_day,
    )


def canonical_target_artifact_sha256(artifact: Any) -> str:
    """Return the only hash representation accepted for target artifacts.

    The caller supplies an in-memory JSON-compatible artifact; this pure helper
    never opens a target file. A missing artifact, non-string mapping key, NaN,
    or other non-JSON value fails closed rather than being coerced into a
    comparison with unrelated bytes.
    """

    if artifact is None:
        raise ValueError("target_artifact_required")
    try:
        canonical_json = json.dumps(
            artifact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("target_artifact_not_canonical_json") from error
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _compute_archived_target_hash(artifact: Any | None, reasons: list[str]) -> str | None:
    if artifact is None:
        # Keep this distinct from a reconstruction-side omission: the archived
        # artifact is the independent comparison anchor, so a caller cannot
        # replace it with a self-declared hash.
        reasons.append("archived_target_artifact_required")
        return None
    try:
        return canonical_target_artifact_sha256(artifact)
    except ValueError as error:
        reasons.append(str(error))
        return None


def _compute_phase_target_hash(reconstruction: PhaseReconstruction) -> tuple[str | None, str | None]:
    try:
        return canonical_target_artifact_sha256(reconstruction.target_artifact), None
    except ValueError as error:
        return None, str(error)


def _append_identity_reasons(
    traces: Sequence[PhaseTrace], expected_identity: tuple[str, ...], reasons: list[str]
) -> None:
    for trace in traces:
        if trace.selection_identity != expected_identity:
            reasons.append(f"phase{trace.phase}_selection_identity_mismatch")


def _append_hash_reasons(
    traces: Sequence[PhaseTrace], archived_target_hash: str, reasons: list[str]
) -> None:
    phase0 = next(trace for trace in traces if trace.phase == 0)
    if phase0.target_hash is not None and phase0.target_hash == archived_target_hash:
        reasons.append("phase0_matches_archived_target_hash")

    first_phase_by_hash: dict[str, int] = {}
    for trace in traces:
        if trace.target_hash is None:
            continue
        prior_phase = first_phase_by_hash.setdefault(trace.target_hash, trace.phase)
        if prior_phase != trace.phase:
            reasons.append(
                f"duplicate_recomputed_target_hash:phase{trace.phase}:phase{prior_phase}"
            )


def _append_hash_scheme_reasons(
    traces: Sequence[PhaseTrace], archived_hash_scheme: str | None, reasons: list[str]
) -> bool:
    """Reject any comparison that is not canonical on both artifact sides.

    An omitted scheme intentionally blocks legacy callers rather than assuming
    that a raw file hash and an in-memory payload hash can be compared.
    """

    valid = True
    if archived_hash_scheme is None:
        reasons.append("archived_target_hash_scheme_required")
        valid = False
    elif archived_hash_scheme != CANONICAL_TARGET_ARTIFACT_HASH_SCHEME:
        reasons.append("archived_target_hash_scheme_not_canonical")
        valid = False

    for trace in traces:
        if trace.target_hash is None:
            continue
        scheme = trace.target_hash_scheme
        if scheme is None:
            reasons.append(f"phase{trace.phase}_target_hash_scheme_required")
            valid = False
            continue
        if archived_hash_scheme is not None and scheme != archived_hash_scheme:
            reasons.append(f"phase{trace.phase}_target_hash_scheme_mismatch")
            valid = False
        if scheme != CANONICAL_TARGET_ARTIFACT_HASH_SCHEME:
            reasons.append(f"phase{trace.phase}_target_hash_scheme_not_canonical")
            valid = False
    return valid


def _coerce_reconstruction(value: PhaseReconstruction | Mapping[str, Any]) -> PhaseReconstruction:
    if isinstance(value, PhaseReconstruction):
        target_hash = value.target_hash
        identity = value.selection_identity
        target_hash_scheme = value.target_hash_scheme
        target_artifact = value.target_artifact
    elif isinstance(value, Mapping):
        target_hash = value.get("target_hash")
        identity = value.get("selection_identity")
        target_hash_scheme = value.get("target_hash_scheme")
        target_artifact = value.get("target_artifact")
    else:
        raise TypeError("phase_reconstruction_must_be_mapping_or_dataclass")
    return PhaseReconstruction(
        target_hash=_normalise_optional_declared_hash(target_hash),
        selection_identity=_normalise_identity(identity),
        target_hash_scheme=_normalise_optional_hash_scheme(target_hash_scheme),
        target_artifact=target_artifact,
    )


def _normalise_calendar(values: Sequence[object]) -> tuple[date, ...]:
    dates = tuple(_to_date(value) for value in values)
    if not dates:
        raise ValueError("trading_calendar_required")
    if len(set(dates)) != len(dates) or tuple(sorted(dates)) != dates:
        raise ValueError("trading_calendar_must_be_sorted_unique")
    return dates


def _normalise_source_dates(values: Sequence[object], calendar: tuple[date, ...]) -> tuple[date, ...]:
    dates = tuple(_to_date(value) for value in values)
    if not dates:
        raise ValueError("source_decision_dates_required")
    if len(set(dates)) != len(dates) or tuple(sorted(dates)) != dates:
        raise ValueError("source_decision_dates_must_be_sorted_unique")
    calendar_days = set(calendar)
    missing = tuple(day.isoformat() for day in dates if day not in calendar_days)
    if missing:
        raise ValueError("source_decision_date_missing_from_calendar:" + ",".join(missing))
    return dates


def _normalise_phases(values: Sequence[int]) -> tuple[int, ...]:
    phases = tuple(values)
    if not phases or 0 not in phases:
        raise ValueError("phase_zero_required")
    if any(not isinstance(phase, int) or isinstance(phase, bool) or phase < 0 for phase in phases):
        raise ValueError("phases_must_be_nonnegative_integers")
    if len(set(phases)) != len(phases) or tuple(sorted(phases)) != phases:
        raise ValueError("phases_must_be_sorted_unique")
    return phases


def _normalise_identity(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("selection_identity_must_be_an_ordered_sequence")
    identity = tuple(str(item).strip() for item in value)
    if not identity or any(not item for item in identity):
        raise ValueError("selection_identity_required")
    return identity


def _normalise_optional_hash_scheme(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target_hash_scheme_must_be_a_nonempty_string")
    return value.strip()


def _normalise_optional_declared_hash(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("declared_target_hash_must_be_a_nonempty_string")
    return value.strip()


def _validate_reconstructors(
    reconstructors: Mapping[int, PhaseReconstructor], phases: tuple[int, ...]
) -> None:
    if not isinstance(reconstructors, Mapping):
        raise ValueError("phase_reconstructors_mapping_required")
    keys = set(reconstructors)
    expected = set(phases)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"phase_reconstructors_mismatch:missing={missing}:extra={extra}")
    if any(not callable(reconstructors[phase]) for phase in phases):
        raise ValueError("phase_reconstructor_not_callable")


def _to_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return date.fromisoformat(str(value)[:10])


__all__ = [
    "CANONICAL_TARGET_ARTIFACT_HASH_SCHEME",
    "CORRECTED_SELECTION_IDENTITY",
    "DEFAULT_PHASES",
    "DecisionMove",
    "PhasePreflightResult",
    "PhaseReconstruction",
    "PhaseTrace",
    "PreflightVerdict",
    "canonical_target_artifact_sha256",
    "preflight_phase_schedule",
]
