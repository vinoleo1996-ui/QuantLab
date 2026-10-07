"""Forward-only T+1-open to T+6-open information evaluation.

This is deliberately not an account adapter.  It turns a score vector that
was already frozen in :class:`ResearchSignalLedger` into one mature,
cross-sectional H=5 information label.  It has no position sizing, cost,
annualisation, order, or portfolio imports.

The reporting schedule is fixed before labels mature: every fifth exchange
session from an explicit anchor is a non-overlapping block.  Daily outcomes may
still be retained for provenance, but a later report cannot choose the best
overlapping windows after observing their returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.data.contracts import timestamp
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import DECISION_TIMEZONE, is_exact_decision_cutoff


H5_OUTCOME_KEY = "forward_h5_open_to_open_information_v1"
# V2 adds a frozen signal identity to prevent reports from pooling H=5
# outcomes from different experts or code/prompt versions.
H5_LABEL_SCHEMA = "quantlab.forward-h5-information-label.v2"
H5_VALUE_SCHEMA = "quantlab.forward-h5-information-outcome.v2"
H5_SIGNAL_IDENTITY_SCHEMA = "quantlab.forward-h5-signal-identity.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPEN_BAR_COLUMNS = frozenset(
    {
        "session",
        "instrument_id",
        "open",
        "quality_status",
        "published_at",
        "first_captured_at",
        "available_at",
        "source_receipt_id",
        "content_sha256",
        "capture_mode",
    }
)


class H5EvaluationError(ValueError):
    """The forward H=5 label cannot be formed under its frozen contract."""


def _session(value: object) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise H5EvaluationError("h5_invalid_session") from exc
    if pd.isna(result):
        raise H5EvaluationError("h5_invalid_session")
    if result.tzinfo is not None:
        result = result.tz_convert(DECISION_TIMEZONE).tz_localize(None)
    return result.normalize()


def _local_time(value: object) -> pd.Timestamp:
    try:
        result = timestamp(value)
    except (TypeError, ValueError) as exc:
        raise H5EvaluationError("h5_invalid_timestamp") from exc
    if pd.isna(result):
        raise H5EvaluationError("h5_invalid_timestamp")
    return result.tz_convert(DECISION_TIMEZONE)


def _session_close(session: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(session.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(hours=18)


def _session_open(session: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(session.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(hours=9, minutes=30)


def _canonical_sha256(value: Mapping[str, object]) -> str:
    try:
        raw = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise H5EvaluationError("h5_canonical_identity_required") from exc
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class H5SignalIdentity:
    """The exact expert/version whose H=5 blocks may be summarized together.

    H=5 is shared plumbing for independently tested mechanisms.  The generic
    outcome key is therefore insufficient as a report key: mixing a theme
    endpoint with a later L2 or text endpoint could make a twenty-block gate
    look satisfied even though no single signal accumulated that evidence.
    """

    feature_id: str
    signal_version: str
    primary_endpoint: str
    extractor_sha256: str

    def __post_init__(self) -> None:
        for field in ("feature_id", "signal_version", "primary_endpoint"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise H5EvaluationError("h5_signal_identity_" + field + "_required")
            object.__setattr__(self, field, value.strip())
        if not isinstance(self.extractor_sha256, str) or not _SHA256.fullmatch(self.extractor_sha256.lower()):
            raise H5EvaluationError("h5_signal_identity_extractor_sha256_required")
        object.__setattr__(self, "extractor_sha256", self.extractor_sha256.lower())

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": H5_SIGNAL_IDENTITY_SCHEMA,
                "feature_id": self.feature_id,
                "signal_version": self.signal_version,
                "primary_endpoint": self.primary_endpoint,
                "extractor_sha256": self.extractor_sha256,
            }
        )

    def to_record(self) -> dict[str, str]:
        return {
            "schema": H5_SIGNAL_IDENTITY_SCHEMA,
            "feature_id": self.feature_id,
            "signal_version": self.signal_version,
            "primary_endpoint": self.primary_endpoint,
            "extractor_sha256": self.extractor_sha256,
            "signal_identity_sha256": self.sha256,
        }


@dataclass(frozen=True)
class H5BlockSchedule:
    """Fixed session calendar and every-five-session reporting anchor.

    The `anchor_session` is selected when a qualified forward source starts.
    Its hash is copied into every label.  Changing the anchor or calendar is a
    new protocol identity rather than a way to choose more favourable blocks.
    """

    anchor_session: object
    trading_sessions: Sequence[object]
    version: str = "forward_h5_block_schedule_v1"

    def __post_init__(self) -> None:
        sessions = tuple(_session(value).date() for value in self.trading_sessions)
        anchor = _session(self.anchor_session).date()
        if not isinstance(self.version, str) or self.version != "forward_h5_block_schedule_v1":
            raise H5EvaluationError("h5_block_schedule_version_invalid")
        if len(sessions) < 7:
            raise H5EvaluationError("h5_block_schedule_requires_seven_sessions")
        if len(set(sessions)) != len(sessions) or tuple(sorted(sessions)) != sessions:
            raise H5EvaluationError("h5_block_schedule_sessions_not_strictly_increasing")
        if anchor not in sessions:
            raise H5EvaluationError("h5_block_schedule_anchor_missing")
        object.__setattr__(self, "anchor_session", anchor)
        object.__setattr__(self, "trading_sessions", sessions)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": self.version,
                "anchor_session": self.anchor_session.isoformat(),
                "trading_sessions": [session.isoformat() for session in self.trading_sessions],
            }
        )

    def _index(self, decision_session: object) -> int:
        session = _session(decision_session).date()
        try:
            return self.trading_sessions.index(session)
        except ValueError as exc:
            raise H5EvaluationError("h5_decision_session_not_in_schedule") from exc

    def is_reporting_block(self, decision_session: object) -> bool:
        index = self._index(decision_session)
        anchor = self.trading_sessions.index(self.anchor_session)
        return index >= anchor and (index - anchor) % 5 == 0

    def sessions_for_label(self, decision_session: object) -> tuple[str, str]:
        """Return the fixed T+1 entry and T+6 exit sessions as ISO dates."""
        index = self._index(decision_session)
        if index + 6 >= len(self.trading_sessions):
            raise H5EvaluationError("h5_exit_session_not_yet_in_schedule")
        return (
            self.trading_sessions[index + 1].isoformat(),
            self.trading_sessions[index + 6].isoformat(),
        )


@dataclass(frozen=True)
class H5Settlement:
    """A mature label, an explicit unavailable label, or a not-yet-mature result."""

    status: str
    observation_id: str
    reason: str | None
    outcome: Mapping[str, object] | None


def _observation_score_vector(
    *, ledger: ResearchSignalLedger, observation_id: str
) -> tuple[dict[str, object], dict[str, float], pd.Timestamp, tuple[str, ...], H5SignalIdentity]:
    if not isinstance(ledger, ResearchSignalLedger):
        raise H5EvaluationError("h5_ledger_required")
    try:
        evidence = ledger.export_observation_evidence(observation_id)
        event = evidence["observation"]
        body = event["body"]
    except (KeyError, TypeError, ValueError) as exc:
        raise H5EvaluationError("h5_observation_evidence_unavailable") from exc
    if not isinstance(event, Mapping) or not isinstance(body, Mapping):
        raise H5EvaluationError("h5_observation_evidence_unavailable")
    try:
        decision = _local_time(body["decision_at"])
        if not is_exact_decision_cutoff(decision, decision):
            raise ValueError("not_decision_cutoff")
        targets = tuple(str(value) for value in body["target_entity_keys"])
        value = body["value"]
        scores = value["score_by_instrument"]
        signal_version = value["signal_version"]
        primary_endpoint = value["primary_endpoint"]
        feature_id = body["feature_id"]
        extractor_sha256 = body["extractor_sha256"]
    except (KeyError, TypeError, ValueError, H5EvaluationError) as exc:
        raise H5EvaluationError("h5_score_observation_schema_invalid") from exc
    if (
        not targets
        or len(set(targets)) != len(targets)
        or not isinstance(scores, Mapping)
        or set(str(key) for key in scores) != set(targets)
        or not isinstance(signal_version, str)
        or not signal_version
        or not isinstance(primary_endpoint, str)
        or not primary_endpoint
    ):
        raise H5EvaluationError("h5_score_observation_schema_invalid")
    try:
        normalized = {str(code): float(scores[code]) for code in scores}
    except (TypeError, ValueError) as exc:
        raise H5EvaluationError("h5_score_observation_nonfinite") from exc
    if not all(math.isfinite(value) for value in normalized.values()):
        raise H5EvaluationError("h5_score_observation_nonfinite")
    identity = H5SignalIdentity(
        feature_id=feature_id,
        signal_version=signal_version,
        primary_endpoint=primary_endpoint,
        extractor_sha256=extractor_sha256,
    )
    return dict(event), normalized, decision, tuple(sorted(targets)), identity


def h5_signal_identity_for_observation(
    *, ledger: ResearchSignalLedger, observation_id: str
) -> H5SignalIdentity:
    """Read the frozen identity a later H=5 report must explicitly select."""
    return _observation_score_vector(ledger=ledger, observation_id=observation_id)[-1]


def _scope_coverage_snapshot(
    *, ledger: ResearchSignalLedger, observation: Mapping[str, object], members: tuple[str, ...]
) -> dict[str, object]:
    """Require one score vector to cover its whole declared decision scope.

    A generic ledger observation may legitimately describe a subset (for
    example, one text citation).  That is not sufficient for the graph H=5
    endpoint: accepting a hand-picked subset would turn a coverage failure
    into a plausible RankIC.  Keep the coverage snapshot in the immutable
    label so a later report can disclose the exact scope it evaluated.
    """
    try:
        body = observation["body"]
        if not isinstance(body, Mapping):
            raise ValueError("observation_body")
        coverage = ledger.coverage(
            str(body["scope_id"]), body["decision_at"], str(body["feature_id"])
        )
        expected = int(coverage["expected"])
        observed = int(coverage["observed"])
        abstained = int(coverage["abstained"])
        missing = tuple(str(value) for value in coverage["missing"])
        observed_keys = tuple(str(value) for value in coverage["observed_entity_keys"])
        membership_sha256 = str(coverage["membership_sha256"])
        complete = (
            expected == len(members)
            and observed == len(members)
            and abstained == 0
            and not missing
            and set(observed_keys) == set(members)
            and len(observed_keys) == len(members)
            and bool(_SHA256.fullmatch(membership_sha256.lower()))
        )
        return {
            "scope_id": str(body["scope_id"]),
            "membership_sha256": membership_sha256.lower(),
            "expected": expected,
            "observed": observed,
            "abstained": abstained,
            "missing": list(missing),
            "complete": complete,
        }
    except (KeyError, TypeError, ValueError):
        return {
            "scope_id": None,
            "membership_sha256": None,
            "expected": None,
            "observed": None,
            "abstained": None,
            "missing": None,
            "complete": False,
        }


def _label_definition(
    *,
    decision: pd.Timestamp,
    entry_session: str,
    exit_session: str,
    schedule: H5BlockSchedule,
    reporting_block: bool,
    signal_identity: H5SignalIdentity,
    scope_coverage: Mapping[str, object],
) -> dict[str, object]:
    return {
        "version": H5_LABEL_SCHEMA,
        "outcome_key": H5_OUTCOME_KEY,
        "decision_session": decision.date().isoformat(),
        "entry_session": entry_session,
        "exit_session": exit_session,
        "entry_rule": "T+1_open",
        "exit_rule": "T+6_open",
        "horizon_sessions": 5,
        "availability_rule": "max(exit_open_source_available_at, T+6_1800_Asia_Shanghai)",
        "block_schedule_sha256": schedule.sha256,
        "reporting_block": reporting_block,
        "signal_identity": signal_identity.to_record(),
        "scope_coverage": dict(scope_coverage),
        "account_execution_authorized": False,
    }


def _invalid_open_frame_reason(
    *,
    open_bars: object,
    members: tuple[str, ...],
    entry_session: str,
    exit_session: str,
) -> tuple[pd.DataFrame | None, str | None]:
    if not isinstance(open_bars, pd.DataFrame) or not _OPEN_BAR_COLUMNS.issubset(open_bars.columns):
        return None, "h5_open_bar_schema_or_coverage_invalid"
    selected = open_bars.copy()
    try:
        selected["instrument_id"] = selected.instrument_id.astype(str)
        selected["_session"] = selected.session.map(_session)
        entry = _session(entry_session)
        exit_ = _session(exit_session)
    except (AttributeError, TypeError, ValueError, H5EvaluationError):
        return None, "h5_open_bar_schema_or_coverage_invalid"
    selected = selected[
        selected.instrument_id.isin(members) & selected._session.isin((entry, exit_))
    ].copy()
    if (
        selected.duplicated(["_session", "instrument_id"]).any()
        or len(selected) != len(members) * 2
        or set(selected[selected._session.eq(entry)].instrument_id) != set(members)
        or set(selected[selected._session.eq(exit_)].instrument_id) != set(members)
    ):
        return None, "h5_open_bar_schema_or_coverage_invalid"
    try:
        numeric = pd.to_numeric(selected.open, errors="coerce")
    except (AttributeError, TypeError, ValueError):
        return None, "h5_open_bar_schema_or_coverage_invalid"
    if (
        not selected.quality_status.astype(str).eq("ok").all()
        or not selected.capture_mode.astype(str).eq("forward_receipt").all()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or (numeric <= 0).any()
    ):
        return None, "h5_open_bar_schema_or_coverage_invalid"
    selected["open"] = numeric.astype(float)
    return selected, None


def _verified_open_receipts(
    *, ledger: ResearchSignalLedger, rows: pd.DataFrame
) -> tuple[tuple[str, ...] | None, pd.Timestamp | None, str | None]:
    """Verify each price row against the receipt written on that same session."""
    receipt_ids: set[str] = set()
    latest: pd.Timestamp | None = None
    try:
        for _, row in rows.iterrows():
            session = _session(row["_session"])
            deadline = _session_close(session)
            evidence = ledger.export_receipt_evidence(str(row["source_receipt_id"]))
            receipt = evidence["receipt"]
            body = receipt["body"]
            raw = evidence["raw"]
            if (
                evidence["canonical_receipt_id"] != str(row["source_receipt_id"])
                or receipt["status"] != "accepted"
                or raw["sha256"] != str(row["content_sha256"]).lower()
                or body["raw_sha256"] != raw["sha256"]
                or _local_time(body["published_at"]) != _local_time(row["published_at"])
                or _local_time(body["first_captured_at"]) != _local_time(row["first_captured_at"])
                or _local_time(receipt["available_at"]) != _local_time(row["available_at"])
            ):
                return None, None, "h5_open_bar_provenance_invalid"
            captured = _local_time(body["first_captured_at"])
            available = _local_time(receipt["available_at"])
            recorded = _local_time(receipt["recorded_at"])
            # A label may wait for T+6 close, but neither of its price inputs
            # may be created then from a historical copy.  Each raw receipt
            # must have reached the ledger on its own market session.
            if (
                captured.date() != session.date()
                or available.date() != session.date()
                or recorded.date() != session.date()
                or captured > deadline
                or available > deadline
                or recorded > deadline
            ):
                return None, None, "h5_open_bar_provenance_invalid"
            latest = max((value for value in (latest, available, recorded) if value is not None))
            receipt_ids.add(str(receipt["event_id"]))
    except (AttributeError, KeyError, TypeError, ValueError, H5EvaluationError):
        return None, None, "h5_open_bar_provenance_invalid"
    return tuple(sorted(receipt_ids)), latest, None


def _record_unavailable(
    *,
    ledger: ResearchSignalLedger,
    observation_id: str,
    label_definition: Mapping[str, object],
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
    available_at: pd.Timestamp,
    reason: str,
) -> H5Settlement:
    try:
        event = ledger.record_outcome(
            observation_id=observation_id,
            outcome_key=H5_OUTCOME_KEY,
            label_definition=label_definition,
            entry_at=entry_at,
            exit_at=exit_at,
            available_at=available_at,
            status="unavailable",
            reason=reason,
            now=ledger.runtime_now(),
        )
    except (TypeError, ValueError) as exc:
        raise H5EvaluationError("h5_unavailable_outcome_recording_failed") from exc
    return H5Settlement("unavailable", observation_id, reason, event)


def settle_h5_open_to_open(
    *,
    ledger: ResearchSignalLedger,
    observation_id: str,
    schedule: H5BlockSchedule,
    open_bars: pd.DataFrame,
) -> H5Settlement:
    """Settle one frozen score vector using T+1/T+6 open prices.

    The call is idempotent once an immutable outcome exists.  Before T+6
    18:00 it returns ``pending`` without writing a label.  After that point,
    a missing/late/invalid source produces an explicit unavailable outcome;
    it never turns into zero return or an inferred score.
    """
    if not isinstance(schedule, H5BlockSchedule):
        raise H5EvaluationError("h5_block_schedule_required")
    event, scores, decision, members, signal_identity = _observation_score_vector(
        ledger=ledger, observation_id=observation_id
    )
    decision_session = decision.date().isoformat()
    try:
        entry_session, exit_session = schedule.sessions_for_label(decision_session)
    except H5EvaluationError as exc:
        if str(exc) == "h5_exit_session_not_yet_in_schedule":
            return H5Settlement("pending", observation_id, "h5_exit_session_not_yet_in_schedule", None)
        raise
    entry_at, exit_at = _session_open(_session(entry_session)), _session_open(_session(exit_session))
    exit_close = _session_close(_session(exit_session))
    scope_coverage = _scope_coverage_snapshot(
        ledger=ledger, observation=event, members=members
    )
    label = _label_definition(
        decision=decision,
        entry_session=entry_session,
        exit_session=exit_session,
        schedule=schedule,
        reporting_block=schedule.is_reporting_block(decision_session),
        signal_identity=signal_identity,
        scope_coverage=scope_coverage,
    )
    now = ledger.runtime_now()
    if now < exit_close:
        return H5Settlement("pending", observation_id, "h5_exit_close_not_reached", None)

    if scope_coverage["complete"] is not True:
        return _record_unavailable(
            ledger=ledger,
            observation_id=observation_id,
            label_definition=label,
            entry_at=entry_at,
            exit_at=exit_at,
            available_at=exit_close,
            reason="h5_score_observation_scope_coverage_incomplete",
        )

    rows, frame_reason = _invalid_open_frame_reason(
        open_bars=open_bars,
        members=members,
        entry_session=entry_session,
        exit_session=exit_session,
    )
    if frame_reason is not None:
        return _record_unavailable(
            ledger=ledger,
            observation_id=observation_id,
            label_definition=label,
            entry_at=entry_at,
            exit_at=exit_at,
            available_at=exit_close,
            reason=frame_reason,
        )
    receipt_ids, evidence_available_at, provenance_reason = _verified_open_receipts(ledger=ledger, rows=rows)
    if provenance_reason is not None:
        return _record_unavailable(
            ledger=ledger,
            observation_id=observation_id,
            label_definition=label,
            entry_at=entry_at,
            exit_at=exit_at,
            available_at=exit_close,
            reason=provenance_reason,
        )
    assert receipt_ids is not None and evidence_available_at is not None
    available_at = max(exit_close, evidence_available_at)
    if now < available_at:
        return H5Settlement("pending", observation_id, "h5_source_receipt_not_yet_available", None)

    entry = rows[rows["_session"].eq(_session(entry_session))].set_index("instrument_id").loc[list(members)]
    exit_ = rows[rows["_session"].eq(_session(exit_session))].set_index("instrument_id").loc[list(members)]
    returns = exit_.open.astype(float) / entry.open.astype(float) - 1.0
    scores_by_member = pd.Series({code: scores[code] for code in members}, dtype=float).loc[list(members)]
    rank_ic = scores_by_member.corr(returns, method="spearman")
    if rank_ic is None or not math.isfinite(float(rank_ic)):
        return _record_unavailable(
            ledger=ledger,
            observation_id=observation_id,
            label_definition=label,
            entry_at=entry_at,
            exit_at=exit_at,
            available_at=available_at,
            reason="h5_cross_sectional_rank_ic_undefined",
        )
    value = {
        "schema": H5_VALUE_SCHEMA,
        "feature_id": signal_identity.feature_id,
        "signal_version": signal_identity.signal_version,
        "signal_primary_endpoint": signal_identity.primary_endpoint,
        "extractor_sha256": signal_identity.extractor_sha256,
        "signal_identity_sha256": signal_identity.sha256,
        "primary_endpoint": "cross_sectional_spearman_rank_ic",
        "cross_sectional_spearman_rank_ic": float(rank_ic),
        "mean_open_to_open_return": float(returns.mean()),
        "median_open_to_open_return": float(returns.median()),
        "return_by_instrument": {code: float(returns.loc[code]) for code in sorted(members)},
        "score_coverage": len(members),
        "label_coverage": len(returns),
        "scope_coverage": dict(scope_coverage),
        "reporting_block": bool(label["reporting_block"]),
        "block_schedule_sha256": schedule.sha256,
        "account_execution_authorized": False,
    }
    try:
        outcome = ledger.record_outcome(
            observation_id=observation_id,
            outcome_key=H5_OUTCOME_KEY,
            label_definition=label,
            entry_at=entry_at,
            exit_at=exit_at,
            available_at=available_at,
            value=value,
            receipt_ids=receipt_ids,
            now=ledger.runtime_now(),
        )
    except (TypeError, ValueError) as exc:
        raise H5EvaluationError("h5_settled_outcome_recording_failed") from exc
    return H5Settlement("settled", observation_id, None, outcome)


def summarize_h5_information_blocks(
    *, ledger: ResearchSignalLedger,
    as_of: object,
    schedule: H5BlockSchedule,
    signal_identity: H5SignalIdentity,
) -> dict[str, object]:
    """Report all mature H=5 labels for one frozen schedule, never an account PnL.

    Twenty *settled non-overlapping reporting blocks* is only the sample-size
    precondition for a later research-registration discussion.  It does not
    grant D2 clearance, user authorisation, portfolio execution, or adoption.
    """
    if not isinstance(ledger, ResearchSignalLedger):
        raise H5EvaluationError("h5_ledger_required")
    if not isinstance(schedule, H5BlockSchedule):
        raise H5EvaluationError("h5_block_schedule_required")
    if not isinstance(signal_identity, H5SignalIdentity):
        raise H5EvaluationError("h5_signal_identity_required")
    cutoff = _local_time(as_of)
    records = ledger.outcomes(cutoff, outcome_key=H5_OUTCOME_KEY)
    matched: list[dict[str, object]] = []
    for event in records:
        body = event.get("body", {})
        label = body.get("label_definition", {}) if isinstance(body, Mapping) else {}
        label_identity = label.get("signal_identity") if isinstance(label, Mapping) else None
        if (
            not isinstance(label, Mapping)
            or label.get("version") != H5_LABEL_SCHEMA
            or label.get("block_schedule_sha256") != schedule.sha256
            or label_identity != signal_identity.to_record()
        ):
            continue
        if event["status"] == "settled":
            value = body.get("value")
            if (
                not isinstance(value, Mapping)
                or value.get("signal_identity_sha256") != signal_identity.sha256
                or value.get("feature_id") != signal_identity.feature_id
                or value.get("signal_version") != signal_identity.signal_version
                or value.get("signal_primary_endpoint") != signal_identity.primary_endpoint
                or value.get("extractor_sha256") != signal_identity.extractor_sha256
            ):
                continue
        matched.append(event)
    reporting = [
        event
        for event in matched
        if bool(event["body"]["label_definition"].get("reporting_block"))
    ]
    reporting.sort(key=lambda event: event["body"]["entry_at"])
    prior_exit: pd.Timestamp | None = None
    for event in reporting:
        entry = _local_time(event["body"]["entry_at"])
        exit_ = _local_time(event["body"]["exit_at"])
        if prior_exit is not None and entry < prior_exit:
            raise H5EvaluationError("h5_reporting_blocks_overlap")
        prior_exit = exit_
    settled = [event for event in reporting if event["status"] == "settled"]
    unavailable = [event for event in reporting if event["status"] == "unavailable"]
    rank_ics = [
        float(event["body"]["value"]["cross_sectional_spearman_rank_ic"])
        for event in settled
    ]
    daily = [
        {
            "outcome_event_id": event["event_id"],
            "status": event["status"],
            "decision_session": event["body"]["label_definition"]["decision_session"],
            "reporting_block": bool(event["body"]["label_definition"]["reporting_block"]),
            "reason": event["body"].get("reason"),
        }
        for event in matched
    ]
    return {
        "schema": "quantlab.forward-h5-information-report.v1",
        "as_of": cutoff.isoformat(),
        "outcome_key": H5_OUTCOME_KEY,
        "block_schedule_sha256": schedule.sha256,
        "signal_identity": signal_identity.to_record(),
        "daily_matured_outcomes": len(matched),
        "daily_settled_outcomes": sum(event["status"] == "settled" for event in matched),
        "daily_unavailable_outcomes": sum(event["status"] == "unavailable" for event in matched),
        "all_daily_outcomes": daily,
        "reporting_blocks_settled": len(settled),
        "reporting_blocks_unavailable": len(unavailable),
        "minimum_required_reporting_blocks": 20,
        "minimum_twenty_blocks_met": len(settled) >= 20,
        "cross_sectional_spearman_rank_ic_mean": None if not rank_ics else float(np.mean(rank_ics)),
        "cross_sectional_spearman_rank_ic_median": None if not rank_ics else float(np.median(rank_ics)),
        "d2_v2_archived_required": True,
        "separate_user_authorization_required": True,
        "account_execution_authorized": False,
        "promotion_approved": False,
    }
