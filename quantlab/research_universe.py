"""Pure, versioned liquidity-pool contracts for forward research signals.

This module deliberately does not reuse :func:`quantlab.data.market.universe_at`.
That function implements the main strategy's 20-session liquidity screen, while
the reference liquidity band has a different 60-observation/rank contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import date
import hashlib
import json
import re
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.research_signal_ledger import ResearchSignalLedger


FORWARD_CONTRACT_VERSION = "forward_u0_v1"
LEGACY_REFERENCE_CONTRACT_VERSION = "reference_u0_legacy_v1"
DECISION_TIMEZONE = "Asia/Shanghai"
DECISION_CUTOFF_HOUR = 18
_POOL_SPEC_FIELDS = (
    "pool_id",
    "rank_lower_exclusive",
    "rank_upper_inclusive",
    "adv_window_sessions",
    "min_adv_cny",
    "required_prior_price_sessions",
    "cadence_sessions",
    "rank_tie_break",
    "rank_before_adv_floor",
)
# These are the only forward liquidity universes authorized by the research
# contract.  Keep the complete tuple here rather than validating a pool name
# alone: otherwise an ADV20 or changed-rank rule could be relabelled as U0.
_FROZEN_POOL_SPEC_VALUES: Mapping[str, tuple[object, ...]] = {
    "U0": ("U0", 3000, 5000, 60, 5_000_000.0, 252, 20, "instrument_id_asc", True),
    "U1": ("U1", 5000, None, 60, 5_000_000.0, 252, 20, "instrument_id_asc", True),
    "U_broad": ("U_broad", 3000, None, 60, 5_000_000.0, 252, 20, "instrument_id_asc", True),
}
_REQUIRED_COLUMNS = frozenset(
    {
        "instrument_id",
        "base_eligible",
        "adv60_cny",
        "adv_observation_count",
        "valid_price_session_count",
        "asof_ok",
        "source_receipt_id",
        "source_receipt_hash",
        "source_published_at",
        "source_first_captured_at",
        "source_available_at",
        "capture_mode",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_COLUMN_ORDER = tuple(sorted(_REQUIRED_COLUMNS))
_NUMERIC_REQUIRED_COLUMNS = frozenset(
    {"adv60_cny", "adv_observation_count", "valid_price_session_count"}
)
_BOOLEAN_REQUIRED_COLUMNS = frozenset({"asof_ok", "base_eligible"})
_RECEIPT_TIMESTAMP_COLUMNS = frozenset(
    {"source_published_at", "source_first_captured_at", "source_available_at"}
)
_FORWARD_INPUT_HASH_SCHEMA = "quantlab.forward_u0.required_rows.v1"
_FORWARD_MEMBERSHIP_HASH_SCHEMA = "quantlab.forward_u0.membership.v1"


@dataclass(frozen=True)
class ExternalProtocolReference:
    """An auditable pointer, never an input data source or merged result."""

    protocol_path: str
    sha256: str
    external_reference_only: bool = True


R2_POOL_MEMBERSHIP_REFERENCE = ExternalProtocolReference(
    protocol_path=(
        "/Users/agiuser/.codex/worktrees/pool-pit-validation/QuantLab/output/"
        "pool_membership_effect_20260929_r2/protocol.json"
    ),
    sha256="b3653299fbe4f00873d038e0d23448ca378a1566aff1bf97a687e649f84c09f4",
)


@dataclass(frozen=True)
class LiquidityPoolSpec:
    """The pool boundary only; lifecycle eligibility belongs to input evidence."""

    pool_id: Literal["U0", "U1", "U_broad"]
    rank_lower_exclusive: int = 3000
    rank_upper_inclusive: int | None = None
    adv_window_sessions: int = 60
    min_adv_cny: float = 5_000_000.0
    required_prior_price_sessions: int = 252
    cadence_sessions: int = 20
    rank_tie_break: str = "instrument_id_asc"
    rank_before_adv_floor: bool = True

    def __post_init__(self) -> None:
        expected = _FROZEN_POOL_SPEC_VALUES.get(self.pool_id)
        actual = tuple(getattr(self, field) for field in _POOL_SPEC_FIELDS)
        # Type equality is deliberate.  These are versioned contract values,
        # not caller-tunable numeric settings with equivalent coercions.
        if expected is None or any(
            type(value) is not type(frozen) or value != frozen
            for value, frozen in zip(actual, expected)
        ):
            raise ValueError("liquidity_pool_spec_not_frozen")
        if self.rank_lower_exclusive < 0:
            raise ValueError("negative_rank_lower_bound")
        if self.rank_upper_inclusive is not None and self.rank_upper_inclusive <= self.rank_lower_exclusive:
            raise ValueError("invalid_rank_interval")
        if self.adv_window_sessions <= 0 or self.required_prior_price_sessions < 0 or self.cadence_sessions <= 0:
            raise ValueError("invalid_pool_window")
        if not np.isfinite(self.min_adv_cny) or self.min_adv_cny <= 0:
            raise ValueError("invalid_min_adv")


U0 = LiquidityPoolSpec("U0", rank_lower_exclusive=3000, rank_upper_inclusive=5000)
U1 = LiquidityPoolSpec("U1", rank_lower_exclusive=5000, rank_upper_inclusive=None)
U_BROAD = LiquidityPoolSpec("U_broad", rank_lower_exclusive=3000, rank_upper_inclusive=None)


def _is_frozen_pool_spec(spec: object) -> bool:
    """Whether an object exactly matches one of the three frozen contracts."""
    if not isinstance(spec, LiquidityPoolSpec):
        return False
    expected = _FROZEN_POOL_SPEC_VALUES.get(spec.pool_id)
    if expected is None:
        return False
    try:
        actual = tuple(getattr(spec, field) for field in _POOL_SPEC_FIELDS)
    except AttributeError:
        return False
    return all(
        type(value) is type(frozen) and value == frozen
        for value, frozen in zip(actual, expected)
    )


def _require_frozen_pool_spec(spec: object, *, type_error: str) -> LiquidityPoolSpec:
    if not isinstance(spec, LiquidityPoolSpec):
        raise TypeError(type_error)
    if not _is_frozen_pool_spec(spec):
        raise ValueError("liquidity_pool_spec_not_frozen")
    return spec


@dataclass(frozen=True)
class UniverseSnapshot:
    contract_version: str
    pool_id: str
    session: date
    decision_at: pd.Timestamp
    cadence_anchor: date
    members: tuple[str, ...]
    rank_by_instrument: Mapping[str, int]
    adv60_by_instrument: Mapping[str, float]
    eligibility_reason_by_instrument: Mapping[str, str]
    input_hash: str
    membership_hash: str
    status: Literal["ok", "abstain"]
    abstain_reason: str | None


def _session(value: object) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("invalid_session")
    if value.tzinfo is not None:
        value = value.tz_convert(DECISION_TIMEZONE).tz_localize(None)
    return value.normalize()


def _local_time(value: object) -> pd.Timestamp | None:
    value = pd.Timestamp(value)
    if pd.isna(value):
        return None
    if value.tzinfo is None:
        return value.tz_localize(DECISION_TIMEZONE)
    return value.tz_convert(DECISION_TIMEZONE)


def _safe_local_time(value: object) -> pd.Timestamp | None:
    """Return a local timestamp or ``None`` for untrusted row evidence."""
    try:
        return _local_time(value)
    except (OverflowError, TypeError, ValueError):
        return None


def is_exact_decision_cutoff(session: object, decision_at: object) -> bool:
    """Whether a receipt is anchored to that session's fixed 18:00 cutoff.

    All forward research inputs use this one, explicit close-of-decision
    contract.  A different same-day clock time is not an alternative cutoff:
    accepting it would make receipt eligibility depend on a caller-controlled
    wall clock.
    """
    try:
        session_stamp = _session(session)
        cutoff = _local_time(decision_at)
    except (TypeError, ValueError):
        return False
    if cutoff is None:
        return False
    expected = pd.Timestamp(session_stamp.date(), tz=DECISION_TIMEZONE) + pd.Timedelta(
        hours=DECISION_CUTOFF_HOUR
    )
    return cutoff == expected


def _valid_count(value: object) -> int | None:
    try:
        count = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(count) or count < 0 or not count.is_integer():
        return None
    return int(count)


def _is_true(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value.lower()))


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_number(value: object) -> Mapping[str, str]:
    """Represent equivalent finite numeric inputs independently of pandas dtype."""
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return {"type": "text", "value": str(value)}
    if not decimal.is_finite():
        return {"type": "nonfinite_number", "value": str(decimal)}
    if decimal == 0:
        decimal = Decimal(0)
    return {"type": "number", "value": format(decimal.normalize(), "f")}


def _canonical_required_value(column: str, value: object) -> object:
    if _is_missing_scalar(value):
        return None
    if column in _NUMERIC_REQUIRED_COLUMNS:
        return _canonical_number(value)
    if column in _BOOLEAN_REQUIRED_COLUMNS:
        if isinstance(value, (bool, np.bool_)):
            return {"type": "bool", "value": bool(value)}
        return {"type": "text", "value": str(value)}
    if column in _RECEIPT_TIMESTAMP_COLUMNS:
        receipt_time = _safe_local_time(value)
        if receipt_time is not None:
            return {"type": "shanghai_timestamp", "value": receipt_time.isoformat()}
        return {"type": "text", "value": str(value)}
    if column == "source_receipt_hash" and _is_sha256(value):
        return {"type": "sha256", "value": str(value).lower()}
    return {"type": "text", "value": str(value)}


def canonical_forward_input_sha256(rows: pd.DataFrame) -> str:
    """Hash all required forward rows in a row-order-independent canonical form.

    Extra dataframe columns are deliberately excluded.  They are not part of
    the frozen U0 input contract and therefore cannot silently change a pool
    identity.  Every required column, including invalid evidence values, is
    included so an abstained snapshot is still bound to the exact input.
    """
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("forward_input_rows_must_be_dataframe")
    missing = _REQUIRED_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError("universe_input_columns_missing:" + ",".join(sorted(missing)))
    canonical_rows = [
        {column: _canonical_required_value(column, row[column]) for column in _REQUIRED_COLUMN_ORDER}
        for _, row in rows.loc[:, _REQUIRED_COLUMN_ORDER].iterrows()
    ]
    canonical_rows.sort(
        key=lambda row: json.dumps(
            row, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    )
    return _canonical_json_sha256(
        {
            "schema": _FORWARD_INPUT_HASH_SCHEMA,
            "columns": _REQUIRED_COLUMN_ORDER,
            "rows": canonical_rows,
        }
    )


def forward_membership_sha256(
    *,
    spec: LiquidityPoolSpec,
    session: object,
    decision_at: object,
    cadence_anchor: object,
    input_hash: str,
    members: Sequence[str],
    status: Literal["ok", "abstain"],
    abstain_reason: str | None,
) -> str:
    """Return the immutable identity of a forward pool snapshot.

    The membership order is the deterministic rank order.  The full pool
    contract and the already-verified canonical input hash are included, so a
    matching list of tickers from another cutoff or pool rule is not relabelled
    as this snapshot.
    """
    spec = _require_frozen_pool_spec(
        spec, type_error="membership_spec_must_be_liquidity_pool_spec"
    )
    if not _is_sha256(input_hash):
        raise ValueError("membership_input_hash_must_be_sha256")
    if status not in ("ok", "abstain"):
        raise ValueError("invalid_membership_status")
    normalized_members = tuple(str(code) for code in members)
    if len(set(normalized_members)) != len(normalized_members):
        raise ValueError("duplicate_membership_instrument")
    if status == "ok" and (not normalized_members or abstain_reason is not None):
        raise ValueError("invalid_ok_membership_identity")
    if status == "abstain" and normalized_members:
        raise ValueError("abstained_membership_must_be_empty")
    when = _session(session)
    anchor = _session(cadence_anchor)
    cutoff = _local_time(decision_at)
    if cutoff is None:
        raise ValueError("invalid_membership_decision_at")
    return _canonical_json_sha256(
        {
            "schema": _FORWARD_MEMBERSHIP_HASH_SCHEMA,
            "contract_version": FORWARD_CONTRACT_VERSION,
            "pool_spec": {
                "pool_id": spec.pool_id,
                "rank_lower_exclusive": spec.rank_lower_exclusive,
                "rank_upper_inclusive": spec.rank_upper_inclusive,
                "adv_window_sessions": spec.adv_window_sessions,
                "min_adv_cny": _canonical_number(spec.min_adv_cny),
                "required_prior_price_sessions": spec.required_prior_price_sessions,
                "cadence_sessions": spec.cadence_sessions,
                "rank_tie_break": spec.rank_tie_break,
                "rank_before_adv_floor": spec.rank_before_adv_floor,
            },
            "session": when.date().isoformat(),
            "decision_at": cutoff.isoformat(),
            "cadence_anchor": anchor.date().isoformat(),
            "input_hash": input_hash.lower(),
            "members": normalized_members,
            "status": status,
            "abstain_reason": abstain_reason,
        }
    )


def due_sessions(
    exchange_sessions: Sequence[object], anchor_session: object, *, step: int = 20
) -> tuple[pd.Timestamp, ...]:
    """Return the exact exchange-session cadence, including the frozen anchor."""
    if step <= 0:
        raise ValueError("invalid_cadence_step")
    calendar = pd.DatetimeIndex([_session(value) for value in exchange_sessions])
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("invalid_exchange_session_calendar")
    anchor = _session(anchor_session)
    if anchor not in calendar:
        raise ValueError("cadence_anchor_not_in_calendar")
    start = int(calendar.get_loc(anchor))
    return tuple(calendar[start::step])


def _abstain(
    *, spec: LiquidityPoolSpec, session: pd.Timestamp, decision_at: pd.Timestamp,
    anchor: pd.Timestamp, input_hash: str, reason: str,
    eligibility: Mapping[str, str] | None = None,
) -> UniverseSnapshot:
    membership_hash = forward_membership_sha256(
        spec=spec,
        session=session,
        decision_at=decision_at,
        cadence_anchor=anchor,
        input_hash=input_hash,
        members=(),
        status="abstain",
        abstain_reason=reason,
    )
    return UniverseSnapshot(
        contract_version=FORWARD_CONTRACT_VERSION,
        pool_id=spec.pool_id,
        session=session.date(),
        decision_at=decision_at,
        cadence_anchor=anchor.date(),
        members=(),
        rank_by_instrument={},
        adv60_by_instrument={},
        eligibility_reason_by_instrument=dict(eligibility or {}),
        input_hash=input_hash.lower(),
        membership_hash=membership_hash,
        status="abstain",
        abstain_reason=reason,
    )


def _same_local_time(left: object, right: object) -> bool:
    """Compare receipt timestamps after explicit Shanghai-time normalization."""
    normalized_left = _safe_local_time(left)
    normalized_right = _safe_local_time(right)
    return normalized_left is not None and normalized_left == normalized_right


def _at_or_before_cutoff(value: object, cutoff: pd.Timestamp) -> bool:
    normalized = _safe_local_time(value)
    return normalized is not None and normalized <= cutoff


def _matches_canonical_ledger_receipt(
    *,
    ledger: ResearchSignalLedger,
    receipt_id: object,
    receipt_hash: object,
    published_at: object,
    first_captured_at: object,
    available_at: object,
    cutoff: pd.Timestamp,
    evidence_cache: dict[str, object],
) -> bool:
    """Verify one row against an accepted, canonical, content-addressed receipt.

    A SHA-shaped column supplied by a dataframe is not evidence by itself.  It
    must resolve to the ledger's canonical (not duplicate-alias) receipt and
    its verified raw blob and timestamps.  Retrieval failures are deliberately
    indistinguishable from mismatches to callers: forward pool construction
    must abstain rather than make an availability judgment from partial proof.
    """
    if not isinstance(receipt_id, str) or not receipt_id:
        return False
    if receipt_id not in evidence_cache:
        try:
            evidence_cache[receipt_id] = ledger.export_receipt_evidence(receipt_id)
        except Exception:
            evidence_cache[receipt_id] = None
    evidence = evidence_cache[receipt_id]
    if not isinstance(evidence, Mapping):
        return False
    try:
        canonical_receipt_id = evidence["canonical_receipt_id"]
        receipt = evidence["receipt"]
        raw = evidence["raw"]
        if evidence.get("kind") != "research_signal_receipt_evidence":
            return False
        if not isinstance(canonical_receipt_id, str) or receipt_id != canonical_receipt_id:
            return False
        if not isinstance(receipt, Mapping) or not isinstance(raw, Mapping):
            return False
        if (
            receipt.get("event_id") != canonical_receipt_id
            or receipt.get("kind") != "receipt"
            or receipt.get("status") != "accepted"
            or receipt.get("canonical_receipt_id", canonical_receipt_id) != canonical_receipt_id
        ):
            return False
        body = receipt.get("body")
        if not isinstance(body, Mapping):
            return False
        raw_sha256 = raw.get("sha256")
        if not _is_sha256(raw_sha256) or receipt_hash != raw_sha256:
            return False
        if not (
            _same_local_time(published_at, body.get("published_at"))
            and _same_local_time(first_captured_at, body.get("first_captured_at"))
            and _same_local_time(available_at, receipt.get("available_at"))
        ):
            return False
        # The row's three asserted timestamps and every material timestamp in
        # the accepted receipt must already have existed by the fixed 18:00
        # decision cutoff.  In particular, a late ledger write cannot be
        # relabelled as an earlier available market cross-section.
        return all(
            _at_or_before_cutoff(value, cutoff)
            for value in (
                published_at,
                first_captured_at,
                available_at,
                body.get("published_at"),
                body.get("first_captured_at"),
                receipt.get("recorded_at"),
                receipt.get("available_at"),
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def build_liquidity_pool(
    *,
    session: object,
    decision_at: object,
    rows: pd.DataFrame,
    spec: LiquidityPoolSpec,
    cadence_anchor: object,
    exchange_sessions: Sequence[object],
    input_hash: str,
    ledger: ResearchSignalLedger,
    contract_version: str = FORWARD_CONTRACT_VERSION,
) -> UniverseSnapshot:
    """Build a forward-only U0/U1/broad snapshot without account side effects.

    Ranking happens on the complete rankable cross-section before the 5m ADV
    floor.  This mirrors the reference band's boundary semantics while keeping
    its legacy historical inputs explicitly separate.
    """
    if contract_version != FORWARD_CONTRACT_VERSION:
        raise ValueError("forward_builder_requires_forward_contract_version")
    spec = _require_frozen_pool_spec(spec, type_error="forward_pool_spec_required")
    if not isinstance(ledger, ResearchSignalLedger):
        raise TypeError("forward_pool_ledger_required")
    missing = _REQUIRED_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError("universe_input_columns_missing:" + ",".join(sorted(missing)))
    if not _is_sha256(input_hash):
        raise ValueError("input_hash_must_be_sha256")
    canonical_input_hash = canonical_forward_input_sha256(rows)
    if input_hash.lower() != canonical_input_hash:
        raise ValueError("input_hash_does_not_match_canonical_required_rows")
    if rows.instrument_id.isna().any() or rows.instrument_id.astype(str).duplicated().any():
        raise ValueError("invalid_or_duplicate_instrument_id")

    snapshot_session = _session(session)
    anchor = _session(cadence_anchor)
    cutoff = _local_time(decision_at)
    if cutoff is None:
        raise ValueError("invalid_decision_at")
    if not is_exact_decision_cutoff(snapshot_session, cutoff):
        return _abstain(
            spec=spec, session=snapshot_session, decision_at=cutoff, anchor=anchor,
            input_hash=canonical_input_hash, reason="decision_at_must_equal_session_1800_shanghai",
        )
    if snapshot_session not in due_sessions(exchange_sessions, anchor, step=spec.cadence_sessions):
        return _abstain(
            spec=spec, session=snapshot_session, decision_at=cutoff, anchor=anchor,
            input_hash=canonical_input_hash, reason="not_due_session",
        )

    frame = rows.copy()
    frame["instrument_id"] = frame.instrument_id.astype(str)
    published = frame.source_published_at.map(_safe_local_time)
    first_captured = frame.source_first_captured_at.map(_safe_local_time)
    availability = frame.source_available_at.map(_safe_local_time)
    asof_valid = frame.asof_ok.eq(True).fillna(False).astype(bool)
    invalid_receipt = (
        frame.source_receipt_hash.isna()
        | ~frame.source_receipt_hash.map(_is_sha256)
        | published.isna()
        | first_captured.isna()
        | availability.isna()
        | published.gt(cutoff)
        | first_captured.gt(cutoff)
        | availability.gt(cutoff)
        | frame.capture_mode.astype(str).ne("forward_receipt")
        | ~asof_valid
    )
    if invalid_receipt.any():
        reasons = {
            code: "late_or_invalid_forward_receipt"
            for code in frame.loc[invalid_receipt, "instrument_id"]
        }
        return _abstain(
            spec=spec, session=snapshot_session, decision_at=cutoff, anchor=anchor,
            input_hash=canonical_input_hash, reason="incomplete_or_nonforward_rank_input", eligibility=reasons,
        )

    receipt_evidence_cache: dict[str, object] = {}
    evidence_mismatch_codes = [
        str(row.instrument_id)
        for row in frame.itertuples(index=False)
        if not _matches_canonical_ledger_receipt(
            ledger=ledger,
            receipt_id=row.source_receipt_id,
            receipt_hash=row.source_receipt_hash,
            published_at=row.source_published_at,
            first_captured_at=row.source_first_captured_at,
            available_at=row.source_available_at,
            cutoff=cutoff,
            evidence_cache=receipt_evidence_cache,
        )
    ]
    if evidence_mismatch_codes:
        reasons = {code: "ledger_receipt_evidence_mismatch" for code in evidence_mismatch_codes}
        return _abstain(
            spec=spec,
            session=snapshot_session,
            decision_at=cutoff,
            anchor=anchor,
            input_hash=canonical_input_hash,
            reason="incomplete_or_nonforward_rank_input",
            eligibility=reasons,
        )

    reasons: dict[str, str] = {}
    rankable = []
    for row in frame.itertuples(index=False):
        code = str(row.instrument_id)
        adv = float(row.adv60_cny) if pd.notna(row.adv60_cny) else float("nan")
        adv_observations = _valid_count(row.adv_observation_count)
        price_sessions = _valid_count(row.valid_price_session_count)
        if not _is_true(row.base_eligible):
            reasons[code] = "base_ineligible"
        elif adv_observations is None:
            reasons[code] = "invalid_adv_observation_count"
        elif adv_observations < spec.adv_window_sessions:
            reasons[code] = "insufficient_adv_observations"
        elif price_sessions is None:
            reasons[code] = "invalid_price_history_count"
        elif price_sessions < spec.required_prior_price_sessions + 1:
            reasons[code] = "insufficient_price_history"
        elif not np.isfinite(adv) or adv <= 0:
            reasons[code] = "invalid_adv60"
        else:
            rankable.append((code, adv))

    if len(rankable) < spec.rank_lower_exclusive + 1:
        return _abstain(
            spec=spec, session=snapshot_session, decision_at=cutoff, anchor=anchor,
            input_hash=canonical_input_hash, reason="insufficient_rankable_names", eligibility=reasons,
        )
    ordered = sorted(rankable, key=lambda item: (-item[1], item[0]))
    ranks = {code: index for index, (code, _) in enumerate(ordered, start=1)}
    advs = {code: adv for code, adv in ordered}
    members = []
    for code, adv in ordered:
        rank = ranks[code]
        inside_lower = rank > spec.rank_lower_exclusive
        inside_upper = spec.rank_upper_inclusive is None or rank <= spec.rank_upper_inclusive
        if not (inside_lower and inside_upper):
            reasons.setdefault(code, "outside_rank_interval")
        elif adv < spec.min_adv_cny:
            reasons[code] = "below_adv_floor_after_rank"
        else:
            members.append(code)
    if not members:
        return _abstain(
            spec=spec, session=snapshot_session, decision_at=cutoff, anchor=anchor,
            input_hash=canonical_input_hash, reason="no_members_after_adv_floor", eligibility=reasons,
        )
    return UniverseSnapshot(
        contract_version=FORWARD_CONTRACT_VERSION,
        pool_id=spec.pool_id,
        session=snapshot_session.date(),
        decision_at=cutoff,
        cadence_anchor=anchor.date(),
        members=tuple(members),
        rank_by_instrument=ranks,
        adv60_by_instrument=advs,
        eligibility_reason_by_instrument=reasons,
        input_hash=canonical_input_hash,
        membership_hash=forward_membership_sha256(
            spec=spec,
            session=snapshot_session,
            decision_at=cutoff,
            cadence_anchor=anchor,
            input_hash=canonical_input_hash,
            members=tuple(members),
            status="ok",
            abstain_reason=None,
        ),
        status="ok",
        abstain_reason=None,
    )


def legacy_reference_snapshot(
    *, session: object, members: Sequence[str], membership_hash: str, cadence_anchor: object
) -> UniverseSnapshot:
    """Represent a hash-pinned historical reference membership without rebuilding it.

    The legacy reference used positive-amount observations and cannot be silently
    relabelled as the forward contract's validated-price-session universe.
    """
    if not _is_sha256(membership_hash):
        raise ValueError("legacy_membership_hash_must_be_sha256")
    normalized = tuple(sorted(str(code) for code in members))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("invalid_legacy_membership")
    when = _session(session)
    anchor = _session(cadence_anchor)
    return UniverseSnapshot(
        contract_version=LEGACY_REFERENCE_CONTRACT_VERSION,
        pool_id="U0",
        session=when.date(),
        decision_at=_local_time(str(when.date()) + " 18:00:00"),
        cadence_anchor=anchor.date(),
        members=normalized,
        rank_by_instrument={},
        adv60_by_instrument={},
        eligibility_reason_by_instrument={},
        input_hash=membership_hash.lower(),
        membership_hash=membership_hash.lower(),
        status="ok",
        abstain_reason=None,
    )
