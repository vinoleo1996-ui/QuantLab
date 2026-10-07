"""Launch-day U0 seed from QuantDB unadjusted daily partitions.

The forward contract allows the most recent 253 exchange sessions to build the
current pool on the day collection starts.  Those partitions are not historical
theme signals.  QuantDB's ``amount`` is in 万元; pool ADV is in CNY.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import (
    FORWARD_CONTRACT_VERSION,
    U0,
    UniverseSnapshot,
    build_liquidity_pool,
    canonical_forward_input_sha256,
)


# Declared before any seed result is read.  The protocol was registered after
# the 2026-09-29 close, so the first due session is the first open session on
# or after this date.  Later sessions stay on the frozen 20-session step.
QUANTDB_FORWARD_ANCHOR_NOT_BEFORE = date(2026, 9, 30)
SEED_SESSION_COUNT = 253
ADV_WINDOW = 60
_WAN_TO_CNY = 10_000.0


class QuantDBU0SeedError(ValueError):
    """The seed partitions do not satisfy the frozen 253-session contract."""


@dataclass(frozen=True)
class QuantDBPartition:
    """One unadjusted session already stored as a ledger receipt."""

    session: pd.Timestamp
    frame: pd.DataFrame
    receipt_id: str
    receipt_hash: str


def resolve_forward_anchor(open_sessions: Sequence[object]) -> pd.Timestamp | None:
    """Return the first open session on or after the declared anchor date."""
    sessions = []
    for value in open_sessions:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise QuantDBU0SeedError("quantdb_anchor_calendar_invalid")
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
        sessions.append(stamp.normalize())
    ordered = tuple(sorted(set(sessions)))
    if len(ordered) != len(sessions):
        raise QuantDBU0SeedError("quantdb_anchor_calendar_invalid")
    for session in ordered:
        if session.date() >= QUANTDB_FORWARD_ANCHOR_NOT_BEFORE:
            return session
    return None


def _session_frame(partition: QuantDBPartition) -> pd.DataFrame:
    frame = partition.frame
    required = {"instrument_id", "amount", "close"}
    if not required.issubset(frame.columns):
        raise QuantDBU0SeedError("quantdb_seed_columns_missing")
    selected = frame.loc[:, ["instrument_id", "amount", "close"]].copy()
    selected["instrument_id"] = selected.instrument_id.astype(str)
    if selected.instrument_id.duplicated().any() or selected.instrument_id.eq("").any():
        raise QuantDBU0SeedError("quantdb_seed_instrument_invalid")
    selected["amount"] = pd.to_numeric(selected.amount, errors="coerce")
    selected["close"] = pd.to_numeric(selected.close, errors="coerce")
    return selected.set_index("instrument_id")


def assemble_quantdb_u0_rows(
    *,
    partitions: Sequence[QuantDBPartition],
    identity: pd.DataFrame,
    provenance: Mapping[str, object],
) -> pd.DataFrame:
    """Turn 253 partitions into receipt-backed pool input rows.

    ``provenance`` is the manifest receipt that lists every partition hash.
    Every row points at that one receipt.  The partition blobs stay in the
    ledger separately; this function does not open them.
    """
    if len(partitions) != SEED_SESSION_COUNT:
        raise QuantDBU0SeedError("quantdb_seed_requires_253_sessions")
    sessions = [pd.Timestamp(item.session).normalize() for item in partitions]
    if sessions != sorted(sessions) or len(set(sessions)) != len(sessions):
        raise QuantDBU0SeedError("quantdb_seed_sessions_not_increasing")
    frames = [_session_frame(item) for item in partitions]
    codes = frames[-1].index.astype(str)
    if len(codes) == 0:
        raise QuantDBU0SeedError("quantdb_seed_decision_session_empty")
    amount = np.vstack([frame.reindex(codes)["amount"].to_numpy(dtype=float) for frame in frames[-ADV_WINDOW:]])
    close = np.vstack([frame.reindex(codes)["close"].to_numpy(dtype=float) for frame in frames])
    valid_amount = np.isfinite(amount) & (amount >= 0)
    valid_close = np.isfinite(close) & (close > 0)
    current_close = close[-1]
    current_amount = amount[-1]
    names = identity["name"].astype(str) if "name" in identity.columns else pd.Series("", index=identity.index)
    eligible = []
    asof = []
    for code, close_value, amount_value in zip(codes, current_close, current_amount):
        if code not in identity.index:
            eligible.append(False)
            asof.append(bool(math.isfinite(close_value) and close_value > 0 and math.isfinite(amount_value) and amount_value >= 0))
            continue
        row = identity.loc[code]
        name = str(names.loc[code]) if code in names.index else ""
        is_st = bool(row.is_st) if not pd.isna(row.is_st) else True
        is_delisted = bool(row.is_delisted) if "is_delisted" in identity.columns and not pd.isna(row.is_delisted) else False
        blocked = is_st or is_delisted or "ST" in name.upper() or "退" in name
        eligible.append(not blocked)
        asof.append(bool(math.isfinite(close_value) and close_value > 0 and math.isfinite(amount_value) and amount_value >= 0))
    adv_count = valid_amount.sum(axis=0)
    with np.errstate(all="ignore"):
        adv = np.where(adv_count > 0, np.nanmean(np.where(valid_amount, amount, np.nan), axis=0) * _WAN_TO_CNY, np.nan)
    required = ("source_receipt_id", "source_receipt_hash", "source_published_at", "source_first_captured_at", "source_available_at")
    if any(key not in provenance or provenance[key] in (None, "") for key in required):
        raise QuantDBU0SeedError("quantdb_seed_manifest_provenance_required")
    return pd.DataFrame(
        {
            "instrument_id": codes.to_numpy(),
            "base_eligible": eligible,
            "adv60_cny": adv,
            "adv_observation_count": adv_count.astype(int),
            "valid_price_session_count": valid_close.sum(axis=0).astype(int),
            "asof_ok": asof,
            "source_receipt_id": str(provenance["source_receipt_id"]),
            "source_receipt_hash": str(provenance["source_receipt_hash"]),
            "source_published_at": provenance["source_published_at"],
            "source_first_captured_at": provenance["source_first_captured_at"],
            "source_available_at": provenance["source_available_at"],
            "capture_mode": "forward_receipt",
        }
    )


def build_quantdb_u0_snapshot(
    *,
    ledger: ResearchSignalLedger,
    rows: pd.DataFrame,
    session: object,
    decision_at: object,
    cadence_anchor: object,
    exchange_sessions: Sequence[object],
) -> UniverseSnapshot:
    """Build the frozen U0 snapshot.  The caller supplies the anchor; this does not choose one."""
    if not isinstance(ledger, ResearchSignalLedger):
        raise QuantDBU0SeedError("quantdb_u0_ledger_required")
    input_hash = canonical_forward_input_sha256(rows)
    return build_liquidity_pool(
        session=session,
        decision_at=decision_at,
        rows=rows,
        spec=U0,
        cadence_anchor=cadence_anchor,
        exchange_sessions=exchange_sessions,
        input_hash=input_hash,
        ledger=ledger,
        contract_version=FORWARD_CONTRACT_VERSION,
    )


__all__ = [
    "ADV_WINDOW",
    "QUANTDB_FORWARD_ANCHOR_NOT_BEFORE",
    "SEED_SESSION_COUNT",
    "QuantDBPartition",
    "QuantDBU0SeedError",
    "assemble_quantdb_u0_rows",
    "build_quantdb_u0_snapshot",
    "resolve_forward_anchor",
]
