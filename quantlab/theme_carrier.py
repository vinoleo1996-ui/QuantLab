"""Forward-only, four-component theme-carrier research signal.

The module is deliberately pure: it reads no files, makes no requests, checks
no wall clock and does not know about accounts.  It receives only receipt-backed
end-of-day data and returns either a fully reproducible score or an abstention.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.research_universe import (
    DECISION_TIMEZONE,
    FORWARD_CONTRACT_VERSION,
    U0,
    UniverseSnapshot,
    forward_membership_sha256,
    is_exact_decision_cutoff,
)
from quantlab.research_signal_ledger import ResearchSignalLedger


THEME_COMPONENTS = (
    "prior_limit_up_premium",
    "broken_board_rate",
    "theme_relative_strength",
    "advancing_breadth",
)
_BAR_COLUMNS = frozenset(
    {
        "session",
        "instrument_id",
        "open",
        "high",
        "low",
        "close",
        "reference_price",
        "limit_up",
        "price_tick",
        "quality_status",
        "published_at",
        "first_captured_at",
        "available_at",
        "source_receipt_id",
        "content_sha256",
        "capture_mode",
    }
)
_MAP_COLUMNS = frozenset(
    {
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
    }
)
_OUT_OF_SCOPE_FIELD_TOKENS = (
    "auction",
    "fund",
    "flow",
    "emotion",
    "leader",
    "freshness",
    "text",
    "stance",
    "sentiment",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THEME_CARRIER_V1_VERSION = "theme_carrier_minimal_v1"
_THEME_CARRIER_V1_MIN_THEME_MEMBERS = 5
_THEME_CARRIER_V1_MIN_YESTERDAY_LIMIT_UP = 1
_THEME_CARRIER_V1_MIN_TODAY_TOUCHED_LIMIT = 1


@dataclass(frozen=True)
class ThemeCarrierConfig:
    """Fixed primary endpoint; changing components is a new experiment."""

    version: str = _THEME_CARRIER_V1_VERSION
    min_theme_members: int = _THEME_CARRIER_V1_MIN_THEME_MEMBERS
    min_yesterday_limit_up: int = _THEME_CARRIER_V1_MIN_YESTERDAY_LIMIT_UP
    min_today_touched_limit: int = _THEME_CARRIER_V1_MIN_TODAY_TOUCHED_LIMIT
    required_components: tuple[str, ...] = THEME_COMPONENTS

    def __post_init__(self) -> None:
        if (
            self.version != _THEME_CARRIER_V1_VERSION
            or self.min_theme_members != _THEME_CARRIER_V1_MIN_THEME_MEMBERS
            or self.min_yesterday_limit_up != _THEME_CARRIER_V1_MIN_YESTERDAY_LIMIT_UP
            or self.min_today_touched_limit != _THEME_CARRIER_V1_MIN_TODAY_TOUCHED_LIMIT
            or self.required_components != THEME_COMPONENTS
        ):
            raise ValueError("theme_carrier_v1_config_is_frozen")


@dataclass(frozen=True)
class ThemeCarrierResult:
    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    decision_session: date
    decision_at: pd.Timestamp
    theme_components: pd.DataFrame
    stock_scores: pd.Series
    coverage: Mapping[str, object]
    receipt_hashes: tuple[str, ...]


@dataclass(frozen=True)
class EntryOrdering:
    """A view of new-entry ordering; it never changes holdings or the pool."""

    baseline_candidate_order: tuple[str, ...]
    retained_holdings: tuple[str, ...]
    new_entry_order: tuple[str, ...]
    applied: bool
    abstain_reason: str | None


def _session(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("invalid_theme_session")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(DECISION_TIMEZONE).tz_localize(None)
    return stamp.normalize()


def _local_time(value: object) -> pd.Timestamp | None:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize(DECISION_TIMEZONE)
    return stamp.tz_convert(DECISION_TIMEZONE)


def _empty_result(
    *, decision_session: pd.Timestamp, decision_at: pd.Timestamp, reason: str,
    coverage: Mapping[str, object] | None = None, receipt_hashes: Iterable[str] = (),
) -> ThemeCarrierResult:
    return ThemeCarrierResult(
        status="abstain",
        abstain_reason=reason,
        decision_session=decision_session.date(),
        decision_at=decision_at,
        theme_components=pd.DataFrame(columns=("theme_id",) + THEME_COMPONENTS),
        stock_scores=pd.Series(dtype=float, name="theme_score_equal_weight_v1"),
        coverage=dict(coverage or {}),
        receipt_hashes=tuple(sorted(set(str(value) for value in receipt_hashes))),
    )


def _required(frame: pd.DataFrame, required: frozenset[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(label + "_columns_missing:" + ",".join(sorted(missing)))


def _has_out_of_scope_input(frame: pd.DataFrame) -> bool:
    return any(
        token in str(column).lower()
        for column in frame.columns
        for token in _OUT_OF_SCOPE_FIELD_TOKENS
    )


def _provenance_is_valid(frame: pd.DataFrame, decision_at: pd.Timestamp) -> bool:
    for column in ("published_at", "first_captured_at", "available_at"):
        values = frame[column].map(_local_time)
        if values.isna().any() or values.gt(decision_at).any():
            return False
    return bool(
        frame.source_receipt_id.notna().all()
        and frame.source_receipt_id.astype(str).ne("").all()
        and frame.content_sha256.notna().all()
        and frame.content_sha256.map(lambda value: isinstance(value, str) and bool(_SHA256.fullmatch(value.lower()))).all()
        and frame.capture_mode.astype(str).eq("forward_receipt").all()
    )


def _ledger_provenance_identity(
    frames: Iterable[pd.DataFrame], *, ledger: ResearchSignalLedger, decision_at: pd.Timestamp
) -> tuple[str, str] | None:
    """Bind row-level receipt metadata to the append-only ledger's raw blobs.

    A syntactically valid receipt id/hash in a DataFrame is not evidence by
    itself.  Each selected row must name the canonical accepted receipt whose
    stored raw digest and time fields exactly match the row and whose accepted
    ledger event was recorded by the fixed decision cutoff.  This intentionally
    reads only the sidecar ledger; it has no market or account dependency.
    """
    if not isinstance(ledger, ResearchSignalLedger):
        return None
    verification = ledger.verify()
    journal_hash = verification.get("journal_hash")
    if not verification.get("verified") or not isinstance(journal_hash, str) or not _SHA256.fullmatch(journal_hash):
        return None
    try:
        for frame in frames:
            for row in frame.itertuples(index=False):
                receipt_id = str(row.source_receipt_id)
                evidence = ledger.export_receipt_evidence(receipt_id)
                receipt = evidence["receipt"]
                body = receipt["body"]
                raw = evidence["raw"]
                if (
                    evidence.get("canonical_receipt_id") != receipt_id
                    or receipt.get("event_id") != receipt_id
                    or receipt.get("status") != "accepted"
                    or raw.get("sha256") != str(row.content_sha256).lower()
                    or body.get("raw_sha256") != raw.get("sha256")
                    or _local_time(body.get("published_at")) != _local_time(row.published_at)
                    or _local_time(body.get("first_captured_at")) != _local_time(row.first_captured_at)
                    or _local_time(receipt.get("available_at")) != _local_time(row.available_at)
                    or _local_time(receipt.get("recorded_at")) is None
                    or _local_time(receipt.get("recorded_at")) > decision_at
                ):
                    return None
    except (KeyError, TypeError, ValueError):
        return None
    return ledger.ledger_id, journal_hash


def _valid_bar_values(frame: pd.DataFrame) -> bool:
    # ``quality_status`` is part of the frozen raw-bar schema.  Treat every
    # status other than an explicit successful validation as unusable rather
    # than allowing a stale, partial, or provider-rejected bar to flow into a
    # plausible-looking theme score.
    if not frame.quality_status.astype(str).eq("ok").all():
        return False
    numeric = ("open", "high", "low", "close", "reference_price", "limit_up", "price_tick")
    converted = frame.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(converted.to_numpy(dtype=float)).all() or (converted <= 0).any().any():
        return False
    return bool((converted.high >= converted.close).all())


def _percentiles(values: Mapping[str, float], *, higher_is_better: bool) -> dict[str, float]:
    """Return directional percentile ranks with average ranks for equal values."""
    series = pd.Series(values, dtype=float)
    ranks = series.rank(method="average", ascending=higher_is_better)
    return {theme: float(ranks.loc[theme] / len(series)) for theme in series.index}


def score_theme_carrier_v1(
    *,
    decision_session: object,
    prior_session: object,
    decision_at: object,
    u0: UniverseSnapshot,
    bars: pd.DataFrame,
    primary_theme_map: pd.DataFrame,
    ledger: ResearchSignalLedger,
    config: ThemeCarrierConfig = ThemeCarrierConfig(),
) -> ThemeCarrierResult:
    """Return the fixed equal-weight four-component theme score or abstain.

    A decision is all-or-nothing: a missing member, ambiguous theme, late
    receipt, or unusable component produces no score.  This avoids silently
    pushing unscored names down a new-entry ranking.
    """
    _required(bars, _BAR_COLUMNS, "theme_carrier_bar")
    _required(primary_theme_map, _MAP_COLUMNS, "theme_carrier_map")
    today, yesterday = _session(decision_session), _session(prior_session)
    cutoff = _local_time(decision_at)
    if cutoff is None:
        raise ValueError("invalid_theme_decision_at")
    if yesterday >= today:
        raise ValueError("prior_session_must_precede_decision_session")
    if not is_exact_decision_cutoff(today, cutoff):
        return _empty_result(
            decision_session=today,
            decision_at=cutoff,
            reason="decision_at_must_equal_session_1800_shanghai",
        )
    if _has_out_of_scope_input(bars) or _has_out_of_scope_input(primary_theme_map):
        return _empty_result(decision_session=today, decision_at=cutoff, reason="out_of_scope_theme_input")
    if u0.pool_id != "U0" or u0.contract_version != FORWARD_CONTRACT_VERSION or u0.status != "ok":
        return _empty_result(decision_session=today, decision_at=cutoff, reason="forward_u0_required")
    if (
        not is_exact_decision_cutoff(u0.session, u0.decision_at)
        or _session(u0.session) != today
        or _local_time(u0.decision_at) != cutoff
    ):
        return _empty_result(decision_session=today, decision_at=cutoff, reason="forward_u0_cutoff_mismatch")
    try:
        expected_u0_membership_hash = forward_membership_sha256(
            spec=U0,
            session=u0.session,
            decision_at=u0.decision_at,
            cadence_anchor=u0.cadence_anchor,
            input_hash=u0.input_hash,
            members=u0.members,
            status=u0.status,
            abstain_reason=u0.abstain_reason,
        )
    except (TypeError, ValueError):
        return _empty_result(decision_session=today, decision_at=cutoff, reason="forward_u0_identity_mismatch")
    if not isinstance(u0.membership_hash, str) or u0.membership_hash.lower() != expected_u0_membership_hash:
        return _empty_result(decision_session=today, decision_at=cutoff, reason="forward_u0_identity_mismatch")
    members = tuple(str(code) for code in u0.members)
    if not members or len(set(members)) != len(members):
        return _empty_result(decision_session=today, decision_at=cutoff, reason="invalid_u0_membership")

    source_bars = bars.copy()
    source_bars["instrument_id"] = source_bars.instrument_id.astype(str)
    source_bars["_session"] = source_bars.session.map(_session)
    selected_bars = source_bars[
        source_bars.instrument_id.isin(members) & source_bars._session.isin((yesterday, today))
    ].copy()
    if selected_bars.duplicated(["_session", "instrument_id"]).any() or len(selected_bars) != len(members) * 2:
        return _empty_result(decision_session=today, decision_at=cutoff, reason="incomplete_or_duplicate_bar_evidence")
    if not _provenance_is_valid(selected_bars, cutoff) or not _valid_bar_values(selected_bars):
        return _empty_result(decision_session=today, decision_at=cutoff, reason="invalid_or_late_bar_provenance")

    source_map = primary_theme_map.copy()
    source_map["instrument_id"] = source_map.instrument_id.astype(str)
    source_map["_session"] = source_map.session.map(_session)
    selected_map = source_map[
        source_map.instrument_id.isin(members) & source_map._session.eq(today)
    ].copy()
    if (
        selected_map.duplicated(["_session", "instrument_id"]).any()
        or len(selected_map) != len(members)
        or not selected_map.mapping_status.astype(str).eq("active").all()
        or selected_map.theme_id.isna().any()
        or selected_map.theme_id.astype(str).eq("").any()
    ):
        return _empty_result(decision_session=today, decision_at=cutoff, reason="incomplete_or_ambiguous_primary_theme")
    if not _provenance_is_valid(selected_map, cutoff):
        return _empty_result(decision_session=today, decision_at=cutoff, reason="invalid_or_late_theme_provenance")
    ledger_identity = _ledger_provenance_identity(
        (selected_bars, selected_map), ledger=ledger, decision_at=cutoff
    )
    if ledger_identity is None:
        return _empty_result(
            decision_session=today, decision_at=cutoff, reason="unlinked_or_tampered_ledger_receipt"
        )

    prior = selected_bars[selected_bars._session.eq(yesterday)].set_index("instrument_id").loc[list(members)]
    current = selected_bars[selected_bars._session.eq(today)].set_index("instrument_id").loc[list(members)]
    close_prior = pd.to_numeric(prior.close, errors="coerce")
    close_current = pd.to_numeric(current.close, errors="coerce")
    returns = close_current / close_prior - 1.0
    prior_limit_up = close_prior >= pd.to_numeric(prior.limit_up, errors="coerce") - pd.to_numeric(prior.price_tick, errors="coerce") / 2
    touched = pd.to_numeric(current.high, errors="coerce") >= pd.to_numeric(current.limit_up, errors="coerce") - pd.to_numeric(current.price_tick, errors="coerce") / 2
    sealed = close_current >= pd.to_numeric(current.limit_up, errors="coerce") - pd.to_numeric(current.price_tick, errors="coerce") / 2
    broken = touched & ~sealed
    if int(prior_limit_up.sum()) == 0:
        return _empty_result(decision_session=today, decision_at=cutoff, reason="no_market_yesterday_limit_up")
    market_limit_return = float(returns.loc[prior_limit_up].mean())

    themes = selected_map.set_index("instrument_id").loc[list(members), "theme_id"].astype(str)
    raw: dict[str, dict[str, float]] = {}
    for theme in sorted(themes.unique()):
        codes = tuple(code for code in members if themes.loc[code] == theme)
        if len(codes) < config.min_theme_members:
            return _empty_result(decision_session=today, decision_at=cutoff, reason="insufficient_theme_members")
        limit_codes = tuple(code for code in codes if bool(prior_limit_up.loc[code]))
        touched_codes = tuple(code for code in codes if bool(touched.loc[code]))
        if len(limit_codes) < config.min_yesterday_limit_up:
            return _empty_result(decision_session=today, decision_at=cutoff, reason="insufficient_theme_yesterday_limit_up")
        if len(touched_codes) < config.min_today_touched_limit:
            return _empty_result(decision_session=today, decision_at=cutoff, reason="insufficient_theme_touched_limit")
        theme_return = float(returns.loc[list(codes)].mean())
        raw[theme] = {
            "prior_limit_up_premium": float(returns.loc[list(limit_codes)].mean() - market_limit_return),
            "broken_board_rate": float(broken.loc[list(touched_codes)].mean()),
            "theme_relative_strength": float(theme_return - returns.mean()),
            "advancing_breadth": float((close_current.loc[list(codes)] > close_prior.loc[list(codes)]).mean()),
            "member_count": float(len(codes)),
            "yesterday_limit_up_count": float(len(limit_codes)),
            "today_touched_limit_count": float(len(touched_codes)),
        }
    if not all(np.isfinite(value) for theme in raw.values() for key, value in theme.items() if key in THEME_COMPONENTS):
        return _empty_result(decision_session=today, decision_at=cutoff, reason="nonfinite_theme_component")

    percentile = {
        "prior_limit_up_premium": _percentiles({theme: values["prior_limit_up_premium"] for theme, values in raw.items()}, higher_is_better=True),
        "broken_board_rate": _percentiles({theme: values["broken_board_rate"] for theme, values in raw.items()}, higher_is_better=False),
        "theme_relative_strength": _percentiles({theme: values["theme_relative_strength"] for theme, values in raw.items()}, higher_is_better=True),
        "advancing_breadth": _percentiles({theme: values["advancing_breadth"] for theme, values in raw.items()}, higher_is_better=True),
    }
    rows = []
    theme_scores: dict[str, float] = {}
    for theme in sorted(raw):
        row = {"theme_id": theme, **raw[theme]}
        for component in THEME_COMPONENTS:
            row[component + "_percentile"] = percentile[component][theme]
        row["theme_score_equal_weight_v1"] = float(
            np.mean([row[component + "_percentile"] for component in THEME_COMPONENTS])
        )
        row["diagnostic_only_components"] = True
        rows.append(row)
        theme_scores[theme] = row["theme_score_equal_weight_v1"]
    stock_scores = pd.Series({code: theme_scores[themes.loc[code]] for code in members}, dtype=float)
    stock_scores.name = "theme_score_equal_weight_v1"
    receipt_hashes = tuple(sorted(set(selected_bars.content_sha256.astype(str)) | set(selected_map.content_sha256.astype(str))))
    return ThemeCarrierResult(
        status="ok",
        abstain_reason=None,
        decision_session=today.date(),
        decision_at=cutoff,
        theme_components=pd.DataFrame(rows).sort_values("theme_id", kind="stable").reset_index(drop=True),
        stock_scores=stock_scores.sort_index(),
        coverage={
            "u0_members": len(members),
            "themes": len(raw),
            "primary_endpoint": "theme_score_equal_weight_v1",
            "component_results_diagnostic_only": True,
            "ledger_id": ledger_identity[0],
            "ledger_journal_hash": ledger_identity[1],
        },
        receipt_hashes=receipt_hashes,
    )


def order_new_entries(
    *, baseline_candidate_order: Sequence[str], retained_holdings: Sequence[str], result: ThemeCarrierResult
) -> EntryOrdering:
    """Reorder only candidates that are not retained holdings.

    This returns an ordering view rather than targets or weights, making it
    impossible for this signal module to modify retention, costs, risk or the
    account executor.
    """
    baseline = tuple(str(code) for code in baseline_candidate_order)
    retained = tuple(str(code) for code in retained_holdings)
    if len(set(baseline)) != len(baseline) or len(set(retained)) != len(retained):
        raise ValueError("duplicate_entry_or_retained_instrument")
    new_entries = tuple(code for code in baseline if code not in set(retained))
    if result.status != "ok":
        return EntryOrdering(baseline, retained, new_entries, False, result.abstain_reason)
    scores = result.stock_scores
    if any(code not in scores.index or not np.isfinite(float(scores.loc[code])) for code in new_entries):
        return EntryOrdering(baseline, retained, new_entries, False, "score_coverage_incomplete")
    baseline_rank = {code: index for index, code in enumerate(baseline)}
    ordered = tuple(sorted(new_entries, key=lambda code: (-float(scores.loc[code]), baseline_rank[code], code)))
    return EntryOrdering(baseline, retained, ordered, True, None)
