"""Forward-only, source-native theme-graph research signal.

``theme_carrier_minimal_v1`` intentionally requires exactly one primary theme
per stock.  This separate module implements the next, incompatible experiment:
every accepted source edge is retained, a stock can belong to multiple themes,
and its score is the maximum score among its eligible source-native themes.
It has no account, execution, network, or wall-clock dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import (
    DECISION_TIMEZONE,
    FORWARD_CONTRACT_VERSION,
    U0,
    UniverseSnapshot,
    forward_membership_sha256,
    is_exact_decision_cutoff,
)


# Frozen names shared semantically with ``theme_carrier_minimal_v1``.  Keep
# this tuple local: the graph is a separate research version and must not
# inherit an accidental future change to the primary-theme module.
THEME_GRAPH_COMPONENTS = (
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
_EDGE_COLUMNS = frozenset(
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
_THEME_CARRIER_GRAPH_V1_VERSION = "theme_carrier_graph_v1"
_THEME_CARRIER_GRAPH_V1_MIN_THEME_MEMBERS = 5
_THEME_CARRIER_GRAPH_V1_MIN_YESTERDAY_LIMIT_UP = 1
_THEME_CARRIER_GRAPH_V1_MIN_TODAY_TOUCHED_LIMIT = 1
_ACTIVE_EDGE_STATUS = "active"
_KNOWN_NO_ELIGIBLE_THEME_STATUS = "known_no_eligible_theme"
_SUPPORTING_RECEIPT_IDS = "supporting_receipt_ids"
_SUPPORTING_RECEIPT_HASHES = "supporting_content_sha256s"
_DETAIL_COLUMNS = (
    "instrument_id",
    "theme_membership_count",
    "eligible_theme_count",
    "winning_theme_id",
    "winning_theme_tie_count",
    "score_status",
)


@dataclass(frozen=True)
class ThemeCarrierGraphConfig:
    """Frozen eligibility policy for the source-native graph endpoint."""

    version: str = _THEME_CARRIER_GRAPH_V1_VERSION
    min_theme_members: int = _THEME_CARRIER_GRAPH_V1_MIN_THEME_MEMBERS
    min_yesterday_limit_up: int = _THEME_CARRIER_GRAPH_V1_MIN_YESTERDAY_LIMIT_UP
    min_today_touched_limit: int = _THEME_CARRIER_GRAPH_V1_MIN_TODAY_TOUCHED_LIMIT
    required_components: tuple[str, ...] = THEME_GRAPH_COMPONENTS

    def __post_init__(self) -> None:
        if (
            self.version != _THEME_CARRIER_GRAPH_V1_VERSION
            or self.min_theme_members != _THEME_CARRIER_GRAPH_V1_MIN_THEME_MEMBERS
            or self.min_yesterday_limit_up != _THEME_CARRIER_GRAPH_V1_MIN_YESTERDAY_LIMIT_UP
            or self.min_today_touched_limit != _THEME_CARRIER_GRAPH_V1_MIN_TODAY_TOUCHED_LIMIT
            or self.required_components != THEME_GRAPH_COMPONENTS
        ):
            raise ValueError("theme_carrier_graph_v1_config_is_frozen")


@dataclass(frozen=True)
class ThemeCarrierGraphResult:
    status: Literal["ok", "abstain"]
    abstain_reason: str | None
    decision_session: date
    decision_at: pd.Timestamp
    theme_components: pd.DataFrame
    stock_scores: pd.Series
    source_native_edges: pd.DataFrame
    stock_theme_details: pd.DataFrame
    coverage: Mapping[str, object]
    receipt_hashes: tuple[str, ...]
    # Row-linked receipts prove the direct source of each bar/edge.  A thin
    # provider adapter can additionally name supporting raw responses (for
    # example, the limit-price and broken-board pages used to validate a daily
    # bar).  Keep both identities on the score result so a later observation
    # cannot silently drop those dependencies from its evidence package.
    receipt_ids: tuple[str, ...] = ()

    @property
    def winning_theme_by_stock(self) -> pd.Series:
        """Deterministic source-native winning theme attribution by stock."""
        if self.stock_theme_details.empty:
            return pd.Series(dtype=object, name="winning_theme_id")
        return self.stock_theme_details.set_index("instrument_id")["winning_theme_id"].copy()

    @property
    def theme_membership_count_by_stock(self) -> pd.Series:
        """Number of accepted source-native theme edges by U0 stock."""
        if self.stock_theme_details.empty:
            return pd.Series(dtype="int64", name="theme_membership_count")
        return self.stock_theme_details.set_index("instrument_id")["theme_membership_count"].copy()


@dataclass(frozen=True)
class ThemeCarrierGraphEntryOrdering:
    """A new-entry ordering view that cannot change retained holdings."""

    baseline_candidate_order: tuple[str, ...]
    retained_holdings: tuple[str, ...]
    new_entry_order: tuple[str, ...]
    applied: bool
    abstain_reason: str | None


def _session(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("invalid_theme_graph_session")
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


def _is_missing(value: object) -> bool:
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _empty_result(
    *,
    decision_session: pd.Timestamp,
    decision_at: pd.Timestamp,
    reason: str,
    coverage: Mapping[str, object] | None = None,
    receipt_hashes: Iterable[str] = (),
    receipt_ids: Iterable[str] = (),
) -> ThemeCarrierGraphResult:
    return ThemeCarrierGraphResult(
        status="abstain",
        abstain_reason=reason,
        decision_session=decision_session.date(),
        decision_at=decision_at,
        theme_components=pd.DataFrame(columns=("theme_id",) + THEME_GRAPH_COMPONENTS),
        stock_scores=pd.Series(dtype=float, name="theme_score_equal_weight_graph_v1"),
        source_native_edges=pd.DataFrame(columns=sorted(_EDGE_COLUMNS)),
        stock_theme_details=pd.DataFrame(columns=_DETAIL_COLUMNS),
        coverage=dict(coverage or {}),
        receipt_hashes=tuple(sorted(set(str(value) for value in receipt_hashes))),
        receipt_ids=tuple(sorted(set(str(value) for value in receipt_ids))),
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
        and frame.content_sha256.map(
            lambda value: isinstance(value, str) and bool(_SHA256.fullmatch(value.lower()))
        ).all()
        and frame.capture_mode.astype(str).eq("forward_receipt").all()
    )


def _ledger_provenance_identity(
    frames: Iterable[pd.DataFrame], *, ledger: ResearchSignalLedger, decision_at: pd.Timestamp
) -> tuple[str, str] | None:
    """Bind every selected raw row to an accepted, on-time ledger receipt."""
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


def _supporting_receipt_identity(
    frames: Iterable[pd.DataFrame], *, ledger: ResearchSignalLedger, decision_at: pd.Timestamp
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Verify optional adapter-declared dependencies without inventing a source.

    Most direct callers need only the receipt linked to each selected row.  A
    provider-specific adapter may also depend on other raw responses to derive
    or cross-check that row.  If either optional supporting column is present,
    both must be present and every id/hash pair must resolve to an accepted,
    on-time ledger receipt.  The return is deliberately optional for backward
    compatible graph fixtures; it is never a caller-supplied unverified list.
    """
    receipt_ids: set[str] = set()
    receipt_hashes: set[str] = set()
    try:
        for frame in frames:
            has_ids = _SUPPORTING_RECEIPT_IDS in frame.columns
            has_hashes = _SUPPORTING_RECEIPT_HASHES in frame.columns
            if has_ids != has_hashes:
                return None
            if not has_ids:
                continue
            for _, row in frame.iterrows():
                raw_ids = row[_SUPPORTING_RECEIPT_IDS]
                raw_hashes = row[_SUPPORTING_RECEIPT_HASHES]
                if not isinstance(raw_ids, (tuple, list)) or not isinstance(raw_hashes, (tuple, list)):
                    return None
                ids = tuple(str(value) for value in raw_ids)
                hashes = tuple(str(value).lower() for value in raw_hashes)
                if not ids or len(ids) != len(hashes) or len(set(ids)) != len(ids):
                    return None
                for receipt_id, digest in zip(ids, hashes):
                    if not _SHA256.fullmatch(digest):
                        return None
                    evidence = ledger.export_receipt_evidence(receipt_id)
                    receipt = evidence["receipt"]
                    raw = evidence["raw"]
                    if (
                        evidence["canonical_receipt_id"] != receipt_id
                        or receipt["status"] != "accepted"
                        or raw["sha256"] != digest
                        or _local_time(receipt["available_at"]) is None
                        or _local_time(receipt["available_at"]) > decision_at
                        or _local_time(receipt["recorded_at"]) is None
                        or _local_time(receipt["recorded_at"]) > decision_at
                    ):
                        return None
                    receipt_ids.add(receipt_id)
                    receipt_hashes.add(digest)
    except (KeyError, TypeError, ValueError):
        return None
    return tuple(sorted(receipt_ids)), tuple(sorted(receipt_hashes))


def _valid_bar_values(frame: pd.DataFrame) -> bool:
    if not frame.quality_status.astype(str).eq("ok").all():
        return False
    numeric = ("open", "high", "low", "close", "reference_price", "limit_up", "price_tick")
    converted = frame.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(converted.to_numpy(dtype=float)).all() or (converted <= 0).any().any():
        return False
    return bool((converted.high >= converted.close).all())


def _percentiles(values: Mapping[str, float], *, higher_is_better: bool) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    ranks = series.rank(method="average", ascending=higher_is_better)
    return {theme: float(ranks.loc[theme] / len(series)) for theme in series.index}


def _u0_validation_reason(
    u0: UniverseSnapshot, *, today: pd.Timestamp, cutoff: pd.Timestamp
) -> str | None:
    if (
        not isinstance(u0, UniverseSnapshot)
        or u0.pool_id != "U0"
        or u0.contract_version != FORWARD_CONTRACT_VERSION
        or u0.status != "ok"
    ):
        return "forward_u0_required"
    if (
        not is_exact_decision_cutoff(u0.session, u0.decision_at)
        or _session(u0.session) != today
        or _local_time(u0.decision_at) != cutoff
    ):
        return "forward_u0_cutoff_mismatch"
    try:
        expected = forward_membership_sha256(
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
        return "forward_u0_identity_mismatch"
    if not isinstance(u0.membership_hash, str) or u0.membership_hash.lower() != expected:
        return "forward_u0_identity_mismatch"
    return None


def _source_native_edges(
    edges: pd.DataFrame, *, members: Sequence[str]
) -> tuple[dict[str, tuple[str, ...]], set[str]] | None:
    """Return every accepted edge, or reject an incomplete/ambiguous source map.

    ``active`` rows carry one source-native edge each.  A stock whose provider
    confirms that no eligible edge exists must instead have exactly one
    ``known_no_eligible_theme`` row with an empty ``theme_id``.  Mixing those
    states, repeating an edge, or omitting a U0 stock is ambiguity, not a
    reason to invent a primary theme or silently assign zero.
    """
    active_by_stock: dict[str, tuple[str, ...]] = {}
    known_no_eligible: set[str] = set()
    for code in members:
        rows = edges[edges.instrument_id.eq(code)]
        if rows.empty:
            return None
        statuses = tuple(rows.mapping_status.astype(str))
        if all(status == _ACTIVE_EDGE_STATUS for status in statuses):
            themes: list[str] = []
            for value in rows.theme_id:
                if _is_missing(value):
                    return None
                theme = str(value).strip()
                if not theme:
                    return None
                themes.append(theme)
            if len(set(themes)) != len(themes):
                return None
            active_by_stock[code] = tuple(themes)
            continue
        if all(status == _KNOWN_NO_ELIGIBLE_THEME_STATUS for status in statuses):
            if len(rows) != 1:
                return None
            value = rows.iloc[0].theme_id
            if not _is_missing(value) and str(value).strip():
                return None
            active_by_stock[code] = ()
            known_no_eligible.add(code)
            continue
        return None
    return active_by_stock, known_no_eligible


def _theme_rows(
    *,
    theme_members: Mapping[str, tuple[str, ...]],
    prior_limit_up: pd.Series,
    touched: pd.Series,
    eligible_theme_raw: Mapping[str, Mapping[str, float]],
    ineligible_reason: Mapping[str, str],
) -> pd.DataFrame:
    percentiles: dict[str, dict[str, float]] = {}
    if eligible_theme_raw:
        percentiles = {
            "prior_limit_up_premium": _percentiles(
                {theme: values["prior_limit_up_premium"] for theme, values in eligible_theme_raw.items()},
                higher_is_better=True,
            ),
            "broken_board_rate": _percentiles(
                {theme: values["broken_board_rate"] for theme, values in eligible_theme_raw.items()},
                higher_is_better=False,
            ),
            "theme_relative_strength": _percentiles(
                {theme: values["theme_relative_strength"] for theme, values in eligible_theme_raw.items()},
                higher_is_better=True,
            ),
            "advancing_breadth": _percentiles(
                {theme: values["advancing_breadth"] for theme, values in eligible_theme_raw.items()},
                higher_is_better=True,
            ),
        }
    rows: list[dict[str, object]] = []
    for theme in sorted(theme_members):
        codes = theme_members[theme]
        raw = eligible_theme_raw.get(theme)
        row: dict[str, object] = {
            "theme_id": theme,
            "member_count": len(codes),
            "yesterday_limit_up_count": int(prior_limit_up.loc[list(codes)].sum()),
            "today_touched_limit_count": int(touched.loc[list(codes)].sum()),
            "eligible_for_score": raw is not None,
            "ineligible_reason": None if raw is not None else ineligible_reason[theme],
            "diagnostic_only_components": True,
        }
        for component in THEME_GRAPH_COMPONENTS:
            row[component] = np.nan if raw is None else raw[component]
            row[component + "_percentile"] = (
                np.nan if raw is None else percentiles[component][theme]
            )
        row["theme_score_equal_weight_graph_v1"] = (
            np.nan
            if raw is None
            else float(np.mean([row[component + "_percentile"] for component in THEME_GRAPH_COMPONENTS]))
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("theme_id", kind="stable").reset_index(drop=True)


def _stock_details(
    *,
    members: Sequence[str],
    active_by_stock: Mapping[str, tuple[str, ...]],
    theme_scores: Mapping[str, float],
) -> tuple[pd.Series, pd.DataFrame]:
    scores: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for code in members:
        all_themes = active_by_stock[code]
        eligible = tuple(theme for theme in all_themes if theme in theme_scores)
        if eligible:
            maximum = max(theme_scores[theme] for theme in eligible)
            winners = tuple(sorted(theme for theme in eligible if theme_scores[theme] == maximum))
            scores[code] = float(maximum)
            row = {
                "instrument_id": code,
                "theme_membership_count": len(all_themes),
                "eligible_theme_count": len(eligible),
                "winning_theme_id": winners[0],
                "winning_theme_tie_count": len(winners),
                "score_status": "eligible_theme",
            }
        else:
            scores[code] = 0.0
            row = {
                "instrument_id": code,
                "theme_membership_count": len(all_themes),
                "eligible_theme_count": 0,
                "winning_theme_id": pd.NA,
                "winning_theme_tie_count": 0,
                "score_status": _KNOWN_NO_ELIGIBLE_THEME_STATUS,
            }
        rows.append(row)
    series = pd.Series(scores, dtype=float, name="theme_score_equal_weight_graph_v1").sort_index()
    details = pd.DataFrame(rows, columns=_DETAIL_COLUMNS).sort_values("instrument_id", kind="stable").reset_index(
        drop=True
    )
    return series, details


def score_theme_carrier_graph_v1(
    *,
    decision_session: object,
    prior_session: object,
    decision_at: object,
    u0: UniverseSnapshot,
    bars: pd.DataFrame,
    theme_edges: pd.DataFrame,
    ledger: ResearchSignalLedger,
    config: ThemeCarrierGraphConfig = ThemeCarrierGraphConfig(),
) -> ThemeCarrierGraphResult:
    """Score the frozen four-component graph endpoint or abstain as a whole.

    Mapping completeness is all-or-nothing across U0.  Eligibility, in
    contrast, is a property of a *known* source-native theme: an undersized
    or otherwise unscorable theme contributes no score, so a stock with no
    eligible theme receives the explicit score ``0.0`` rather than being
    silently dropped from the ranking.
    """
    _required(bars, _BAR_COLUMNS, "theme_carrier_graph_bar")
    _required(theme_edges, _EDGE_COLUMNS, "theme_carrier_graph_edge")
    today, yesterday = _session(decision_session), _session(prior_session)
    cutoff = _local_time(decision_at)
    if cutoff is None:
        raise ValueError("invalid_theme_graph_decision_at")
    if yesterday >= today:
        raise ValueError("prior_session_must_precede_decision_session")
    if not is_exact_decision_cutoff(today, cutoff):
        return _empty_result(
            decision_session=today,
            decision_at=cutoff,
            reason="decision_at_must_equal_session_1800_shanghai",
        )
    if _has_out_of_scope_input(bars) or _has_out_of_scope_input(theme_edges):
        return _empty_result(
            decision_session=today, decision_at=cutoff, reason="out_of_scope_theme_graph_input"
        )
    u0_reason = _u0_validation_reason(u0, today=today, cutoff=cutoff)
    if u0_reason is not None:
        return _empty_result(decision_session=today, decision_at=cutoff, reason=u0_reason)
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
        return _empty_result(
            decision_session=today, decision_at=cutoff, reason="incomplete_or_duplicate_bar_evidence"
        )
    if not _provenance_is_valid(selected_bars, cutoff) or not _valid_bar_values(selected_bars):
        return _empty_result(
            decision_session=today, decision_at=cutoff, reason="invalid_or_late_bar_provenance"
        )

    source_edges = theme_edges.copy()
    source_edges["instrument_id"] = source_edges.instrument_id.astype(str)
    source_edges["_session"] = source_edges.session.map(_session)
    selected_edges = source_edges[
        source_edges.instrument_id.isin(members) & source_edges._session.eq(today)
    ].copy()
    edge_state = _source_native_edges(selected_edges, members=members)
    if edge_state is None:
        return _empty_result(
            decision_session=today,
            decision_at=cutoff,
            reason="incomplete_or_ambiguous_theme_graph_mapping",
        )
    active_by_stock, explicit_no_eligible = edge_state
    if not _provenance_is_valid(selected_edges, cutoff):
        return _empty_result(
            decision_session=today,
            decision_at=cutoff,
            reason="invalid_or_late_theme_graph_provenance",
        )
    ledger_identity = _ledger_provenance_identity(
        (selected_bars, selected_edges), ledger=ledger, decision_at=cutoff
    )
    if ledger_identity is None:
        return _empty_result(
            decision_session=today, decision_at=cutoff, reason="unlinked_or_tampered_ledger_receipt"
        )
    supporting_receipts = _supporting_receipt_identity(
        (selected_bars, selected_edges), ledger=ledger, decision_at=cutoff
    )
    if supporting_receipts is None:
        return _empty_result(
            decision_session=today, decision_at=cutoff, reason="invalid_supporting_theme_graph_provenance"
        )

    prior = selected_bars[selected_bars._session.eq(yesterday)].set_index("instrument_id").loc[list(members)]
    current = selected_bars[selected_bars._session.eq(today)].set_index("instrument_id").loc[list(members)]
    close_prior = pd.to_numeric(prior.close, errors="coerce")
    close_current = pd.to_numeric(current.close, errors="coerce")
    returns = close_current / close_prior - 1.0
    prior_limit_up = close_prior >= pd.to_numeric(prior.limit_up, errors="coerce") - pd.to_numeric(
        prior.price_tick, errors="coerce"
    ) / 2
    touched = pd.to_numeric(current.high, errors="coerce") >= pd.to_numeric(
        current.limit_up, errors="coerce"
    ) - pd.to_numeric(current.price_tick, errors="coerce") / 2
    sealed = close_current >= pd.to_numeric(current.limit_up, errors="coerce") - pd.to_numeric(
        current.price_tick, errors="coerce"
    ) / 2
    broken = touched & ~sealed

    theme_member_lists: dict[str, list[str]] = {}
    for code in members:
        for theme in active_by_stock[code]:
            theme_member_lists.setdefault(theme, []).append(code)
    theme_members = {theme: tuple(codes) for theme, codes in theme_member_lists.items()}

    eligible_codes: dict[str, tuple[str, ...]] = {}
    ineligible_reason: dict[str, str] = {}
    for theme, codes in theme_members.items():
        limit_codes = tuple(code for code in codes if bool(prior_limit_up.loc[code]))
        touched_codes = tuple(code for code in codes if bool(touched.loc[code]))
        if len(codes) < config.min_theme_members:
            ineligible_reason[theme] = "insufficient_theme_members"
        elif len(limit_codes) < config.min_yesterday_limit_up:
            ineligible_reason[theme] = "insufficient_theme_yesterday_limit_up"
        elif len(touched_codes) < config.min_today_touched_limit:
            ineligible_reason[theme] = "insufficient_theme_touched_limit"
        else:
            eligible_codes[theme] = codes

    eligible_theme_raw: dict[str, dict[str, float]] = {}
    if eligible_codes:
        # Every eligible theme has at least one prior-limit-up member, so this
        # should be nonempty.  Keep the explicit failure branch as a guard
        # against future contract changes that relax that frozen threshold.
        if int(prior_limit_up.sum()) == 0:
            return _empty_result(
                decision_session=today, decision_at=cutoff, reason="no_market_yesterday_limit_up"
            )
        market_limit_return = float(returns.loc[prior_limit_up].mean())
        for theme, codes in eligible_codes.items():
            limit_codes = tuple(code for code in codes if bool(prior_limit_up.loc[code]))
            touched_codes = tuple(code for code in codes if bool(touched.loc[code]))
            theme_return = float(returns.loc[list(codes)].mean())
            raw = {
                "prior_limit_up_premium": float(returns.loc[list(limit_codes)].mean() - market_limit_return),
                "broken_board_rate": float(broken.loc[list(touched_codes)].mean()),
                "theme_relative_strength": float(theme_return - returns.mean()),
                "advancing_breadth": float(
                    (close_current.loc[list(codes)] > close_prior.loc[list(codes)]).mean()
                ),
            }
            if not all(np.isfinite(value) for value in raw.values()):
                return _empty_result(
                    decision_session=today, decision_at=cutoff, reason="nonfinite_theme_component"
                )
            eligible_theme_raw[theme] = raw

    components = _theme_rows(
        theme_members=theme_members,
        prior_limit_up=prior_limit_up,
        touched=touched,
        eligible_theme_raw=eligible_theme_raw,
        ineligible_reason=ineligible_reason,
    )
    theme_scores = {
        str(row["theme_id"]): float(row["theme_score_equal_weight_graph_v1"])
        for _, row in components[components.eligible_for_score].iterrows()
    }
    stock_scores, stock_details = _stock_details(
        members=members,
        active_by_stock=active_by_stock,
        theme_scores=theme_scores,
    )
    direct_receipt_ids = set(selected_bars.source_receipt_id.astype(str)) | set(selected_edges.source_receipt_id.astype(str))
    direct_receipt_hashes = set(selected_bars.content_sha256.astype(str)) | set(selected_edges.content_sha256.astype(str))
    receipt_ids = tuple(sorted(direct_receipt_ids | set(supporting_receipts[0])))
    receipt_hashes = tuple(sorted(direct_receipt_hashes | set(supporting_receipts[1])))
    return ThemeCarrierGraphResult(
        status="ok",
        abstain_reason=None,
        decision_session=today.date(),
        decision_at=cutoff,
        theme_components=components,
        stock_scores=stock_scores,
        source_native_edges=selected_edges.drop(columns="_session").sort_values(
            ["instrument_id", "mapping_status", "theme_id"], kind="stable", na_position="last"
        ).reset_index(drop=True),
        stock_theme_details=stock_details,
        coverage={
            "u0_members": len(members),
            "source_native_themes": len(theme_members),
            "active_theme_edges": sum(len(themes) for themes in active_by_stock.values()),
            "explicit_known_no_eligible_theme_stocks": len(explicit_no_eligible),
            "stocks_with_zero_eligible_theme": int(stock_details.eligible_theme_count.eq(0).sum()),
            "eligible_themes": len(theme_scores),
            "primary_endpoint": "theme_score_equal_weight_graph_v1",
            "component_results_diagnostic_only": True,
            "ledger_id": ledger_identity[0],
            "ledger_journal_hash": ledger_identity[1],
            "raw_receipt_count": len(receipt_ids),
        },
        receipt_hashes=receipt_hashes,
        receipt_ids=receipt_ids,
    )


def order_graph_new_entries(
    *,
    baseline_candidate_order: Sequence[str],
    retained_holdings: Sequence[str],
    result: ThemeCarrierGraphResult,
) -> ThemeCarrierGraphEntryOrdering:
    """Change only new-entry order, retaining baseline order for equal scores."""
    baseline = tuple(str(code) for code in baseline_candidate_order)
    retained = tuple(str(code) for code in retained_holdings)
    if len(set(baseline)) != len(baseline) or len(set(retained)) != len(retained):
        raise ValueError("duplicate_entry_or_retained_instrument")
    retained_set = set(retained)
    new_entries = tuple(code for code in baseline if code not in retained_set)
    if result.status != "ok":
        return ThemeCarrierGraphEntryOrdering(baseline, retained, new_entries, False, result.abstain_reason)
    scores = result.stock_scores
    if any(code not in scores.index or not np.isfinite(float(scores.loc[code])) for code in new_entries):
        return ThemeCarrierGraphEntryOrdering(baseline, retained, new_entries, False, "score_coverage_incomplete")
    baseline_rank = {code: index for index, code in enumerate(baseline)}
    ordered = tuple(sorted(new_entries, key=lambda code: (-float(scores.loc[code]), baseline_rank[code])))
    return ThemeCarrierGraphEntryOrdering(baseline, retained, ordered, True, None)


# Match the minimal endpoint's entry-ordering spelling while keeping this
# module's graph-specific result type isolated from its primary-theme API.
order_new_entries = order_graph_new_entries
