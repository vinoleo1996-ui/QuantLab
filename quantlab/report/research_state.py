"""Allowlisted, read-only projection of QuantLab research state.

This module deliberately does not discover files beneath ``output/``.  A
research result is visible only after its exact path and minimum schema have
been added to :data:`ALLOWLIST`.  It also never converts a conditional result,
a completed test suite, or a source-side promotion flag into adoption approval.

``load_research_catalog`` is the small I/O boundary.  ``build_research_catalog``
is pure: callers can supply captured :class:`ArtifactRecord` objects in tests
or another read-only adapter without touching the filesystem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


CATALOG_SCHEMA_VERSION = "1.0"
TARGET_NET_CAGR = 0.15


@dataclass(frozen=True)
class ArtifactSpec:
    """One permitted input; relative paths are intentionally compile-time data."""

    artifact_id: str
    relative_path: str
    label: str
    kind: str
    required_keys: tuple[str, ...] = ()
    required_marker: str | None = None
    integrity_level: str = "declared_artifact"


@dataclass(frozen=True)
class ArtifactRecord:
    """A read-only snapshot of a single allowlisted artifact."""

    artifact_id: str
    relative_path: str
    available: bool
    schema_valid: bool
    content_sha256: str | None
    payload: Mapping[str, Any] | str | None
    error: str | None = None


@dataclass(frozen=True)
class ResearchCatalog:
    """JSON-ready research state without a dependency on the web dashboard."""

    source_policy: Mapping[str, Any]
    current_execution_state: Mapping[str, Any]
    adoption: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    evidence_references: tuple[Mapping[str, Any], ...]
    open_questions: tuple[Mapping[str, Any], ...]
    integrity_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible representation for an API/UI."""

        # JSON round-tripping prevents a caller from mutating nested mappings
        # retained by this immutable catalog instance.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False, allow_nan=False))


# This is the entire input surface.  Do not replace it with a directory scan:
# individual raw fills, status files, incomplete attempts, and old run pointers
# do not carry sufficient context to describe research state safely.
ALLOWLIST: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        "continuity_ledger",
        "docs/RESEARCH_CONTINUITY.md",
        "研究接续台账",
        "markdown",
        required_marker="# QuantLab 研究接续台账",
        integrity_level="current_ledger_text",
    ),
    ArtifactSpec(
        "market_data_manifest",
        "data/local_market/1c85fd9aceaa03c3c4863bec/manifest.json",
        "冻结市场数据身份",
        "json",
        ("snapshot_id", "as_of", "panels", "accounting_gaps", "data_gaps"),
        integrity_level="declared_dataset_manifest",
    ),
    ArtifactSpec(
        "cadence_comparison",
        "output/continuity_review_20260927/cadence13/comparison_with13/comparison.json",
        "主账户换手节奏比较",
        "json",
        ("status", "promotion_approved", "arms"),
        integrity_level="recorded_comparison_artifact",
    ),
    ArtifactSpec(
        "continuity_attribution",
        "output/continuity_review_20260927/attribution_final/summary.json",
        "主账户事后归因",
        "json",
        ("status", "promotion_approved", "account_replayed"),
        integrity_level="recorded_posthoc_artifact",
    ),
    ArtifactSpec(
        "reference_pool_diagnostic",
        "output/reference_selection_20260927/pool_market_run2/DELIVERY.json",
        "参考池价格行为诊断",
        "json",
        ("status", "promotion_approved", "ruling", "unresolved_gates"),
        integrity_level="recorded_delivery_artifact",
    ),
    ArtifactSpec(
        "expert_router_delivery",
        "output/expert_router_20260927/DELIVERY.json",
        "原专家路由实验",
        "json",
        ("model_adopted", "automatic_trading", "result"),
        integrity_level="recorded_delivery_artifact",
    ),
    ArtifactSpec(
        "expert_selection_delivery",
        "output/expert_selection_20260927/DELIVERY.json",
        "互补专家实验",
        "json",
        ("status", "promotion_approved", "candidate_decisions", "frozen_ledger"),
        integrity_level="recorded_delivery_artifact",
    ),
    ArtifactSpec(
        "reference_challenge_delivery",
        "output/reference_challenge_20260927/DELIVERY.json",
        "盈利参考挑战",
        "json",
        ("status", "promotion_approved", "account_runs"),
        integrity_level="recorded_delivery_artifact",
    ),
    ArtifactSpec(
        "account_2025_delivery",
        "output/account_2025_20260927/FINAL_DELIVERY.json",
        "2025 新账户交付",
        "json",
        ("status", "promotion_approved", "main_new_accounts_run", "reference_new_accounts_run", "strategy_changed"),
        integrity_level="recorded_delivery_artifact",
    ),
    ArtifactSpec(
        "stable_book_decision",
        "output/stable_book_20260928/DECISION.json",
        "连续双腿错峰账户裁定",
        "json",
        ("decision", "promotion_approved", "reasons"),
        integrity_level="declared_decision_no_manifest",
    ),
    ArtifactSpec(
        "continuous_band_decision",
        "output/continuous_band_20260928/DECISION.json",
        "连续 50 只宽篮裁定",
        "json",
        ("decision", "promotion_approved", "reasons"),
        integrity_level="declared_decision_no_manifest",
    ),
    ArtifactSpec(
        "extended_2018_band_decision",
        "output/continuous_band_2018ext_20260928/DECISION.json",
        "补足 2018 年宽篮裁定",
        "json",
        ("decision", "promotion_approved", "reasons"),
        integrity_level="declared_decision_no_manifest",
    ),
    ArtifactSpec(
        "full_pool_decision",
        "output/full_pool_20260928/DECISION.json",
        "全池 50 只裁定",
        "json",
        ("decision", "promotion_approved", "reasons"),
        integrity_level="declared_decision_no_manifest",
    ),
)

_SPEC_BY_ID = {spec.artifact_id: spec for spec in ALLOWLIST}

_EVIDENCE_LABELS = {
    "continuity_ledger": "research_ledger_not_adoption_approval",
    "market_data_manifest": "provenance_only_not_live_market_data",
    "cadence_comparison": "conditional_historical_account_result",
    "continuity_attribution": "conditional_posthoc_not_account_replay",
    "reference_pool_diagnostic": "conditional_diagnostic_not_portfolio_alpha",
    "expert_router_delivery": "conditional_research_not_adopted",
    "expert_selection_delivery": "conditional_research_no_promotion",
    "reference_challenge_delivery": "conditional_comparison_not_adoption",
    "account_2025_delivery": "conditional_account_evidence_strategy_unchanged",
    "stable_book_decision": "conditional_closed_research_decision",
    "continuous_band_decision": "conditional_closed_research_decision",
    "extended_2018_band_decision": "conditional_closed_research_decision",
    "full_pool_decision": "conditional_closed_research_decision",
}


def _resolved_allowlisted_path(root: Path, spec: ArtifactSpec) -> Path:
    path = (root / spec.relative_path).resolve()
    if path != root and root not in path.parents:
        # The list is compile-time, but keep this guard if a future edit makes a
        # path escape the chosen project root.
        raise ValueError(f"allowlist_path_escapes_root:{spec.artifact_id}")
    return path


def _valid_payload(spec: ArtifactSpec, payload: Mapping[str, Any] | str) -> bool:
    if spec.kind == "markdown":
        return isinstance(payload, str) and bool(spec.required_marker) and spec.required_marker in payload
    return isinstance(payload, Mapping) and all(key in payload for key in spec.required_keys)


def collect_allowlisted_artifacts(repo_root: str | Path) -> dict[str, ArtifactRecord]:
    """Read only exact allowlisted paths; missing evidence is represented, never guessed.

    There is intentionally no ``glob``, ``rglob``, directory listing, or fallback
    to ``output/runs/latest.json`` here.
    """

    root = Path(repo_root).resolve()
    records: dict[str, ArtifactRecord] = {}
    for spec in ALLOWLIST:
        path = _resolved_allowlisted_path(root, spec)
        if not path.is_file():
            records[spec.artifact_id] = ArtifactRecord(
                artifact_id=spec.artifact_id,
                relative_path=spec.relative_path,
                available=False,
                schema_valid=False,
                content_sha256=None,
                payload=None,
                error="missing",
            )
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            payload: Mapping[str, Any] | str = text if spec.kind == "markdown" else json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            records[spec.artifact_id] = ArtifactRecord(
                artifact_id=spec.artifact_id,
                relative_path=spec.relative_path,
                available=True,
                schema_valid=False,
                content_sha256=None,
                payload=None,
                error="unreadable_or_invalid",
            )
            continue
        records[spec.artifact_id] = ArtifactRecord(
            artifact_id=spec.artifact_id,
            relative_path=spec.relative_path,
            available=True,
            schema_valid=_valid_payload(spec, payload),
            content_sha256=sha256(raw).hexdigest(),
            payload=payload,
            error=None if _valid_payload(spec, payload) else "schema_invalid",
        )
    return records


def _record(records: Mapping[str, ArtifactRecord], artifact_id: str) -> ArtifactRecord:
    """Give pure builders a safe missing record when a caller omits one."""

    record = records.get(artifact_id)
    if record is not None:
        return record
    spec = _SPEC_BY_ID[artifact_id]
    return ArtifactRecord(
        artifact_id=artifact_id,
        relative_path=spec.relative_path,
        available=False,
        schema_valid=False,
        content_sha256=None,
        payload=None,
        error="not_collected",
    )


def _mapping(record: ArtifactRecord) -> Mapping[str, Any]:
    return record.payload if record.schema_valid and isinstance(record.payload, Mapping) else {}


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    # Artifact strings are display metadata, not HTML.  Keeping a bounded text
    # value also prevents a damaged artifact from becoming a giant UI payload.
    return value[:500]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metrics(source: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, float | int | str]:
    values: dict[str, float | int | str] = {}
    for name in names:
        value = source.get(name)
        number = _finite_number(value)
        if number is not None:
            values[name] = number
        elif name in ("start", "end") and isinstance(value, str):
            values[name] = value
    return values


def _source_claimed_promotion(record: ArtifactRecord) -> bool:
    payload = _mapping(record)
    if payload.get("promotion_approved") is True:
        return True
    if record.artifact_id == "expert_selection_delivery":
        decisions = payload.get("candidate_decisions")
        if isinstance(decisions, Mapping):
            return any(
                isinstance(decision, Mapping) and decision.get("promotion_approved") is True
                for decision in decisions.values()
            )
    return False


def _source_state(record: ArtifactRecord, *keys: str) -> str | None:
    payload = _mapping(record)
    for key in keys:
        state = _text(payload.get(key))
        if state is not None:
            return state
    return None


def _evidence_reference(spec: ArtifactSpec, record: ArtifactRecord) -> dict[str, Any]:
    if not record.available:
        availability = "missing"
        integrity = "not_available"
    elif not record.schema_valid:
        availability = "invalid"
        integrity = "schema_invalid"
    else:
        availability = "available"
        integrity = spec.integrity_level
    return {
        "id": spec.artifact_id,
        "label": spec.label,
        "path": spec.relative_path,
        "availability": availability,
        "integrity": integrity,
        "evidence_label": _EVIDENCE_LABELS[spec.artifact_id] if record.schema_valid else "unavailable_evidence",
        "content_sha256": record.content_sha256,
        "caveat": "Explicit allowlisted evidence; this projection does not independently certify research conclusions.",
    }


def _candidate(
    *,
    candidate_id: str,
    label: str,
    role: str,
    source: ArtifactRecord,
    recorded_status: str | None,
    summary: str,
    metrics: Mapping[str, Any] | None = None,
    recorded_decision: str | None = None,
    closed: bool = False,
) -> dict[str, Any]:
    if source.schema_valid:
        evidence_label = "conditional_research_closed" if closed else "conditional_research_only"
    else:
        evidence_label = "unavailable_evidence_not_adoptable"
    return {
        "id": candidate_id,
        "label": label,
        "role": role,
        "evidence_label": evidence_label,
        "recorded_status": recorded_status if source.schema_valid else "evidence_unavailable",
        "recorded_decision": recorded_decision if source.schema_valid else None,
        "summary": summary,
        "metrics": dict(metrics or {}),
        "source_artifact": source.artifact_id,
        "source_path": source.relative_path,
        "source_claimed_promotion": _source_claimed_promotion(source),
        # This is a non-negotiable projection invariant.  A partial artifact
        # cannot grant adoption even if it accidentally claims otherwise.
        "promotion_approved": False,
        "adoptable": False,
    }


def _cadence_candidates(record: ArtifactRecord) -> list[dict[str, Any]]:
    payload = _mapping(record)
    arms = payload.get("arms") if isinstance(payload.get("arms"), Mapping) else {}
    definitions = (
        ("quality_pb_monthly", "quality/PB 月度", "qP2_monthly"),
        ("quality_pb_hold90", "quality/PB 月度 + 最低持有 90 日", "qP2_monthly_hold90"),
        ("cadence_13_weeks", "quality/PB 每 13 周", "cadence13"),
    )
    rows = []
    for candidate_id, label, arm_id in definitions:
        arm = arms.get(arm_id) if isinstance(arms.get(arm_id), Mapping) else {}
        rows.append(_candidate(
            candidate_id=candidate_id,
            label=label,
            role="main_account_research_candidate",
            source=record,
            recorded_status=_text(arm.get("status")),
            summary="同一历史样本的扣费研究账户结果；数据缺口和非独立验证阻断采用。",
            metrics=_metrics(arm, ("start", "end", "observations", "net_cagr", "max_drawdown")),
        ))
    return rows


def _expert_selection_candidates(record: ArtifactRecord) -> list[dict[str, Any]]:
    payload = _mapping(record)
    decisions = payload.get("candidate_decisions") if isinstance(payload.get("candidate_decisions"), Mapping) else {}
    definitions = (
        ("nonlinear_a1", "A1 非线性价量专家", "A1", False),
        ("nonlinear_a2", "A2 DoubleEnsemble 专家", "A2", True),
        ("retention_sorting_s", "S 保留排序", "S", True),
    )
    rows = []
    for candidate_id, label, decision_id, closed in definitions:
        decision = decisions.get(decision_id) if isinstance(decisions.get(decision_id), Mapping) else {}
        rows.append(_candidate(
            candidate_id=candidate_id,
            label=label,
            role="complementary_expert_candidate",
            source=record,
            recorded_status=_source_state(record, "status"),
            recorded_decision=_text(decision.get("decision")),
            summary="冻结研究实验；即使账户与测试记录完整，也没有取得可采用的稳定增量证据。",
            closed=closed,
        ))
    return rows


def _decision_candidate(
    record: ArtifactRecord,
    candidate_id: str,
    label: str,
) -> dict[str, Any]:
    payload = _mapping(record)
    return _candidate(
        candidate_id=candidate_id,
        label=label,
        role="closed_research_candidate",
        source=record,
        recorded_status="recorded_decision",
        recorded_decision=_text(payload.get("decision")),
        summary="已记录为未达到门槛的条件性研究裁定；不能从单个较好时钟或口径提升为采用。",
        metrics=_metrics(payload, ("cagr", "median_cagr", "max_drawdown", "cost2_cagr")),
        closed=True,
    )


def _data_gap_counts(record: ArtifactRecord) -> tuple[int | None, int | None]:
    payload = _mapping(record)
    if not payload:
        return None, None
    accounting = payload.get("accounting_gaps")
    data = payload.get("data_gaps")
    accounting_count = len(accounting) if isinstance(accounting, (list, tuple, Mapping)) else None
    data_count = len(data) if isinstance(data, (list, tuple, Mapping)) else None
    return accounting_count, data_count


def build_research_catalog(records: Mapping[str, ArtifactRecord]) -> ResearchCatalog:
    """Construct a catalog from already-read allowlisted records.

    This is intentionally deterministic and filesystem-free.  Unknown records
    are ignored; only identifiers declared in :data:`ALLOWLIST` can affect the
    projection.
    """

    selected = {spec.artifact_id: _record(records, spec.artifact_id) for spec in ALLOWLIST}
    cadence = selected["cadence_comparison"]
    expert_selection = selected["expert_selection_delivery"]
    router = selected["expert_router_delivery"]
    reference_challenge = selected["reference_challenge_delivery"]
    market_data = selected["market_data_manifest"]
    pool_diagnostic = selected["reference_pool_diagnostic"]
    account_2025 = selected["account_2025_delivery"]

    candidates = _cadence_candidates(cadence)
    candidates.extend(_expert_selection_candidates(expert_selection))
    router_payload = _mapping(router)
    candidates.append(_candidate(
        candidate_id="conditional_expert_router",
        label="原专家条件路由",
        role="research_candidate",
        source=router,
        recorded_status=_source_state(router, "result"),
        summary="相对原专家有记录改善，但没有通过盈利参考与风险门槛；模型未采用、未自动交易。",
    ))
    candidates.append(_candidate(
        candidate_id="profit_reference_challenge",
        label="盈利参考挑战",
        role="comparison_reference",
        source=reference_challenge,
        recorded_status=_source_state(reference_challenge, "status"),
        summary="比较基准而非生产策略；不同账户条件与条件性数据不构成采用许可。",
        metrics=_metrics(_mapping(reference_challenge), ("account_runs",)),
    ))
    candidates.extend((
        _decision_candidate(selected["stable_book_decision"], "stable_two_leg_book", "连续双腿错峰账户"),
        _decision_candidate(selected["continuous_band_decision"], "continuous_band_50", "连续 50 只宽篮"),
        _decision_candidate(selected["extended_2018_band_decision"], "extended_band_2018", "补足 2018 年宽篮"),
        _decision_candidate(selected["full_pool_decision"], "full_pool_50", "全池 50 只"),
    ))

    accounting_gaps, data_gaps = _data_gap_counts(market_data)
    cadence_arms = _mapping(cadence).get("arms")
    hold90 = cadence_arms.get("qP2_monthly_hold90") if isinstance(cadence_arms, Mapping) else {}
    hold90_cagr = _finite_number(hold90.get("net_cagr")) if isinstance(hold90, Mapping) else None
    performance_outcome = "blocked" if hold90_cagr is not None and hold90_cagr < TARGET_NET_CAGR else "unverified"
    data_outcome = "blocked" if (accounting_gaps or data_gaps) else "unverified"
    pool_payload = _mapping(pool_diagnostic)
    unresolved_gates = pool_payload.get("unresolved_gates") if isinstance(pool_payload.get("unresolved_gates"), list) else []

    adoption = {
        "status": "not_adoptable",
        "promotion_approved": False,
        "decision": "No allowlisted artifact grants strategy adoption, automatic trading, or a production-default change.",
        "gates": (
            {
                "id": "net_performance_target",
                "label": "扣费后长期年化 15%",
                "outcome": performance_outcome,
                "evidence_label": "conditional_or_missing_evidence",
                "observed_hold90_net_cagr": hold90_cagr,
                "source_artifacts": ("cadence_comparison",),
            },
            {
                "id": "pit_and_corporate_actions",
                "label": "严格 PIT、行业有效日与完整权益/公司行为",
                "outcome": data_outcome,
                "evidence_label": "blocked_or_unverified_data_acceptance",
                "accounting_gap_count": accounting_gaps,
                "data_gap_count": data_gaps,
                "source_artifacts": ("market_data_manifest", "continuity_ledger"),
            },
            {
                "id": "execution_pressure",
                "label": "实际持仓/退出执行压力",
                "outcome": "unverified",
                "evidence_label": "unverified_execution_evidence",
                "note": "未实际命中的压力场景不能作为压力通过。",
                "source_artifacts": ("expert_selection_delivery", "reference_pool_diagnostic"),
            },
            {
                "id": "independent_forward_record",
                "label": "冻结前向记录",
                "outcome": "unverified",
                "evidence_label": "unverified_forward_validation",
                "note": "历史重叠和已查看窗口不是独立盲样本。",
                "source_artifacts": ("continuity_ledger", "expert_selection_delivery"),
            },
            {
                "id": "promotion_control",
                "label": "生产授权与自动交易",
                "outcome": "blocked",
                "evidence_label": "promotion_forced_false",
                "note": "本投影始终输出 promotion_approved=false。",
                "source_artifacts": ("expert_router_delivery", "account_2025_delivery"),
            },
        ),
    }

    recorded_artifact_states = []
    for artifact_id, keys in (
        ("cadence_comparison", ("status",)),
        ("expert_selection_delivery", ("status",)),
        ("account_2025_delivery", ("status",)),
    ):
        record = selected[artifact_id]
        recorded_artifact_states.append({
            "artifact_id": artifact_id,
            "recorded_state": _source_state(record, *keys) if record.schema_valid else "evidence_unavailable",
            "is_live_process_state": False,
        })
    current_execution_state = {
        "observation": "not_observed_by_catalog",
        "scope": "read_only_artifact_projection",
        "account_or_process_state": "requires_explicit_runtime_probe",
        "legacy_latest_run_pointer": "intentionally_not_read",
        "note": "A historical artifact or a single latest-run pointer cannot prove current process state, holdings, signals, or adoption state.",
        "recorded_artifact_states": tuple(recorded_artifact_states),
    }

    open_questions = (
        {
            "id": "data_acceptance",
            "question": "能否补齐严格 PIT、行业有效日和完整权益/公司行为，以满足数据验收？",
            "evidence_label": "unverified_open_question",
            "source_artifacts": ("continuity_ledger", "market_data_manifest"),
        },
        {
            "id": "execution_pressure",
            "question": "实际持仓和退出在注册压力场景下是否会真正命中流动性/锁仓约束？",
            "evidence_label": "unverified_open_question",
            "source_artifacts": ("expert_selection_delivery", "reference_pool_diagnostic"),
        },
        {
            "id": "forward_record",
            "question": "如何形成可审计的冻结前向记录，而不把已查看的历史窗口冒充独立验证？",
            "evidence_label": "unverified_open_question",
            "source_artifacts": ("continuity_ledger", "expert_selection_delivery"),
        },
        {
            "id": "reference_causality",
            "question": "参考收益究竟依赖股票池、保留机制还是 P2 风险层？须一次只验证一个依赖。",
            "evidence_label": "unverified_open_question",
            "source_artifacts": ("continuity_ledger", "reference_challenge_delivery"),
        },
        {
            "id": "a1_stability_and_replacement_cost",
            "question": "A1 的近期失效与替换成本是否有可复核解释，且能否在独立证据上成立？",
            "evidence_label": "unverified_open_question",
            "source_artifacts": ("expert_selection_delivery",),
        },
    )

    warnings = []
    for spec in ALLOWLIST:
        record = selected[spec.artifact_id]
        if _source_claimed_promotion(record):
            warnings.append(f"{spec.artifact_id}:source_claimed_promotion")
        if record.available and not record.schema_valid:
            warnings.append(f"{spec.artifact_id}:schema_invalid")

    return ResearchCatalog(
        source_policy={
            "mode": "explicit_allowlist_only",
            "schema_version": CATALOG_SCHEMA_VERSION,
            "allowlisted_paths": tuple(spec.relative_path for spec in ALLOWLIST),
            "excluded_inputs": (
                "output/runs/**",
                "raw fills, targets, and daily returns",
                "status.json and incomplete attempts",
                "directory scans and fallback artifact discovery",
            ),
        },
        current_execution_state=current_execution_state,
        adoption=adoption,
        candidates=tuple(candidates),
        evidence_references=tuple(_evidence_reference(spec, selected[spec.artifact_id]) for spec in ALLOWLIST),
        open_questions=open_questions,
        integrity_warnings=tuple(warnings),
    )


def load_research_catalog(repo_root: str | Path) -> ResearchCatalog:
    """Read the fixed allowlist and return a safe research-state projection."""

    return build_research_catalog(collect_allowlisted_artifacts(repo_root))


__all__ = [
    "ALLOWLIST",
    "ArtifactRecord",
    "ArtifactSpec",
    "ResearchCatalog",
    "build_research_catalog",
    "collect_allowlisted_artifacts",
    "load_research_catalog",
]
