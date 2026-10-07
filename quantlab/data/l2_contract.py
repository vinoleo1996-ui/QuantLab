"""Fail-closed contracts for an externally supplied L2 score experiment.

This module intentionally stops at data, labels, and a ranking score artifact.
It has no dependency on the account engine, portfolio weights, or the frozen
expert router.  A later, separately preregistered adapter may consume a valid
``score`` only to order new entries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.data.contracts import (
    InstrumentRules,
    asof_records,
    decision_time,
    fingerprint,
    timestamp,
)


class L2ContractError(ValueError):
    """The L2 experiment must abstain rather than infer a missing contract."""


# These are the reviewed QuantMind catalog *keys*, not copied calculations or
# a claim that a provider has supplied a qualified historical field.
L2_FEATURE_KEYS = (
    "micro_vpin_vol_ratio",
    "micro_vpin_amount_ratio",
    "micro_pin",
    "micro_zone_distribution",
    "micro_zone_vol_ratio_T4",
    "micro_zone_vol_ratio_T6",
    "micro_zone_vol_ratio_T5",
    "micro_zone_vol_ratio_T3",
    "micro_zone_rv_ratio_close",
    "vol_price_divergence",
    "micro_open_gap",
    "micro_impact_decay_half_life",
    "micro_liquidity_daily_pattern",
    "flow_imbalance_revert_speed",
)

RECEIPT_COLUMNS = (
    "decision_date",
    "instrument_id",
    "feature_key",
    "value",
    "unit",
    "source_field",
    "available_at",
    "source_version",
    "raw_response_sha256",
    "ledger_id",
    "ledger_receipt_id",
    "provider",
    "provider_trial_id",
    "source_as_of",
    "quality_status",
)

SCORE_COLUMNS = (
    "decision_date",
    "instrument_id",
    "score",
    "score_rank",
    "model_id",
    "model_family",
    "model_config_sha256",
    "model_contract_sha256",
    "training_as_of",
    "training_label_available_at",
    "feature_manifest_sha256",
    "label_contract_sha256",
    "dataset_identity_sha256",
    "matrix_row_sha256",
    "materialized_matrix_sha256",
    "horizon_sessions",
    "status",
    "abstain_reason",
)

MATERIALIZED_MATRIX_SCHEMA = "quantlab.l2.materialized-matrix.v1"
_MATERIALIZED_MATRIX_ATTR = "materialized_matrix_sha256"
_MATERIALIZED_MATRIX_SCHEMA_ATTR = "materialized_matrix_schema"
_MATERIALIZED_MATRIX_MANIFEST_ATTR = "feature_manifest_sha256"
_MATRIX_IDENTITY_COLUMNS = (
    "decision_date",
    "instrument_id",
    *L2_FEATURE_KEYS,
    "feature_available_at",
    "source_as_of_max",
    "receipt_set_sha256",
    "ledger_available_at_max",
    "ledger_recorded_at_max",
    "ledger_receipt_set_sha256",
    "dataset_identity_sha256",
    "feature_manifest_sha256",
    "status",
    "abstain_reason",
)

# A score artifact must not quietly carry a position/execution policy beside a
# ranking score.  Keep this explicit rather than trying to infer semantics from
# a numeric column later in the account path.
FORBIDDEN_SCORE_COLUMNS = frozenset(
    {
        "weight",
        "weights",
        "position_score",
        "position_weight",
        "fusion_score",
        "p3",
        "kelly",
        "kelly_fraction",
        "topk_dropout",
        "stop_loss",
        "auto_model_switch",
        "fallback_model_id",
        "fallback_model_family",
        "fallback_score",
        "fallback_used",
        "fallback_reason",
        "execution_policy",
        "allocation_policy",
        "risk_policy",
    }
)

_FORBIDDEN_SCORE_POLICY_TOKENS = frozenset(
    {
        "allocation",
        "buy",
        "dropout",
        "execution",
        "fallback",
        "fusion",
        "kelly",
        "loss",
        "order",
        "p3",
        "position",
        "rating",
        "recommendation",
        "risk",
        "sell",
        "stop",
        "target",
        "topk",
        "trade",
        "weight",
        "weights",
    }
)


def _normalized_score_column(value: object) -> str:
    """Normalise namespaced/case variants before applying policy-field bans."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _forbidden_score_columns(columns: Iterable[object]) -> list[str]:
    forbidden_normalized = {_normalized_score_column(name) for name in FORBIDDEN_SCORE_COLUMNS}
    forbidden_collapsed = {name.replace("_", "") for name in forbidden_normalized}
    rejected = []
    for column in columns:
        normalized = _normalized_score_column(column)
        collapsed = normalized.replace("_", "")
        # A namespaced policy field such as quality.position.position_score is
        # still a policy field; only the plain score artifact is admissible.
        if (
            normalized in forbidden_normalized
            or collapsed in forbidden_collapsed
            or any(
                normalized == name
                or normalized.startswith(name + "_")
                or normalized.endswith("_" + name)
                or ("_" + name + "_") in normalized
                for name in forbidden_normalized
            )
            or any(token in normalized.split("_") for token in _FORBIDDEN_SCORE_POLICY_TOKENS)
        ):
            rejected.append(str(column))
    return sorted(rejected)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_IDENTITY = re.compile(r"^sha256:[0-9a-f]{64}$")


def _default_permission_adapter_identity() -> str:
    """Freeze the source bytes of the default permission adapter without importing it."""
    try:
        spec = importlib.util.find_spec("quantlab.backtest.ledger")
        origin = None if spec is None else spec.origin
        if not origin:
            raise FileNotFoundError("missing_default_permission_adapter_source")
        source = Path(origin)
        return "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise L2ContractError("l2_default_permission_adapter_source_unavailable") from exc


def _session(value) -> pd.Timestamp:
    """Normalise a scalar to an exchange-session date without UTC drift."""
    try:
        value = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(value):
        return pd.NaT
    if value.tzinfo is not None:
        value = value.tz_convert("Asia/Shanghai").tz_localize(None)
    return value.normalize()


def _timestamp_or_nat(value) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        return timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT


def _valid_sha256(value) -> bool:
    return bool(_SHA256.fullmatch(str(value).lower()))


def _valid_code_identity(value) -> bool:
    return bool(_CODE_IDENTITY.fullmatch(str(value).lower()))


def _reasons(values: Iterable[str]) -> str:
    return "|".join(sorted({str(value) for value in values if value}))


def _identity_timestamp(value) -> str | None:
    stamp = _timestamp_or_nat(value)
    return None if pd.isna(stamp) else stamp.isoformat()


def _identity_number(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _optional_sha256(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
    return str(value).lower() if _valid_sha256(value) else None


def _missing_identity(value) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _matrix_row_identity(row: Mapping[str, object]) -> str:
    """Hash a materialized row after normalising its time and numeric scalars."""
    try:
        decision = _session(row["decision_date"])
    except (KeyError, TypeError, ValueError):
        decision = pd.NaT
    payload = {
        "schema": MATERIALIZED_MATRIX_SCHEMA,
        "decision_date": None if pd.isna(decision) else decision.date().isoformat(),
        "instrument_id": str(row.get("instrument_id", "")),
        "features": {key: _identity_number(row.get(key)) for key in L2_FEATURE_KEYS},
        "feature_available_at": _identity_timestamp(row.get("feature_available_at")),
        "source_as_of_max": _identity_timestamp(row.get("source_as_of_max")),
        "receipt_set_sha256": _optional_sha256(row.get("receipt_set_sha256")),
        "ledger_available_at_max": _identity_timestamp(row.get("ledger_available_at_max")),
        "ledger_recorded_at_max": _identity_timestamp(row.get("ledger_recorded_at_max")),
        "ledger_receipt_set_sha256": _optional_sha256(row.get("ledger_receipt_set_sha256")),
        "dataset_identity_sha256": _optional_sha256(row.get("dataset_identity_sha256")),
        "feature_manifest_sha256": _optional_sha256(row.get("feature_manifest_sha256")),
        "status": str(row.get("status", "")),
        "abstain_reason": str(row.get("abstain_reason", "")),
    }
    return fingerprint(payload)


def _materialized_matrix_identity(matrix: pd.DataFrame) -> str:
    entries = []
    for _, row in matrix.iterrows():
        decision = _session(row.decision_date)
        entries.append(
            (
                decision.date().isoformat(),
                str(row.instrument_id),
                str(row.matrix_row_sha256).lower(),
            )
        )
    return fingerprint(
        {
            "schema": MATERIALIZED_MATRIX_SCHEMA,
            "rows": sorted(entries),
        }
    )


@dataclass(frozen=True)
class L2FeatureSpec:
    """One versioned L2 field; all source semantics are declared up front."""

    key: str
    unit: str
    dtype: str
    source_field: str
    source_version: str
    known_at_field: str = "available_at"
    missing_policy: str = "abstain"

    def __post_init__(self):
        if self.key not in L2_FEATURE_KEYS:
            raise L2ContractError("unknown_l2_feature_key:" + str(self.key))
        for field in ("unit", "dtype", "source_field", "source_version", "known_at_field"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise L2ContractError("feature_spec_missing:" + field + ":" + self.key)
        if self.known_at_field != "available_at":
            raise L2ContractError("feature_known_at_field_must_be_available_at:" + self.key)
        if self.missing_policy != "abstain":
            raise L2ContractError("feature_missing_policy_must_abstain:" + self.key)


@dataclass(frozen=True)
class L2FeatureManifest:
    """Identity of the trial data, not a provider assertion of PIT validity."""

    provider: str
    trial_id: str
    source_dataset_id: str
    source_release_id: str
    fields: tuple[L2FeatureSpec, ...]
    schema_version: str = "l2-feature-manifest-v1"

    def __post_init__(self):
        for field in ("provider", "trial_id", "source_dataset_id", "source_release_id", "schema_version"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise L2ContractError("manifest_missing:" + field)
        if tuple(spec.key for spec in self.fields) != L2_FEATURE_KEYS:
            raise L2ContractError("manifest_must_have_exact_fixed_l2_feature_keys")
        if len({spec.key for spec in self.fields}) != len(self.fields):
            raise L2ContractError("manifest_duplicate_feature_key")

    @property
    def sha256(self) -> str:
        return fingerprint(asdict(self))

    @property
    def by_key(self) -> dict[str, L2FeatureSpec]:
        return {spec.key: spec for spec in self.fields}


def fixed_l2_manifest(*, provider: str, trial_id: str, source_dataset_id: str,
                      source_release_id: str, field_metadata: Mapping[str, Mapping[str, str]]) -> L2FeatureManifest:
    """Build the fixed 14-field manifest without inventing provider metadata."""
    if set(field_metadata) != set(L2_FEATURE_KEYS):
        raise L2ContractError("field_metadata_keys_must_match_fixed_l2_manifest")
    fields = tuple(
        L2FeatureSpec(key=key, **dict(field_metadata[key]))
        for key in L2_FEATURE_KEYS
    )
    return L2FeatureManifest(provider, trial_id, source_dataset_id, source_release_id, fields)


@dataclass(frozen=True)
class L2ScoreModelContract:
    """Freeze the sole permitted scorer before score artifacts are admitted.

    A failed CatBoost scorer must produce an abstention row.  It must never
    silently substitute a linear/tree fallback whose identity or configuration
    is not the preregistered L2 scorer.
    """

    model_id: str
    model_config_sha256: str
    model_family: str = "catboost"
    fallback_model_id: str | None = None
    fallback_policy: str = "abstain"
    schema_version: str = "l2-catboost-score-model-v1"

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise L2ContractError("l2_model_id_missing")
        if self.model_family != "catboost":
            raise L2ContractError("l2_model_must_be_catboost")
        if not _valid_sha256(self.model_config_sha256):
            raise L2ContractError("l2_model_config_hash_invalid")
        if self.fallback_model_id is not None:
            raise L2ContractError("l2_model_fallback_forbidden")
        if self.fallback_policy != "abstain":
            raise L2ContractError("l2_model_fallback_policy_must_abstain")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise L2ContractError("l2_model_schema_version_missing")

    @property
    def sha256(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class L2LabelContract:
    """A horizon-specific, open-to-open label whose outcome cannot mature early."""

    horizon_sessions: int = 5
    execution_lag_sessions: int = 1
    entry_offset_sessions: int = 1
    exit_offset_sessions: int = 6
    decision_cutoff: str = "18:00 Asia/Shanghai"
    return_basis: str = "gross_open_to_open_cash_adjusted"
    allow_open_assumption: bool = False
    permission_adapter_identity: str = field(default_factory=_default_permission_adapter_identity)
    name: str = "l2_open_to_open_v1"

    def __post_init__(self):
        if self.horizon_sessions not in (5, 20):
            raise L2ContractError("l2_horizon_must_be_5_or_20")
        if self.execution_lag_sessions != 1 or self.entry_offset_sessions != 1:
            raise L2ContractError("l2_execution_lag_must_be_next_open")
        if self.exit_offset_sessions != self.horizon_sessions + self.execution_lag_sessions:
            raise L2ContractError("l2_exit_offset_must_equal_horizon_plus_execution_lag")
        if self.decision_cutoff != "18:00 Asia/Shanghai":
            raise L2ContractError("l2_decision_cutoff_must_be_1800_shanghai")
        if self.return_basis != "gross_open_to_open_cash_adjusted":
            raise L2ContractError("l2_return_basis_must_be_gross_open_to_open_cash_adjusted")
        if not isinstance(self.allow_open_assumption, bool):
            raise L2ContractError("l2_allow_open_assumption_must_be_bool")
        if not _valid_code_identity(self.permission_adapter_identity):
            raise L2ContractError("l2_permission_adapter_identity_must_be_code_hash")
        if not self.name:
            raise L2ContractError("l2_label_contract_name_missing")

    @property
    def embargo_sessions(self) -> int:
        return self.horizon_sessions + self.execution_lag_sessions

    @property
    def sha256(self) -> str:
        return fingerprint(asdict(self) | {"embargo_sessions": self.embargo_sessions})

    @classmethod
    def h5(cls) -> "L2LabelContract":
        return cls(horizon_sessions=5, exit_offset_sessions=6, name="l2_h5_open_to_open_v1")

    @classmethod
    def h20(cls) -> "L2LabelContract":
        return cls(horizon_sessions=20, exit_offset_sessions=21, name="l2_h20_open_to_open_v1")


def validate_horizon_pair(h5: L2LabelContract, h20: L2LabelContract) -> None:
    """Prevent an H=5 model artifact from being relabelled as the H=20 sleeve."""
    if h5.horizon_sessions != 5 or h20.horizon_sessions != 20:
        raise L2ContractError("l2_h5_and_h20_contracts_required")
    if h5.sha256 == h20.sha256 or h5.embargo_sessions == h20.embargo_sessions:
        raise L2ContractError("l2_horizon_contracts_not_separate")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise L2ContractError(label + "_must_be_dataframe")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise L2ContractError(label + "_columns_missing:" + ",".join(missing))


def _require_receipt_ledger(receipt_ledger):
    """Use the canonical append-only ledger, not a caller-provided hash oracle."""
    from quantlab.research_signal_ledger import ResearchSignalLedger

    if not isinstance(receipt_ledger, ResearchSignalLedger):
        raise L2ContractError("l2_receipt_ledger_required")
    return receipt_ledger


def _canonical_ledger_receipt(receipt_ledger, receipt_id: str, cache: dict[str, dict | None]) -> dict | None:
    """Read one receipt through the ledger's integrity-checking public export."""
    if receipt_id in cache:
        return cache[receipt_id]
    try:
        evidence = receipt_ledger.export_receipt_evidence(receipt_id)
        receipt = evidence["receipt"]
        raw = evidence["raw"]
        body = receipt["body"]
        canonical_id = evidence["canonical_receipt_id"]
        result = dict(
            ledger_id=evidence["ledger_id"],
            canonical_receipt_id=canonical_id,
            event_hash=receipt["event_hash"],
            raw_sha256=raw["sha256"],
            receipt_raw_sha256=body["raw_sha256"],
            available_at=_timestamp_or_nat(receipt["available_at"]),
            recorded_at=_timestamp_or_nat(receipt["recorded_at"]),
            kind=receipt["kind"],
            status=receipt["status"],
            receipt_event_id=receipt["event_id"],
        )
        if (
            not isinstance(result["ledger_id"], str)
            or not result["ledger_id"].strip()
            or result["canonical_receipt_id"] != result["receipt_event_id"]
            or result["kind"] != "receipt"
            or result["status"] != "accepted"
            or not _valid_sha256(result["event_hash"])
            or not _valid_sha256(result["raw_sha256"])
            or str(result["raw_sha256"]).lower() != str(result["receipt_raw_sha256"]).lower()
            or pd.isna(result["available_at"])
            or pd.isna(result["recorded_at"])
        ):
            raise ValueError("invalid_canonical_receipt_evidence")
    except (KeyError, TypeError, ValueError, OSError):
        result = None
    cache[receipt_id] = result
    return result


def validate_l2_receipts(receipts: pd.DataFrame, manifest: L2FeatureManifest, *,
                         receipt_ledger) -> pd.DataFrame:
    """Annotate each receipt; malformed rows remain visible as abstention evidence."""
    _require_columns(receipts, RECEIPT_COLUMNS, "l2_receipt")
    ledger = _require_receipt_ledger(receipt_ledger)
    out = receipts.copy()
    out["decision_date"] = out["decision_date"].map(_session)
    out["_available_at"] = out["available_at"].map(_timestamp_or_nat)
    out["_source_as_of"] = out["source_as_of"].map(_timestamp_or_nat)
    out["_decision_at"] = out["decision_date"].map(lambda d: decision_time(d) if pd.notna(d) else pd.NaT)
    specs = manifest.by_key
    status, reasons = [], []
    ledger_cache: dict[str, dict | None] = {}
    ledger_metadata: list[dict] = []
    for _, row in out.iterrows():
        row_reasons: list[str] = []
        metadata: dict = {}
        key = row.feature_key
        spec = specs.get(key)
        if pd.isna(row.decision_date):
            row_reasons.append("invalid_decision_date")
        if not isinstance(row.instrument_id, str) or not row.instrument_id.strip():
            row_reasons.append("invalid_instrument_id")
        if spec is None:
            row_reasons.append("unknown_feature_key")
        else:
            if str(row.unit) != spec.unit:
                row_reasons.append("unit_mismatch:" + str(key))
            if str(row.source_field) != spec.source_field:
                row_reasons.append("source_field_mismatch:" + str(key))
            if str(row.source_version) != spec.source_version:
                row_reasons.append("source_version_mismatch:" + str(key))
        try:
            finite = np.isfinite(float(row.value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            row_reasons.append("nonfinite_feature_value")
        if pd.isna(row._available_at):
            row_reasons.append("missing_feature_available_at")
        elif not pd.isna(row._decision_at) and row._available_at > row._decision_at:
            row_reasons.append("feature_available_after_decision")
        if pd.isna(row._source_as_of):
            row_reasons.append("missing_source_as_of")
        elif not pd.isna(row._decision_at) and row._source_as_of > row._decision_at:
            row_reasons.append("source_as_of_after_decision")
        if pd.notna(row._source_as_of) and pd.notna(row._available_at) and row._source_as_of > row._available_at:
            row_reasons.append("source_as_of_after_available_at")
        if str(row.provider) != manifest.provider:
            row_reasons.append("provider_mismatch")
        if str(row.provider_trial_id) != manifest.trial_id:
            row_reasons.append("provider_trial_mismatch")
        if str(row.quality_status) != "ok":
            row_reasons.append("receipt_quality_not_ok")
        if not _valid_sha256(row.raw_response_sha256):
            row_reasons.append("invalid_raw_response_sha256")
        ledger_id = row["ledger_id"]
        receipt_id = row["ledger_receipt_id"]
        if not isinstance(ledger_id, str) or not ledger_id.strip():
            row_reasons.append("missing_ledger_id")
        if not isinstance(receipt_id, str) or not receipt_id.strip():
            row_reasons.append("missing_ledger_receipt_id")
            evidence = None
        else:
            evidence = _canonical_ledger_receipt(ledger, receipt_id, ledger_cache)
            if evidence is None:
                row_reasons.append("ledger_receipt_not_verified")
        if evidence is not None:
            ledger_reasons: list[str] = []
            if ledger_id != evidence["ledger_id"]:
                ledger_reasons.append("ledger_id_mismatch")
            if str(row.raw_response_sha256).lower() != str(evidence["raw_sha256"]).lower():
                ledger_reasons.append("raw_sha256_not_ledger_verified")
            if pd.notna(row._available_at) and row._available_at < evidence["available_at"]:
                ledger_reasons.append("feature_available_before_ledger_receipt")
            if pd.notna(row._decision_at) and evidence["recorded_at"] > row._decision_at:
                ledger_reasons.append("ledger_receipt_recorded_after_decision")
            row_reasons.extend(ledger_reasons)
            if not ledger_reasons:
                metadata = dict(
                    ledger_id=evidence["ledger_id"],
                    canonical_receipt_id=evidence["canonical_receipt_id"],
                    event_hash=str(evidence["event_hash"]).lower(),
                    raw_sha256=str(evidence["raw_sha256"]).lower(),
                    available_at=evidence["available_at"],
                    recorded_at=evidence["recorded_at"],
                )
        ledger_metadata.append(metadata)
        status.append("ok" if not row_reasons else "abstain")
        reasons.append(_reasons(row_reasons))
    out["validation_status"] = status
    out["validation_reason"] = reasons
    out["_ledger_canonical_receipt_id"] = [item.get("canonical_receipt_id") for item in ledger_metadata]
    out["_ledger_event_hash"] = [item.get("event_hash") for item in ledger_metadata]
    out["_ledger_raw_sha256"] = [item.get("raw_sha256") for item in ledger_metadata]
    out["_ledger_available_at"] = [item.get("available_at", pd.NaT) for item in ledger_metadata]
    out["_ledger_recorded_at"] = [item.get("recorded_at", pd.NaT) for item in ledger_metadata]
    duplicate = out.duplicated(["decision_date", "instrument_id", "feature_key"], keep=False)
    if duplicate.any():
        out.loc[duplicate, "validation_status"] = "abstain"
        out.loc[duplicate, "validation_reason"] = out.loc[duplicate, "validation_reason"].map(
            lambda value: _reasons([value, "duplicate_feature_receipt"])
        )
    return out


def materialize_l2_matrix(receipts: pd.DataFrame, manifest: L2FeatureManifest, *,
                          receipt_ledger) -> pd.DataFrame:
    """Materialise complete 14-field cross-sections; every defect is an abstention."""
    checked = validate_l2_receipts(receipts, manifest, receipt_ledger=receipt_ledger)
    columns = [
        "decision_date", "instrument_id", *L2_FEATURE_KEYS,
        "feature_available_at", "source_as_of_max", "receipt_set_sha256",
        "ledger_available_at_max", "ledger_recorded_at_max", "ledger_receipt_set_sha256",
        "dataset_identity_sha256", "feature_manifest_sha256", "status", "abstain_reason",
        "matrix_row_sha256",
    ]
    rows = []
    for (day, code), block in checked.groupby(["decision_date", "instrument_id"], dropna=False, sort=True):
        reasons = list(block.loc[block.validation_status.ne("ok"), "validation_reason"])
        values: dict[str, float] = {}
        available, source_asof, hashes = [], [], []
        ledger_available, ledger_recorded, ledger_identities = [], [], []
        for key in L2_FEATURE_KEYS:
            matches = block[block.feature_key.eq(key)]
            if len(matches) == 0:
                reasons.append("missing_feature:" + key)
                continue
            if len(matches) != 1:
                reasons.append("duplicate_feature_receipt:" + key)
                continue
            row = matches.iloc[0]
            if row.validation_status != "ok":
                reasons.append(row.validation_reason)
                continue
            values[key] = float(row.value)
            available.append(row._available_at)
            source_asof.append(row._source_as_of)
            hashes.append(str(row.raw_response_sha256).lower())
            if (
                not isinstance(row["_ledger_canonical_receipt_id"], str)
                or not _valid_sha256(row["_ledger_event_hash"])
                or not _valid_sha256(row["_ledger_raw_sha256"])
                or pd.isna(row["_ledger_available_at"])
                or pd.isna(row["_ledger_recorded_at"])
            ):
                reasons.append("ledger_receipt_identity_missing:" + key)
                continue
            ledger_available.append(row["_ledger_available_at"])
            ledger_recorded.append(row["_ledger_recorded_at"])
            ledger_identities.append(dict(
                feature_key=key,
                ledger_id=str(row["ledger_id"]),
                canonical_receipt_id=row["_ledger_canonical_receipt_id"],
                event_hash=str(row["_ledger_event_hash"]).lower(),
                raw_sha256=str(row["_ledger_raw_sha256"]).lower(),
                available_at=row["_ledger_available_at"].isoformat(),
                recorded_at=row["_ledger_recorded_at"].isoformat(),
            ))
        reason = _reasons(reasons)
        ok = not reason and len(values) == len(L2_FEATURE_KEYS) and len(ledger_identities) == len(L2_FEATURE_KEYS)
        receipt_set = fingerprint(sorted(hashes)) if hashes else None
        ledger_receipt_set = fingerprint(ledger_identities) if len(ledger_identities) == len(L2_FEATURE_KEYS) else None
        identity = fingerprint({
            "feature_manifest_sha256": manifest.sha256,
            "receipt_set_sha256": receipt_set,
            "ledger_receipt_set_sha256": ledger_receipt_set,
            "decision_date": str(day),
            "instrument_id": str(code),
        }) if receipt_set and ledger_receipt_set else None
        row = dict(
            decision_date=day,
            instrument_id=str(code),
            feature_available_at=max(available) if available else pd.NaT,
            source_as_of_max=max(source_asof) if source_asof else pd.NaT,
            receipt_set_sha256=receipt_set,
            ledger_available_at_max=max(ledger_available) if ledger_available else pd.NaT,
            ledger_recorded_at_max=max(ledger_recorded) if ledger_recorded else pd.NaT,
            ledger_receipt_set_sha256=ledger_receipt_set,
            dataset_identity_sha256=identity,
            feature_manifest_sha256=manifest.sha256,
            status="ok" if ok else "abstain",
            abstain_reason="" if ok else reason,
        )
        for key in L2_FEATURE_KEYS:
            row[key] = values.get(key, np.nan) if ok else np.nan
        row["matrix_row_sha256"] = _matrix_row_identity(row)
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    out.attrs[_MATERIALIZED_MATRIX_SCHEMA_ATTR] = MATERIALIZED_MATRIX_SCHEMA
    out.attrs[_MATERIALIZED_MATRIX_MANIFEST_ATTR] = manifest.sha256
    out.attrs[_MATERIALIZED_MATRIX_ATTR] = _materialized_matrix_identity(out)
    return out


def _rule_map(dataset, day: pd.Timestamp, cache: dict[pd.Timestamp, Mapping[str, InstrumentRules]],
              rules_for_date: Callable[[pd.Timestamp], Mapping[str, InstrumentRules]] | None) -> Mapping[str, InstrumentRules]:
    if day in cache:
        return cache[day]
    try:
        if rules_for_date is not None:
            rules = dict(rules_for_date(day))
        else:
            opening = timestamp(day) + pd.Timedelta(hours=9, minutes=30)
            records = asof_records(dataset.instruments, opening, effective=opening)
            rules = {code: InstrumentRules.from_record(row) for code, row in records.iterrows()}
    except (KeyError, TypeError, ValueError):
        rules = {}
    cache[day] = rules
    return rules


def _gap_in_path(dataset, code: str, entry: pd.Timestamp, exit_: pd.Timestamp) -> bool:
    for gap in getattr(dataset, "manifest", {}).get("data_gaps", []):
        if gap.get("instrument_id") != code:
            continue
        date = _session(gap.get("date"))
        if pd.notna(date) and entry <= date <= exit_:
            return True
    return False


def _action_adjustment(dataset, code: str, entry: pd.Timestamp, exit_: pd.Timestamp, maturity: pd.Timestamp):
    """Mirror the existing price-path label convention without inventing events."""
    actions = getattr(dataset, "actions", pd.DataFrame())
    if actions is None or actions.empty:
        return 1.0, 0.0, ""
    required = {"instrument_id", "ex_date", "record_date", "available_at", "cash_per_share", "share_ratio"}
    if not required.issubset(actions.columns):
        return None, None, "action_contract_missing"
    rows = actions[actions.instrument_id.eq(code)].copy()
    if rows.empty:
        return 1.0, 0.0, ""
    rows["_ex"] = rows.ex_date.map(_session)
    rows = rows[(rows._ex >= entry) & (rows._ex <= exit_)]
    if rows.empty:
        return 1.0, 0.0, ""
    rows["_available"] = rows.available_at.map(_timestamp_or_nat)
    if rows._available.isna().any() or (rows._available > maturity).any():
        return None, None, "corporate_action_not_available_by_label_maturity"
    shares, cash = 1.0, 0.0
    for _, action in rows.sort_values(["_ex", "_available"]).iterrows():
        record = _session(action.record_date)
        # Retain the existing label convention: a record date before the entry
        # does not manufacture an entitlement for a later entrant.
        if pd.isna(record) or record < entry:
            continue
        if str(action.get("action_type", "distribution")) != "distribution":
            return None, None, "unsupported_corporate_action"
        try:
            payout, ratio = float(action.cash_per_share), float(action.share_ratio)
        except (TypeError, ValueError):
            return None, None, "invalid_corporate_action"
        if not np.isfinite(payout) or not np.isfinite(ratio) or payout < 0 or ratio < 0:
            return None, None, "invalid_corporate_action"
        cash += shares * payout
        shares *= 1.0 + ratio
    return shares, cash, ""


def build_executable_labels(dataset, contract: L2LabelContract | None = None, *,
                            codes: Sequence[str] | None = None,
                            rules_for_date: Callable[[pd.Timestamp], Mapping[str, InstrumentRules]] | None = None,
                            allow_open_assumption: bool | None = None,
                            permission_adapter: Callable | None = None,
                            permission_adapter_identity: str | None = None) -> pd.DataFrame:
    """Build status-explicit H=5/H=20 labels; unknown execution never becomes zero.

    The label is a gross, corporate-action-adjusted open-price path.  It is not
    a net account PnL target: lot size, cash, participation, and fees require a
    separate account experiment and must not leak into this information test.
    """
    contract = contract or L2LabelContract.h5()
    if allow_open_assumption is not None:
        if not isinstance(allow_open_assumption, bool) or allow_open_assumption != contract.allow_open_assumption:
            raise L2ContractError("l2_open_assumption_contract_mismatch")
    effective_open_assumption = contract.allow_open_assumption
    if permission_adapter is None:
        if contract.permission_adapter_identity != _default_permission_adapter_identity():
            raise L2ContractError("l2_custom_permission_adapter_required")
        from quantlab.backtest.ledger import trade_permission
        permission = trade_permission
    else:
        if not callable(permission_adapter):
            raise L2ContractError("l2_permission_adapter_not_callable")
        if not _valid_code_identity(permission_adapter_identity):
            raise L2ContractError("l2_permission_adapter_identity_required")
        if permission_adapter_identity != contract.permission_adapter_identity:
            raise L2ContractError("l2_permission_adapter_identity_mismatch")
        permission = permission_adapter
    calendar = pd.DatetimeIndex(getattr(dataset, "calendar", ()))
    if calendar.empty or not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise L2ContractError("invalid_label_calendar")
    all_codes = sorted(codes if codes is not None else getattr(dataset, "panels", {}).keys())
    rules_cache: dict[pd.Timestamp, Mapping[str, InstrumentRules]] = {}
    rows = []
    for j, signal in enumerate(calendar):
        entry_index, exit_index = j + contract.entry_offset_sessions, j + contract.exit_offset_sessions
        for code in all_codes:
            row = dict(
                decision_date=signal,
                instrument_id=str(code),
                horizon_sessions=contract.horizon_sessions,
                execution_lag_sessions=contract.execution_lag_sessions,
                entry_date=calendar[entry_index] if entry_index < len(calendar) else pd.NaT,
                exit_date=calendar[exit_index] if exit_index < len(calendar) else pd.NaT,
                entry_open=np.nan,
                exit_open=np.nan,
                target_gross_return=np.nan,
                label_end=calendar[exit_index] if exit_index < len(calendar) else pd.NaT,
                label_available_at=pd.NaT,
                entry_permission=False,
                exit_permission=False,
                status="not_matured",
                reason="label_endpoint_outside_calendar",
                label_contract_sha256=contract.sha256,
            )
            if exit_index >= len(calendar) or entry_index >= len(calendar):
                rows.append(row)
                continue
            panel = getattr(dataset, "panels", {}).get(code)
            if panel is None:
                row.update(status="abstain", reason="missing_panel")
                rows.append(row)
                continue
            panel = panel.reindex(calendar)
            entry, exit_ = calendar[entry_index], calendar[exit_index]
            entry_bar, exit_bar = panel.loc[entry].to_dict(), panel.loc[exit_].to_dict()
            if entry_bar.get("quality_status") != "ok":
                row.update(status="abstain", reason="entry_bar_quality_not_ok")
                rows.append(row)
                continue
            if exit_bar.get("quality_status") != "ok":
                row.update(status="abstain", reason="exit_bar_quality_not_ok")
                rows.append(row)
                continue
            entry_rule = _rule_map(dataset, entry, rules_cache, rules_for_date).get(code)
            exit_rule = _rule_map(dataset, exit_, rules_cache, rules_for_date).get(code)
            if entry_rule is None:
                row.update(status="abstain", reason="entry_rule_unavailable")
                rows.append(row)
                continue
            entry_ok, entry_reason = permission(
                entry_bar, entry_rule, "buy", entry, allow_open_assumption=effective_open_assumption
            )
            row["entry_permission"] = bool(entry_ok)
            if not entry_ok:
                row.update(status="abstain", reason="entry_" + entry_reason)
                rows.append(row)
                continue
            if exit_rule is None:
                row.update(status="abstain", reason="exit_rule_unavailable")
                rows.append(row)
                continue
            exit_ok, exit_reason = permission(
                exit_bar, exit_rule, "sell", exit_, allow_open_assumption=effective_open_assumption
            )
            row["exit_permission"] = bool(exit_ok)
            if not exit_ok:
                row.update(status="abstain", reason="exit_" + exit_reason)
                rows.append(row)
                continue
            entry_open, exit_open = entry_bar.get("open", np.nan), exit_bar.get("open", np.nan)
            if not np.isfinite(entry_open) or not np.isfinite(exit_open) or entry_open <= 0 or exit_open <= 0:
                row.update(status="abstain", reason="missing_open_price")
                rows.append(row)
                continue
            exit_available = _timestamp_or_nat(exit_bar.get("available_at"))
            maturity = decision_time(exit_)
            if pd.isna(exit_available) or exit_available > maturity:
                row.update(status="abstain", reason="exit_bar_not_available_by_label_maturity")
                rows.append(row)
                continue
            if _gap_in_path(dataset, str(code), entry, exit_):
                row.update(status="abstain", reason="accounting_gap_in_label_path")
                rows.append(row)
                continue
            shares, cash, action_reason = _action_adjustment(dataset, str(code), entry, exit_, maturity)
            if action_reason:
                row.update(status="abstain", reason=action_reason)
                rows.append(row)
                continue
            available_at = max(exit_available, maturity)
            row.update(
                entry_open=float(entry_open),
                exit_open=float(exit_open),
                target_gross_return=float((exit_open * shares + cash) / entry_open - 1.0),
                label_available_at=available_at,
                status="observed",
                reason="",
            )
            rows.append(row)
    return pd.DataFrame(rows)


def validate_l2_labels(labels: pd.DataFrame, contract: L2LabelContract, *,
                       calendar: Sequence[pd.Timestamp] | None = None) -> pd.DataFrame:
    required = (
        "decision_date", "instrument_id", "horizon_sessions", "execution_lag_sessions",
        "entry_date", "exit_date", "target_gross_return", "label_end",
        "label_available_at", "status", "reason", "label_contract_sha256",
    )
    _require_columns(labels, required, "l2_label")
    out = labels.copy()
    if out.duplicated(["decision_date", "instrument_id"]).any():
        raise L2ContractError("duplicate_l2_label_identity")
    if not out.horizon_sessions.eq(contract.horizon_sessions).all():
        raise L2ContractError("l2_label_horizon_mismatch")
    if not out.execution_lag_sessions.eq(contract.execution_lag_sessions).all():
        raise L2ContractError("l2_label_execution_lag_mismatch")
    if not out.label_contract_sha256.eq(contract.sha256).all():
        raise L2ContractError("l2_label_contract_hash_mismatch")
    observed = out.status.eq("observed")
    if observed.any():
        for field in ("entry_date", "exit_date", "label_end", "label_available_at"):
            if out.loc[observed, field].map(lambda value: pd.notna(value)).eq(False).any():
                raise L2ContractError("observed_l2_label_missing:" + field)
        if not np.isfinite(out.loc[observed, "target_gross_return"].astype(float)).all():
            raise L2ContractError("observed_l2_label_nonfinite_target")
        for _, row in out.loc[observed].iterrows():
            exit_ = _session(row.exit_date)
            if _session(row.label_end) != exit_:
                raise L2ContractError("l2_label_end_mismatch")
            if _timestamp_or_nat(row.label_available_at) < decision_time(exit_):
                raise L2ContractError("l2_label_matured_before_exit_close")
    if calendar is not None:
        cal = pd.DatetimeIndex(calendar)
        for _, row in out.loc[observed].iterrows():
            signal, entry, exit_ = _session(row.decision_date), _session(row.entry_date), _session(row.exit_date)
            positions = cal.get_indexer([signal, entry, exit_])
            if (positions < 0).any():
                raise L2ContractError("l2_label_date_outside_calendar")
            if positions[1] - positions[0] != contract.entry_offset_sessions:
                raise L2ContractError("l2_label_entry_offset_mismatch")
            if positions[2] - positions[0] != contract.exit_offset_sessions:
                raise L2ContractError("l2_label_exit_offset_mismatch")
    return out


def purge_l2_training_rows(labels: pd.DataFrame, boundary_at, contract: L2LabelContract) -> pd.DataFrame:
    """Use each label's real endpoint *and* availability timestamp for embargo."""
    checked = validate_l2_labels(labels, contract)
    boundary = _timestamp_or_nat(boundary_at)
    if pd.isna(boundary):
        raise L2ContractError("invalid_l2_split_boundary")
    ends = checked.label_end.map(_session)
    available = checked.label_available_at.map(_timestamp_or_nat)
    before_boundary_day = ends < boundary.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    mature_before_boundary = available < boundary
    keep = checked.status.eq("observed") & before_boundary_day & mature_before_boundary
    out = checked.loc[keep].copy()
    out.attrs.update(
        embargo_sessions=contract.embargo_sessions,
        boundary_at=boundary.isoformat(),
        total_rows=int(len(checked)),
        retained_rows=int(len(out)),
        purged_rows=int((~keep).sum()),
        purge_rule="observed_and_actual_label_end_before_boundary_and_label_available_at_before_boundary",
    )
    return out


def _validate_qualified_materialized_matrix(
    matrix: pd.DataFrame, manifest: L2FeatureManifest
) -> tuple[pd.DataFrame, str]:
    """Accept only a complete, hash-bound output of ``materialize_l2_matrix``."""
    _require_columns(matrix, (*_MATRIX_IDENTITY_COLUMNS, "matrix_row_sha256"), "l2_matrix")
    if matrix.empty:
        raise L2ContractError("l2_qualified_matrix_empty")
    if matrix.attrs.get(_MATERIALIZED_MATRIX_SCHEMA_ATTR) != MATERIALIZED_MATRIX_SCHEMA:
        raise L2ContractError("l2_qualified_matrix_schema_required")
    if matrix.attrs.get(_MATERIALIZED_MATRIX_MANIFEST_ATTR) != manifest.sha256:
        raise L2ContractError("l2_qualified_matrix_manifest_identity_mismatch")
    declared_identity = matrix.attrs.get(_MATERIALIZED_MATRIX_ATTR)
    if not _valid_sha256(declared_identity):
        raise L2ContractError("l2_qualified_matrix_identity_required")
    checked = matrix.copy()
    checked["decision_date"] = checked.decision_date.map(_session)
    if checked.decision_date.isna().any() or checked.duplicated(["decision_date", "instrument_id"]).any():
        raise L2ContractError("invalid_or_duplicate_l2_matrix_identity")
    for _, row in checked.iterrows():
        if not isinstance(row.instrument_id, str) or not row.instrument_id.strip():
            raise L2ContractError("l2_matrix_invalid_instrument_id")
        if row.feature_manifest_sha256 != manifest.sha256:
            raise L2ContractError("l2_matrix_feature_manifest_hash_mismatch")
        if not _valid_sha256(row.matrix_row_sha256):
            raise L2ContractError("l2_matrix_row_identity_invalid")
        if str(row.matrix_row_sha256).lower() != _matrix_row_identity(row.to_dict()):
            raise L2ContractError("l2_matrix_row_identity_mismatch")
        feature_values = pd.to_numeric(pd.Series({key: row[key] for key in L2_FEATURE_KEYS}), errors="coerce")
        if row.status == "ok":
            if str(row.abstain_reason):
                raise L2ContractError("l2_matrix_ok_row_has_abstain_reason")
            if not np.isfinite(feature_values.to_numpy(dtype=float)).all():
                raise L2ContractError("l2_matrix_ok_row_nonfinite_feature")
            if not all(
                _valid_sha256(row[field])
                for field in (
                    "receipt_set_sha256",
                    "ledger_receipt_set_sha256",
                    "dataset_identity_sha256",
                )
            ):
                raise L2ContractError("l2_matrix_ok_row_identity_invalid")
            feature_available = _timestamp_or_nat(row.feature_available_at)
            source_as_of = _timestamp_or_nat(row.source_as_of_max)
            ledger_available = _timestamp_or_nat(row.ledger_available_at_max)
            ledger_recorded = _timestamp_or_nat(row.ledger_recorded_at_max)
            decision = decision_time(row.decision_date)
            if (
                pd.isna(feature_available)
                or pd.isna(source_as_of)
                or pd.isna(ledger_available)
                or pd.isna(ledger_recorded)
                or feature_available > decision
                or source_as_of > feature_available
                or ledger_available > decision
                or ledger_recorded > decision
            ):
                raise L2ContractError("l2_matrix_ok_row_availability_invalid")
        elif row.status == "abstain":
            if not str(row.abstain_reason):
                raise L2ContractError("l2_matrix_abstain_reason_missing")
            if np.isfinite(feature_values.to_numpy(dtype=float)).any():
                raise L2ContractError("l2_matrix_abstain_row_has_feature")
        else:
            raise L2ContractError("l2_matrix_unknown_status")
    computed_identity = _materialized_matrix_identity(checked)
    if str(declared_identity).lower() != computed_identity:
        raise L2ContractError("l2_qualified_matrix_identity_mismatch")
    return checked, computed_identity


def validate_l2_scores(scores: pd.DataFrame, manifest: L2FeatureManifest, contract: L2LabelContract,
                       *, model_contract: L2ScoreModelContract,
                       matrix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate a ranking-only score artifact before any future adapter reads it."""
    _require_columns(scores, SCORE_COLUMNS, "l2_score")
    if not isinstance(model_contract, L2ScoreModelContract):
        raise L2ContractError("l2_score_model_contract_required")
    if matrix is None:
        raise L2ContractError("l2_score_qualified_matrix_required")
    matrix_index, matrix_identity = _validate_qualified_materialized_matrix(matrix, manifest)
    forbidden = _forbidden_score_columns(scores.columns)
    if forbidden:
        raise L2ContractError("l2_score_contains_position_or_policy_columns:" + ",".join(forbidden))
    unexpected = sorted(set(scores.columns) - set(SCORE_COLUMNS))
    if unexpected:
        raise L2ContractError("l2_score_unexpected_columns:" + ",".join(unexpected))
    out = scores.copy()
    if out.empty:
        raise L2ContractError("l2_score_empty")
    out["decision_date"] = out.decision_date.map(_session)
    if out.decision_date.isna().any() or out.duplicated(["decision_date", "instrument_id"]).any():
        raise L2ContractError("invalid_or_duplicate_l2_score_identity")
    if not out.horizon_sessions.eq(contract.horizon_sessions).all():
        raise L2ContractError("l2_score_horizon_mismatch")
    if not out.feature_manifest_sha256.eq(manifest.sha256).all():
        raise L2ContractError("l2_score_feature_manifest_hash_mismatch")
    if not out.label_contract_sha256.eq(contract.sha256).all():
        raise L2ContractError("l2_score_label_contract_hash_mismatch")
    if not out.model_id.eq(model_contract.model_id).all():
        raise L2ContractError("l2_score_model_id_mismatch")
    if not out.model_family.eq(model_contract.model_family).all():
        raise L2ContractError("l2_score_model_family_mismatch")
    if not out.model_config_sha256.eq(model_contract.model_config_sha256).all():
        raise L2ContractError("l2_score_model_config_hash_mismatch")
    if not out.model_contract_sha256.eq(model_contract.sha256).all():
        raise L2ContractError("l2_score_model_contract_hash_mismatch")
    for _, row in out.iterrows():
        if not isinstance(row.instrument_id, str) or not row.instrument_id.strip():
            raise L2ContractError("l2_score_invalid_instrument_id")
        if not _valid_sha256(row.matrix_row_sha256):
            raise L2ContractError("l2_score_matrix_row_identity_invalid")
        if not _valid_sha256(row.materialized_matrix_sha256) or row.materialized_matrix_sha256.lower() != matrix_identity:
            raise L2ContractError("l2_score_materialized_matrix_identity_mismatch")
        cutoff, label_cutoff = _timestamp_or_nat(row.training_as_of), _timestamp_or_nat(row.training_label_available_at)
        decision = decision_time(row.decision_date)
        if pd.isna(cutoff) or pd.isna(label_cutoff) or cutoff >= decision or label_cutoff >= decision:
            raise L2ContractError("l2_score_training_not_matured_before_decision")
        if label_cutoff > cutoff:
            raise L2ContractError("l2_score_training_labels_not_available_by_training_as_of")
        if row.status == "ok":
            try:
                score, score_rank = float(row.score), float(row.score_rank)
            except (TypeError, ValueError):
                raise L2ContractError("l2_score_nonfinite") from None
            if not np.isfinite(score) or not np.isfinite(score_rank):
                raise L2ContractError("l2_score_nonfinite")
            if not 0.0 <= score_rank <= 1.0:
                raise L2ContractError("l2_score_rank_outside_unit_interval")
            if str(row.abstain_reason):
                raise L2ContractError("l2_score_ok_row_has_abstain_reason")
        elif row.status == "abstain":
            if np.isfinite(pd.to_numeric(pd.Series([row.score]), errors="coerce").iloc[0]) or np.isfinite(pd.to_numeric(pd.Series([row.score_rank]), errors="coerce").iloc[0]):
                raise L2ContractError("l2_score_abstain_row_has_score")
            if not str(row.abstain_reason):
                raise L2ContractError("l2_score_abstain_reason_missing")
        else:
            raise L2ContractError("l2_score_unknown_status")
    matrix_index = matrix_index.set_index(["decision_date", "instrument_id"])
    for _, row in out.iterrows():
        key = (row.decision_date, row.instrument_id)
        if key not in matrix_index.index:
            raise L2ContractError("l2_score_without_feature_matrix_row")
        source = matrix_index.loc[key]
        if str(source.matrix_row_sha256).lower() != str(row.matrix_row_sha256).lower():
            raise L2ContractError("l2_score_matrix_row_identity_mismatch")
        expected_dataset = _optional_sha256(source.dataset_identity_sha256)
        if expected_dataset is None:
            if not _missing_identity(row.dataset_identity_sha256):
                raise L2ContractError("l2_score_abstained_matrix_identity_must_be_empty")
        elif (
            not _valid_sha256(row.dataset_identity_sha256)
            or str(row.dataset_identity_sha256).lower() != expected_dataset
        ):
            raise L2ContractError("l2_score_matrix_identity_mismatch")
        if row.status == "ok" and source.status != "ok":
            raise L2ContractError("l2_score_from_abstained_feature_row")
    return out


@dataclass(frozen=True)
class EntryOnlyOrder:
    """A future adapter's input: retention is immutable, only fresh-entry order exists."""

    retained_holdings: tuple[str, ...]
    ordered_new_entries: tuple[str, ...]


def order_new_entries(retained_holdings: Sequence[str], fixed_entry_candidates: Sequence[str],
                      scores: Mapping[str, float] | pd.Series) -> EntryOnlyOrder:
    """Sort a fixed new-entry set without selecting weights or modifying holdings."""
    def codes(values: object, label: str) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise L2ContractError(label + "_must_be_sequence")
        try:
            result = tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise L2ContractError(label + "_must_be_sequence") from exc
        if any(not isinstance(code, str) or not code or code != code.strip() for code in result):
            raise L2ContractError(label + "_invalid_code")
        if len(set(result)) != len(result):
            raise L2ContractError(label + "_duplicate_code")
        return result

    retained = codes(retained_holdings, "entry_only_retained")
    candidates = codes(fixed_entry_candidates, "entry_only_candidate")
    if len(set(retained)) != len(retained) or len(set(candidates)) != len(candidates):
        raise L2ContractError("entry_only_sort_duplicate_code")
    overlap = sorted(set(retained).intersection(candidates))
    if overlap:
        raise L2ContractError("retained_holding_in_new_entry_pool:" + ",".join(overlap))
    if isinstance(scores, pd.Series):
        score_items = list(scores.items())
    elif isinstance(scores, Mapping):
        score_items = list(scores.items())
    else:
        raise L2ContractError("entry_only_scores_must_be_mapping_or_series")
    score_codes = codes(tuple(code for code, _ in score_items), "entry_only_score_index")
    if set(score_codes) != set(candidates):
        missing = sorted(set(candidates) - set(score_codes))
        unexpected = sorted(set(score_codes) - set(candidates))
        if missing and not unexpected:
            raise L2ContractError("entry_only_score_missing:" + ",".join(missing))
        if unexpected and not missing:
            raise L2ContractError("entry_only_score_unexpected:" + ",".join(unexpected))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise L2ContractError("entry_only_score_universe_mismatch:" + ";".join(details))
    values: dict[str, float] = {}
    for code, value in score_items:
        if isinstance(value, (bool, np.bool_)):
            raise L2ContractError("entry_only_score_nonfinite")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise L2ContractError("entry_only_score_nonfinite") from exc
        if not np.isfinite(numeric):
            raise L2ContractError("entry_only_score_nonfinite")
        values[code] = numeric
    if set(values) != set(candidates):
        raise L2ContractError("entry_only_score_universe_mismatch")
    if not np.isfinite(np.array([values[code] for code in candidates], dtype=float)).all():
        raise L2ContractError("entry_only_score_nonfinite")
    ordered = tuple(sorted(candidates, key=lambda code: (-values[code], code)))
    return EntryOnlyOrder(retained, ordered)


def order_new_entries_from_l2_score_artifact(
    *,
    retained_holdings: Sequence[str],
    fixed_entry_candidates: Sequence[str],
    decision_date,
    scores: pd.DataFrame,
    manifest: L2FeatureManifest,
    contract: L2LabelContract,
    model_contract: L2ScoreModelContract,
    matrix: pd.DataFrame,
) -> EntryOnlyOrder:
    """Create an entry-only order from a verified, separately trained H=20 score.

    This is deliberately an adapter *boundary*, not an account call.  H=5 is
    an information-only experiment and must never be promoted by reusing its
    rows as a 20-session entry sorter.  The returned object retains the same
    no-weights/no-policy semantics as :func:`order_new_entries`.
    """
    if not isinstance(contract, L2LabelContract) or contract.horizon_sessions != 20:
        raise L2ContractError("l2_entry_order_requires_h20_contract")
    session = _session(decision_date)
    if pd.isna(session):
        raise L2ContractError("l2_entry_order_invalid_decision_date")
    if isinstance(fixed_entry_candidates, (str, bytes)):
        raise L2ContractError("l2_entry_order_candidates_must_be_sequence")
    try:
        candidates = tuple(fixed_entry_candidates)
    except TypeError as exc:
        raise L2ContractError("l2_entry_order_candidates_must_be_sequence") from exc
    if any(not isinstance(code, str) or not code or code != code.strip() for code in candidates):
        raise L2ContractError("l2_entry_order_invalid_candidate")
    if len(set(candidates)) != len(candidates):
        raise L2ContractError("l2_entry_order_duplicate_candidate")

    checked = validate_l2_scores(
        scores,
        manifest,
        contract,
        model_contract=model_contract,
        matrix=matrix,
    )
    rows = checked.loc[checked.decision_date.eq(session)].copy()
    by_instrument = {str(row.instrument_id): row for _, row in rows.iterrows()}
    values: dict[str, float] = {}
    for code in candidates:
        row = by_instrument.get(code)
        if row is None:
            raise L2ContractError("l2_entry_order_score_missing:" + code)
        if row.status != "ok":
            raise L2ContractError("l2_entry_order_score_not_observed:" + code)
        values[code] = float(row.score)
    return order_new_entries(retained_holdings, candidates, values)
