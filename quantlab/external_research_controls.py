"""Fail-closed controls for externally inspired research mechanisms.

This module deliberately does not import an account engine, router, position
policy, or live-data client.  It freezes the research-side boundaries that must
be satisfied before a separately authorised account experiment can exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.data.contracts import timestamp
from quantlab.research_universe import (
    FORWARD_CONTRACT_VERSION,
    U0,
    U1,
    LiquidityPoolSpec,
    UniverseSnapshot,
    forward_membership_sha256,
    is_exact_decision_cutoff,
)
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.theme_carrier import (
    THEME_COMPONENTS,
    EntryOrdering,
    ThemeCarrierResult,
)


class ExternalResearchContractError(ValueError):
    """An external-mechanism proposal failed a predeclared research boundary."""


_D2_DECISION_SCHEMA = "quantlab.d2.decision.v1"
_D2_ARCHIVE_SCHEMA = "quantlab.d2.full-matrix-archive.v1"
_D2_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "status",
        "protocol_sha256",
        "selection_basis",
        "archive_sha256",
        "archive_schema_version",
        "archive_identity_sha256",
    }
)
_D2_ARCHIVE_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "protocol_sha256",
        "selection_basis",
        "full_matrix_sha256",
        "archive_identity_sha256",
    }
)
_THEME_CARRIER_ENTRY_MECHANISM = "theme_carrier_minimal_v1"


def _sha256(value: object, name: str) -> str:
    value = str(value or '')
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
        raise ExternalResearchContractError(f'{name}_sha256_required')
    return value.lower()


def _canonical_payload_sha256(value: object, name: str) -> str:
    """Hash a portable manifest or action log without repr/order ambiguity."""
    try:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExternalResearchContractError(f"{name}_canonical_payload_required") from exc
    return sha256(payload).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalResearchContractError(f"{name}_required")
    return value.strip()


def _fixed_decision_at(value: object, error: str) -> pd.Timestamp:
    """Accept only the shared close-of-decision timestamp.

    Text and agent evidence is intentionally held to the same 18:00 Shanghai
    boundary as the forward market signal. A merely parseable earlier clock
    time would otherwise create a second, caller-selected decision regime.
    """
    try:
        decision = timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ExternalResearchContractError(error) from exc
    if not is_exact_decision_cutoff(decision, decision):
        raise ExternalResearchContractError(error)
    return decision


def _strict_json_object(raw: bytes, name: str) -> dict[str, object]:
    """Parse a signed JSON artifact without accepting duplicate or NaN keys."""

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExternalResearchContractError(f"{name}_json_required") from exc
    if not isinstance(decoded, dict):
        raise ExternalResearchContractError(f"{name}_object_required")
    return decoded


def _d2_archive_identity(artifact: Mapping[str, object]) -> str:
    if set(artifact) != _D2_ARCHIVE_FIELDS:
        raise ExternalResearchContractError("d2_archive_artifact_schema_invalid")
    if artifact.get("schema_version") != _D2_ARCHIVE_SCHEMA:
        raise ExternalResearchContractError("d2_archive_schema_invalid")
    _text(artifact.get("run_id"), "d2_archive_run_id")
    _sha256(artifact.get("protocol_sha256"), "d2_archive_protocol")
    if artifact.get("selection_basis") != "pre_registered_full_matrix":
        raise ExternalResearchContractError("d2_archive_selection_basis_invalid")
    _sha256(artifact.get("full_matrix_sha256"), "d2_full_matrix")
    identity_payload = {
        key: artifact[key]
        for key in sorted(_D2_ARCHIVE_FIELDS - {"archive_identity_sha256"})
    }
    expected = _canonical_payload_sha256(identity_payload, "d2_archive_identity")
    if _sha256(artifact.get("archive_identity_sha256"), "d2_archive_identity") != expected:
        raise ExternalResearchContractError("d2_archive_identity_hash_mismatch")
    return expected


@dataclass(frozen=True)
class D2ArchiveGate:
    """Evidence that D2 was archived before a *registration* can be recorded.

    It is intentionally not an account-start capability.  A later, separately
    authorised account adapter must establish its own execution authority.
    """

    run_id: str
    status: str
    protocol_sha256: str
    decision_sha256: str
    decision_manifest: Mapping[str, object]
    archive_path: str

    def require_registration_evidence(self) -> dict[str, str | bool]:
        if self.status != 'archived_decided':
            raise ExternalResearchContractError('d2_not_archived')
        run_id = _text(self.run_id, 'd2_run_id')
        protocol_sha = _sha256(self.protocol_sha256, 'd2_protocol')
        decision_sha = _sha256(self.decision_sha256, 'd2_decision')
        if not isinstance(self.decision_manifest, Mapping):
            raise ExternalResearchContractError('d2_decision_manifest_required')
        manifest = dict(self.decision_manifest)
        if set(manifest) != _D2_DECISION_FIELDS:
            raise ExternalResearchContractError('d2_decision_manifest_schema_invalid')
        if manifest.get('schema_version') != _D2_DECISION_SCHEMA:
            raise ExternalResearchContractError('d2_decision_schema_invalid')
        if _canonical_payload_sha256(manifest, 'd2_decision_manifest') != decision_sha:
            raise ExternalResearchContractError('d2_decision_manifest_hash_mismatch')
        if (
            manifest.get('run_id') != run_id
            or manifest.get('status') != 'archived_decided'
            or manifest.get('protocol_sha256') != protocol_sha
        ):
            raise ExternalResearchContractError('d2_decision_manifest_identity_mismatch')
        if manifest.get('selection_basis') != 'pre_registered_full_matrix':
            raise ExternalResearchContractError('d2_selection_basis_invalid')
        archive_sha = _sha256(manifest.get('archive_sha256'), 'd2_archive')
        path = Path(_text(self.archive_path, 'd2_archive_path'))
        if path.is_symlink() or not path.is_file():
            raise ExternalResearchContractError('d2_archive_artifact_path_invalid')
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ExternalResearchContractError('d2_archive_artifact_unreadable') from exc
        if sha256(raw).hexdigest() != archive_sha:
            raise ExternalResearchContractError('d2_archive_artifact_hash_mismatch')
        artifact = _strict_json_object(raw, 'd2_archive_artifact')
        archive_identity = _d2_archive_identity(artifact)
        if (
            artifact.get('run_id') != run_id
            or artifact.get('protocol_sha256') != protocol_sha
            or artifact.get('selection_basis') != manifest['selection_basis']
            or manifest.get('archive_schema_version') != artifact.get('schema_version')
            or _sha256(manifest.get('archive_identity_sha256'), 'd2_manifest_archive_identity')
            != archive_identity
        ):
            raise ExternalResearchContractError('d2_archive_artifact_identity_mismatch')
        return {
            'd2_archived': True,
            'd2_run_id': run_id,
            'd2_protocol_sha256': protocol_sha,
            'd2_decision_sha256': decision_sha,
            'd2_archive_sha256': archive_sha,
            'd2_archive_identity_sha256': archive_identity,
            'account_execution_authorized': False,
        }

    def require_account_permission(self) -> None:
        """Compatibility shim that deliberately refuses to become execution authority."""
        self.require_registration_evidence()
        raise ExternalResearchContractError('d2_gate_registration_only')


@dataclass(frozen=True)
class FixedSurface:
    """All non-ranking dimensions that an atomic entry-order A/B must hold fixed."""

    pool_contract: str
    fees_contract: str
    holdings_contract: str
    risk_contract: str
    retention_contract: str
    cadence_contract: str
    execution_contract: str
    immutable_identities: Mapping[str, str]

    _REQUIRED_IDENTITIES = frozenset({
        'universe_definition', 'membership', 'candidate_set', 'retained_holdings',
        'fees', 'holdings', 'risk', 'retention', 'cadence', 'execution', 'code', 'data',
    })

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == 'immutable_identities':
                continue
            if not isinstance(value, str) or not value:
                raise ExternalResearchContractError(f'fixed_surface_missing:{name}')
        identities = self.immutable_identities
        if not isinstance(identities, Mapping) or set(identities) != self._REQUIRED_IDENTITIES:
            raise ExternalResearchContractError('fixed_surface_identity_set_invalid')
        for name, value in identities.items():
            _sha256(value, f'fixed_surface_{name}')


def _strict_instrument_sequence(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ExternalResearchContractError(f'{name}_must_be_sequence')
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ExternalResearchContractError(f'{name}_must_be_sequence') from exc
    if any(not isinstance(code, str) or not code or code != code.strip() for code in result):
        raise ExternalResearchContractError(f'{name}_contains_invalid_instrument')
    if len(set(result)) != len(result):
        raise ExternalResearchContractError(f'{name}_contains_duplicate_instrument')
    return result


def entry_only_instrument_set_sha256(instruments: Sequence[str]) -> str:
    """Canonical identity of a fixed entry/retention set, independent of order."""
    values = _strict_instrument_sequence(instruments, 'entry_only_instruments')
    return _canonical_payload_sha256(tuple(sorted(values)), 'entry_only_instrument_set')


_ENTRY_UNIVERSE_SPECS: Mapping[str, LiquidityPoolSpec] = {"U0": U0, "U1": U1}


def _entry_universe_spec(pool_id: object) -> LiquidityPoolSpec:
    spec = _ENTRY_UNIVERSE_SPECS.get(str(pool_id))
    if spec is None:
        raise ExternalResearchContractError("theme_carrier_entry_pool_must_be_u0_or_u1")
    return spec


def entry_universe_contract_sha256(pool_id: str) -> str:
    """Identity for one frozen atomic-entry pool, never a relabelled broad pool."""
    spec = _entry_universe_spec(pool_id)
    return _canonical_payload_sha256(
        {
            'contract_version': FORWARD_CONTRACT_VERSION,
            'pool_id': spec.pool_id,
            'rank_lower_exclusive': spec.rank_lower_exclusive,
            'rank_upper_inclusive': spec.rank_upper_inclusive,
            'adv_window_sessions': spec.adv_window_sessions,
            'min_adv_cny': spec.min_adv_cny,
            'required_prior_price_sessions': spec.required_prior_price_sessions,
            'cadence_sessions': spec.cadence_sessions,
            'rank_tie_break': spec.rank_tie_break,
            'rank_before_adv_floor': spec.rank_before_adv_floor,
        },
        'theme_carrier_entry_universe_contract',
    )


def u0_entry_universe_contract_sha256() -> str:
    """Compatibility helper for the primary U0 atomic arm."""
    return entry_universe_contract_sha256("U0")


def u1_entry_universe_contract_sha256() -> str:
    """Identity for the independent U1 replication arm."""
    return entry_universe_contract_sha256("U1")


@dataclass(frozen=True, init=False)
class ThemeCarrierEntryArtifact:
    """Concrete output of the frozen theme-carrier entry-only mechanism.

    ``u0=`` remains an accepted construction keyword for existing U0 callers.
    New code should use ``universe=``: the same frozen mechanism can be
    evaluated in U0 and, separately, in U1 as a replication arm.  The artifact
    itself never decides whether U1 may replace U0; that policy lives in the
    A/B validator below.
    """

    universe: UniverseSnapshot
    result: ThemeCarrierResult
    ordering: EntryOrdering
    mechanism_id: str

    def __init__(
        self,
        u0: UniverseSnapshot | None = None,
        result: ThemeCarrierResult | None = None,
        ordering: EntryOrdering | None = None,
        mechanism_id: str = _THEME_CARRIER_ENTRY_MECHANISM,
        *,
        universe: UniverseSnapshot | None = None,
    ) -> None:
        # ``u0`` is deliberately only a compatibility spelling.  Accepting
        # both values could let a caller silently bind an artifact to a
        # different universe, so require exactly one.
        if (universe is None) == (u0 is None):
            raise ExternalResearchContractError(
                "theme_carrier_artifact_requires_exactly_one_universe"
            )
        if result is None or ordering is None:
            raise ExternalResearchContractError('theme_carrier_artifact_result_and_ordering_required')
        object.__setattr__(self, "universe", universe if universe is not None else u0)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "ordering", ordering)
        object.__setattr__(self, "mechanism_id", mechanism_id)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.mechanism_id != _THEME_CARRIER_ENTRY_MECHANISM:
            raise ExternalResearchContractError('theme_carrier_mechanism_must_be_minimal_v1')

    @property
    def u0(self) -> UniverseSnapshot:
        """Legacy alias; callers must validate the pool before treating it as U0."""
        return self.universe

    @property
    def candidate_set_sha256(self) -> str:
        return entry_only_instrument_set_sha256(self.ordering.baseline_candidate_order)

    @property
    def retained_holdings_sha256(self) -> str:
        return entry_only_instrument_set_sha256(self.ordering.retained_holdings)


def _timestamp_equal(left: object, right: object) -> bool:
    try:
        return timestamp(left) == timestamp(right)
    except (TypeError, ValueError):
        return False


def _validate_theme_carrier_entry_artifact(
    artifact: object, *, surface: FixedSurface, expected_pool_id: str | None = None
) -> dict[str, str]:
    """Verify the actual theme result and entry ordering, not a mechanism label.

    U0 is the primary arm and U1 is only a separately registered replication
    arm, but both must carry a concrete, hash-bound output of the *same*
    frozen four-component mechanism.  This validator verifies an artifact for
    the surface's actual pool instead of relabelling an U0 result as U1.
    """
    if not isinstance(artifact, ThemeCarrierEntryArtifact):
        raise ExternalResearchContractError('theme_carrier_entry_artifact_required')
    pool_id = expected_pool_id or surface.pool_contract
    spec = _entry_universe_spec(pool_id)
    universe, result, ordering = artifact.universe, artifact.result, artifact.ordering
    if not isinstance(universe, UniverseSnapshot):
        raise ExternalResearchContractError('theme_carrier_universe_snapshot_required')
    if (
        universe.pool_id != pool_id
        or universe.contract_version != FORWARD_CONTRACT_VERSION
        or universe.status != 'ok'
        or universe.abstain_reason is not None
        or not is_exact_decision_cutoff(universe.session, universe.decision_at)
    ):
        raise ExternalResearchContractError('theme_carrier_verified_' + str(pool_id).lower() + '_required')
    try:
        expected_membership = forward_membership_sha256(
            spec=spec,
            session=universe.session,
            decision_at=universe.decision_at,
            cadence_anchor=universe.cadence_anchor,
            input_hash=universe.input_hash,
            members=universe.members,
            status=universe.status,
            abstain_reason=universe.abstain_reason,
        )
    except (TypeError, ValueError) as exc:
        raise ExternalResearchContractError('theme_carrier_universe_identity_invalid') from exc
    if not isinstance(universe.membership_hash, str) or universe.membership_hash.lower() != expected_membership:
        raise ExternalResearchContractError('theme_carrier_universe_membership_hash_mismatch')
    members = _strict_instrument_sequence(universe.members, 'theme_carrier_universe_members')
    if not members:
        raise ExternalResearchContractError('theme_carrier_universe_members_empty')
    if not isinstance(result, ThemeCarrierResult):
        raise ExternalResearchContractError('theme_carrier_result_required')
    if (
        result.status != 'ok'
        or result.abstain_reason is not None
        or str(result.stock_scores.name) != 'theme_score_equal_weight_v1'
        or str(result.coverage.get('primary_endpoint')) != 'theme_score_equal_weight_v1'
        or result.coverage.get('component_results_diagnostic_only') is not True
        or not _timestamp_equal(result.decision_at, universe.decision_at)
        or str(result.decision_session) != str(universe.session)
    ):
        raise ExternalResearchContractError('theme_carrier_result_not_frozen_primary_endpoint')
    if not isinstance(result.coverage.get('ledger_id'), str) or not result.coverage['ledger_id'].strip():
        raise ExternalResearchContractError('theme_carrier_ledger_identity_missing')
    _sha256(result.coverage.get('ledger_journal_hash'), 'theme_carrier_ledger_journal')
    if not result.receipt_hashes or any(
        not isinstance(value, str) or _sha256(value, 'theme_carrier_receipt') != value.lower()
        for value in result.receipt_hashes
    ):
        raise ExternalResearchContractError('theme_carrier_receipt_identity_invalid')
    components = result.theme_components
    required_columns = {
        'theme_id',
        *THEME_COMPONENTS,
        *(component + '_percentile' for component in THEME_COMPONENTS),
        'theme_score_equal_weight_v1',
        'diagnostic_only_components',
    }
    if not isinstance(components, pd.DataFrame) or components.empty or not required_columns.issubset(components.columns):
        raise ExternalResearchContractError('theme_carrier_component_artifact_invalid')
    if components.theme_id.isna().any() or components.theme_id.astype(str).eq('').any() or components.theme_id.duplicated().any():
        raise ExternalResearchContractError('theme_carrier_component_artifact_invalid')
    if not components.diagnostic_only_components.eq(True).all():
        raise ExternalResearchContractError('theme_carrier_component_not_diagnostic_only')
    percentile_columns = [component + '_percentile' for component in THEME_COMPONENTS]
    numeric_columns = [*THEME_COMPONENTS, *percentile_columns, 'theme_score_equal_weight_v1']
    numeric = components.loc[:, numeric_columns].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or not (
        numeric.loc[:, percentile_columns].ge(0.0).all().all()
        and numeric.loc[:, percentile_columns].le(1.0).all().all()
    ):
        raise ExternalResearchContractError('theme_carrier_component_artifact_invalid')
    expected_equal_weight = numeric.loc[:, percentile_columns].mean(axis=1)
    if not np.allclose(
        numeric['theme_score_equal_weight_v1'].to_numpy(dtype=float),
        expected_equal_weight.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ExternalResearchContractError('theme_carrier_primary_endpoint_not_equal_weight')
    if not isinstance(result.stock_scores, pd.Series):
        raise ExternalResearchContractError('theme_carrier_stock_scores_required')
    scores = result.stock_scores.copy()
    score_codes = _strict_instrument_sequence(scores.index.tolist(), 'theme_carrier_score_instruments')
    if set(score_codes) != set(members):
        raise ExternalResearchContractError('theme_carrier_score_coverage_not_universe_membership')
    try:
        score_values = scores.astype(float)
    except (TypeError, ValueError) as exc:
        raise ExternalResearchContractError('theme_carrier_stock_scores_nonfinite') from exc
    if not np.isfinite(score_values.to_numpy()).all():
        raise ExternalResearchContractError('theme_carrier_stock_scores_nonfinite')
    if not isinstance(ordering, EntryOrdering):
        raise ExternalResearchContractError('theme_carrier_entry_ordering_required')
    baseline = _strict_instrument_sequence(ordering.baseline_candidate_order, 'theme_carrier_baseline_candidates')
    retained = _strict_instrument_sequence(ordering.retained_holdings, 'theme_carrier_retained_holdings')
    ordered = _strict_instrument_sequence(ordering.new_entry_order, 'theme_carrier_new_entry_order')
    if not ordering.applied or ordering.abstain_reason is not None:
        raise ExternalResearchContractError('theme_carrier_entry_ordering_not_applied')
    if not set(baseline).issubset(members) or not set(retained).issubset(members):
        raise ExternalResearchContractError('theme_carrier_entry_ordering_outside_universe')
    new_entries = tuple(code for code in baseline if code not in set(retained))
    if set(ordered) != set(new_entries) or len(ordered) != len(new_entries):
        raise ExternalResearchContractError('theme_carrier_entry_ordering_membership_changed')
    baseline_rank = {code: index for index, code in enumerate(baseline)}
    expected_order = tuple(
        sorted(new_entries, key=lambda code: (-float(score_values.loc[code]), baseline_rank[code], code))
    )
    if ordered != expected_order:
        raise ExternalResearchContractError('theme_carrier_entry_ordering_not_score_derived')
    expected_surface = {
        'universe_definition': entry_universe_contract_sha256(pool_id),
        'membership': expected_membership,
        'candidate_set': artifact.candidate_set_sha256,
        'retained_holdings': artifact.retained_holdings_sha256,
    }
    for name, expected in expected_surface.items():
        if surface.immutable_identities.get(name) != expected:
            raise ExternalResearchContractError('theme_carrier_surface_identity_mismatch:' + name)
    identity = _canonical_payload_sha256(
        {
            'mechanism_id': artifact.mechanism_id,
            'pool_id': pool_id,
            'universe_membership_sha256': expected_membership,
            'universe_contract_sha256': entry_universe_contract_sha256(pool_id),
            'component_rows': components.sort_values('theme_id', kind='stable').to_dict(orient='records'),
            'stock_scores': sorted((str(code), float(score_values.loc[code])) for code in score_codes),
            'coverage': dict(result.coverage),
            'receipt_hashes': tuple(sorted(result.receipt_hashes)),
            'baseline_candidate_order': baseline,
            'retained_holdings': retained,
            'new_entry_order': ordered,
        },
        'theme_carrier_entry_artifact',
    )
    result_identity = {
        'theme_carrier_entry_artifact_sha256': identity,
        'universe_pool_id': pool_id,
        'universe_membership_sha256': expected_membership,
        'universe_contract_sha256': entry_universe_contract_sha256(pool_id),
        'candidate_set_sha256': artifact.candidate_set_sha256,
        'retained_holdings_sha256': artifact.retained_holdings_sha256,
    }
    # Preserve the primary-arm names for existing result consumers while the
    # generic names above keep U1 from being represented as an U0 artifact.
    if pool_id == 'U0':
        result_identity.update(
            u0_membership_sha256=expected_membership,
            u0_contract_sha256=entry_universe_contract_sha256('U0'),
        )
    else:
        result_identity.update(
            u1_membership_sha256=expected_membership,
            u1_contract_sha256=entry_universe_contract_sha256('U1'),
        )
    return result_identity


def validate_entry_sort_ab(baseline: Mapping, treatment: Mapping, *, allow_u1_replication: bool = False) -> dict:
    """Allow exactly the verified theme-carrier entry ordering to change in an atomic A/B.

    U1 is deliberately a separately labelled replication arm.  It can never
    silently replace U0 because it looked better on already-observed history.
    """
    base_surface = baseline.get('surface')
    trial_surface = treatment.get('surface')
    if not isinstance(base_surface, FixedSurface) or not isinstance(trial_surface, FixedSurface):
        raise ExternalResearchContractError('ab_fixed_surface_required')
    base_surface.validate(); trial_surface.validate()
    if 'entry_sorter' in baseline or 'entry_sorter' in treatment:
        raise ExternalResearchContractError('ab_entry_sorter_label_forbidden')
    arm = trial_surface.pool_contract
    if arm not in ('U0', 'U1'):
        raise ExternalResearchContractError('first_atomic_arm_must_be_u0_or_u1')
    if base_surface.pool_contract != 'U0':
        raise ExternalResearchContractError('ab_baseline_must_be_u0')
    allowed_pool_change = arm == 'U1' and allow_u1_replication
    for name, expected in asdict(base_surface).items():
        if name == 'immutable_identities':
            continue
        actual = getattr(trial_surface, name)
        if name == 'pool_contract' and allowed_pool_change:
            continue
        if actual != expected:
            raise ExternalResearchContractError(f'ab_surface_changed:{name}')
    base_ids, trial_ids = dict(base_surface.immutable_identities), dict(trial_surface.immutable_identities)
    if arm == 'U0' and trial_ids != base_ids:
        raise ExternalResearchContractError('ab_surface_changed:immutable_identities')
    if arm == 'U1':
        if treatment.get('replication_of') != 'U0':
            raise ExternalResearchContractError('u1_replication_of_u0_required')
        if treatment.get('replace_u0') is not False:
            raise ExternalResearchContractError('u1_cannot_replace_u0')
        # Policy identities (fees/risk/cadence/retention rule) remain fixed.
        # U1 has a different actual candidate set and may have a different
        # current retained set because its members are disjoint from U0; the
        # latter is therefore optional rather than a hidden impossible
        # invariant for a valid independent arm.
        required_u1_identity_changes = {'universe_definition', 'membership', 'candidate_set'}
        allowed_u1_identity_changes = required_u1_identity_changes | {'retained_holdings'}
        changed_identities = {name for name in base_ids if base_ids[name] != trial_ids[name]}
        if not required_u1_identity_changes.issubset(changed_identities) or not changed_identities.issubset(
            allowed_u1_identity_changes
        ):
            raise ExternalResearchContractError('u1_replication_identity_boundary_invalid')
        _sha256(treatment.get('u1_replication_protocol_sha256'), 'u1_replication_protocol')
        u0_source_identity = _validate_theme_carrier_entry_artifact(
            treatment.get('u0_source_theme_carrier_entry_artifact'),
            surface=base_surface,
            expected_pool_id='U0',
        )
        artifact_identity = _validate_theme_carrier_entry_artifact(
            treatment.get('theme_carrier_entry_artifact'),
            surface=trial_surface,
            expected_pool_id='U1',
        )
        u0_source = treatment.get('u0_source_theme_carrier_entry_artifact')
        u1_artifact = treatment.get('theme_carrier_entry_artifact')
        assert isinstance(u0_source, ThemeCarrierEntryArtifact)
        assert isinstance(u1_artifact, ThemeCarrierEntryArtifact)
        if (
            u0_source.mechanism_id != u1_artifact.mechanism_id
            or u0_source.universe.session != u1_artifact.universe.session
            or not _timestamp_equal(u0_source.universe.decision_at, u1_artifact.universe.decision_at)
            or u0_source.universe.cadence_anchor != u1_artifact.universe.cadence_anchor
        ):
            raise ExternalResearchContractError('u1_replication_mechanism_or_decision_alignment_invalid')
        artifact_identity.update(
            u0_source_theme_carrier_entry_artifact_sha256=u0_source_identity[
                'theme_carrier_entry_artifact_sha256'
            ],
            u0_source_membership_sha256=u0_source_identity['u0_membership_sha256'],
            u0_source_contract_sha256=u0_source_identity['u0_contract_sha256'],
        )
    elif treatment.get('replace_u0'):
        raise ExternalResearchContractError('u0_replacement_not_a_first_ab_decision')
    else:
        artifact_identity = _validate_theme_carrier_entry_artifact(
            treatment.get('theme_carrier_entry_artifact'), surface=base_surface
        )
    return dict(
        arm=arm,
        mechanism_id=_THEME_CARRIER_ENTRY_MECHANISM,
        changed_fields=('theme_carrier_minimal_v1_entry_order',),
        **artifact_identity,
        account_start_allowed=False,
        account_execution_authorized=False,
        note='This validates only an atomic research A/B registration; it does not approve an account run or adoption.',
    )


def register_atomic_entry_sort_experiment(
    *,
    experiment_id: str,
    preregistration_sha256: str,
    baseline: Mapping,
    treatment: Mapping,
    d2_gate: D2ArchiveGate,
    allow_u1_replication: bool = False,
) -> dict:
    """Create a hash-bound research registration, never an execution permit.

    Future account adapters must call this first instead of treating a surface
    comparison as proof that D2 was archived.  It records D2's immutable
    decision identity next to the separate experiment preregistration while
    deliberately returning no authority to trade or promote a strategy.
    """
    if not isinstance(d2_gate, D2ArchiveGate):
        raise ExternalResearchContractError('d2_archive_gate_required')
    _text(experiment_id, 'atomic_experiment_id')
    _sha256(preregistration_sha256, 'atomic_preregistration')
    d2_evidence = d2_gate.require_registration_evidence()
    comparison = validate_entry_sort_ab(
        baseline, treatment, allow_u1_replication=allow_u1_replication
    )
    manifest = dict(d2_gate.decision_manifest)
    return dict(
        experiment_id=experiment_id,
        preregistration_sha256=preregistration_sha256.lower(),
        comparison=comparison,
        d2_run_id=d2_gate.run_id,
        d2_decision_sha256=d2_gate.decision_sha256,
        d2_archive_sha256=manifest['archive_sha256'],
        d2_archive_path=d2_gate.archive_path,
        d2_selection_basis=manifest['selection_basis'],
        d2_archive_identity_sha256=d2_evidence['d2_archive_identity_sha256'],
        registration_valid=True,
        account_execution_authorized=False,
        promotion_approved=False,
    )


def _verified_ledger_receipt(
    ledger: object, receipt_id: object, decision_at: object
) -> dict[str, str]:
    """Resolve one canonical, timely receipt from the append-only ledger.

    A dict with a receipt-shaped ``status`` and ``raw_sha256`` is not evidence:
    it has to be re-derived from :meth:`ResearchSignalLedger.export_receipt_evidence`
    after the ledger's blob and hash-chain verification succeeds.
    """
    if not isinstance(ledger, ResearchSignalLedger):
        raise ExternalResearchContractError('ledger_evidence_required')
    decision = _fixed_decision_at(
        decision_at, 'ledger_evidence_decision_cutoff_required'
    )
    verification = ledger.verify()
    journal_hash = verification.get('journal_hash')
    if (
        not verification.get('verified')
        or not isinstance(journal_hash, str)
        or _sha256(journal_hash, 'ledger_journal') != journal_hash.lower()
    ):
        raise ExternalResearchContractError('ledger_evidence_integrity_invalid')
    try:
        requested_id = _text(receipt_id, 'ledger_receipt_id')
        evidence = ledger.export_receipt_evidence(requested_id)
        receipt = evidence.get('receipt')
        raw = evidence.get('raw')
        if not isinstance(receipt, Mapping) or not isinstance(raw, Mapping):
            raise ValueError('receipt_evidence_shape_invalid')
        canonical_id = _text(evidence.get('canonical_receipt_id'), 'canonical_receipt_id')
        if (
            evidence.get('kind') != 'research_signal_receipt_evidence'
            or evidence.get('ledger_id') != ledger.ledger_id
            or evidence.get('requested_receipt_id') != requested_id
            or canonical_id != requested_id
            or receipt.get('event_id') != canonical_id
            or receipt.get('kind') != 'receipt'
            or receipt.get('status') != 'accepted'
        ):
            raise ValueError('receipt_evidence_not_canonical_accepted')
        body = receipt.get('body')
        if not isinstance(body, Mapping):
            raise ValueError('receipt_body_invalid')
        raw_sha256 = _sha256(raw.get('sha256'), 'ledger_receipt_raw')
        if body.get('raw_sha256') != raw_sha256:
            raise ValueError('receipt_raw_identity_mismatch')
        source_family = _text(body.get('source_family'), 'ledger_source_family')
        origin_id = _text(body.get('origin_id'), 'ledger_origin_id')
        for name, value in (
            ('ledger_receipt_published_at', body.get('published_at')),
            ('ledger_receipt_first_captured_at', body.get('first_captured_at')),
            ('ledger_receipt_available_at', receipt.get('available_at')),
            ('ledger_receipt_recorded_at', receipt.get('recorded_at')),
        ):
            observed = timestamp(value)
            if pd.isna(observed) or observed > decision:
                raise ValueError(name + '_not_available_by_decision')
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalResearchContractError('ledger_receipt_evidence_unverified') from exc
    return dict(
        receipt_id=requested_id,
        source_family=source_family,
        origin_id=origin_id,
        raw_sha256=raw_sha256,
        ledger_id=ledger.ledger_id,
        ledger_journal_hash=journal_hash.lower(),
    )


class TextConsensus:
    """Two independent, non-reprint ledger receipts are minimum text consensus.

    ``sources`` remains a sequence of small references for compatibility, but
    it no longer supplies source identity or timing.  Only ``receipt_id`` is
    required; a caller may mark a known reprint to exclude it, but cannot use a
    hand-shaped ``is_reprint=False`` value to establish independence.  The
    ledger supplies all factual provenance.  Existing map-only callers fail
    closed as an abstention rather than receiving an accidental confirmation.
    """

    @staticmethod
    def confirm(
        sources: Iterable[Mapping], decision_at: object, *, ledger: ResearchSignalLedger | None = None
    ) -> dict:
        try:
            decision = _fixed_decision_at(decision_at, 'text_decision_cutoff_required')
        except ExternalResearchContractError:
            return dict(status='abstain', reason='text_decision_cutoff_required', independent_sources=0)
        if not isinstance(ledger, ResearchSignalLedger):
            return dict(status='abstain', reason='text_ledger_evidence_required', independent_sources=0)
        try:
            references = list(sources)
        except TypeError:
            return dict(status='abstain', reason='text_source_identity_missing', independent_sources=0)
        accepted: list[dict[str, str]] = []
        for source in references:
            if not isinstance(source, Mapping) or not source.get('receipt_id'):
                return dict(status='abstain', reason='text_source_identity_missing', independent_sources=0)
            reprint_marker = source.get('is_reprint', False)
            # This marker never establishes independence, but if supplied it
            # must still be an unambiguous JSON boolean.  Treating strings
            # such as ``"true"`` as false would let a known reprint slip
            # through a caller's malformed serialization.
            if not isinstance(reprint_marker, bool):
                return dict(status='abstain', reason='text_reprint_status_invalid', independent_sources=0)
            # The ledger's independent origin and source-family fields are
            # authoritative.  A caller can only make the result more
            # conservative by marking a known reprint; it cannot manufacture
            # a second independent origin by declaring ``False``.
            if reprint_marker:
                continue
            try:
                accepted.append(_verified_ledger_receipt(ledger, source['receipt_id'], decision))
            except ExternalResearchContractError:
                return dict(status='abstain', reason='text_receipt_evidence_unverified', independent_sources=0)
        families = {x['source_family'] for x in accepted}
        origins = {x['origin_id'] for x in accepted}
        # ``origin_id`` is the ledger's original-source identity, so a known
        # reprint shares it with its primary item.  Raw identity is a second,
        # ledger-derived guard: callers must not be able to register one
        # unchanged payload under two hand-shaped source maps and call that
        # pair independent corroboration.
        raw_contents = {x['raw_sha256'] for x in accepted}
        if (
            len(accepted) < 2
            or len(families) < 2
            or len(origins) < 2
            or len(raw_contents) < 2
        ):
            return dict(status='abstain', reason='two_independent_non_reprint_sources_required',
                        independent_sources=min(len(families), len(origins)))
        return dict(
            status='confirmed',
            independent_sources=min(len(families), len(origins)),
            receipt_ids=tuple(sorted(x['receipt_id'] for x in accepted)),
            ledger_id=ledger.ledger_id,
            ledger_journal_hash=ledger.verify()['journal_hash'],
        )


_FORBIDDEN_AGENT_FIELDS = frozenset({
    'rating', 'recommendation', 'trade', 'trade_action', 'buy', 'sell', 'order',
    'weight', 'target_weight', 'position', 'position_score', 'fusion_score',
    'orders', 'entry_order', 'entry_ordering', 'new_entry_order', 'ranking',
    'rank', 'action', 'next_action', 'signal', 'allocation',
})

# Settled results may be named only by the declared top-level
# ``outcome_feedback_ids`` field.  Keeping aliases and nested copies out of a
# research brief prevents a model response from smuggling a pending label into
# the next prompt without ledger settlement and time checks.
_UNREGISTERED_AGENT_OUTCOME_FIELDS = frozenset({
    'outcome', 'outcomes', 'outcome_id', 'outcome_ids', 'outcome_status',
    'outcome_feedback', 'outcome_feedback_id', 'outcome_feedback_ids',
    'feedback_outcome', 'feedback_outcomes', 'settlement', 'settlement_id',
    'settlement_status', 'settled_outcome', 'settled_outcomes',
})


def _nested_forbidden_agent_fields(value: object) -> set[str]:
    """Disallow an order/rating payload at every structured brief level."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace('.', '_').replace('-', '_')
            if normalized in _FORBIDDEN_AGENT_FIELDS:
                found.add(normalized)
            found.update(_nested_forbidden_agent_fields(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_nested_forbidden_agent_fields(item))
    return found


def _nested_unregistered_outcome_feedback_fields(
    value: object, *, top_level: bool = False
) -> set[str]:
    """Find outcome-like fields that bypass the settled-ledger channel.

    ``outcome_feedback_ids`` is intentionally allowed only on the brief
    envelope, where :func:`validate_multi_agent_brief` re-resolves every id
    against ``ResearchSignalLedger.settled_outcomes``.  The same name inside a
    fact/catalyst/disagreement object would otherwise avoid that check.
    """
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace('.', '_').replace('-', '_')
            if normalized in _UNREGISTERED_AGENT_OUTCOME_FIELDS and not (
                top_level and normalized == 'outcome_feedback_ids'
            ):
                found.add(normalized)
            found.update(_nested_unregistered_outcome_feedback_fields(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_nested_unregistered_outcome_feedback_fields(item))
    return found


def validate_multi_agent_brief(
    brief: Mapping,
    fixed_pool_members: set[str],
    outcome_status: Mapping[str, str],
    *,
    ledger: ResearchSignalLedger | None = None,
    receipt_evidence: Mapping[str, Mapping[str, object]] | None = None,
) -> dict:
    """Restrict a multi-agent chain to research evidence, not a trading verdict.

    ``receipt_evidence`` is retained only as a backwards-compatible keyword;
    it is deliberately not trusted.  Citations are always re-resolved from an
    integrity-checked :class:`ResearchSignalLedger`, so a hand-shaped map can
    no longer turn an arbitrary string into an accepted citation.
    """
    if not isinstance(brief, Mapping):
        raise ExternalResearchContractError('agent_brief_mapping_required')
    forbidden = _nested_forbidden_agent_fields(brief)
    if forbidden:
        raise ExternalResearchContractError('agent_direct_rating_forbidden')
    unregistered_outcomes = _nested_unregistered_outcome_feedback_fields(brief, top_level=True)
    if unregistered_outcomes:
        raise ExternalResearchContractError('agent_unregistered_outcome_feedback_forbidden')
    instrument = brief.get('instrument_id')
    if instrument not in fixed_pool_members:
        raise ExternalResearchContractError('agent_instrument_outside_fixed_pool')
    decision_at = _fixed_decision_at(
        brief.get('decision_at'), 'agent_brief_decision_cutoff_required'
    )
    abstain_reason = brief.get('abstain_reason')
    if abstain_reason is not None:
        abstain_reason = _text(abstain_reason, 'agent_abstention_reason')
    # Model/prompt identity belongs to every brief, including an abstention:
    # otherwise a chain could hide a model or prompt drift precisely on dates
    # where it contributed no positive signal.
    _text(brief.get('model_id'), 'agent_model_id')
    _sha256(brief.get('prompt_sha256'), 'agent_prompt')
    if abstain_reason is None:
        for field in ('facts', 'citations', 'catalysts', 'invalidation_conditions', 'disagreements'):
            if not isinstance(brief.get(field), list):
                raise ExternalResearchContractError(f'agent_brief_missing:{field}')
        if not isinstance(ledger, ResearchSignalLedger):
            raise ExternalResearchContractError('agent_citation_ledger_required')
        citation_ids = tuple(str(value) for value in brief['citations'])
        if not citation_ids or len(set(citation_ids)) != len(citation_ids):
            raise ExternalResearchContractError('agent_citations_invalid')
        citation_evidence = []
        for receipt_id in citation_ids:
            try:
                citation_evidence.append(_verified_ledger_receipt(ledger, receipt_id, decision_at))
            except ExternalResearchContractError as exc:
                raise ExternalResearchContractError('agent_citation_receipt_unverified')
        # ``receipt_evidence`` is intentionally unused.  Leaving the keyword
        # in the signature keeps legacy callers from crashing with TypeError,
        # while requiring real ledger evidence keeps their old maps fail-closed.
    else:
        citation_evidence = []
    feedback = brief.get('outcome_feedback_ids', [])
    if isinstance(feedback, (str, bytes)):
        raise ExternalResearchContractError('agent_outcome_feedback_ids_invalid')
    try:
        feedback = tuple(feedback)
    except TypeError as exc:
        raise ExternalResearchContractError('agent_outcome_feedback_ids_invalid') from exc
    if (
        any(not isinstance(key, str) or not key for key in feedback)
        or len(set(feedback)) != len(feedback)
    ):
        raise ExternalResearchContractError('agent_outcome_feedback_ids_invalid')
    if feedback:
        if not isinstance(ledger, ResearchSignalLedger):
            raise ExternalResearchContractError('agent_outcome_ledger_required')
        try:
            settled_ids = {
                str(row['event_id'])
                for row in ledger.settled_outcomes(decision_at)
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalResearchContractError('unsettled_outcome_feedback') from exc
        # ``outcome_status`` is retained only for a compatible call surface;
        # it is not evidence. The feedback event itself must be settled and
        # available in the append-only ledger by this fixed decision time.
        if not set(feedback).issubset(settled_ids):
            raise ExternalResearchContractError('unsettled_outcome_feedback')
    result = dict(status='abstain' if abstain_reason else 'ok', instrument_id=instrument,
                  feedback_outcomes=tuple(feedback), direct_ordering_allowed=False)
    if citation_evidence:
        result.update(
            citation_receipt_ids=tuple(row['receipt_id'] for row in citation_evidence),
            citation_ledger_id=ledger.ledger_id,
            citation_ledger_journal_hash=ledger.verify()['journal_hash'],
        )
    return result


class MiroFishRunContract:
    """Summarise K=8 simulations from action logs, never unused persona parameters."""

    _FORBIDDEN = frozenset({'stance', 'sentiment_bias'})
    _COVERAGE_SCHEMA = 'quantlab.mirofish.ledger-coverage.v1'

    @classmethod
    def _has_forbidden_parameter(cls, value: object) -> bool:
        if isinstance(value, Mapping):
            return any(
                str(key).lower() in cls._FORBIDDEN or cls._has_forbidden_parameter(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(cls._has_forbidden_parameter(item) for item in value)
        return False

    @classmethod
    def factors_from_action_log(cls, *, rounds: int, actions: Iterable[Mapping]) -> dict:
        """Score one action log.  This does not launch a K=8 simulation."""
        logged = list(actions)
        if cls._has_forbidden_parameter({"actions": logged}):
            raise ExternalResearchContractError("mirofish_unused_parameter_forbidden")
        if not isinstance(rounds, int) or rounds < 1:
            raise ExternalResearchContractError("mirofish_action_log_invalid")
        actors = []
        for action in logged:
            if not isinstance(action, Mapping) or not action.get("actor_id") or not action.get("action"):
                raise ExternalResearchContractError("mirofish_action_log_invalid")
            actors.append(str(action["actor_id"]))
        if not actors:
            raise ExternalResearchContractError("mirofish_empty_action_log")
        counts = {actor: actors.count(actor) for actor in set(actors)}
        total = len(actors)
        return dict(
            propagation_range=len(counts),
            action_density=total / rounds,
            concentration=sum((count / total) ** 2 for count in counts.values()),
            simulation_runs=0,
            account_result=False,
        )

    @staticmethod
    def action_log_sha256(*, rounds: int, actions: Iterable[Mapping]) -> str:
        return _canonical_payload_sha256(
            dict(rounds=rounds, actions=list(actions)), 'mirofish_action_log'
        )

    @classmethod
    def ledger_coverage_evidence(
        cls,
        *,
        ledger: ResearchSignalLedger,
        scope_id: str,
        decision_at: object,
        feature_id: str,
    ) -> dict[str, object]:
        """Materialise one hash-bound simulation-coverage snapshot from the ledger.

        This is intentionally a small adapter rather than a second coverage
        calculator: the eligible numerator and denominator come directly from
        the append-only text/observation ledger and are tied to its current
        journal hash.  It does not launch a simulation or make a signal.
        """
        if not isinstance(ledger, ResearchSignalLedger):
            raise ExternalResearchContractError('mirofish_ledger_coverage_required')
        try:
            decision = timestamp(decision_at)
            scope = _text(scope_id, 'mirofish_scope_id')
            feature = _text(feature_id, 'mirofish_feature_id')
            verification = ledger.verify()
            journal_hash = verification.get('journal_hash')
            if (
                not verification.get('verified')
                or not isinstance(journal_hash, str)
                or _sha256(journal_hash, 'mirofish_ledger_journal') != journal_hash.lower()
            ):
                raise ValueError('ledger_integrity_invalid')
            observed = ledger.coverage(scope, decision, feature)
            expected = observed.get('expected')
            observed_count = observed.get('observed')
            abstained = observed.get('abstained')
            membership_hash = observed.get('membership_sha256')
            if (
                not isinstance(expected, int)
                or expected <= 0
                or not isinstance(observed_count, int)
                or not isinstance(abstained, int)
                or observed_count < 0
                or abstained < 0
                or observed_count + abstained > expected
            ):
                raise ValueError('coverage_counts_invalid')
            membership_hash = _sha256(membership_hash, 'mirofish_membership')
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalResearchContractError('mirofish_ledger_coverage_unverified') from exc
        identity = _canonical_payload_sha256(
            {
                'schema_version': cls._COVERAGE_SCHEMA,
                'ledger_id': ledger.ledger_id,
                'ledger_journal_hash': journal_hash.lower(),
                'scope_id': scope,
                'decision_at': decision.isoformat(),
                'feature_id': feature,
                'membership_sha256': membership_hash,
                'expected': expected,
                'observed': observed_count,
                'abstained': abstained,
                'missing': tuple(observed.get('missing', ())),
                'observed_entity_keys': tuple(observed.get('observed_entity_keys', ())),
                'abstained_entity_keys': tuple(observed.get('abstained_entity_keys', ())),
            },
            'mirofish_ledger_coverage',
        )
        return dict(
            schema_version=cls._COVERAGE_SCHEMA,
            scope_id=scope,
            decision_at=decision.isoformat(),
            feature_id=feature,
            ledger_id=ledger.ledger_id,
            ledger_journal_hash=journal_hash.lower(),
            membership_sha256=membership_hash,
            expected=expected,
            observed=observed_count,
            abstained=abstained,
            eligible_rate=observed_count / expected,
            denominator_hash=identity,
        )

    @classmethod
    def _require_ledger_coverage(
        cls, coverage: Mapping, *, ledger: ResearchSignalLedger | None
    ) -> dict[str, object]:
        if not isinstance(ledger, ResearchSignalLedger):
            raise ExternalResearchContractError('mirofish_ledger_coverage_required')
        if not isinstance(coverage, Mapping):
            raise ExternalResearchContractError('mirofish_coverage_evidence_required')
        required = {
            'schema_version', 'scope_id', 'decision_at', 'feature_id', 'ledger_id',
            'ledger_journal_hash', 'membership_sha256', 'expected', 'observed', 'abstained',
            'eligible_rate', 'denominator_hash',
        }
        if not required.issubset(coverage):
            raise ExternalResearchContractError('mirofish_coverage_evidence_required')
        expected = cls.ledger_coverage_evidence(
            ledger=ledger,
            scope_id=coverage['scope_id'],
            decision_at=coverage['decision_at'],
            feature_id=coverage['feature_id'],
        )
        for field in required - {'eligible_rate'}:
            if coverage.get(field) != expected[field]:
                raise ExternalResearchContractError('mirofish_ledger_coverage_mismatch:' + field)
        rate = coverage.get('eligible_rate')
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate != expected['eligible_rate']:
            raise ExternalResearchContractError('mirofish_ledger_coverage_mismatch:eligible_rate')
        return expected

    @staticmethod
    def _metric(run: Mapping) -> dict:
        if MiroFishRunContract._has_forbidden_parameter(run):
            raise ExternalResearchContractError('mirofish_unused_parameter_forbidden')
        _text(run.get('run_id'), 'mirofish_run_id')
        if not isinstance(run.get('seed'), int):
            raise ExternalResearchContractError('mirofish_seed_required')
        _text(run.get('model_id'), 'mirofish_model_id')
        _sha256(run.get('prompt_sha256'), 'mirofish_prompt')
        rounds = run.get('rounds')
        actions = run.get('actions')
        if not isinstance(rounds, int) or rounds < 1 or not isinstance(actions, list):
            raise ExternalResearchContractError('mirofish_action_log_invalid')
        actors=[]
        for action in actions:
            if not isinstance(action, Mapping) or MiroFishRunContract._has_forbidden_parameter(action):
                raise ExternalResearchContractError('mirofish_unused_parameter_forbidden')
            if not action.get('actor_id') or not action.get('action'):
                raise ExternalResearchContractError('mirofish_action_log_invalid')
            actors.append(str(action['actor_id']))
        if not actors:
            raise ExternalResearchContractError('mirofish_empty_action_log')
        expected_hash = MiroFishRunContract.action_log_sha256(rounds=rounds, actions=actions)
        if run.get('action_log_hash') != expected_hash:
            raise ExternalResearchContractError('mirofish_action_log_hash_mismatch')
        counts={actor: actors.count(actor) for actor in set(actors)}
        total=len(actors)
        return dict(propagation_range=len(counts), action_density=total / rounds,
                    concentration=sum((count / total) ** 2 for count in counts.values()))

    @staticmethod
    def summarize(
        *,
        coverage: Mapping,
        minimum_coverage: float,
        runs: Iterable[Mapping],
        ledger: ResearchSignalLedger | None = None,
    ) -> dict:
        if not isinstance(minimum_coverage, (float, int)) or not 0 < minimum_coverage <= 1:
            raise ExternalResearchContractError('mirofish_preregistered_coverage_required')
        coverage_evidence = MiroFishRunContract._require_ledger_coverage(coverage, ledger=ledger)
        rate=coverage_evidence['eligible_rate']
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < minimum_coverage:
            raise ExternalResearchContractError('mirofish_text_coverage_not_met')
        runs=list(runs)
        if len(runs) != 8:
            raise ExternalResearchContractError('mirofish_requires_k8')
        if len({str(run.get('run_id')) for run in runs}) != 8:
            raise ExternalResearchContractError('mirofish_run_id_not_unique')
        if len({run.get('seed') for run in runs}) != 8:
            raise ExternalResearchContractError('mirofish_seed_not_unique')
        metrics=[MiroFishRunContract._metric(run) for run in runs]
        if len({str(run.get('action_log_hash')) for run in runs}) != 8:
            raise ExternalResearchContractError('mirofish_action_log_not_unique')
        def iqr(name):
            values=np.array([row[name] for row in metrics], dtype=float)
            return float(np.quantile(values,.75)-np.quantile(values,.25))
        return dict(runs=8, metrics=metrics,
                    iqr={name:iqr(name) for name in ('propagation_range','action_density','concentration')},
                    coverage_evidence_sha256=coverage_evidence['denominator_hash'],
                    eligibility='research_only_not_a_trading_signal')


class MainlineMatrix:
    """Enforce the predeclared five-phase by two-cost reporting matrix."""

    PHASES=(0, 4, 8, 12, 16)
    COSTS=('base', 'double')

    @classmethod
    def evaluate(cls, cells: Iterable[Mapping]) -> dict:
        observed={}
        for cell in cells:
            key=(cell.get('phase'), cell.get('cost'))
            if key in observed:
                raise ExternalResearchContractError('mainline_matrix_duplicate_cell')
            observed[key]=cell
        required={(phase,cost) for phase in cls.PHASES for cost in cls.COSTS}
        if set(observed) != required:
            raise ExternalResearchContractError('mainline_matrix_incomplete')
        annualized=[]
        drawdown_losses=[]
        for key, cell in observed.items():
            annual=cell.get('after_fee_annualized'); drawdown=cell.get('max_drawdown')
            if not isinstance(annual,(int,float)) or not math.isfinite(annual):
                raise ExternalResearchContractError('mainline_annualized_invalid')
            if not isinstance(drawdown,(int,float)) or not math.isfinite(drawdown):
                raise ExternalResearchContractError('mainline_drawdown_invalid')
            drawdown_loss=abs(float(drawdown))
            if drawdown_loss > .25:
                raise ExternalResearchContractError('mainline_drawdown_exceeds_25pct')
            annualized.append(float(annual))
            drawdown_losses.append(drawdown_loss)
        phase_medians={phase: median([observed[(phase,cost)]['after_fee_annualized'] for cost in cls.COSTS])
                       for phase in cls.PHASES}
        best=max(observed.values(), key=lambda x:x['after_fee_annualized'])
        worst_phase=min(phase_medians, key=phase_medians.get)
        return dict(selection_metric='median_after_fee_annualized',
                    median_after_fee_annualized=float(median(annualized)),
                    worst_phase=worst_phase,
                    worst_phase_median_after_fee_annualized=float(phase_medians[worst_phase]),
                    highest_cell_disclosure=dict(phase=best['phase'],cost=best['cost'],
                                                 after_fee_annualized=float(best['after_fee_annualized'])),
                    maximum_drawdown=float(max(drawdown_losses)),
                    promotion_approved=False)


def validate_fusion_or_structure(kind: str, experts: Iterable[Mapping], *, signal_fixed: bool) -> dict:
    """Do not mix a new signal test with fusion or holding-structure tuning."""
    if kind not in ('fusion', 'structure'):
        raise ExternalResearchContractError('unknown_post_increment_experiment')
    experts=list(experts)
    if not experts or any(not row.get('pre_registered_net_increment') for row in experts):
        raise ExternalResearchContractError('single_expert_increment_missing')
    if not signal_fixed:
        raise ExternalResearchContractError('signal_not_fixed')
    if kind == 'fusion' and len(experts) < 2:
        raise ExternalResearchContractError('fusion_requires_two_independently_proven_experts')
    return dict(allowed=True, kind=kind, expert_ids=tuple(row.get('expert_id') for row in experts),
                promotion_approved=False)


def validate_research_disclosure(disclosure: Mapping) -> dict:
    """A result must preserve failures and identities, rather than only its best cell."""
    required=('search_family','all_outcomes','coverage','code_identity','data_identity','model_identity','decision_impact')
    for field in required:
        if field not in disclosure or disclosure[field] in (None, '', []):
            raise ExternalResearchContractError(f'disclosure_missing:{field}')
    if not isinstance(disclosure['all_outcomes'], list):
        raise ExternalResearchContractError('disclosure_outcomes_invalid')
    return dict(complete=True, promotion_approved=False,
                disclosure_hash=sha256(repr(sorted(disclosure.items(), key=lambda item:item[0])).encode()).hexdigest())
