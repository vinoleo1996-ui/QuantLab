"""Contract tests for the isolated L2 score research path.

These tests intentionally do not construct an account or invoke the frozen
router.  They exercise provenance, executable-label timing, and score-only
boundaries before an L2 provider trial can be admitted to training.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.data.contracts import InstrumentRules, decision_time
from quantlab.data.l2_contract import (
    L2ContractError,
    L2LabelContract,
    L2ScoreModelContract,
    L2_FEATURE_KEYS,
    build_executable_labels,
    fixed_l2_manifest,
    materialize_l2_matrix,
    order_new_entries,
    order_new_entries_from_l2_score_artifact,
    purge_l2_training_rows,
    validate_horizon_pair,
    validate_l2_labels,
    validate_l2_scores,
)
from quantlab.research_signal_ledger import ResearchSignalLedger


RAW_BYTES = b"l2-provider-trial-receipt"
RAW_HASH = hashlib.sha256(RAW_BYTES).hexdigest()


class L2ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._ledger_now = "2024-01-02 17:59:00+08:00"
        self.ledger = ResearchSignalLedger(
            Path(self.temp.name) / "research-signal-ledger", now_fn=lambda: self._ledger_now
        )

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        metadata = {
            key: dict(
                unit="ratio",
                dtype="float64",
                source_field="trial." + key,
                source_version="trial-release-1",
            )
            for key in L2_FEATURE_KEYS
        }
        return fixed_l2_manifest(
            provider="qualified-trial-provider",
            trial_id="trial-20260929",
            source_dataset_id="trial-snapshot-1",
            source_release_id="release-1",
            field_metadata=metadata,
        )

    def ledger_receipt(self, day):
        self._ledger_now = day + " 17:59:00+08:00"
        receipt = self.ledger.record_receipt(
            source_id="l2-trial-provider",
            source_family="l2-trial-provider",
            origin_id="l2-trial-" + day,
            canonical_url="https://provider.test/l2/" + day,
            raw=RAW_BYTES,
            content_type="application/json",
            published_at=day + " 17:00:00+08:00",
            first_captured_at=day + " 17:59:00+08:00",
            available_at=day + " 17:59:00+08:00",
            availability_basis="provider_trial_capture",
            now=day + " 17:59:00+08:00",
        )
        return receipt["canonical_receipt_id"] if "canonical_receipt_id" in receipt else receipt["event_id"]

    def receipts(self, *, day="2024-01-02", code="S000"):
        manifest = self.manifest()
        receipt_id = self.ledger_receipt(day)
        rows = []
        for index, spec in enumerate(manifest.fields, start=1):
            rows.append(
                dict(
                    decision_date=day,
                    instrument_id=code,
                    feature_key=spec.key,
                    value=float(index),
                    unit=spec.unit,
                    source_field=spec.source_field,
                    available_at=day + " 17:59:00+08:00",
                    source_version=spec.source_version,
                    raw_response_sha256=RAW_HASH,
                    ledger_id=self.ledger.ledger_id,
                    ledger_receipt_id=receipt_id,
                    provider=manifest.provider,
                    provider_trial_id=manifest.trial_id,
                    source_as_of=day + " 17:59:00+08:00",
                    quality_status="ok",
                )
            )
        return pd.DataFrame(rows)

    def rule_map(self, ds):
        rules = {
            row.instrument_id: InstrumentRules.from_record(row)
            for _, row in ds.instruments.drop_duplicates("instrument_id", keep="last").iterrows()
        }
        return lambda _day: rules

    def score_model(self):
        return L2ScoreModelContract(
            model_id="catboost-score-fixed-v1",
            model_config_sha256=RAW_HASH,
        )

    def score_frame(self, manifest, matrix, *, contract=None):
        contract = contract or L2LabelContract.h5()
        model = self.score_model()
        source = matrix.iloc[0]
        return pd.DataFrame(
            [
                dict(
                    decision_date=source.decision_date,
                    instrument_id=source.instrument_id,
                    score=.3,
                    score_rank=1.,
                    model_id=model.model_id,
                    model_family=model.model_family,
                    model_config_sha256=model.model_config_sha256,
                    model_contract_sha256=model.sha256,
                    training_as_of="2024-01-01 18:00:00+08:00",
                    training_label_available_at="2024-01-01 18:00:00+08:00",
                    feature_manifest_sha256=manifest.sha256,
                    label_contract_sha256=contract.sha256,
                    dataset_identity_sha256=source.dataset_identity_sha256,
                    matrix_row_sha256=source.matrix_row_sha256,
                    materialized_matrix_sha256=matrix.attrs['materialized_matrix_sha256'],
                    horizon_sessions=contract.horizon_sessions,
                    status="ok",
                    abstain_reason="",
                )
            ]
        )

    def test_manifest_requires_exact_fixed_14_key_order_and_complete_metadata(self):
        manifest = self.manifest()
        self.assertEqual(tuple(field.key for field in manifest.fields), L2_FEATURE_KEYS)
        self.assertEqual(len(manifest.sha256), 64)
        metadata = {
            key: dict(unit="ratio", dtype="float64", source_field=key, source_version="v1")
            for key in L2_FEATURE_KEYS[:-1]
        }
        with self.assertRaisesRegex(L2ContractError, "field_metadata_keys"):
            fixed_l2_manifest(
                provider="p", trial_id="t", source_dataset_id="d", source_release_id="r", field_metadata=metadata
            )
        metadata = {
            key: dict(unit="ratio", dtype="float64", source_field=key, source_version="v1")
            for key in L2_FEATURE_KEYS
        }
        del metadata[L2_FEATURE_KEYS[0]]["unit"]
        with self.assertRaisesRegex(TypeError, "unit"):
            fixed_l2_manifest(
                provider="p", trial_id="t", source_dataset_id="d", source_release_id="r", field_metadata=metadata
            )

    def test_missing_value_or_required_feature_materializes_an_abstention_not_an_imputation(self):
        frame = self.receipts()
        frame = frame.iloc[:-1].copy()
        frame.loc[frame.index[0], "value"] = np.nan
        matrix = materialize_l2_matrix(frame, self.manifest(), receipt_ledger=self.ledger)
        self.assertEqual(len(matrix), 1)
        row = matrix.iloc[0]
        self.assertEqual(row.status, "abstain")
        self.assertIn("missing_feature:" + L2_FEATURE_KEYS[-1], row.abstain_reason)
        self.assertIn("nonfinite_feature_value", row.abstain_reason)
        self.assertTrue(row[list(L2_FEATURE_KEYS)].isna().all())

    def test_known_at_trial_hash_source_and_duplicate_defects_are_fail_closed(self):
        frame = self.receipts()
        frame.loc[frame.feature_key.eq(L2_FEATURE_KEYS[0]), "available_at"] = "2024-01-02 18:01:00+08:00"
        frame.loc[frame.feature_key.eq(L2_FEATURE_KEYS[1]), "raw_response_sha256"] = "not-a-hash"
        frame.loc[frame.feature_key.eq(L2_FEATURE_KEYS[2]), "provider_trial_id"] = "wrong-trial"
        frame = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
        matrix = materialize_l2_matrix(frame, self.manifest(), receipt_ledger=self.ledger)
        row = matrix.iloc[0]
        self.assertEqual(row.status, "abstain")
        self.assertIn("feature_available_after_decision", row.abstain_reason)
        self.assertIn("invalid_raw_response_sha256", row.abstain_reason)
        self.assertIn("provider_trial_mismatch", row.abstain_reason)
        self.assertIn("duplicate_feature_receipt", row.abstain_reason)

    def test_source_as_of_cannot_lag_its_available_at(self):
        frame = self.receipts()
        first = frame.feature_key.eq(L2_FEATURE_KEYS[0])
        frame.loc[first, "available_at"] = "2024-01-02 17:58:00+08:00"
        frame.loc[first, "source_as_of"] = "2024-01-02 17:59:00+08:00"
        row = materialize_l2_matrix(frame, self.manifest(), receipt_ledger=self.ledger).iloc[0]
        self.assertEqual(row.status, "abstain")
        self.assertIn("source_as_of_after_available_at", row.abstain_reason)

    def test_well_formed_forged_raw_hash_cannot_bypass_canonical_ledger_blob(self):
        frame = self.receipts()
        frame.loc[frame.feature_key.eq(L2_FEATURE_KEYS[0]), "raw_response_sha256"] = hashlib.sha256(
            b"forged-but-well-formed"
        ).hexdigest()
        row = materialize_l2_matrix(frame, self.manifest(), receipt_ledger=self.ledger).iloc[0]
        self.assertEqual(row.status, "abstain")
        self.assertIn("raw_sha256_not_ledger_verified", row.abstain_reason)

    def test_l2_materialization_requires_a_canonical_research_signal_ledger(self):
        with self.assertRaisesRegex(L2ContractError, "receipt_ledger_required"):
            materialize_l2_matrix(self.receipts(), self.manifest(), receipt_ledger=None)

    def test_future_receipts_do_not_rewrite_a_materialized_prior_cross_section(self):
        first = materialize_l2_matrix(self.receipts(), self.manifest(), receipt_ledger=self.ledger)
        later = self.receipts(day="2024-01-03", code="S001")
        combined = materialize_l2_matrix(
            pd.concat([self.receipts(), later], ignore_index=True),
            self.manifest(),
            receipt_ledger=self.ledger,
        )
        left = first.set_index(["decision_date", "instrument_id"]).loc[(pd.Timestamp("2024-01-02"), "S000")]
        right = combined.set_index(["decision_date", "instrument_id"]).loc[(pd.Timestamp("2024-01-02"), "S000")]
        pd.testing.assert_series_equal(left, right)

    def test_h5_label_is_exact_next_open_to_sixth_next_open_and_matures_at_exit(self):
        ds = dataset(days=26, stocks=2)
        contract = L2LabelContract.h5()
        labels = build_executable_labels(ds, contract, rules_for_date=self.rule_map(ds))
        observed = labels[labels.status.eq("observed")]
        self.assertFalse(observed.empty)
        row = observed.iloc[0]
        signal = pd.Timestamp(row.decision_date)
        index = ds.calendar.get_loc(signal)
        self.assertEqual(pd.Timestamp(row.entry_date), ds.calendar[index + 1])
        self.assertEqual(pd.Timestamp(row.exit_date), ds.calendar[index + 6])
        panel = ds.panels[row.instrument_id]
        self.assertAlmostEqual(row.target_gross_return, panel.loc[ds.calendar[index + 6], "open"] / panel.loc[ds.calendar[index + 1], "open"] - 1)
        self.assertGreaterEqual(pd.Timestamp(row.label_available_at), decision_time(ds.calendar[index + 6]))
        validate_l2_labels(labels, contract, calendar=ds.calendar)

    def test_unavailable_open_permission_is_an_abstention_not_a_zero_return(self):
        ds = dataset(days=20, stocks=2)
        contract = L2LabelContract.h5()

        def deny(_bar, _rule, side, _date, **_kwargs):
            return False, "limit_up" if side == "buy" else "limit_down"

        with patch("quantlab.backtest.ledger.trade_permission", side_effect=deny):
            labels = build_executable_labels(ds, contract, rules_for_date=self.rule_map(ds))
        blocked = labels[labels.reason.eq("entry_limit_up")]
        self.assertFalse(blocked.empty)
        self.assertTrue(blocked.status.eq("abstain").all())
        self.assertTrue(blocked.target_gross_return.isna().all())

    def test_actual_label_end_and_late_availability_both_enforce_embargo(self):
        contract = L2LabelContract.h5()
        rows = pd.DataFrame(
            [
                dict(
                    decision_date="2024-01-02", instrument_id="A", horizon_sessions=5, execution_lag_sessions=1,
                    entry_date="2024-01-03", exit_date="2024-01-10", target_gross_return=.1,
                    label_end="2024-01-10", label_available_at="2024-01-14 18:00:00+08:00",
                    status="observed", reason="", label_contract_sha256=contract.sha256,
                ),
                dict(
                    decision_date="2024-01-03", instrument_id="B", horizon_sessions=5, execution_lag_sessions=1,
                    entry_date="2024-01-04", exit_date="2024-01-11", target_gross_return=.1,
                    label_end="2024-01-11", label_available_at="2024-01-15 18:00:00+08:00",
                    status="observed", reason="", label_contract_sha256=contract.sha256,
                ),
            ]
        )
        kept = purge_l2_training_rows(rows, "2024-01-15 18:00:00+08:00", contract)
        self.assertEqual(list(kept.instrument_id), ["A"])
        self.assertEqual(kept.attrs["embargo_sessions"], 6)
        self.assertEqual(kept.attrs["purged_rows"], 1)

    def test_h5_and_h20_contracts_and_labels_cannot_be_mixed(self):
        h5, h20 = L2LabelContract.h5(), L2LabelContract.h20()
        validate_horizon_pair(h5, h20)
        with self.assertRaisesRegex(L2ContractError, "exit_offset"):
            L2LabelContract(horizon_sessions=20, exit_offset_sessions=6)
        labels = pd.DataFrame(
            [
                dict(
                    decision_date="2024-01-02", instrument_id="A", horizon_sessions=5, execution_lag_sessions=1,
                    entry_date="2024-01-03", exit_date="2024-01-10", target_gross_return=.1,
                    label_end="2024-01-10", label_available_at="2024-01-10 18:00:00+08:00",
                    status="observed", reason="", label_contract_sha256=h5.sha256,
                )
            ]
        )
        with self.assertRaisesRegex(L2ContractError, "horizon_mismatch"):
            validate_l2_labels(labels, h20)

    def test_score_artifact_is_ranking_only_and_bound_to_qualified_feature_identity(self):
        manifest = self.manifest()
        matrix = materialize_l2_matrix(self.receipts(), manifest, receipt_ledger=self.ledger)
        source = matrix.iloc[0]
        h5 = L2LabelContract.h5()
        score_model = self.score_model()
        scores = pd.DataFrame(
            [
                dict(
                    decision_date=source.decision_date,
                    instrument_id=source.instrument_id,
                    score=.3,
                    score_rank=1.,
                    model_id="catboost-score-fixed-v1",
                    model_family="catboost",
                    model_config_sha256=RAW_HASH,
                    model_contract_sha256=score_model.sha256,
                    training_as_of="2024-01-01 18:00:00+08:00",
                    training_label_available_at="2024-01-01 18:00:00+08:00",
                    feature_manifest_sha256=manifest.sha256,
                    label_contract_sha256=h5.sha256,
                    dataset_identity_sha256=source.dataset_identity_sha256,
                    matrix_row_sha256=source.matrix_row_sha256,
                    materialized_matrix_sha256=matrix.attrs['materialized_matrix_sha256'],
                    horizon_sessions=5,
                    status="ok",
                    abstain_reason="",
                )
            ]
        )
        checked = validate_l2_scores(scores, manifest, h5, model_contract=score_model, matrix=matrix)
        self.assertEqual(checked.iloc[0].score, .3)
        bad = scores.assign(position_score=.7)
        with self.assertRaisesRegex(L2ContractError, "position_or_policy"):
            validate_l2_scores(bad, manifest, h5, model_contract=score_model, matrix=matrix)
        namespaced = scores.assign(**{"quality.position.position_score": .7})
        with self.assertRaisesRegex(L2ContractError, "position_or_policy"):
            validate_l2_scores(namespaced, manifest, h5, model_contract=score_model, matrix=matrix)
        with self.assertRaisesRegex(L2ContractError, "horizon_mismatch"):
            validate_l2_scores(scores, manifest, L2LabelContract.h20(), model_contract=score_model, matrix=matrix)
        with self.assertRaisesRegex(L2ContractError, "labels_not_available_by_training_as_of"):
            validate_l2_scores(
                scores.assign(training_label_available_at="2024-01-01 18:01:00+08:00"),
                manifest,
                h5,
                model_contract=score_model,
                matrix=matrix,
            )

    def test_scores_require_an_untampered_qualified_materialized_matrix_and_row_binding(self):
        manifest = self.manifest()
        matrix = materialize_l2_matrix(self.receipts(), manifest, receipt_ledger=self.ledger)
        scores = self.score_frame(manifest, matrix)
        model = self.score_model()
        with self.assertRaisesRegex(L2ContractError, "qualified_matrix_required"):
            validate_l2_scores(scores, manifest, L2LabelContract.h5(), model_contract=model)
        tampered_matrix = matrix.copy()
        tampered_matrix.loc[0, L2_FEATURE_KEYS[0]] = 999.0
        with self.assertRaisesRegex(L2ContractError, "matrix_row_identity_mismatch"):
            validate_l2_scores(
                scores, manifest, L2LabelContract.h5(), model_contract=model, matrix=tampered_matrix
            )
        with self.assertRaisesRegex(L2ContractError, "score_matrix_row_identity_mismatch"):
            validate_l2_scores(
                scores.assign(matrix_row_sha256="f" * 64),
                manifest,
                L2LabelContract.h5(),
                model_contract=model,
                matrix=matrix,
            )
        with self.assertRaisesRegex(L2ContractError, "score_materialized_matrix_identity_mismatch"):
            validate_l2_scores(
                scores.assign(materialized_matrix_sha256="f" * 64),
                manifest,
                L2LabelContract.h5(),
                model_contract=model,
                matrix=matrix,
            )

    def test_score_artifact_rejects_prefixed_policy_fields_and_all_unregistered_columns(self):
        manifest = self.manifest()
        matrix = materialize_l2_matrix(self.receipts(), manifest, receipt_ledger=self.ledger)
        scores = self.score_frame(manifest, matrix)
        model = self.score_model()
        with self.assertRaisesRegex(L2ContractError, "position_or_policy"):
            validate_l2_scores(
                scores.assign(weight_for_new_entry=.2),
                manifest,
                L2LabelContract.h5(),
                model_contract=model,
                matrix=matrix,
            )
        with self.assertRaisesRegex(L2ContractError, "unexpected_columns"):
            validate_l2_scores(
                scores.assign(research_note="only a note"),
                manifest,
                L2LabelContract.h5(),
                model_contract=model,
                matrix=matrix,
            )

    def test_catboost_score_contract_forbids_fallback_and_model_identity_mismatch(self):
        model = self.score_model()
        self.assertEqual(model.model_family, "catboost")
        with self.assertRaisesRegex(L2ContractError, "fallback_forbidden"):
            L2ScoreModelContract(
                model_id="catboost-score-fixed-v1",
                model_config_sha256=RAW_HASH,
                fallback_model_id="linear-fallback-v1",
            )

        manifest = self.manifest()
        matrix = materialize_l2_matrix(self.receipts(), manifest, receipt_ledger=self.ledger)
        source = matrix.iloc[0]
        scores = pd.DataFrame(
            [dict(
                decision_date=source.decision_date,
                instrument_id=source.instrument_id,
                score=.3,
                score_rank=1.,
                model_id=model.model_id,
                model_family=model.model_family,
                model_config_sha256=model.model_config_sha256,
                model_contract_sha256=model.sha256,
                training_as_of="2024-01-01 18:00:00+08:00",
                training_label_available_at="2024-01-01 18:00:00+08:00",
                feature_manifest_sha256=manifest.sha256,
                label_contract_sha256=L2LabelContract.h5().sha256,
                dataset_identity_sha256=source.dataset_identity_sha256,
                matrix_row_sha256=source.matrix_row_sha256,
                materialized_matrix_sha256=matrix.attrs['materialized_matrix_sha256'],
                horizon_sessions=5,
                status="ok",
                abstain_reason="",
            )]
        )
        with self.assertRaisesRegex(L2ContractError, "model_id_mismatch"):
            validate_l2_scores(
                scores.assign(model_id="linear-fallback-v1"), manifest, L2LabelContract.h5(),
                model_contract=model, matrix=matrix,
            )
        with self.assertRaisesRegex(L2ContractError, "model_config_hash_mismatch"):
            validate_l2_scores(
                scores.assign(model_config_sha256="f" * 64), manifest, L2LabelContract.h5(),
                model_contract=model, matrix=matrix,
            )
        with self.assertRaisesRegex(L2ContractError, "fallback"):
            validate_l2_scores(
                scores.assign(fallback_model_id="linear-fallback-v1"), manifest, L2LabelContract.h5(),
                model_contract=model, matrix=matrix,
            )

    def test_execution_assumptions_and_permission_adapter_must_be_hashed_into_label_contract(self):
        ds = dataset(days=20, stocks=2)
        rules = self.rule_map(ds)
        default = L2LabelContract.h5()
        self.assertRegex(default.permission_adapter_identity, r"^sha256:[0-9a-f]{64}$")
        open_assumption = L2LabelContract(
            horizon_sessions=5,
            exit_offset_sessions=6,
            allow_open_assumption=True,
        )
        self.assertNotEqual(default.sha256, open_assumption.sha256)
        with self.assertRaisesRegex(L2ContractError, "open_assumption_contract_mismatch"):
            build_executable_labels(
                ds,
                default,
                rules_for_date=rules,
                allow_open_assumption=True,
            )

        def deny_permission(_bar, _rule, _side, _date, **_kwargs):
            return False, "test_permission_denied"

        custom_identity = "sha256:" + RAW_HASH
        custom = L2LabelContract(
            horizon_sessions=5,
            exit_offset_sessions=6,
            permission_adapter_identity=custom_identity,
        )
        self.assertNotEqual(default.sha256, custom.sha256)
        with self.assertRaisesRegex(L2ContractError, "permission_adapter_identity_required"):
            build_executable_labels(ds, custom, rules_for_date=rules, permission_adapter=deny_permission)
        with self.assertRaisesRegex(L2ContractError, "permission_adapter_identity_mismatch"):
            build_executable_labels(
                ds,
                custom,
                rules_for_date=rules,
                permission_adapter=deny_permission,
                permission_adapter_identity="sha256:" + "f" * 64,
            )
        labels = build_executable_labels(
            ds,
            custom,
            rules_for_date=rules,
            permission_adapter=deny_permission,
            permission_adapter_identity=custom_identity,
        )
        self.assertTrue(labels.label_contract_sha256.eq(custom.sha256).all())
        matured = labels.label_end.notna()
        denied = labels.loc[matured & labels.reason.eq("entry_test_permission_denied")]
        self.assertFalse(denied.empty)
        self.assertTrue(denied.status.eq("abstain").all())

    def test_entry_only_order_preserves_retained_holdings_and_fixed_candidate_set(self):
        order = order_new_entries(("HELD_B", "HELD_A"), ("NEW_C", "NEW_A", "NEW_B"),
                                  {"NEW_A": .1, "NEW_B": .8, "NEW_C": .8})
        self.assertEqual(order.retained_holdings, ("HELD_B", "HELD_A"))
        self.assertEqual(order.ordered_new_entries, ("NEW_B", "NEW_C", "NEW_A"))
        with self.assertRaisesRegex(L2ContractError, "retained_holding"):
            order_new_entries(("HELD_A",), ("HELD_A", "NEW_A"), {"HELD_A": .9, "NEW_A": .1})
        with self.assertRaisesRegex(L2ContractError, "score_missing"):
            order_new_entries((), ("NEW_A",), {})

    def test_only_h20_validated_score_artifact_can_create_an_entry_order(self):
        manifest = self.manifest()
        matrix = materialize_l2_matrix(self.receipts(), manifest, receipt_ledger=self.ledger)
        model = self.score_model()
        h20 = L2LabelContract.h20()
        scores = self.score_frame(manifest, matrix, contract=h20)
        code = matrix.iloc[0].instrument_id

        order = order_new_entries_from_l2_score_artifact(
            retained_holdings=("HELD_A",),
            fixed_entry_candidates=(code,),
            decision_date=matrix.iloc[0].decision_date,
            scores=scores,
            manifest=manifest,
            contract=h20,
            model_contract=model,
            matrix=matrix,
        )
        self.assertEqual(order.retained_holdings, ("HELD_A",))
        self.assertEqual(order.ordered_new_entries, (code,))

        with self.assertRaisesRegex(L2ContractError, "entry_order_requires_h20"):
            order_new_entries_from_l2_score_artifact(
                retained_holdings=(),
                fixed_entry_candidates=(code,),
                decision_date=matrix.iloc[0].decision_date,
                scores=self.score_frame(manifest, matrix, contract=L2LabelContract.h5()),
                manifest=manifest,
                contract=L2LabelContract.h5(),
                model_contract=model,
                matrix=matrix,
            )

        abstained = scores.assign(status="abstain", score=np.nan, score_rank=np.nan,
                                  abstain_reason="model_abstained")
        with self.assertRaisesRegex(L2ContractError, "entry_order_score_not_observed"):
            order_new_entries_from_l2_score_artifact(
                retained_holdings=(),
                fixed_entry_candidates=(code,),
                decision_date=matrix.iloc[0].decision_date,
                scores=abstained,
                manifest=manifest,
                contract=h20,
                model_contract=model,
                matrix=matrix,
            )

    def test_entry_only_order_rejects_ambiguous_codes_duplicate_indexes_and_extra_scores(self):
        with self.assertRaisesRegex(L2ContractError, "candidate_invalid_code"):
            order_new_entries((), ("NEW_A", " "), {"NEW_A": .1, " ": .2})
        with self.assertRaisesRegex(L2ContractError, "score_index_duplicate_code"):
            order_new_entries(
                (),
                ("NEW_A", "NEW_B"),
                pd.Series([.1, .2, .3], index=["NEW_A", "NEW_A", "NEW_B"]),
            )
        with self.assertRaisesRegex(L2ContractError, "score_unexpected"):
            order_new_entries((), ("NEW_A",), {"NEW_A": .1, "OUTSIDE": .2})
        with self.assertRaisesRegex(L2ContractError, "scores_must_be_mapping_or_series"):
            order_new_entries((), ("NEW_A",), [(.1,)])


if __name__ == "__main__":
    unittest.main()
