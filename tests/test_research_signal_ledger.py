"""Behavioral contracts for the append-only external-signal research ledger.

The tests intentionally use only captured bytes and a temporary directory.  A
ledger receipt is evidence of a forward capture, not a market-data request or
an account instruction.
"""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quantlab.research_signal_ledger import ResearchSignalLedger


DAY = "2026-01-05"
DECISION = DAY + " 18:00:00+08:00"


class ResearchSignalLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "ledger"
        # Signal-event timestamps are intentionally ledger-owned.  Each test
        # controls that clock through ``now_fn`` rather than relying on the
        # public call-site ``now`` argument, which must not backdate evidence.
        self._runtime_now = pd.Timestamp(DAY + " 17:00:00+08:00")
        self.ledger = ResearchSignalLedger(self.root, now_fn=lambda: self._runtime_now)

    def tearDown(self):
        self.temp.cleanup()

    def _set_runtime_now(self, value):
        self._runtime_now = pd.Timestamp(value)

    def _scope(self, members=("stock:000001",), *, now=DECISION):
        self._set_runtime_now(now)
        return self.ledger.declare_scope(
            "U0",
            DECISION,
            members,
            definition_version="u0-adv60-v1",
            source_hash=hashlib.sha256(b"frozen-membership-snapshot").hexdigest(),
            now=now,
        )

    def _receipt(
        self,
        *,
        raw=b'{"headline":"capture"}',
        source_id="publisher-a",
        source_family="publisher-a",
        origin_id="origin-a",
        published=DAY + " 17:00:00+08:00",
        first=DAY + " 17:10:00+08:00",
        available=None,
        now=None,
    ):
        available = available or first
        self._set_runtime_now(now or first)
        return self.ledger.record_receipt(
            source_id=source_id,
            source_family=source_family,
            origin_id=origin_id,
            canonical_url="https://example.test/" + source_id,
            raw=raw,
            content_type="application/json",
            published_at=published,
            first_captured_at=first,
            available_at=available,
            availability_basis="first_capture",
            now=now or first,
        )

    def _entity(self, receipt, key="stock:000001", *, status="resolved", now=DAY + " 17:30:00+08:00"):
        self._set_runtime_now(now)
        return self.ledger.record_entity(
            receipt["canonical_receipt_id"],
            entity_type="stock",
            canonical_id=key if status == "resolved" else None,
            display_name=key,
            mapping_method="fixture-mapper",
            mapping_version="fixture-v1",
            status=status,
            reason="ambiguous_ticker" if status == "abstained" else None,
            now=now,
        )

    @staticmethod
    def _rule():
        return {"kind": "rule", "version": "theme-carrier-minimal-v1", "code_hash": "fixture"}

    def _observe(self, receipt, entity, *, key="theme:000001", target=("stock:000001",), now=DAY + " 17:57:00+08:00", **kwargs):
        extractor = kwargs.pop("extractor", self._rule())
        self._set_runtime_now(now)
        return self.ledger.record_observation(
            observation_key=key,
            scope_id="U0",
            decision_at=DECISION,
            feature_id="theme_carrier_minimal_v1",
            target_entity_keys=target,
            receipt_ids=[receipt["event_id"]],
            entity_ids=[entity["event_id"]],
            value=0.25,
            unit="percentile",
            extractor=extractor,
            now=now,
            **kwargs,
        )

    def _complete_eligible_signal(self, *, caller_now):
        """Write scope, receipt, and entity with real pre-cutoff ledger times."""
        self._set_runtime_now(DAY + " 17:50:00+08:00")
        self.ledger.declare_scope(
            "U0",
            DECISION,
            ["stock:000001"],
            definition_version="u0-adv60-v1",
            source_hash=hashlib.sha256(b"frozen-membership-snapshot").hexdigest(),
            now=caller_now,
        )
        self._set_runtime_now(DAY + " 17:53:00+08:00")
        receipt = self.ledger.record_receipt(
            source_id="publisher-a",
            source_family="publisher-a",
            origin_id="complete-signal",
            canonical_url="https://example.test/complete-signal",
            raw=b"complete signal",
            content_type="application/json",
            published_at=DAY + " 17:00:00+08:00",
            first_captured_at=DAY + " 17:52:00+08:00",
            available_at=DAY + " 17:52:00+08:00",
            availability_basis="first_capture",
            now=caller_now,
        )
        self._set_runtime_now(DAY + " 17:55:00+08:00")
        entity = self.ledger.record_entity(
            receipt["canonical_receipt_id"],
            entity_type="stock",
            canonical_id="stock:000001",
            display_name="stock:000001",
            mapping_method="fixture-mapper",
            mapping_version="fixture-v1",
            now=caller_now,
        )
        return receipt, entity

    def test_receipt_blob_chain_and_exact_refetch_dedupe(self):
        first = self._receipt(raw=b"captured payload")
        duplicate = self._receipt(
            raw=b"captured payload",
            first=DAY + " 17:11:00+08:00",
            now=DAY + " 17:11:00+08:00",
        )

        self.assertEqual(first["status"], "accepted")
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(duplicate["body"]["duplicate_of"], first["event_id"])
        self.assertEqual(duplicate["canonical_receipt_id"], first["event_id"])
        self.assertEqual(len(list((self.root / "blobs").iterdir())), 1)
        self.assertTrue(self.ledger.verify()["verified"])
        before = self.ledger.verify()["events"]
        evidence = self.ledger.export_receipt_evidence(duplicate["event_id"])
        self.assertEqual(evidence["requested_receipt_id"], duplicate["event_id"])
        self.assertEqual(evidence["canonical_receipt_id"], first["event_id"])
        self.assertEqual(evidence["raw"]["sha256"], first["body"]["raw_sha256"])
        self.assertEqual(self.ledger.verify()["events"], before)

    def test_transport_evidence_is_persisted_but_cannot_be_used_as_a_signal_source(self):
        self._set_runtime_now(DAY + " 17:12:00+08:00")
        attempt = self.ledger.record_transport_evidence(
            collection_id="forward-eod-20260105",
            source_manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
            source_manifest_event_sha256=hashlib.sha256(b"manifest-event").hexdigest(),
            session=DAY,
            decision_at=DECISION,
            status="failed",
            failure_reason="page_fetch_failed",
            attempts=(
                {
                    "page_number": 1,
                    "canonical_url": "https://example.test/page/1",
                    "captured_at": DAY + " 17:10:00+08:00",
                    "raw_sha256": hashlib.sha256(b"first-page").hexdigest(),
                    "fetch_error": None,
                },
                {
                    "page_number": 2,
                    "canonical_url": "https://example.test/page/2",
                    "captured_at": DAY + " 17:11:00+08:00",
                    "raw_sha256": None,
                    "fetch_error": "TimeoutError",
                },
            ),
            first_captured_at=DAY + " 17:10:00+08:00",
            available_at=DAY + " 17:11:00+08:00",
            now=DAY + " 17:12:00+08:00",
        )

        reopened = ResearchSignalLedger(self.root)
        evidence = reopened.export_receipt_evidence(attempt["event_id"])
        self.assertEqual(evidence["receipt"]["body"]["receipt_role"], "transport_evidence")
        self.assertEqual(evidence["receipt"]["body"]["metadata"]["transport"]["failure_reason"], "page_fetch_failed")
        self.assertEqual(len(evidence["receipt"]["body"]["metadata"]["transport"]["attempts"]), 2)
        with self.assertRaisesRegex(ValueError, "receipt_not_source_response"):
            reopened.record_entity(
                attempt["event_id"],
                entity_type="stock",
                canonical_id="stock:000001",
                display_name="stock:000001",
                mapping_method="forbidden",
                mapping_version="v1",
                now=DAY + " 17:30:00+08:00",
            )
        self.assertTrue(reopened.verify()["verified"])

    def test_zero_page_failed_transport_retains_reported_failure_evidence(self):
        self._set_runtime_now(DAY + " 17:13:00+08:00")
        event = self.ledger.record_transport_evidence(
            collection_id="forward-eod-20260105-preflight",
            source_manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
            source_manifest_event_sha256=hashlib.sha256(b"manifest-event").hexdigest(),
            session=DAY,
            decision_at=DECISION,
            status="failed",
            failure_reason="transport_registration_failed",
            attempts=(),
            first_captured_at=None,
            available_at=None,
            reported_at=DAY + " 17:12:00+08:00",
            now=DAY + " 16:00:00+08:00",
        )

        body = event["body"]
        transport = body["metadata"]["transport"]
        self.assertEqual(event["available_at"], DAY + "T17:12:00+08:00")
        self.assertEqual(transport["failure_reported_at"], DAY + "T17:12:00+08:00")
        self.assertEqual(transport["attempts"], [])
        self.assertEqual(event["recorded_at"], DAY + "T17:13:00+08:00")
        self.assertEqual(transport["receipt_written_at"], event["recorded_at"])
        self.assertTrue(self.ledger.verify()["verified"])

        with self.assertRaisesRegex(ValueError, "transport_attempts_required"):
            self.ledger.record_transport_evidence(
                collection_id="forward-eod-20260105-missing-clock",
                source_manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
                source_manifest_event_sha256=hashlib.sha256(b"manifest-event").hexdigest(),
                session=DAY,
                decision_at=DECISION,
                status="failed",
                failure_reason="transport_registration_failed",
                attempts=(),
                first_captured_at=None,
                available_at=None,
            )
        with self.assertRaisesRegex(ValueError, "transport_attempts_required"):
            self.ledger.record_transport_evidence(
                collection_id="forward-eod-20260105-empty-prepared",
                source_manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
                source_manifest_event_sha256=hashlib.sha256(b"manifest-event").hexdigest(),
                session=DAY,
                decision_at=DECISION,
                status="prepared",
                failure_reason=None,
                attempts=(),
                first_captured_at=None,
                available_at=None,
                reported_at=DAY + " 17:12:00+08:00",
            )

    def test_missing_published_time_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "missing_published_at"):
            self._receipt(published=None)

    def test_scope_requires_a_real_sha256_formatted_source_hash(self):
        for source_hash in ("frozen-membership-sha256", "a" * 63, "g" * 64):
            with self.subTest(source_hash=source_hash):
                with self.assertRaisesRegex(ValueError, "invalid_scope_source_hash"):
                    self.ledger.declare_scope(
                        "U0",
                        DECISION,
                        ["stock:000001"],
                        definition_version="u0-adv60-v1",
                        source_hash=source_hash,
                        now=DECISION,
                    )

    def test_complete_pre_cutoff_signal_is_observed_using_ledger_write_clock(self):
        # Every call lies that it happened after 18:00.  The injected ledger
        # clock proves the real writes completed by 17:57, so the feature is
        # eligible and the stored timestamps must retain those real times.
        caller_now = DAY + " 18:05:00+08:00"
        receipt, entity = self._complete_eligible_signal(caller_now=caller_now)

        self._set_runtime_now(DAY + " 17:57:00+08:00")
        result = self.ledger.record_observation(
            observation_key="pre-cutoff",
            scope_id="U0",
            decision_at=DECISION,
            feature_id="theme_carrier_minimal_v1",
            target_entity_keys=["stock:000001"],
            receipt_ids=[receipt["event_id"]],
            entity_ids=[entity["event_id"]],
            value=0.25,
            unit="percentile",
            extractor=self._rule(),
            now=caller_now,
        )

        self.assertEqual(result["status"], "observed")
        self.assertEqual(receipt["recorded_at"], DAY + "T17:53:00+08:00")
        self.assertEqual(entity["recorded_at"], DAY + "T17:55:00+08:00")
        self.assertEqual(result["recorded_at"], DAY + "T17:57:00+08:00")
        self.assertEqual(result["body"]["value"], 0.25)

    def test_post_cutoff_observation_is_converted_to_value_less_abstention(self):
        receipt, entity = self._complete_eligible_signal(caller_now=DAY + " 17:57:00+08:00")

        # The caller tries to backdate the feature, but the ledger's actual
        # clock is already after the 18:00 freeze.
        self._set_runtime_now(DAY + " 18:01:00+08:00")
        result = self.ledger.record_observation(
            observation_key="post-cutoff-observed",
            scope_id="U0",
            decision_at=DECISION,
            feature_id="theme_carrier_minimal_v1",
            target_entity_keys=["stock:000001"],
            receipt_ids=[receipt["event_id"]],
            entity_ids=[entity["event_id"]],
            value=0.25,
            unit="percentile",
            extractor=self._rule(),
            now=DAY + " 17:57:00+08:00",
        )

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["recorded_at"], DAY + "T18:01:00+08:00")
        self.assertEqual(result["body"]["reason"], "observation_write_after_decision_cutoff")
        self.assertIsNone(result["body"]["value"])
        coverage = self.ledger.coverage("U0", DECISION, "theme_carrier_minimal_v1")
        self.assertEqual((coverage["observed"], coverage["abstained"]), (0, 1))

    def test_post_cutoff_explicit_abstention_is_allowed_with_ledger_timestamp(self):
        self._set_runtime_now(DAY + " 17:50:00+08:00")
        self.ledger.declare_scope(
            "U0",
            DECISION,
            ["stock:000001"],
            definition_version="u0-adv60-v1",
            source_hash=hashlib.sha256(b"frozen-membership-snapshot").hexdigest(),
            now=DAY + " 17:50:00+08:00",
        )

        self._set_runtime_now(DAY + " 18:02:00+08:00")
        result = self.ledger.abstain(
            observation_key="late-explicit-abstention",
            scope_id="U0",
            decision_at=DECISION,
            feature_id="theme_carrier_minimal_v1",
            target_entity_keys=["stock:000001"],
            stage="receipt",
            reason="late_source_receipt",
            now=DAY + " 17:57:00+08:00",
        )

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["recorded_at"], DAY + "T18:02:00+08:00")
        self.assertEqual(result["body"]["reason"], "late_source_receipt")
        self.assertIsNone(result["body"]["value"])

    def test_reprint_is_not_a_second_independent_source_and_abstains(self):
        first = self._receipt(raw=b"original", origin_id="wire-story-1")
        reprint = self._receipt(
            raw=b"publisher-b rewrite",
            source_id="publisher-b",
            source_family="publisher-b",
            origin_id="wire-story-1",
            first=DAY + " 17:20:00+08:00",
        )
        entity = self._entity(first)
        self._scope()

        result = self.ledger.record_observation(
            observation_key="text:000001",
            scope_id="U0",
            decision_at=DECISION,
            feature_id="text_consensus_v1",
            target_entity_keys=["stock:000001"],
            receipt_ids=[first["event_id"], reprint["event_id"]],
            entity_ids=[entity["event_id"]],
            value=1.0,
            unit="consensus_count",
            extractor=self._rule(),
            min_independent_sources=2,
            now=DAY + " 18:01:00+08:00",
        )

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["body"]["reason"], "insufficient_independent_sources")
        coverage = self.ledger.coverage("U0", DECISION, "text_consensus_v1")
        self.assertEqual(coverage["observed"], 0)
        self.assertEqual(coverage["abstained"], 1)
        self.assertEqual(coverage["reason_counts"], {"insufficient_independent_sources": 1})

    def test_two_stories_from_one_source_family_are_not_text_consensus(self):
        self._scope()
        first = self._receipt(raw=b"source-a-1", origin_id="story-1")
        second = self._receipt(
            raw=b"source-a-2",
            source_id="publisher-a-second-feed",
            source_family="publisher-a",
            origin_id="story-2",
            first=DAY + " 17:20:00+08:00",
        )
        entity = self._entity(first)
        result = self.ledger.record_observation(
            observation_key="same-family",
            scope_id="U0",
            decision_at=DECISION,
            feature_id="text_consensus_v1",
            target_entity_keys=["stock:000001"],
            receipt_ids=[first["event_id"], second["event_id"]],
            entity_ids=[entity["event_id"]],
            value=1.0,
            unit="consensus_count",
            extractor=self._rule(),
            min_independent_sources=2,
            now=DAY + " 18:01:00+08:00",
        )
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["body"]["reason"], "insufficient_independent_sources")

    def test_identical_captured_content_cannot_be_relabelled_as_two_text_sources(self):
        """A generic multi-source observation must not bypass TextConsensus.

        ``record_observation`` is the ledger-facing API used to persist a
        consensus feature.  Different hand-entered source identities do not
        make two byte-identical captures independent evidence.
        """
        self._scope()
        first = self._receipt(raw=b"same wire story", origin_id="wire:1")
        relabelled = self._receipt(
            raw=b"same wire story",
            source_id="publisher-b",
            source_family="publisher-b",
            origin_id="publisher-b:99",
            first=DAY + " 17:20:00+08:00",
            now=DAY + " 17:20:00+08:00",
        )
        entity = self._entity(first)
        result = self.ledger.record_observation(
            observation_key="same-content-different-identities",
            scope_id="U0",
            decision_at=DECISION,
            feature_id="text_consensus_v1",
            target_entity_keys=["stock:000001"],
            receipt_ids=[first["event_id"], relabelled["event_id"]],
            entity_ids=[entity["event_id"]],
            value=1.0,
            unit="consensus_count",
            extractor=self._rule(),
            min_independent_sources=2,
            now=DAY + " 18:01:00+08:00",
        )

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["body"]["reason"], "duplicate_content_evidence")
        self.assertIsNone(result["body"]["value"])

    def test_late_and_ambiguous_inputs_abstain_without_a_zero_value(self):
        self._scope(("stock:000001", "stock:000002"))
        late = self._receipt(
            source_id="late-source",
            source_family="late-source",
            origin_id="late-source-1",
            first=DAY + " 18:01:00+08:00",
            available=DAY + " 18:01:00+08:00",
            now=DAY + " 18:01:00+08:00",
        )
        late_entity = self._entity(late)
        late_result = self._observe(late, late_entity, key="late", now=DAY + " 18:02:00+08:00")
        self.assertEqual(late_result["status"], "abstained")
        self.assertEqual(late_result["body"]["reason"], "late_source_receipt")
        self.assertIsNone(late_result["body"]["value"])

        early = self._receipt(
            source_id="ambiguous-source",
            source_family="ambiguous-source",
            origin_id="ambiguous-source-1",
            raw=b"ambiguous entity",
        )
        ambiguous = self._entity(early, key="stock:000002", status="abstained")
        ambiguous_result = self._observe(
            early,
            ambiguous,
            key="ambiguous",
            target=("stock:000002",),
        )
        self.assertEqual(ambiguous_result["status"], "abstained")
        self.assertEqual(ambiguous_result["body"]["reason"], "ambiguous_entity")
        self.assertIsNone(ambiguous_result["body"]["value"])

    def test_resolved_entity_mapping_must_be_linked_to_this_observations_receipts(self):
        self._scope()
        signal_receipt = self._receipt()
        mapping_receipt = self._receipt(
            raw=b"separate mapping proof",
            source_id="mapping-source",
            source_family="mapping-source",
            origin_id="mapping-origin",
        )
        entity = self._entity(mapping_receipt)

        result = self._observe(signal_receipt, entity, key="mapping-not-linked")

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["body"]["reason"], "entity_mapping_receipt_not_linked")
        self.assertIsNone(result["body"]["value"])

    def test_resolved_entity_mapping_must_match_every_observed_target_key(self):
        self._scope(("stock:000001", "stock:000002", "stock:000003"))
        receipt = self._receipt()
        wrong_entity = self._entity(receipt, key="stock:000002")

        mismatched = self._observe(
            receipt,
            wrong_entity,
            key="mapping-target-mismatch",
            target=("stock:000003",),
        )
        self.assertEqual(mismatched["status"], "abstained")
        self.assertEqual(mismatched["body"]["reason"], "entity_mapping_target_mismatch")
        self.assertIsNone(mismatched["body"]["value"])

        correct_entity = self._entity(receipt, key="stock:000001")
        partial = self.ledger.record_observation(
            observation_key="mapping-target-partial",
            scope_id="U0",
            decision_at=DECISION,
            feature_id="theme_carrier_minimal_v1",
            target_entity_keys=["stock:000001", "stock:000002"],
            receipt_ids=[receipt["event_id"]],
            entity_ids=[correct_entity["event_id"]],
            value=0.25,
            unit="percentile",
            extractor=self._rule(),
            now=DAY + " 18:01:00+08:00",
        )
        self.assertEqual(partial["status"], "abstained")
        self.assertEqual(partial["body"]["reason"], "entity_mapping_target_mismatch")
        self.assertIsNone(partial["body"]["value"])

    def test_coverage_reports_observed_abstained_and_missing_members(self):
        self._scope(("stock:000001", "stock:000002", "stock:000003"))
        first = self._receipt()
        entity = self._entity(first)
        self._observe(first, entity, key="observed")
        self.ledger.abstain(
            observation_key="missing-source",
            scope_id="U0",
            decision_at=DECISION,
            feature_id="theme_carrier_minimal_v1",
            target_entity_keys=["stock:000002"],
            stage="receipt",
            reason="missing_source_provenance",
            now=DAY + " 18:01:00+08:00",
        )

        coverage = self.ledger.coverage("U0", DECISION, "theme_carrier_minimal_v1")
        self.assertEqual(coverage["expected"], 3)
        self.assertEqual(coverage["observed"], 1)
        self.assertEqual(coverage["abstained"], 1)
        self.assertEqual(coverage["missing"], ["stock:000003"])
        self.assertEqual(coverage["reason_counts"], {"missing_source_provenance": 1})

    def test_historical_backfill_and_conflicting_terminal_observations_are_rejected(self):
        self._scope()
        receipt = self._receipt()
        entity = self._entity(receipt)
        first = self._observe(receipt, entity, key="immutable")
        repeat = self._observe(receipt, entity, key="immutable")
        self.assertEqual(first["event_id"], repeat["event_id"])
        with self.assertRaisesRegex(ValueError, "immutable_event_conflict"):
            self.ledger.record_observation(
                observation_key="immutable",
                scope_id="U0",
                decision_at=DECISION,
                feature_id="theme_carrier_minimal_v1",
                target_entity_keys=["stock:000001"],
                receipt_ids=[receipt["event_id"]],
                entity_ids=[entity["event_id"]],
                value=0.9,
                unit="percentile",
                extractor=self._rule(),
                now=DAY + " 18:01:00+08:00",
            )
        with self.assertRaisesRegex(ValueError, "historical_observation_forbidden"):
            self._observe(receipt, entity, key="tomorrow", now="2026-01-06 18:01:00+08:00")

    def test_outcomes_need_maturity_and_feedback_only_returns_settled_rows(self):
        self._scope()
        signal_receipt = self._receipt()
        entity = self._entity(signal_receipt)
        observation = self._observe(signal_receipt, entity)
        outcome_receipt = self._receipt(
            source_id="price-source",
            source_family="price-source",
            origin_id="price-20260112",
            raw=b"outcome price",
            first="2026-01-12 10:00:00+08:00",
            available="2026-01-12 10:00:00+08:00",
            now="2026-01-12 10:00:00+08:00",
        )
        args = dict(
            observation_id=observation["event_id"],
            outcome_key="open_to_open_h5",
            label_definition={"version": "t1-open-to-t6-open-v1"},
            entry_at="2026-01-06 09:30:00+08:00",
            exit_at="2026-01-12 09:30:00+08:00",
            available_at="2026-01-12 10:00:00+08:00",
            value=0.12,
            receipt_ids=[outcome_receipt["event_id"]],
        )
        with self.assertRaisesRegex(ValueError, "outcome_not_matured"):
            self.ledger.record_outcome(**args, now="2026-01-12 09:59:00+08:00")
        outcome = self.ledger.record_outcome(**args, now="2026-01-12 10:01:00+08:00")
        self.assertEqual(outcome["status"], "settled")
        unavailable = self.ledger.record_outcome(
            observation_id=observation["event_id"],
            outcome_key="open_to_open_h20",
            label_definition={"version": "t1-open-to-t21-open-v1"},
            entry_at="2026-01-06 09:30:00+08:00",
            exit_at="2026-01-12 09:30:00+08:00",
            available_at="2026-01-12 10:00:00+08:00",
            status="unavailable",
            reason="missing_exit_quote",
            now="2026-01-12 10:01:00+08:00",
        )
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertEqual(self.ledger.settled_outcomes("2026-01-12 09:59:00+08:00"), [])
        settled = self.ledger.settled_outcomes("2026-01-12 10:01:00+08:00")
        self.assertEqual([row["event_id"] for row in settled], [outcome["event_id"]])

    def test_blob_tamper_is_detected_and_sql_update_is_rejected(self):
        receipt = self._receipt(raw=b"protected bytes")
        with sqlite3.connect(self.root / "ledger.sqlite3") as db:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append_only_events"):
                db.execute("UPDATE events SET status='forged' WHERE event_id=?", (receipt["event_id"],))
        blob = self.root / receipt["body"]["blob_path"]
        blob.write_bytes(b"tampered")
        verification = self.ledger.verify()
        self.assertFalse(verification["verified"])
        self.assertTrue(any("blob_hash_mismatch" in reason for reason in verification["reasons"]))

    def test_llm_extractor_requires_model_prompt_and_response_receipt(self):
        self._scope()
        receipt = self._receipt()
        entity = self._entity(receipt)
        with self.assertRaisesRegex(ValueError, "llm_extractor_metadata_missing"):
            self._observe(receipt, entity, key="llm", extractor={"kind": "llm", "version": "v1"})
        unlinked = self._receipt(
            raw=b"different llm response",
            source_id="llm-response-source",
            source_family="llm-response-source",
            origin_id="llm-response-origin",
        )
        with self.assertRaisesRegex(ValueError, "llm_response_receipt_not_linked"):
            self._observe(
                receipt,
                entity,
                key="llm-unlinked",
                extractor={
                    "kind": "llm",
                    "version": "v1",
                    "model_id": "hosted-model",
                    "prompt_sha256": hashlib.sha256(b"prompt").hexdigest(),
                    "response_receipt_ids": [unlinked["event_id"]],
                },
            )
        with self.assertRaisesRegex(ValueError, "invalid_extractor_prompt_sha256"):
            self._observe(
                receipt,
                entity,
                key="llm-invalid-prompt-hash",
                extractor={
                    "kind": "llm",
                    "version": "v1",
                    "model_id": "hosted-model",
                    "prompt_sha256": "not-a-sha256",
                    "response_receipt_ids": [receipt["event_id"]],
                },
            )
        result = self._observe(
            receipt,
            entity,
            key="llm-ok",
            extractor={
                "kind": "llm",
                "version": "v1",
                "model_id": "hosted-model",
                "prompt_sha256": hashlib.sha256(b"prompt").hexdigest(),
                "response_receipt_ids": [receipt["event_id"]],
            },
        )
        self.assertEqual(result["status"], "observed")


if __name__ == "__main__":
    unittest.main()
