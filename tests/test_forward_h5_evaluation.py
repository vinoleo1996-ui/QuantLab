"""Forward-only H=5 information-label contract tests.

These tests deliberately construct ledger receipts on their actual sessions.
They never invoke an account, a position policy, or an annualisation helper.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quantlab.forward_h5_evaluation import (
    H5BlockSchedule,
    H5_OUTCOME_KEY,
    h5_signal_identity_for_observation,
    settle_h5_open_to_open,
    summarize_h5_information_blocks,
)
from quantlab.research_signal_ledger import ResearchSignalLedger


MEMBERS = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")
SESSIONS = (
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
    "2026-01-09",
    "2026-01-12",
    "2026-01-13",
    "2026-01-14",
    "2026-01-15",
    "2026-01-16",
    "2026-01-19",
    "2026-01-20",
)


class ForwardH5EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = pd.Timestamp("2026-01-05 17:00:00+08:00")
        self.ledger = ResearchSignalLedger(
            Path(self.temp.name) / "ledger", now_fn=lambda: self.now
        )
        self.schedule = H5BlockSchedule(
            anchor_session=SESSIONS[0], trading_sessions=SESSIONS
        )

    def tearDown(self):
        self.temp.cleanup()

    def _at(self, value: str) -> None:
        self.now = pd.Timestamp(value)

    @staticmethod
    def _decision_at(session: str) -> str:
        return session + " 18:00:00+08:00"

    def _receipt(self, *, session: str, label: str, hour: str = "16:00:00") -> dict:
        capture = session + " " + hour + "+08:00"
        self._at(capture)
        return self.ledger.record_receipt(
            source_id="qualified-price-trial",
            source_family="qualified-price-trial",
            origin_id=label,
            canonical_url="https://provider.example.test/" + label,
            raw=("raw:" + label).encode("utf-8"),
            content_type="application/json",
            published_at=capture,
            first_captured_at=capture,
            available_at=capture,
            availability_basis="qualified_trial_capture",
        )

    def _observation(
        self,
        *,
        session: str = SESSIONS[0],
        feature_id: str = "theme_carrier_graph_v1",
        signal_version: str = "theme_carrier_graph_v1",
        primary_endpoint: str = "theme_score_equal_weight_graph_v1",
        members: tuple[str, ...] = MEMBERS,
    ) -> dict:
        decision_at = self._decision_at(session)
        self._at(session + " 17:20:00+08:00")
        self.ledger.declare_scope(
            "U0",
            decision_at,
            MEMBERS,
            definition_version="forward-u0-v1",
            source_hash=hashlib.sha256(("scope:" + session).encode()).hexdigest(),
        )
        source = self._receipt(session=session, label="score-" + session, hour="17:25:00")
        entities = []
        for code in members:
            self._at(session + " 17:30:00+08:00")
            entities.append(
                self.ledger.record_entity(
                    source["event_id"],
                    entity_type="stock_theme_score",
                    canonical_id=code,
                    display_name=code,
                    mapping_method="qualified_theme_graph",
                    mapping_version="theme_carrier_graph_v1",
                )["event_id"]
            )
        self._at(session + " 17:45:00+08:00")
        return self.ledger.record_observation(
            observation_key=feature_id + ":" + session,
            scope_id="U0",
            decision_at=decision_at,
            feature_id=feature_id,
            target_entity_keys=members,
            receipt_ids=[source["event_id"]],
            entity_ids=entities,
            value={
                "schema": "quantlab.theme-carrier-ledger-observation.v2",
                "signal_version": signal_version,
                "primary_endpoint": primary_endpoint,
                "score_by_instrument": {
                    code: score
                    for code, score in {
                        "000001.SZ": 0.1,
                        "000002.SZ": 0.2,
                        "000003.SZ": 0.3,
                        "000004.SZ": 0.4,
                    }.items()
                    if code in members
                },
            },
            unit="theme_score_equal_weight_v1",
            extractor={
                "kind": "rule",
                "version": signal_version,
                "code_hash": "fixture",
            },
        )

    def _open_rows(self, *, session: str, receipt: dict, opens: dict[str, float]) -> pd.DataFrame:
        stamp = session + " 16:00:00+08:00"
        return pd.DataFrame(
            [
                {
                    "session": session,
                    "instrument_id": code,
                    "open": value,
                    "quality_status": "ok",
                    "published_at": stamp,
                    "first_captured_at": stamp,
                    "available_at": stamp,
                    "source_receipt_id": receipt["event_id"],
                    "content_sha256": receipt["body"]["raw_sha256"],
                    "capture_mode": "forward_receipt",
                }
                for code, value in opens.items()
            ]
        )

    def _complete_open_evidence(self) -> pd.DataFrame:
        entry = self._receipt(session=SESSIONS[1], label="entry-open", hour="16:00:00")
        exit_ = self._receipt(session=SESSIONS[6], label="exit-open", hour="16:00:00")
        return pd.concat(
            [
                self._open_rows(
                    session=SESSIONS[1],
                    receipt=entry,
                    opens={code: 10.0 for code in MEMBERS},
                ),
                self._open_rows(
                    session=SESSIONS[6],
                    receipt=exit_,
                    opens={
                        "000001.SZ": 10.1,
                        "000002.SZ": 10.2,
                        "000003.SZ": 10.3,
                        "000004.SZ": 10.4,
                    },
                ),
            ],
            ignore_index=True,
        )

    def test_settles_only_after_t6_close_from_forward_open_receipts(self):
        observation = self._observation()
        bars = self._complete_open_evidence()

        self._at("2026-01-13 17:59:00+08:00")
        pending = settle_h5_open_to_open(
            ledger=self.ledger,
            observation_id=observation["event_id"],
            schedule=self.schedule,
            open_bars=bars,
        )
        self.assertEqual((pending.status, pending.reason), ("pending", "h5_exit_close_not_reached"))

        self._at("2026-01-13 18:05:00+08:00")
        settled = settle_h5_open_to_open(
            ledger=self.ledger,
            observation_id=observation["event_id"],
            schedule=self.schedule,
            open_bars=bars,
        )
        self.assertEqual(settled.status, "settled")
        self.assertIsNotNone(settled.outcome)
        outcome = settled.outcome["body"]
        self.assertEqual(outcome["outcome_key"], H5_OUTCOME_KEY)
        self.assertEqual(outcome["entry_at"], "2026-01-06T09:30:00+08:00")
        self.assertEqual(outcome["exit_at"], "2026-01-13T09:30:00+08:00")
        self.assertAlmostEqual(outcome["value"]["cross_sectional_spearman_rank_ic"], 1.0)
        self.assertTrue(outcome["value"]["reporting_block"])
        self.assertEqual(self.ledger.settled_outcomes("2026-01-13 18:04:00+08:00"), [])
        self.assertEqual(len(self.ledger.settled_outcomes("2026-01-13 18:05:00+08:00")), 1)

    def test_missing_or_late_price_evidence_is_a_recorded_unavailable_not_a_zero(self):
        observation = self._observation()
        self._at("2026-01-13 18:05:00+08:00")
        result = settle_h5_open_to_open(
            ledger=self.ledger,
            observation_id=observation["event_id"],
            schedule=self.schedule,
            open_bars=pd.DataFrame(),
        )
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.reason, "h5_open_bar_schema_or_coverage_invalid")
        event = result.outcome
        self.assertEqual(event["status"], "unavailable")
        self.assertIsNone(event["body"]["value"])
        self.assertEqual(event["body"]["reason"], "h5_open_bar_schema_or_coverage_invalid")
        visible = self.ledger.outcomes("2026-01-13 18:05:00+08:00", outcome_key=H5_OUTCOME_KEY)
        self.assertEqual([(row["event_id"], row["status"]) for row in visible], [(event["event_id"], "unavailable")])

    def test_block_schedule_is_fixed_non_overlapping_and_summary_never_annualizes(self):
        self.assertTrue(self.schedule.is_reporting_block(SESSIONS[0]))
        self.assertFalse(self.schedule.is_reporting_block(SESSIONS[1]))
        self.assertTrue(self.schedule.is_reporting_block(SESSIONS[5]))
        self.assertEqual(self.schedule.sessions_for_label(SESSIONS[0]), (SESSIONS[1], SESSIONS[6]))
        self.assertEqual(self.schedule.sessions_for_label(SESSIONS[5]), (SESSIONS[6], SESSIONS[11]))

        observation = self._observation()
        bars = self._complete_open_evidence()
        self._at("2026-01-13 18:05:00+08:00")
        settle_h5_open_to_open(
            ledger=self.ledger,
            observation_id=observation["event_id"],
            schedule=self.schedule,
            open_bars=bars,
        )
        summary = summarize_h5_information_blocks(
            ledger=self.ledger,
            as_of="2026-01-13 18:05:00+08:00",
            schedule=self.schedule,
            signal_identity=h5_signal_identity_for_observation(
                ledger=self.ledger, observation_id=observation["event_id"]
            ),
        )
        self.assertEqual(summary["reporting_blocks_settled"], 1)
        self.assertFalse(summary["minimum_twenty_blocks_met"])
        self.assertFalse(summary["account_execution_authorized"])
        self.assertNotIn("annualized", summary)
        self.assertNotIn("annualized_return", summary)

    def test_summary_cannot_pool_different_experts_into_one_h5_block_count(self):
        theme = self._observation()
        l2 = self._observation(
            feature_id="l2_h5_score_v1",
            signal_version="l2_h5_score_v1",
            primary_endpoint="l2_score",
        )
        bars = self._complete_open_evidence()
        self._at("2026-01-13 18:05:00+08:00")
        settle_h5_open_to_open(
            ledger=self.ledger,
            observation_id=theme["event_id"],
            schedule=self.schedule,
            open_bars=bars,
        )
        settle_h5_open_to_open(
            ledger=self.ledger,
            observation_id=l2["event_id"],
            schedule=self.schedule,
            open_bars=bars,
        )

        summary = summarize_h5_information_blocks(
            ledger=self.ledger,
            as_of="2026-01-13 18:05:00+08:00",
            schedule=self.schedule,
            signal_identity=h5_signal_identity_for_observation(
                ledger=self.ledger, observation_id=theme["event_id"]
            ),
        )
        self.assertEqual(summary["reporting_blocks_settled"], 1)
        self.assertEqual(summary["daily_matured_outcomes"], 1)
        self.assertEqual(summary["signal_identity"]["feature_id"], "theme_carrier_graph_v1")

    def test_partial_scope_score_is_recorded_as_unavailable_not_a_cherry_picked_rank_ic(self):
        partial = self._observation(members=MEMBERS[:3])
        bars = self._complete_open_evidence()
        self._at("2026-01-13 18:05:00+08:00")
        result = settle_h5_open_to_open(
            ledger=self.ledger,
            observation_id=partial["event_id"],
            schedule=self.schedule,
            open_bars=bars,
        )
        self.assertEqual(
            (result.status, result.reason),
            ("unavailable", "h5_score_observation_scope_coverage_incomplete"),
        )
        self.assertIsNone(result.outcome["body"]["value"])
        self.assertFalse(result.outcome["body"]["label_definition"]["scope_coverage"]["complete"])

    def test_entry_price_backfilled_after_its_own_session_is_rejected(self):
        observation = self._observation()
        # The entry page is written only on T+6.  It may describe a historical
        # open but cannot be retroactively accepted as forward evidence.
        self._at("2026-01-13 16:00:00+08:00")
        entry = self.ledger.record_receipt(
            source_id="qualified-price-trial",
            source_family="qualified-price-trial",
            origin_id="late-entry-copy",
            canonical_url="https://provider.example.test/late-entry-copy",
            raw=b"raw:late-entry-copy",
            content_type="application/json",
            published_at="2026-01-06 16:00:00+08:00",
            first_captured_at="2026-01-06 16:00:00+08:00",
            available_at="2026-01-06 16:00:00+08:00",
            availability_basis="qualified_trial_capture",
        )
        exit_ = self._receipt(session=SESSIONS[6], label="exit-on-time", hour="16:01:00")
        bars = pd.concat(
            [
                self._open_rows(
                    session=SESSIONS[1], receipt=entry, opens={code: 10.0 for code in MEMBERS}
                ),
                self._open_rows(
                    session=SESSIONS[6], receipt=exit_, opens={code: 10.1 for code in MEMBERS}
                ),
            ],
            ignore_index=True,
        )
        # The content timestamps are genuinely historical; the receipt's
        # immutable recorded_at, not caller-supplied row values, is why it
        # remains invalid for a forward label.
        self._at("2026-01-13 18:05:00+08:00")
        result = settle_h5_open_to_open(
            ledger=self.ledger,
            observation_id=observation["event_id"],
            schedule=self.schedule,
            open_bars=bars,
        )
        self.assertEqual((result.status, result.reason), ("unavailable", "h5_open_bar_provenance_invalid"))


if __name__ == "__main__":
    unittest.main()
