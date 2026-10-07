"""Regression tests for the no-account corrected-band phase preflight."""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import corrected_band_phase_preflight as preflight


def _calendar(days=16):
    first = date(2026, 1, 2)
    return tuple(first + timedelta(days=index) for index in range(days))


def _artifact(label):
    return {"targets": [{"target_id": label}]}


_UNSET = object()
ARCHIVED_WRONG_ORDER_ARTIFACT = _artifact("archived-wrong-order")
ARCHIVED_WRONG_ORDER_HASH = preflight.canonical_target_artifact_sha256(
    ARCHIVED_WRONG_ORDER_ARTIFACT
)


def _reconstructor(
    artifact_id,
    identity=preflight.CORRECTED_SELECTION_IDENTITY,
    target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
    target_artifact=_UNSET,
    declared_target_hash=_UNSET,
    seen=None,
    phase=None,
):
    if target_artifact is _UNSET:
        target_artifact = _artifact(artifact_id)
    if declared_target_hash is _UNSET:
        declared_target_hash = (
            None
            if target_artifact is None
            else preflight.canonical_target_artifact_sha256(target_artifact)
        )

    def reconstruct(moves):
        if seen is not None:
            seen[phase] = tuple((move.source_date, move.scheduled_date) for move in moves)
        return preflight.PhaseReconstruction(
            target_hash=declared_target_hash,
            selection_identity=identity,
            target_hash_scheme=target_hash_scheme,
            target_artifact=target_artifact,
        )

    return reconstruct


class CorrectedBandPhasePreflightTests(unittest.TestCase):
    def test_canonical_target_artifact_hash_is_independent_of_mapping_key_order(self):
        first = {
            "targets": [{"symbol": "000001", "weight": 0.5}],
            "selection_identity": ["-financial_score", "liquidity_rank"],
        }
        same_content_different_order = {
            "selection_identity": ["-financial_score", "liquidity_rank"],
            "targets": [{"weight": 0.5, "symbol": "000001"}],
        }

        self.assertEqual(
            preflight.canonical_target_artifact_sha256(first),
            preflight.canonical_target_artifact_sha256(same_content_different_order),
        )

    def test_blocks_when_archived_target_artifact_is_absent(self):
        calendar = _calendar()
        phase_artifact = _artifact("rebuilt-phase-0")
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1],),
            {0: _reconstructor("rebuilt-phase-0", target_artifact=phase_artifact)},
            archived_target_hash=preflight.canonical_target_artifact_sha256(phase_artifact),
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=None,
            phases=(0,),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("archived_target_artifact_required", result.verdict.reasons)
        self.assertFalse(result.verdict.account_replay_started)

    def test_blocks_when_phase_target_artifact_is_absent(self):
        calendar = _calendar()
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1],),
            {
                0: _reconstructor(
                    "rebuilt-phase-0",
                    target_artifact=None,
                    declared_target_hash=None,
                ),
            },
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0,),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("phase0_target_artifact_required", result.verdict.reasons)
        self.assertFalse(result.verdict.account_replay_started)

    def test_phase_zero_compares_computed_hashes_not_caller_declared_hashes(self):
        calendar = _calendar()
        artifact = {"targets": [{"signal_date": "2026-01-03", "weights": {"A": 0.9}}]}
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1],),
            {
                0: lambda _moves: preflight.PhaseReconstruction(
                    target_hash="payload-serialization-hash",
                    selection_identity=preflight.CORRECTED_SELECTION_IDENTITY,
                    target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
                    target_artifact=artifact,
                ),
            },
            archived_target_hash="legacy-file-bytes-hash",
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=artifact,
            phases=(0,),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("phase0_matches_archived_target_hash", result.verdict.reasons)
        self.assertIn("archived_declared_target_hash_mismatch", result.verdict.reasons)
        self.assertIn("phase0_declared_target_hash_mismatch", result.verdict.reasons)

    def test_blocks_phase_zero_when_its_target_hash_scheme_differs_from_the_archived_artifact(self):
        calendar = _calendar()
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1], calendar[4]),
            {
                0: _reconstructor(
                    "archived-wrong-order",
                    target_hash_scheme="sha256:file-bytes-targets-json-v1",
                ),
            },
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0,),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("phase0_target_hash_scheme_mismatch", result.verdict.reasons)
        self.assertFalse(result.verdict.account_replay_started)

    def test_blocks_legacy_call_without_an_archived_target_hash_scheme(self):
        calendar = _calendar()
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1], calendar[4]),
            {0: _reconstructor("archived-wrong-order")},
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0,),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("archived_target_hash_scheme_required", result.verdict.reasons)
        self.assertNotIn("phase0_matches_archived_target_hash", result.verdict.reasons)

    def test_blocks_phase_zero_without_a_declared_target_hash_scheme(self):
        calendar = _calendar()
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1], calendar[4]),
            {0: _reconstructor("archived-wrong-order", target_hash_scheme=None)},
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0,),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("phase0_target_hash_scheme_required", result.verdict.reasons)
        self.assertNotIn("phase0_matches_archived_target_hash", result.verdict.reasons)

    def test_blocks_matching_noncanonical_legacy_hash_schemes(self):
        calendar = _calendar()
        legacy_scheme = "sha256:file-bytes-targets-json-v1"
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1], calendar[4]),
            {0: _reconstructor("archived-wrong-order", target_hash_scheme=legacy_scheme)},
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=legacy_scheme,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0,),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("archived_target_hash_scheme_not_canonical", result.verdict.reasons)
        self.assertIn("phase0_target_hash_scheme_not_canonical", result.verdict.reasons)
        self.assertNotIn("phase0_matches_archived_target_hash", result.verdict.reasons)

    def test_default_phase_set_reconstructs_every_nonzero_phase_from_the_same_sources(self):
        calendar = _calendar(days=30)
        source_dates = (calendar[2], calendar[5])
        result = preflight.preflight_phase_schedule(
            calendar,
            source_dates,
            {phase: _reconstructor(f"rebuilt-phase-{phase}") for phase in preflight.DEFAULT_PHASES},
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
        )

        self.assertEqual(result.verdict.status, "ready")
        self.assertEqual(tuple(trace.phase for trace in result.traces), preflight.DEFAULT_PHASES)
        for phase in (4, 8, 12, 16):
            self.assertEqual(
                tuple(move.scheduled_date for move in result.trace_for(phase).moves),
                tuple(calendar[calendar.index(source) + phase] for source in source_dates),
            )

    def test_moves_each_original_decision_on_the_full_calendar_not_a_sliced_schedule(self):
        calendar = _calendar()
        source_dates = (calendar[1], calendar[4], calendar[7])
        seen = {}
        result = preflight.preflight_phase_schedule(
            calendar,
            source_dates,
            {
                0: _reconstructor("rebuilt-phase-0", seen=seen, phase=0),
                4: _reconstructor("rebuilt-phase-4", seen=seen, phase=4),
            },
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0, 4),
        )

        self.assertEqual(result.verdict.status, "ready")
        self.assertEqual(
            seen[4],
            (
                (calendar[1], calendar[5]),
                (calendar[4], calendar[8]),
                (calendar[7], calendar[11]),
            ),
        )
        self.assertEqual(
            tuple(move.scheduled_date for move in result.trace_for(4).moves),
            (calendar[5], calendar[8], calendar[11]),
        )

    def test_rejects_a_phase_that_reuses_another_recomputed_target_hash(self):
        calendar = _calendar()
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1], calendar[4]),
            {
                0: _reconstructor("rebuilt-targets"),
                4: _reconstructor("rebuilt-targets"),
            },
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0, 4),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("duplicate_recomputed_target_hash:phase4:phase0", result.verdict.reasons)
        self.assertFalse(result.verdict.account_replay_started)

    def test_records_tail_overflow_without_silently_dropping_its_source_decision(self):
        calendar = _calendar(days=7)
        seen = {}
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1], calendar[6]),
            {
                0: _reconstructor("phase-0", seen=seen, phase=0),
                4: _reconstructor("phase-4", seen=seen, phase=4),
            },
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0, 4),
        )

        phase4 = result.trace_for(4)
        self.assertEqual(result.verdict.status, "ready")
        self.assertEqual(phase4.tail_overflow_count, 1)
        self.assertEqual(phase4.moves[-1].source_date, calendar[6])
        self.assertIsNone(phase4.moves[-1].scheduled_date)
        self.assertEqual(phase4.moves[-1].omission_reason, "tail_overflow")
        self.assertEqual(seen[4], ((calendar[1], calendar[5]),))

    def test_phase_zero_requires_the_corrected_financial_then_liquidity_identity(self):
        calendar = _calendar()
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1], calendar[4]),
            {
                0: _reconstructor("rebuilt-phase-0", identity=("-financial_score", "input_order")),
                4: _reconstructor("rebuilt-phase-4"),
            },
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0, 4),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("phase0_selection_identity_mismatch", result.verdict.reasons)

    def test_phase_zero_must_not_match_the_archived_wrong_order_target_hash(self):
        calendar = _calendar()
        result = preflight.preflight_phase_schedule(
            calendar,
            (calendar[1], calendar[4]),
            {
                0: _reconstructor("archived-wrong-order"),
                4: _reconstructor("rebuilt-phase-4"),
            },
            archived_target_hash=ARCHIVED_WRONG_ORDER_HASH,
            archived_target_hash_scheme=preflight.CANONICAL_TARGET_ARTIFACT_HASH_SCHEME,
            archived_target_artifact=ARCHIVED_WRONG_ORDER_ARTIFACT,
            phases=(0, 4),
        )

        self.assertEqual(result.verdict.status, "blocked")
        self.assertIn("phase0_matches_archived_target_hash", result.verdict.reasons)


if __name__ == "__main__":
    unittest.main()
