import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import (
    LEGACY_REFERENCE_CONTRACT_VERSION,
    R2_POOL_MEMBERSHIP_REFERENCE,
    LiquidityPoolSpec,
    U0,
    U1,
    U_BROAD,
    build_liquidity_pool,
    canonical_forward_input_sha256,
    due_sessions,
    forward_membership_sha256,
    legacy_reference_snapshot,
)


def rankable_rows(receipt_evidence, count=6002):
    receipt = receipt_evidence["receipt"]
    receipt_body = receipt["body"]
    raw = receipt_evidence["raw"]
    rows = []
    for index in range(count):
        rows.append(
            {
                "instrument_id": f"{index:06d}.SZ",
                "base_eligible": True,
                "adv60_cny": float(10_000_000 - index),
                "adv_observation_count": 60,
                "valid_price_session_count": 253,
                "asof_ok": True,
                "source_receipt_id": receipt_evidence["canonical_receipt_id"],
                "source_receipt_hash": raw["sha256"],
                "source_published_at": receipt_body["published_at"],
                "source_first_captured_at": receipt_body["first_captured_at"],
                "source_available_at": receipt["available_at"],
                "capture_mode": "forward_receipt",
            }
        )
    return pd.DataFrame(rows)


def canonical_input_hash(rows):
    return canonical_forward_input_sha256(rows)


class ResearchUniverseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # The ledger owns write time.  This fixture controls that clock rather
        # than passing a caller-controlled `now` value to manufacture an
        # historical receipt.
        self._ledger_now = "2026-08-31 17:10:00+08:00"
        self.ledger = ResearchSignalLedger(
            Path(self.temp.name) / "ledger", now_fn=lambda: self._ledger_now
        )
        receipt = self.ledger.record_receipt(
            source_id="forward-market-feed",
            source_family="forward-market-feed",
            origin_id="u0-batch-20260831",
            canonical_url="https://example.test/u0-batch-20260831",
            raw=b'{"batch":"u0-rank-input-20260831"}',
            content_type="application/json",
            published_at="2026-08-31 16:30:00+08:00",
            first_captured_at="2026-08-31 17:00:00+08:00",
            available_at="2026-08-31 17:00:00+08:00",
            availability_basis="first_capture",
            now="2026-08-31 17:10:00+08:00",
        )
        self.receipt_evidence = self.ledger.export_receipt_evidence(receipt["event_id"])

    def tearDown(self):
        self.temp.cleanup()

    def _rows(self, count=6002):
        return rankable_rows(self.receipt_evidence, count)

    def test_r2_protocol_is_metadata_only_and_not_a_local_pool_input(self):
        self.assertEqual(
            R2_POOL_MEMBERSHIP_REFERENCE.sha256,
            "b3653299fbe4f00873d038e0d23448ca378a1566aff1bf97a687e649f84c09f4",
        )
        self.assertTrue(R2_POOL_MEMBERSHIP_REFERENCE.external_reference_only)
        self.assertIn("pool_membership_effect_20260929_r2", R2_POOL_MEMBERSHIP_REFERENCE.protocol_path)

    def test_u0_u1_and_broad_have_exact_rank_boundaries(self):
        calendar = pd.bdate_range("2026-08-03", periods=21)
        rows = self._rows()
        common = dict(
            session=calendar[-1],
            decision_at="2026-08-31 18:00:00+08:00",
            rows=rows,
            cadence_anchor=calendar[0],
            exchange_sessions=calendar,
            input_hash=canonical_input_hash(rows),
            ledger=self.ledger,
        )

        u0 = build_liquidity_pool(spec=U0, **common)
        u1 = build_liquidity_pool(spec=U1, **common)
        broad = build_liquidity_pool(spec=U_BROAD, **common)

        self.assertEqual(u0.status, "ok")
        self.assertIn("003000.SZ", u0.members)
        self.assertIn("004999.SZ", u0.members)
        self.assertNotIn("002999.SZ", u0.members)
        self.assertNotIn("005000.SZ", u0.members)
        self.assertIn("005000.SZ", u1.members)
        self.assertEqual(set(broad.members), set(u0.members) | set(u1.members))
        self.assertFalse(set(u0.members) & set(u1.members))

    def test_only_the_three_frozen_pool_specs_are_constructible_or_usable(self):
        u0_fields = dict(
            pool_id="U0",
            rank_lower_exclusive=3000,
            rank_upper_inclusive=5000,
            adv_window_sessions=60,
            min_adv_cny=5_000_000.0,
            required_prior_price_sessions=252,
            cadence_sessions=20,
            rank_tie_break="instrument_id_asc",
            rank_before_adv_floor=True,
        )
        deviations = (
            {"pool_id": "U0_relabelled"},
            {"rank_lower_exclusive": 3001},
            {"rank_upper_inclusive": 4999},
            {"adv_window_sessions": 20},
            {"min_adv_cny": 5_000_001.0},
            {"required_prior_price_sessions": 251},
            {"cadence_sessions": 21},
            {"rank_tie_break": "market_cap_desc"},
            {"rank_before_adv_floor": False},
        )
        for deviation in deviations:
            with self.subTest(deviation=deviation):
                with self.assertRaisesRegex(ValueError, "liquidity_pool_spec_not_frozen"):
                    LiquidityPoolSpec(**{**u0_fields, **deviation})

        # Normal construction is blocked above, but public hashing/building
        # boundaries must reject a forged dataclass too rather than trusting
        # its U0 label.
        forged = object.__new__(LiquidityPoolSpec)
        for field in U0.__dataclass_fields__:
            object.__setattr__(forged, field, getattr(U0, field))
        object.__setattr__(forged, "adv_window_sessions", 20)
        input_hash = hashlib.sha256(b"forged-u0-input").hexdigest()
        with self.assertRaisesRegex(ValueError, "liquidity_pool_spec_not_frozen"):
            forward_membership_sha256(
                spec=forged,
                session="2026-08-31",
                decision_at="2026-08-31 18:00:00+08:00",
                cadence_anchor="2026-08-03",
                input_hash=input_hash,
                members=("003000.SZ",),
                status="ok",
                abstain_reason=None,
            )

        calendar = pd.bdate_range("2026-08-03", periods=21)
        rows = self._rows()
        with self.assertRaisesRegex(ValueError, "liquidity_pool_spec_not_frozen"):
            build_liquidity_pool(
                session=calendar[-1],
                decision_at="2026-08-31 18:00:00+08:00",
                rows=rows,
                spec=forged,
                cadence_anchor=calendar[0],
                exchange_sessions=calendar,
                input_hash=canonical_input_hash(rows),
                ledger=self.ledger,
            )

    def test_forward_input_and_membership_identities_are_canonical_sha256(self):
        calendar = pd.bdate_range("2026-08-03", periods=21)
        rows = self._rows()
        shuffled = rows.sample(frac=1.0, random_state=19).reset_index(drop=True)
        input_hash = canonical_input_hash(rows)
        self.assertEqual(input_hash, canonical_input_hash(shuffled))
        self.assertRegex(input_hash, r"^[0-9a-f]{64}$")
        with_extra_column = rows.assign(unrelated_diagnostic="not_part_of_the_contract")
        self.assertEqual(input_hash, canonical_input_hash(with_extra_column))
        changed_input = rows.copy()
        changed_input.loc[0, "valid_price_session_count"] = 254
        self.assertNotEqual(input_hash, canonical_input_hash(changed_input))

        common = dict(
            session=calendar[-1],
            decision_at="2026-08-31 18:00:00+08:00",
            spec=U0,
            cadence_anchor=calendar[0],
            exchange_sessions=calendar,
            ledger=self.ledger,
        )
        snapshot = build_liquidity_pool(rows=rows, input_hash=input_hash, **common)
        reordered = build_liquidity_pool(rows=shuffled, input_hash=canonical_input_hash(shuffled), **common)
        self.assertEqual(snapshot.status, "ok")
        self.assertEqual(snapshot.input_hash, input_hash)
        self.assertRegex(snapshot.membership_hash, r"^[0-9a-f]{64}$")
        self.assertEqual(snapshot.membership_hash, reordered.membership_hash)
        self.assertEqual(
            snapshot.membership_hash,
            forward_membership_sha256(
                spec=U0,
                session=calendar[-1],
                decision_at="2026-08-31 18:00:00+08:00",
                cadence_anchor=calendar[0],
                input_hash=input_hash,
                members=snapshot.members,
                status="ok",
                abstain_reason=None,
            ),
        )

        changed_snapshot = build_liquidity_pool(
            rows=changed_input,
            input_hash=canonical_input_hash(changed_input),
            **common,
        )
        self.assertEqual(snapshot.members, changed_snapshot.members)
        self.assertNotEqual(snapshot.membership_hash, changed_snapshot.membership_hash)

        with self.assertRaisesRegex(ValueError, "input_hash_does_not_match_canonical_required_rows"):
            build_liquidity_pool(
                rows=rows,
                input_hash=hashlib.sha256(b"unrelated-input").hexdigest(),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "input_hash_must_be_sha256"):
            build_liquidity_pool(rows=rows, input_hash="input-sha", **common)

    def test_adv_floor_applies_after_rank_without_promoting_rank_5001(self):
        calendar = pd.bdate_range("2026-08-03", periods=21)
        rows = self._rows(5002)
        # A name inside U0 fails the ADV floor. Rank 5001 remains outside
        # U0; the band must not backfill it after the floor removes names.
        rows.loc[3000:, "adv60_cny"] = [5_000_999.0 - offset for offset in range(len(rows.loc[3000:]))]
        shuffled = rows.sample(frac=1.0, random_state=7)
        snapshot = build_liquidity_pool(
            session=calendar[-1],
            decision_at="2026-08-31 18:00:00+08:00",
            rows=shuffled,
            spec=U0,
            cadence_anchor=calendar[0],
            exchange_sessions=calendar,
            input_hash=canonical_input_hash(shuffled),
            ledger=self.ledger,
        )

        self.assertEqual(snapshot.status, "ok")
        self.assertIn("003000.SZ", snapshot.members)
        self.assertNotIn("004000.SZ", snapshot.members)
        self.assertNotIn("005000.SZ", snapshot.members)
        self.assertEqual(snapshot.eligibility_reason_by_instrument["004000.SZ"], "below_adv_floor_after_rank")

    def test_non_due_and_incomplete_rank_inputs_abstain_fail_closed(self):
        calendar = pd.bdate_range("2026-08-03", periods=21)
        rows = self._rows()
        nondue = build_liquidity_pool(
            session=calendar[-2], decision_at="2026-08-28 18:00:00+08:00", rows=rows,
            spec=U0, cadence_anchor=calendar[0], exchange_sessions=calendar, input_hash=canonical_input_hash(rows),
            ledger=self.ledger,
        )
        self.assertEqual((nondue.status, nondue.abstain_reason), ("abstain", "not_due_session"))

        rows.loc[0, "source_receipt_hash"] = ""
        incomplete = build_liquidity_pool(
            session=calendar[-1], decision_at="2026-08-31 18:00:00+08:00", rows=rows,
            spec=U0, cadence_anchor=calendar[0], exchange_sessions=calendar, input_hash=canonical_input_hash(rows),
            ledger=self.ledger,
        )
        self.assertEqual(incomplete.status, "abstain")
        self.assertEqual(incomplete.abstain_reason, "incomplete_or_nonforward_rank_input")

        rows = self._rows()
        rows.loc[0, "source_receipt_hash"] = "not-a-content-hash"
        invalid_hash = build_liquidity_pool(
            session=calendar[-1], decision_at="2026-08-31 18:00:00+08:00", rows=rows,
            spec=U0, cadence_anchor=calendar[0], exchange_sessions=calendar, input_hash=canonical_input_hash(rows),
            ledger=self.ledger,
        )
        self.assertEqual((invalid_hash.status, invalid_hash.abstain_reason),
                         ("abstain", "incomplete_or_nonforward_rank_input"))

    def test_forward_pool_rows_must_match_canonical_ledger_receipt_evidence(self):
        calendar = pd.bdate_range("2026-08-03", periods=21)
        common = dict(
            session=calendar[-1],
            decision_at="2026-08-31 18:00:00+08:00",
            spec=U0,
            cadence_anchor=calendar[0],
            exchange_sessions=calendar,
            ledger=self.ledger,
        )

        def build(rows):
            return build_liquidity_pool(rows=rows, input_hash=canonical_input_hash(rows), **common)

        forged = self._rows(3001)
        forged.loc[0, "source_receipt_hash"] = hashlib.sha256(b"forged").hexdigest()
        forged_snapshot = build(forged)
        self.assertEqual(
            (forged_snapshot.status, forged_snapshot.abstain_reason),
            ("abstain", "incomplete_or_nonforward_rank_input"),
        )
        self.assertEqual(
            forged_snapshot.eligibility_reason_by_instrument["000000.SZ"],
            "ledger_receipt_evidence_mismatch",
        )

        for column, value in (
            ("source_published_at", "2026-08-31 16:31:00+08:00"),
            ("source_first_captured_at", "2026-08-31 16:59:00+08:00"),
            ("source_available_at", "2026-08-31 17:01:00+08:00"),
        ):
            with self.subTest(column=column):
                mismatched_time = self._rows(3001)
                mismatched_time.loc[0, column] = value
                snapshot = build(mismatched_time)
                self.assertEqual(
                    snapshot.eligibility_reason_by_instrument["000000.SZ"],
                    "ledger_receipt_evidence_mismatch",
                )

        duplicate = self.ledger.record_receipt(
            source_id="forward-market-feed",
            source_family="forward-market-feed",
            origin_id="u0-batch-20260831",
            canonical_url="https://example.test/u0-batch-20260831",
            raw=b'{"batch":"u0-rank-input-20260831"}',
            content_type="application/json",
            published_at="2026-08-31 16:30:00+08:00",
            first_captured_at="2026-08-31 17:00:00+08:00",
            available_at="2026-08-31 17:00:00+08:00",
            availability_basis="first_capture",
            now="2026-08-31 17:20:00+08:00",
        )
        self.assertEqual(duplicate["status"], "duplicate")
        alias_rows = self._rows(3001)
        alias_rows.loc[0, "source_receipt_id"] = duplicate["event_id"]
        alias_snapshot = build(alias_rows)
        self.assertEqual(
            alias_snapshot.eligibility_reason_by_instrument["000000.SZ"],
            "ledger_receipt_evidence_mismatch",
        )

    def test_ledger_is_mandatory_cached_and_must_have_existed_by_cutoff(self):
        calendar = pd.bdate_range("2026-08-03", periods=21)
        rows = self._rows(3001)
        common = dict(
            session=calendar[-1],
            decision_at="2026-08-31 18:00:00+08:00",
            rows=rows,
            spec=U0,
            cadence_anchor=calendar[0],
            exchange_sessions=calendar,
            input_hash=canonical_input_hash(rows),
        )
        with self.assertRaisesRegex(TypeError, "forward_pool_ledger_required"):
            build_liquidity_pool(ledger=None, **common)

        calls = []
        export = self.ledger.export_receipt_evidence

        def counted_export(receipt_id):
            calls.append(receipt_id)
            return export(receipt_id)

        self.ledger.export_receipt_evidence = counted_export
        try:
            cached = build_liquidity_pool(ledger=self.ledger, **common)
        finally:
            del self.ledger.export_receipt_evidence
        self.assertEqual(cached.status, "ok")
        self.assertEqual(calls, [self.receipt_evidence["canonical_receipt_id"]])

        self._ledger_now = "2026-08-31 18:01:00+08:00"
        late = self.ledger.record_receipt(
            source_id="late-ledger-market-feed",
            source_family="late-ledger-market-feed",
            origin_id="u0-batch-late-record",
            canonical_url="https://example.test/u0-batch-late-record",
            raw=b'{"batch":"u0-rank-input-late-record"}',
            content_type="application/json",
            published_at="2026-08-31 16:30:00+08:00",
            first_captured_at="2026-08-31 17:00:00+08:00",
            available_at="2026-08-31 17:00:00+08:00",
            availability_basis="first_capture",
            now="2026-08-31 18:01:00+08:00",
        )
        late_evidence = self.ledger.export_receipt_evidence(late["event_id"])
        late_rows = self._rows(3001)
        late_rows.loc[:, "source_receipt_id"] = late_evidence["canonical_receipt_id"]
        late_rows.loc[:, "source_receipt_hash"] = late_evidence["raw"]["sha256"]
        late_rows.loc[:, "source_published_at"] = late_evidence["receipt"]["body"]["published_at"]
        late_rows.loc[:, "source_first_captured_at"] = late_evidence["receipt"]["body"]["first_captured_at"]
        late_rows.loc[:, "source_available_at"] = late_evidence["receipt"]["available_at"]
        late_snapshot = build_liquidity_pool(
            ledger=self.ledger,
            rows=late_rows,
            input_hash=canonical_input_hash(late_rows),
            **{key: value for key, value in common.items() if key not in {"rows", "input_hash"}},
        )
        self.assertEqual(
            late_snapshot.eligibility_reason_by_instrument["000000.SZ"],
            "ledger_receipt_evidence_mismatch",
        )

    def test_decision_cutoff_must_be_this_session_at_1800_shanghai(self):
        calendar = pd.bdate_range("2026-08-03", periods=21)
        rows = self._rows()
        common = dict(
            session=calendar[-1],
            rows=rows,
            spec=U0,
            cadence_anchor=calendar[0],
            exchange_sessions=calendar,
            input_hash=canonical_input_hash(rows),
            ledger=self.ledger,
        )
        for invalid_cutoff in (
            "2026-08-31 17:59:00+08:00",
            "2026-08-31 18:01:00+08:00",
            "2026-09-01 18:00:00+08:00",
        ):
            with self.subTest(invalid_cutoff=invalid_cutoff):
                snapshot = build_liquidity_pool(decision_at=invalid_cutoff, **common)
                self.assertEqual(
                    (snapshot.status, snapshot.abstain_reason),
                    ("abstain", "decision_at_must_equal_session_1800_shanghai"),
                )

    def test_shanghai_dated_sessions_remain_on_the_same_local_exchange_date(self):
        calendar = pd.bdate_range("2026-08-03", periods=21, tz="Asia/Shanghai")
        rows = self._rows()
        snapshot = build_liquidity_pool(
            session=calendar[-1],
            decision_at="2026-08-31 18:00:00+08:00",
            rows=rows,
            spec=U0,
            cadence_anchor=calendar[0],
            exchange_sessions=calendar,
            input_hash=canonical_input_hash(rows),
            ledger=self.ledger,
        )
        self.assertEqual(snapshot.status, "ok")
        self.assertEqual(snapshot.session.isoformat(), "2026-08-31")

    def test_invalid_history_counter_is_excluded_with_an_explicit_reason(self):
        calendar = pd.bdate_range("2026-08-03", periods=21)
        rows = self._rows()
        rows.loc[0, "adv_observation_count"] = float("nan")
        snapshot = build_liquidity_pool(
            session=calendar[-1], decision_at="2026-08-31 18:00:00+08:00", rows=rows,
            spec=U0, cadence_anchor=calendar[0], exchange_sessions=calendar, input_hash=canonical_input_hash(rows),
            ledger=self.ledger,
        )
        self.assertEqual(snapshot.status, "ok")
        self.assertEqual(snapshot.eligibility_reason_by_instrument["000000.SZ"], "invalid_adv_observation_count")
        self.assertNotIn("000000.SZ", snapshot.rank_by_instrument)

    def test_exchange_session_cadence_and_legacy_membership_are_explicitly_separate(self):
        calendar = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-07"])
        self.assertEqual(due_sessions(calendar, calendar[0], step=2), (calendar[0], calendar[2]))
        legacy = legacy_reference_snapshot(
            session=calendar[2],
            cadence_anchor=calendar[0],
            members=["000002.SZ", "000001.SZ"],
            membership_hash=hashlib.sha256(b"archived-heads-membership").hexdigest(),
        )
        self.assertEqual(legacy.contract_version, LEGACY_REFERENCE_CONTRACT_VERSION)
        self.assertEqual(legacy.members, ("000001.SZ", "000002.SZ"))
        with self.assertRaisesRegex(ValueError, "requires_forward_contract"):
            build_liquidity_pool(
                session=calendar[2], decision_at="2026-08-05 18:00:00+08:00", rows=self._rows(),
                spec=U0, cadence_anchor=calendar[0], exchange_sessions=calendar,
                input_hash=canonical_input_hash(self._rows()), ledger=self.ledger,
                contract_version=LEGACY_REFERENCE_CONTRACT_VERSION,
            )


if __name__ == "__main__":
    unittest.main()
