import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from quantlab.research_universe import (
    FORWARD_CONTRACT_VERSION,
    U0,
    UniverseSnapshot,
    forward_membership_sha256,
)
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.theme_carrier import ThemeCarrierConfig, order_new_entries, score_theme_carrier_v1


DECISION = pd.Timestamp("2026-08-31")
PRIOR = pd.Timestamp("2026-08-28")
DECISION_AT = "2026-08-31 18:00:00+08:00"
TECH_CODES = ("A.SZ", "B.SZ", "E.SZ", "F.SZ", "G.SZ")
ENERGY_CODES = ("C.SZ", "D.SZ", "H.SZ", "I.SZ", "J.SZ")
ALL_CODES = ("A.SZ", "B.SZ", "C.SZ", "D.SZ", "E.SZ", "F.SZ", "G.SZ", "H.SZ", "I.SZ", "J.SZ")


def u0_snapshot():
    input_hash = hashlib.sha256(b"forward-u0-test-input").hexdigest()
    return UniverseSnapshot(
        contract_version=FORWARD_CONTRACT_VERSION,
        pool_id="U0",
        session=DECISION.date(),
        decision_at=pd.Timestamp(DECISION_AT),
        cadence_anchor=DECISION.date(),
        members=ALL_CODES,
        rank_by_instrument={code: index + 3001 for index, code in enumerate(ALL_CODES)},
        adv60_by_instrument={code: 6_000_000.0 for code in ALL_CODES},
        eligibility_reason_by_instrument={},
        input_hash=input_hash,
        membership_hash=forward_membership_sha256(
            spec=U0,
            session=DECISION,
            decision_at=DECISION_AT,
            cadence_anchor=DECISION,
            input_hash=input_hash,
            members=ALL_CODES,
            status="ok",
            abstain_reason=None,
        ),
        status="ok",
        abstain_reason=None,
    )


def bar(session, code, close, limit_up, high=None):
    return {
        "session": session,
        "instrument_id": code,
        "open": close,
        "high": close if high is None else high,
        "low": close,
        "close": close,
        "reference_price": close,
        "limit_up": limit_up,
        "price_tick": 0.01,
        "quality_status": "ok",
        "published_at": "2026-08-31 15:00:00+08:00",
        "first_captured_at": "2026-08-31 16:00:00+08:00",
        "available_at": "2026-08-31 17:00:00+08:00",
        "source_receipt_id": f"receipt-{session:%Y%m%d}-{code}",
        "content_sha256": hashlib.sha256(f"bar-{session:%Y%m%d}-{code}".encode()).hexdigest(),
        "capture_mode": "forward_receipt",
    }


def fixture_inputs():
    bars = [bar(PRIOR, code, 11.0, 11.0) for code in ALL_CODES]
    bars.extend(bar(DECISION, code, 10.5, 11.0, high=11.0) for code in TECH_CODES)
    bars.extend(bar(DECISION, code, 12.1, 12.1, high=12.1) for code in ENERGY_CODES)
    mappings = []
    for code, theme in tuple((code, "tech") for code in TECH_CODES) + tuple(
        (code, "energy") for code in ENERGY_CODES
    ):
        mappings.append(
            {
                "session": DECISION,
                "instrument_id": code,
                "theme_id": theme,
                "mapping_status": "active",
                "published_at": "2026-08-31 15:00:00+08:00",
                "first_captured_at": "2026-08-31 16:00:00+08:00",
                "available_at": "2026-08-31 17:00:00+08:00",
                "source_receipt_id": f"map-{code}",
                "content_sha256": hashlib.sha256(f"map-{code}".encode()).hexdigest(),
                "capture_mode": "forward_receipt",
            }
        )
    return pd.DataFrame(bars), pd.DataFrame(mappings)


class ThemeCarrierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._ledger_now = "2026-08-31 17:50:00+08:00"
        self.ledger = ResearchSignalLedger(
            Path(self.temp.name) / "ledger", now_fn=lambda: self._ledger_now
        )

    def tearDown(self):
        self.temp.cleanup()

    def inputs(self):
        # These are the canonical on-time fixture receipts.  Tests that need
        # a late receipt explicitly advance the ledger-owned clock below.
        self._ledger_now = "2026-08-31 17:50:00+08:00"
        bars, mappings = fixture_inputs()
        for kind, frame in (("bar", bars), ("map", mappings)):
            for index, row in frame.iterrows():
                if kind == "bar":
                    raw = f"bar-{pd.Timestamp(row.session):%Y%m%d}-{row.instrument_id}".encode()
                else:
                    raw = f"map-{row.instrument_id}".encode()
                receipt = self.ledger.record_receipt(
                    source_id=f"fixture-{kind}-{index}",
                    source_family=f"fixture-{kind}",
                    origin_id=f"fixture-{kind}-{index}",
                    canonical_url=f"https://example.test/{kind}/{index}",
                    raw=raw,
                    content_type="application/json",
                    published_at=row.published_at,
                    first_captured_at=row.first_captured_at,
                    available_at=row.available_at,
                    availability_basis="fixture_forward_capture",
                    now=row.first_captured_at,
                )
                frame.at[index, "source_receipt_id"] = receipt["event_id"]
                frame.at[index, "content_sha256"] = receipt["body"]["raw_sha256"]
        return bars, mappings

    def score(self, **kwargs):
        return score_theme_carrier_v1(ledger=self.ledger, **kwargs)

    def test_equal_weight_theme_endpoint_uses_only_the_four_declared_components(self):
        bars, mappings = self.inputs()
        result = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(set(result.stock_scores.index), set(u0_snapshot().members))
        components = result.theme_components.set_index("theme_id")
        endpoint_columns = [
            "prior_limit_up_premium_percentile",
            "broken_board_rate_percentile",
            "theme_relative_strength_percentile",
            "advancing_breadth_percentile",
        ]
        self.assertAlmostEqual(
            components.loc["energy", "theme_score_equal_weight_v1"],
            components.loc["energy", endpoint_columns].mean(),
        )
        self.assertGreater(result.stock_scores["C.SZ"], result.stock_scores["A.SZ"])

    def test_v1_config_is_frozen_and_average_ranks_ignore_theme_id_ties(self):
        for kwargs in (
            {"version": "theme_carrier_minimal_v2"},
            {"min_theme_members": 6},
            {"min_yesterday_limit_up": 2},
            {"min_today_touched_limit": 2},
            {"required_components": tuple(reversed(ThemeCarrierConfig().required_components))},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "theme_carrier_v1_config_is_frozen"):
                    ThemeCarrierConfig(**kwargs)

        bars, mappings = self.inputs()
        for session in (PRIOR, DECISION):
            mask = bars.session.eq(session)
            bars.loc[mask, ["open", "high", "low", "close", "reference_price", "limit_up"]] = (
                (11.0, 11.0, 11.0, 11.0, 11.0, 11.0)
                if session == PRIOR
                else (12.1, 12.1, 12.1, 12.1, 12.1, 12.1)
            )
        common = dict(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            config=ThemeCarrierConfig(),
        )
        tied = self.score(bars=bars, primary_theme_map=mappings, **common)
        renamed = mappings.copy()
        renamed["theme_id"] = renamed.theme_id.replace({"tech": "zzz", "energy": "aaa"})
        renamed_result = self.score(bars=bars, primary_theme_map=renamed, **common)
        self.assertEqual((tied.status, renamed_result.status), ("ok", "ok"))
        pd.testing.assert_series_equal(tied.stock_scores, renamed_result.stock_scores)
        self.assertTrue((tied.theme_components["theme_score_equal_weight_v1"] == 0.75).all())

    def test_late_or_nonforward_receipt_abstains_instead_of_imputing_scores(self):
        bars, mappings = self.inputs()
        bars.loc[0, "available_at"] = "2026-08-31 18:01:00+08:00"
        result = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )

        self.assertEqual((result.status, result.abstain_reason), ("abstain", "invalid_or_late_bar_provenance"))
        self.assertTrue(result.stock_scores.empty)

        bars, mappings = self.inputs()
        mappings.loc[0, "content_sha256"] = "not-a-content-hash"
        invalid_hash = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )
        self.assertEqual((invalid_hash.status, invalid_hash.abstain_reason),
                         ("abstain", "invalid_or_late_theme_provenance"))

    def test_any_non_ok_bar_quality_status_abstains_for_the_whole_decision(self):
        bars, mappings = self.inputs()
        bars.loc[0, "quality_status"] = "stale"

        result = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )

        self.assertEqual(
            (result.status, result.abstain_reason),
            ("abstain", "invalid_or_late_bar_provenance"),
        )
        self.assertTrue(result.stock_scores.empty)

    def test_valid_looking_row_hash_must_still_match_an_accepted_ledger_receipt(self):
        bars, mappings = self.inputs()
        bars.loc[0, "content_sha256"] = hashlib.sha256(b"forged-but-well-formed").hexdigest()
        result = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )
        self.assertEqual((result.status, result.abstain_reason), ("abstain", "unlinked_or_tampered_ledger_receipt"))

    def test_receipt_written_after_cutoff_abstains_even_when_its_content_times_are_early(self):
        # A receiver can attach old source timestamps to an event written only
        # after 18:00.  The accepted ledger event itself must have existed by
        # the decision cutoff, for both bar and entity/theme evidence.
        for kind in ("bar", "map"):
            with self.subTest(kind=kind):
                bars, mappings = self.inputs()
                frame = bars if kind == "bar" else mappings
                row = frame.iloc[0]
                raw = (
                    f"bar-{pd.Timestamp(row.session):%Y%m%d}-{row.instrument_id}".encode()
                    if kind == "bar"
                    else f"map-{row.instrument_id}".encode()
                )
                self._ledger_now = "2026-08-31 18:01:00+08:00"
                late = self.ledger.record_receipt(
                    source_id=f"late-{kind}-fixture",
                    source_family=f"late-{kind}-fixture",
                    origin_id=f"late-{kind}-{row.instrument_id}",
                    canonical_url=f"https://example.test/late/{kind}/{row.instrument_id}",
                    raw=raw,
                    content_type="application/json",
                    published_at=row.published_at,
                    first_captured_at=row.first_captured_at,
                    available_at=row.available_at,
                    availability_basis="fixture_forward_capture",
                    now="2026-08-31 18:01:00+08:00",
                )
                frame.at[frame.index[0], "source_receipt_id"] = late["event_id"]
                frame.at[frame.index[0], "content_sha256"] = late["body"]["raw_sha256"]
                result = self.score(
                    decision_session=DECISION,
                    prior_session=PRIOR,
                    decision_at=DECISION_AT,
                    u0=u0_snapshot(),
                    bars=bars,
                    primary_theme_map=mappings,
                    config=ThemeCarrierConfig(),
                )
                self.assertEqual(
                    (result.status, result.abstain_reason),
                    ("abstain", "unlinked_or_tampered_ledger_receipt"),
                )

    def test_decision_cutoff_must_be_this_session_at_1800_and_match_u0(self):
        bars, mappings = self.inputs()
        common = dict(
            decision_session=DECISION,
            prior_session=PRIOR,
            u0=u0_snapshot(),
            bars=bars,
            primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )
        for invalid_cutoff in (
            "2026-08-31 17:59:00+08:00",
            "2026-08-31 18:01:00+08:00",
            "2026-09-01 18:00:00+08:00",
        ):
            with self.subTest(invalid_cutoff=invalid_cutoff):
                result = self.score(decision_at=invalid_cutoff, **common)
                self.assertEqual(
                    (result.status, result.abstain_reason),
                    ("abstain", "decision_at_must_equal_session_1800_shanghai"),
                )

        stale_u0 = replace(u0_snapshot(), decision_at=pd.Timestamp("2026-08-31 17:59:00+08:00"))
        result = self.score(decision_at=DECISION_AT, u0=stale_u0, **{
            key: value for key, value in common.items() if key != "u0"
        })
        self.assertEqual((result.status, result.abstain_reason), ("abstain", "forward_u0_cutoff_mismatch"))

        tampered_u0 = replace(u0_snapshot(), membership_hash=hashlib.sha256(b"tampered-u0").hexdigest())
        result = self.score(decision_at=DECISION_AT, u0=tampered_u0, **{
            key: value for key, value in common.items() if key != "u0"
        })
        self.assertEqual((result.status, result.abstain_reason), ("abstain", "forward_u0_identity_mismatch"))

    def test_sealed_limit_is_not_broken_and_future_rows_cannot_change_today_score(self):
        bars, mappings = self.inputs()
        kwargs = dict(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )
        base = self.score(bars=bars, **kwargs)
        by_theme = base.theme_components.set_index("theme_id")
        self.assertEqual(by_theme.loc["tech", "broken_board_rate"], 1.0)
        self.assertEqual(by_theme.loc["energy", "broken_board_rate"], 0.0)

        future = pd.concat(
            [bars, pd.DataFrame([bar(DECISION + pd.Timedelta(days=1), "A.SZ", 1.0, 1.1, high=1.1)])],
            ignore_index=True,
        )
        later = self.score(bars=future, **kwargs)
        pd.testing.assert_series_equal(base.stock_scores, later.stock_scores)
        pd.testing.assert_frame_equal(base.theme_components, later.theme_components)

    def test_entry_ordering_changes_only_new_entries_and_abstention_is_baseline_exact(self):
        bars, mappings = self.inputs()
        result = self.score(
            decision_session=DECISION, prior_session=PRIOR, decision_at=DECISION_AT,
            u0=u0_snapshot(), bars=bars, primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )
        ranked = order_new_entries(
            baseline_candidate_order=ALL_CODES,
            retained_holdings=("A.SZ",), result=result,
        )
        self.assertTrue(ranked.applied)
        self.assertEqual(ranked.baseline_candidate_order, ALL_CODES)
        self.assertEqual(ranked.retained_holdings, ("A.SZ",))
        self.assertEqual(ranked.new_entry_order, ("C.SZ", "D.SZ", "H.SZ", "I.SZ", "J.SZ", "B.SZ", "E.SZ", "F.SZ", "G.SZ"))

        bars.loc[0, "capture_mode"] = "historical_panel"
        abstained = self.score(
            decision_session=DECISION, prior_session=PRIOR, decision_at=DECISION_AT,
            u0=u0_snapshot(), bars=bars, primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )
        unchanged = order_new_entries(
            baseline_candidate_order=ALL_CODES,
            retained_holdings=("A.SZ",), result=abstained,
        )
        self.assertFalse(unchanged.applied)
        self.assertEqual(unchanged.new_entry_order, ("B.SZ", "C.SZ", "D.SZ", "E.SZ", "F.SZ", "G.SZ", "H.SZ", "I.SZ", "J.SZ"))

    def test_out_of_scope_auction_or_emotion_field_is_rejected_not_silently_used(self):
        bars, mappings = self.inputs()
        bars["auction_score"] = 0.5
        result = self.score(
            decision_session=DECISION, prior_session=PRIOR, decision_at=DECISION_AT,
            u0=u0_snapshot(), bars=bars, primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )
        self.assertEqual((result.status, result.abstain_reason), ("abstain", "out_of_scope_theme_input"))

    def test_duplicate_or_ambiguous_primary_theme_abstains_for_the_whole_decision(self):
        bars, mappings = self.inputs()
        mappings = pd.concat([mappings, mappings.iloc[[0]].assign(theme_id="other")], ignore_index=True)
        result = self.score(
            decision_session=DECISION, prior_session=PRIOR, decision_at=DECISION_AT,
            u0=u0_snapshot(), bars=bars, primary_theme_map=mappings,
            config=ThemeCarrierConfig(),
        )
        self.assertEqual((result.status, result.abstain_reason), ("abstain", "incomplete_or_ambiguous_primary_theme"))


if __name__ == "__main__":
    unittest.main()
