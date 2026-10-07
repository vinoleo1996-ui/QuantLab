import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import (
    FORWARD_CONTRACT_VERSION,
    U0,
    UniverseSnapshot,
    forward_membership_sha256,
)
from quantlab.theme_carrier_graph import (
    ThemeCarrierGraphConfig,
    order_graph_new_entries,
    score_theme_carrier_graph_v1,
)


DECISION = pd.Timestamp("2026-08-31")
PRIOR = pd.Timestamp("2026-08-28")
DECISION_AT = "2026-08-31 18:00:00+08:00"
TECH_CODES = ("A.SZ", "B.SZ", "E.SZ", "F.SZ", "G.SZ")
ENERGY_CODES = ("A.SZ", "C.SZ", "D.SZ", "H.SZ", "I.SZ", "J.SZ")
ALL_CODES = ("A.SZ", "B.SZ", "C.SZ", "D.SZ", "E.SZ", "F.SZ", "G.SZ", "H.SZ", "I.SZ", "J.SZ")


def u0_snapshot(codes=ALL_CODES):
    input_hash = hashlib.sha256(b"forward-u0-theme-graph-test-input").hexdigest()
    return UniverseSnapshot(
        contract_version=FORWARD_CONTRACT_VERSION,
        pool_id="U0",
        session=DECISION.date(),
        decision_at=pd.Timestamp(DECISION_AT),
        cadence_anchor=DECISION.date(),
        members=tuple(codes),
        rank_by_instrument={code: index + 3001 for index, code in enumerate(codes)},
        adv60_by_instrument={code: 6_000_000.0 for code in codes},
        eligibility_reason_by_instrument={},
        input_hash=input_hash,
        membership_hash=forward_membership_sha256(
            spec=U0,
            session=DECISION,
            decision_at=DECISION_AT,
            cadence_anchor=DECISION,
            input_hash=input_hash,
            members=tuple(codes),
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
        "source_receipt_id": f"bar-{session:%Y%m%d}-{code}",
        "content_sha256": hashlib.sha256(f"bar-{session:%Y%m%d}-{code}".encode()).hexdigest(),
        "capture_mode": "forward_receipt",
    }


def edge(code, theme, status="active"):
    return {
        "session": DECISION,
        "instrument_id": code,
        "theme_id": theme,
        "mapping_status": status,
        "published_at": "2026-08-31 15:00:00+08:00",
        "first_captured_at": "2026-08-31 16:00:00+08:00",
        "available_at": "2026-08-31 17:00:00+08:00",
        "source_receipt_id": f"map-{code}-{theme or 'none'}",
        "content_sha256": hashlib.sha256(f"map-{code}-{theme or 'none'}".encode()).hexdigest(),
        "capture_mode": "forward_receipt",
    }


def fixture_inputs(codes=ALL_CODES):
    bars = [bar(PRIOR, code, 11.0, 11.0) for code in codes]
    for code in codes:
        if code in TECH_CODES:
            bars.append(bar(DECISION, code, 10.5, 11.0, high=11.0))
        else:
            bars.append(bar(DECISION, code, 12.1, 12.1, high=12.1))
    mappings = [edge(code, "tech") for code in TECH_CODES]
    mappings.extend(edge(code, "energy") for code in ENERGY_CODES)
    return pd.DataFrame(bars), pd.DataFrame(mappings)


class ThemeCarrierGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._ledger_now = "2026-08-31 17:50:00+08:00"
        self.ledger = ResearchSignalLedger(
            Path(self.temp.name) / "ledger", now_fn=lambda: self._ledger_now
        )

    def tearDown(self):
        self.temp.cleanup()

    def inputs(self, codes=ALL_CODES):
        self._ledger_now = "2026-08-31 17:50:00+08:00"
        bars, mappings = fixture_inputs(codes)
        for kind, frame in (("bar", bars), ("map", mappings)):
            for index, row in frame.iterrows():
                raw = (
                    f"bar-{pd.Timestamp(row.session):%Y%m%d}-{row.instrument_id}".encode()
                    if kind == "bar"
                    else f"map-{row.instrument_id}-{row.theme_id or 'none'}".encode()
                )
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
                )
                frame.at[index, "source_receipt_id"] = receipt["event_id"]
                frame.at[index, "content_sha256"] = receipt["body"]["raw_sha256"]
        return bars, mappings

    def score(self, **kwargs):
        return score_theme_carrier_graph_v1(ledger=self.ledger, **kwargs)

    def test_source_native_many_to_many_edges_choose_the_best_eligible_theme(self):
        bars, mappings = self.inputs()
        result = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            theme_edges=mappings,
            config=ThemeCarrierGraphConfig(),
        )

        self.assertEqual(result.status, "ok")
        themes = result.theme_components.set_index("theme_id")
        self.assertAlmostEqual(
            themes.loc["energy", "theme_score_equal_weight_graph_v1"],
            themes.loc[
                "energy",
                [
                    "prior_limit_up_premium_percentile",
                    "broken_board_rate_percentile",
                    "theme_relative_strength_percentile",
                    "advancing_breadth_percentile",
                ],
            ].mean(),
        )
        details = result.stock_theme_details.set_index("instrument_id")
        self.assertEqual(int(details.loc["A.SZ", "theme_membership_count"]), 2)
        self.assertEqual(int(details.loc["A.SZ", "eligible_theme_count"]), 2)
        self.assertEqual(details.loc["A.SZ", "winning_theme_id"], "energy")
        self.assertGreater(result.stock_scores["A.SZ"], result.stock_scores["B.SZ"])
        self.assertEqual(result.coverage["active_theme_edges"], len(mappings))
        self.assertEqual(
            set(result.source_native_edges[["instrument_id", "theme_id"]].itertuples(index=False, name=None)),
            set(mappings[["instrument_id", "theme_id"]].itertuples(index=False, name=None)),
        )
        self.assertEqual(
            set(result.source_native_edges.content_sha256), set(mappings.content_sha256)
        )

    def test_known_or_unscorable_theme_is_a_real_zero_not_an_abstention(self):
        codes = ALL_CODES + ("K.SZ", "L.SZ")
        bars, mappings = self.inputs(codes)
        mappings = pd.concat(
            [
                mappings,
                pd.DataFrame(
                    [
                        edge("K.SZ", "orphan"),
                        edge("L.SZ", None, "known_no_eligible_theme"),
                    ]
                ),
            ],
            ignore_index=True,
        )
        for position, raw in ((-2, b"map-K.SZ-orphan"), (-1, b"map-L.SZ-none")):
            row = mappings.iloc[position]
            receipt = self.ledger.record_receipt(
                source_id=f"fixture-map-zero-{position}",
                source_family="fixture-map",
                origin_id=f"fixture-map-zero-{position}",
                canonical_url=f"https://example.test/map/zero/{position}",
                raw=raw,
                content_type="application/json",
                published_at=row.published_at,
                first_captured_at=row.first_captured_at,
                available_at=row.available_at,
                availability_basis="fixture_forward_capture",
            )
            mappings.at[mappings.index[position], "source_receipt_id"] = receipt["event_id"]
            mappings.at[mappings.index[position], "content_sha256"] = receipt["body"]["raw_sha256"]

        result = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(codes),
            bars=bars,
            theme_edges=mappings,
        )

        self.assertEqual(result.status, "ok")
        details = result.stock_theme_details.set_index("instrument_id")
        self.assertEqual(result.stock_scores["K.SZ"], 0.0)
        self.assertEqual(result.stock_scores["L.SZ"], 0.0)
        self.assertEqual(int(details.loc["K.SZ", "theme_membership_count"]), 1)
        self.assertEqual(int(details.loc["K.SZ", "eligible_theme_count"]), 0)
        self.assertTrue(pd.isna(details.loc["K.SZ", "winning_theme_id"]))
        self.assertEqual(details.loc["K.SZ", "score_status"], "known_no_eligible_theme")
        self.assertEqual(int(details.loc["L.SZ", "theme_membership_count"]), 0)
        self.assertEqual(int(details.loc["L.SZ", "eligible_theme_count"]), 0)

    def test_missing_or_ambiguous_edge_coverage_abstains_for_the_whole_decision(self):
        bars, mappings = self.inputs()
        missing = mappings[mappings.instrument_id.ne("J.SZ")].copy()
        result = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            theme_edges=missing,
        )
        self.assertEqual(
            (result.status, result.abstain_reason),
            ("abstain", "incomplete_or_ambiguous_theme_graph_mapping"),
        )

        duplicate = pd.concat([mappings, mappings.iloc[[0]]], ignore_index=True)
        ambiguous = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            theme_edges=duplicate,
        )
        self.assertEqual(
            (ambiguous.status, ambiguous.abstain_reason),
            ("abstain", "incomplete_or_ambiguous_theme_graph_mapping"),
        )

        status_ambiguous = mappings.copy()
        status_ambiguous.loc[0, "mapping_status"] = "ambiguous"
        ambiguous_status = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            theme_edges=status_ambiguous,
        )
        self.assertEqual(
            (ambiguous_status.status, ambiguous_status.abstain_reason),
            ("abstain", "incomplete_or_ambiguous_theme_graph_mapping"),
        )

    def test_receipts_are_ledger_bound_and_late_or_tampered_mapping_cannot_score(self):
        bars, mappings = self.inputs()
        mappings.loc[0, "content_sha256"] = hashlib.sha256(b"tampered-theme-edge").hexdigest()
        tampered = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            theme_edges=mappings,
        )
        self.assertEqual(
            (tampered.status, tampered.abstain_reason),
            ("abstain", "unlinked_or_tampered_ledger_receipt"),
        )

        bars, mappings = self.inputs()
        self._ledger_now = "2026-08-31 18:01:00+08:00"
        row = mappings.iloc[0]
        receipt = self.ledger.record_receipt(
            source_id="late-map",
            source_family="late-map",
            origin_id="late-map-A.SZ-tech",
            canonical_url="https://example.test/map/late",
            raw=b"map-A.SZ-tech",
            content_type="application/json",
            published_at=row.published_at,
            first_captured_at=row.first_captured_at,
            available_at=row.available_at,
            availability_basis="fixture_forward_capture",
        )
        mappings.at[mappings.index[0], "source_receipt_id"] = receipt["event_id"]
        mappings.at[mappings.index[0], "content_sha256"] = receipt["body"]["raw_sha256"]
        late = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=bars,
            theme_edges=mappings,
        )
        self.assertEqual(
            (late.status, late.abstain_reason),
            ("abstain", "unlinked_or_tampered_ledger_receipt"),
        )

    def test_entry_order_only_reorders_new_names_and_preserves_baseline_order_for_ties(self):
        codes = ALL_CODES + ("K.SZ",)
        bars, mappings = self.inputs(codes)
        mappings = pd.concat(
            [mappings, pd.DataFrame([edge("K.SZ", None, "known_no_eligible_theme")])],
            ignore_index=True,
        )
        row = mappings.iloc[-1]
        receipt = self.ledger.record_receipt(
            source_id="fixture-map-order-none",
            source_family="fixture-map",
            origin_id="fixture-map-order-none",
            canonical_url="https://example.test/map/order-none",
            raw=b"map-K.SZ-none",
            content_type="application/json",
            published_at=row.published_at,
            first_captured_at=row.first_captured_at,
            available_at=row.available_at,
            availability_basis="fixture_forward_capture",
        )
        mappings.at[mappings.index[-1], "source_receipt_id"] = receipt["event_id"]
        mappings.at[mappings.index[-1], "content_sha256"] = receipt["body"]["raw_sha256"]
        result = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(codes),
            bars=bars,
            theme_edges=mappings,
        )

        baseline = ("K.SZ", "B.SZ", "E.SZ", "A.SZ", "C.SZ", "D.SZ")
        ordered = order_graph_new_entries(
            baseline_candidate_order=baseline,
            retained_holdings=("A.SZ",),
            result=result,
        )
        self.assertTrue(ordered.applied)
        self.assertEqual(ordered.retained_holdings, ("A.SZ",))
        self.assertEqual(ordered.new_entry_order, ("C.SZ", "D.SZ", "B.SZ", "E.SZ", "K.SZ"))

        abstained = self.score(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(codes),
            bars=bars.drop(index=0),
            theme_edges=mappings,
        )
        unchanged = order_graph_new_entries(
            baseline_candidate_order=baseline,
            retained_holdings=("A.SZ",),
            result=abstained,
        )
        self.assertFalse(unchanged.applied)
        self.assertEqual(unchanged.new_entry_order, ("K.SZ", "B.SZ", "E.SZ", "C.SZ", "D.SZ"))


if __name__ == "__main__":
    unittest.main()
