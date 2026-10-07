"""Behaviour tests for the Tushare-only forward theme-market thin adapter.

The fixtures are synthetic raw API responses.  They prove parser and
provenance behaviour only; they are not an authorized Tushare trial or a
forward research observation.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quantlab.data.provider_trial import (
    CapabilityDeclaration,
    HistoricalBarSeed,
    PageReconciliation,
    PermissionDeclaration,
    ProviderTrialSubmission,
    QualifiedProviderTrial,
    SampleReceipt,
    TermsIdentity,
    ThemeMarketEvidence,
    acceptance_requirements,
    qualify_provider_trial,
)
from quantlab.data.tushare_theme_market_adapter import (
    TUSHARE_THEME_MARKET_PROVIDER_ID,
    TushareRawResponse,
    TushareThemeMarketError,
    capture_tushare_theme_market_graph_v1,
)
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import (
    FORWARD_CONTRACT_VERSION,
    U0,
    UniverseSnapshot,
    forward_membership_sha256,
)
from quantlab.theme_carrier_graph import score_theme_carrier_graph_v1


DECISION = pd.Timestamp("2026-09-28")
PRIOR = pd.Timestamp("2026-09-25")
DECISION_AT = "2026-09-28 18:00:00+08:00"
CAPTURED_AT = "2026-09-28 16:20:00+08:00"
TS_CODES = tuple(f"{number:06d}.SZ" for number in range(1, 12))
INSTRUMENTS = tuple("sz." + code[:6] for code in TS_CODES)
THEME_TECH = "885001.TI"
THEME_ENERGY = "885002.TI"


def digest(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def raw_payload(fields: list[str], items: list[list[object]]) -> bytes:
    return json.dumps(
        {"code": 0, "msg": "", "data": {"fields": fields, "items": items}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def response(
    endpoint: str,
    response_id: str,
    parameters: dict[str, str],
    fields: list[str],
    items: list[list[object]],
) -> TushareRawResponse:
    return TushareRawResponse(
        endpoint=endpoint,
        response_id=response_id,
        request_parameters=parameters,
        canonical_url="https://api.tushare.pro",
        raw=raw_payload(fields, items),
        content_type="application/json",
        published_at=CAPTURED_AT,
        first_captured_at=CAPTURED_AT,
        available_at=CAPTURED_AT,
        availability_basis="first_capture_at_provider_api",
    )


def u0_snapshot(codes: tuple[str, ...] = INSTRUMENTS) -> UniverseSnapshot:
    input_hash = digest("tushare-theme-adapter-u0")
    return UniverseSnapshot(
        contract_version=FORWARD_CONTRACT_VERSION,
        pool_id="U0",
        session=DECISION.date(),
        decision_at=pd.Timestamp(DECISION_AT),
        cadence_anchor=DECISION.date(),
        members=codes,
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
            members=codes,
            status="ok",
            abstain_reason=None,
        ),
        status="ok",
        abstain_reason=None,
    )


class TushareThemeMarketAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._ledger_now = "2026-09-28 17:50:00+08:00"
        self.ledger = ResearchSignalLedger(
            Path(self.temp.name) / "ledger", now_fn=lambda: self._ledger_now
        )

    def tearDown(self):
        self.temp.cleanup()

    def qualified_trial(self):
        sample_names = ("seed", "page", "mapping", "limit-up", "failed-board", "bars")
        samples = tuple(
            SampleReceipt(
                receipt_id="sample-" + name,
                raw_sha256=digest("sample-raw:" + name),
                source_locator="https://trial.example.test/tushare/" + name,
                captured_at="2026-09-28T16:00:00+08:00",
            )
            for name in sample_names
        )
        by_name = {sample.receipt_id.removeprefix("sample-"): sample.raw_sha256 for sample in samples}
        capabilities = tuple(
            CapabilityDeclaration(
                capability=name,
                available=True,
                evidence_locator="https://trial.example.test/tushare/capabilities#" + name,
                evidence_sha256=digest("capability:" + name),
            )
            for name in (
                "stock_theme_multi_mapping",
                "active_theme_semantics",
                "limit_up_events",
                "failed_board_events",
                "daily_bars",
                "paged_snapshot_reconciliation",
                "u0_membership_coverage",
            )
        )
        submission = ProviderTrialSubmission(
            source_class="theme_market",
            provider_id=TUSHARE_THEME_MARKET_PROVIDER_ID,
            trial_id="tushare-trial-20260928-a",
            source_dataset_id="tushare-pro-ths-theme-market",
            source_release_id="tushare-release-20260928",
            source_release_sha256=digest("tushare-release-20260928"),
            terms=TermsIdentity(
                contract_id="tushare-trial-contract",
                version="2026-09-28",
                locator="https://trial.example.test/tushare/terms",
                sha256=digest("tushare-terms"),
            ),
            automated_access=PermissionDeclaration(
                permission="automated_access",
                granted=True,
                scope="read-only Tushare Pro trial API",
                evidence_locator="https://trial.example.test/tushare/terms#api",
                evidence_sha256=digest("tushare-api-permission"),
            ),
            raw_retention=PermissionDeclaration(
                permission="raw_retention",
                granted=True,
                scope="immutable raw Tushare research responses",
                evidence_locator="https://trial.example.test/tushare/terms#retention",
                evidence_sha256=digest("tushare-retention-permission"),
            ),
            capabilities=capabilities,
            sample_receipts=samples,
            theme_market=ThemeMarketEvidence(
                historical_bar_seed=HistoricalBarSeed(
                    seed_id="tushare-u0-seed",
                    bar_set_sha256=digest("tushare-bar-seed"),
                    calendar_sha256=digest("tushare-calendar"),
                    instrument_coverage_sha256=digest("tushare-coverage"),
                    raw_receipt_sha256=by_name["seed"],
                    session_count=253,
                    first_session="2025-09-27",
                    last_session="2026-09-28",
                ),
                page_reconciliation=PageReconciliation(
                    snapshot_id="tushare-snapshot",
                    coverage_sha256=digest("tushare-page-coverage"),
                    expected_page_count=1,
                    observed_page_numbers=(1,),
                    page_receipt_sha256s=(by_name["page"],),
                    expected_instrument_count=100,
                    observed_unique_instrument_count=100,
                    duplicate_instrument_count=0,
                    u0_membership_sha256=digest("tushare-u0-membership"),
                    expected_u0_member_count=len(INSTRUMENTS),
                    mapped_u0_member_count=len(INSTRUMENTS),
                    unmapped_u0_member_count=0,
                ),
                theme_mapping_receipt_sha256=by_name["mapping"],
                limit_up_receipt_sha256=by_name["limit-up"],
                failed_board_receipt_sha256=by_name["failed-board"],
                market_bar_receipt_sha256=by_name["bars"],
            ),
        )
        return qualify_provider_trial(submission)

    def responses(self):
        previous_date = PRIOR.strftime("%Y%m%d")
        decision_date = DECISION.strftime("%Y%m%d")
        index = response(
            "ths_index",
            "themes",
            {"type": "N"},
            ["ts_code", "name", "count", "exchange", "list_date", "type"],
            [
                [THEME_TECH, "科技", 5, "A", "20200101", "N"],
                [THEME_ENERGY, "能源", 6, "A", "20200101", "N"],
            ],
        )
        members_tech = response(
            "ths_member",
            "member-tech",
            {"ts_code": THEME_TECH},
            ["ts_code", "con_code", "con_name"],
            [[THEME_TECH, code, "股票" + code[:6]] for code in TS_CODES[:5]],
        )
        members_energy = response(
            "ths_member",
            "member-energy",
            {"ts_code": THEME_ENERGY},
            ["ts_code", "con_code", "con_name"],
            [[THEME_ENERGY, code, "股票" + code[:6]] for code in TS_CODES[4:10]],
        )
        daily_fields = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount"]
        prior_daily_items = []
        today_daily_items = []
        for index_number, code in enumerate(TS_CODES, start=1):
            prior_close = 11.0 if index_number in (1, 6) else 10.0
            prior_daily_items.append(
                [code, previous_date, prior_close, prior_close, prior_close, prior_close, 10.0, 100000.0]
            )
            if index_number <= 4:
                today_daily_items.append([code, decision_date, 10.5, 11.0, 10.2, 10.5, prior_close, 100000.0])
            elif index_number <= 10:
                today_daily_items.append([code, decision_date, 11.0, 11.0, 10.5, 11.0, prior_close, 100000.0])
            else:
                today_daily_items.append([code, decision_date, 10.2, 10.5, 10.1, 10.4, prior_close, 100000.0])
        prior_daily = response("daily", "daily-prior", {"trade_date": previous_date}, daily_fields, prior_daily_items)
        current_daily = response("daily", "daily-current", {"trade_date": decision_date}, daily_fields, today_daily_items)
        limit_fields = ["ts_code", "trade_date", "up_limit", "down_limit"]
        prior_limits = response(
            "stk_limit",
            "limits-prior",
            {"trade_date": previous_date},
            limit_fields,
            [[code, previous_date, 11.0, 9.0] for code in TS_CODES],
        )
        current_limits = response(
            "stk_limit",
            "limits-current",
            {"trade_date": decision_date},
            limit_fields,
            [[code, decision_date, 11.0, 9.0] for code in TS_CODES],
        )
        board = response(
            "limit_list_d",
            "boards-current",
            {"trade_date": decision_date},
            ["ts_code", "trade_date", "limit", "open_times"],
            [
                [code, decision_date, "Z" if index_number <= 4 else "U", 1 if index_number <= 4 else 0]
                for index_number, code in enumerate(TS_CODES[:10], start=1)
            ],
        )
        return (index, members_tech, members_energy, prior_daily, current_daily, prior_limits, current_limits, board)

    def capture(self, **changes):
        arguments = dict(
            qualified_trial=self.qualified_trial(),
            ledger=self.ledger,
            u0=u0_snapshot(),
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            responses=self.responses(),
        )
        arguments.update(changes)
        return capture_tushare_theme_market_graph_v1(**arguments)

    def test_materializes_qualified_raw_responses_into_many_to_many_graph_input(self):
        result = self.capture()

        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.abstain_reason)
        self.assertIsNotNone(result.graph_input)
        graph_input = result.graph_input
        self.assertEqual(set(graph_input.theme_edges.instrument_id), set(INSTRUMENTS))
        shared = graph_input.theme_edges[graph_input.theme_edges.instrument_id.eq("sz.000005")]
        self.assertEqual(set(shared.theme_id), {THEME_TECH, THEME_ENERGY})
        unknown = graph_input.theme_edges[graph_input.theme_edges.instrument_id.eq("sz.000011")]
        self.assertEqual(unknown.mapping_status.tolist(), ["known_no_eligible_theme"])
        self.assertEqual(len(graph_input.receipt_ids), 8)
        self.assertEqual(len(graph_input.receipt_hashes), 8)
        self.assertTrue(self.ledger.verify()["verified"])

        scored = score_theme_carrier_graph_v1(
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            u0=u0_snapshot(),
            bars=graph_input.bars,
            theme_edges=graph_input.theme_edges,
            ledger=self.ledger,
        )
        self.assertEqual(scored.status, "ok")
        self.assertEqual(scored.stock_scores.loc["sz.000011"], 0.0)
        self.assertEqual(set(scored.receipt_ids), set(graph_input.receipt_ids))
        self.assertEqual(set(scored.receipt_hashes), set(graph_input.receipt_hashes))

    def test_missing_provider_theme_member_response_abstains_after_recording_available_raw_evidence(self):
        responses = tuple(item for item in self.responses() if item.response_id != "member-energy")
        result = self.capture(responses=responses)

        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.abstain_reason, "tushare_theme_member_response_missing")
        self.assertIsNone(result.graph_input)
        self.assertEqual(len(result.receipt_ids), 7)
        self.assertTrue(self.ledger.verify()["verified"])

    def test_malformed_provider_body_is_retained_before_the_period_abstains(self):
        responses = list(self.responses())
        index = next(i for i, item in enumerate(responses) if item.endpoint == "ths_index")
        responses[index] = replace(responses[index], raw=b"{not-json")

        result = self.capture(responses=tuple(responses))

        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.abstain_reason, "tushare_raw_response_invalid")
        self.assertEqual(len(result.receipt_ids), 8)
        self.assertTrue(self.ledger.verify()["verified"])

    def test_board_receipt_must_agree_with_daily_high_close_and_limit_price(self):
        responses = list(self.responses())
        index = next(i for i, item in enumerate(responses) if item.endpoint == "limit_list_d")
        responses[index] = replace(
            responses[index],
            raw=raw_payload(
                ["ts_code", "trade_date", "limit", "open_times"],
                [[code, DECISION.strftime("%Y%m%d"), "U", 0] for code in TS_CODES[:10]],
            ),
        )
        result = self.capture(responses=tuple(responses))

        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.abstain_reason, "tushare_board_event_inconsistent")
        self.assertTrue(self.ledger.verify()["verified"])

    def test_adapter_rechecks_that_the_trial_was_qualified_and_retention_was_granted(self):
        trial = self.qualified_trial()
        forged = QualifiedProviderTrial(trial.submission, tuple())
        with self.assertRaisesRegex(TushareThemeMarketError, "tushare_trial_not_qualified"):
            self.capture(qualified_trial=forged)

        no_retention = replace(trial.submission, raw_retention=replace(trial.submission.raw_retention, granted=False))
        forged_no_retention = QualifiedProviderTrial(
            no_retention,
            tuple(item.requirement_id for item in acceptance_requirements("theme_market")),
        )
        with self.assertRaisesRegex(TushareThemeMarketError, "tushare_trial_not_qualified"):
            self.capture(qualified_trial=forged_no_retention)


if __name__ == "__main__":
    unittest.main()
