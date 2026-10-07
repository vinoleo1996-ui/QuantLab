"""Synthetic QuantDB responses for the concept-graph adapter.

These fixtures prove the limit rule and the ledger gate.  They are not a
QuantDB capture and not an account result.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quantlab.data.quantdb_theme_market_adapter import (
    QuantDBRawResponse,
    capture_quantdb_theme_graph_v1,
)
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import (
    FORWARD_CONTRACT_VERSION,
    U0,
    UniverseSnapshot,
    forward_membership_sha256,
)


DECISION = pd.Timestamp("2026-09-28")
PRIOR = pd.Timestamp("2026-09-24")
REFERENCE = pd.Timestamp("2026-09-23")
DECISION_AT = "2026-09-28 18:00:00+08:00"
CAPTURED_AT = "2026-09-28 16:10:00+08:00"
CALENDAR = (
    "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
    "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-28",
)
CODES = tuple(f"{number:06d}.SZ" for number in range(1, 11))
INSTRUMENTS = tuple("sz." + code[:6] for code in CODES)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def response(dataset: str, response_id: str, parameters: dict[str, str], raw: bytes, content_type: str) -> QuantDBRawResponse:
    path = {
        "sector_concept": "/api/v1/data/preview",
        "daily_manifest": "/api/v1/data/download/manifest",
        "daily_unadjusted": "/api/v1/data/download",
        "stock_list": "/api/v1/data/stock-list",
        "trade_calendar": "/api/v1/data/calendar",
    }[dataset]
    return QuantDBRawResponse(
        dataset=dataset,
        response_id=response_id,
        request_parameters=parameters,
        canonical_url="https://www.quantdb.cn" + path,
        raw=raw,
        content_type=content_type,
        published_at=CAPTURED_AT,
        first_captured_at=CAPTURED_AT,
        available_at=CAPTURED_AT,
        availability_basis="first_capture_of_quantdb_response_no_vendor_publish_time",
    )


def u0_snapshot() -> UniverseSnapshot:
    input_hash = digest("quantdb-theme-adapter-u0")
    return UniverseSnapshot(
        contract_version=FORWARD_CONTRACT_VERSION,
        pool_id="U0",
        session=DECISION.date(),
        decision_at=pd.Timestamp(DECISION_AT),
        cadence_anchor=DECISION.date(),
        members=INSTRUMENTS,
        rank_by_instrument={code: index + 3001 for index, code in enumerate(INSTRUMENTS)},
        adv60_by_instrument={code: 6_000_000.0 for code in INSTRUMENTS},
        eligibility_reason_by_instrument={},
        input_hash=input_hash,
        membership_hash=forward_membership_sha256(
            spec=U0,
            session=DECISION,
            decision_at=DECISION_AT,
            cadence_anchor=DECISION,
            input_hash=input_hash,
            members=INSTRUMENTS,
            status="ok",
            abstain_reason=None,
        ),
        status="ok",
        abstain_reason=None,
    )


def bars(session: pd.Timestamp, close_for) -> bytes:
    rows = []
    for code in CODES:
        close = close_for(code)
        rows.append({
            "symbol": code,
            "time": session.strftime("%Y-%m-%d"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000,
            "amount": 10.0,
        })
    return parquet_bytes(pd.DataFrame(rows))


def decision_bars() -> bytes:
    rows = []
    for code in CODES:
        if code == "000001.SZ":
            price = (12.10, 12.10, 12.10, 12.10)
        elif code in {"000002.SZ", "000006.SZ"}:
            price = (10.50, 11.44, 10.40, 11.00)
        elif code == "000005.SZ":
            price = (11.20, 11.50, 11.00, 11.40)
        else:
            price = (10.50, 10.80, 10.40, 10.70)
        rows.append({
            "symbol": code,
            "time": DECISION.strftime("%Y-%m-%d"),
            "open": price[0],
            "high": price[1],
            "low": price[2],
            "close": price[3],
            "volume": 1000,
            "amount": 10.0,
        })
    return parquet_bytes(pd.DataFrame(rows))


def sector_payload(industry_only: str | None = None, drop: str | None = None) -> bytes:
    rows = []
    for code in CODES[:6]:
        rows.append({"SectorCode": "881001.SH", "SectorName": "概念甲", "SectorType": "概念板块", "Symbol": code})
    for code in CODES[4:]:
        if code == industry_only:
            rows.append({"SectorCode": "881101.SH", "SectorName": "行业甲", "SectorType": "行业板块(二级)", "Symbol": code})
            continue
        if code == drop:
            continue
        rows.append({"SectorCode": "881002.SH", "SectorName": "概念乙", "SectorType": "概念板块", "Symbol": code})
    return json.dumps({"data": rows}, ensure_ascii=False).encode("utf-8")


def responses(industry_only: str | None = None, drop: str | None = None) -> tuple[QuantDBRawResponse, ...]:
    calendar = json.dumps({
        "data": [{"trade_date": day, "is_open": 1} for day in CALENDAR],
    }).encode("utf-8")
    stocks = json.dumps({
        "data": [
            {"symbol": code, "name": "合成", "list_date": "1991-04-03", "is_st": 0, "is_delisted": 0}
            for code in CODES
        ],
    }).encode("utf-8")
    return (
        response("daily_manifest", "manifest", {"category_id": "1", "sub_category": "daily_unadjusted", "layout": "v2"}, json.dumps({"files": []}).encode("utf-8"), "application/json"),
        response("trade_calendar", "calendar", {"start_date": "2026-09-14", "end_date": "2026-09-28"}, calendar, "application/json"),
        response("stock_list", "stocks", {"limit": "10"}, stocks, "application/json"),
        response("sector_concept", "concepts", {"category_id": "2", "sub_category": "sector_concept"}, sector_payload(industry_only, drop), "application/json"),
        response("daily_unadjusted", "daily-reference", {"category_id": "1", "sub_category": "daily_unadjusted", "trade_date": "20260923"}, bars(REFERENCE, lambda _code: 10.0), "application/vnd.apache.parquet"),
        response("daily_unadjusted", "daily-prior", {"category_id": "1", "sub_category": "daily_unadjusted", "trade_date": "20260924"}, bars(PRIOR, lambda code: 11.0 if code in {"000001.SZ", "000005.SZ"} else 10.40), "application/vnd.apache.parquet"),
        response("daily_unadjusted", "daily-decision", {"category_id": "1", "sub_category": "daily_unadjusted", "trade_date": "20260928"}, decision_bars(), "application/vnd.apache.parquet"),
    )


class QuantDBThemeMarketAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._ledger_now = "2026-09-28 17:00:00+08:00"
        self.ledger = ResearchSignalLedger(Path(self.temp.name) / "ledger", now_fn=lambda: self._ledger_now)

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, **changes):
        arguments = dict(
            ledger=self.ledger,
            u0=u0_snapshot(),
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            responses=responses(),
        )
        arguments.update(changes)
        return capture_quantdb_theme_graph_v1(**arguments)

    def test_limit_comes_from_prior_close_and_graph_scores(self):
        captured = self.capture()
        self.assertEqual(captured.status, "ok", captured.abstain_reason)
        assert captured.graph_input is not None and captured.graph_result is not None
        self.assertEqual(len(captured.graph_input.receipt_ids), 7)
        prior = captured.graph_input.bars[captured.graph_input.bars.session.eq(PRIOR)]
        self.assertTrue((prior.limit_up == 11.0).all())
        current = captured.graph_input.bars[
            captured.graph_input.bars.instrument_id.eq("sz.000002")
            & captured.graph_input.bars.session.eq(DECISION)
        ]
        self.assertEqual(float(current.limit_up.iloc[0]), 11.44)
        self.assertEqual(captured.graph_result.status, "ok")
        self.assertGreater(int(captured.coverage["touched_count"]), 0)
        self.assertNotIn("zt_price", captured.graph_input.bars.columns)

    def test_late_receipt_abstains_without_dropping_the_diagnostic(self):
        self._ledger_now = "2026-09-29 09:00:00+08:00"
        captured = self.capture()
        self.assertEqual(captured.status, "abstain")
        self.assertEqual(captured.abstain_reason, "quantdb_receipt_late_for_decision")
        self.assertIsNone(captured.graph_result)
        self.assertGreater(int(captured.coverage["broken_count"]), 0)
        self.assertGreaterEqual(len(captured.receipt_ids), 6)
        self.assertTrue(self.ledger.verify()["verified"])

    def test_placeholder_list_date_does_not_reject_the_stock_file(self):
        items = list(responses())
        stock_index = next(index for index, item in enumerate(items) if item.dataset == "stock_list")
        payload = json.loads(items[stock_index].raw)
        payload["data"].append({
            "symbol": "000011.SZ", "name": "缺上市日", "list_date": "0", "is_st": 0, "is_delisted": 0,
        })
        items[stock_index] = QuantDBRawResponse(
            dataset=items[stock_index].dataset,
            response_id=items[stock_index].response_id,
            request_parameters=items[stock_index].request_parameters,
            canonical_url=items[stock_index].canonical_url,
            raw=json.dumps(payload).encode("utf-8"),
            content_type=items[stock_index].content_type,
            published_at=items[stock_index].published_at,
            first_captured_at=items[stock_index].first_captured_at,
            available_at=items[stock_index].available_at,
            availability_basis=items[stock_index].availability_basis,
        )
        captured = self.capture(responses=tuple(items))
        self.assertEqual(captured.status, "ok", captured.abstain_reason)

    def test_symbol_missing_from_sector_file_is_unconfirmed(self):
        captured = self.capture(responses=responses(drop="000010.SZ"))
        self.assertEqual(captured.abstain_reason, "quantdb_theme_membership_unconfirmed")

    def test_industry_only_symbol_is_a_known_empty_concept_not_a_guess(self):
        captured = self.capture(responses=responses(industry_only="000010.SZ"))
        self.assertEqual(captured.status, "ok", captured.abstain_reason)
        assert captured.graph_result is not None
        self.assertEqual(float(captured.graph_result.stock_scores.loc["sz.000010"]), 0.0)
        details = captured.graph_result.stock_theme_details
        self.assertEqual(
            details.set_index("instrument_id").loc["sz.000010", "score_status"],
            "known_no_eligible_theme",
        )

    def test_beijing_member_is_excluded_without_voiding_the_score(self):
        members = INSTRUMENTS + ("bj.920001",)
        input_hash = digest("quantdb-theme-adapter-u0-bj")
        snapshot = UniverseSnapshot(
            contract_version=FORWARD_CONTRACT_VERSION,
            pool_id="U0",
            session=DECISION.date(),
            decision_at=pd.Timestamp(DECISION_AT),
            cadence_anchor=DECISION.date(),
            members=members,
            rank_by_instrument={code: index + 3001 for index, code in enumerate(members)},
            adv60_by_instrument={code: 6_000_000.0 for code in members},
            eligibility_reason_by_instrument={},
            input_hash=input_hash,
            membership_hash=forward_membership_sha256(
                spec=U0,
                session=DECISION,
                decision_at=DECISION_AT,
                cadence_anchor=DECISION,
                input_hash=input_hash,
                members=members,
                status="ok",
                abstain_reason=None,
            ),
            status="ok",
            abstain_reason=None,
        )
        captured = self.capture(u0=snapshot)
        self.assertEqual(captured.status, "ok", captured.abstain_reason)
        self.assertEqual(captured.coverage["u0_member_count"], 11)
        self.assertEqual(captured.coverage["excluded_member_count"], 1)
        self.assertEqual(captured.coverage["excluded_reason_counts"], {"limit_rule_undefined": 1})
        self.assertEqual(captured.coverage["scored_member_count"], 10)
        assert captured.graph_result is not None
        self.assertNotIn("bj.920001", set(captured.graph_result.stock_scores.index))


if __name__ == "__main__":
    unittest.main()
