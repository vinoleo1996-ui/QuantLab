"""End-to-end contracts for the registered raw-market transport bridge.

These tests use only in-memory raw bytes.  They intentionally do not make a
network request, construct an account, or enter a historical replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from quantlab.forward_market_capture import capture_eastmoney_eod_snapshot
from quantlab.forward_market_transport import (
    FailedRawEastmoneyCollection,
    RawEastmoneyHttpsResponse,
    collect_registered_eastmoney_eod_pages,
)
from quantlab.forward_source_manifest import (
    PreparedRawEastmoneyCollection,
    SourceManifestJournal,
    fixed_eastmoney_eod_manifest_v1,
)
from quantlab.research_signal_ledger import ResearchSignalLedger


SESSION = "2026-10-01"
DECISION_AT = "2026-10-01 18:00:00+08:00"


def _row(code: int) -> dict[str, object]:
    return {
        "f12": str(code).zfill(6),
        "f14": "测试" + str(code).zfill(6),
        "f2": 11.0,
        "f17": 11.0,
        "f15": 11.0,
        "f16": 11.0,
        "f18": 10.0,
        "f5": 1000.0,
        "f6": 10_000_000.0,
        "f51": 11.0,
    }


def _raw_page(rows: list[dict[str, object]], *, total: int | None = None) -> bytes:
    return json.dumps(
        {"data": {"total": len(rows) if total is None else total, "diff": rows}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class _FixedClock:
    def __init__(self, value: str):
        self.value = value

    def __call__(self) -> str:
        return self.value


class _SequenceClock:
    """Return one ledger-owned runtime timestamp per materialisation step."""

    def __init__(self, *values: str):
        self.values = list(values)

    def __call__(self) -> str:
        if not self.values:
            raise AssertionError("fixture_runtime_clock_exhausted")
        return self.values.pop(0)


class _FakeOpener:
    def __init__(self, manifest, raws: list[bytes | BaseException]):
        self.manifest = manifest
        self.raws = list(raws)

    def __call__(self, url: str) -> RawEastmoneyHttpsResponse:
        response = self.raws.pop(0)
        if isinstance(response, BaseException):
            raise response
        return RawEastmoneyHttpsResponse(
            request_url=url,
            response_url=url,
            raw=response,
        )


class ForwardMarketTransportBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.journal = SourceManifestJournal(self.root / "source-manifests")
        self.manifest = fixed_eastmoney_eod_manifest_v1()

    def tearDown(self):
        self.temp.cleanup()

    def _collect(self, *, raws: list[bytes | BaseException], received_at: str):
        return collect_registered_eastmoney_eod_pages(
            journal=self.journal,
            manifest=self.manifest,
            session=SESSION,
            decision_at=DECISION_AT,
            collection_id="bridge-fixture-20261001",
            opener=_FakeOpener(self.manifest, raws),
            clock=_FixedClock(received_at),
        )

    def _ledger(self, written_at: str) -> ResearchSignalLedger:
        return ResearchSignalLedger(
            self.root / "research-signal-ledger",
            now_fn=lambda: written_at,
        )

    def test_registered_collection_is_the_only_path_that_materializes_page_receipts(self):
        raw = _raw_page([_row(1)])
        collection = self._collect(
            raws=[raw],
            received_at="2026-10-01 17:40:00+08:00",
        )
        ledger = self._ledger("2026-10-01 17:56:00+08:00")

        capture = capture_eastmoney_eod_snapshot(
            ledger=ledger,
            journal=self.journal,
            collection=collection,
        )

        self.assertEqual((capture.status, capture.abstain_reason), ("ok", None))
        self.assertEqual(capture.source_manifest_sha256, self.manifest.manifest_sha256)
        self.assertIsNotNone(capture.transport_receipt_id)
        self.assertEqual(len(capture.receipt_ids), 1)
        page_evidence = ledger.export_receipt_evidence(capture.receipt_ids[0])
        self.assertEqual(page_evidence["receipt"]["body"]["raw_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(
            page_evidence["receipt"]["body"]["metadata"]["source_manifest_event_sha256"],
            collection.manifest_event_sha256,
        )
        self.assertEqual(page_evidence["receipt"]["recorded_at"], "2026-10-01T17:56:00+08:00")
        self.assertEqual(ledger.verify()["verified"], True)

    def test_late_close_page_is_persisted_but_is_not_eligible_for_its_named_decision(self):
        raw = _raw_page([_row(1)])
        collection = self._collect(
            raws=[raw],
            received_at="2026-10-01 17:51:00+08:00",
        )
        self.assertIsInstance(collection, PreparedRawEastmoneyCollection)
        ledger = self._ledger("2026-10-01 17:56:00+08:00")

        capture = capture_eastmoney_eod_snapshot(
            ledger=ledger,
            journal=self.journal,
            collection=collection,
        )

        self.assertEqual(capture.status, "abstain")
        self.assertEqual(
            capture.abstain_reason,
            "not_eligible_for_decision_at:close_page_received_outside_window",
        )
        self.assertEqual(len(capture.receipt_ids), 1)
        evidence = ledger.export_receipt_evidence(capture.receipt_ids[0])
        self.assertEqual(evidence["receipt"]["recorded_at"], "2026-10-01T17:56:00+08:00")
        self.assertEqual(evidence["receipt"]["body"]["first_captured_at"], "2026-10-01T17:51:00+08:00")
        self.assertTrue(ledger.verify()["verified"])

    def test_failed_collection_persists_partial_raw_pages_and_manifest_bound_failure_after_reopen(self):
        first_raw = _raw_page([_row(index) for index in range(1, 101)], total=101)
        collection = self._collect(
            raws=[first_raw, TimeoutError("fixture timeout")],
            received_at="2026-10-01 17:49:00+08:00",
        )
        self.assertIsInstance(collection, FailedRawEastmoneyCollection)
        ledger = self._ledger("2026-10-01 17:57:00+08:00")

        capture = capture_eastmoney_eod_snapshot(
            ledger=ledger,
            journal=self.journal,
            collection=collection,
        )

        self.assertEqual((capture.status, capture.abstain_reason), ("abstain", "page_fetch_failed"))
        self.assertEqual(len(capture.receipt_ids), 1)
        self.assertIsNotNone(capture.transport_receipt_id)
        transport = ledger.export_receipt_evidence(capture.transport_receipt_id or "")["receipt"]
        details = transport["body"]["metadata"]["transport"]
        self.assertEqual(details["status"], "failed")
        self.assertEqual(details["failure_reason"], "page_fetch_failed")
        self.assertEqual(details["source_manifest_sha256"], self.manifest.manifest_sha256)
        self.assertEqual(details["receipt_written_at"], "2026-10-01T17:57:00+08:00")
        self.assertEqual(len(details["attempts"]), 2)
        self.assertEqual(details["attempts"][0]["canonical_url"], self.manifest.canonical_url(1))
        self.assertEqual(details["attempts"][0]["raw_sha256"], hashlib.sha256(first_raw).hexdigest())
        self.assertEqual(details["attempts"][1]["fetch_error"], "TimeoutError")
        page = ledger.export_receipt_evidence(capture.receipt_ids[0])["receipt"]
        self.assertEqual(page["body"]["metadata"]["transport_receipt_id"], capture.transport_receipt_id)
        self.assertEqual(page["recorded_at"], "2026-10-01T17:57:00+08:00")
        reopened = ResearchSignalLedger(self.root / "research-signal-ledger")
        self.assertTrue(reopened.verify()["verified"])
        self.assertEqual(
            reopened.export_receipt_evidence(capture.transport_receipt_id or "")["receipt"]["body"]["receipt_role"],
            "transport_evidence",
        )

    def test_zero_attempt_failure_persists_a_reported_failure_envelope(self):
        # Reopening a manifest registered after this collection clock is a
        # failure before page one is opened.  It must leave a transport
        # receipt (with an empty attempt list), but never manufacture a raw
        # page receipt.
        self.journal.register(
            self.manifest,
            recorded_at="2026-10-01 16:30:00+08:00",
        )
        collection = self._collect(
            raws=[],
            received_at="2026-10-01 16:00:00+08:00",
        )
        self.assertIsInstance(collection, FailedRawEastmoneyCollection)
        self.assertEqual((collection.failure_reason, collection.attempts), ("registration_after_capture", ()))
        self.assertEqual(str(collection.reported_at), "2026-10-01 16:00:00+08:00")
        ledger = self._ledger("2026-10-01 17:57:00+08:00")

        capture = capture_eastmoney_eod_snapshot(
            ledger=ledger,
            journal=self.journal,
            collection=collection,
        )

        self.assertEqual(
            (capture.status, capture.abstain_reason),
            ("abstain", "not_eligible_for_decision_at:transport_failed_before_page_receipt"),
        )
        self.assertEqual(capture.receipt_ids, ())
        transport = ledger.export_receipt_evidence(capture.transport_receipt_id or "")["receipt"]
        details = transport["body"]["metadata"]["transport"]
        self.assertEqual(details["attempts"], [])
        self.assertEqual(details["failure_reported_at"], "2026-10-01T16:00:00+08:00")
        self.assertEqual(transport["recorded_at"], "2026-10-01T17:57:00+08:00")
        self.assertTrue(ledger.verify()["verified"])

    def test_after_decision_write_is_retained_but_cannot_backfill_that_decision(self):
        raw = _raw_page([_row(1)])
        collection = self._collect(
            raws=[raw],
            received_at="2026-10-01 17:49:00+08:00",
        )
        ledger = self._ledger("2026-10-01 18:01:00+08:00")

        capture = capture_eastmoney_eod_snapshot(
            ledger=ledger,
            journal=self.journal,
            collection=collection,
        )

        self.assertEqual(capture.status, "abstain")
        self.assertEqual(
            capture.abstain_reason,
            "not_eligible_for_decision_at:receipt_written_after_decision_cutoff",
        )
        self.assertEqual(len(capture.receipt_ids), 1)
        page = ledger.export_receipt_evidence(capture.receipt_ids[0])["receipt"]
        self.assertEqual(page["body"]["first_captured_at"], "2026-10-01T17:49:00+08:00")
        self.assertEqual(page["recorded_at"], "2026-10-01T18:01:00+08:00")
        self.assertNotEqual(page["body"]["first_captured_at"], page["recorded_at"])
        self.assertTrue(ledger.verify()["verified"])

    def test_each_page_uses_its_own_ledger_clock_and_a_crossing_write_abstains(self):
        first_raw = _raw_page([_row(index) for index in range(1, 101)], total=101)
        second_raw = _raw_page([_row(101)], total=101)
        collection = self._collect(
            raws=[first_raw, second_raw],
            received_at="2026-10-01 17:49:00+08:00",
        )
        # Transport evidence, page 1 and page 2 are distinct append-only
        # writes.  The second page crosses the decision cutoff even though
        # both HTTPS responses arrived within the close-market window.
        ledger = ResearchSignalLedger(
            self.root / "research-signal-ledger",
            now_fn=_SequenceClock(
                "2026-10-01 17:57:00+08:00",
                "2026-10-01 17:59:59+08:00",
                "2026-10-01 18:00:01+08:00",
            ),
        )

        capture = capture_eastmoney_eod_snapshot(
            ledger=ledger,
            journal=self.journal,
            collection=collection,
        )

        self.assertEqual(
            (capture.status, capture.abstain_reason),
            ("abstain", "not_eligible_for_decision_at:receipt_written_after_decision_cutoff"),
        )
        self.assertEqual(capture.ledger_recorded_at, "2026-10-01T17:57:00+08:00")
        page_writes = {
            ledger.export_receipt_evidence(receipt_id)["receipt"]["body"]["metadata"]["page_number"]:
            ledger.export_receipt_evidence(receipt_id)["receipt"]["recorded_at"]
            for receipt_id in capture.receipt_ids
        }
        self.assertEqual(page_writes, {
            1: "2026-10-01T17:59:59+08:00",
            2: "2026-10-01T18:00:01+08:00",
        })
        self.assertEqual(capture.decision_eligibility, "not_eligible_for_decision_at")
        self.assertTrue(ledger.verify()["verified"])

    def test_raw_pages_cannot_bypass_the_manifest_transport_bridge(self):
        ledger = self._ledger("2026-10-01 17:57:00+08:00")

        with self.assertRaises(TypeError):
            capture_eastmoney_eod_snapshot(  # type: ignore[call-arg] -- prohibited former API
                ledger=ledger,
                session=SESSION,
                decision_at=DECISION_AT,
                pages=(),
                collection_id="bypass",
                recorded_at="2026-10-01 17:57:00+08:00",
            )

        self.assertEqual(ledger.verify()["events"], 0)


if __name__ == "__main__":
    unittest.main()
