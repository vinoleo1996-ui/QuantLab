"""The 253-session QuantDB seed builds U0 and keeps ST names out of the ranking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quantlab.data.quantdb_u0_seed import (
    QuantDBPartition,
    QuantDBU0SeedError,
    assemble_quantdb_u0_rows,
    build_quantdb_u0_snapshot,
    resolve_forward_anchor,
)
from quantlab.research_signal_ledger import ResearchSignalLedger


def _sessions(count: int = 253) -> list[pd.Timestamp]:
    return list(pd.bdate_range("2024-01-02", periods=count))


def _partitions(sessions: list[pd.Timestamp], codes: list[str]) -> list[QuantDBPartition]:
    frame = pd.DataFrame({"instrument_id": codes, "amount": 600.0, "close": 10.0})
    return [
        QuantDBPartition(session, frame.copy(), f"partition-{index}", "a" * 64)
        for index, session in enumerate(sessions)
    ]


class QuantDBU0SeedTests(unittest.TestCase):
    def test_anchor_is_the_first_open_session_on_or_after_the_declared_date(self):
        self.assertIsNone(resolve_forward_anchor(["2026-09-24", "2026-09-28", "2026-09-29"]))
        anchor = resolve_forward_anchor(["2026-09-29", "2026-09-30", "2026-10-09"])
        self.assertEqual(anchor, pd.Timestamp("2026-09-30"))

    def test_short_history_is_rejected(self):
        sessions = _sessions(252)
        with self.assertRaises(QuantDBU0SeedError):
            assemble_quantdb_u0_rows(
                partitions=_partitions(sessions, ["sz.000001"]),
                identity=pd.DataFrame({"is_st": [False], "is_delisted": [False], "name": ["普通"]}, index=["sz.000001"]),
                provenance={
                    "source_receipt_id": "manifest",
                    "source_receipt_hash": "b" * 64,
                    "source_published_at": "2026-09-30 16:00:00+08:00",
                    "source_first_captured_at": "2026-09-30 16:00:00+08:00",
                    "source_available_at": "2026-09-30 16:00:00+08:00",
                },
            )

    def test_wan_amount_becomes_cny_and_u0_excludes_st(self):
        sessions = _sessions()
        codes = [f"sz.{index:06d}" for index in range(3010)]
        identity = pd.DataFrame(
            {
                "is_st": [code == "sz.003009" for code in codes],
                "is_delisted": False,
                "name": ["ST样本" if code == "sz.003009" else "普通" for code in codes],
            },
            index=codes,
        )
        provenance = {
            "source_receipt_id": "manifest-1",
            "source_receipt_hash": "c" * 64,
            "source_published_at": "2026-09-30 16:10:00+08:00",
            "source_first_captured_at": "2026-09-30 16:10:00+08:00",
            "source_available_at": "2026-09-30 16:10:00+08:00",
        }
        rows = assemble_quantdb_u0_rows(
            partitions=_partitions(sessions, codes),
            identity=identity,
            provenance=provenance,
        )
        self.assertEqual(float(rows.loc[rows.instrument_id.eq("sz.000001"), "adv60_cny"].iloc[0]), 6_000_000.0)
        self.assertFalse(bool(rows.loc[rows.instrument_id.eq("sz.003009"), "base_eligible"].iloc[0]))
        decision = sessions[-1]
        captured_at = pd.Timestamp(decision.date()).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
        recorded_at = pd.Timestamp(decision.date()).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=17)
        with tempfile.TemporaryDirectory() as temp:
            ledger = ResearchSignalLedger(Path(temp) / "ledger", now_fn=lambda: recorded_at)
            recorded = ledger.record_receipt(
                source_id="quantdb.u0-seed",
                source_family="quantdb",
                origin_id="quantdb:u0-seed:manifest",
                canonical_url="https://www.quantdb.cn/api/v1/data/download",
                raw=b'{"partitions":[]}',
                content_type="application/json",
                published_at=captured_at,
                first_captured_at=captured_at,
                available_at=captured_at,
                availability_basis="first_capture_of_quantdb_response_no_vendor_publish_time",
            )
            evidence = ledger.export_receipt_evidence(recorded["event_id"])
            rows["source_receipt_id"] = evidence["canonical_receipt_id"]
            rows["source_receipt_hash"] = evidence["raw"]["sha256"]
            rows["source_published_at"] = captured_at
            rows["source_first_captured_at"] = captured_at
            rows["source_available_at"] = evidence["receipt"]["available_at"]
            snapshot = build_quantdb_u0_snapshot(
                ledger=ledger,
                rows=rows,
                session=decision,
                decision_at=pd.Timestamp(decision.date()).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=18),
                cadence_anchor=sessions[12],
                exchange_sessions=sessions,
            )
        self.assertEqual(snapshot.status, "ok", snapshot.abstain_reason)
        self.assertEqual(len(snapshot.members), 9)
        self.assertNotIn("sz.003009", snapshot.members)


if __name__ == "__main__":
    unittest.main()
