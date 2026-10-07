import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quantlab.portfolio import grok_paper


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture(root: Path) -> None:
    """Create the four fixed ledger artifacts without any replay fallback."""

    root.mkdir(parents=True, exist_ok=True)
    current_rows = []
    initial_rows = []
    for offset in range(50):
        code = f"sh.{600000 + offset:06d}"
        current_rows.append({
            "code": code,
            "name": f"股票{offset}",
            "implied_shares": 100.0,
            "original_shares": 100,
            "last_price": 10.0,
            "adj_factor": 1.0,
            "market_value": 1000.0,
            "weight": 1000.0 / 50100.0,
            "cost_basis_total": 950.0,
            "unrealized_pnl": 50.0,
            "return_since_entry": 50.0 / 950.0,
            "entry_date": "2026-09-22",
            "entry_price": 9.5,
            "flags_today": "",
        })
        initial_rows.append({
            "code": code,
            "name": f"股票{offset}",
            "board": "SH main",
            "industry_csrc2012": f"C{offset:02d} 测试行业{offset}",
            "shares": 100,
            "entry_date": "2026-09-22",
            "entry_price": 9.5,
            "market_value": 950.0,
            "buy_cost": 0.0,
            "cost_basis_total": 950.0,
            "cost_per_share_incl_fees": 9.5,
            "weight_of_capital": 950.0 / 60000.0,
            "seed1_fill_order": offset + 1,
            "entry_rank_by_20d_amount": offset + 1,
            "adv20_amount_cny": 1_000_000.0,
            "listing_date": "2020-01-01",
            "red_flags_at_entry": "",
        })

    _write_csv(root / "current_positions.csv", current_rows, list(current_rows[0]))
    _write_csv(root / "initial_positions_2026-09-22.csv", initial_rows, list(initial_rows[0]))
    (root / "current_state.json").write_text(json.dumps({
        "account": "paper_qn_backband_N50_seed1",
        "as_of": "2026-09-28",
        "nav": 50100.0,
        "cash": 100.0,
        "invested": 50000.0,
        "n_positions": 50,
        "positions": current_rows,
    }), encoding="utf-8")
    (root / "initial_state_2026-09-22.json").write_text(json.dumps({
        "account": "paper_qn_backband_N50_seed1",
        "as_of": "2026-09-22",
        "capital": 60000.0,
        "cash": 100.0,
        "invested_value": 47500.0,
        "nav": 47600.0,
        "n_positions": 50,
        "N": 50,
        "positions": initial_rows,
        "source": "/fixture/seed1_initial_N50.csv",
        "source_sha256": "a" * 64,
    }), encoding="utf-8")


class GrokPaperLedgerTests(unittest.TestCase):
    def test_loads_complete_fixed_ledger_and_joins_build_time_industry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ledger"
            _write_fixture(root)

            ledger = grok_paper.load_grok_paper_ledger(root)

        self.assertTrue(ledger["displayable"])
        self.assertEqual(ledger["snapshot_name"], "grok_paper_ledger")
        self.assertEqual(ledger["source_kind"], "grok_paper_ledger")
        self.assertEqual(ledger["status"], "paper_simulation_unverified")
        self.assertEqual(ledger["as_of"], "2026-09-28")
        self.assertEqual(ledger["summary"]["holdings_count"], 50)
        self.assertEqual(len(ledger["rows"]), 50)
        first = ledger["rows"][0]
        self.assertEqual(first["code"], "sh.600000")
        self.assertEqual(first["public_code"], "600000")
        self.assertEqual(first["quantity"], 100.0)
        self.assertEqual(first["cost"], 9.5)
        self.assertEqual(first["cost_basis_total"], 950.0)
        self.assertEqual(first["mark"], 10.0)
        self.assertEqual(first["price"], 10.0)
        self.assertEqual(first["entry_date"], "2026-09-22")
        self.assertEqual(first["build_time_industry"], "C00 测试行业0")
        self.assertEqual(first["industry"], "C00 测试行业0")
        source = ledger["provenance"]["files"]["current_positions"]
        self.assertEqual(source["path"], str(root / "current_positions.csv"))
        self.assertEqual(len(source["sha256"]), 64)

    def test_uses_the_fixed_default_grok_ledger_root_when_not_injected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "grokbot_isolated" / "paper" / "ledger"
            _write_fixture(root)
            with patch.object(grok_paper, "DEFAULT_LEDGER_ROOT", root):
                ledger = grok_paper.load_grok_paper_ledger()

        self.assertTrue(ledger["displayable"])
        self.assertEqual(ledger["provenance"]["ledger_root"], str(root))

    def test_fails_closed_when_current_position_count_is_not_fifty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ledger"
            _write_fixture(root)
            with (root / "current_positions.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            _write_csv(root / "current_positions.csv", rows[:-1], list(rows[0]))

            ledger = grok_paper.load_grok_paper_ledger(root)

        self.assertFalse(ledger["displayable"])
        self.assertEqual(ledger["status"], "unavailable")
        self.assertEqual(ledger["rows"], [])
        self.assertIn("current_positions_count", ledger["errors"][0])

    def test_fails_closed_when_build_time_industry_schema_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ledger"
            _write_fixture(root)
            with (root / "initial_positions_2026-09-22.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            fieldnames = [name for name in rows[0] if name != "industry_csrc2012"]
            _write_csv(root / "initial_positions_2026-09-22.csv", rows, fieldnames)

            ledger = grok_paper.load_grok_paper_ledger(root)

        self.assertFalse(ledger["displayable"])
        self.assertEqual(ledger["rows"], [])
        self.assertIn("initial_positions_schema", ledger["errors"][0])

    def test_preserves_an_absent_build_time_industry_as_missing_not_a_guess(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ledger"
            _write_fixture(root)
            with (root / "initial_positions_2026-09-22.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["industry_csrc2012"] = ""
            _write_csv(root / "initial_positions_2026-09-22.csv", rows, list(rows[0]))
            state_path = root / "initial_state_2026-09-22.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["positions"][0]["industry_csrc2012"] = ""
            state_path.write_text(json.dumps(state), encoding="utf-8")

            ledger = grok_paper.load_grok_paper_ledger(root)

        self.assertTrue(ledger["displayable"])
        self.assertIsNone(ledger["rows"][0]["build_time_industry"])
        self.assertEqual(ledger["rows"][0]["build_time_industry_status"], "missing_in_build_ledger")

    def test_missing_ledger_never_uses_a_weight_or_results_pickle_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ledger"
            root.mkdir()
            (root / "results.pkl").write_bytes(b"not a ledger")

            ledger = grok_paper.load_grok_paper_ledger(root)

        self.assertFalse(ledger["displayable"])
        self.assertEqual(ledger["rows"], [])
        self.assertEqual(ledger["status"], "unavailable")
        self.assertIn("source_missing:current_positions", ledger["errors"][0])


if __name__ == "__main__":
    unittest.main()
