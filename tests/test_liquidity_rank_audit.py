"""Liquidity-rank audit stays a membership check and does not form holdings."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "liquidity_rank_audit.py"
    spec = importlib.util.spec_from_file_location("liquidity_rank_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiquidityRankAuditTests(unittest.TestCase):
    def test_short_universe_is_not_bought_and_financials_stay_empty(self):
        module = _load()
        records, status = module.band_audit([(6_000_000.0, "000001.SZ")])
        self.assertEqual(records, [])
        self.assertEqual(status["status"], "not_bought")
        self.assertFalse(status["financial_tier_used"])

    def test_amount_floor_does_not_backfill_and_ranks_stay_sorted(self):
        module = _load()
        observations = [(6_000_000.0 if index <= 4000 else 1_000_000.0, f"{index:06d}.SZ") for index in range(1, 5201)]
        records, status = module.band_audit(observations)
        selected = [row for row in records if row["in_liquidity_band"]]
        self.assertTrue(status["rank_order_ok"])
        self.assertTrue(all(row["financial_score"] is None and row["financial_available_at"] is None for row in records))
        self.assertTrue(all(row["portfolio_selected"] is False for row in records))
        self.assertLess(len(selected), 2000)
        self.assertTrue(all(later["liquidity_rank"] >= earlier["liquidity_rank"] for earlier, later in zip(selected, selected[1:])))

    def test_cadence_walks_both_ways_from_the_anchor(self):
        module = _load()
        calendar = pd.bdate_range("2018-01-02", periods=80)
        days = module.cadence_days(calendar, anchor=calendar[40], start=calendar[0], end=calendar[-1], step=20)
        self.assertEqual(days[0], pd.Timestamp(calendar[0]))
        self.assertIn(pd.Timestamp(calendar[40]), days)
        self.assertEqual(days[2], pd.Timestamp(calendar[40]))
        self.assertLessEqual(days[-1], pd.Timestamp(calendar[-1]))


if __name__ == "__main__":
    unittest.main()
