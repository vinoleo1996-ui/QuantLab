"""Entry-order difference between the liquidity book and a ranked refill."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_theme_history.py"
    spec = importlib.util.spec_from_file_location("simulate_theme_history", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThemeHistoryTargetTests(unittest.TestCase):
    def test_retained_names_stay_and_openings_follow_the_rank(self):
        module = _load()
        self.assertEqual(module.rebalance_codes(("a", "c"), ["b", "a"]), ("a", "c", "b"))
        self.assertEqual(module.book_targets(["b", "a"], ["a", "c", "b", "d"], limit=3), ("a", "b", "c"))
        self.assertEqual(module.performance([100, 110], ["2025-01-01", "2026-01-01"])["cagr"] > 0, True)

    def test_derived_open_carries_the_status_time_the_ledger_requires(self):
        from quantlab.backtest.ledger import trade_permission

        module = _load()
        bar = module.derived_open_bar(
            day="2025-09-15", open_price=10.2, reference=10.0, ratio=0.1, is_st=False, listing_session=60,
        )
        allowed, reason = trade_permission(
            bar, module._rule("sz.000001", 0.1), "buy", "2025-09-15", allow_open_assumption=True,
        )
        self.assertEqual((allowed, reason), (True, ""))

    def test_full_window_candidate_keeps_a_stable_twelve_percent_book(self):
        module = _load()
        kept = module.candidate_decision(0.16, -0.21, [0.13, 0.20, 0.08])
        self.assertEqual(kept["candidate_retained"], True)
        self.assertEqual(kept["promotion_approved"], False)
        below = module.candidate_decision(0.11, -0.20, [0.10, 0.12, 0.09])
        self.assertEqual(below["candidate_retained"], False)
        negative = module.candidate_decision(0.14, -0.20, [0.13, 0.20, -0.01])
        self.assertEqual(negative["candidate_retained"], False)
        deep = module.candidate_decision(0.19, -0.33, [0.12, 0.18, 0.11])
        self.assertEqual(deep["candidate_retained"], False)


if __name__ == "__main__":
    unittest.main()
