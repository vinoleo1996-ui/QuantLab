"""Price-factor reorder uses a name's own bars and leaves retention alone."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_price_factor_reference.py"
    spec = importlib.util.spec_from_file_location("simulate_price_factor_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PriceFactorReferenceTests(unittest.TestCase):
    def test_broken_board_ranks_below_an_advance(self):
        module = _load()
        scores = module.price_factor_priorities({
            "a": {"prior_limit_up_premium": 0, "broken_board_rate": 1, "advancing_breadth": 0},
            "b": {"prior_limit_up_premium": 0, "broken_board_rate": 0, "advancing_breadth": 1},
        })
        self.assertGreater(scores["b"], scores["a"])

    def test_limit_touch_without_a_seal_is_a_broken_board(self):
        module = _load()
        current = pd.Series({"close": 10.5, "high": 11.0, "limit_up": 11.0, "reference_price": 10.0, "limit_ratio_at_open": 0.1})
        previous = pd.Series({"close": 11.0, "high": 11.0, "limit_up": 11.0, "reference_price": 10.0, "limit_ratio_at_open": 0.1})
        facts = module.bar_components(current, previous)
        self.assertEqual(facts["prior_limit_up_premium"], 1.0)
        self.assertEqual(facts["broken_board_rate"], 1.0)
        self.assertEqual(facts["advancing_breadth"], 0.0)


if __name__ == "__main__":
    unittest.main()
