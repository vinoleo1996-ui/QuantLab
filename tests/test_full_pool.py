"""The full pool buys without a 3001-name minimum and is not the old tail."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import full_pool as pool


class FullPoolTests(unittest.TestCase):
    def test_spacing_covers_both_ends_of_a_short_universe(self):
        ranked = [f'{i:06d}.SZ' for i in range(2797)]
        industries = {code: str(i % 40) for i, code in enumerate(ranked)}
        chosen = pool.select_full_pool(ranked, industries, current=())
        self.assertEqual(len(chosen), 50)
        self.assertIn(ranked[0], chosen)
        self.assertIn(ranked[-1], chosen)
        self.assertLess(ranked.index(chosen[1]), 2797 // 2)

    def test_january_grid_is_a_decision_day(self):
        calendar = pd.bdate_range('2018-01-02', periods=40)
        days = pool.decision_days(calendar, start=calendar[0], end=calendar[-1], step=20)
        self.assertEqual(days[0], pd.Timestamp(calendar[0]))
        self.assertIn(pd.Timestamp(calendar[20]), days)

    def test_held_names_stay_when_they_remain_in_the_pool(self):
        ranked = [f'{i:06d}.SZ' for i in range(100)]
        industries = {code: code for code in ranked}
        first = pool.select_full_pool(ranked, industries, current=())
        second = pool.select_full_pool(ranked, industries, current=first)
        self.assertEqual(second, first)


if __name__ == '__main__':
    unittest.main()
