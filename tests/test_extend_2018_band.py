"""Earlier liquidity ranks stay on the frozen rule and carry into the saved heads."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import continuous_band as band
import extend_2018_band as early


class Extend2018BandTests(unittest.TestCase):
    def test_grid_stops_before_the_frozen_anchor(self):
        calendar = pd.bdate_range('2017-01-02', '2019-03-01')
        # The real exchange calendar is checked in prepare; this only checks the walk.
        days = early.early_decision_days(calendar, anchor=calendar[-1], step=20, start=calendar[0])
        self.assertNotIn(calendar[-1], days)
        self.assertEqual((calendar[-1] - days[-1]).days > 0, True)

    def test_rank_keeps_3001_to_5000_and_refuses_a_short_universe(self):
        observations = [(float(6000 - i), f'{i:06d}.SZ') for i in range(6000)]
        # amount is 6000, 5999, ... so the highest amount sorts first.
        # Give them all at least 5 million by shifting.
        observations = [(amount + 5_000_000, symbol) for amount, symbol in observations]
        chosen = early.liquidity_band(observations)
        self.assertEqual(len(chosen), 2000)
        self.assertEqual(chosen[0], observations[3000][1])
        self.assertEqual(chosen[-1], observations[4999][1])
        self.assertIsNone(early.liquidity_band(observations[:3000]))

    def test_amount_floor_removes_names_without_relaxing_the_rank(self):
        observations = [(10_000_000 - i, f'{i:06d}.SZ') for i in range(3000)]
        kept = (5_000_000.0, '003000.SZ')
        observations.append(kept)
        observations.extend((100.0, f'{i:06d}.SZ') for i in range(3001, 4000))
        chosen = early.liquidity_band(observations)
        self.assertEqual(chosen, (kept[1],))

    def test_early_holdings_carry_into_the_frozen_head(self):
        codes = [f'{i:06d}.SZ' for i in range(80)]
        industries = {code: code for code in codes}
        heads = {
            '2018-12-26': {code: (1.0 if i >= 30 else 0.0) for i, code in enumerate(codes)},
            '2019-03-01': {code: 1.0 for code in codes},
        }
        groups = {day: dict(industries) for day in heads}
        carried = band.build_targets(pd.to_datetime(['2018-12-26', '2019-03-01']), heads, groups)
        restarted = band.build_targets(pd.to_datetime(['2019-03-01']), heads, groups)
        self.assertTrue(set(carried[0]['weights']) <= set(carried[1]['weights']))
        self.assertNotEqual(set(carried[1]['weights']), set(restarted[0]['weights']))


if __name__ == '__main__':
    unittest.main()
