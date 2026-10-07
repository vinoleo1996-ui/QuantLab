import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab import research_trials as T
from quantlab.research_statistics import newey_west, win_rates


class TrialLedger(unittest.TestCase):
    def test_append_only_counts_and_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / 'ledger.csv'
            T.register('a', 'x', ledger=ledger)
            T.register('a', 'y', role='diagnostic', confirm_uses=1, ledger=ledger)
            T.register('b', 'x', ledger=ledger)
            with self.assertRaises(ValueError):
                T.register('a', 'x', ledger=ledger)
            T.record_outcome('a', 'x', 'pass', ledger=ledger)
            self.assertEqual(T.count('a', ledger=ledger), 2)
            self.assertEqual(T.count('a', ledger=ledger, include_diagnostics=False), 1)
            self.assertEqual(T.count(ledger=ledger), 3)
            self.assertEqual(T.confirm_uses('a', ledger=ledger), 1)
            self.assertEqual(len(T.rows(ledger)), 4)
            with self.assertRaises(ValueError):
                T.register('a', 'z', outcome='great', ledger=ledger)


class WinRatesAndNeweyWest(unittest.TestCase):
    def test_win_rates_three_definitions(self):
        days = pd.bdate_range('2020-01-01', '2021-12-31')
        r = pd.Series(np.where(days.month % 2 == 0, 0.001, -0.0005), index=days)
        out = win_rates(r, benchmark=pd.Series(0.0, index=days))
        self.assertEqual(out['months'], 24)
        self.assertAlmostEqual(out['monthly_positive'], 0.5)
        self.assertAlmostEqual(out['monthly_beat'], 0.5)
        self.assertEqual(out['rolling_12m_positive'], 1.0)

    def test_newey_west_mean_and_slope(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000)
        y = 0.5 + 2 * x + rng.normal(scale=0.1, size=2000)
        beta, t = newey_west(y, x)
        self.assertAlmostEqual(beta[1], 2, places=1)
        self.assertGreater(t[0], 50)
        beta, t = newey_west(rng.normal(size=2000))
        self.assertLess(abs(t[0]), 4)


if __name__ == '__main__':
    unittest.main()
