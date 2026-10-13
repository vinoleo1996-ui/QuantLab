import unittest

import numpy as np
import pandas as pd

from quantlab.p4_alloc import allocate, target_weights

DAYS = pd.bdate_range('2020-01-01', periods=400)


def etf(seed=0):
    rng = np.random.default_rng(seed)
    scales = {'gold': .01, 'nasdaq': .015, 'sp500': .012, 'treasury': .002}
    return pd.DataFrame({k: rng.normal(.0002, s, len(DAYS)) for k, s in scales.items()}, index=DAYS)


class AllocTests(unittest.TestCase):
    def test_equity_share_fixed_and_low_vol_etf_gets_most_of_the_rest(self):
        w = target_weights(etf().iloc[:200], 0.60)
        self.assertAlmostEqual(w[0], 0.60)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertEqual(int(np.argmax(w[1:])), 3)

    def test_flat_world_keeps_nav_at_one_minus_costs_only(self):
        flat = pd.DataFrame(0.0, index=DAYS, columns=['gold', 'nasdaq', 'sp500', 'treasury'])
        nav, trace = allocate(pd.Series(1.0, index=DAYS), flat)
        self.assertAlmostEqual(nav.iloc[-1], 1.0, places=6)
        self.assertEqual(trace['disaster_triggers'], 0)

    def test_disaster_line_cuts_equity_and_recovers(self):
        equity = pd.Series(1.0, index=DAYS)
        equity.iloc[30:60] = np.linspace(1.0, 0.45, 30)
        equity.iloc[60:] = np.linspace(0.45, 1.2, len(DAYS) - 60)
        flat = pd.DataFrame(0.0, index=DAYS, columns=['gold', 'nasdaq', 'sp500', 'treasury'])
        nav, trace = allocate(equity, flat)
        self.assertEqual(trace['disaster_triggers'], 1)
        self.assertLess(trace['mean_weights']['equity'], 0.60)

    def test_qdii_above_premium_threshold_cannot_grow(self):
        flat = pd.DataFrame(0.0, index=DAYS, columns=['gold', 'nasdaq', 'sp500', 'treasury'])
        premium = pd.DataFrame({'nasdaq': 0.10, 'sp500': 0.0}, index=DAYS)
        rising = pd.Series(np.linspace(1.0, 3.0, len(DAYS)), index=DAYS)
        nav, trace = allocate(rising, flat, premium=premium)
        self.assertGreater(trace['qdii_pauses'], 0)
        self.assertLess(trace['mean_weights']['nasdaq'], trace['mean_weights']['sp500'])
        self.assertAlmostEqual(sum(trace['mean_weights'].values()), 1.0, places=3)

    def test_rebalance_uses_only_past_returns(self):
        base = etf()
        changed = base.copy()
        changed.iloc[300:] *= 5
        equity = pd.Series(np.linspace(1, 2, len(DAYS)), index=DAYS)
        a, _ = allocate(equity, base)
        b, _ = allocate(equity, changed)
        pd.testing.assert_series_equal(a.iloc[:300], b.iloc[:300])


if __name__ == '__main__':
    unittest.main()
