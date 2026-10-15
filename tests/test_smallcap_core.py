"""Invariants of the small-cap simulator and candidate pools."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab.smallcap import universe
from quantlab.smallcap.sim import Costs, Market, simulate


def market(opens, closes, up=None, down=None, last_bar=None, no_limits=()):
    opens, closes = np.asarray(opens, float), np.asarray(closes, float)
    n, k = opens.shape
    nan = np.full((n, k), np.nan)
    return Market(pd.bdate_range('2024-01-02', periods=n), pd.Index([f'c{j}' for j in range(k)]),
                  opens, closes, np.ones((n, k)), nan if up is None else np.asarray(up, float),
                  nan if down is None else np.asarray(down, float), np.full((n, k), np.inf),
                  last_bar or {}, set(no_limits))


class SimulatorInvariants(unittest.TestCase):
    def test_no_signal_keeps_cash(self):
        result = simulate(market([[10.] * 2] * 5, [[10.] * 2] * 5), {})
        self.assertTrue(np.allclose(result['nav'], 600_000.))
        self.assertEqual(result['fills'], 0)

    def test_fill_at_next_open_with_slippage_lots_and_commission(self):
        m = market([[10.]] * 3, [[10.]] * 3)
        result = simulate(m, {m.dates[0]: {'c0': 0.5}})
        notional = 30_000 * 10.03
        expected = 600_000. - notional - notional * 0.00025 + 30_000 * 10.
        self.assertAlmostEqual(result['nav'].iloc[1], expected, places=6)
        self.assertEqual(result['fills'], 1)

    def test_limit_up_open_blocks_buy_and_limit_down_open_blocks_sell(self):
        m = market([[10.]] * 3, [[10.]] * 3, up=[[10.]] * 3, down=[[9.]] * 3)
        self.assertEqual(simulate(m, {m.dates[0]: {'c0': 0.5}})['fills'], 0)
        m = market([[10.], [10.], [9.], [9.]], [[10.]] * 4, up=[[11.]] * 4, down=[[9.]] * 4)
        result = simulate(m, {m.dates[0]: {'c0': 0.5}, m.dates[1]: {}})
        self.assertEqual(result['fills'], 1)

    def test_extra_columns_ignore_limits(self):
        m = market([[10.]] * 3, [[10.]] * 3, up=[[10.]] * 3, no_limits=('c0',))
        self.assertEqual(simulate(m, {m.dates[0]: {'c0': 0.5}})['fills'], 1)

    def test_gross_never_exceeds_cap(self):
        m = market([[10.] * 2] * 4, [[10.] * 2] * 4)
        result = simulate(m, {m.dates[0]: {'c0': 0.6, 'c1': 0.6}})
        self.assertLessEqual(result['gross'].max(), Costs().gross_cap + 1e-9)

    def test_delisted_name_is_liquidated_with_haircut(self):
        m = market([[10.]] * 4, [[10.]] * 4)
        m.last_bar = {0: m.dates[1]}
        result = simulate(m, {m.dates[0]: {'c0': 0.5}})
        self.assertEqual(result['events'][0]['type'], 'delisted_liquidation')
        self.assertAlmostEqual(result['events'][0]['value'], 30_000 * 10. * 0.5)


def features(n, st=()):
    codes = [f'{i:06d}' for i in range(n)]
    day = pd.Timestamp('2024-01-02')
    adv = pd.DataFrame([[1e9 - i * 1e5 for i in range(n)]], index=[day], columns=codes)
    history = pd.DataFrame([[300] * n], index=[day], columns=codes)
    flags = pd.DataFrame([[1. if c in st else 0. for c in codes]], index=[day], columns=codes)
    return day, codes, {'adv': adv, 'history': history, 'cap': adv, 'st': flags}


class PoolDefinitions(unittest.TestCase):
    def test_absolute_band_shrinks_when_the_market_is_small(self):
        day, codes, f = features(3200)
        self.assertEqual(universe.pool_on(day, f, 'adv'), codes[3000:3200])
        self.assertEqual(len(universe.pool_on(day, f, 'adv', band='relative')), 3200 - 1920)
        day, codes, f = features(2999)
        self.assertEqual(universe.pool_on(day, f, 'adv'), [])
        self.assertEqual(len(universe.pool_on(day, f, 'adv', band='relative')), 2999 - 1800)

    def test_relative_band_matches_the_audit_rule(self):
        import smallcap_audit as AU
        day, _, f = features(4321)
        self.assertEqual(universe.pool_on(day, f, 'adv', band='relative'), AU.relative_pool_on(day, f, 'adv'))

    def test_st_is_removed_before_ranking(self):
        day, codes, f = features(100, st={'000099', '000010'})
        pool = universe.pool_on(day, f, 'adv', band='relative', exclude_st=True)
        self.assertNotIn('000099', pool)
        self.assertEqual(len(pool), 98 - 59)
        self.assertEqual(pool[0], codes[60])

    def test_adv_domains_cut_the_tradable_list_into_three_bands(self):
        day, codes, f = features(100)
        groups = universe.domains_on(day, f, exclude_st=True, min_history=252)
        self.assertEqual(groups['liquid'], codes[:40])
        self.assertEqual(groups['mid'], codes[40:70])
        self.assertEqual(groups['micro'], codes[70:])
        self.assertEqual(groups['liquid'][0], codes[0])

    def test_domains_drop_st_and_beijing_before_the_cut(self):
        day, codes, f = features(100, st={'000010'})
        f['adv']['bj.430001'] = 1e12
        f['history']['bj.430001'] = 300
        f['st']['bj.430001'] = 0.0
        f['cap']['bj.430001'] = 1e12
        groups = universe.domains_on(day, f, exclude_st=True, min_history=252)
        pooled = groups['liquid'] + groups['mid'] + groups['micro']
        self.assertNotIn('000010', pooled)
        self.assertNotIn('bj.430001', pooled)
        self.assertEqual(len(pooled), 99)
        self.assertEqual(groups['liquid'][0], codes[0])

    def test_min_history_is_a_parameter(self):
        day, codes, f = features(100)
        f['history'].iloc[0, 99] = 252
        self.assertNotIn(codes[99], universe.pool_on(day, f, 'adv', band='relative'))
        self.assertIn(codes[99], universe.pool_on(day, f, 'adv', band='relative', min_history=252))


if __name__ == '__main__':
    unittest.main()
