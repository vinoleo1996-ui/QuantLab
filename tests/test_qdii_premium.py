import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import p4_alloc as R


NAV = pd.DataFrame([
    # 513100 is split on 2022-01-14: unit NAV is not restated, adj NAV is.
    {'ts_code': '513100.SH', 'ann_date': '20200101', 'nav_date': '20191231', 'unit_nav': 1.0, 'adj_nav': 1.0},
    {'ts_code': '513100.SH', 'ann_date': '20220113', 'nav_date': '20220112', 'unit_nav': 1.0, 'adj_nav': 1.0},
    {'ts_code': '513100.SH', 'ann_date': '20220114', 'nav_date': '20220113', 'unit_nav': 1.0, 'adj_nav': 0.2},
    {'ts_code': '513100.SH', 'ann_date': '20220124', 'nav_date': '20220121', 'unit_nav': 1.0, 'adj_nav': 0.2},
    {'ts_code': '513500.SH', 'ann_date': '20200101', 'nav_date': '20191231', 'unit_nav': 1.0, 'adj_nav': 1.0},
])
BARS = pd.DataFrame([
    {'ts_code': '513100.SH', 'trade_date': '20200101', 'close': 1.0, 'adj_factor': 1.0},
    {'ts_code': '513100.SH', 'trade_date': '20220113', 'close': 1.0, 'adj_factor': 1.0},
    {'ts_code': '513100.SH', 'trade_date': '20220114', 'close': 0.2, 'adj_factor': 1.0},
    {'ts_code': '513100.SH', 'trade_date': '20220124', 'close': 0.2, 'adj_factor': 1.0},
    {'ts_code': '513500.SH', 'trade_date': '20200101', 'close': 1.0, 'adj_factor': 1.0},
    {'ts_code': '513500.SH', 'trade_date': '20200214', 'close': 1.05, 'adj_factor': 1.0},
])


def fake_snapshot(name):
    return (NAV if name == 'qdii_nav' else BARS).copy()


class QdiiPremium(unittest.TestCase):
    def frame(self, basis):
        sessions = pd.bdate_range('2020-01-01', '2022-12-30')
        with patch('quantlab.cloud.load_snapshot', side_effect=fake_snapshot):
            return R.qdii_premium_frame(sessions, basis=basis)

    def test_legacy_split_artifact_is_masked(self):
        frame = self.frame('legacy')
        self.assertGreaterEqual(frame.attrs['masked']['nasdaq'], 1)
        self.assertAlmostEqual(frame.loc['2022-01-14', 'nasdaq'], 0.0, places=3)

    def test_adjusted_basis_removes_split_artifact(self):
        frame = self.frame('adjusted')
        self.assertAlmostEqual(frame.loc['2022-01-14', 'nasdaq'], 0.0, places=3)
        self.assertAlmostEqual(frame.loc['2022-01-24', 'nasdaq'], 0.0, places=3)
        self.assertEqual(frame.attrs['masked']['nasdaq'], 0)
        self.assertEqual(frame.attrs['basis'], 'adjusted')

    def test_real_premium_is_preserved(self):
        frame = self.frame('adjusted')
        self.assertAlmostEqual(frame.loc['2020-02-14', 'sp500'], 0.05, places=3)

    def test_unknown_basis_is_rejected(self):
        with self.assertRaises(ValueError):
            self.frame('bogus')


if __name__ == '__main__':
    unittest.main()
