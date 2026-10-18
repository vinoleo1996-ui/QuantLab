import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from p5_base_orders import industry_map_at


class OrderIndustryPit(unittest.TestCase):
    def test_future_reclassification_is_not_used(self):
        scores = pd.DataFrame([
            {'date': '2024-01-02', 'code': 'sz.000001', 'industry': '银行'},
            {'date': '2024-01-02', 'code': 'sz.000002', 'industry': '电子'},
            {'date': '2024-06-03', 'code': 'sz.000001', 'industry': '计算机'},
        ])
        self.assertEqual(industry_map_at(scores, '2024-02-01'),
                         {'sz.000001': '银行', 'sz.000002': '电子'})
        self.assertEqual(industry_map_at(scores, '2024-06-03')['sz.000001'], '计算机')

    def test_no_snapshot_before_signal_returns_empty(self):
        scores = pd.DataFrame([{'date': '2024-01-02', 'code': 'sz.000001', 'industry': '银行'}])
        self.assertEqual(industry_map_at(scores, '2023-12-29'), {})


if __name__ == '__main__':
    unittest.main()
