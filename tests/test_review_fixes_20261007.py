import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


class StLimitRule(unittest.TestCase):
    def test_main_board_st_moves_to_ten_percent_on_2026_07_06(self):
        from quantlab.data.exchange_rules import limit_ratio
        self.assertEqual(limit_ratio('sh.600000', '2026-07-03', True, 100), .05)
        self.assertEqual(limit_ratio('sh.600000', '2026-07-06', True, 100), .10)
        self.assertEqual(limit_ratio('sz.000001', '2026-09-01', True, 100), .10)
        self.assertEqual(limit_ratio('sz.000001', '2026-07-03', False, 100), .10)

    def test_other_boards_unchanged(self):
        from quantlab.data.exchange_rules import limit_ratio
        self.assertEqual(limit_ratio('sz.300001', '2019-06-03', True, 100), .05)
        self.assertEqual(limit_ratio('sz.300001', '2026-07-06', True, 100), .20)
        self.assertEqual(limit_ratio('sh.688001', '2026-07-06', True, 100), .20)

    def test_ledger_accepts_both_rule_versions(self):
        from quantlab.data.exchange_rules import RULE_VERSION, RULE_VERSIONS
        self.assertIn('cn_daily_v1', RULE_VERSIONS)
        self.assertIn(RULE_VERSION, RULE_VERSIONS)


class _Store:
    def load(self, *args):
        return None

    def save(self, *args):
        pass


class AkshareFloatCapUnit(unittest.TestCase):
    def test_float_cap_in_yuan_and_six_billion_floor_applies(self):
        from quantlab.data.sources import DataFeed
        spot = pd.DataFrame({'代码': ['600001', '600002'], '名称': ['甲', '乙'], '最新价': [10., 10.],
                             '总市值': [9e9, 9e9], '流通市值': [7e9, 3e9], '成交额': [5e8, 5e8],
                             '换手率': [1., 1.], '涨跌幅': [0., 0.]})
        feed = DataFeed(store=_Store(), source='akshare')
        fake = SimpleNamespace(stock_zh_a_spot_em=lambda: spot.copy())
        with patch.object(DataFeed, '_ak', return_value=fake), \
                patch.object(DataFeed, 'fetch_industry_a', return_value=None):
            out = feed._a_candidates()
        self.assertEqual(list(out.code), ['600001'])
        self.assertEqual(float(out.float_mktcap.iloc[0]), 7e9)


class SinaFloatCapFallback(unittest.TestCase):
    def test_missing_float_cap_falls_back_to_total_cap_once(self):
        import quantlab.data.sina as sn
        rows = [{'code': f'{600000 + i}', 'name': f's{i}', 'trade': 10., 'settlement': 10., 'amount': 1e8,
                 'mktcap': 800000., 'nmc': (None if i == 0 else 500000.)} for i in range(60)]
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: rows)
        with patch.object(sn.requests, 'get', return_value=response), patch.object(sn.time, 'sleep'):
            out = sn.fetch_spot_candidates()
        self.assertEqual(float(out.loc['600000', 'float_mktcap']), 8e9)
        self.assertEqual(float(out.loc['600001', 'float_mktcap']), 5e9)
        self.assertEqual(float(out.loc['600000', 'mktcap']), 8e9)


if __name__ == '__main__':
    unittest.main()
