import importlib.util
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


class LocalAdapterUsesExchangeRules(unittest.TestCase):
    def test_main_board_st_limit_follows_rule_switch(self):
        from quantlab.data.local_adapter import normalize_panel
        dates = pd.bdate_range('2026-06-25', '2026-07-10')
        raw = pd.DataFrame(dict(open=10., high=10., low=10., close=10., amount=1e8, volume=1e7,
                                reference_price=10., feature_close=10.), index=dates)
        states = pd.DataFrame(dict(effective_on=dates, is_st=True, is_suspended=False, source_name='t'))
        p, _ = normalize_panel('sh.600000', raw, states, {}, dates)
        before, after = p.loc['2026-07-03'], p.loc['2026-07-06']
        self.assertEqual(before.limit_ratio_at_open, .05)
        self.assertEqual(before.limit_up, 10.5)
        self.assertEqual(after.limit_ratio_at_open, .10)
        self.assertEqual(after.limit_up, 11.)
        self.assertTrue(p.limit_ratio_at_open.iloc[:5].isna().all())

    def test_star_and_chinext_unchanged(self):
        from quantlab.data.local_adapter import normalize_panel
        dates = pd.bdate_range('2020-08-10', '2020-09-04')
        raw = pd.DataFrame(dict(open=10., high=10., low=10., close=10., amount=1e8, volume=1e7,
                                reference_price=10., feature_close=10.), index=dates)
        states = pd.DataFrame(dict(effective_on=dates, is_st=True, is_suspended=False, source_name='t'))
        p, _ = normalize_panel('sz.300001', raw, states, {}, dates)
        self.assertEqual(p.loc['2020-08-21'].limit_ratio_at_open, .05)
        self.assertEqual(p.loc['2020-08-24'].limit_ratio_at_open, .20)
        q, _ = normalize_panel('sh.688001', raw, states, {}, dates)
        self.assertEqual(q.loc['2020-08-21'].limit_ratio_at_open, .20)


class SmallcapPanelRestatesStLimits(unittest.TestCase):
    def test_stale_five_percent_rows_after_switch_become_ten_percent(self):
        import numpy as np
        from quantlab.smallcap.panel import _restate_st_limits
        dates = pd.to_datetime(['2026-07-03', '2026-07-06'])
        frame = pd.DataFrame(dict(limit_ratio_at_open=[.05, .05], reference_price=[10., 10.]), index=dates)
        out = pd.DataFrame(dict(is_st=[1., 1.], limit_up=[10.5, 10.5], limit_down=[9.5, 9.5]), index=dates)
        fixed = _restate_st_limits('sh.600000', frame, out.copy())
        self.assertEqual(fixed.loc['2026-07-03'].limit_up, 10.5)
        self.assertEqual(fixed.loc['2026-07-06'].limit_up, 11.)
        self.assertEqual(fixed.loc['2026-07-06'].limit_down, 9.)
        star = _restate_st_limits('sh.688001', frame.assign(limit_ratio_at_open=.20), out.copy())
        self.assertTrue(np.allclose(star.limit_up, 10.5))


class BeijingCodePrefix(unittest.TestCase):
    def test_92_codes_are_beijing(self):
        from quantlab.data_os.identity import a_share_identity
        self.assertEqual(a_share_identity('920001')['exchange'], 'BJSE')
        self.assertEqual(a_share_identity('830001')['exchange'], 'BJSE')
        self.assertEqual(a_share_identity('600000')['exchange'], 'XSHG')
        self.assertEqual(a_share_identity('900901')['exchange'], 'XSHG')
        self.assertEqual(a_share_identity('000001')['exchange'], 'XSHE')


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


@unittest.skipUnless(importlib.util.find_spec('requests'), 'requests not installed')
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
