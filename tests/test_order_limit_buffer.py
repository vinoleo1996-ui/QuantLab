import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from p5_base_orders import limit_price


class OrderLimitBuffer(unittest.TestCase):
    def test_buy_buffer_rounds_down_to_cent(self):
        self.assertEqual(limit_price(10.00, 'buy', buffer=0.01), (10.10, None))

    def test_buy_is_capped_at_upper_limit(self):
        self.assertEqual(limit_price(10.00, 'buy', upper=10.05, buffer=0.02),
                         (10.05, 'rule_band'))

    def test_sell_is_capped_at_lower_limit(self):
        self.assertEqual(limit_price(10.00, 'sell', lower=9.90, buffer=0.02),
                         (9.90, 'rule_band'))

    def test_etf_tick_is_milliyuan(self):
        self.assertEqual(limit_price(4.000, 'buy', buffer=0.01, tick=0.001), (4.040, None))

    def test_missing_open_is_not_priced(self):
        price, reason = limit_price(float('nan'), 'buy')
        self.assertFalse(price == price)
        self.assertEqual(reason, 'no_open')


if __name__ == '__main__':
    unittest.main()
