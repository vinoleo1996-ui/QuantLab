import unittest

import pandas as pd

from quantlab.etf_account import replay


def bars(rows):
    return pd.DataFrame(rows, columns=['date', 'code', 'open_raw', 'open_hfq', 'close_hfq']).assign(
        date=lambda d: pd.to_datetime(d.date))


class EtfAccountTest(unittest.TestCase):
    def test_next_open_lots_fees_and_distribution(self):
        data = bars([('2020-01-01', 'A', 1.0, 1.0, 1.0), ('2020-01-02', 'A', 1.0, 1.0, 1.0),
                     ('2020-01-03', 'A', 0.9, 1.0, 1.1)])
        result = replay(data, {'2020-01-01': {'A': 0.5}}, '2020-01-01', '2020-01-03', cash=10_000)
        nav = result['nav']
        self.assertEqual(nav.iloc[0], 10_000)
        fee = 5.0 + 5000 * 0.0005
        self.assertAlmostEqual(nav.iloc[1], 10_000 - fee)
        self.assertAlmostEqual(nav.iloc[2], 5000 - fee + 5000 * 1.1)

    def test_sells_before_buys_and_exit(self):
        data = bars([(d, c, 1.0, 1.0, 1.0) for d in ('2020-01-01', '2020-01-02', '2020-01-03') for c in 'AB'])
        result = replay(data, {'2020-01-01': {'A': 1.0}, '2020-01-02': {'B': 0.9}}, '2020-01-01', '2020-01-03',
                        cash=10_000)
        self.assertEqual(result['trades'], 3)
        self.assertLess(result['nav'].iloc[-1], 10_000)
        self.assertGreater(result['nav'].iloc[-1], 9_950)


    def test_missing_session_keeps_last_close(self):
        data = bars([('2020-01-01', 'A', 1.0, 1.0, 1.0), ('2020-01-01', 'B', 1.0, 1.0, 1.0),
                     ('2020-01-02', 'A', 1.0, 1.0, 1.0), ('2020-01-02', 'B', 1.0, 1.0, 1.0),
                     ('2020-01-03', 'B', 1.0, 1.0, 1.0), ('2020-01-06', 'A', 1.0, 1.0, 1.0),
                     ('2020-01-06', 'B', 1.0, 1.0, 1.0)])
        nav = replay(data, {'2020-01-01': {'A': 0.5}}, '2020-01-01', '2020-01-06', cash=10_000)['nav']
        self.assertAlmostEqual(nav.loc['2020-01-03'], nav.loc['2020-01-02'])


if __name__ == '__main__':
    unittest.main()
