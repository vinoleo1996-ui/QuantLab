import unittest
from types import SimpleNamespace

import pandas as pd

from quantlab.broad import dividend_yield, month_ends, top_pool


def panel(days, codes, **fields):
    frames = {name: pd.DataFrame(values, index=days, columns=codes, dtype=float) for name, values in fields.items()}
    return SimpleNamespace(field=frames.__getitem__)


class BroadTest(unittest.TestCase):
    def test_pool_uses_same_day_cap_and_excludes_st_bj_and_short_history(self):
        days = pd.date_range('2020-01-01', periods=3)
        codes = ['sh.1', 'sz.2', 'bj.3', 'sh.4']
        p = panel(days, codes,
                  tradestatus=[[1, 1, 1, 1]] * 3, volume=[[1, 1, 1, 1]] * 3,
                  is_st=[[0, 0, 0, 1]] * 3,
                  float_mktcap=[[1, 2, 9, 9], [1, 2, 9, 9], [3, 2, 9, 9]])
        self.assertEqual(top_pool(p, days[-1:], size=1, min_history=3)[days[-1]], ['sh.1'])
        self.assertEqual(top_pool(p, days[1:2], size=1, min_history=2)[days[1]], ['sz.2'])
        self.assertEqual(top_pool(p, days[1:2], size=5, min_history=3)[days[1]], [])

    def test_dividend_yield_trailing_year_and_later_bonus_dilution(self):
        days = pd.to_datetime(['2021-06-30', '2022-06-30'])
        close = pd.DataFrame({'sh.1': [10.0, 10.0]}, index=days)
        actions = pd.DataFrame({'instrument_id': ['sh.1', 'sh.1', 'sh.1'],
                                'ex_date': ['2020-05-01', '2021-01-10', '2021-05-10'],
                                'cash_per_share': [9.0, 0.4, 0.0], 'share_ratio': [0.0, 0.0, 1.0]})
        result = dividend_yield(actions, close, days)
        self.assertAlmostEqual(result.loc[days[0], 'sh.1'], 0.02)
        self.assertAlmostEqual(result.loc[days[1], 'sh.1'], 0.0)

    def test_month_ends(self):
        sessions = pd.to_datetime(['2020-01-30', '2020-01-31', '2020-02-27', '2020-02-28'])
        self.assertEqual(list(month_ends(sessions, '2020-01-01', '2020-02-27')),
                         list(pd.to_datetime(['2020-01-31', '2020-02-27'])))


if __name__ == '__main__':
    unittest.main()
