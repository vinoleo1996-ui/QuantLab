import unittest

import numpy as np
import pandas as pd

from quantlab.alpha.features import Fundamentals, Industries, PriceVolume, cross_section

DAYS = pd.bdate_range('2023-01-02', periods=320)
CODES = ['sh.600000', 'sz.000001', 'sz.000002']


def fields(seed=0):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, .02, (len(DAYS), 3)), axis=0)), index=DAYS, columns=CODES)
    return dict(close=close, adj_close=close * 2, high=close * 1.02, low=close * .98,
                amount=pd.DataFrame(1e8, index=DAYS, columns=CODES),
                turnover=pd.DataFrame(rng.uniform(.5, 3, (len(DAYS), 3)), index=DAYS, columns=CODES),
                float_mktcap=pd.DataFrame(5e9, index=DAYS, columns=CODES))


def fin_rows():
    rows = []
    for end, available, eps, profit in [('20221231', '2023-03-30 18:00', 1.0, 100.0),
                                        ('20230331', '2023-04-28 18:00', 0.3, 30.0),
                                        ('20220331', '2022-04-28 18:00', 0.2, 20.0)]:
        rows.append(dict(ts_code='600000.SH', end_date=end, available_at=f'{available}:00+08:00', basic_eps=eps,
                         n_income_attr_p=profit, revenue=profit * 10, bps=5.0, roe=8.0, grossprofit_margin=30.0,
                         debt_to_assets=40.0, ocf_to_or=0.1))
    rows.append(dict(rows[1], available_at='2023-06-01 18:00:00+08:00', basic_eps=0.9, n_income_attr_p=90.0))
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_future_prices_do_not_change_todays_features(self):
        day = DAYS[280]
        base = PriceVolume(fields()).on(day, CODES)
        changed = fields()
        for name in changed:
            changed[name] = changed[name].copy()
            changed[name].loc[changed[name].index > day] *= 3.0
        pd.testing.assert_frame_equal(base, PriceVolume(changed).on(day, CODES))
        self.assertTrue(base.notna().all().all())

    def test_financial_row_is_usable_from_its_available_time(self):
        fundamentals = Fundamentals(fin_rows())
        close = pd.Series({'sh.600000': 10.0})
        before = fundamentals.on(pd.Timestamp('2023-04-27'), ['sh.600000'], close).iloc[0]
        on_day = fundamentals.on(pd.Timestamp('2023-04-28'), ['sh.600000'], close).iloc[0]
        self.assertAlmostEqual(before.ep_ttm, 0.1)
        self.assertAlmostEqual(on_day.ep_ttm, (0.3 + 1.0 - 0.2) / 10.0)
        self.assertAlmostEqual(on_day.profit_yoy, 0.5)

    def test_later_restatement_is_not_seen_early(self):
        fundamentals = Fundamentals(fin_rows())
        close = pd.Series({'sh.600000': 10.0})
        may = fundamentals.on(pd.Timestamp('2023-05-15'), ['sh.600000'], close).iloc[0]
        june = fundamentals.on(pd.Timestamp('2023-06-02'), ['sh.600000'], close).iloc[0]
        self.assertAlmostEqual(may.profit_yoy, 0.5)
        self.assertAlmostEqual(june.profit_yoy, 3.5)

    def test_after_hours_row_waits_for_next_day(self):
        rows = fin_rows().iloc[:1].assign(available_at='2023-03-30 18:01:00+08:00')
        fundamentals = Fundamentals(rows)
        close = pd.Series({'sh.600000': 10.0})
        self.assertTrue(np.isnan(fundamentals.on(pd.Timestamp('2023-03-30'), ['sh.600000'], close).iloc[0].ep_ttm))
        self.assertAlmostEqual(fundamentals.on(pd.Timestamp('2023-03-31'), ['sh.600000'], close).iloc[0].ep_ttm, 0.1)

    def test_industry_membership_respects_out_date(self):
        table = pd.DataFrame(dict(instrument_id=['sh.600000', 'sh.600000', 'sz.000001'],
                                  sw_code=['A', 'B', 'A'], in_date=['20200101', '20230301', '20200101'],
                                  out_date=['20230301', None, None],
                                  available_at=['2020-01-01 18:00:00+08:00'] * 3))
        industries = Industries(table)
        self.assertEqual(industries.on('2023-02-28')['sh.600000'], 'A')
        self.assertEqual(industries.on('2023-03-01')['sh.600000'], 'B')

    def test_cross_section_has_every_family(self):
        data = fields()
        price = PriceVolume(data)
        table = pd.DataFrame(dict(instrument_id=CODES, sw_code=['A', 'A', 'B'], in_date='20200101', out_date=None,
                                  available_at='2020-01-01 18:00:00+08:00'))
        frame = cross_section(DAYS[300], CODES, price, Fundamentals(fin_rows()), Industries(table), data['close'])
        self.assertEqual(len(frame), 3)
        self.assertIn('ep_ttm', frame)
        self.assertIn('rel_ind_ret_20', frame)


if __name__ == '__main__':
    unittest.main()
