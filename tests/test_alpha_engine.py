import unittest

import numpy as np
import pandas as pd

from quantlab.alpha.labels import forward_open_returns
from quantlab.alpha.models import GBDTEnsemble, LinearICIR, percentile_frame
from quantlab.alpha.portfolio import random_order, schedule, select

DAYS = pd.bdate_range('2024-01-01', periods=30)


def price_fields():
    close = pd.DataFrame({'a': np.linspace(10, 13, 30), 'b': 10.0}, index=DAYS)
    return dict(close=close, adj_close=close * 2, open=close.copy(), tradestatus=pd.DataFrame(1.0, index=DAYS, columns=['a', 'b']),
                limit_up=close * 1.1)


def synthetic(n_days=60, n_codes=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.bdate_range('2020-01-01', periods=n_days):
        signal = rng.normal(size=n_codes)
        rows.append(pd.DataFrame(dict(date=day, code=[f'c{i}' for i in range(n_codes)], good=signal,
                                      noise=rng.normal(size=n_codes),
                                      label=signal + rng.normal(scale=2.0, size=n_codes))))
    frame = pd.concat(rows, ignore_index=True)
    frame = percentile_frame(frame, ['good', 'noise'])
    frame['label'] = frame.groupby('date')['label'].rank(pct=True)
    return frame


class LabelTests(unittest.TestCase):
    def test_label_is_open_to_open_over_twenty_sessions(self):
        fields = price_fields()
        label = forward_open_returns(fields)
        expected = fields['open'].a.iloc[21] / fields['open'].a.iloc[1] - 1
        self.assertAlmostEqual(label.a.iloc[0], expected)
        self.assertTrue(np.isnan(label.a.iloc[-1]))

    def test_suspended_or_limit_up_entry_is_unlabeled(self):
        fields = price_fields()
        fields['tradestatus'].loc[DAYS[1], 'a'] = 0.0
        fields['open'].loc[DAYS[2], 'b'] = fields['limit_up'].loc[DAYS[2], 'b']
        label = forward_open_returns(fields)
        self.assertTrue(np.isnan(label.a.iloc[0]))
        self.assertTrue(np.isnan(label.b.iloc[1]))
        self.assertFalse(np.isnan(label.b.iloc[0]))


class ModelTests(unittest.TestCase):
    def test_linear_weights_follow_the_informative_feature(self):
        model = LinearICIR().fit(synthetic(), ['good', 'noise'])
        self.assertGreater(model.weights['good'], 1.0)
        self.assertLess(abs(model.weights['noise']), 0.5)

    def test_gbdt_learns_the_signal_out_of_sample(self):
        train, test = synthetic(seed=1), synthetic(n_days=10, seed=2)
        model = GBDTEnsemble().fit(train, ['good', 'noise'], seeds=(1, 2))
        score = model.predict(test)
        self.assertGreater(np.corrcoef(score, test['label'])[0, 1], 0.2)


class PortfolioTests(unittest.TestCase):
    def test_held_names_inside_buffer_are_kept_and_industry_cap_holds(self):
        ordered = [f's{i}' for i in range(200)]
        industries = {code: 'big' if i < 30 else f'i{i % 20}' for i, code in enumerate(ordered)}
        chosen = select(ordered, industries, current=['s150', 's90'], holdings=50, buffer=100, cap=12)
        self.assertIn('s90', chosen)
        self.assertNotIn('s150', chosen)
        self.assertEqual(sum(industries[c] == 'big' for c in chosen), 12)
        self.assertEqual(len(chosen), 50)

    def test_hysteresis_keeps_names_in_top_half_and_enters_only_from_top_decile(self):
        from quantlab.alpha.portfolio import select_hysteresis
        ordered = [f's{i}' for i in range(1000)]
        industries = {c: f'i{i % 20}' for i, c in enumerate(ordered)}
        current = ['s400', 's499', 's500', 's900']
        chosen = select_hysteresis(ordered, industries, current, holdings=50)
        self.assertIn('s400', chosen)
        self.assertIn('s499', chosen)
        self.assertNotIn('s500', chosen)
        self.assertNotIn('s900', chosen)
        self.assertEqual(len(chosen), 50)
        self.assertTrue(all(int(c[1:]) < 100 for c in chosen if c not in current))
        again = select_hysteresis(ordered, industries, chosen, holdings=50)
        self.assertEqual(set(again), set(chosen))

    def test_schedule_is_equal_weight_within_gross(self):
        names = [f's{i}' for i in range(80)]
        out = schedule(['d1'], lambda day: names, lambda day: {c: f'i{k % 8}' for k, c in enumerate(names)})
        self.assertEqual(len(out['d1']), 50)
        self.assertEqual(len(schedule(['d1'], lambda day: names, lambda day: {})['d1']), 12)
        self.assertAlmostEqual(sum(out['d1'].values()), 0.90)

    def test_random_order_is_stable_per_seed(self):
        codes = [f's{i}' for i in range(20)]
        self.assertEqual(random_order(codes, 1), random_order(list(reversed(codes)), 1))
        self.assertNotEqual(random_order(codes, 1), random_order(codes, 2))


if __name__ == '__main__':
    unittest.main()
