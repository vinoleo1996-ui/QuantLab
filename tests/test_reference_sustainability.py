import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference_sustainability', ROOT/'scripts/reference_sustainability.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReferenceOrderingTests(unittest.TestCase):
    def test_binary_priority_does_not_cross_financial_classes(self):
        source = {f'{i:06d}.SZ': float(i % 2) for i in range(100)}
        result = m.priority_scores(source)
        self.assertGreater(min(result[k] for k in source if source[k] == 1),
                           max(result[k] for k in source if source[k] == 0))
        self.assertEqual(result, m.priority_scores(dict(reversed(list(source.items())))))
        with self.assertRaises(ValueError): m.priority_scores({'x': .5})

    def test_order_among_present_names_ignores_future_names(self):
        source = {f'{i:06d}.SZ': float(i % 2) for i in range(30)}
        extended = dict(source, future_name=1.)
        ordered = lambda scores: sorted(scores, key=lambda k: (-scores[k], k))
        self.assertEqual(ordered(m.priority_scores(source)),
                         [k for k in ordered(m.priority_scores(extended)) if k in source])

    def test_archive_match_checks_dates_and_weights(self):
        target = [{'signal_date':'2022-01-20','weights':{'000001.SZ':.9},'sell_only':False,'max_exposure':.9}]
        m.require_targets_equal(target, {'targets':target})
        for bad in ([], [dict(target[0], weights={'000002.SZ':.9})],
                    [dict(target[0], signal_date='2022-01-21')]):
            with self.assertRaises(ValueError): m.require_targets_equal(bad, {'targets':target})

    def test_policy_keeps_industry_caps_and_retained_names(self):
        heads = {'2022-01-20':{f'{i:06d}.SZ':float(i % 2) for i in range(220)},
                 '2022-02-21':{f'{i:06d}.SZ':float(i % 2) for i in range(220)}}
        industries = {d:{s:str(int(s[:6]) % 11) for s in cross} for d,cross in heads.items()}
        schedules = m.make_targets(heads, industries, ['2022-01-20','2022-02-21'], neutral=True)
        from collections import Counter
        self.assertEqual(schedules[0]['weights'], schedules[1]['weights'])
        self.assertEqual(len(schedules[0]['weights']), 100)
        self.assertAlmostEqual(sum(schedules[0]['weights'].values()), .9)
        self.assertLessEqual(max(Counter(industries['2022-01-20'][s] for s in schedules[0]['weights']).values()),15)


if __name__ == '__main__': unittest.main()
