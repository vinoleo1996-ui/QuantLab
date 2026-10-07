import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference_financial_priority', ROOT/'scripts/reference_financial_priority.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FinancialPriorityTests(unittest.TestCase):
    def test_unknown_is_not_economic_failure(self):
        self.assertEqual(m.financial_state(None, 2), 'unknown')
        self.assertEqual(m.financial_state(float('nan'), 2), 'unknown')
        self.assertEqual(m.financial_state(0, 3), 'pass')
        self.assertEqual(m.financial_state(-1, 2), 'fail')
        self.assertEqual(m.financial_state(1, 0), 'fail')
        self.assertEqual(m.financial_state(1, 3.01), 'fail')
        # Known disproof is recorded even when the other field is unavailable.
        self.assertEqual(m.financial_state(-1, None), 'fail')

    def test_common_coverage_keeps_observed_failures_and_excludes_missing(self):
        inputs = dict(heads={'2022-01-20':{'a':1., 'b':0., 'c':0., 'd':0.}},
            industries={'2022-01-20':dict.fromkeys('abcd','sector')},
            financial_snapshots=[['a','2022-01-20',3,2], ['b','2022-01-20',-2,2],
                                 ['c','2022-01-20',None,2], ['d','2022-01-20',-2,None]])
        heads, groups, coverage = m.common_inputs(inputs)
        self.assertEqual(heads, {'2022-01-20':{'a':1.,'b':0.}})
        self.assertEqual(set(groups['2022-01-20']), {'a','b'})
        self.assertEqual(coverage[0]['known_both'], 2)
        self.assertEqual(coverage[0]['observed_fail'], 2)
        self.assertEqual(coverage[0]['unknown'], 1)
        inputs['heads']['2022-01-20']['a'] = 0.
        with self.assertRaisesRegex(ValueError, 'archived_financial_score'):
            m.common_inputs(inputs)

    def test_only_financial_preference_changes_targets_and_future_is_irrelevant(self):
        codes = [f'{i:06d}.SZ' for i in range(250)]
        groups = dict(zip(codes, (str(i%20) for i in range(250))))
        values = {s:float(i%2) for i,s in enumerate(codes)}
        day = '2022-01-20'
        heads, industries = {day:values}, {day:groups}
        a = m.make_pair(heads, industries, [day])
        self.assertNotEqual(a['financial'][0]['weights'], a['neutral'][0]['weights'])
        self.assertEqual(len(a['financial'][0]['weights']),100)
        self.assertEqual(len(a['neutral'][0]['weights']),100)
        self.assertTrue(all(values[s] == 1 for s in a['financial'][0]['weights']))
        all_same = m.make_pair({day:dict.fromkeys(codes,1.)}, industries, [day])
        self.assertEqual(all_same['financial'], all_same['neutral'])
        future = '2022-02-21'
        b = m.make_pair(dict(heads, **{future:dict.fromkeys(codes,0.)}),
                        dict(industries, **{future:groups}), [day,future])
        for arm in a:
            self.assertEqual(a[arm][0], b[arm][0])

    def test_coverage_shortage_is_shared_without_future_backfill(self):
        day='2019-03-01'; codes=[f'{i:06d}.SZ' for i in range(37)]
        result=m.make_pair({day:dict.fromkeys(codes,0.)},
            {day:{s:str(i%5) for i,s in enumerate(codes)}}, [day])
        self.assertEqual(result['financial'], result['neutral'])
        self.assertEqual(len(result['financial'][0]['weights']),37)
        self.assertAlmostEqual(sum(result['financial'][0]['weights'].values()),.9)


if __name__ == '__main__': unittest.main()
