import unittest
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('selection_under_test',
    Path(__file__).resolve().parents[1] / 'quantlab/strategy/expert_router.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
select_reference_basket = module.select_reference_basket


class SelectionPolicyTests(unittest.TestCase):
    def test_consistent_rank_retains_high_score_outside_old_code_cutoff(self):
        heads = {str(i): 1. for i in range(8)}
        priorities = {str(i): i / 10 for i in range(8)}
        got = select_reference_basket(heads, ('0', '6'), {}, priorities=priorities,
            holdings=2, retention_rank=3, industry_slots=2,
            selection_policy='score_consistent_retention')
        self.assertEqual(set(got), {'6', '7'})
        old = select_reference_basket(heads, ('0', '6'), {}, priorities=priorities,
            holdings=2, retention_rank=3, industry_slots=2)
        self.assertEqual(set(old), {'0', '7'})

    def test_disabled_policy_matches_reference_and_financial_tiers_stay_first(self):
        heads = {'a': 2., 'b': 2., 'c': 1., 'd': 1.}
        scores = {'a': .1, 'b': .2, 'c': .99, 'd': .98}
        args = dict(priorities=scores, holdings=2, retention_rank=2, industry_slots=2)
        base = select_reference_basket(heads, (), {}, **args)
        explicit = select_reference_basket(heads, (), {}, **args,
            selection_policy='reference_retention')
        self.assertEqual(base, explicit)
        consistent = select_reference_basket(heads, (), {}, **args,
            selection_policy='score_consistent_retention')
        self.assertEqual(consistent, ('b', 'a'))

    def test_missing_scores_keep_slots_and_industry_cap_applies(self):
        heads = {c: 1. for c in 'abcde'}
        got = select_reference_basket(heads, (), {'e': 'X', 'a': 'X'},
            priorities={'a': .1, 'c': .2, 'e': .9}, holdings=3,
            retention_rank=3, industry_slots=1,
            selection_policy='score_consistent_retention')
        self.assertEqual(got, ('e', 'b'))  # unknown industry is one shared group

    def test_unknown_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown_selection_policy'):
            select_reference_basket({'a': 1.}, (), {}, selection_policy='typo')


if __name__ == '__main__':
    unittest.main()
