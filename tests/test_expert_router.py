"""Behavior of the research-only, causal expert mixture."""
import importlib.util
import unittest
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "router_under_test", Path(__file__).resolve().parents[1] / "quantlab/strategy/expert_router.py")
router_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = router_module
spec.loader.exec_module(router_module)

import numpy as np
import pandas as pd


class ExpertRouterTests(unittest.TestCase):
    def test_missing_labels_abstain_and_do_not_become_zero_returns(self):
        from router_under_test import ExpertRouter, RouterPolicy, cohort_from_outcomes
        x = pd.DataFrame({'a': np.linspace(.01, 1, 100), 'b': np.linspace(1, .01, 100)})
        y = x.a.copy()
        y.iloc[:20] = np.nan
        row = cohort_from_outcomes('2020-01-01', '2020-02-01', 'normal', x, y)
        self.assertEqual(row.status, 'insufficient_label_coverage')
        decision = ExpertRouter([row], RouterPolicy(min_cohorts=1)).decide('2020-03-01', 'normal', x)
        self.assertEqual(decision.status, 'insufficient_matured_history')
        self.assertFalse(decision.weights)
        self.assertTrue(decision.scores.empty)

    def test_unrealised_label_is_unavailable_until_endpoint(self):
        from router_under_test import ExpertRouter, RouterPolicy, cohort_from_outcomes
        x = pd.DataFrame({'a': np.linspace(.01, 1, 100), 'b': np.linspace(1, .01, 100)})
        row = cohort_from_outcomes('2020-01-01', '2020-02-01', 'normal', x, x.a)
        router = ExpertRouter([row], RouterPolicy(min_cohorts=1))
        self.assertEqual(router.decide('2020-01-31', 'normal', x).status, 'insufficient_matured_history')
        result = router.decide('2020-02-01', 'normal', x)
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(sum(result.weights.values()), 1)
        self.assertTrue(all(0 <= v <= 1 for v in result.weights.values()))

    def test_reference_retention_financial_tiers_and_unknown_slots_are_preserved(self):
        from router_under_test import select_reference_basket
        heads = dict(A=1., B=1., C=1., D=0., E=0.)
        sectors = {c:c for c in heads}
        # B has no score and keeps its middle slot. D/E cannot cross the financial tier.
        chosen = select_reference_basket(heads, (), sectors, priorities=dict(A=.1,C=.9,D=1.,E=.8),
                                          holdings=2, retention_rank=3)
        self.assertEqual(chosen, ('C', 'B'))
        retained = select_reference_basket(heads, ('A',), sectors, priorities=dict(A=.1,C=.9,D=1.),
                                           holdings=2, retention_rank=3)
        self.assertEqual(retained, ('A', 'B'))
        original = select_reference_basket(heads, (), sectors, holdings=2, retention_rank=3)
        self.assertEqual(original, ('A', 'B'))

    def test_solver_failure_never_creates_an_equal_weight_signal(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from router_under_test import ExpertRouter, RouterPolicy, cohort_from_outcomes
        x = pd.DataFrame({'a': np.linspace(.01, 1, 100), 'b': np.linspace(1, .01, 100)})
        row = cohort_from_outcomes('2020-01-01','2020-02-01','normal',x,x.a)
        with patch('router_under_test.minimize', return_value=SimpleNamespace(
                x=np.array([.5,.5]),success=False,message='test failure')):
            value=ExpertRouter([row],RouterPolicy(min_cohorts=1)).decide('2020-02-02','normal',x)
        self.assertEqual(value.status,'solver_failed')
        self.assertTrue(value.scores.empty)

    def test_future_expert_schema_and_outcomes_cannot_change_past_decision(self):
        from router_under_test import ExpertRouter, RouterPolicy, cohort_from_outcomes
        x = pd.DataFrame({'a': np.linspace(.01, 1, 100), 'b': np.linspace(1, .01, 100)})
        p = RouterPolicy(min_cohorts=1)
        old = cohort_from_outcomes('2020-01-01', '2020-02-01', 'normal', x, x.a)
        future = cohort_from_outcomes('2030-01-01', '2030-02-01', 'normal',
                                      x.rename(columns={'a': 'new_future_expert'}), -x.a)
        a = ExpertRouter([old], p).decide('2020-02-02', 'normal', x)
        b = ExpertRouter([old, future], p).decide('2020-02-02', 'normal', x)
        self.assertEqual(a.weights, b.weights)
        pd.testing.assert_series_equal(a.scores, b.scores)

    def test_market_context_changes_learned_weights_and_stock_ranking(self):
        self.assertIsNotNone(router_module,
                             'The contextual routing algorithm is not implemented')
        from router_under_test import ExpertRouter, cohort_from_outcomes
        x = pd.DataFrame({'trend': np.linspace(.01, 1, 100),
                          'defensive': np.linspace(1, .01, 100)})
        history = []
        for i, day in enumerate(pd.date_range('2020-01-01', periods=24, freq='MS')):
            state = 'active' if i % 2 else 'defensive'
            y = x.trend if state == 'active' else x.defensive
            history.append(cohort_from_outcomes(day, day + pd.Timedelta(days=20),
                                                state, x, y))
        router = ExpertRouter(history)
        up = router.decide('2022-02-01', 'active', x)
        down = router.decide('2022-02-01', 'defensive', x)
        self.assertEqual(up.status, 'ok')
        self.assertGreater(up.weights['trend'], up.weights['defensive'])
        self.assertLess(down.weights['trend'], down.weights['defensive'])
        self.assertEqual(up.scores.idxmax(), 99)
        self.assertEqual(down.scores.idxmax(), 0)


if __name__ == '__main__':
    unittest.main()
