import unittest

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.mandate import Mandate
from quantlab.strategy.legacy import allocate_legacy
from quantlab.strategy.risk import RiskManager
from quantlab.strategy.risk_policy import PolicyController


def market():
    dates = pd.bdate_range('2020-01-01', periods=500)
    changes = np.r_[np.tile([.001, -.001], 150), np.full(150, .002),
                    np.tile([-.035, .012], 25)]
    return pd.Series(100*np.exp(np.cumsum(changes)), index=dates)


class RiskPolicyTests(unittest.TestCase):
    def test_volatility_threshold_uses_prior_252_complete_observations(self):
        benchmark=market()
        policy=PolicyController('P2',Config(),benchmark)
        self.assertFalse(policy.market_snapshot(benchmark.index[271])['ready'])
        first=policy.market_snapshot(benchmark.index[272])
        self.assertTrue(first['ready'])
        changed=benchmark.copy();changed.iloc[272]*=2
        second=PolicyController('P2',Config(),changed).market_snapshot(benchmark.index[272])
        self.assertEqual(first['historical_vol_q80'],second['historical_vol_q80'])
        self.assertNotEqual(first['realized_vol_20'],second['realized_vol_20'])

    def test_p0_preserves_legacy_daily_risk(self):
        cfg, benchmark = Config(), market()
        old = RiskManager(cfg.risk, cfg.research, cfg.portfolio.target_vol)
        policy = PolicyController('P0', cfg, benchmark)
        for date, nav in zip(benchmark.index, [100, 90, 85, 82, 80, 90]):
            expected = old.evaluate(nav, date=date)
            actual = policy.evaluate(date, nav)
            for key, value in expected.items():
                self.assertEqual(actual.risk[key], value, key)
            self.assertEqual(actual.soft_stock_weight, .06)

    def test_fixed_relaxation_requires_explicit_mandate(self):
        cfg = Config()
        cfg.research.max_stock_weight = .08
        cfg.portfolio.max_sector_weight = .35
        cfg.research.max_cluster_weight = .35
        cfg.portfolio.target_vol = .18
        with self.assertRaisesRegex(ValueError, 'mandate_security_limit'):
            Mandate.from_config(cfg)
        mandate = Mandate.from_config(cfg, risk_policy='P1')
        self.assertEqual(mandate.risk_policy, 'P1')
        self.assertEqual(mandate.config().research.max_stock_weight, .08)
        self.assertNotEqual(Mandate.from_config(Config()).version,
                            Mandate.from_config(Config(), risk_policy='P1').version)
        cfg.research.max_stock_weight = .081
        with self.assertRaises(ValueError):
            Mandate.from_config(cfg, risk_policy='P1')

    def test_future_prices_do_not_change_past_budgets(self):
        full = market()
        shortened = full.iloc[:380]
        altered = full.copy()
        altered.iloc[380:] *= np.linspace(.1, 20, len(altered)-380)
        a = PolicyController('P2', Config(), shortened)
        b = PolicyController('P2', Config(), altered)
        for date in shortened.index[300:]:
            self.assertEqual(a.evaluate(date, 100).as_dict(), b.evaluate(date, 100).as_dict())

    def test_promotion_requires_five_sessions_and_duplicate_calls_do_not_count(self):
        close = market()
        policy = PolicyController('P2', Config(), close)
        for date in close.index[320:324]:
            self.assertEqual(policy.evaluate(date, 100).state, 'normal')
            self.assertEqual(policy.evaluate(date, 100).state, 'normal')
        budget = policy.evaluate(close.index[324], 100)
        self.assertEqual(budget.state, 'active')
        self.assertEqual(budget.max_stock_weight, .10)
        self.assertEqual(budget.max_cluster_weight, .35)

    def test_data_gap_blocks_increase_and_recovery_is_stepwise(self):
        close = market()
        policy = PolicyController('P2', Config(), close)
        budget = policy.evaluate(close.index[319], 100, exit_blocked=True)
        self.assertEqual(budget.state, 'defensive')
        self.assertFalse(budget.risk['allow_new_stock'])
        for date in close.index[320:324]:
            self.assertEqual(policy.evaluate(date, 100).state, 'defensive')
        self.assertEqual(policy.evaluate(close.index[324], 100).state, 'normal')
        for date in close.index[325:329]:
            self.assertEqual(policy.evaluate(date, 100).state, 'normal')
        self.assertEqual(policy.evaluate(close.index[329], 100).state, 'active')

    def test_market_demotion_requires_two_sessions_per_step(self):
        close = market()
        policy = PolicyController('P2', Config(), close)
        for date in close.index[320:325]:
            policy.evaluate(date, 100)
        defensive_dates = [date for date in close.index[450:]
                           if policy.market_snapshot(date)['desired_state'] == 'defensive']
        self.assertGreaterEqual(len(defensive_dates), 4)
        self.assertEqual(policy.evaluate(defensive_dates[0], 100).state, 'active')
        self.assertEqual(policy.evaluate(defensive_dates[1], 100).state, 'normal')
        self.assertEqual(policy.evaluate(defensive_dates[2], 100).state, 'normal')
        self.assertEqual(policy.evaluate(defensive_dates[3], 100).state, 'defensive')

    def test_incomplete_history_cannot_promote(self):
        close = market()
        policy = PolicyController('P2', Config(), close)
        for date in close.index[:100]:
            budget = policy.evaluate(date, 100)
            self.assertEqual(budget.state, 'normal')
            self.assertIn('market_history_incomplete', budget.risk['reasons'])

    def test_drawdown_25_requires_manual_restore_and_preserves_peak(self):
        close = market()
        policy = PolicyController('P1', Config(), close)
        policy.evaluate(close.index[0], 100)
        for i, (nav, level, gross) in enumerate([(88, 1, 1.), (82, 2, .60),
                                                 (78, 3, .30), (75, 4, 0.)], 1):
            budget = policy.evaluate(close.index[i], nav)
            self.assertEqual(budget.risk['level'], level)
            self.assertEqual(budget.gross_scale, gross)
        self.assertTrue(policy.evaluate(close.index[5], 90).risk['manual_review'])
        with self.assertRaises(ValueError):
            policy.approve_resume(approved_by='')
        policy.approve_resume(approved_by='operator')
        budget = policy.evaluate(close.index[6], 90)
        self.assertFalse(budget.risk['manual_review'])
        self.assertAlmostEqual(budget.risk['drawdown'], -.10)
        self.assertEqual(policy.risk_manager.peak, 100)

    def test_drawdown_blocks_market_promotion_and_recovery_requires_five_sessions(self):
        close = market()
        policy = PolicyController('P2', Config(), close)
        policy.evaluate(close.index[319], 100)
        for date in close.index[320:326]:
            budget = policy.evaluate(date, 88)
            self.assertEqual(budget.state, 'normal')
            self.assertEqual(budget.risk['level'], 1)
        for date in close.index[326:330]:
            self.assertEqual(policy.evaluate(date, 92).risk['level'], 1)
            self.assertEqual(policy.evaluate(date, 92).risk['level'], 1)
        self.assertEqual(policy.evaluate(close.index[330], 92).risk['level'], 0)

    def test_weekend_calls_do_not_advance_drawdown_recovery(self):
        policy = PolicyController('P1', Config(), market())
        policy.evaluate('2020-01-02', 100)
        policy.evaluate('2020-01-03', 88)
        self.assertEqual(policy.evaluate('2020-01-04', 92).risk['recovery_count'], 0)
        self.assertEqual(policy.evaluate('2020-01-05', 92).risk['recovery_count'], 0)
        self.assertEqual(policy.evaluate('2020-01-06', 92).risk['recovery_count'], 1)


class AllocationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.codes = [f's{i:02d}' for i in range(20)]
        random = np.random.default_rng(22)
        self.returns = pd.DataFrame(random.normal(size=(120, 20)) *
                                    np.r_[.0001, np.full(19, .005)], columns=self.codes)
        self.exposures = pd.DataFrame(np.eye(20), columns=self.codes)
        self.budget = PolicyController('P1', self.cfg, market()).evaluate('2021-01-01', 100)

    def test_explicit_p0_budget_preserves_previous_allocation(self):
        old, status, _ = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, exposures=self.exposures)
        p0 = PolicyController('P0', self.cfg, market()).evaluate('2021-01-01', 100)
        actual, other_status, _ = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, exposures=self.exposures, budget=p0)
        self.assertEqual((status, other_status), ('ok', 'ok'))
        np.testing.assert_allclose(old, actual, atol=1e-12)

    def test_relaxation_reaches_weights_above_old_relative_cap(self):
        old, old_status, old_detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, exposures=self.exposures)
        new, status, detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, exposures=self.exposures, budget=self.budget)
        self.assertEqual((old_status, status), ('ok', 'ok'))
        self.assertIsNone(detail['relative_weight_cap'])
        self.assertAlmostEqual(old_detail['relative_weight_cap'], .065)
        self.assertGreater(new.max(), .065*new.sum()+1e-5)
        self.assertLessEqual(new.max(), .07+1e-7)
        self.assertAlmostEqual(new.sum(), .98, places=6)

    def test_soft_stock_limit_does_not_force_existing_locked_holding_to_sell(self):
        prior = {self.codes[0]: .075}
        weights, status, detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, previous=prior, locked=prior,
            exposures=self.exposures, budget=self.budget)
        self.assertEqual(status, 'ok', detail)
        self.assertAlmostEqual(weights.iloc[0], .075, places=6)

    def test_hard_locked_breach_remains_explicitly_infeasible(self):
        prior = {self.codes[0]: .081}
        weights, status, detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, previous=prior, locked=prior,
            exposures=self.exposures, budget=self.budget)
        self.assertEqual(status, 'locked_holdings_violate_constraints')
        self.assertEqual(weights.sum(), 0)

    def test_soft_industry_limit_preserves_locked_existing_exposure(self):
        exposure = pd.DataFrame({c: {'shared' if i < 5 else c: 1.}
                                 for i, c in enumerate(self.codes)}).fillna(0.)
        prior = dict.fromkeys(self.codes[:5], .068)
        weights, status, detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, previous=prior, locked=prior,
            exposures=exposure, budget=self.budget)
        self.assertEqual(status, 'ok', detail)
        self.assertAlmostEqual(weights.iloc[:5].sum(), .34, places=6)

    def test_unknown_sector_is_not_relaxed_with_known_sectors(self):
        unknown = pd.DataFrame([np.ones(20)], index=['UNKNOWN'], columns=self.codes)
        weights, status, detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, exposures=unknown, budget=self.budget)
        self.assertEqual(status, 'ok', detail)
        self.assertAlmostEqual(weights.sum(), .23, places=6)

    def test_equal_weight_is_explicit_comparator_on_same_constraints(self):
        weights, status, detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, exposures=self.exposures, budget=self.budget,
            equal_weight=True)
        self.assertEqual(status, 'ok', detail)
        self.assertTrue(np.allclose(weights, .98/20, atol=1e-6))
        self.assertIn('equal_weight', detail['policy'])

    def test_increase_eligibility_preserves_old_holdings_but_blocks_new_buy(self):
        prior = {self.codes[0]: .065}
        weights, status, detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, previous=prior, locked=prior,
            exposures=self.exposures, budget=self.budget,
            allowed_increases=set(self.codes[2:]))
        self.assertEqual(status, 'ok', detail)
        self.assertAlmostEqual(weights.iloc[0], .065, places=6)
        self.assertEqual(weights.iloc[1], 0.)

    def test_unknown_downside_group_remains_capped(self):
        from quantlab.strategy.crowding import RiskSnapshot
        groups = RiskSnapshot('2021-01-01', dict.fromkeys(self.codes, 'UNKNOWN'),
                              pd.Series(dtype=float), pd.Series(dtype=float), [])
        weights, status, detail = allocate_legacy(self.returns, self.cfg,
            selected=self.codes, exposures=self.exposures, risk_snapshot=groups,
            budget=self.budget)
        self.assertEqual(status, 'ok', detail)
        self.assertAlmostEqual(weights.sum(), .28, places=6)


if __name__ == '__main__':
    unittest.main()
