import importlib.util
from pathlib import Path
import unittest
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/continuity_attribution.py'


class AttributionContracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'attribution implementation missing')
        spec = importlib.util.spec_from_file_location('continuity_attribution', SCRIPT)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_returns_use_exchange_sessions_not_next_nonmissing_observation(self):
        calendar = pd.date_range('2020-01-01', periods=5, freq='B')
        prices = pd.Series([10., 11., float('nan'), 13., 14.], index=calendar)
        self.assertEqual(self.m.forward_outcome(prices, calendar, calendar[0], 2),
                         (None, 'missing_price_path'))
        self.assertEqual(self.m.forward_outcome(prices, calendar, calendar[4], 2),
                         (None, 'end_censored'))
        value, reason = self.m.forward_outcome(prices, calendar, calendar[0], 1)
        self.assertAlmostEqual(value, .1)
        self.assertEqual(reason, 'observed')

    def test_known_action_gap_does_not_become_measured_alpha(self):
        calendar = pd.date_range('2020-01-01', periods=4, freq='B')
        prices = pd.Series([10., 9., 10., 12.], index=calendar)
        self.assertEqual(self.m.forward_outcome(prices, calendar, calendar[0], 2,
                                               [calendar[1]]),
                         (None, 'known_data_gap'))

    def test_realized_categories_and_repurchase_do_not_double_count_costs(self):
        snapshots = [dict(date=f'2020-01-0{i}', quantities=q) for i, q in
                     [(1, {}), (2, {'A':100}), (3, {'A':200}), (4, {}), (5, {'A':100})]]
        fills = [dict(date=f'2020-01-0{i}', instrument_id='A', side=side,
                      reason=reason, notional=100., fees=1., slippage_cost=2.)
                 for i, side, reason in [(2,'buy','legacy_repaired'),
                                        (3,'buy','legacy_repaired'),
                                        (4,'sell','risk_exit'),
                                        (5,'buy','legacy_repaired')]]
        rows = self.m.classify_fills(dict(account=snapshots, fills=fills))
        self.assertEqual([r['category'] for r in rows],
                         ['addition', 'weight_increase', 'risk_exit', 'repurchase'])
        self.assertEqual(rows[-1]['previous_exit_reason'], 'risk_exit')
        self.assertEqual(sum(r['cost'] for r in rows), 12.)

    def test_score_buckets_are_deterministic_and_preserve_ties(self):
        labels = self.m.score_buckets(pd.Series({'A':3., 'B':3., 'C':1., 'D':0.}), 2)
        self.assertEqual(labels['A'], labels['B'])
        self.assertGreater(labels['A'], labels['D'])

    def test_cash_causes_are_multilabel_and_no_candidate_not_inferred_from_missing(self):
        row = dict(flat=True, level=3, manual_review=False, allow_new_stock=False)
        labels = self.m.cash_labels(row, ['linear_constraints_infeasible'], True, None)
        self.assertIn('L3_recovery_lock', labels)
        self.assertIn('constraint_conflict', labels)
        self.assertIn('data_block', labels)
        self.assertNotIn('no_candidates', labels)

    def test_cash_audit_selects_only_registered_arms_and_requires_every_one(self):
        protocol = {'protocol': {'arms': {'main': {}, 'control': {}}}}
        outcomes = {'main': {'run_id': 'a'}, 'control': {'run_id': 'b'},
                    'reference_original': {'summary': {}}}
        self.assertEqual(list(self.m.registered_outcomes(outcomes, protocol)),
                         ['main', 'control'])
        with self.assertRaisesRegex(ValueError, 'missing_registered_arm:control'):
            self.m.registered_outcomes({'main': outcomes['main']}, protocol)
        outcomes['control'] = {'summary': {}}
        with self.assertRaisesRegex(ValueError, 'malformed_registered_arm:control'):
            self.m.registered_outcomes(outcomes, protocol)

    def test_cash_idle_days_retain_last_decision_causes_with_provenance(self):
        dates = ['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07']
        run = dict(account=[dict(date=d, quantities={}, cash=100., nav=100., stale=[]) for d in dates],
                   risk_history=[dict(date=d, level=0, manual_review=False, allow_new_stock=True,
                                      eligible_count=None, recovery_count=0, reasons=[]) for d in dates],
                   allocation_issues=[dict(date=dates[1], reason='linear_constraints_infeasible')],
                   signal_trace=[dict(date=dates[2], scores={}, status='allocation_failed')],
                   data_issues=[dict(date=dates[0], reason='unexplained_ex_reference')])
        rows, _ = self.m.cash_rows(run, 'main')
        self.assertEqual(rows[0]['primary'], 'waiting_for_first_decision')
        self.assertIn('constraint_conflict', rows[1]['labels'])
        self.assertIn('no_candidates', rows[3]['labels'])
        self.assertNotIn('constraint_conflict', rows[3]['labels'])
        self.assertNotIn('data_block', rows[0]['labels'])
        self.assertEqual(rows[3]['last_decision_date'], dates[2])
        self.assertIsNone(rows[3]['eligible_count'])
        self.assertIn('last_decision.empty_scores', rows[3]['label_evidence'])
        self.assertIn(dates[2], rows[3]['label_evidence'])
        run['signal_trace'] = []
        rows, _ = self.m.cash_rows(run, 'main')
        self.assertIn('constraint_conflict', rows[3]['labels'])
        self.assertEqual(rows[3]['last_decision_date'], dates[1])

    def test_reconstruction_cannot_pass_when_one_stored_score_is_missing(self):
        stored = pd.Series({'A': .2, 'B': .4})
        with self.assertRaisesRegex(ValueError, 'score_reconstruction_coverage'):
            self.m.validate_scores(pd.Series({'A': .2}), stored, {'lowvol': 1.},
                                   {'lowvol': 1.}, '2020-01-02')
        self.assertEqual(self.m.validate_scores(stored, stored, {'lowvol': 1.},
                                                {'lowvol': 1.}, '2020-01-02'), 0.)

    def test_actual_groups_include_unscored_names_without_changing_sector_baseline(self):
        d = '2020-01-02'
        scored = pd.DataFrame([dict(date=d, instrument_id='A', sector='S', return_20=.1,
                                    coverage_20='observed', return_60=.2, coverage_60='observed')])
        run = dict(account=[dict(date=d, weights={'A': .2, 'B': .3, 'C': .1})],
                   signal_trace=[dict(date=d, selected=['A', 'B'])])
        ledger = pd.DataFrame([dict(date=d, instrument_id=c, industry='S') for c in ['A','B','C']])
        membership = self.m.requested_membership(run, scored, ledger)
        self.assertEqual(len(membership), 3)
        self.assertEqual(membership.in_scored.sum(), 1)
        for c, r, status in [('B', .5, 'observed'), ('C', None, 'known_data_gap')]:
            for h in (20, 60):
                membership.loc[membership.instrument_id.eq(c), f'return_{h}'] = r
                membership.loc[membership.instrument_id.eq(c), f'coverage_{h}'] = status
        calendar = pd.date_range(d, periods=70, freq='B')
        benchmark = pd.Series(1., index=calendar)
        grouped = self.m.aggregate_requested(membership, scored, benchmark, calendar)
        held = grouped[(grouped.group == 'actual_held') & (grouped.horizon == 20)].iloc[0]
        self.assertEqual(held.requested_count, 3)
        self.assertEqual(held.scored_count, 1)
        self.assertEqual(held.valid_count, 2)
        self.assertAlmostEqual(held.total_nav_weight, .6)
        self.assertAlmostEqual(held.observed_nav_weight, .5)
        self.assertAlmostEqual(held.omitted_nav_weight, .1)
        self.assertAlmostEqual(held.scored_nav_weight, .2)
        self.assertAlmostEqual(held.return_mean, .34)
        self.assertAlmostEqual(held.sector_residual, .24)
        selected = grouped[(grouped.group == 'selected') & (grouped.horizon == 20)].iloc[0]
        self.assertEqual(selected.requested_count, 2)
        self.assertAlmostEqual(selected.return_mean, .3)

    def test_partial_risk_reduction_then_readd_is_tagged_without_new_cost_category(self):
        calendar = pd.date_range('2020-01-02', periods=8, freq='B')
        trades = pd.DataFrame([
            dict(date=str(calendar[0].date()), instrument_id='A', side='sell', category='risk_exit', cost=2.),
            dict(date=str(calendar[1].date()), instrument_id='A', side='sell', category='risk_exit', cost=3.),
            dict(date=str(calendar[3].date()), instrument_id='A', side='buy', category='weight_increase', cost=4.)])
        run = {'account': [dict(date=d, quantities={'A': 100}) for d in calendar]}
        audit = self.m.risk_buybacks(trades, run, calendar)
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit.buy_id.nunique(), 1)
        self.assertEqual(audit.lag_sessions.tolist(), [3, 2])
        self.assertTrue(audit.within_20.all())
        self.assertEqual(set(audit.risk_exit_type), {'partial_reduction'})
        self.assertEqual(set(audit.buy_category), {'weight_increase'})
        self.assertEqual(trades.cost.sum(), 9.)


if __name__ == '__main__':
    unittest.main()
