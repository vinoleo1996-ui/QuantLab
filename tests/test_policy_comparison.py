import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.policy_comparison import (ARMS, CONTRASTS, REFERENCE_ARMS, frozen_protocol,
    account_checks, stress_windows, verdict, _stress_dataset, experiment_health, lagged_budget_attribution)
from quantlab.research_statistics import (paired_returns, simultaneous_block_ci,
    deflated_sharpe, family_sharpe, fixed_two_year_segments)


class PolicyStatisticsTests(unittest.TestCase):
    def test_declared_family_uses_shared_draws_and_keeps_dependent_hypotheses(self):
        rng = np.random.default_rng(74)
        excess = np.repeat(rng.normal(.0003, .002, 40), 10)
        frame = pd.DataFrame({'a': excess, 'identical_a': excess, 'negative_a': -excess})
        result = simultaneous_block_ci(frame, block=20, samples=300)
        again = simultaneous_block_ci(frame, block=20, samples=300)
        self.assertEqual(result, again)
        self.assertEqual(result['comparisons']['a'], result['comparisons']['identical_a'])
        self.assertAlmostEqual(result['comparisons']['negative_a']['estimate'],
                               -result['comparisons']['a']['estimate'])
        self.assertLess(result['comparisons']['a']['simultaneous_lower'],
                        result['comparisons']['a']['simultaneous_upper'])
        self.assertEqual(simultaneous_block_ci(frame.iloc[:59], block=20, samples=100)['status'],
                         'insufficient_data')

    def test_pairing_does_not_drop_missing_account_days(self):
        dates = pd.bdate_range('2021-01-01', periods=250)
        a = pd.Series(.001, index=dates)
        runs = {'a': {'daily_return': a}, 'b': {'daily_return': a.iloc[1:]}}
        with self.assertRaisesRegex(ValueError, 'calendar_mismatch'):
            paired_returns(runs, {'test': ('a', 'b')})
        runs['b']['daily_return'] = a.copy()
        self.assertTrue(paired_returns(runs, {'test': ('a', 'b')}).eq(0).all().all())
        runs['b']['daily_return'].iloc[7] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing_returns'):
            paired_returns(runs, {'test': ('a', 'b')})

    def test_dsr_measures_tail_moments_and_does_not_claim_complete_search(self):
        rng = np.random.default_rng(81)
        values = rng.normal(.001, .008, 800)
        values[::40] -= .07
        result = deflated_sharpe(values, trials=8)
        self.assertEqual(result['periods_per_year'], 252)
        self.assertLess(result['observed_skew'], -.5)
        self.assertGreater(result['observed_raw_kurtosis'], 3)
        self.assertEqual(result['status'], 'historical_search_incomplete')
        self.assertFalse(result['promotion_approved'])
        self.assertGreaterEqual(deflated_sharpe(values, trials=1)['value'], result['value'])
        self.assertEqual(deflated_sharpe(np.zeros(400), trials=8)['status'], 'insufficient_variation')
        family = family_sharpe({'a': {'daily_return': values}, 'b': {'daily_return': values + .001}}, registered_trials=8)
        self.assertEqual(family['a']['dispersion_source'], 'observed_declared_family')


class PolicyProtocolTests(unittest.TestCase):
    def test_three_stages_exclude_2026_observation_and_policy_uses_own_drawdown(self):
        dates = pd.bdate_range('2020-01-02', '2026-06-30')
        segments = fixed_two_year_segments(pd.Series(.0001, index=dates))
        self.assertEqual(len(segments['stages']), 3)
        self.assertTrue(all(s['complete'] for s in segments['stages']))
        self.assertFalse(segments['observation_2026H1']['counts_toward_three_stages'])
        summary = dict(accounting={'passed': True}, risk_policy='P2', net_profit=50000., conditional_metrics={'max_drawdown': -.23})
        pair = dict(annualized_net_excess=.04, segments=segments['stages'])
        evidence = dict(accounting_verified=True, warnings=[])
        p2 = verdict(summary, pair, {}, {}, {}, data_assessment=evidence)
        self.assertNotIn('drawdown_target_exceeded', p2['reasons'])
        summary['risk_policy'] = 'P0'
        self.assertIn('drawdown_target_exceeded', verdict(summary, pair, {}, {}, {}, data_assessment=evidence)['reasons'])

    def test_risk_attribution_uses_prior_close_budget(self):
        dates = pd.bdate_range('2020-01-02', periods=3)
        a = dict(daily_return=pd.Series([0., .1, -.1], index=dates), risk_history=[
            dict(date=d, risk_budget=dict(state=s, target_vol=v))
            for d,s,v in zip(dates,['normal','active','defensive'],[.18,.22,.12])])
        b = dict(daily_return=pd.Series(0., index=dates))
        report = lagged_budget_attribution(a,b)
        self.assertGreater(report['by_state']['normal']['annualized_mean_net_excess'], 0)
        self.assertLess(report['by_state']['active']['annualized_mean_net_excess'], 0)
        self.assertNotIn('defensive',report['by_state'])

    def test_empty_cash_replay_and_unused_candidate_are_not_experiments(self):
        run = dict(account=[dict(quantities={})] * 100, fills=[], signal_trace=[])
        self.assertFalse(experiment_health(run)['passed'])
        run.update(account=[dict(quantities={'A': 100})], fills=[dict(side='buy', quantity=100)],
                   signal_trace=[dict(scores={'A': 1}, target_weights={'A': .05})])
        self.assertTrue(experiment_health(run)['passed'])
        run['candidate'] = 'industry_rotation'
        self.assertIn('candidate_not_exercised', experiment_health(run)['reasons'])

    def test_frozen_matrix_is_bounded_and_quality_has_coverage_control(self):
        ds = dataset()
        protocol = frozen_protocol(ds)
        self.assertEqual(len(ARMS) + len(REFERENCE_ARMS), 10)
        self.assertLessEqual(protocol['registered_configurations'], protocol['max_configurations'])
        self.assertEqual(CONTRASTS['quality_signal'], ('qualityP2', 'qualityCoverageP2'))
        self.assertEqual(protocol['history_status'], 'all_historical_periods_retrospective_not_blind')
        ds.manifest['research_scope'] = {'sample_size': 120, 'scope': 'historical_cohort'}
        with self.assertRaisesRegex(ValueError, 'fixed_sample'):
            frozen_protocol(ds)

    def test_account_checks_detect_fictitious_cash_and_early_execution(self):
        run = dict(account=[dict(nav=100., cash=40., receivables=0., quantities={'a': 3}, marks={'a': 20.})], fills=[])
        self.assertTrue(account_checks(run)['passed'])
        run['account'][0]['cash'] = 41.
        self.assertFalse(account_checks(run)['passed'])
        run['fills'] = [dict(date='2021-01-04', signal_date='2021-01-04')]
        self.assertIn('same_day_or_early_execution', account_checks(run)['errors'])

    def test_adverse_window_must_contain_an_actual_exit(self):
        dates = pd.bdate_range('2021-01-04', periods=40)
        r = pd.Series(0., index=dates)
        r.iloc[10:15] = -.02
        base = dict(daily_return=r, fills=[dict(side='sell', date=str(dates[10].date()))])
        windows = stress_windows(base, dates)
        self.assertEqual(windows['adverse'], list(dates[10:15]))
        base['fills'] = []
        self.assertEqual(stress_windows(base, dates)['adverse'], [])

    def test_missing_quotes_changes_only_held_panels_and_preserves_recovery(self):
        ds = dataset(days=70, stocks=2)
        sessions = list(ds.calendar[20:25])
        base = dict(account=[dict(date=ds.calendar[19], quantities={'S000': 100})])
        changed, codes = _stress_dataset(ds, base, sessions, 'missing_held_quotes_5')
        self.assertEqual(codes, ['S000'])
        self.assertFalse(changed.panels['S000'].index.isin(sessions).any())
        self.assertTrue(ds.panels['S000'].index.isin(sessions).any())
        pd.testing.assert_frame_equal(changed.panels['S001'], ds.panels['S001'])
        self.assertEqual(changed.panels['S000'].loc[ds.calendar[25], 'close'], ds.panels['S000'].loc[ds.calendar[25], 'close'])

    def test_positive_point_estimate_cannot_pass_without_family_and_stress(self):
        summary = dict(accounting={'passed': True}, net_profit=50000., conditional_metrics={'max_drawdown': -.1})
        pair = dict(annualized_net_excess=.04, segments=[dict(sessions=120, annualized_net_excess=.04)] * 3)
        result = verdict(summary, pair, {'simultaneous_lower': -.01}, {'simultaneous_lower': .01}, {},
                         data_assessment={'accounting_verified': True, 'warnings': []})
        self.assertEqual(result['status'], 'unproven')
        self.assertIn('family_adjusted_increment_not_proven', result['reasons'])
        self.assertIn('stress_missing_failed_or_not_exercised', result['reasons'])

    def test_reference_hash_recovery_never_claims_nav_recreates_trades(self):
        module_path = Path(__file__).resolve().parents[1] / 'scripts/audit_reference1882.py'
        spec = importlib.util.spec_from_file_location('reference1882_audit_test', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / 'output').mkdir()
            original = b'original = 1\n'
            with tarfile.open(repo / 'output/source.tar.gz', 'w:gz') as tar:
                item = tarfile.TarInfo('snapshot/strategy.py')
                item.size = len(original)
                tar.addfile(item, io.BytesIO(original))
            (repo / 'strategy.py').write_text('changed = 2\n')
            report = dict(code_sha256={'strategy.py': hashlib.sha256(original).hexdigest()},
                          signal_sha256='hash-only', runs=[dict(signal='quality_filtered_equal', cost_multiplier=1,
                                                               annualized_return=.1882, equity_curve=[['2022-01-04', 500000, 500000]])])
            (repo / 'report.json').write_text(json.dumps(report))
            result = module.audit_reference(repo, 'report.json')
            self.assertEqual(result['source_recovery']['missing'], [])
            self.assertEqual(result['source_recovery']['recovered']['strategy.py']['kind'], 'source_archive')
            self.assertEqual(result['status'], 'cannot_exactly_reproduce')
            self.assertIn('original_dated_target_schedule_missing', result['reproduction_blockers'])
            self.assertEqual((repo / 'strategy.py').read_text(), 'changed = 2\n')


if __name__ == '__main__': unittest.main()
