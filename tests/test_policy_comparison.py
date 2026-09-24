import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.policy_comparison import (ARMS, CONTRASTS, REFERENCE_ARMS, frozen_protocol,
    account_checks, stress_windows, verdict, _stress_dataset, experiment_health, lagged_budget_attribution,
    STRESS_NAMES, comparison_payload, summarize_run, run_stress_suite, recorded_stress_results,
    validate_stress_artifact, _record_once)
from quantlab.config import Config
from quantlab.research import ExperimentRegistry
from quantlab.report.runs import persist_run, atomic_json
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


def small_account_run():
    """An invested, reconciled account with an actual next-session buy."""
    dates = pd.bdate_range('2021-01-04', periods=20)
    states = []
    for i, day in enumerate(dates):
        price = 10. + i / 10
        shares = 0 if i == 0 else 50
        cash = 1000. if not shares else 495.
        nav = cash + shares * price
        states.append(dict(date=day, nav=nav, cash=cash, receivables=0., payables=0.,
            quantities={'A': shares} if shares else {}, marks={'A': price} if shares else {},
            weights={'A': shares * price / nav} if shares else {}))
    nav = pd.Series([s['nav'] / 1000 for s in states], index=dates)
    fills = [dict(order_id='buy1', instrument_id='A', date=dates[1], signal_date=dates[0],
                  side='buy', quantity=50, notional=505., fees=0., slippage_cost=0., reason='rebalance')]
    return dict(run_id='small-run', status='research_unvalidated', data_version='fixture-data',
        model_version='fixture-model', as_of=str(dates[-1].date()),
        config={'capital': {'init_cash': 1000.}}, risk_policy='P0', account=states,
        daily_return=nav.pct_change().fillna(nav.iloc[0] - 1), nav=nav, calendar=dates,
        fills=fills, orders=[], signal_trace=[dict(date=dates[0], scores={'A': 1.}, target_weights={'A': .5})],
        risk_history=[], run_health=dict(allocation_failures=0, forecast_failures=0, data_issue_count=0))


class PolicyComparisonHealthAndResumeTests(unittest.TestCase):
    def test_healthy_candidate_cannot_pass_against_a_broken_control(self):
        healthy = dict(accounting={'passed': True}, experiment_health={'passed': True},
            run_health=dict(allocation_failures=0, forecast_failures=0, data_issue_count=0),
            net_profit=100., conditional_metrics={'max_drawdown': -.1})
        pair = dict(annualized_net_excess=.04,
                    segments=[dict(complete=True, annualized_net_excess=.04)] * 3)
        stress = {name:dict(constraints_passed=True, run={'net_profit': 50.}) for name in STRESS_NAMES}
        args = (healthy, pair, {'simultaneous_lower': .01}, {'simultaneous_lower': .01}, stress)
        evidence = dict(accounting_verified=True, warnings=[])
        passed = verdict(*args, data_assessment=evidence, control_summary=healthy)
        self.assertEqual(passed['status'], 'retrospective_candidate_for_forward_observation')
        for field, reason in [('allocation_failures','control_allocation_failures'),
                              ('data_issue_count','control_held_data_gaps'),
                              ('forecast_failures','control_invalid_or_warmup_decisions')]:
            with self.subTest(field=field):
                control = deepcopy(healthy)
                control['run_health'][field] = 1
                result = verdict(*args, data_assessment=evidence, control_summary=control)
                self.assertEqual(result['status'], 'unproven')
                self.assertIn(reason, result['reasons'])
                self.assertFalse(result['comparison_health']['control']['passed'])
                self.assertTrue(result['comparison_health']['candidate']['passed'])
                self.assertEqual(pair['annualized_net_excess'], .04)
        self.assertIn('control_run_health_unavailable',
                      verdict(*args, data_assessment=evidence)['reasons'])

    def test_in_memory_payload_drops_large_evidence_and_retains_same_statistical_inputs(self):
        run = small_account_run()
        run['standardized_features'] = object()
        run['weekly_observations'] = object()
        compact = comparison_payload(run)
        self.assertEqual(set(compact), {'run_id','account','fills','daily_return','risk_history'})
        for key in compact:
            self.assertIs(compact[key], run[key])
        self.assertIn('signal_trace', run)

    def test_stress_interruption_keeps_finished_scenario_and_resume_skips_it(self):
        ds = dataset(days=30, stocks=2)
        base = small_account_run()
        names = ('cost_double','liquidity_30pct','execution_delay_2')
        with tempfile.TemporaryDirectory() as temp, patch('quantlab.policy_comparison.STRESS_NAMES', names):
            registry = ExperimentRegistry(Path(temp) / 'registry')
            trial = registry.register('stress', {}, 'data', 'fixed', 'model')
            json_path = Path(temp) / 'stress.json'
            saved = {}
            def checkpoint(name, outcome):
                _record_once(registry, trial, dict(fold='stress:'+name, **outcome))
                saved[name] = outcome
                atomic_json(json_path, saved)
            interrupted = Mock(side_effect=[small_account_run(), KeyboardInterrupt()])
            with self.assertRaises(KeyboardInterrupt):
                run_stress_suite(ds, Config(), {'risk_policy':'P0'}, base, interrupted, on_result=checkpoint)
            self.assertEqual(list(saved), ['cost_double'])
            self.assertEqual(interrupted.call_count, 2)
            resumed = recorded_stress_results(registry, trial, json.loads(json_path.read_text()))
            replay = Mock(side_effect=[small_account_run(),small_account_run()])
            verifier = Mock()
            result = run_stress_suite(ds, Config(), {'risk_policy':'P0'}, base, replay,
                resume=resumed, verify_resume=verifier, on_result=checkpoint)
            self.assertEqual(set(result), set(names))
            self.assertEqual(replay.call_count, 2)
            self.assertEqual(replay.call_args_list[0].kwargs['liquidity_mult'], .3)
            self.assertEqual(replay.call_args_list[1].kwargs['execution_delay'], 2)
            verifier.assert_called_once_with('cost_double', resumed['cost_double'])
            self.assertEqual(set(json.loads(json_path.read_text())), set(names))
            with self.assertRaisesRegex(ValueError, 'stress_resume_verifier_required'):
                run_stress_suite(ds, Config(), {'risk_policy':'P0'}, base, replay, resume=resumed)

    def test_registry_recovers_gap_before_json_write_and_rejects_changed_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = ExperimentRegistry(Path(temp) / 'registry')
            trial = registry.register('stress', {}, 'data', 'fixed', 'model')
            outcome = dict(status='failed', constraints_passed=False, reason='test_failure')
            _record_once(registry, trial, dict(fold='stress:cost_double', **outcome))
            self.assertEqual(recorded_stress_results(registry, trial, {}), {'cost_double':outcome})
            with self.assertRaisesRegex(ValueError, 'stress_checkpoint_differs_from_registry'):
                recorded_stress_results(registry, trial, {'cost_double':dict(outcome, reason='changed')})
            with self.assertRaisesRegex(ValueError, 'stress_checkpoint_not_registered'):
                recorded_stress_results(registry, trial, {'liquidity_30pct':outcome})
            # Failure is terminal under the same registered scenario identity.
            with self.assertRaisesRegex(ValueError, 'recorded_outcome_differs_on_resume'):
                _record_once(registry, trial, dict(fold='stress:cost_double', **dict(outcome, status='completed')))

    def test_completed_stress_resume_requires_matching_immutable_artifact(self):
        with tempfile.TemporaryDirectory() as temp, patch('quantlab.report.runs.provenance', return_value={}):
            run = small_account_run()
            folder = persist_run(run, root=temp)
            summary = summarize_run(run)
            self.assertTrue(summary['accounting']['passed'])
            self.assertTrue(summary['experiment_health']['passed'])
            outcome = dict(status='completed', run_id=run['run_id'], run=summary)
            validate_stress_artifact(outcome, temp)
            with self.assertRaisesRegex(ValueError, 'stress_artifact_summary_mismatch'):
                validate_stress_artifact(dict(outcome, run=dict(summary, net_profit=999999)), temp)
            with self.assertRaises(FileNotFoundError):
                validate_stress_artifact(dict(outcome, run_id='missing-artifact'), temp)
            with self.assertRaisesRegex(ValueError, 'completed_stress_artifact_missing'):
                validate_stress_artifact({'status':'completed'}, temp)
            with self.assertRaisesRegex(ValueError, 'invalid_stress_artifact_path'):
                validate_stress_artifact(dict(outcome, run_id='../outside'), temp)
            (folder/'result.pkl').write_bytes((folder/'result.pkl').read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact_checksum_mismatch'):
                validate_stress_artifact(outcome, temp)


if __name__ == '__main__': unittest.main()
