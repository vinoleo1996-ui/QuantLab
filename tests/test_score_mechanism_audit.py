import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import textwrap
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/score_mechanism_audit.py'


class ScoreMechanismContracts(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('score_mechanism_audit', SCRIPT)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_source_hash_failure_and_attempt_reuse_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'evidence'
            source.write_text('archived evidence')
            with self.assertRaisesRegex(ValueError, 'source_hash_mismatch'):
                self.m.verify_hashes({source: 'wrong'})
            hashes = {source: self.m.sha(source)}
            self.assertEqual(self.m.verify_hashes(hashes)[str(source)], hashes[source])
            attempt = root / 'run1'
            self.m.start_attempt(attempt, {'promotion_approved': False})
            with self.assertRaisesRegex(ValueError, 'attempt_exists'):
                self.m.start_attempt(attempt, {})

    def test_saved_scores_require_complete_finite_reconstruction(self):
        expected = pd.Series({'A': .2, 'B': .3})
        for actual in [pd.Series({'A': .2}), pd.Series({'A': .2, 'B': np.inf}),
                       pd.Series({'A': .2, 'B': np.nan})]:
            with self.assertRaisesRegex(ValueError, 'score_reconstruction_coverage'):
                self.m.validate_reconstruction(actual, expected, 'fixture')
        with self.assertRaisesRegex(ValueError, 'score_reconstruction_mismatch'):
            self.m.validate_reconstruction(expected + .001, expected, 'fixture')
        self.assertEqual(self.m.validate_reconstruction(expected, expected, 'fixture'), 0.)

    def test_model_comparisons_use_common_scores_and_pre_outcome_tied_buckets(self):
        frame = pd.DataFrame(dict(date=['2020-01-02'] * 20,
                                  score_a=np.arange(20, dtype=float),
                                  score_b=np.arange(20, dtype=float),
                                  return_20=np.arange(20) / 100,
                                  residual_20=np.arange(20) / 200,
                                  coverage_20=['observed'] * 20))
        frame.loc[19, 'score_b'] = np.nan
        frame.loc[16:18, ['score_a', 'score_b']] = 18
        metrics, membership = self.m.model_diagnostics(frame, ('a', 'b'), (20,))
        self.assertEqual(metrics.common_finite_score_count.tolist(), [19, 19])
        self.assertEqual(metrics.future_observed_count.tolist(), [19, 19])
        self.assertEqual(membership.loc[16:18, 'bucket_a'].nunique(), 1)
        frame.loc[0:6, 'return_20'] = np.nan
        frame.loc[0:6, 'coverage_20'] = 'known_data_gap'
        changed, altered = self.m.model_diagnostics(frame, ('a', 'b'), (20,))
        pd.testing.assert_series_equal(membership.bucket_a, altered.bucket_a)
        self.assertEqual(changed.future_observed_count.tolist(), [12, 12])
        self.assertEqual(changed.q5_minus_q3.tolist(), metrics.q5_minus_q3.tolist())

    def test_constant_and_insufficient_rank_information_stays_null(self):
        for n, constant, reason in [(9, False, 'insufficient_observations'),
                                    (12, True, 'constant_score')]:
            frame = pd.DataFrame(dict(score_a=np.ones(n) if constant else np.arange(n),
                                      return_20=np.arange(n) / 100,
                                      residual_20=np.arange(n) / 100,
                                      coverage_20=['observed'] * n))
            metrics, _ = self.m.model_diagnostics(frame, ('a',), (20,))
            self.assertTrue(pd.isna(metrics.rankic.iloc[0]))
            self.assertEqual(metrics.rankic_status.iloc[0], reason)
            self.assertTrue(pd.isna(metrics.q5_minus_q3.iloc[0]))

    def test_proxy_is_group_then_row_mean_and_ignores_future_ic(self):
        hist = pd.DataFrame({'f1': [.2, .4, 99.], 'f2': [-.2, np.nan, -99.],
                             'f3': [.5, .5, 99.]},
                            index=pd.to_datetime(['2020-01-02', '2020-02-03', '2020-03-02']))
        specs = {'two': ['f1', 'f2'], 'one': ['f3'], 'all': None}
        weights, factors = self.m.weight_proxy(hist, '2020-02-03', list(hist), specs)
        by = weights.set_index('expert')
        self.assertAlmostEqual(by.loc['two', 'signed_proxy'], .2)
        self.assertAlmostEqual(by.loc['one', 'signed_proxy'], .5)
        self.assertAlmostEqual(by.loc['all', 'signed_proxy'], (.5 / 3 + .9 / 2) / 2)
        self.assertEqual(by.loc['two', 'ic_rows'], 2)
        future_changed = hist.copy()
        future_changed.iloc[-1] = -500
        unchanged, _ = self.m.weight_proxy(future_changed, '2020-02-03', list(hist), specs)
        pd.testing.assert_frame_equal(weights, unchanged)
        self.assertAlmostEqual(weights.full_precision_weight.sum(), 1.)
        contribution = factors[factors.expert.eq('two')].contribution_to_group_mean.sum()
        self.assertAlmostEqual(contribution, .2)

    def test_actual_frozen_experts_reconstruct_scores_ranks_and_candidate_order(self):
        # A fresh interpreter prevents the full suite's current quantlab imports
        # from substituting for the explicitly required frozen package.
        code = textwrap.dedent('''
            import importlib.util
            from pathlib import Path
            import numpy as np
            import pandas as pd
            root = Path.cwd()
            spec = importlib.util.spec_from_file_location('audit', root / 'scripts/score_mechanism_audit.py')
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            experts, scoring = m.load_frozen_api()
            from quantlab.config import CONFIG
            d = pd.Timestamp('2020-02-03')
            names = list(dict.fromkeys(f for group in experts.EXPERT_SPECS.values() if group for f in group))
            codes = [f'S{i:02}' for i in range(14)]
            full = {f: pd.DataFrame([np.sin(np.arange(14) * (i + 1) / 9) + np.arange(14) / 50],
                                    index=[d], columns=codes) for i, f in enumerate(names)}
            row = pd.DataFrame([{f: (i + 1) / 20 for i, f in enumerate(names)}], index=[d])
            ic = pd.DataFrame([{f: (i - 4) / 20 for i, f in enumerate(names)}] * 4,
                              index=pd.to_datetime(['2019-11-29','2019-12-31','2020-01-23','2020-03-02']))
            hist = ic.loc[:d]
            pen, _ = scoring.factor_crowding_penalty(hist)
            pen = pen.reindex(pen.index.union([d])).sort_index().ffill().loc[[d]]
            score, ew = experts.experts_score(full, row, pen, hist, cfg=CONFIG.factors)
            eligible = codes[2:12]
            tops = experts.expert_top_union({k: v.reindex(columns=eligible) for k,v in full.items()},
                                             row, pen, None, 3, cfg=CONFIG.factors)
            union = sorted(set(c for group in tops.values() for c in group))
            trace = dict(date=str(d.date()), standardized_features={k: v.loc[d].to_dict() for k,v in full.items()},
                         factor_weights=row.loc[d].to_dict(), expert_weights=ew,
                         expert_candidates=tops, scores=score.loc[d, union].to_dict())
            saved = pd.DataFrame(dict(instrument_id=eligible, score=score.loc[d, eligible].values))
            panel, proxy, factors, check = m.reconstruct_trace(trace, ic, saved, CONFIG.factors, 3)
            assert check['max_score_error'] <= 1e-10 and check['candidate_order_equal']
            for name in experts.EXPERT_SPECS:
                raw = experts._group_score(full, row, pen, experts.EXPERT_SPECS[name]).loc[d]
                np.testing.assert_allclose(panel['score_' + name], raw.rank(pct=True).reindex(eligible))
            assert sorted(proxy.factor_count.tolist()) == [2,2,3,4,11]
            bad = dict(trace, expert_candidates={**tops, 'momentum': list(reversed(tops['momentum']))})
            try:
                m.reconstruct_trace(bad, ic, saved, CONFIG.factors, 3)
            except ValueError as exc:
                assert 'candidate_order_mismatch' in str(exc)
            else:
                raise AssertionError('candidate order discrepancy accepted')
            changed = ic.copy(); changed.loc[changed.index > d] = 900.
            again = m.reconstruct_trace(trace, changed, saved, CONFIG.factors, 3)
            pd.testing.assert_frame_equal(proxy, again[1])
        ''')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_matured_quality_includes_equal_endpoint_but_excludes_later_outcomes(self):
        dates = ['2020-01-02', '2020-02-03', '2020-03-02']
        metrics = pd.DataFrame(dict(date=dates, model=['a'] * 3, horizon=[20] * 3,
                                    endpoint=['2020-02-03', '2020-03-03', '2020-04-03'],
                                    rankic=[.1, .2, .9], rankic_status=['observed'] * 3))
        result = self.m.matured_quality(metrics, ['2020-02-03', '2020-03-02'])
        self.assertEqual(result.observable_cohorts.tolist(), [1, 1])
        self.assertEqual(result.actual_rankic_mean.tolist(), [.1, .1])
        self.assertEqual(result.latest_endpoint.tolist(), ['2020-02-03'] * 2)
        self.assertTrue(result.history_status.eq('insufficient_12_cohort_history').all())
        changed = metrics.copy(); changed.loc[1:, 'rankic'] = -99
        pd.testing.assert_frame_equal(result, self.m.matured_quality(changed, ['2020-02-03', '2020-03-02']))

    def test_matched_factor_mean_is_not_the_composite_rankic(self):
        x = np.arange(15, dtype=float)
        panel = pd.DataFrame(dict(date=['2020-01-02'] * 15, factor_f1=x, factor_f2=-x,
                                  factor_f3=x, score_two=x, score_one=x, score_all=x,
                                  return_20=x / 100, coverage_20=['observed'] * 15,
                                  residual_20=x / 100))
        factors, experts = self.m.matched_factor_diagnostics(
            panel, {'two': ['f1', 'f2'], 'one': ['f3'], 'all': None}, (20,))
        pair = experts.set_index('expert')
        self.assertAlmostEqual(pair.loc['two', 'group_factor_rankic_mean'], 0.)
        self.assertAlmostEqual(pair.loc['two', 'expert_rankic'], 1.)
        self.assertAlmostEqual(pair.loc['all', 'group_factor_rankic_mean'], 1 / 3)
        self.assertEqual(set(factors.observed_count), {15})
        panel.loc[0, 'factor_f3'] = np.nan
        factors, experts = self.m.matched_factor_diagnostics(
            panel, {'two': ['f1', 'f2'], 'one': ['f3'], 'all': None}, (20,))
        self.assertEqual(set(factors.observed_count), {14})
        self.assertEqual(set(experts.observed_count), {14})

    def test_complete_selected_exceptions_survive_with_unknown_financial_status(self):
        scored = pd.DataFrame(dict(date=['2020-01-02'] * 3, instrument_id=['A', 'B', 'C'],
                                   score=[.9, .8, .1], score_mixture=[.9, .8, .1],
                                   in_expert_union=[True, True, False], quality_pass=[True, False, None],
                                   financial_covered=[True, True, None],
                                   selected=[True, True, False], held_weight=[0., .1, 0.], sector=['S'] * 3,
                                   return_20=[.1, .2, .3], coverage_20=['observed'] * 3))
        requested = pd.DataFrame(dict(date=['2020-01-02'] * 3, instrument_id=['A', 'B', 'D'],
                                     selected=[True, True, True], held_weight=[0., .1, .2], sector=['S'] * 3,
                                     return_20=[.1, .2, .9], coverage_20=['observed'] * 3,
                                     in_scored=[True, True, False]))
        all_members = self.m.complete_membership(scored, requested, (20,))
        by = all_members.set_index('instrument_id')
        self.assertEqual(by.loc['D', 'financial_status'], 'unknown')
        self.assertFalse(by.loc['D', 'in_scored'])
        self.assertAlmostEqual(by.loc['D', 'sector_peer_mean_20'], .2)
        self.assertAlmostEqual(by.loc['D', 'residual_20'], .7)
        self.assertEqual(by.loc['B', 'financial_status'], 'known_fail')
        self.assertEqual(by.loc['C', 'financial_status'], 'unknown')
        self.assertEqual(all_members.selected.sum(), 3)
        stages, transitions = self.m.funnel_diagnostics(all_members, (20,))
        selected = stages[stages.stage.eq('selected')].iloc[0]
        self.assertEqual(selected.requested_count, 3)
        self.assertEqual(selected.common_finite_score_count, 2)
        self.assertEqual(selected.future_observed_all, 3)
        self.assertAlmostEqual(selected.all_member_return_mean, .4)
        self.assertEqual(selected.financial_unknown_count, 1)
        self.assertEqual(transitions.iloc[-1].child_outside_parent_count, 2)

    def test_kept_removed_comparison_only_uses_shared_observable_dates(self):
        panel = pd.DataFrame(dict(date=['2020-01-02'] * 3 + ['2020-02-03'] * 2,
                                  parent=[True, True, False, True, True],
                                  child=[True, False, True, True, False],
                                  return_20=[.1, .2, .8, .9, np.nan],
                                  residual_20=[.01, .02, .08, .09, np.nan],
                                  coverage_20=['observed'] * 4 + ['missing_price_path']))
        transition = self.m.paired_transition(panel, panel.parent, panel.child, 'test', (20,))
        self.assertEqual(transition.paired_observable.tolist(), [True, False])
        self.assertAlmostEqual(transition.kept_minus_removed.mean(), -.1)
        self.assertEqual(transition.child_outside_parent_count.tolist(), [1, 0])
        self.assertEqual(transition.removed_count.tolist(), [1, 1])
        self.assertEqual(transition.removed_observed_count.tolist(), [1, 0])

    def test_date_equal_summaries_do_not_pool_cross_section_counts(self):
        metrics = pd.DataFrame(dict(date=['2020-01-02', '2020-02-03', '2022-01-03'],
                                    phase=['2020-21', '2020-21', '2022-23'],
                                    model=['a'] * 3, horizon=[20] * 3,
                                    value=[.1, .3, np.nan], requested_count=[100, 1000, 9000]))
        result = self.m.summarize_dates(metrics, ['model', 'horizon'], ['value'])
        overall = result[result.phase.eq('overall')].iloc[0]
        self.assertAlmostEqual(overall['mean'], .2)
        self.assertEqual(overall.valid_dates, 2)
        self.assertEqual(overall.requested_dates, 3)
        empty = result[result.phase.eq('2022-23')].iloc[0]
        self.assertTrue(pd.isna(empty['mean']))
        self.assertEqual(empty.valid_dates, 0)

    def test_uncovered_archived_entry_is_unknown_not_economic_failure(self):
        scored = pd.DataFrame(dict(date=['2020-01-02'] * 2, instrument_id=['A', 'B'],
                                   in_expert_union=[True, True], quality_pass=[False, False],
                                   financial_covered=[False, True], selected=[True, True],
                                   held_weight=[0., 0.], sector=['S', 'S']))
        requested = scored[['date', 'instrument_id', 'selected', 'held_weight', 'sector']].copy()
        members = self.m.complete_membership(scored, requested, ())
        self.assertEqual(members.financial_status.tolist(), ['unknown', 'known_fail'])

    def test_failed_attempt_persists_reason_and_cannot_be_executed_again(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            def fail(_):
                raise ValueError('fixture_reconstruction_mismatch')
            with self.assertRaisesRegex(ValueError, 'fixture_reconstruction_mismatch'):
                self.m.execute_attempt(output, {'promotion_approved': False}, fail)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertIn('fixture_reconstruction_mismatch', status['failure_reason'])
            self.assertFalse((output / 'summary.json').exists())
            self.assertTrue((output / 'manifest.json').is_file())
            with self.assertRaisesRegex(ValueError, 'attempt_exists'):
                self.m.execute_attempt(output, {}, fail)

    def test_object_typed_nullable_rankics_remain_unknown_in_asof_history(self):
        metrics = pd.DataFrame(dict(date=['2020-01-02', '2020-02-03'], model=['a', 'a'],
                                    horizon=[60, 60], endpoint=['2020-03-30', None],
                                    rankic=pd.Series([None, .2], dtype=object),
                                    rankic_status=['insufficient_observations', 'observed']))
        result = self.m.matured_quality(metrics, ['2020-04-01'])
        self.assertEqual(result.matured_cohorts.iloc[0], 1)
        self.assertEqual(result.observable_cohorts.iloc[0], 0)
        self.assertTrue(pd.isna(result.actual_rankic_mean.iloc[0]))


if __name__ == '__main__':
    unittest.main()
