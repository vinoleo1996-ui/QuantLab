import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/candidate_construction_audit.py'


def fixture():
    return pd.DataFrame({
        'date': ['2020-01-02'] * 4 + ['2022-01-03'] * 4,
        'instrument_id': ['D', 'C', 'B', 'A'] * 2,
        'score_mixture': [.1, .8, .8, .9] * 2,
        'in_expert_union': [True, True, False, False] + [True, False, False, False],
        'sector': ['s1', 's1', 's2', None] * 2,
        'bucket_mixture': [1., 4., 4., 5.] * 2,
        'financial_status': ['pass', 'known_fail', 'unknown', 'unknown', 'pass', 'unknown', 'unknown', 'unknown'],
        'return_20': [.1, .2, .4, .8] * 2,
        'residual_20': [.05, .15, .2, .3] * 2,
        'coverage_20': ['observed'] * 8,
    })


class CandidateConstructionContracts(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('candidate_audit', SCRIPT)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_daily_equal_size_uses_score_then_symbol_and_reports_boundary_tie(self):
        source = fixture()
        archived = source.loc[source.in_expert_union, ['date', 'instrument_id']]
        members, dates = self.m.construct_membership(source, archived, expected_dates=2)
        self.assertEqual(dates.A_count.tolist(), [2, 1])
        self.assertEqual(dates.B_count.tolist(), [2, 1])
        self.assertEqual(members.loc[members.in_B, 'instrument_id'].tolist(), ['A', 'B', 'A'])
        self.assertEqual(dates.boundary_tie_count.tolist(), [2, 1])
        self.assertEqual(dates.boundary_tie_selected.iloc[0], ['B'])

    def test_invalid_scores_duplicate_keys_or_archived_union_are_rejected(self):
        original = fixture()
        archived = original.loc[original.in_expert_union, ['date', 'instrument_id']]
        variants = []
        for value in [np.nan, np.inf, -np.inf]:
            bad = original.copy()
            bad.loc[0, 'score_mixture'] = value
            variants.append((bad, archived, 'nonfinite_score'))
        variants.append((pd.concat([original, original.iloc[:1]]), archived, 'duplicate_key'))
        variants.append((original, archived.iloc[1:], 'union_mismatch'))
        for scores, archive, reason in variants:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.m.construct_membership(scores, archive, expected_dates=2)
        with self.assertRaisesRegex(ValueError, 'date_count'):
            self.m.construct_membership(original, archived, expected_dates=74)

    def test_shuffle_and_future_missingness_cannot_change_selected_members(self):
        scores = fixture()
        archived = scores.loc[scores.in_expert_union, ['date', 'instrument_id']]
        before, before_dates = self.m.construct_membership(scores, archived, 2)
        changed = scores.sample(frac=1, random_state=29)
        changed.loc[:, ['return_20', 'residual_20']] = np.nan
        changed.loc[:, 'coverage_20'] = 'unmatured'
        after, after_dates = self.m.construct_membership(changed, archived, 2)
        columns = ['date', 'instrument_id', 'in_A', 'in_B', 'membership_group']
        pd.testing.assert_frame_equal(before[columns], after[columns])
        pd.testing.assert_frame_equal(before_dates, after_dates)

    def test_observed_means_keep_missing_members_and_residual_coverage_separate(self):
        scores = fixture()
        scores.loc[2, 'coverage_20'] = 'known_data_gap'
        scores.loc[3, 'residual_20'] = np.nan
        archived = scores.loc[scores.in_expert_union, ['date', 'instrument_id']]
        members, _ = self.m.construct_membership(scores, archived, 2)
        metrics, pairs, _ = self.m.measure(members, horizons=(20,))
        b = metrics[(metrics.date == '2020-01-02') & (metrics.group == 'B')].iloc[0]
        self.assertEqual((b.requested_count, b.raw_count, b.residual_count), (2, 1, 0))
        self.assertAlmostEqual(b.raw_mean, .8)
        self.assertTrue(pd.isna(b.residual_mean))
        pair = pairs.iloc[0]
        self.assertAlmostEqual(pair.raw_difference, .65)
        self.assertTrue(pair.raw_paired)
        self.assertFalse(pair.raw_complete)
        self.assertFalse(pair.residual_paired)
        self.assertFalse(pair.residual_complete)

    def test_paired_dates_and_complete_sensitivity_never_mix_cohorts(self):
        scores = fixture()
        scores.loc[0:1, 'coverage_20'] = 'known_data_gap'
        scores.loc[7, 'residual_20'] = np.nan
        archived = scores.loc[scores.in_expert_union, ['date', 'instrument_id']]
        members, _ = self.m.construct_membership(scores, archived, 2)
        metrics, pairs, _ = self.m.measure(members, horizons=(20,))
        self.assertFalse(pairs.iloc[0].raw_paired)
        self.assertTrue(pd.isna(pairs.iloc[0].raw_difference))
        self.assertTrue(pairs.iloc[1].raw_complete)
        self.assertFalse(pairs.iloc[1].residual_complete)
        empty = metrics[metrics.group.eq('common')]
        self.assertEqual(empty.requested_count.tolist(), [0, 0])
        self.assertTrue(empty[['raw_mean', 'residual_mean']].isna().all().all())
        summary = self.m.summarize(pairs)
        missing = summary[(summary.phase == '2020-21') & summary.measure.eq('raw')]
        self.assertEqual(missing.date_count.tolist(), [0, 0])
        self.assertTrue(missing['mean'].isna().all())
        complete = summary[(summary.phase == '2022-23') & summary.measure.eq('raw') & summary.subset.eq('complete')].iloc[0]
        self.assertEqual(complete.date_count, 1)
        self.assertAlmostEqual(complete['mean'], .7)

    def test_predeclared_decision_requires_all_main_and_coverage_directions(self):
        cells = pd.DataFrame([dict(phase=p, horizon=h, measure=m, subset=s, mean=.01, date_count=1)
                              for p in ['2020-21', '2022-23', '2024-25', '2026H1']
                              for h in [20, 60] for m in ['raw', 'residual']
                              for s in ['paired', 'complete']])
        cells.loc[cells.phase.eq('2026H1'), 'mean'] = -.5
        self.assertEqual(self.m.decide(cells)['decision'], 'conditional_signal_only')
        missing = cells.copy()
        missing.loc[1, ['mean', 'date_count']] = [np.nan, 0]
        self.assertEqual(self.m.decide(missing)['decision'], 'insufficient_evidence')
        reverse = cells.copy()
        reverse.loc[1, 'mean'] = 0
        self.assertEqual(self.m.decide(reverse)['decision'], 'not_supported_close_priority')
        missing.loc[0, 'mean'] = -.01
        decision = self.m.decide(missing)
        self.assertEqual(decision['decision'], 'not_supported_close_priority')
        self.assertEqual(decision['missing_complete_cells'], 1)
        self.assertEqual(self.m.decide(cells.iloc[2:])['decision'], 'insufficient_evidence')

    def test_archived_financial_unknown_and_shared_field_checks(self):
        scores = fixture()
        archive = scores.copy()
        archive.loc[3, 'financial_status'] = 'pass'
        with self.assertRaisesRegex(ValueError, 'financial_outside_union'):
            self.m.attach_financial(scores, archive)
        archive.loc[3, 'financial_status'] = 'unknown'
        result = self.m.attach_financial(scores, archive)
        self.assertEqual(result.loc[3, 'financial_status'], 'unknown')
        archive.loc[0, 'return_20'] = 99.
        with self.assertRaisesRegex(ValueError, 'shared_field_mismatch'):
            self.m.attach_financial(scores, archive)
        archive = pd.concat([scores, scores.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'duplicate_key'):
            self.m.attach_financial(scores, archive)

    def test_profiles_include_unknown_sector_and_financial_counts(self):
        scores = fixture()
        archived = scores.loc[scores.in_expert_union, ['date', 'instrument_id']]
        members, _ = self.m.construct_membership(scores, archived, 2)
        profiles, finance = self.m.profiles(members)
        b = profiles[(profiles.date == '2020-01-02') & profiles.group.eq('B')]
        unknown = b[b.dimension.eq('sector') & b.category.eq('unknown')].iloc[0]
        self.assertEqual(unknown.member_count, 1)
        self.assertAlmostEqual(unknown.member_share, .5)
        self.assertAlmostEqual(unknown.score_mean, .9)
        a = finance[(finance.date == '2020-01-02') & finance.group.eq('A')].iloc[0]
        self.assertEqual((a['pass'], a.known_fail, a.unknown), (1, 1, 0))
        common = finance[finance.group.eq('common')].iloc[0]
        self.assertEqual(common.requested_count, 0)
        self.assertTrue(pd.isna(common.known_fraction))

    def test_attempt_source_mismatch_preserves_failure_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.json'
            source.write_text('{}')
            protocol = {'sources': {str(source): 'wrong'}, 'promotion_approved': False}
            attempt = root / 'run1'
            with self.assertRaisesRegex(ValueError, 'source_hash_mismatch'):
                self.m.execute_attempt(attempt, protocol, lambda output: self.fail('must not analyze'))
            self.assertEqual(json.loads((attempt / 'status.json').read_text())['status'], 'technical_failure')
            self.assertEqual(json.loads((attempt / 'protocol.json').read_text()), protocol)
            manifest = json.loads((attempt / 'manifest.json').read_text())['sha256']
            self.assertIn('status.json', manifest)
            with self.assertRaisesRegex(ValueError, 'attempt_exists'):
                self.m.execute_attempt(attempt, protocol, lambda output: None)
            self.assertEqual(self.m.verify_hashes({source: self.m.sha(source)})[str(source)], self.m.sha(source))

    def test_source_metadata_requires_successful_all_date_reconstruction_and_no_promotion(self):
        summary = dict(status='complete_conditional_offline_diagnostic', promotion_approved=False,
                       decisions=2, scored_rows=8, reconstruction=dict(all_passed=True, decisions=2,
                       max_score_error=0., max_saved_eligible_error=0., max_full_precision_mixture_error=0.,
                       candidate_order_equal=True, rounded_weights_equal=True))
        checks = dict(complete=True, expected_decisions=2, checks=[
            dict(date='2020-01-02', scored_count=4, union_count=2, max_score_error=0., max_saved_eligible_error=0.,
                 full_precision_mixture_error=0., candidate_order_equal=True, rounded_weights_equal=True),
            dict(date='2022-01-03', scored_count=4, union_count=1, max_score_error=0., max_saved_eligible_error=0.,
                 full_precision_mixture_error=0., candidate_order_equal=True, rounded_weights_equal=True)])
        self.m.validate_source_metadata(summary, checks, fixture(), expected_dates=2, expected_rows=8)
        summary['promotion_approved'] = True
        with self.assertRaisesRegex(ValueError, 'source_not_approved_for_diagnostic'):
            self.m.validate_source_metadata(summary, checks, fixture(), 2, 8)
        summary['promotion_approved'] = False
        checks['checks'][0]['candidate_order_equal'] = False
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            self.m.validate_source_metadata(summary, checks, fixture(), 2, 8)

    def test_end_to_end_archive_outputs_are_manifested_and_selection_is_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            scores = fixture()
            scores['return_60'] = scores.return_20
            scores['residual_60'] = scores.residual_20
            scores['coverage_60'] = scores.coverage_20
            scores.to_parquet(source / 'score_rank_membership.parquet', index=False)
            scores.to_parquet(source / 'complete_funnel_membership.parquet', index=False)
            scores.loc[scores.in_expert_union, ['date', 'instrument_id']].to_parquet(source / 'archived_expert_candidates.parquet', index=False)
            summary = dict(status='complete_conditional_offline_diagnostic', promotion_approved=False,
                           decisions=2, scored_rows=8, reconstruction=dict(all_passed=True, decisions=2,
                           max_score_error=0., max_saved_eligible_error=0., max_full_precision_mixture_error=0.,
                           candidate_order_equal=True, rounded_weights_equal=True))
            checks = dict(complete=True, expected_decisions=2, checks=[dict(date=d, scored_count=4,
                union_count=n, max_score_error=0., max_saved_eligible_error=0., full_precision_mixture_error=0.,
                candidate_order_equal=True, rounded_weights_equal=True) for d, n in [('2020-01-02', 2), ('2022-01-03', 1)]])
            (source / 'summary.json').write_text(json.dumps(summary))
            (source / 'reconstruction_checks.json').write_text(json.dumps(checks))
            protocol = dict(sources={str(p): self.m.sha(p) for p in source.iterdir()}, promotion_approved=False)
            output = root / 'run1'
            result = self.m.execute_attempt(output, protocol, lambda out: self.m.analyze_archives(out, source, 2, 8))
            self.assertEqual(result['decision']['decision'], 'insufficient_evidence')
            self.assertFalse(result['account_replayed'])
            self.assertEqual(result['membership']['A'], 3)
            members = pd.read_parquet(output / 'membership.parquet')
            self.assertEqual(len(members), 6)
            self.assertEqual(set(members.loc[members.in_B, 'financial_status']), {'unknown'})
            manifest = json.loads((output / 'manifest.json').read_text())['sha256']
            self.assertEqual(manifest['summary.json'], self.m.sha(output / 'summary.json'))
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'complete')
            self.assertLessEqual((output / 'protocol.json').stat().st_mtime_ns, (output / 'membership.parquet').stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError, 'attempt_exists'):
                self.m.execute_attempt(output, protocol, lambda out: None)

    def test_registration_anchors_manifest_then_verifies_every_manifest_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence.json'
            evidence.write_text('{}')
            summary = root / 'summary.json'
            summary.write_text('{}')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'sha256': {'evidence.json': self.m.sha(evidence)}}))
            anchors = {str(summary): self.m.sha(summary), str(manifest): self.m.sha(manifest)}
            protocol = self.m.registered_protocol(root, anchors, {'test': SCRIPT})
            self.assertIn(str(evidence), protocol['sources'])
            self.assertEqual(protocol['decision_rules']['main_cells'], 12)
            evidence.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source_hash_mismatch'):
                self.m.registered_protocol(root, anchors, {'test': SCRIPT})
            manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'source_hash_mismatch'):
                self.m.registered_protocol(root, anchors, {'test': SCRIPT})

    def test_identical_arms_keep_empty_exclusive_groups_null(self):
        scores = fixture().iloc[:4].copy()
        scores['in_expert_union'] = [False, False, True, True]
        archived = scores.loc[scores.in_expert_union, ['date', 'instrument_id']]
        members, _ = self.m.construct_membership(scores, archived, 1)
        metrics, pairs, coverage = self.m.measure(members, horizons=(20,))
        empty = metrics[metrics.group.isin(['A_only', 'B_only'])]
        self.assertEqual(empty.requested_count.tolist(), [0, 0])
        self.assertTrue(empty[['raw_mean', 'residual_mean']].isna().all().all())
        self.assertEqual(pairs.raw_difference.tolist(), [0.])
        self.assertEqual(pairs.residual_difference.tolist(), [0.])
        self.assertTrue(pairs.raw_complete.all())


if __name__ == '__main__':
    unittest.main()
