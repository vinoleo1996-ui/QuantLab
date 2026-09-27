"""Contract tests for the reference observer, without market outcome data."""
import importlib
import importlib.util
import json
import unittest
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def fixture():
    inputs = {'heads': {'2024-12-12': {'A': 1., 'B': 1., 'C': 0.}},
              'industries': {'2024-12-12': {'A': 'I', 'B': 'J', 'C': 'I'}}}
    targets = {'targets': [{'signal_date': '2024-12-31', 'weights': {'A': .45, 'C': .45},
                            'max_exposure': .9, 'sell_only': False}]}
    old = [{'period': 'recent', 'signal_date': '2024-12-31', 'information_date': '2024-12-12',
            'period_end': '2025-12-31', 'heads': inputs['heads']['2024-12-12'],
            'industries': inputs['industries']['2024-12-12'], 'selected': ['A', 'B']}]
    return inputs, targets, old


class ReferenceDiagnosticTests(unittest.TestCase):
    def test_warm_start_uses_information_snapshot_and_reference_members(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.reference_selection_diagnostic'),
                             'reference observer implementation is missing')
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        inputs, targets, old = fixture()
        rows = api.build_decisions('recent', '2025-12-31', inputs, targets, old)
        self.assertEqual(rows[0]['information_date'], '2024-12-12')
        self.assertEqual(rows[0]['selected'], ['A', 'C'])
        self.assertEqual(rows[0]['old_index'], 0)
        self.assertEqual(rows[0]['retained_count'], 0)

    def test_reuse_rejects_any_changed_primary_sampling_identity(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        changes = [
            ('heads', {'A': 0., 'B': 1., 'C': 0.}),
            ('industries', {'A': 'J', 'B': 'J', 'C': 'I'}),
            ('information_date', '2024-12-13'),
            ('signal_date', '2024-12-30'),
            ('period', 'another'),
            ('period_end', '2025-12-30'),
            ('selected', ['A']),
        ]
        for field, value in changes:
            with self.subTest(field=field):
                inputs, targets, old = fixture()
                old[0][field] = value
                with self.assertRaisesRegex(ValueError, 'identity'):
                    api.build_decisions('recent', '2025-12-31', inputs, targets, old)

    def test_memberships_reuse_primary_but_match_reference_strata(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        self.assertTrue(hasattr(api, 'make_memberships'), 'member freeze is missing')
        inputs, targets, old = fixture()
        specs = api.build_decisions('recent', '2025-12-31', inputs, targets, old)
        original = {'0_primary': np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int32),
                    '0_strata': np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int32)}
        result = api.make_memberships(specs, original, draws=3)
        np.testing.assert_array_equal(result['0_primary'], original['0_primary'])
        for row in result['0_strata']:
            self.assertEqual(set(row.tolist()), {0, 2})
        again = api.make_memberships(specs, original, draws=3)
        np.testing.assert_array_equal(result['0_strata'], again['0_strata'])
        original['0_primary'][0] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            api.make_memberships(specs, original, draws=3)

    def test_labels_reconstruct_observer_and_enforce_calendar_endpoints(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        self.assertTrue(hasattr(api, 'reconstruct_labels'), 'label reconstruction is missing')
        inputs, targets, old = fixture()
        specs = api.build_decisions('recent', '2025-12-31', inputs, targets, old)
        calendar = pd.bdate_range('2024-12-31', periods=31)
        rows = [dict(period='recent', signal_date='2024-12-31', instrument=c,
                     horizon=h, value=(.1 if h == 20 else None),
                     reason=('observed' if h == 20 else 'immature'), selected=(c == 'B'))
                for c in ['A', 'B', 'C'] for h in [20, 60]]
        rebuilt = api.reconstruct_labels(pd.DataFrame(rows), specs, calendar)
        self.assertEqual(set(rebuilt.loc[rebuilt.selected, 'instrument']), {'A', 'C'})
        self.assertEqual(set(rebuilt.loc[rebuilt.horizon.eq(20), 'endpoint']), {'2025-01-28'})
        self.assertTrue(rebuilt.loc[rebuilt.horizon.eq(60), 'endpoint'].isna().all())
        rows[1]['value'], rows[1]['reason'] = .5, 'observed'
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            api.reconstruct_labels(pd.DataFrame(rows), specs, calendar)

    def test_missing_member_invalidates_whole_basket_without_renormalization(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        self.assertTrue(hasattr(api, 'basket_statistics'), 'basket statistics are missing')
        values = np.array([.2, .4, np.nan])
        draws = np.array([[0, 1], [0, 2]])
        result = api.basket_statistics(values, np.array([0, 2]), draws)
        self.assertIsNone(result['selected_mean'])
        self.assertIsNone(result['selected_minus_random'])
        self.assertEqual(result['paired_draws'], 0)
        self.assertEqual(result['random_full_known_fraction'], .5)
        self.assertAlmostEqual(result['selected_lower_bound'], -.4)
        self.assertEqual(result['selected_upper_bound'], 'unbounded')
        complete = api.basket_statistics(values, np.array([0, 1]), draws)
        self.assertEqual(complete['paired_draws'], 1)
        self.assertAlmostEqual(complete['selected_mean'], .3)
        self.assertAlmostEqual(complete['random_conditional_mean'], .3)
        self.assertAlmostEqual(complete['selected_minus_random'], 0.)
        self.assertAlmostEqual(complete['random_average_lower_bound'], -.05)

    def test_calendar_bootstrap_matches_original_eight_and_twelve_is_wider(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        self.assertTrue(hasattr(api, 'bootstrap_comparisons'), 'bootstrap interface is missing')
        from scripts.router_random_diagnostic import bootstrap
        calendar = pd.bdate_range('2024-01-01', periods=120)
        values = [(calendar[i], value) for i, value in [(0, -.04), (22, .02), (50, .07), (90, -.01)]]
        original = bootstrap(values, calendar, 20, 19)
        result = api.bootstrap_comparisons(values, calendar, 20, 19)
        for key in original:
            self.assertEqual(result[key], original[key], key)
        twelve = api.bootstrap_comparisons(values, calendar, 20, 19, family_tests=12)
        self.assertLessEqual(twelve['ci_family95_pp'][0], result['ci_family95_pp'][0])
        self.assertGreaterEqual(twelve['ci_family95_pp'][1], result['ci_family95_pp'][1])
        self.assertEqual(twelve['calendar_sessions'], 120)
        self.assertEqual(twelve['observed_dates'], 4)

    def test_outcome_content_cannot_load_before_membership_freeze(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        self.assertTrue(hasattr(api, 'load_frozen_labels'), 'freeze barrier is missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'unreadable.parquet'
            source.write_text('not valid parquet; must never be reached')
            with self.assertRaisesRegex(ValueError, 'memberships_not_frozen'):
                api.load_frozen_labels(root, source, [], pd.bdate_range('2024-01-01', periods=80))
            self.assertFalse((root / 'label_read_event.json').exists())

    def test_analysis_keeps_unknown_period_cells_and_literal_price_difference(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        self.assertTrue(hasattr(api, 'analyse'), 'analysis assembly is missing')
        inputs, targets, old = fixture()
        specs = api.build_decisions('recent', '2025-12-31', inputs, targets, old)
        calendar = pd.bdate_range('2024-12-31', periods=31)
        rows = [dict(period='recent', signal_date='2024-12-31', instrument=c,
                     horizon=h, value=(v if h == 20 else None),
                     reason=('observed' if h == 20 else 'immature'))
                for c, v in [('A', .2), ('B', .4), ('C', .8)] for h in [20, 60]]
        labels = api.reconstruct_labels(pd.DataFrame(rows), specs, calendar)
        members = {'0_primary': np.array([[0, 1], [1, 0]]),
                   '0_strata': np.array([[0, 2], [2, 0]])}
        per_date, summary = api.analyse(specs, members, labels, calendar)
        self.assertEqual(len(per_date), 4)
        self.assertAlmostEqual(summary[0]['mean_pp'], 20.)
        self.assertEqual(summary[0]['decision'], 'insufficient')
        self.assertIsNone(summary[2]['mean_pp'])
        self.assertEqual(summary[2]['observed_dates'], 0)
        self.assertEqual(per_date[2]['pool_missing_reasons'], {'immature': 3})

    def test_single_run_rejects_existing_artifact_before_loading_inputs(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        self.assertTrue(hasattr(api, 'run'), 'single-run orchestrator is missing')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'partial_result.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'output_not_empty'):
                api.run(output)

    def test_future_snapshot_cannot_change_warm_start(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        inputs, targets, old = fixture()
        inputs['heads']['2025-01-01'] = {'FUTURE': 1.}
        inputs['industries']['2025-01-01'] = {'FUTURE': 'X'}
        result = api.build_decisions('recent', '2025-12-31', inputs, targets, old)
        self.assertEqual(result[0]['information_date'], '2024-12-12')
        self.assertNotIn('FUTURE', result[0]['heads'])
        del inputs['heads']['2024-12-12']
        with self.assertRaisesRegex(ValueError, 'timing'):
            api.build_decisions('recent', '2025-12-31', inputs, targets, old)

    def test_changed_membership_file_blocks_outcome_read(self):
        api = importlib.import_module('scripts.reference_selection_diagnostic')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member = root / 'memberships.npz'
            member.write_bytes(b'original members')
            expected = api.sha(member)
            (root / 'membership_identity.json').write_text(json.dumps({
                'stage': 'memberships_frozen_before_outcomes',
                'hashes': {'memberships.npz': expected}}))
            member.write_bytes(b'changed members')
            with self.assertRaisesRegex(ValueError, 'frozen_hash_changed'):
                api.load_frozen_labels(root, root / 'no_labels.parquet', [],
                                       pd.bdate_range('2024-01-01', periods=80))
            self.assertFalse((root / 'label_read_event.json').exists())

if __name__ == '__main__':
    unittest.main()
