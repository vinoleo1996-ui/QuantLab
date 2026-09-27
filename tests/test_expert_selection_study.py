"""Shared account controls and the two independent research interventions."""
import copy
import fcntl
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import expert_selection_study as s


class StudyTests(unittest.TestCase):
    def test_reference_identity_and_seed_vintage(self):
        codes = [f'{i:06d}.SZ' for i in range(200)]
        inputs = {'heads': {d: dict.fromkeys(codes, 1.) for d in ('2024-12-12', '2025-01-10')},
                  'industries': {d: {c: str(i % 10) for i, c in enumerate(codes)}
                                 for d in ('2024-12-12', '2025-01-10')}}
        days = ['2024-12-31', '2025-01-02', '2025-01-10']
        seen = []
        def provider(day):
            seen.append(day)
            return {}, {'status': 'none'}
        actual, trace = s.build_targets(inputs, days, provider, 'R')
        self.assertEqual(actual, s.reference.build_targets(inputs, days, 50))
        self.assertEqual(seen, ['2024-12-12', '2025-01-10'])
        self.assertTrue(trace[0]['warm_start'])
        priorities = {c: i / 200 for i, c in enumerate(codes)}
        old, _ = s.base.build_routed_targets(inputs, days, lambda d: (priorities, {}))
        new, _ = s.build_targets(inputs, days, lambda d: (priorities, {}), 'R')
        self.assertEqual(old, new)

    def test_s_changes_retention_targets_orders_and_fills(self):
        from tests.fixtures_v2 import dataset
        ds = dataset(60, 200)
        ds.panels = {c.lower(): p for c, p in ds.panels.items()}
        ds.instruments['instrument_id'] = ds.instruments.instrument_id.str.lower()
        codes = sorted(ds.panels)
        days = [str(ds.calendar[i].date()) for i in (25, 45)]
        inputs = {'heads': {d: dict.fromkeys(codes, 1.) for d in days},
                  'industries': {d: {c: c for c in codes} for d in days}}
        def provider(day):
            # Initially high code ranks; at the next decision the same old
            # targets remain the highest scores, despite code rank >150.
            return {c: (i+1)/len(codes) for i, c in enumerate(codes)}, {'status': 'ok'}
        schedules, results = [], []
        for arm in ('R', 'S'):
            targets, trace = s.build_targets(inputs, ds.calendar[25:], provider, arm)
            schedules.append(targets)
            results.append(s.replay.replay_reference(ds, {'targets': targets}, start=days[0],
                end=ds.calendar[-1], profile=s.replay.ReplayProfile.common(50)))
            if arm == 'S':
                self.assertEqual(trace[1]['retained_count'], 50)
                self.assertEqual(trace[1]['removals'], [])
        # A score shift is needed for old-vs-new retention to differ: R refills
        # high scores if they are not retained, so test a changed candidate list.
        inputs['heads'][days[1]].update({c: 1. for c in codes})
        def shifted(day):
            rank = {c: (i+1)/len(codes) for i,c in enumerate(codes)}
            if day == days[1]:
                rank = {c: ((i+80) % 200 + 1)/200 for i,c in enumerate(codes)}
            return rank, {'status': 'ok'}
        results, schedules = [], []
        for arm in ('R','S'):
            targets, _ = s.build_targets(inputs, ds.calendar[25:], shifted, arm)
            schedules.append(targets)
            results.append(s.replay.replay_reference(ds, {'targets':targets}, start=days[0],
                end=ds.calendar[-1], profile=s.replay.ReplayProfile.common(50)))
        self.assertNotEqual(schedules[0], schedules[1])
        self.assertNotEqual(results[0]['signal_trace'][1]['orders'], results[1]['signal_trace'][1]['orders'])
        self.assertNotEqual(results[0]['fills'], results[1]['fills'])
        self.assertTrue(all(row['cash'] >= 0 for result in results for row in result['account']))

    def fixture(self):
        from tests.fixtures_v2 import dataset
        ds = dataset(150, 1)
        dates = ds.calendar[[0, 21, 42, 63, 84]]
        rows, new = [], {}
        for day in dates:
            values = {}
            for i in range(40):
                code = f'sz.{i:06d}'
                row = dict(date=str(day.date()), instrument_id=code, return_20=i/1000,
                    coverage_20='observed', score_mixture=(i+1)/40)
                row.update({'score_'+e: (i+1)/40 for e in s.base.EXPERTS})
                rows.append(row); values[code] = (40-i)/40
            new[pd.Timestamp(day)] = dict(predictions=pd.Series(values), evidence={
                'status':'ok', 'as_of':str(s.decision_time(day)),
                'latest_label_available_at':str(s.decision_time(day)-pd.Timedelta(days=1)),
                'cohort_dates':[str((day-pd.Timedelta(days=30)).date())], 'version':'fixture'})
        p = s.router.RouterPolicy(min_cohorts=1)
        return ds, pd.DataFrame(rows), new, p

    def test_sixth_missing_falls_back_without_shrinking_original_pool(self):
        ds, frame, new, p = self.fixture()
        day = ds.calendar[65]
        new[ds.calendar[63]]['predictions'].iloc[0] = np.nan
        provider = s.ExpandedExperts(frame, ds.calendar, ds.benchmark, new, policy=p)
        expected = provider.baseline.predict(day, 'conditional_router')
        scores, evidence = provider.predict(day)
        self.assertEqual(scores, expected[0])
        self.assertEqual(evidence['fallback_reason'], 'missing_nonlinear_common_members')
        cohort = [c for c in provider.model.cohorts if c.signal_date == s.decision_time(ds.calendar[63])][0]
        self.assertEqual(cohort.requested, 40)
        self.assertEqual(cohort.status, 'missing_nonlinear_common_members')

    def test_future_unmatured_labels_do_not_change_sixth_route(self):
        ds, frame, new, p = self.fixture()
        day = ds.calendar[65]
        a = s.ExpandedExperts(frame, ds.calendar, ds.benchmark, new, policy=p).predict(day)
        frame.loc[frame.date >= str(ds.calendar[63].date()), 'return_20'] *= -200
        new[ds.calendar[84]]['predictions'] *= 0
        b = s.ExpandedExperts(frame, ds.calendar, ds.benchmark, new, policy=p).predict(day)
        self.assertEqual(a,b)
        self.assertEqual(a[1]['status'], 'ok')
        self.assertEqual(len(a[1]['weights']),6)

    def test_training_cutoff_and_insufficient_history_are_explicit(self):
        ds, frame, new, p = self.fixture()
        new[ds.calendar[63]]['evidence']['latest_label_available_at'] = str(s.decision_time(ds.calendar[64]))
        with self.assertRaisesRegex(ValueError, 'nonlinear_prediction_looks_ahead'):
            s.ExpandedExperts(frame, ds.calendar, ds.benchmark, new, policy=p)
        # Keep current sixth prediction, but none of its matured history. The
        # five-expert R still has adequate history and must be used in full.
        new = {ds.calendar[42]:new[ds.calendar[42]]}
        provider = s.ExpandedExperts(frame, ds.calendar, ds.benchmark, new, policy=p)
        result=provider.predict(ds.calendar[45])
        self.assertEqual(result[1]['fallback_reason'], 'insufficient_matured_history')

    def test_reuse_requires_every_execution_identity(self):
        identity = {'targets':'x','profile':{'initial_cash':600000},'period':['a','b'],
                    'parent_manifest':'hash','ledger':'hash2','replay':'hash3','cost_mult':1.}
        self.assertTrue(s.can_reuse(identity,copy.deepcopy(identity)))
        for key in identity:
            changed=copy.deepcopy(identity); changed[key]='changed'
            self.assertFalse(s.can_reuse(identity,changed))

    def test_solver_failure_returns_whole_R(self):
        ds, frame, new, p = self.fixture()
        provider=s.ExpandedExperts(frame,ds.calendar,ds.benchmark,new,policy=p)
        expected=provider.baseline.predict(ds.calendar[65],'conditional_router')[0]
        with patch.object(provider.model,'decide',return_value=s.router.RoutedDecision('solver_failed')):
            values,evidence=provider.predict(ds.calendar[65])
        self.assertEqual(values,expected)
        self.assertEqual(evidence['fallback_reason'],'solver_failed')

    def test_writer_lock_blocks_both_stages_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(s,'OUT',Path(tmp)):
            path=Path(tmp)/'study/writer.lock';path.parent.mkdir()
            @s.exclusive
            def writer():return 'done'
            with path.open('w') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'already_running'):writer()
            self.assertEqual(writer(),'done')

    def test_partial_result_is_preserved_and_recovered_only_by_identity(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(s,'progress'):
            folder=Path(tmp)/'R';folder.mkdir()
            payload={'study_protocol_sha256':'study','targets':[dict(signal_date='2025-01-02',weights={'sz.000001':.9})]}
            identity={'period':['2025-01-02','2025-01-31'],'cost_mult':1.,'profile':{'initial_cash':600000}}
            result=dict(study_protocol_sha256='study',parent_data_version=s.PARENT_VERSION,
                protocol=dict(profile=identity['profile'],start=identity['period'][0],end=identity['period'][1],
                    cost_multiplier=1.,target_schedule_sha256=s.replay.fingerprint(s.replay.read_targets(payload)[1])))
            (folder/'result.pkl').write_bytes(pickle.dumps(result))
            archive,recovered=s.preserve_partial(folder,identity,payload)
            self.assertEqual(recovered,result)
            self.assertFalse(folder.exists());self.assertTrue((archive/'result.pkl').exists())
            folder.mkdir();(folder/'result.pkl').write_bytes(pickle.dumps(result))
            _,recovered=s.preserve_partial(folder,dict(identity,cost_mult=2.),payload)
            self.assertIsNone(recovered)

    def test_comparison_resume_does_not_recompute_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(s,'OUT',Path(tmp)),patch.object(s,'PERIODS',{}),\
             patch.object(s,'verify',return_value=({'arms':['R','S']},{})),patch.object(s,'progress'):
            dest=s.stage_dir('rs');dest.mkdir(parents=True)
            (dest/'protocol.json').write_text('{}')
            value=dict(comparison_identity=dict(protocol_sha256=s.shared.sha256(dest/'protocol.json'),completed_accounts={}),marker='saved')
            path=dest/'comparison.json';path.write_text(json.dumps(value))
            original=path.read_bytes()
            self.assertEqual(s.compare('rs'),value)
            self.assertEqual(path.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
