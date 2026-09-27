"""The router must reach real next-session orders and fills."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class RoutedReferenceTests(unittest.TestCase):
    def test_no_expert_overlay_exactly_reproduces_reference_and_warm_start(self):
        import reference_expert_router as s
        codes = [f'{i:06d}.SZ' for i in range(200)]
        inputs = {'heads': {d: {c: float(i % 2) for i, c in enumerate(codes)}
                            for d in ('2024-12-12', '2025-01-10')},
                  'industries': {d: {c: str(i % 10) for i, c in enumerate(codes)}
                                 for d in ('2024-12-12', '2025-01-10')}}
        seen = []
        def provider(day):
            seen.append(day)
            return {}, {'status': 'unavailable'}
        days = ['2024-12-31', '2025-01-02', '2025-01-10']
        actual, traces = s.build_routed_targets(inputs, days, provider)
        expected = s.reference.build_targets(inputs, days, 50)
        self.assertEqual(actual, expected)
        self.assertEqual(seen, ['2024-12-12', '2025-01-10'])
        self.assertTrue(traces[0]['warm_start'])

    def test_archived_experts_reject_stale_and_ignore_future_changes(self):
        import reference_expert_router as s
        self.assertTrue(hasattr(s, 'ArchivedExperts'), 'Dated expert archive is not wired')
        from tests.fixtures_v2 import dataset
        ds = dataset(120, 1)
        x = []
        for day in (ds.calendar[0], ds.calendar[21], ds.calendar[45]):
            for i in range(60):
                row = dict(date=str(day.date()), instrument_id=f'sz.{i:06d}',
                           return_20=i/1000, coverage_20='observed', score_mixture=(i+1)/60)
                row.update({'score_'+e: (i+1)/60 for e in s.EXPERTS})
                x.append(row)
        frame = pd.DataFrame(x)
        p = s.router.RouterPolicy(min_cohorts=1)
        a = s.ArchivedExperts(frame, ds.calendar, ds.benchmark, policy=p)
        values, evidence = a.predict(ds.calendar[23], 'conditional_router')
        self.assertTrue(values)
        self.assertEqual(evidence['score_date'], str(ds.calendar[21].date()))
        changed = frame.copy()
        future = changed.date > str(ds.calendar[23].date())
        changed.loc[future, 'return_20'] *= -100
        changed.loc[future, 'score_momentum'] = 0
        b = s.ArchivedExperts(changed, ds.calendar, ds.benchmark, policy=p)
        self.assertEqual((values, evidence), b.predict(ds.calendar[23], 'conditional_router'))
        self.assertFalse(a.predict(ds.calendar[90], 'conditional_router')[0])
        self.assertEqual(a.predict(ds.calendar[90], 'original_experts')[1]['status'], 'stale_expert_snapshot')

    def test_contextual_predictions_change_targets_orders_and_actual_fills(self):
        self.assertIsNotNone(importlib.util.find_spec('reference_expert_router'),
                             'Router is not connected to reference account targets')
        import reference_expert_router as s
        from tests.fixtures_v2 import dataset
        ds = dataset(45, 60)
        ds.panels = {c.lower(): p for c, p in ds.panels.items()}
        ds.instruments['instrument_id'] = ds.instruments.instrument_id.str.lower()
        codes = sorted(ds.panels)
        x = pd.DataFrame({'trend': np.linspace(.01, 1, 60),
                          'defensive': np.linspace(1, .01, 60)}, index=codes)
        cohorts = []
        for i, day in enumerate(pd.date_range('2020-01-01', periods=24, freq='MS')):
            state = 'active' if i % 2 else 'defensive'
            y = x.trend if state == 'active' else x.defensive
            cohorts.append(s.router.cohort_from_outcomes(day, day + pd.Timedelta(days=20), state, x, y))
        model = s.router.ExpertRouter(cohorts)
        day = str(ds.calendar[25].date())
        inputs = {'heads': {day: dict.fromkeys(codes, 1.)},
                  'industries': {day: {c: c for c in codes}}}
        schedules, accounts = [], []
        for state in ('active', 'defensive'):
            def provider(as_of):
                decision = model.decide(as_of, state, x)
                return decision.scores.to_dict(), decision.evidence
            targets, _ = s.build_routed_targets(inputs, [day], provider)
            result = s.replay.replay_reference(ds, {'targets': targets}, start=day,
                end=ds.calendar[35], profile=s.replay.ReplayProfile.common(50))
            schedules.append(targets)
            accounts.append(result)
        self.assertNotEqual(schedules[0], schedules[1])
        self.assertNotEqual(accounts[0]['signal_trace'][0]['orders'], accounts[1]['signal_trace'][0]['orders'])
        bought = [{f['instrument_id'] for f in a['fills'] if f['side'] == 'buy'} for a in accounts]
        self.assertIn('s059', bought[0]); self.assertNotIn('s000', bought[0])
        self.assertIn('s000', bought[1]); self.assertNotIn('s059', bought[1])
        for result in accounts:
            self.assertFalse(result['account'][0]['quantities'])
            self.assertTrue(all(pd.Timestamp(f['date']) > pd.Timestamp(day) for f in result['fills']))
            self.assertTrue(all(a['cash'] >= 0 for a in result['account']))


if __name__ == '__main__':
    unittest.main()
