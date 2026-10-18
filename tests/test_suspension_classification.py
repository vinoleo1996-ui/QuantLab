import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from stable_book import data_quality
from quantlab.data.suspensions import SuspensionEvidence, load_daily_evidence


class FakeEvidence:
    def __init__(self, statuses, complete=True):
        self.statuses = {code: {pd.Timestamp(day): status for day, status in rows.items()}
                         for code, rows in statuses.items()}
        self.complete = complete

    def get(self, code):
        if code not in self.statuses:
            return None
        return {'complete': self.complete}

    def session_status(self, code, day):
        if code not in self.statuses:
            return 'no_evidence'
        return self.statuses[code].get(pd.Timestamp(day), 'missing_row')


def snapshots(code, days):
    return [{'date': pd.Timestamp(day), 'stale': [code]} for day in days]


def evidence_for(code, days, status='suspended'):
    return FakeEvidence({code: {pd.Timestamp(day): status for day in days}})


class RealSuspensionEvidence(unittest.TestCase):
    def test_000616_boundaries_and_metadata(self):
        evidence = SuspensionEvidence()
        self.assertEqual(evidence.session_status('sz.000616', '2018-01-23'), 'traded')
        self.assertEqual(evidence.session_status('sz.000616', '2018-01-24'), 'suspended')
        self.assertEqual(evidence.session_status('sz.000616', '2018-07-24'), 'suspended')
        self.assertEqual(evidence.session_status('sz.000616', '2018-07-25'), 'traded')
        meta = evidence.metadata('sz.000616')
        self.assertTrue(meta['complete'])
        self.assertEqual(meta['coverage_start'], '2014-01-02')
        self.assertEqual(meta['coverage_end'], '2026-06-30')

    def test_code_without_any_cache_is_absent(self):
        self.assertIsNone(SuspensionEvidence().get('sz.999999'))

    def test_supplemental_cache_fills_primary_gaps(self):
        evidence = SuspensionEvidence()
        self.assertEqual(evidence.session_status('sz.000655', '2018-04-12'), 'suspended')
        self.assertEqual(evidence.session_status('sh.688237', '2026-07-16'), 'suspended')
        meta = evidence.metadata('sh.688237')
        self.assertIn('raw_ext_20261018', meta['path'])
        self.assertEqual(meta['coverage_start'], '2014-01-02')
        self.assertEqual(meta['coverage_end'], '2026-09-22')

    def test_nan_volume_on_suspended_row_counts_as_suspended(self):
        evidence = SuspensionEvidence()
        self.assertEqual(evidence.session_status('sh.600723', '2021-09-17'), 'suspended')

    def test_nan_volume_on_traded_row_is_not_suspended(self):
        evidence = SuspensionEvidence()
        self.assertEqual(evidence.session_status('sh.600656', '2016-05-12'), 'traded')


class SuspensionClassification(unittest.TestCase):
    def test_long_suspended_run_passes_quality_gate(self):
        code = 'sz.000616'
        days = pd.bdate_range('2018-01-24', periods=120)
        # The run must close inside the window. A suspension still open at the last session is a
        # supply gap until a delisting record explains it (see tests/test_stale_gate.py).
        out = data_quality(snapshots(code, days)
                           + [{'date': pd.bdate_range('2018-01-24', periods=121)[-1], 'stale': []}],
                           [], [], True, suspension_evidence=evidence_for(code, days))
        self.assertEqual(out['stale_held_max_days'], 120)
        self.assertEqual(out['suspended_held_max_days'], 120)
        self.assertEqual(out['unexplained_stale_held_max_days'], 0)
        self.assertTrue(out['suspension_coverage_complete'])
        self.assertTrue(out['data_quality_ok'])

    def test_missing_raw_file_is_unexplained_and_incomplete(self):
        code = 'sh.600017'
        days = pd.bdate_range('2018-01-24', periods=3)
        out = data_quality(snapshots(code, days), [], [], True,
                           suspension_evidence=FakeEvidence({}))
        self.assertEqual(out['unexplained_stale_held_max_days'], 3)
        self.assertFalse(out['suspension_coverage_complete'])
        self.assertEqual(out['raw_uncovered_stale_codes'], [code])
        self.assertFalse(out['data_quality_ok'])

    def test_one_contradictory_session_makes_whole_run_unexplained(self):
        code = 'sz.000616'
        days = pd.bdate_range('2018-01-24', periods=6)
        statuses = {day: 'suspended' for day in days}
        statuses[days[2]] = 'traded'
        out = data_quality(snapshots(code, days), [], [], True,
                           suspension_evidence=FakeEvidence({code: statuses}))
        self.assertEqual(out['unexplained_stale_held_max_days'], 6)
        self.assertTrue(out['suspension_coverage_complete'])
        self.assertEqual(out['unexplained_runs'][0]['reason'], 'not_suspended_on_session')
        self.assertEqual(out['unexplained_runs'][0]['contradictions'], [str(days[2].date())])
        self.assertFalse(out['data_quality_ok'])

    def test_missing_row_makes_coverage_incomplete(self):
        code = 'sz.000616'
        days = pd.bdate_range('2018-01-24', periods=3)
        statuses = {days[0]: 'suspended', days[1]: 'suspended'}
        out = data_quality(snapshots(code, days), [], [], True,
                           suspension_evidence=FakeEvidence({code: statuses}))
        self.assertEqual(out['unexplained_runs'][0]['reason'], 'missing_evidence')
        self.assertEqual(out['unexplained_runs'][0]['missing'], [str(days[2].date())])
        self.assertFalse(out['suspension_coverage_complete'])
        self.assertFalse(out['data_quality_ok'])

    def test_run_crossing_delist_date_is_unexplained(self):
        code = 'sz.000616'
        days = pd.bdate_range('2018-07-16', periods=6)
        out = data_quality(snapshots(code, days), [], [], True,
                           suspension_evidence=evidence_for(code, days),
                           delistings={code: days[-1]})
        self.assertEqual(out['unexplained_stale_held_max_days'], 6)
        self.assertEqual(out['unexplained_runs'][0]['reason'], 'run_crosses_delist_date')
        self.assertFalse(out['data_quality_ok'])

    def test_five_day_unexplained_boundary_is_unchanged(self):
        code = 'sz.000001'
        five = pd.bdate_range('2020-01-02', periods=5)
        six = pd.bdate_range('2020-01-02', periods=6)
        self.assertTrue(data_quality(snapshots(code, five), [], [], True)['data_quality_ok'])
        self.assertFalse(data_quality(snapshots(code, six), [], [], True)['data_quality_ok'])

    def test_legacy_without_evidence_keeps_stale_as_unexplained(self):
        code = 'sz.000616'
        days = pd.bdate_range('2018-01-24', periods=120)
        out = data_quality(snapshots(code, days), [], [], True, suspension_evidence=None)
        self.assertEqual(out['unexplained_stale_held_max_days'], 120)
        self.assertEqual(out['suspended_held_max_days'], 0)
        self.assertFalse(out['data_quality_ok'])

    def test_corrupt_raw_cache_is_unexplained_not_a_crash(self):
        code = 'sz.000616'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sz.000616_daily.csv'
            path.write_text('date,tradestatus,volume,amount,close,preclose\n'
                            '2018-01-24,0,0,0,10.0,10.0\n')
            (Path(tmp) / 'sz.000616_daily.json').write_text(json.dumps({
                'error_code': '0', 'sha256': hashlib.sha256(b'not-the-file').hexdigest(),
                'requested_start': '2014-01-02', 'requested_end': '2026-06-30', 'rows': 1}))
            evidence = load_daily_evidence(code, root=tmp)
            self.assertIsNotNone(evidence)
            self.assertFalse(evidence['complete'])
            self.assertTrue(evidence['reason'].startswith('raw_cache_unreadable'))
            days = pd.bdate_range('2018-01-24', periods=3)
            out = data_quality(snapshots(code, days), [], [], True,
                               suspension_evidence=SuspensionEvidence(root=tmp))
            self.assertEqual(out['unexplained_stale_held_max_days'], 3)
            self.assertFalse(out['suspension_coverage_complete'])
            self.assertFalse(out['data_quality_ok'])


if __name__ == '__main__':
    unittest.main()
