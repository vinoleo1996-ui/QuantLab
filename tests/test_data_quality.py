import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from stable_book import data_quality


class DataQualityGate(unittest.TestCase):
    def test_clean_replay_passes(self):
        out = data_quality([{'stale': []}, {'stale': []}], [], [], True)
        self.assertTrue(out['data_quality_ok'])
        self.assertEqual(out['stale_held_max_days'], 0)

    def test_missing_panel_blocks_evidence(self):
        self.assertFalse(data_quality([{'stale': []}], ['sz.000001'], [], True)['data_quality_ok'])

    def test_stale_run_above_five_days_blocks_evidence(self):
        snapshots = [{'stale': ['sz.000001']} for _ in range(6)]
        out = data_quality(snapshots, [], [{'reason': 'stale_close'}], True)
        self.assertEqual(out['stale_held_max_days'], 6)
        self.assertFalse(out['data_quality_ok'])
        self.assertEqual(data_quality(snapshots[:5], [], [], True)['data_quality_ok'], True)

    def test_unsettled_delisting_blocks_evidence(self):
        out = data_quality([{'stale': []}], [], [{'reason': 'delisting_settlement_unverified'}], True)
        self.assertTrue(out['unsettled_delisting'])
        self.assertFalse(out['data_quality_ok'])

    def test_failed_account_check_blocks_evidence(self):
        self.assertFalse(data_quality([{'stale': []}], [], [], False)['data_quality_ok'])


if __name__ == '__main__':
    unittest.main()
