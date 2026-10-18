import csv
import tempfile
import unittest
from pathlib import Path

from quantlab import research_trials as T


class LedgerValidation(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.ledger = Path(self.folder.name) / 'ledger.csv'
        self.prereg = Path(self.folder.name) / 'prereg.md'
        self.prereg.write_text('rule')

    def append(self, **changes):
        rows = T.rows(self.ledger)
        row = dict(rows[-1] if rows else {column: '' for column in T.COLUMNS})
        row.update(changes)
        exists = self.ledger.exists()
        with open(self.ledger, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=T.COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def set_last(self, **changes):
        rows = T.rows(self.ledger)
        rows[-1].update(changes)
        with open(self.ledger, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=T.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_clean_ledger_passes(self):
        T.register('f', 't', prereg=str(self.prereg), ledger=self.ledger)
        self.assertEqual(T.validate_ledger(self.ledger), [])

    def test_invalid_outcome_is_an_error_until_a_correction_exists(self):
        T.register('f', 't', prereg=str(self.prereg), ledger=self.ledger)
        self.set_last(outcome='improvement')
        self.assertTrue(any('invalid outcome' in e for e in T.validate_ledger(self.ledger)))
        T.record_outcome('f', 't', 'pass', ledger=self.ledger)
        self.assertEqual(T.validate_ledger(self.ledger), [])
        report = T.ledger_report(self.ledger)
        self.assertTrue(any('improvement' in line for line in report['corrected']))

    def test_duplicate_base_id_is_an_error(self):
        T.register('f', 't', prereg=str(self.prereg), ledger=self.ledger)
        self.append(prereg_sha256=T.sha256_file(self.prereg))
        self.assertTrue(any('duplicate base trial_id' in e for e in T.validate_ledger(self.ledger)))

    def test_outcome_without_base_is_an_error(self):
        T.register('f', 't', prereg=str(self.prereg), ledger=self.ledger)
        self.append(trial_id='missing#outcome1', outcome='pass')
        self.assertTrue(any('without base' in e for e in T.validate_ledger(self.ledger)))

    def test_placeholder_prereg_and_hash_are_errors(self):
        T.register('f', 't', prereg=str(self.prereg), ledger=self.ledger)
        self.set_last(prereg='docs/prereg/—', prereg_sha256='—')
        errors = T.validate_ledger(self.ledger)
        self.assertTrue(any('missing prereg' in e for e in errors))
        self.assertTrue(any('placeholder prereg_sha256' in e for e in errors))


class P5RealizableLedger(unittest.TestCase):
    def test_diagnostic_base_and_outcome_are_registered_with_current_hash(self):
        rows = [r for r in T.rows() if r['family'] == 'P5_realizable']
        base = [r for r in rows if r['trial_id'] == 'P5_realizable']
        outcome = [r for r in rows if r['trial_id'] == 'P5_realizable#outcome1']
        self.assertEqual(len(base), 1)
        self.assertEqual(len(outcome), 1)
        self.assertEqual(base[0]['role'], 'diagnostic')
        self.assertEqual(base[0]['outcome'], 'insufficient_data')
        self.assertEqual(outcome[0]['outcome'], 'insufficient_data')
        self.assertEqual(base[0]['prereg'], 'docs/prereg/P5_REALIZABLE.md')
        self.assertEqual(base[0]['prereg_sha256'], T.sha256_file(T.ROOT / base[0]['prereg']))
        self.assertIn('withdrawn', base[0]['note'])
        self.assertIn('worst-path', base[0]['note'])


if __name__ == '__main__':
    unittest.main()
