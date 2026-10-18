import csv
import tempfile
import unittest
from pathlib import Path

from quantlab import research_trials as T


class EffectiveOutcome(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.ledger = Path(self.folder.name) / 'ledger.csv'
        self.prereg = Path(self.folder.name) / 'prereg.md'
        self.prereg.write_text('rule')

    def append(self, **changes):
        rows = T.rows(self.ledger)
        row = dict(rows[-1])
        row.update(changes)
        with open(self.ledger, 'a', newline='', encoding='utf-8') as handle:
            csv.DictWriter(handle, fieldnames=T.COLUMNS).writerow(row)

    def test_latest_correction_wins(self):
        T.register('f', 't', prereg=str(self.prereg), ledger=self.ledger)
        self.append(trial_id='t#outcome1', outcome='registered')
        T.record_outcome('f', 't', 'pending', ledger=self.ledger)
        T.record_outcome('f', 't', 'pass', ledger=self.ledger)
        self.assertEqual(T.effective_outcome('f', 't', self.ledger), 'pass')

    def test_confirm_uses_reads_the_effective_row(self):
        T.register('f', 't', prereg=str(self.prereg), confirm_uses=1, ledger=self.ledger)
        self.assertEqual(T.confirm_uses('f', self.ledger), 1)
        self.append(trial_id='t#outcome1', outcome='pending', confirm_uses=0)
        self.assertEqual(T.confirm_uses('f', self.ledger), 0)


if __name__ == '__main__':
    unittest.main()
