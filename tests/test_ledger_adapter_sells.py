"""Targets handed to stable_book.replay_account must keep listing names that are no longer targeted."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import stable_book as book
from tests.fixtures_v2 import dataset


class LedgerAdapterSells(unittest.TestCase):
    def test_unlisted_held_name_is_share_locked_by_the_ledger(self):
        ds = dataset(days=40, stocks=4)
        days = list(ds.calendar[:40])
        targets = [
            {'signal_date': str(days[22].date()), 'weights': {'S000': 0.4}, 'active': ['S000'], 'sell': [], 'sleeve': 0},
            {'signal_date': str(days[30].date()), 'weights': {'S001': 0.4}, 'active': ['S001'], 'sell': [], 'sleeve': 0},
        ]
        result = book.replay_account(ds, targets, start=days[0], end=days[-1])
        last = result['account'][-1]['quantities']
        self.assertGreater(last.get('S000', 0), 0)

    def test_adapters_list_every_previously_targeted_name(self):
        import p1_b_turn50
        import p1_m2
        schedule = {pd.Timestamp('2020-01-02'): {'sz.000001': 0.4},
                    pd.Timestamp('2020-02-03'): {'sz.000002': 0.4},
                    pd.Timestamp('2020-03-02'): {'sz.000003': 0.4}}
        for adapter in (p1_m2.to_targets, p1_b_turn50.to_targets):
            rows = adapter(schedule)
            self.assertEqual(rows[1]['sell'], ['000001.SZ'])
            self.assertEqual(rows[2]['sell'], ['000001.SZ', '000002.SZ'])

    def test_listed_sell_clears_the_old_name(self):
        ds = dataset(days=40, stocks=4)
        days = list(ds.calendar[:40])
        targets = [
            {'signal_date': str(days[22].date()), 'weights': {'S000': 0.4}, 'active': ['S000'], 'sell': [], 'sleeve': 0},
            {'signal_date': str(days[30].date()), 'weights': {'S001': 0.4}, 'active': ['S001'], 'sell': ['S000'],
             'sleeve': 0},
        ]
        result = book.replay_account(ds, targets, start=days[0], end=days[-1])
        self.assertEqual(result['account'][-1]['quantities'].get('S000', 0), 0)


if __name__ == '__main__':
    unittest.main()
