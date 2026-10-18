import unittest
from unittest import mock

from quantlab import p4_ledger

TARGETS = [{'signal_date': '2024-01-02', 'weights': {'sh.600000': 0.5},
            'active': ['sh.600000'], 'sell': []}]


class P4LedgerTests(unittest.TestCase):
    def forwarded(self, **kwargs):
        """Capture the arguments ``replay`` hands to the account replay, without touching a panel."""
        seen = {}

        def fake_replay_account(dataset, targets, **call):
            seen.update(call)
            return {'summary': {}}

        with mock.patch.object(p4_ledger.book, 'replay_account', fake_replay_account):
            p4_ledger.replay(TARGETS, start='2024-01-02', end='2024-01-03',
                             dataset=object(), **kwargs)
        return seen

    def test_account_parameters_are_forwarded_not_swapped_into_globals(self):
        before = (p4_ledger.book.GROSS, p4_ledger.book.CASH_BUFFER, p4_ledger.book.INITIAL_CASH)
        call = self.forwarded(capital_cny=800_000.0, gross=0.97)
        self.assertEqual(call['capital'], 800_000.0)
        self.assertEqual(call['gross'], 0.97)
        self.assertIsNone(call['cash_buffer'])
        self.assertEqual((p4_ledger.book.GROSS, p4_ledger.book.CASH_BUFFER,
                          p4_ledger.book.INITIAL_CASH), before)

    def test_missing_capital_falls_back_to_the_mandate(self):
        self.assertEqual(self.forwarded()['capital'], p4_ledger.MANDATE.capital_cny)

    def test_sell_list_keeps_every_previously_targeted_name(self):
        rows = p4_ledger.to_targets({'2024-01-02': {'sh.600000': .5, 'sz.000001': .4},
                                     '2024-02-01': {'sz.000001': .9},
                                     '2024-03-01': {'sz.000002': .9}})
        self.assertEqual(rows[0]['sell'], [])
        self.assertEqual(rows[1]['sell'], ['sh.600000'])
        self.assertEqual(rows[2]['sell'], ['sh.600000', 'sz.000001'])
        self.assertEqual(rows[2]['active'], ['sz.000002'])


if __name__ == '__main__':
    unittest.main()
