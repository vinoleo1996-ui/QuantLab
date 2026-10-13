import unittest

from quantlab import p4_ledger


class P4LedgerTests(unittest.TestCase):
    def test_capital_is_swapped_only_inside_the_context(self):
        before = p4_ledger.book.INITIAL_CASH
        with p4_ledger.capital(800_000):
            self.assertEqual(p4_ledger.book.INITIAL_CASH, 800_000.0)
        self.assertEqual(p4_ledger.book.INITIAL_CASH, before)
        with self.assertRaises(RuntimeError), p4_ledger.capital(1.0):
            raise RuntimeError('boom')
        self.assertEqual(p4_ledger.book.INITIAL_CASH, before)

    def test_exposure_swaps_gross_and_cash_buffer_and_restores(self):
        self.assertEqual((p4_ledger.book.GROSS, p4_ledger.book.CASH_BUFFER), (0.90, 0.10))
        with p4_ledger.exposure(0.97):
            self.assertEqual(p4_ledger.book.GROSS, 0.97)
            self.assertAlmostEqual(p4_ledger.book.CASH_BUFFER, 0.03)
        self.assertEqual((p4_ledger.book.GROSS, p4_ledger.book.CASH_BUFFER), (0.90, 0.10))

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
