import unittest

import pandas as pd

from quantlab.backtest.ledger import Ledger, Lot


class DelistingSettlement(unittest.TestCase):
    def ledger(self, haircut=0.0):
        ledger = Ledger(cash=0.0, delistings=pd.DataFrame(
            {'code': ['sz.000001'], 'delist_date': [pd.Timestamp('2020-01-03')]}),
            delisting_haircut=haircut)
        ledger.lots['sz.000001'] = [Lot(100, 0, 10.0, 0)]
        ledger.marks['sz.000001'] = 10.0
        return ledger

    def test_holding_is_settled_at_last_close_and_cash_increases(self):
        ledger = self.ledger()
        self.assertEqual(ledger.settle_delistings('2020-01-02', {'sz.000001': 99.0}), [])
        self.assertEqual(ledger.quantity('sz.000001'), 100)
        settled = ledger.settle_delistings('2020-01-03', {'sz.000001': 12.0})
        self.assertEqual(settled, ['sz.000001'])
        self.assertEqual(ledger.quantity('sz.000001'), 0)
        self.assertAlmostEqual(ledger.cash, 1200.0)
        self.assertEqual(ledger.fills[-1]['reason'], 'delisting')
        self.assertEqual(ledger.settle_delistings('2020-01-06', {}), [])

    def test_missing_price_falls_back_to_last_mark_and_haircut_applies(self):
        ledger = self.ledger(haircut=0.5)
        ledger.settle_delistings('2020-01-03', {})
        self.assertAlmostEqual(ledger.cash, 500.0)
        self.assertEqual(ledger.fills[-1]['haircut'], 0.5)

    def test_missing_price_is_reported_not_silently_dropped(self):
        ledger = Ledger(cash=0.0, delistings=pd.DataFrame(
            {'code': ['sz.000001'], 'delist_date': [pd.Timestamp('2020-01-03')]}))
        ledger.lots['sz.000001'] = [Lot(100, 0, 10.0, 0)]
        self.assertEqual(ledger.settle_delistings('2020-01-03', {}), [])
        self.assertEqual(ledger.quantity('sz.000001'), 100)
        self.assertEqual(ledger.data_issues[-1]['reason'], 'delisting_settlement_unverified')


if __name__ == '__main__':
    unittest.main()
