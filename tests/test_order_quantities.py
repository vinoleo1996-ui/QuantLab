import unittest
from dataclasses import asdict, replace

import pandas as pd

from quantlab.backtest.ledger import Ledger, Lot
from quantlab.data.contracts import InstrumentRules
from quantlab.strategy.risk_trades import round_risk_sales
from test_ledger import RULE, bar


STAR = replace(RULE, instrument_id='sh.688001', buy_lot=1, sell_lot=1,
               min_buy_quantity=200, min_sell_quantity=200)


class OrderQuantityTests(unittest.TestCase):
    def test_minimum_declaration_is_distinct_from_increment(self):
        self.assertEqual(STAR.order_quantity(199, 'buy'), 0)
        self.assertEqual(STAR.order_quantity(201, 'buy'), 201)
        self.assertEqual(STAR.order_quantity(199, 'sell', holding=1000), 0)
        self.assertEqual(STAR.order_quantity(201, 'sell', holding=1000), 201)
        self.assertEqual(STAR.order_quantity(199, 'sell', holding=199), 199)
        self.assertEqual(RULE.order_quantity(150, 'sell', holding=1050), 100)
        self.assertEqual(RULE.order_quantity(50, 'sell', holding=1050), 0)
        self.assertEqual(RULE.order_quantity(50, 'sell', holding=50), 50)

    def test_old_contract_defaults_and_invalid_minimum(self):
        row = asdict(RULE)
        row.pop('min_buy_quantity'); row.pop('min_sell_quantity')
        self.assertEqual(InstrumentRules.from_record(row).minimum_quantity('sell'), 100)
        row['min_sell_quantity'] = 0
        with self.assertRaisesRegex(ValueError, 'invalid order minimum'):
            InstrumentRules.from_record(row)

    def test_ledger_does_not_submit_undersized_star_order(self):
        ledger = Ledger(20000, slippage=0, cash_buffer=0, min_trade_value=0)
        ledger.marks = {STAR.instrument_id: 10.}
        ledger.queue_targets({STAR.instrument_id: .0995}, '2024-01-02', 'initial', {STAR.instrument_id: STAR})
        self.assertEqual(ledger.orders, [])
        self.assertEqual(ledger.events[-1]['reason'], 'below_minimum_order_quantity')
        ledger.queue_targets({STAR.instrument_id: .1005}, '2024-01-03', 'initial', {STAR.instrument_id: STAR})
        self.assertEqual(ledger.orders[-1].quantity, 201)
        ledger.execute('2024-01-04', 1, {STAR.instrument_id: bar('2024-01-04')},
                       {STAR.instrument_id: STAR}, {STAR.instrument_id: 1e8})
        self.assertEqual(ledger.quantity(STAR.instrument_id), 201)
        self.assertGreaterEqual(ledger.cash, 0.)

    def test_risk_sales_obey_minimum_without_using_buy_multiples(self):
        code = STAR.instrument_id
        matrix = pd.DataFrame([[1.]], index=['gross'], columns=[code])
        for cap, expected in ((.4995, 800), (.3995, 799)):
            shares, proof = round_risk_sales({code: 1000}, {code: 10.}, 20000., {code: .5}, {code: STAR},
                linear_limits=[(matrix, pd.Series({'gross': cap}))])
            self.assertEqual(shares[code], expected)
            self.assertTrue(proof['projected_constraints_passed'])
            ledger = Ledger(20000, slippage=0, cash_buffer=0, min_trade_value=0)
            ledger.cash=10000.; ledger.marks={code:10.}; ledger.lots={code:[Lot(1000,0,10.,0)]}
            ledger.queue_targets({code: shares[code]*10/20000}, '2024-01-02', 'risk_exit',
                                 {code: STAR}, target_quantities=shares)
            self.assertEqual(ledger.orders[-1].quantity, 1000-expected)

