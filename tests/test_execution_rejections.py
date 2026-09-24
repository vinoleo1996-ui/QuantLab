"""Execution-attempt history survives the later completion of an order."""
import unittest

from quantlab.backtest.ledger import Ledger, Lot
from tests.test_ledger import RULE, bar


class ExecutionRejectionHistory(unittest.TestCase):
    def sell_account(self, *, available_session=0, slippage=0.):
        ledger = Ledger(1000., slippage=slippage, cash_buffer=0., participation=1.)
        ledger.lots = {'A': [Lot(1000, available_session, 10., 0, '2024-01-01')]}
        ledger.marks = {'A': 10.}
        ledger.queue_targets({}, '2024-01-01', 'risk_exit', {'A': RULE})
        return ledger

    def test_slippage_rejection_survives_later_fill_without_duplicate_event(self):
        ledger = self.sell_account(slippage=.003)
        ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02', 9.01)}, {'A': RULE}, {'A': 1e8})
        self.assertEqual(ledger.cash, 1000.)
        self.assertEqual(ledger.quantity('A'), 1000)
        self.assertEqual(ledger.fills, [])
        self.assertEqual(ledger.orders[0].reject_reason, 'slippage_reaches_price_limit')
        rejected = [x for x in ledger.events if x['type'] == 'execution_rejected']
        self.assertEqual(len(rejected), 1)
        event = rejected[0].copy()
        self.assertEqual(event['reason'], 'slippage_reaches_price_limit')
        self.assertEqual((event['date'], event['order_id'], event['instrument_id'], event['side'], event['remaining']),
                         ('2024-01-02', 'O00000001', 'A', 'sell', 1000))
        self.assertEqual(event['execution_context']['raw_open'], 9.01)
        self.assertEqual(event['execution_context']['slipped_price'], 8.98)
        self.assertEqual(event['execution_context']['limit_down'], 9.)
        ledger.execute('2024-01-03', 2, {'A': bar('2024-01-03')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual(ledger.cash, 10960.01)
        self.assertEqual(ledger.quantity('A'), 0)
        self.assertEqual(ledger.orders[0].status, 'filled')
        self.assertEqual(ledger.orders[0].reject_reason, '')
        self.assertEqual([(f['quantity'], f['price'], f['fees']) for f in ledger.fills], [(1000, 9.97, 9.99)])
        self.assertEqual(ledger.export()['corporate_action_events'], [event])

    def test_t1_rejection_is_retained_after_shares_become_sellable(self):
        ledger = self.sell_account(available_session=2)
        ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((ledger.cash, ledger.quantity('A'), ledger.fills), (1000., 1000, []))
        self.assertEqual(len(ledger.events), 1)
        event = ledger.events[0].copy()
        self.assertEqual(event['reason'], 'settlement_or_liquidity_limit')
        self.assertIn('settlement_limit', event['causes'])
        self.assertEqual(event['execution_context']['available_quantity'], 0)
        ledger.execute('2024-01-03', 2, {'A': bar('2024-01-03')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((ledger.cash, ledger.quantity('A')), (10990., 0))
        self.assertEqual(ledger.orders[0].reject_reason, '')
        self.assertEqual(ledger.events, [event])

    def test_cash_and_exposure_rejections_record_observed_cause(self):
        for initial_cash, weight, can_buy, cause in [(1000., 1., None, 'cash_limit'),
                                                   (6000., .5, lambda *args: False, 'buy_permission_rejected')]:
            with self.subTest(cause=cause):
                ledger = Ledger(initial_cash, slippage=0, cash_buffer=0, participation=1)
                ledger.marks = {'A': 10.}
                ledger.queue_targets({'A': weight}, '2024-01-01', 'initial', {'A': RULE})
                ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, {'A': RULE}, {'A': 1e8}, can_buy)
                self.assertEqual((ledger.cash, ledger.quantity('A'), ledger.fills), (initial_cash, 0, []))
                self.assertEqual(len(ledger.events), 1)
                self.assertEqual(ledger.events[0]['reason'], 'cash_or_exposure_limit')
                self.assertIn(cause, ledger.events[0]['causes'])
                self.assertEqual(ledger.orders[0].status, 'expired')
                ledger.execute('2024-01-03', 2, {'A': bar('2024-01-03')}, {'A': RULE}, {'A': 1e8}, can_buy)
                self.assertEqual(len(ledger.events), 1)  # No second attempt on expired orders.

    def test_partial_sell_records_unfilled_remainder_once_then_completes(self):
        ledger = self.sell_account()
        ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, {'A': RULE}, {'A': 5000.})
        self.assertEqual((ledger.cash, ledger.quantity('A')), (5992.5, 500))
        self.assertEqual((ledger.orders[0].status, ledger.orders[0].reject_reason), ('partial', ''))
        self.assertEqual(len(ledger.events), 1)
        event = ledger.events[0].copy()
        self.assertEqual((event['outcome'], event['attempted_quantity'], event['filled_quantity'], event['remaining']),
                         ('partial', 1000, 500, 500))
        self.assertEqual(event['causes'], ['liquidity_limit'])
        ledger.execute('2024-01-03', 2, {'A': bar('2024-01-03')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((ledger.cash, ledger.quantity('A')), (10985., 0))
        self.assertEqual([f['quantity'] for f in ledger.fills], [500, 500])
        self.assertEqual(ledger.events, [event])

    def test_partial_buy_keeps_expiry_and_liquidity_reason(self):
        ledger = Ledger(100000., slippage=0, cash_buffer=0, participation=1)
        ledger.marks = {'A': 10.}
        ledger.queue_targets({'A': .1}, '2024-01-01', 'initial', {'A': RULE})
        ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, {'A': RULE}, {'A': 5000.})
        self.assertEqual((ledger.cash, ledger.quantity('A')), (94995., 500))
        self.assertEqual((ledger.orders[0].status, ledger.orders[0].reject_reason), ('expired', 'partial_day_order'))
        self.assertEqual(len(ledger.events), 1)
        self.assertEqual(ledger.events[0]['reason'], 'partial_day_order')
        self.assertEqual(ledger.events[0]['causes'], ['liquidity_limit'])
        self.assertEqual((ledger.events[0]['filled_quantity'], ledger.events[0]['remaining']), (500, 500))

    def test_repeated_limit_rejections_have_one_event_per_attempt(self):
        ledger = self.sell_account()
        for day, session in [('2024-01-02', 1), ('2024-01-03', 2)]:
            ledger.execute(day, session, {'A': bar(day, 9.)}, {'A': RULE}, {'A': 1e8})
        self.assertEqual([(e['date'], e['reason'], e['remaining']) for e in ledger.events],
                         [('2024-01-02', 'limit_down', 1000), ('2024-01-03', 'limit_down', 1000)])
        self.assertEqual(ledger.fills, [])
        ledger.execute('2024-01-04', 3, {'A': bar('2024-01-04')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((ledger.cash, ledger.quantity('A')), (10990., 0))
        self.assertEqual(len(ledger.events), 2)

    def test_unknown_rules_and_missing_liquidity_remain_traceable(self):
        for rules, adv, reason in [({}, {'A': 1e8}, 'unknown_rules'),
                                   ({'A': RULE}, {}, 'missing_prior_liquidity')]:
            with self.subTest(reason=reason):
                ledger = self.sell_account()
                ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, rules, adv)
                self.assertEqual((ledger.cash, ledger.quantity('A'), ledger.fills), (1000., 1000, []))
                self.assertEqual(len(ledger.events), 1)
                self.assertEqual(ledger.events[0]['reason'], reason)
                ledger.execute('2024-01-03', 2, {'A': bar('2024-01-03')}, {'A': RULE}, {'A': 1e8})
                self.assertEqual((ledger.cash, ledger.quantity('A')), (10990., 0))
                self.assertEqual(ledger.orders[0].reject_reason, '')
                self.assertEqual(len(ledger.events), 1)

    def test_tax_cash_rejection_is_retained_when_the_later_tax_rate_allows_sale(self):
        ledger = self.sell_account()
        ledger.lots['A'][0].dividends = [dict(per_share=100., ex_date='2024-01-01')]
        ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((ledger.cash, ledger.quantity('A'), ledger.fills), (1000., 1000, []))
        self.assertEqual(len(ledger.events), 1)
        self.assertEqual(ledger.events[0]['reason'], 'insufficient_cash_for_tax')
        self.assertEqual(ledger.events[0]['execution_context']['dividend_tax'], 20000.)
        ledger.execute('2025-01-03', 252, {'A': bar('2025-01-03')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((ledger.cash, ledger.quantity('A')), (10990., 0))
        self.assertEqual(ledger.fills[0]['dividend_tax'], 0.)
        self.assertEqual(ledger.orders[0].reject_reason, '')
        self.assertEqual(len(ledger.events), 1)

    def test_orders_not_yet_eligible_are_not_execution_rejections(self):
        ledger = self.sell_account()
        ledger.execute('2024-01-01', 0, {'A': bar('2024-01-01')}, {'A': RULE}, {'A': 1e8})
        ledger.orders[0].execute_not_before_session = 3
        ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((ledger.cash, ledger.quantity('A'), ledger.fills), (1000., 1000, []))
        self.assertEqual(ledger.events, [])

    def test_partial_settlement_and_cash_limits_do_not_change_fills(self):
        ledger = self.sell_account()
        ledger.lots['A'] = [Lot(400, 0, 10., 0), Lot(600, 2, 10., 1)]
        ledger.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((ledger.cash, ledger.quantity('A')), (4993., 600))
        self.assertEqual(ledger.events[0]['causes'], ['settlement_limit'])
        self.assertEqual((ledger.events[0]['filled_quantity'], ledger.events[0]['remaining']), (400, 600))
        buyer = Ledger(2000., slippage=0, cash_buffer=0, participation=1)
        buyer.marks = {'A': 10.}
        buyer.queue_targets({'A': 1.}, '2024-01-01', 'initial', {'A': RULE})
        buyer.execute('2024-01-02', 1, {'A': bar('2024-01-02')}, {'A': RULE}, {'A': 1e8})
        self.assertEqual((buyer.cash, buyer.quantity('A')), (995., 100))
        self.assertEqual(buyer.events[0]['causes'], ['cash_limit'])
        self.assertEqual((buyer.events[0]['filled_quantity'], buyer.events[0]['remaining']), (100, 100))


if __name__ == '__main__':
    unittest.main()
