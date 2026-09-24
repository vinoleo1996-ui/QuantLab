import unittest
from dataclasses import replace

import pandas as pd

from quantlab.backtest.ledger import Ledger, Lot, trade_permission
from quantlab.data.contracts import InstrumentRules
from quantlab.strategy.risk import RiskManager
from quantlab.strategy.scoring import composite_score, weight_row


RULE = InstrumentRules("A", "stock", 100, 100, 1, 0.01, 0.00025, 5.0, 0.10)


def bar(date, price=10., status="trading", ref=10.):
    return {"open": price, "close": price, "open_status": status,
            "open_status_available_at": str(pd.Timestamp(date).date()) + " 09:25:00",
            "reference_price": ref, "limit_up": ref * 1.1, "limit_down": ref * .9}


class LedgerContracts(unittest.TestCase):
    def account(self):
        a = Ledger(100_000, slippage=0, cash_buffer=0, participation=1)
        a.mark("2024-01-01", {"A": 10, "B": 10})
        return a

    def test_risk_exit_does_not_round_unchanged_holdings_into_sales(self):
        # Real replay: 1800 shares round-trip through NAV/weight as
        # 1799.9999999999998. Odd lots from bonus shares must also be retained.
        for quantity in (1800, 1850):
            with self.subTest(quantity=quantity):
                a = Ledger(359425.16, min_trade_value=0)
                a.lots = {'A': [Lot(quantity, 0, 5.89, 0)],
                          'B': [Lot(100, 0, 2427.9, 0)]}
                a.marks = {'A': 5.89, 'B': 2427.9}
                rules = {'A': RULE, 'B': replace(RULE, instrument_id='B')}
                target = {'A': a.weights()['A'], 'B': 0.}
                a.queue_targets(target, '2024-01-02', 'risk_exit', rules)
                self.assertEqual([(o.instrument_id, o.side, o.quantity) for o in a.orders],
                                 [('B', 'sell', 100)])
                self.assertEqual(a.quantity('A'), quantity)

    def test_lot_boundary_roundoff_and_real_reduction_remain_distinct(self):
        a = self.account(); a.min_trade_value = 0
        # A mathematically whole-lot target can be represented just below 0.3.
        a.queue_targets({'A': .3-1e-16}, '2024-01-02', 'initial', {'A': RULE})
        self.assertEqual(a.orders[-1].quantity, 3000)
        a.lots = {'A': [Lot(3050, 0, 10., 0)]}; a.cash = 69500.
        a.queue_targets({'A': .29999}, '2024-01-03', 'risk_exit', {'A': RULE})
        self.assertEqual(a.orders[-1].side, 'sell')
        self.assertEqual(a.orders[-1].quantity, 100)

    def test_risk_exit_survives_rebalance_partial_fill_and_retry(self):
        a=self.account();a.cash=90_000;a.lots={'A':[Lot(1000,0,10,0)]};rules={'A':RULE}
        a.queue_targets({},'2024-01-02','risk_exit',rules)
        original=a.orders[0]
        a.execute('2024-01-03',1,{'A':bar('2024-01-03',status='halted')},rules,{'A':1e8})
        a.queue_targets({'A':.5},'2024-01-03','opportunity',rules)
        self.assertEqual(original.status,'pending')
        self.assertFalse(any(o.side=='buy' and o.status=='pending' for o in a.orders))
        a.execute('2024-01-04',2,{'A':bar('2024-01-04')},rules,{'A':5000})
        self.assertEqual(a.quantity('A'),500)
        a.queue_targets({},'2024-01-04','risk_exit',rules)
        self.assertEqual(sum(o.remaining for o in a.orders if o.side=='sell' and o.status in ('pending','partial')),500)
        a.execute('2024-01-05',3,{'A':bar('2024-01-05')},rules,{'A':1e8})
        self.assertEqual(a.quantity('A'),0)
        self.assertEqual(sum(f['quantity'] for f in a.fills),1000)
        self.assertTrue(all(f['signal_date']=='2024-01-02' for f in a.fills))

    def test_full_rotation_pays_both_sides(self):
        a = self.account(); rules = {"A": RULE, "B": replace(RULE, instrument_id="B")}
        a.queue_targets({"A": .9}, "2024-01-01", "initial", rules)
        a.execute("2024-01-02", 1, {"A": bar("2024-01-02")}, rules, {"A": 1e8})
        a.queue_targets({"B": .9}, "2024-01-02", "rotation", rules)
        a.execute("2024-01-03", 2, {c: bar("2024-01-03") for c in rules}, rules, {c: 1e8 for c in rules})
        self.assertEqual(a.quantity("A"), 0)
        self.assertEqual([f["side"] for f in a.fills], ["buy", "sell", "buy"])
        self.assertGreater(sum(f["fees"] for f in a.fills), 0)
        self.assertAlmostEqual(a.nav(), a.initial_cash - sum(f["fees"] for f in a.fills))

    def test_blocked_sale_cannot_finance_new_buy(self):
        a = self.account(); rules = {"A": RULE, "B": replace(RULE, instrument_id="B")}
        a.queue_targets({"A": .99}, "2024-01-01", "initial", rules)
        a.execute("2024-01-02", 1, {"A": bar("2024-01-02")}, rules, {"A": 1e8})
        old = a.quantity("A")
        a.queue_targets({"B": .99}, "2024-01-02", "stop", rules)
        a.execute("2024-01-03", 2, {"A": bar("2024-01-03", 9), "B": bar("2024-01-03")}, rules, {c: 1e8 for c in rules})
        self.assertEqual(a.quantity("A"), old)
        self.assertGreaterEqual(a.cash, 0)
        self.assertLessEqual(sum(a.weights().values()), 1.0 + 1e-10)
        self.assertTrue(any(o.reject_reason == "limit_down" for o in a.orders))

    def test_drift_missing_quote_and_resumption(self):
        a = self.account(); a.cash = 80_000
        a.lots = {"A": [Lot(1000, 0, 10, 0)], "B": [Lot(1000, 0, 10, 0)]}
        a.close("2024-01-01", 0, {"A": 10, "B": 10})
        a.close("2024-01-02", 1, {"A": float("nan"), "B": 11})
        self.assertEqual(a.quantity("B"), 1000)
        self.assertNotAlmostEqual(a.weights()["A"], a.weights()["B"])
        a.close("2024-01-03", 2, {"A": 8, "B": 10})
        self.assertEqual(a.nav(), 98_000)
        self.assertTrue(a.data_issues)

    def test_end_signal_waits_and_t1_is_enforced(self):
        a = self.account(); rules = {"A": RULE}
        a.queue_targets({"A": .5}, "2024-01-02", "buy", rules)
        a.execute("2024-01-02", 1, {"A": bar("2024-01-02")}, rules, {"A": 1e8})
        self.assertFalse(a.fills)
        a.execute("2024-01-03", 2, {"A": bar("2024-01-03")}, rules, {"A": 1e8})
        self.assertEqual(a.available("A", 2), 0)
        self.assertGreater(a.available("A", 3), 0)

    def test_etf_no_stock_tax_and_t0(self):
        r = replace(RULE, asset_type="equity_etf", sell_delay=0)
        a = self.account()
        a.queue_targets({"A": .5}, "2024-01-01", "buy", {"A": r})
        a.execute("2024-01-02", 1, {"A": bar("2024-01-02")}, {"A": r}, {"A": 1e8})
        self.assertEqual(a.available("A", 1), a.quantity("A"))
        self.assertEqual(a.fee("2024-01-02", r, "sell", 50000)[1], 0)
        self.assertEqual(a.fee("2023-08-25", RULE, "sell", 50000)[1], 50)
        self.assertEqual(a.fee("2023-08-28", RULE, "sell", 50000)[1], 25)

    def test_missing_amount_not_halt_and_future_amount_not_permission(self):
        b = bar("2024-01-02"); b["amount"] = float("nan")
        self.assertTrue(trade_permission(b, RULE, "buy", "2024-01-02")[0])
        b["open_status"] = "unknown"; b["amount"] = 1e9
        self.assertFalse(trade_permission(b, RULE, "buy", "2024-01-02")[0])

    def test_dividend_bonus_and_payment_conserve_value(self):
        a = self.account(); a.cash = 90_000; a.lots = {"A": [Lot(1000, 0, 10, 0)]}
        a.close("2024-01-02", 0, {"A": 10})
        events = pd.DataFrame([dict(action_id="D1", instrument_id="A", record_date="2024-01-02",
                         ex_date="2024-01-03", pay_date="2024-01-05", share_date="2024-01-04",
                         available_at="2024-01-01", cash_per_share=1., share_ratio=1.)])
        calendar = pd.bdate_range("2024-01-02", "2024-01-08")
        a.corporate_actions("2024-01-03", 1, events, calendar)
        a.close("2024-01-03", 1, {"A": 4.5})
        self.assertEqual(a.quantity("A"), 2000)
        self.assertEqual(a.available("A", 1), 1000)
        self.assertEqual(a.nav(), 100_000)
        self.assertEqual(a.cash, 90_000)
        a.corporate_actions("2024-01-05", 3, events, calendar)
        self.assertEqual(a.cash, 91_000); self.assertEqual(a.nav(), 100_000)


class SignalAndRiskContracts(unittest.TestCase):
    def test_scores_transmit_and_nan_weights_fail(self):
        d = pd.Timestamp("2024-01-01")
        f = {"trend": pd.DataFrame([[1, 2, 3]], index=[d], columns=list("ABC"))}
        w = weight_row(pd.Series({"trend": 1.}), d)
        positive = composite_score(f, w, None).iloc[0]
        negative = composite_score(f, -w, None).iloc[0]
        self.assertEqual(positive.idxmax(), negative.idxmin())
        with self.assertRaisesRegex(ValueError, "missing_factor_weights"):
            composite_score(f, w * float("nan"), None)
        constant = composite_score({"trend": f["trend"] * 0}, w, None)
        self.assertTrue(constant.isna().all().all())

    def test_daily_peak_hysteresis_and_manual_halt(self):
        r = RiskManager()
        r.evaluate(100, date="2024-01-01")
        r.evaluate(120, date="2024-01-02")
        self.assertEqual(r.evaluate(99, date="2024-01-03")["level"], 2)
        for d in pd.bdate_range("2024-01-04", periods=4):
            state = r.evaluate(110, date=d)
            r.evaluate(110, date=d)  # Duplicate processing cannot advance hysteresis.
            self.assertEqual(state["level"], 2)
        self.assertEqual(r.evaluate(110, date="2024-01-10")["level"], 1)
        state = r.evaluate(90, date="2024-01-11")
        self.assertTrue(state["manual_review"])
        self.assertEqual(r.evaluate(125, date="2024-01-12")["gross_scale"], 0)


if __name__ == "__main__": unittest.main()
