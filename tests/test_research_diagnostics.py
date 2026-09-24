import unittest

import pandas as pd

from quantlab.report.research_diagnostics import (
    drawdown_recovery, halfyear_returns, realized_concentration, turnover_and_exits)


class AccountDiagnosticTests(unittest.TestCase):
    def test_recovered_worst_drawdown_does_not_hide_current_unrecovered_loss(self):
        dates = pd.bdate_range('2024-01-02',periods=6)
        result = drawdown_recovery(pd.Series([1.,1.2,.9,1.1,1.2,1.1],index=dates))
        self.assertAlmostEqual(result['deepest']['depth'],-.25)
        self.assertTrue(result['deepest']['recovered'])
        self.assertEqual(result['deepest']['peak_to_recovery_sessions'],3)
        self.assertEqual(result['deepest']['trough_to_recovery_sessions'],2)
        self.assertTrue(result['unrecovered_at_end'])
        self.assertEqual(result['current']['elapsed_sessions'],1)
        first_fee = drawdown_recovery(pd.Series([.99,.99],index=dates[:2]))
        self.assertEqual(first_fee['deepest']['peak_basis'],'initial_capital')
        self.assertEqual(first_fee['deepest']['elapsed_sessions'],2)
        self.assertIsNone(first_fee['deepest']['peak_to_recovery_sessions'])

    def test_partial_half_year_is_not_a_complete_halfyear_result(self):
        calendar = pd.bdate_range('2023-01-02','2023-12-29')
        dates = calendar[calendar>='2023-03-01']
        returns = pd.Series(-.001,index=dates)
        returns.loc['2023-07-01':]=.0002
        result = halfyear_returns(returns,calendar)
        self.assertEqual(result['worst_observed']['period'],'2023H1')
        self.assertFalse(result['worst_observed']['complete'])
        self.assertEqual(result['worst_complete']['period'],'2023H2')
        self.assertAlmostEqual(result['worst_complete']['net_return'],
                               (1+returns.loc['2023-07-01':]).prod()-1)

    def test_concentration_uses_actual_drift_and_never_fills_missing_industry_or_groups(self):
        dates = pd.bdate_range('2023-01-02',periods=3)
        run = dict(account=[dict(date=dates[0],weights={}),dict(date=dates[1],weights={'A':.4,'B':.3}),
                            dict(date=dates[2],weights={'A':.6,'B':.2})],
            risk_history=[dict(date=dates[1],actual_sector_exposure={'UNKNOWN':.4,'I':.3})],
            signal_trace=[dict(date=dates[1],groups={'A':'g1','B':'g2'})])
        result = realized_concentration(run)
        self.assertEqual(result['single_security']['maximum'],.6)
        self.assertEqual(result['industry']['missing_sessions'],1)
        self.assertIsNone(result['daily'][-1]['max_industry'])
        self.assertEqual(result['unknown_industry']['maximum'],.4)
        self.assertEqual(result['group']['status'],'not_computable')
        self.assertIsNone(result['group']['maximum'])

    def test_two_way_turnover_and_pending_exit_clocks_use_actual_fills(self):
        dates = pd.bdate_range('2023-01-02',periods=4)
        orders = [dict(order_id='done',instrument_id='A',side='sell',reason='risk_exit',
                       signal_date=str(dates[0].date()),quantity=100,remaining=0,status='filled'),
                  dict(order_id='wait',instrument_id='B',side='sell',reason='risk_exit',
                       signal_date=str(dates[0].date()),quantity=100,remaining=80,status='partial'),
                  dict(order_id='cancel',instrument_id='C',side='sell',reason='rebalance',
                       signal_date=str(dates[0].date()),quantity=100,remaining=100,status='cancelled'),
                  dict(order_id='last',instrument_id='D',side='sell',reason='risk_exit',
                       signal_date=str(dates[-1].date()),quantity=100,remaining=100,status='pending')]
        fills = [dict(order_id='initial_buy',date=dates[0],side='buy',quantity=100,notional=100.),
                 dict(order_id='done',date=dates[2],side='sell',quantity=100,notional=200.),
                 dict(order_id='wait',date=dates[1],side='sell',quantity=20,notional=40.)]
        run = dict(account=[dict(date=d,nav=n) for d,n in zip(dates,[1000.,900.,800.,700.])],
                   calendar=dates,config={'capital':{'init_cash':1000.}},orders=orders,fills=fills,
                   capital_diagnostics={'risk_exits':[dict(order_id='done',elapsed_sessions=2)]})
        turnover, exits = turnover_and_exits(run)
        self.assertAlmostEqual(turnover['annualized_two_way'],252/4*(100/1000+200/900+40/1000))
        self.assertEqual(exits['all_sells']['completed_orders'],1)
        self.assertEqual(exits['all_sells']['completed_median_sessions'],2)
        self.assertEqual(exits['all_sells']['pending_orders'],2)
        self.assertEqual(exits['all_sells']['cancelled_or_other_incomplete_orders'],1)
        rows = {row['order_id']:row for row in exits['orders']}
        self.assertEqual(rows['wait']['elapsed_sessions'],3)
        self.assertEqual(rows['last']['elapsed_sessions'],0)
        self.assertIsNone(rows['cancel']['elapsed_sessions'])
        self.assertTrue(rows['done']['capital_diagnostic_consistent'])


if __name__=='__main__': unittest.main()
