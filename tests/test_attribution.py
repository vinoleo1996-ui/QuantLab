import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quantlab.backtest.ledger import Ledger
from quantlab.data.contracts import InstrumentRules
from quantlab.report.attribution import concentration_diagnostics, instrument_returns


def distribution_run(*, cash_per_share=1., share_ratio=1., sell=True):
    """Buy before the record date; accrue, pay later, then sell with dividend tax."""
    calendar=pd.bdate_range('2024-01-02','2024-01-10')
    rule=InstrumentRules('A','stock',100,100,1,.01,.00025,5.,None)
    ledger=Ledger(2000,slippage=0,cash_buffer=0,participation=1,min_trade_value=0)
    actions=pd.DataFrame([dict(action_id='D1',instrument_id='A',record_date='2024-01-03',
        ex_date='2024-01-04',pay_date='2024-01-08',share_date='2024-01-04',
        available_at='2024-01-02',cash_per_share=cash_per_share,share_ratio=share_ratio,
        tax_policy='gross_cn_individual')])
    for session,date in enumerate(calendar):
        price=10. if date<pd.Timestamp('2024-01-04') else (10.-cash_per_share)/(1.+share_ratio)
        ledger.corporate_actions(date,session,actions,calendar)
        bar={'open':price,'close':price,'open_status':'trading',
             'open_status_available_at':str(date.date())+' 09:25:00'}
        ledger.execute(date,session,{'A':bar},{'A':rule},{'A':1e8})
        ledger.close(date,session,{'A':price})
        if date==calendar[0]:ledger.queue_targets({'A':.5},date,'buy',{'A':rule})
        if sell and date==pd.Timestamp('2024-01-08'):ledger.queue_targets({},date,'sell',{'A':rule})
    nav=pd.Series([state['nav'] for state in ledger.snapshots],index=calendar)
    returns=nav.div(nav.shift(1).fillna(ledger.initial_cash))-1.
    return dict(config={'capital':{'init_cash':ledger.initial_cash}},daily_return=returns,
                input_actions=actions,**ledger.export())


class AttributionContracts(unittest.TestCase):
    def test_compact_run_requires_nonzero_action_mapping(self):
        run=distribution_run();run.pop('input_actions')
        with self.assertRaisesRegex(ValueError,'attribution_missing_action_mapping'):
            instrument_returns(run)

    def test_explicit_actions_restore_compact_run_without_mutating_it(self):
        run=distribution_run();actions=run.pop('input_actions')
        contributions=instrument_returns(run,actions=actions)
        pd.testing.assert_series_equal(contributions.sum(axis=1),run['daily_return'],check_names=False,atol=1e-12)
        self.assertNotIn('input_actions',run)

    def test_concentration_report_uses_dataset_mapping_for_compact_runs(self):
        strategy=distribution_run();control=distribution_run()
        actions=strategy.pop('input_actions');control.pop('input_actions')
        ds=SimpleNamespace(actions=actions,instruments=pd.DataFrame())
        with patch('quantlab.report.attribution.asof_records',return_value=pd.DataFrame()), \
             patch('quantlab.report.attribution.exposure_matrix',return_value=pd.DataFrame({'A':[1.]},index=['BANK'])):
            report=concentration_diagnostics(strategy,control,ds)
        self.assertLess(report['reconciliation_max_error'],1e-12)

    def test_concentration_rejects_individually_valid_different_calendars(self):
        strategy=distribution_run();control=distribution_run()
        # The removed first session is all cash. Both accounts still reconcile
        # independently, but their reporting intervals are genuinely different.
        control['account']=control['account'][1:]
        control['daily_return']=control['daily_return'].iloc[1:]
        for run in (strategy,control):
            pd.testing.assert_series_equal(instrument_returns(run).sum(axis=1),run['daily_return'],
                                           check_names=False,atol=1e-12)
        ds=SimpleNamespace(actions=strategy['input_actions'],instruments=pd.DataFrame())
        with patch('quantlab.report.attribution.asof_records',return_value=pd.DataFrame()), \
             patch('quantlab.report.attribution.exposure_matrix',return_value=pd.DataFrame({'A':[1.]},index=['BANK'])):
            with self.assertRaisesRegex(ValueError,'attribution_comparison_calendar_mismatch'):
                concentration_diagnostics(strategy,control,ds)

    def test_dividend_accrual_payment_and_selling_tax_reconcile(self):
        run=distribution_run();contributions=instrument_returns(run)
        self.assertEqual(contributions.loc['2024-01-04'].sum(),0.)
        self.assertEqual(contributions.loc['2024-01-08'].sum(),0.)
        sold=next(fill for fill in run['fills'] if fill['side']=='sell')
        self.assertEqual(sold['dividend_tax'],20.)
        self.assertAlmostEqual(contributions.loc['2024-01-09','A'],-sold['fees']/1995.)
        prior_nav=pd.Series([2000.]+[s['nav'] for s in run['account'][:-1]],index=run['daily_return'].index)
        pnl=contributions.sum(axis=1)*prior_nav
        self.assertAlmostEqual(pnl.sum(),-sum(f['fees'] for f in run['fills']))

    def test_split_only_does_not_create_profit(self):
        run=distribution_run(cash_per_share=0.,share_ratio=1.,sell=False)
        self.assertEqual(run['account'][1]['quantities']['A'],100)
        self.assertEqual(run['account'][2]['quantities']['A'],200)
        contributions=instrument_returns(run)
        self.assertEqual(contributions.loc['2024-01-04'].sum(),0.)
        pd.testing.assert_series_equal(contributions.sum(axis=1),run['daily_return'],check_names=False,atol=1e-12)

    def test_unexplained_nav_or_return_is_rejected(self):
        run=distribution_run();run['account'][2]['nav']+=1.
        with self.assertRaisesRegex(ValueError,'attribution_reconciliation_failed'):
            instrument_returns(run)
        run=distribution_run();run['daily_return'].iloc[2]+=.01
        with self.assertRaisesRegex(ValueError,'attribution_reconciliation_failed'):
            instrument_returns(run)

    def test_missing_account_day_is_not_filled_with_zero_return(self):
        run=distribution_run();run['account'].pop(2)
        with self.assertRaisesRegex(ValueError,'attribution_calendar_mismatch'):
            instrument_returns(run)

    def test_ambiguous_action_identity_is_rejected(self):
        run=distribution_run();different=run['input_actions'].copy();different['instrument_id']='B'
        run['input_actions']=pd.concat([run['input_actions'],different],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'attribution_ambiguous_action_mapping'):
            instrument_returns(run)


if __name__=='__main__':unittest.main()
