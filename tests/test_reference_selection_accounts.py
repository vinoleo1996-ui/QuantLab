"""Hand-calculated accounting and target-path regression fixtures."""
import copy
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/reference_selection_accounts.py'

def load_module():
    if not SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location('reference_selection_accounts', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

accounts = load_module()

def state(day, nav, cash, q=0, mark=0., basis=0., receivables=0.):
    return dict(date=pd.Timestamp(day), nav=nav, cash=cash, receivables=receivables,
                quantities={'sz.000001': q} if q else {},
                marks={'sz.000001': mark} if q else {},
                cost_basis={'sz.000001': basis} if q else {},
                weights={'sz.000001': q*mark/nav} if q else {})

def fixture():
    # Buy 10 for 100 + 2 fee. Sell 4 for 52 - 1 fee after a 10 cash dividend.
    # Remaining basis 61.2; end value 78 -> unrealized16.8, disposal10.2,
    # dividend10; net37. Slippage3 is disclosed and already in fill prices.
    return dict(config={'capital': {'init_cash': 1000.}},
        account=[state('2025-01-01',1000,1000),state('2025-01-02',1008,898,10,11,10.2),
                 state('2025-01-03',1037,949,6,13,10.2,10),state('2025-01-06',1037,959,6,13,10.2)],
        daily_return=pd.Series([0.,.008,1037/1008-1,0.],index=pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03','2025-01-06'])),
        fills=[dict(order_id='b',date='2025-01-02',instrument_id='sz.000001',side='buy',quantity=10,notional=100.,fees=2.,slippage_cost=1.,signal_date='2025-01-01',commission=2.,stamp_tax=0.,dividend_tax=0.),
               dict(order_id='s',date='2025-01-03',instrument_id='sz.000001',side='sell',quantity=4,notional=52.,fees=1.,slippage_cost=2.,signal_date='2025-01-02',commission=.5,stamp_tax=.5,dividend_tax=0.)],
        corporate_action_events=[dict(type='corporate_action',date='2025-01-03',action_id='div',cash=10.,bonus_shares=0)],
        orders=[])

class AccountingTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(accounts, 'Task2 accounting parser must be implemented')
        return accounts

    def test_dividend_is_accrued_once_fees_are_in_basis_and_slippage_not_doubled(self):
        m=self.module(); rows,trades,check=m.attribute_account(fixture(),{'div':'sz.000001'})
        self.assertAlmostEqual(rows.pnl.sum(),37.)
        self.assertAlmostEqual(rows.realized_disposal_pnl.sum(),10.2)
        self.assertAlmostEqual(rows.dividend_accrual.sum(),10.)
        self.assertAlmostEqual(rows.unrealized_change.sum(),16.8)
        self.assertAlmostEqual(trades.fees.sum(),3.)
        self.assertAlmostEqual(trades.slippage_cost.sum(),3.)
        self.assertLess(check['max_daily_pnl_error'],1e-8)
        self.assertEqual(set(trades.category),{'new_entry','same_name_reduction'})

    def test_bonus_shares_preserve_basis_and_do_not_invent_realized_profit(self):
        m=self.module(); run=fixture();run['fills']=run['fills'][:1]
        run['account']=run['account'][:2]+[state('2025-01-03',1008,898,20,5.5,5.1)]
        run['daily_return']=pd.Series([0.,.008,0.],index=pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03']))
        run['corporate_action_events']=[dict(type='corporate_action',date='2025-01-03',action_id='bonus',cash=0.,bonus_shares=10)]
        rows,_,_=m.attribute_account(run,{'bonus':'sz.000001'})
        self.assertAlmostEqual(rows.pnl.sum(),8.)
        self.assertAlmostEqual(rows.realized_disposal_pnl.sum(),0.)
        self.assertAlmostEqual(rows.unrealized_change.sum(),8.)

    def test_missing_basis_and_unexplained_basis_movement_fail_closed(self):
        m=self.module();run=fixture();del run['account'][1]['cost_basis']
        with self.assertRaisesRegex(ValueError,'cost_basis'):m.attribute_account(run,{'div':'sz.000001'})
        run=fixture();run['account'][-1]['cost_basis']['sz.000001']=9.
        with self.assertRaisesRegex(ValueError,'basis_without_sale'):m.attribute_account(run,{'div':'sz.000001'})

    def test_missing_action_mapping_and_quantity_mismatch_fail_closed(self):
        m=self.module()
        with self.assertRaisesRegex(ValueError,'action_mapping'):m.attribute_account(fixture(),{})
        run=fixture();run['account'][2]['quantities']['sz.000001']=7
        with self.assertRaisesRegex(ValueError,'quantity_conservation'):m.attribute_account(run,{'div':'sz.000001'})

    def test_reentry_is_distinct_from_new_entry(self):
        m=self.module();run=fixture();run['fills'][1].update(quantity=10,notional=130.)
        run['account'][2]=state('2025-01-03',1037,1027,receivables=10.)
        run['fills'].append(dict(run['fills'][0],order_id='b2',date='2025-01-06',signal_date='2025-01-03'))
        run['account'][3]=state('2025-01-06',1035,935,10,10,10.2)
        run['daily_return'].iloc[-1]=1035/1037-1
        _,trades,_=m.attribute_account(run,{'div':'sz.000001'})
        self.assertEqual(trades.category.tolist(),['new_entry','full_exit','reentry'])

    def test_drawdown_uses_initial_cash_and_unrecovered_is_censored(self):
        m=self.module();states=[state('2024-12-31',100,100),state('2025-01-02',90,90),state('2025-01-03',110,110),state('2025-01-06',88,88)]
        result=m.account_metrics(states,100.,'2025_recent')
        self.assertAlmostEqual(result['max_drawdown'],-.2)
        self.assertEqual(result['sessions'],3)
        self.assertAlmostEqual(result['annual']['2025']['net_profit'],-12.)
        self.assertTrue(result['unrecovered_at_end'])
        self.assertEqual(result['recovery_episodes'][0]['recovery_date'],'2025-01-03')
        self.assertIsNone(result['recovery_episodes'][-1]['recovery_date'])

    def test_outside150_admission_next_target_exit_and_final_censor(self):
        m=self.module();heads={f'{i:06d}.SZ':1. for i in range(1,153)}
        inputs={'heads':{'2025-01-01':heads,'2025-01-03':heads},'industries':{'2025-01-01':{},'2025-01-03':{}}}
        targets=[dict(signal_date='2025-01-02',weights={'000151.SZ':.45,'000001.SZ':.45}),dict(signal_date='2025-01-03',weights={'000152.SZ':.45,'000001.SZ':.45})]
        decisions,admissions=m.target_paths(targets,inputs)
        a=next(x for x in admissions if x['instrument_id']=='sz.000151')
        b=next(x for x in admissions if x['instrument_id']=='sz.000152')
        self.assertEqual(a['original_rank'],151)
        self.assertTrue(a['exited_next_target']);self.assertTrue(a['next_eligible'])
        self.assertFalse(a['next_top150']);self.assertFalse(a['end_censored'])
        self.assertTrue(b['end_censored']);self.assertIsNone(b['next_eligible'])
        self.assertEqual(decisions[0]['information_date'],'2025-01-01')
        self.assertEqual(decisions[1]['roles']['sz.000001'],'retained')

    def test_role_transfer_does_not_count_old_unrealized_profit_as_new_gain(self):
        m=self.module();rows,trades,_=m.attribute_account(fixture(),{'div':'sz.000001'})
        inputs={'heads':{'2025-01-01':{'000001.SZ':1.},'2025-01-02':{'000001.SZ':1.}},'industries':{'2025-01-01':{'000001.SZ':'bank'},'2025-01-02':{'000001.SZ':'bank'}}}
        targets=[dict(signal_date=x,weights={'000001.SZ':.9}) for x in ['2025-01-01','2025-01-02']]
        decisions,_=m.target_paths(targets,inputs)
        enriched,end,transfers=m.attach_roles(rows,fixture()['account'],decisions)
        self.assertAlmostEqual(enriched.groupby('role').pnl.sum()['new_admission'],8.)
        self.assertAlmostEqual(enriched.groupby('role').pnl.sum()['retained'],29.)
        self.assertAlmostEqual(transfers.groupby('role').unrealized_transfer.sum()['retained'],8.)
        self.assertAlmostEqual(end.unrealized_pnl.sum(),16.8)

if __name__=='__main__':unittest.main()
