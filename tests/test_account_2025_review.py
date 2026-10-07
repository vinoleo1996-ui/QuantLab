"""New-account boundaries and first-day accounting, independent of old results."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import account_2025_review as review


def sample_result():
    import pandas as pd
    observations=[('2024-12-31',600000),('2025-01-02',594000),
        ('2025-12-31',660000),('2026-01-05',627000),('2026-09-22',633600)]
    snapshots=[dict(date=d,nav=n,cash=n,quantities={},weights={},marks={}) for d,n in observations]
    return dict(account=snapshots,config={'capital':{'init_cash':600000}},
        nav=pd.Series([n/600000 for _,n in observations],index=pd.to_datetime([d for d,_ in observations])),
        fills=[dict(date='2025-01-02',signal_date='2024-12-31',reason='legacy_repaired',fees=15.,slippage_cost=40.,notional=1000.)],
        risk_history=[],status='unverified_data_gaps')


class NewAccountTests(unittest.TestCase):
    def test_first_trading_day_loss_is_included_and_year_two_uses_prior_year_boundary(self):
        result=sample_result()
        summary,daily=review.summarize(result)
        self.assertAlmostEqual(daily.iloc[0],-.01)
        self.assertAlmostEqual(summary['total_net_return'],.056)
        self.assertAlmostEqual(summary['net_profit'],33600)
        self.assertAlmostEqual(summary['stages']['2025']['net_return'],.1)
        self.assertAlmostEqual(summary['stages']['2026']['net_return'],-.04)
        self.assertEqual(summary['stages']['2026']['initial_boundary_nav'],660000)
        self.assertAlmostEqual(summary['max_drawdown'],-.05)
        self.assertEqual(summary['fees'],15)
        self.assertEqual(summary['slippage'],40)
        self.assertFalse(summary['promotion_approved'])

    def test_old_positions_prior_profit_claims_or_fills_cannot_be_discarded(self):
        for change in [dict(nav=600001),dict(quantities={'S':100}),dict(receivables=1),dict(payables=1)]:
            result=sample_result()
            result['account'][0].update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'economic_activity'):
                review.reporting_account(result)
        result=sample_result()
        result['fills'][0]['date']='2024-12-31'
        with self.assertRaisesRegex(ValueError,'economic_activity'):
            review.reporting_account(result)

    def test_more_than_one_prior_account_session_is_rejected(self):
        result=sample_result()
        earlier=deepcopy(result['account'][0]);earlier['date']='2024-12-30'
        result['account'].insert(0,earlier)
        with self.assertRaisesRegex(ValueError,'exactly_one'):
            review.reporting_account(result)

    def test_first_session_requires_an_actual_prior_month_end(self):
        import pandas as pd
        cal=pd.to_datetime(['2024-12-30','2024-12-31','2025-01-02','2025-01-03','2025-02-03'])
        prep,dates=review.account_window(cal,'2025-01-02','2025-01-03')
        self.assertEqual(prep,'2024-12-31')
        self.assertEqual(len(dates),2)
        with self.assertRaisesRegex(ValueError,'prior_month_end'):
            review.account_window(cal,'2025-01-03','2025-02-03')

    def test_extension_date_masks_decode_categories_without_changing_source_values(self):
        import importlib.util
        import pandas as pd
        path=ROOT/'output/account_2025_20260927/extension/build_extension.py'
        spec=importlib.util.spec_from_file_location('account_extension_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        original=pd.Series(pd.Categorical(['2024-12-31','2025-01-02']))
        mask=module.dates(original)<=pd.Timestamp('2025-01-01')
        self.assertEqual(mask.tolist(),[True,False])
        self.assertIsInstance(original.dtype,pd.CategoricalDtype)
        self.assertEqual(original.tolist(),['2024-12-31','2025-01-02'])

    def test_new_price_panel_does_not_duplicate_existing_support_history(self):
        import importlib.util
        import pandas as pd
        path=ROOT/'output/account_2025_20260927/extension/build_extension.py'
        spec=importlib.util.spec_from_file_location('account_extension_history_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old=pd.DataFrame([dict(instrument_id='A',effective_at='2020-01-01',sector='a'),
                          dict(instrument_id='B',effective_at='2020-01-01',sector='b')])
        new=pd.concat([old,pd.DataFrame([dict(instrument_id='C',effective_at='2026-06-01',sector='c'),
                          dict(instrument_id='B',effective_at='2026-07-01',sector='b2')])],ignore_index=True)
        combined,details=module.merge_support_history('industry_records',old,new,{'B','C'},'effective_at')
        self.assertEqual(len(combined),4)
        self.assertEqual(details['new_rows'],2)
        pd.testing.assert_frame_equal(combined.iloc[:2].reset_index(drop=True),old)
        self.assertFalse(combined.duplicated(['instrument_id','effective_at','sector']).any())
        changed=new.copy();changed.loc[1,'sector']='revised'
        with self.assertRaises(AssertionError):
            module.merge_support_history('industry_records',old,changed,{'B','C'},'effective_at')

    def test_real_frozen_engine_has_full_warmup_and_no_inherited_account(self):
        program=r'''
import sys
sys.path.insert(0,'scripts')
import account_2025_review as r
r.shared.load_frozen()
sys.path.insert(0,str(r.shared.FROZEN/'tests'))
from fixtures_v2 import dataset
import pandas as pd
ds=dataset(days=540,stocks=20)
ds.manifest['source']='synthetic_new_account_contract'
ds.instruments['sector']=ds.instruments.instrument_id
ds.financials=pd.DataFrame([dict(instrument_id=code,available_at='2022-01-01',effective_at='2022-01-01',
    source='TEST_FIXTURE',revision='1',quality_status='ok',metric='net_profit',value=1e8,
    announced_on='2022-01-01',report_period='2021-12-31') for code in ds.panels])
for panel in ds.panels.values():panel['pb_mrq']=1.
result=r.replay_account(ds,start='2024-12-02',end='2024-12-10')
assert len(result['account'])==8
assert str(pd.Timestamp(result['account'][0]['date']).date())=='2024-11-29'
assert result['account'][0]['cash']==600000 and not result['account'][0]['quantities']
assert result['account'][0]['nav']==600000
assert result['risk_history'][0]['level']==0 and result['risk_history'][0]['drawdown']==0
assert result['ic_matrix'].index.min()<pd.Timestamp('2024-01-01')
assert len(result['fills'])>0,result['signal_trace']
assert min(f['date'] for f in result['fills'])=='2024-12-02'
assert min(f['signal_date'] for f in result['fills'])=='2024-11-29'
r.reporting_account(result,start='2024-12-02',end='2024-12-10',prep='2024-11-29')
assert result['config']['capital']['rebalance_rule']=='monthly'
print('frozen fresh-account boundary passed')
'''
        completed=subprocess.run([sys.executable,'-c',program],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stdout+'\n'+completed.stderr)


if __name__=='__main__':unittest.main()
