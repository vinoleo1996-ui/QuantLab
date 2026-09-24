import unittest
import pandas as pd
import numpy as np
from fixtures_v2 import dataset
from quantlab.strategy.candidates import financial_entry_evidence, candidate_decision, industry_rotation
from quantlab.report.advice import rotation_advice


class CandidateTests(unittest.TestCase):
    def test_publication_time_uses_the_shared_china_local_contract(self):
        ds=dataset(100,3);d=ds.calendar[90]
        for p in ds.panels.values():p['pb_mrq']=2.
        ds.financials=pd.DataFrame([dict(instrument_id=c,metric='net_profit',value=10.,
            report_period='2022-12-31',announced_on=d-pd.Timedelta(days=1),
            available_at=at,revision='1') for c,at in zip(['S000','S001','S002'],
            [str(d.date())+' 17:00:00',str(d.date())+' 09:00:00+00:00',str(d.date())+' 10:01:00+00:00'])])
        evidence=financial_entry_evidence(ds,d,list(ds.panels))
        self.assertTrue(evidence['S000']['passes'])
        self.assertTrue(evidence['S001']['passes'])
        self.assertFalse(evidence['S002']['covered'])

    def test_financial_coverage_is_not_quality_and_late_release_waits(self):
        ds=dataset(100,3);d=ds.calendar[90]
        for p in ds.panels.values(): p['pb_mrq']=2.
        ds.financials=pd.DataFrame([
            dict(instrument_id='S000',metric='net_profit',value=-10.,report_period='2022-12-31',announced_on=d-pd.Timedelta(days=1),available_at=None,revision='1'),
            dict(instrument_id='S001',metric='net_profit',value=10.,report_period='2022-12-31',announced_on=d,available_at=None,revision='1')])
        e=financial_entry_evidence(ds,d,['S000','S001','S002'])
        self.assertTrue(e['S000']['covered']);self.assertFalse(e['S000']['passes'])
        self.assertFalse(e['S001']['covered']);self.assertFalse(e['S002']['covered'])
        score=pd.Series({'S000':3.,'S001':2.,'S002':1.})
        _,coverage,_=candidate_decision(ds,{},d,score,'quality_coverage')
        _,quality,_=candidate_decision(ds,{},d,score,'quality_pb')
        self.assertEqual(coverage,{'S000'});self.assertEqual(quality,set())

    def test_future_industry_change_cannot_rewrite_signal(self):
        ds=dataset(140,20);d=ds.calendar[110]
        close=pd.DataFrame({c:p.close for c,p in ds.panels.items()})
        before,_=industry_rotation(ds,close,d)
        future=ds.instruments.copy();future['effective_at']=ds.calendar[120];future['available_at']=ds.calendar[120];future['sector']='future_sector'
        ds.instruments=pd.concat([ds.instruments,future],ignore_index=True)
        after,_=industry_rotation(ds,close,d)
        pd.testing.assert_series_equal(before,after)

    def test_single_industry_is_no_adjustment(self):
        ds=dataset(100,6);ds.instruments['sector']='only'
        close=pd.DataFrame({c:p.close for c,p in ds.panels.items()})
        ranks,reason=industry_rotation(ds,close,ds.calendar[90])
        self.assertTrue(ranks.empty)
        self.assertEqual(reason['reason'],'insufficient_comparable_industries')

    def test_advice_ranks_values_and_preserves_actual_order(self):
        out=rotation_advice(dict(account=[dict(quantities={'a':100},weights={'a':.05},cash=1000.,marks={'a':10.})],
            weekly_observations=[dict(date='2026-01-01',scores={'a':1.,'z':4.})],
            orders=[dict(instrument_id='b',side='buy',status='pending',remaining=100,quantity=100,estimated_notional=2000.,reason='legacy_repaired')]))
        self.assertEqual(out['holdings'][0]['current_rank'],2)
        self.assertEqual(out['orders'][0]['remaining_notional'],2000.)
        self.assertFalse(out['executable'])


if __name__=='__main__': unittest.main()
