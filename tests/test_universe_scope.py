import unittest
from types import SimpleNamespace
from copy import deepcopy
import pandas as pd
import numpy as np

from fixtures_v2 import dataset
from quantlab.config import CONFIG
from quantlab.data.market import universe_at


class UniverseScope(unittest.TestCase):
    def test_listing_age_counts_exchange_sessions_with_inclusive_date_boundary(self):
        ds=dataset(days=90,stocks=6);cfg=deepcopy(CONFIG)
        cfg.research.universe_mode='pit_liquidity';cfg.universe.a_min_listing_days=20
        boundary=ds.calendar[20]
        ds.instruments['listing_date']=[str(boundary.date()),ds.calendar[21],
            str((boundary-pd.Timedelta(days=2)).date()),str(ds.calendar[50].date()),
            str(boundary+pd.Timedelta(hours=12)),boundary.strftime('%Y%m%d')]
        amount=pd.DataFrame(1e9,index=ds.calendar,columns=ds.panels)
        selected,_=universe_at(ds,ds.calendar[39],amount,cfg)
        self.assertEqual(selected,['S000','S002','S005'])

    def test_bulk_daily_state_preserves_unknown_and_does_not_rewrite_ineligible_rows(self):
        ds=dataset(days=80,stocks=7);cfg=deepcopy(CONFIG)
        cfg.research.universe_mode='pit_liquidity'
        ds.instruments['state_basis']='daily_observation_required'
        ds.instruments.loc[4,'active']=False
        ds.instruments.loc[4,'is_st']=True
        ds.instruments.loc[4,'listing_date']='invalid_but_inactive'
        ds.instruments.loc[5,'quality_status']='bad'
        ds.instruments.loc[6,'state_basis']='record'
        for i,panel in enumerate(ds.panels.values()):panel['is_st_at_open']=float(i in (1,3,5,6))
        ds.panels['S002'].loc[ds.calendar[70],'is_st_at_open']=np.nan
        amount=pd.DataFrame(1e9,index=ds.calendar,columns=ds.panels)
        del ds.panels['S003']
        selected,records=universe_at(ds,ds.calendar[70],amount,cfg)
        self.assertEqual(selected,['S000','S006'])
        self.assertFalse(records.loc['S000','is_st'])
        self.assertTrue(records.loc['S001','is_st'])
        self.assertTrue(pd.isna(records.loc['S002','is_st']))
        self.assertFalse(records.loc['S003','is_st'])
        self.assertTrue(records.loc['S004','is_st'])
        self.assertFalse(records.loc['S005','is_st'])
        self.assertFalse(records.loc['S006','is_st'])
        self.assertEqual(records.is_st.dtype,object)

    def test_collection_does_not_truncate_or_drop_growth_boards(self):
        from scripts.collect_free_research import select_member_codes
        base=[f'{exchange}.{prefix}{i:03d}' for exchange,prefix in
              [('sh','600'),('sh','601'),('sh','603'),('sz','000'),('sz','001'),('sz','002')]
              for i in range(1000)][:5201]
        members=pd.DataFrame({'code':base+
                                      ['sz.300308','301001.SZ','sh.688981','bj.920001','sh.000300']})
        codes=select_member_codes(members)
        self.assertEqual(len(codes),5205)
        for c in ('sz.300308','sz.301001','sh.688981','bj.920001'): self.assertIn(c,codes)
        self.assertNotIn('sh.000300',codes)
        self.assertEqual(len(select_member_codes(members,120)),120)
        self.assertEqual(select_member_codes(members,120),select_member_codes(members.iloc[::-1],120))

    def test_comparison_has_no_120_member_requirement(self):
        from quantlab.legacy_comparison import comparison_protocol
        ds = SimpleNamespace(panels={str(i): None for i in range(5201)}, manifest={
            'research_scope': {'scope': 'provided_historical_members_no_sampling'}})
        protocol = comparison_protocol(ds)
        self.assertEqual(protocol['universe']['securities_with_panels'], 5201)
        self.assertEqual(protocol['universe']['market_coverage'], 'not_established')

    def test_old_fixed_sample_requires_explicit_diagnostic_choice(self):
        from quantlab.legacy_comparison import comparison_protocol
        ds = SimpleNamespace(panels={'A': None}, manifest={
            'research_scope': {'scope': 'fixed_initial_mainboard_cohort; not all-market'}})
        with self.assertRaisesRegex(ValueError, 'diagnostic_sample'):
            comparison_protocol(ds)
        self.assertTrue(comparison_protocol(ds, diagnostic_sample=True)['diagnostic_sample'])

    def test_new_listings_enter_and_later_delistings_do_not_erase_history(self):
        ds = dataset(days=90, stocks=5)
        cfg = deepcopy(CONFIG)
        cfg.research.universe_mode = 'pit_liquidity'
        cfg.universe.a_min_listing_days = 20
        amount = pd.DataFrame(1e9, index=ds.calendar, columns=ds.panels)
        new_date, delist_date = ds.calendar[40], ds.calendar[75]
        ds.instruments.loc[ds.instruments.instrument_id=='S001', 'listing_date'] = str(new_date.date())
        gone = ds.instruments[ds.instruments.instrument_id=='S002'].iloc[0].copy()
        gone['active'] = False
        gone['effective_at'] = str(delist_date.date())
        gone['available_at'] = str(delist_date.date())+' 18:00:00'
        ds.instruments = pd.concat([ds.instruments, pd.DataFrame([gone])], ignore_index=True)
        before, _ = universe_at(ds, ds.calendar[39], amount, cfg)
        after, _ = universe_at(ds, ds.calendar[65], amount, cfg)
        last, _ = universe_at(ds, ds.calendar[80], amount, cfg)
        self.assertNotIn('S001', before)
        self.assertIn('S001', after)
        self.assertIn('S002', before)
        self.assertNotIn('S002', last)
        amount.loc[ds.calendar[60:81], 'S003'] = 0
        self.assertNotIn('S003', universe_at(ds, ds.calendar[80], amount, cfg)[0])


if __name__ == '__main__': unittest.main()
