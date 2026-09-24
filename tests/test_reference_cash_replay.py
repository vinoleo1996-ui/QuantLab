from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.reference_cash_replay import replay_reference, read_targets, ReferenceLedger
from quantlab.data.contracts import InstrumentRules


def reference_fixture(days=70):
    ds = dataset(days=days,stocks=2)
    mapping = {'S000':'sh.600001','S001':'sh.600002'}
    ds.panels = {mapping[c]:p for c,p in ds.panels.items()}
    ds.instruments.instrument_id = ds.instruments.instrument_id.map(mapping)
    for p in ds.panels.values():
        p['open']=p['close']=p['reference_price']=10.
        p['limit_up']=11.;p['limit_down']=9.;p['amount']=1e8
    return ds


def targets(ds, rows):
    return dict(source_commit='frozen',targets=[dict(signal_date=str(ds.calendar[i].date()),weights=w,
                                                   sell_only=False,max_exposure=.9) for i,w in rows])


class ReferenceCashTests(unittest.TestCase):
    def test_original_reference_has_no_new_drawdown_or_single_stock_stops(self):
        ds = reference_fixture()
        p = ds.panels['sh.600001']
        p.loc[ds.calendar[25]:,'open']=p.loc[ds.calendar[25]:,'close']=4.
        p.loc[ds.calendar[25]:,'reference_price']=4.
        p.loc[ds.calendar[25]:,'limit_up']=4.4;p.loc[ds.calendar[25]:,'limit_down']=3.6
        result = replay_reference(ds,targets(ds,[(21,{'600001.SH':.45})]),start=ds.calendar[0],end=ds.calendar[60])
        self.assertTrue(result['fills'])
        self.assertTrue(all(f['side']=='buy' for f in result['fills']))
        quantities=[s['quantities'].get('sh.600001',0) for s in result['account'] if s['date']>=ds.calendar[22]]
        self.assertEqual(len(set(quantities)),1)
        self.assertGreater(quantities[0],0)
        self.assertFalse(result['executable'])
        self.assertIn('original_no_drawdown',result['protocol']['risk_policy'])

    def test_unsold_reference_positions_cannot_fund_new_positions(self):
        ds = reference_fixture()
        day = ds.calendar[31]
        ds.panels['sh.600001'].loc[day,'open_status']='halted'
        result = replay_reference(ds,targets(ds,[(21,{'600001.SH':.9}),(30,{'600002.SH':.9})]),
                                  start=ds.calendar[0],end=day)
        last = result['account'][-1]
        self.assertGreater(last['quantities']['sh.600001'],0)
        self.assertNotIn('sh.600002',last['quantities'])
        self.assertGreaterEqual(last['cash'],0)
        self.assertTrue(any(o['side']=='sell' and o['status']=='pending' for o in result['orders']))

    def test_last_signal_remains_pending_and_lots_are_actual_hundreds(self):
        ds = reference_fixture()
        result = replay_reference(ds,targets(ds,[(21,{'600001.SH':.45}),(40,{'600002.SH':.45})]),
                                  start=ds.calendar[0],end=ds.calendar[40])
        self.assertTrue(all(f['quantity']%100==0 for f in result['fills'] if f['side']=='buy'))
        self.assertTrue(all(pd.Timestamp(f['date'])>pd.Timestamp(f['signal_date']) for f in result['fills']))
        self.assertTrue(any(o['signal_date']==str(ds.calendar[40].date()) and o['status']=='pending' for o in result['orders']))

    def test_historical_stamp_transfer_and_cost_stress_are_explicit(self):
        rule = InstrumentRules('sh.600001','stock',100,100,1,.01,.0003,5.,.1)
        base, double = ReferenceLedger(cost_mult=1), ReferenceLedger(cost_mult=2)
        self.assertEqual(base.fee('2021-01-04',rule,'sell',100000),(32.,100.))
        self.assertEqual(base.fee('2024-01-04',rule,'sell',100000),(31.,50.))
        self.assertEqual(double.fee('2024-01-04',rule,'sell',100000),(62.,100.))

    def test_accounting_gaps_measure_pre_session_shares_without_adjusting_profit(self):
        ds = reference_fixture()
        schedule = targets(ds,[(21,{'600001.SH':.45})])
        clean = replay_reference(ds,schedule,start=ds.calendar[0],end=ds.calendar[40])
        ds.manifest['accounting_gaps'] = [dict(date=str(ds.calendar[i].date()),instrument_id='sh.600001',
            reason='unexplained_reference_price_change',reference_gap_per_share=.2) for i in (22,25)]
        marked = replay_reference(ds,schedule,start=ds.calendar[0],end=ds.calendar[40])
        exposed = [x for x in marked['data_issues'] if 'reference_gap_exposure' in x]
        self.assertEqual(len(exposed),1)  # No shares before the initial buy on day 22.
        self.assertEqual(exposed[0]['held_shares'],22500)
        self.assertEqual(exposed[0]['reference_gap_exposure'],4500.)
        self.assertIn('not_profit_correction_or_bias_bound',exposed[0]['meaning'])
        pd.testing.assert_series_equal(clean['nav'],marked['nav'])
        self.assertEqual(marked['status'],'conditional_reference_data_gaps')

    def test_new_target_needs_same_day_known_close_even_when_open_is_valid(self):
        ds = reference_fixture()
        ds.panels['sh.600001'].loc[ds.calendar[21],'close']=np.nan
        result = replay_reference(ds,targets(ds,[(21,{'600001.SH':.45})]),
                                  start=ds.calendar[0],end=ds.calendar[40])
        self.assertFalse(result['fills'])
        self.assertFalse(result['orders'])
        self.assertEqual(result['signal_trace'][0]['skipped_targets']['sh.600001'],
                         'missing_reliable_signal_close')

    def test_missing_signal_close_preserves_old_shares_and_disallows_additions(self):
        ds = reference_fixture()
        ds.panels['sh.600001'].loc[ds.calendar[30],'close']=np.nan
        result = replay_reference(ds,targets(ds,[(21,{'600001.SH':.45}),(30,{'600001.SH':.9})]),
                                  start=ds.calendar[0],end=ds.calendar[40])
        self.assertEqual(len(result['fills']),1)
        self.assertEqual(result['account'][-1]['quantities']['sh.600001'],22500)
        self.assertFalse(result['signal_trace'][-1]['orders'])
        self.assertIn('sh.600001',result['signal_trace'][-1]['skipped_targets'])

    def test_schedule_normalization_rejects_duplicate_or_leveraged_targets(self):
        payload=dict(targets=[dict(signal_date='2021-01-04',weights={'600001.SH':.91})])
        with self.assertRaisesRegex(ValueError,'exposure'):
            read_targets(payload)
        payload['targets'][0]['weights']={'600001.SH':.45}
        _,schedule=read_targets(payload)
        self.assertEqual(schedule[pd.Timestamp('2021-01-04')],{'sh.600001':.45})
        payload['targets']*=2
        with self.assertRaisesRegex(ValueError,'unique'):
            read_targets(payload)
        payload['targets_sha256']='invalid'
        with self.assertRaisesRegex(ValueError,'checksum'):
            read_targets(payload)


if __name__=='__main__':unittest.main()
