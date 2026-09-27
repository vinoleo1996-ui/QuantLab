"""Perturb real reference execution without changing decisions or cash rules."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import expert_selection_stress as s


class ExecutionStressTests(unittest.TestCase):
    def fixture(self,amount=1e8):
        from tests.fixtures_v2 import dataset
        ds=dataset(65,2)
        ds.panels={c.lower():p.copy() for c,p in ds.panels.items()}
        ds.instruments['instrument_id']=ds.instruments.instrument_id.str.lower()
        for panel in ds.panels.values():
            panel.loc[:,'open']=20.;panel.loc[:,'close']=20.
            panel.loc[:,'reference_price']=20.;panel.loc[:,'limit_up']=22.;panel.loc[:,'limit_down']=18.
            panel.loc[:,'amount']=amount
        targets={'targets':[dict(signal_date=str(ds.calendar[25].date()),weights={'s000':.9}),
                            dict(signal_date=str(ds.calendar[35].date()),weights={'s001':.9})]}
        return ds,targets

    def replay(self,ds,targets,case,lock=None,end=49):
        return s.simulate(ds,targets,start=str(ds.calendar[25].date()),end=str(ds.calendar[end].date()),
            case=case,lock_plan=lock,protocol_hash='test-stress',arm='S')

    def test_neutral_wrapper_preserves_account_and_fills_and_restores_class(self):
        ds,targets=self.fixture()
        original=s.replay.ProfileLedger
        baseline=s.replay.replay_reference(ds,targets,start=ds.calendar[25],end=ds.calendar[49],profile=s.replay.ReplayProfile.common(50))
        neutral=self.replay(ds,targets,'neutral')
        self.assertEqual(neutral['account'],baseline['account'])
        self.assertEqual(neutral['fills'],baseline['fills'])
        self.assertIs(s.replay.ProfileLedger,original)
        self.assertNotEqual(neutral['run_id'],baseline['run_id'])

    def test_lock_hits_real_exit_five_days_and_cannot_fund_replacement(self):
        ds,targets=self.fixture()
        base=self.replay(ds,targets,'neutral')
        plan=s.choose_lock(base,targets,ds.calendar)
        self.assertEqual(plan['instrument_id'],'s000')
        self.assertEqual(plan['start_session'],36)
        locked=self.replay(ds,targets,'lock5',plan)
        prohibited=set(plan['lock_dates'])
        self.assertFalse(any(f['side']=='sell' and f['instrument_id']=='s000' and f['date'] in prohibited for f in locked['fills']))
        self.assertFalse(any(f['side']=='buy' and f['instrument_id']=='s001' and f['date'] in prohibited for f in locked['fills']))
        self.assertEqual(locked['stress_evidence']['lock']['rejected_sessions'],5)
        self.assertTrue(locked['stress_evidence']['exercised'])
        self.assertIsNotNone(locked['stress_evidence']['lock']['completed_exit_date'])
        self.assertTrue(all(a['cash']>=0 for a in locked['account']))
        self.assertTrue(all(a['marks'].get('s000',20.)==20. for a in locked['account']))

    def test_extra_delay_keeps_original_quantity_and_original_signal(self):
        ds,targets=self.fixture()
        # A later execution price differs from the original sizing price.
        # Delay may change fills, but must never recompute the requested shares.
        ds.panels['s000'].loc[ds.calendar[27],'open']=21.
        base=self.replay(ds,targets,'neutral')
        delayed=self.replay(ds,targets,'delay1')
        self.assertEqual(base['orders'][0]['quantity'],delayed['orders'][0]['quantity'])
        self.assertEqual(base['orders'][0]['signal_date'],delayed['orders'][0]['signal_date'])
        self.assertEqual(pd.Timestamp(delayed['fills'][0]['date']),ds.calendar[27])
        self.assertGreater(delayed['stress_evidence']['delay']['withheld_attempts'],0)
        self.assertTrue(all(ds.calendar.get_loc(pd.Timestamp(f['date'])) >= ds.calendar.get_loc(pd.Timestamp(f['signal_date']))+2 for f in delayed['fills']))

    def test_prior_adv_reduction_caps_real_fills_without_changing_targets(self):
        ds,targets=self.fixture(amount=10e6)
        normal=self.replay(ds,targets,'neutral')
        reduced=self.replay(ds,targets,'adv03')
        self.assertLess(reduced['fills'][0]['quantity'],normal['fills'][0]['quantity'])
        self.assertEqual(normal['orders'][0]['quantity'],reduced['orders'][0]['quantity'])
        self.assertTrue(reduced['stress_evidence']['exercised'])
        self.assertGreater(reduced['stress_evidence']['liquidity']['bound_order_sessions'],0)
        self.assertTrue(all(f['notional']<=10e6*.3*.01+.01 for f in reduced['fills']))
        unconstrained=self.replay(*self.fixture(amount=1e12),'adv03')
        self.assertFalse(unconstrained['stress_evidence']['exercised'])
        self.assertEqual(unconstrained['stress_evidence']['status'],'not_exercised')

    def test_cost2_is_existing_multiplier_and_last_day_cannot_fill(self):
        ds,targets=self.fixture()
        actual=self.replay(ds,targets,'cost2')
        expected=s.replay.replay_reference(ds,targets,start=ds.calendar[25],end=ds.calendar[49],
            profile=s.replay.ReplayProfile.common(50),cost_mult=2.)
        self.assertEqual(actual['account'],expected['account'])
        for a,b in zip(actual['fills'],expected['fills']):
            self.assertEqual(a['price'],b['price']);self.assertEqual(a['fees'],b['fees'])
        last={'targets':[dict(signal_date=str(ds.calendar[49].date()),weights={'s000':.9})]}
        self.assertEqual(self.replay(ds,last,'delay1')['fills'],[])

    def test_no_actual_exit_is_explicitly_unexercised_and_inputs_unchanged(self):
        ds,targets=self.fixture()
        targets['targets']=targets['targets'][:1]
        baseline=self.replay(ds,targets,'neutral')
        self.assertIsNone(s.choose_lock(baseline,targets,ds.calendar))
        before=ds.panels['s000'].copy(deep=True)
        result=self.replay(ds,targets,'lock5')
        self.assertFalse(result['stress_evidence']['exercised'])
        pd.testing.assert_frame_equal(before,ds.panels['s000'])

    def test_exception_restores_frozen_replay_class(self):
        ds,targets=self.fixture()
        original=s.replay.ProfileLedger
        with patch.object(s.replay,'replay_reference',side_effect=RuntimeError('fixture interrupted')):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                self.replay(ds,targets,'delay1')
        self.assertIs(s.replay.ProfileLedger,original)

    def test_adv_odd_lot_and_t1_counterexample_is_not_capacity_hit(self):
        from quantlab.backtest.ledger import Lot, Order
        from quantlab.data.contracts import InstrumentRules
        ds,_=self.fixture()
        calendar=ds.calendar
        raw=dict(delay_orders=[],delay_withheld=[],lock_sessions=[],lock_attempts=[],liquidity_attempts=[])
        rule=InstrumentRules.from_record(ds.instruments.iloc[0])
        with s.ledger_override(calendar,'adv03',None,raw):
            ledger=s.replay.ProfileLedger(600000.,slippage=0.,participation=.01,
                profile=s.replay.ReplayProfile.common(50))
            ledger.lots={'s000':[Lot(200,0,20.,0),Lot(50,2,20.,0)]}
            ledger.marks={'s000':20.}
            ledger.orders=[Order('exit','s000','sell',250,250,str(calendar[0].date()),'reference_rebalance')]
            ledger.execute(calendar[1],1,{'s000':ds.panels['s000'].loc[calendar[1]].to_dict()},
                           {'s000':rule},{'s000':1_800_000.})
        row=raw['liquidity_attempts'][0]
        self.assertEqual(ledger.fills[0]['quantity'],200)
        self.assertEqual(ledger.quantity('s000'),50)
        self.assertFalse(row['capacity_exhausted'])
        self.assertEqual(row['available_after'],0)
        self.assertEqual(row['extra_executable_original_capacity'],0)
        # The50-share entire remainder is legal once available; a full lot
        # must not be required when checking its remaining1400yuan capacity.
        ledger.lots['s000'][1].available_session=1
        quantity=s.executable_quantity(ledger,ledger.orders[0],calendar[1],1,
            ds.panels['s000'].loc[calendar[1]].to_dict(),rule,5400.,4000.)
        self.assertEqual(quantity,50)

    def test_native_five_day_halt_is_no_incremental_lock_shock(self):
        ds,targets=self.fixture()
        ds.panels['s000'].loc[ds.calendar[36:41],'open_status']='halted'
        baseline=self.replay(ds,targets,'neutral')
        plan=s.choose_lock(baseline,targets,ds.calendar)
        self.assertEqual(plan['start_session'],36)
        stressed=self.replay(ds,targets,'lock5',plan)
        self.assertEqual(stressed['account'],baseline['account'])
        self.assertEqual(stressed['stress_evidence']['lock']['rejected_sessions'],5)
        self.assertFalse(stressed['stress_evidence']['exercised'])
        self.assertEqual(stressed['stress_evidence']['lock']['incremental_blocked_baseline_quantity'],0)

    def test_simulation_checkpoint_recovers_report_failure_and_rejects_changed_bytes(self):
        ds,targets=self.fixture()
        result=self.replay(ds,targets,'delay1')
        with tempfile.TemporaryDirectory() as tmp,patch.object(s.study,'progress'):
            root=Path(tmp);target=root/'targets.json';target.write_text(json.dumps(targets))
            plan=dict(target_path=str(target),target_sha256=s.shared.sha256(target),case='delay1',arm='S',
                      start=str(ds.calendar[25].date()),end=str(ds.calendar[49].date()))
            identity=s.checkpoint_identity('test-stress',plan)
            result['stress_checkpoint_identity']=identity
            folder=root/'valid'
            s.save_simulation(folder,result,identity,plan)
            # No report/complete.json: precisely a report failure checkpoint.
            recovered=s.recover_partial(folder,identity,plan)
            self.assertEqual(recovered['account'],result['account'])
            damaged=root/'changed';s.save_simulation(damaged,result,identity,plan)
            with (damaged/'result.pkl').open('ab') as handle:handle.write(b'changed_after_checkpoint')
            self.assertIsNone(s.recover_partial(damaged,identity,plan))
            unbound=root/'no_checkpoint';s.save_simulation(unbound,result,identity,plan)
            (unbound/'simulation_complete.json').unlink()
            self.assertIsNone(s.recover_partial(unbound,identity,plan))


if __name__=='__main__':unittest.main()
