"""Stress inputs must remain valid exchange prices and exercise real constraints."""
from unittest.mock import patch
import unittest
import pandas as pd

from fixtures_v2 import dataset
from quantlab.backtest.ledger import Ledger, Lot, trade_permission
from quantlab.config import Config
from quantlab.data.contracts import InstrumentRules
from quantlab.data.exchange_rules import limit_prices
from quantlab.data.market import panels_for_research
from quantlab.policy_comparison import _stress_dataset, run_stress_suite


def stress_fixture():
    ds=dataset(days=80,stocks=1);code='sz.000001'
    ds.panels={code:ds.panels['S000']};ds.instruments['instrument_id']=code
    ds.manifest['feature_price_basis']='historical_adjustment_snapshot'
    p=ds.panels[code]
    p['reference_price']=p['close']=20.05
    p['open']=20.05*.99;p['high']=20.05;p['low']=19.84;p['feature_close']=100.
    p['limit_up'],p['limit_down']=limit_prices(20.05,.1,.01)
    p['open_status']='derived';p['open_status_basis']='daily_open_assumption';p['rule_version']='cn_daily_v1'
    p['is_st_at_open']=False;p['listing_session_lower_bound']=60
    sessions=list(ds.calendar[70:75])
    base=dict(account=[dict(date=ds.calendar[69],quantities={code:100})])
    return ds,code,sessions,base,InstrumentRules.from_record(ds.instruments.iloc[0])


class StressIntegrityTests(unittest.TestCase):
    def test_gap_respects_real_limit_and_ohlc_range(self):
        ds,code,sessions,base,rule=stress_fixture()
        original=ds.panels[code].copy()
        changed,_=_stress_dataset(ds,base,sessions,'opening_gap_10pct')
        row=changed.panels[code].loc[sessions[0]]
        self.assertEqual(changed.validate('free_research'),[])
        self.assertGreaterEqual(row.open,row.limit_down)
        self.assertLessEqual(row.open,row.limit_up)
        self.assertLessEqual(row.low,min(row.open,row.close))
        self.assertGreaterEqual(row.high,max(row.open,row.close))
        self.assertEqual(trade_permission(row.to_dict(),rule,'sell',sessions[0],allow_open_assumption=True),
                         (False,'limit_down'))
        self.assertGreater(changed.manifest['stress_variant']['clipped_price_cells'],0)
        pd.testing.assert_frame_equal(ds.panels[code],original)

    def test_correlated_fall_rebuilds_tick_limits_and_also_changes_feature_prices(self):
        ds,code,sessions,base,rule=stress_fixture()
        changed,_=_stress_dataset(ds,base,sessions,'correlated_fall_5')
        original_features=panels_for_research(ds)[0]
        stressed_features=panels_for_research(changed)[0]
        self.assertFalse(original_features.equals(stressed_features))
        for date in ds.panels[code].index[70:]:
            row=changed.panels[code].loc[date]
            upper,lower=limit_prices(row.reference_price,.1,.01)
            self.assertAlmostEqual(row.limit_up,upper)
            self.assertAlmostEqual(row.limit_down,lower)
            self.assertTrue(trade_permission(row.to_dict(),rule,'sell',date,allow_open_assumption=True)[0])
            ratio=row.close/ds.panels[code].loc[date,'close']
            self.assertAlmostEqual(stressed_features.loc[date,code]/original_features.loc[date,code],ratio)
        self.assertLess(changed.panels[code].loc[sessions[-1],'close']/20.05,.91)

    def test_gap_does_not_invent_a_quote_when_the_original_open_is_missing(self):
        ds,code,sessions,base,rule=stress_fixture()
        ds.panels[code].loc[sessions[0],'open']=float('nan')
        changed,_=_stress_dataset(ds,base,sessions,'opening_gap_10pct')
        self.assertTrue(pd.isna(changed.panels[code].loc[sessions[0],'open']))
        self.assertEqual(changed.manifest['stress_variant']['price_impacts'][code]['status'],'not_exercised')

    def test_sell_lock_keeps_real_cash_available_but_does_not_invent_sale_proceeds(self):
        ds=dataset(days=80,stocks=2);date=ds.calendar[70]
        rules={row.instrument_id:InstrumentRules.from_record(row) for _,row in ds.instruments.iterrows()}
        ledger=Ledger(cash=10000.,slippage=0.,cash_buffer=0.)
        ledger.lots['S000']=[Lot(9000,0,10.,0,str(ds.calendar[0].date()))]
        ledger.marks={'S000':10.,'S001':10.}
        ledger.queue_targets({'S001':.2},ds.calendar[69],'risk_exit',rules)
        bars={c:dict(p.loc[date],open=10.,reference_price=10.,limit_up=11.,limit_down=9.) for c,p in ds.panels.items()}
        bars['S000']['stress_sell_blocked']=True
        ledger.execute(date,70,bars,rules,{'S000':1e9,'S001':1e9})
        self.assertEqual(ledger.quantity('S000'),9000)
        self.assertEqual(ledger.quantity('S001'),900)
        self.assertEqual(ledger.cash,995.)
        events=[x for x in ledger.events if x.get('type')=='execution_rejected' and x.get('reason')=='stress_sell_lock']
        self.assertTrue(events)
        self.assertEqual(events[0]['remaining'],9000)

    def test_lock_hit_uses_actual_rejection_events_and_keeps_buy_side_separate(self):
        ds=dataset(days=80,stocks=1);dates=ds.calendar[:40]
        base=dict(daily_return=pd.Series(0.,index=dates),
            fills=[dict(side='sell',date=str(dates[10].date()),order_id='BASE_DIFFERENT')],
            account=[dict(date=dates[9],quantities={'S000':100})])
        captured=[]
        def replay(changed,cfg,spec,**kwargs):
            captured.append(kwargs)
            full='blocked_sessions' in kwargs
            window=kwargs.get('blocked_sessions',kwargs.get('blocked_sell_sessions'))
            day=window[0];signal=day-pd.Timedelta(days=1)
            event=dict(type='execution_rejected',date=str(day.date()),order_id='STRESS_ID',
                instrument_id='S000',side='sell',remaining=100,
                reason='unknown_or_halted_open' if full else 'stress_sell_lock')
            return dict(run_id='synthetic',orders=[dict(order_id='STRESS_ID',instrument_id='S000',side='sell',quantity=100,
                signal_date=str(signal.date()),execute_not_before=str(day.date()))],corporate_action_events=[event],
                fills=[] if full else [dict(side='buy',date=str(day.date()))])
        with (patch('quantlab.policy_comparison.STRESS_NAMES',('fixed_lock_5','adverse_exit_lock_5','full_halt_5')),
              patch('quantlab.policy_comparison.summarize_run',return_value=dict(accounting={'passed':True},conditional_metrics={'max_drawdown':-.01}))):
            result=run_stress_suite(ds,Config(),dict(risk_policy='P2'),base,replay)
        self.assertEqual(len(captured),3)
        self.assertIn('blocked_sell_sessions',captured[0])
        self.assertIn('blocked_sessions',captured[2])
        for value in result.values():
            self.assertTrue(value['constraints_passed'])
            self.assertEqual(value['requested_exit_orders'],1)


if __name__=='__main__':unittest.main()
