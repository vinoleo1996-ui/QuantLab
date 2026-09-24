import unittest
from copy import deepcopy
import pandas as pd
from fixtures_v2 import dataset
from quantlab.config import CONFIG


class LayeredResearch(unittest.TestCase):
    def test_missing_industry_only_disables_industry_module(self):
        from quantlab.data.qualification import research_assessment, qualification
        ds=dataset();ds.manifest['source']='free_sample';ds.instruments['sector']='UNKNOWN'
        result=research_assessment(ds)
        self.assertEqual(result['status'],'research_ready')
        self.assertFalse(result['modules']['industry'])
        self.assertTrue(result['modules']['price_volume'])
        self.assertEqual(qualification(ds)['status'],'blocked_data')
        ds.panels['S000']=ds.panels['S000'].drop(columns='amount')
        self.assertEqual(research_assessment(ds)['status'],'blocked_data')

    def test_derived_price_limit_does_not_prove_open_execution(self):
        from quantlab.backtest.ledger import trade_permission
        from quantlab.data.contracts import InstrumentRules
        from quantlab.data.exchange_rules import limit_prices
        from dataclasses import replace
        ds=dataset();d=ds.calendar[0];rule=InstrumentRules.from_record(ds.instruments.iloc[0])
        rule=replace(rule,instrument_id='sz.000001')
        bar=ds.panels['S000'].iloc[0].to_dict()
        bar.update(open_status='derived',open_status_basis='daily_open_assumption',rule_version='cn_daily_v1',is_st_at_open=False,listing_session_lower_bound=60)
        bar['limit_up'],bar['limit_down']=limit_prices(bar['reference_price'],.1)
        self.assertFalse(trade_permission(bar,rule,'buy',d)[0])
        self.assertTrue(trade_permission(bar,rule,'buy',d,allow_open_assumption=True)[0])
        bar['open']=bar['limit_up']
        self.assertEqual(trade_permission(bar,rule,'buy',d,allow_open_assumption=True),(False,'limit_up'))

    def test_rule_transition_and_listing_exceptions(self):
        from quantlab.data.exchange_rules import limit_ratio,limit_prices
        self.assertEqual(limit_ratio('300001','2020-08-21',False,100),.1)
        self.assertEqual(limit_ratio('300001','2020-08-24',True,100),.2)
        self.assertEqual(limit_ratio('600001','2020-08-24',True,100),.05)
        self.assertIsNone(limit_ratio('600001','2023-04-10',False,1))
        self.assertEqual(limit_prices(10.05,.1),(11.06,9.05))

    def test_rule_conflicting_open_cannot_be_clipped_into_a_fill(self):
        from quantlab.backtest.ledger import trade_permission
        from quantlab.data.contracts import InstrumentRules
        from dataclasses import replace
        ds=dataset();d=ds.calendar[0]
        rule=replace(InstrumentRules.from_record(ds.instruments.iloc[0]),instrument_id='sh.600225')
        bar=ds.panels['S000'].iloc[0].to_dict()
        bar.update(open_status='derived',open_status_basis='daily_open_assumption',rule_version='cn_daily_v1',
                   is_st_at_open=False,listing_session_lower_bound=60,reference_price=1.24,limit_up=1.36,limit_down=1.12)
        for price in (.62,1.50):
            bar['open']=price
            for side in ('buy','sell'):
                self.assertEqual(trade_permission(bar,rule,side,d,allow_open_assumption=True),
                                 (False,'derived_open_outside_rule_bounds'))

    def test_missing_equity_event_excludes_labels_but_not_earlier_features(self):
        from quantlab.strategy.opportunity import forward_labels,opportunity_features
        ds=dataset(days=360,stocks=6);before=opportunity_features(ds,profile='price_volume')
        event=ds.calendar[330]
        ds.manifest['data_gaps']=[dict(instrument_id='S000',date=str(event.date()),reason='unexplained_ex_reference')]
        labels=forward_labels(ds)
        self.assertNotIn((ds.calendar[315],'S000'),labels.index)
        after=opportunity_features(ds,profile='price_volume')
        pd.testing.assert_frame_equal(before.loc[:ds.calendar[329]],after.loc[:ds.calendar[329]])
        self.assertTrue(after.loc[(event,'S000')].isna().all())

    def test_free_opportunity_produces_orders_with_unknown_industry_cap(self):
        from quantlab.backtest.engine import Engine
        from quantlab.strategy.opportunity import Forecast
        ds=dataset(days=90);ds.manifest['source']='free_sample';ds.instruments['sector']='UNKNOWN'
        cfg=deepcopy(CONFIG);cfg.research.feature_profile='price_volume';cfg.research.universe_mode='pit_liquidity'
        d=ds.calendar[64]
        f=Forecast(str(d),20,pd.Series({c:.20 if c=='S000' else -.1 for c in ds.panels}),'ok',[],str(ds.calendar[60]),'free_fixture',dict(feature_profile='price_volume'))
        run=Engine(cfg).run(ds,forecasts={d:f},start=d,end=ds.calendar[65],data_mode='free_research',review_override=[d])
        self.assertGreater(len(run['fills']),0)
        self.assertLessEqual(sum(run['account'][-1]['weights'].values()),.250001)
        self.assertFalse(run['promotion_approved'])

    def test_price_volume_model_runs_without_industry_and_survives_truncation(self):
        from quantlab.strategy.opportunity import opportunity_features,normalized_features,OpportunityModel,forward_labels
        ds=dataset(days=440,stocks=8);ds.instruments['sector']='UNKNOWN'
        raw=opportunity_features(ds,profile='price_volume')
        features=normalized_features(raw);labels=forward_labels(ds)
        model=OpportunityModel(profile='price_volume').fit(features,labels,ds.calendar[260:390],ds.calendar[415])
        f=model.predict(features,ds.calendar[420],list(ds.panels))
        self.assertEqual(f.status,'ok')
        self.assertEqual(f.trace['feature_profile'],'price_volume')
        short=deepcopy(ds);short.manifest['as_of']=str(ds.calendar[420].date())
        truncated=opportunity_features(short,profile='price_volume')
        pd.testing.assert_frame_equal(raw.loc[:ds.calendar[420]],truncated)


if __name__=='__main__':unittest.main()
