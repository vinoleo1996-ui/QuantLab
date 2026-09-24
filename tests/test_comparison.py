import unittest
from copy import deepcopy
import pandas as pd
import numpy as np
from fixtures_v2 import dataset
from quantlab.config import CONFIG
from quantlab.strategy.opportunity import Forecast
from quantlab.comparison import rank_matched, restrict_forecast, reference_scores
from quantlab.backtest.engine import prepare


class MatchedComparison(unittest.TestCase):
    def test_unhealthy_control_does_not_invalidate_healthy_candidate_selection(self):
        from quantlab.comparison import validation_objective
        dates=pd.bdate_range('2022-01-03',periods=4)
        health=dict(allocation_failures=0,forecast_failures=0,data_issue_count=0)
        def run(returns,issues=0):
            r=pd.Series(returns,index=dates)
            return dict(daily_return=r,nav=(1+r).cumprod(),run_health=dict(health,data_issue_count=issues))
        a=run([.01,0,-.002,.004]);b=run([.005,0,-.002,.004])
        good=run([.002]*4);bad=run([.002]*4,1)
        va=validation_objective(a,good);vb=validation_objective(b,good)
        self.assertGreater(va['selection_score'],vb['selection_score'])
        self.assertGreater(va['net_excess'],vb['net_excess'])
        impaired=validation_objective(a,bad)
        self.assertEqual(impaired['selection_score'],va['selection_score'])
        self.assertIsNone(impaired['net_excess'])
        self.assertEqual(impaired['comparison_status'],'control_unusable')
        self.assertIsNone(validation_objective(bad,good)['selection_score'])

    def test_security_with_only_future_bars_remains_unknown(self):
        from quantlab.data.market import panels_for_research
        ds=dataset(days=120,stocks=3)
        ds.manifest['as_of']=str(ds.calendar[80].date())
        ds.panels['S000']=ds.panels['S000'].iloc[100:]
        close,amount,turnover=panels_for_research(ds)
        self.assertTrue(close['S000'].isna().all())
        self.assertTrue(amount['S000'].isna().all())

    def test_ranking_changes_identity_not_return_distribution_or_training_cutoff(self):
        f=Forecast('2024-02-01',20,pd.Series({'a':-.01,'b':.03,'c':.02}),'ok',[],
                   '2024-01-31','frozen',{'feature_profile':'price_volume'})
        changed=rank_matched(f,pd.Series({'a':3.,'b':1.,'c':2.}),'reference')
        self.assertEqual(changed.expected_returns['a'],.03)
        np.testing.assert_array_equal(np.sort(changed.expected_returns),np.sort(f.expected_returns))
        self.assertEqual(changed.training_cutoff,f.training_cutoff)
        self.assertEqual(changed.horizon,f.horizon)
        self.assertEqual(rank_matched(f,pd.Series(1.,index=['a','b','c']),'constant').status,'invalid')
        restricted=restrict_forecast(f,pd.Series({'a':1.,'c':2.}),pd.Series({'a':2.,'c':1.}))
        self.assertEqual(set(restricted.expected_returns.index),{'a','c'})
        self.assertEqual(restricted.trace['original_forecast_count'],3)

    def test_future_quotes_do_not_rewrite_repaired_legacy_or_trend_rank(self):
        ds=dataset(days=340,stocks=12);cfg=deepcopy(CONFIG)
        cfg.research.feature_profile='price_volume';cfg.research.universe_mode='pit_liquidity'
        d=ds.calendar[300]
        old=reference_scores(ds,prepare(ds,cfg),cfg,[d])
        other=deepcopy(ds)
        for c,p in other.panels.items():p.loc[p.index>d,'close']*=1.5
        new=reference_scores(other,prepare(other,cfg),cfg,[d])
        self.assertGreater(old[0][d].nunique(),1)
        pd.testing.assert_series_equal(old[0][d],new[0][d])
        pd.testing.assert_series_equal(old[1][d],new[1][d])


if __name__=='__main__':unittest.main()
