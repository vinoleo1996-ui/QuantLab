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
