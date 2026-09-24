import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from quantlab.config import CONFIG
from quantlab.strategy.experts import experts_score,expert_top_union
from quantlab.strategy.scoring import weight_row
from quantlab.strategy.opportunity import opportunity_features,normalized_features
from fixtures_v2 import dataset


class ReviewRegressions(unittest.TestCase):
    def test_open_execution_cannot_see_close_time_etf_exposure_revision(self):
        from quantlab.backtest.engine import Engine,prepare
        ds=dataset(days=70,stocks=6);c='S000';d=ds.calendar[45]
        ds.instruments.loc[ds.instruments.instrument_id==c,'asset_type']='equity_etf'
        ds.exposures=pd.DataFrame([
            dict(instrument_id=c,effective_at=str(ds.calendar[0].date()),available_at='2023-01-01',sector='old',weight=1.),
        ])
        cfg=deepcopy(CONFIG);before=prepare(ds,cfg)
        revised=deepcopy(ds)
        revised.exposures=pd.concat([ds.exposures,pd.DataFrame([
            dict(instrument_id=c,effective_at=str(d.date()),available_at=str(d.date())+' 16:00:00',sector='new',weight=1.),
        ])],ignore_index=True)
        after=prepare(revised,cfg)
        pd.testing.assert_frame_equal(before['exposures'][d],after['exposures'][d])
        schedule={ds.calendar[44]:{c:.2}}
        a=Engine(cfg).run(ds,strategy='schedule',target_schedule=schedule,start=ds.calendar[44],end=d)
        b=Engine(cfg).run(revised,strategy='schedule',target_schedule=schedule,start=ds.calendar[44],end=d)
        self.assertEqual(a['fills'],b['fills'])

    def test_etf_same_effective_date_revision_replaces_snapshot(self):
        from quantlab.data.market import exposure_matrix
        ds=dataset();c='S000';d=ds.calendar[45]
        ds.instruments.loc[ds.instruments.instrument_id==c,'asset_type']='equity_etf'
        ds.exposures=pd.DataFrame([
            dict(instrument_id=c,effective_at='2023-01-01',available_at='2023-01-01',sector='old',weight=1.),
            dict(instrument_id=c,effective_at='2023-01-01',available_at=str(d.date())+' 16:00:00',sector='new',weight=1.),
        ])
        exp=exposure_matrix(ds,d,ds.instruments.set_index('instrument_id'),[c])
        self.assertEqual(exp[c].to_dict(),{'new':1.})

    def test_crowding_filter_keeps_95_percent(self):
        date=pd.Timestamp('2024-01-01');cols=list(range(100))
        f={'x':pd.DataFrame([np.arange(100)],index=[date],columns=cols)}
        row=weight_row(pd.Series({'x':1.}),date)
        out=expert_top_union(f,row,None,pd.Series(np.arange(100),index=cols),100,{'one':['x']},cfg=deepcopy(CONFIG.factors))
        self.assertEqual(len(out['one']),95)
        cfg=deepcopy(CONFIG.factors);cfg.crowding_quantile=.5
        self.assertEqual(len(expert_top_union(f,row,None,pd.Series(np.arange(100)),100,{'one':['x']},cfg=cfg)['one']),50)

    def test_expert_uncertainty_and_injected_prior_affect_weights(self):
        date=pd.Timestamp('2024-01-01');f={k:pd.DataFrame([v],index=[date],columns=list('ABC')) for k,v in {'a':[1,2,3],'b':[3,2,1]}.items()}
        row=weight_row(pd.Series({'a':1.,'b':1.}),date)
        ic=pd.DataFrame({'a':np.tile([-.12,.2],6),'b':np.tile([-.16,.16],6)})
        cfg=deepcopy(CONFIG.factors)
        _,weights=experts_score(f,row,None,ic,{'a':['a'],'b':['b']},cfg=cfg)
        self.assertLess(weights['a']/weights['b'],3.)
        cfg.expert_prior_strength=100.
        _,shrunk=experts_score(f,row,None,ic,{'a':['a'],'b':['b']},cfg=cfg)
        self.assertLess(abs(shrunk['a']-.5),abs(weights['a']-.5))

    def test_sufficient_warmup_gives_identical_last_features(self):
        ds=dataset(days=600,stocks=20)
        full=normalized_features(opportunity_features(ds))
        shorter=deepcopy(ds);shorter.calendar=ds.calendar[100:]
        shorter.panels={c:p.iloc[100:] for c,p in ds.panels.items()};shorter.benchmark=ds.benchmark.iloc[100:]
        recomputed=normalized_features(opportunity_features(shorter))
        pd.testing.assert_frame_equal(full.loc[ds.calendar[500]:],recomputed.loc[ds.calendar[500]:],atol=1e-8,rtol=1e-8)

if __name__=='__main__':unittest.main()
