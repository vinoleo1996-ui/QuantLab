"""Uncertain pair risks must constrain exposure without erasing known evidence."""
import unittest

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.strategy.crowding import exit_risk
from quantlab.strategy.legacy import allocate_legacy
from quantlab.strategy.risk_policy import PolicyController


class DownsideUncertaintyTests(unittest.TestCase):
    def test_one_sixty_session_candidate_does_not_idle_the_whole_account(self):
        dates=pd.bdate_range('2024-01-02',periods=180)
        codes=[f'O{i:02}' for i in range(19)]+['NEW']
        close=pd.DataFrame(20*np.exp(np.cumsum(np.random.default_rng(932).normal(0,.003,(180,20)),axis=0)),
                           index=dates,columns=codes)
        close.loc[dates[:-60],'NEW']=np.nan
        market=pd.Series(100*np.cumprod(1+np.tile([.001,.001,-.001,.001,-.001],36)),index=dates)
        amount=close.notna().astype(float)*1e8
        snapshot=exit_risk(close,amount/1e8,amount,market,dates[-1],codes,include_penalties=False)
        self.assertEqual(snapshot.groups['NEW'],'UNKNOWN')
        self.assertEqual(sum(g=='UNKNOWN' for g in snapshot.groups.values()),1)
        returns=close.pct_change(fill_method=None).tail(120)
        cfg=Config()
        matrix=snapshot.matrix(codes)
        for policy in ['P0','P1','P2']:
            with self.subTest(policy=policy):
                budget=PolicyController(policy,cfg,market).evaluate(dates[-1],600000.)
                weights,status,detail=allocate_legacy(returns,cfg,selected=codes,
                    exposures=pd.DataFrame(np.eye(20),index=codes,columns=codes),
                    risk_snapshot=snapshot,budget=budget)
                self.assertEqual(status,'ok',detail)
                self.assertGreater(weights.sum(),.97)
                # Unknown may co-fall with each known group, even though the known
                # groups themselves have observed correlations below the threshold.
                for group in set(snapshot.groups.values())-{'UNKNOWN'}:
                    self.assertEqual(matrix.loc[group,'NEW'],1.)
                    self.assertLessEqual(float(matrix.loc[group]@weights),budget.soft_cluster_weight+1e-6)
                cap=budget.unknown_cluster_weight-(.02 if policy!='P0' else 0.)
                self.assertLessEqual(float(matrix.loc['UNKNOWN']@weights),cap+1e-6)

    def test_pairwise_missing_history_keeps_explicit_joint_constraint(self):
        dates=pd.bdate_range('2024-01-02',periods=121)
        codes=['A','B','C','NEW']
        close=pd.DataFrame(20*np.exp(np.cumsum(np.random.default_rng(234).normal(0,.005,(121,4)),axis=0)),
                           index=dates,columns=codes)
        # A and B each have more than 30 down observations, but their overlap
        # has fewer than 30. C has a complete independent history.
        close.loc[dates[72:],'A']=np.nan
        close.loc[dates[:49],'B']=np.nan
        close.loc[dates[:109],'NEW']=np.nan
        market=pd.Series(100*.999**np.arange(121),index=dates)
        amount=pd.DataFrame(1e8,index=dates,columns=codes)
        snapshot=exit_risk(close,amount/1e8,amount,market,dates[-1],codes,include_penalties=False)
        self.assertNotEqual(snapshot.groups['A'],'UNKNOWN')
        self.assertNotEqual(snapshot.groups['B'],'UNKNOWN')
        self.assertNotEqual(snapshot.groups['C'],'UNKNOWN')
        self.assertNotEqual(snapshot.groups['A'],snapshot.groups['B'])
        joint=snapshot.matrix(codes)
        rows=joint[(joint.A==1)&(joint.B==1)&(joint.C==0)&(joint.NEW==1)]
        self.assertEqual(len(rows),1)
        self.assertTrue(any('pair_history_incomplete:A:B' in reason for reason in snapshot.reasons))
        weights=pd.Series(dict(A=.15,B=.15,C=.05,NEW=.08))
        self.assertGreater(float(rows.iloc[0]@weights),.35)

    def test_future_observations_do_not_remove_past_uncertainty(self):
        dates=pd.bdate_range('2024-01-02',periods=180)
        codes=['A','B','NEW']
        close=pd.DataFrame(20*np.exp(np.cumsum(np.random.default_rng(732).normal(0,.005,(180,3)),axis=0)),
                           index=dates,columns=codes)
        close.loc[dates[:100],'NEW']=np.nan
        market=pd.Series(100*.999**np.arange(180),index=dates)
        amount=pd.DataFrame(1e8,index=dates,columns=codes)
        cutoff=dates[120]
        full=exit_risk(close,amount/1e8,amount,market,cutoff,codes,include_penalties=False)
        prefix=exit_risk(close.loc[:cutoff],amount.loc[:cutoff]/1e8,amount.loc[:cutoff],
                         market.loc[:cutoff],cutoff,codes,include_penalties=False)
        self.assertEqual(full.groups,prefix.groups)
        self.assertEqual(full.reasons,prefix.reasons)
        pd.testing.assert_frame_equal(full.matrix(codes),prefix.matrix(codes))


if __name__=='__main__':
    unittest.main()
