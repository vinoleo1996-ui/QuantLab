"""Pool-market diagnostic: as-of/availability contracts, the 2026-09-27 NaT crash
regression, and the skew-robust decomposition of the primary statistic.

The primary statistic is heads observed MEAN minus market observed MEDIAN. It
therefore embeds the market's own average-vs-typical gap; SkewRobustStatisticTests
pins the exact decomposition primary == location_mean + market_own_skew so a large
primary can never be reported without the tail that produced it.
"""
import unittest
import json
import numpy as np
import pandas as pd

from scripts.reference_pool_market_diagnostic import (
    market_members, valid_prices, summarize_values, compare_values,
    json_safe, skew_robust_contrasts)
from scripts.router_random_diagnostic import forward_label


class PoolMarketTests(unittest.TestCase):
    def records(self):
        return pd.DataFrame([dict(instrument_id='sz.000001',effective_at='2022-01-01',available_at='2022-01-01 18:00',active=True,asset_type='stock',revision=0)])
    def quotes(self):
        return {'sz.000001':dict(feature_close=10.,available_at='2022-01-03 18:00',quality_status='ok')}
    def test_asof_future_invariance(self):
        r=self.records(); q=self.quotes(); base=market_members(r,q,'2022-01-03')[0]
        future=r.iloc[0].to_dict();future.update(effective_at='2022-01-04',available_at='2022-01-04 18:00',active=False)
        self.assertEqual(base,market_members(pd.concat([r,pd.DataFrame([future])]),q,'2022-01-03')[0]);self.assertEqual(base,['sz.000001'])
    def test_effective_and_available_separate(self):
        r=self.records();r.loc[0,'effective_at']='2022-01-04';self.assertEqual(market_members(r,self.quotes(),'2022-01-03')[0],[])
        r=self.records();r.loc[0,'available_at']='2022-01-04';self.assertEqual(market_members(r,self.quotes(),'2022-01-03')[0],[])
    def test_delisting_before_signal_excluded(self):
        r=self.records();r.loc[0,'active']=False;self.assertEqual(market_members(r,self.quotes(),'2022-01-03')[0],[])
    def test_late_bad_and_nonpositive_quote(self):
        for field,value in [('available_at','2022-01-03 18:01'),('quality_status','bad'),('feature_close',0),('feature_close',float('nan'))]:
            q=self.quotes();q['sz.000001'][field]=value;self.assertEqual(market_members(self.records(),q,'2022-01-03')[0],[])
    def test_timezone_equivalence(self):
        q=self.quotes();q['sz.000001']['available_at']='2022-01-03T10:00:00Z';self.assertEqual(market_members(self.records(),q,'2022-01-03')[0],['sz.000001'])
    def test_ambiguous_records_rejected(self):
        r=self.records();other=r.copy();other['active']=False
        with self.assertRaises(ValueError):market_members(pd.concat([r,other]),self.quotes(),'2022-01-03')
    def test_unknown_not_zero_or_removed_from_denominator(self):
        s=summarize_values([.2,np.nan],'mean');self.assertEqual(s['requested'],2);self.assertEqual(s['observed'],1);self.assertAlmostEqual(s['conditional'],.2);self.assertAlmostEqual(s['lower_bound'],-.4);self.assertEqual(s['upper_bound'],'unbounded')
    def test_mean_versus_median(self):
        c=compare_values([0,0,.9],[0,0,.9]);self.assertAlmostEqual(c['conditional_difference'],.3)
    def test_median_bounds_original_count(self):
        s=summarize_values([0,.1,.2,np.nan],'median');self.assertAlmostEqual(s['lower_bound'],.05);self.assertAlmostEqual(s['upper_bound'],.15)
    def test_future_path_and_missing_semantics(self):
        cal=pd.bdate_range('2022-01-03',periods=65);p=pd.Series(np.arange(65)+100.,index=cal)
        self.assertAlmostEqual(forward_label(p,cal,cal[0],20,cal[-1],[])[0],.2)
        p.iloc[30:]=999;self.assertAlmostEqual(forward_label(p,cal,cal[0],20,cal[-1],[])[0],.2)
        p.iloc[1]=np.nan;self.assertEqual(forward_label(p,cal,cal[0],20,cal[-1],[])[1],'missing_path')
        self.assertEqual(forward_label(p,cal,cal[0],20,cal[-1],[cal[2]])[1],'known_gap')
        self.assertEqual(forward_label(p,cal,cal[0],20,cal[10],[])[1],'period_end')
    def test_quality_and_late_price_mask(self):
        p=pd.DataFrame({'feature_close':[1,2,3],'available_at':['2022-01-03 18:00','2022-01-04 18:01','2022-01-05 18:00'],'quality_status':['ok','ok','bad']},index=pd.bdate_range('2022-01-03',periods=3))
        v=valid_prices(p);self.assertEqual(v.iloc[0],1);self.assertTrue(v.iloc[1:].isna().all())


class CrashRegressionTests(unittest.TestCase):
    """Reproduces the 2026-09-27T18:04 crash: NaT endpoint must serialise, not abort."""
    def test_nat_endpoint_becomes_json_null_not_nan(self):
        self.assertIsNone(json_safe(pd.NaT))
        self.assertIsNone(json_safe(float('nan')))
        # A real signal date is NOT a missing endpoint and must stay a value.
        self.assertEqual(json_safe(pd.Timestamp('2026-09-04')),'2026-09-04T00:00:00')
    def test_real_date_endpoint_survives_as_string(self):
        self.assertEqual(json_safe(pd.Timestamp('2026-07-01')),'2026-07-01T00:00:00')
    def test_crash_payload_now_serialises_under_allow_nan_false(self):
        # Exactly the shape that aborted attempt 1: a per_date row whose endpoint
        # is NaT because the 60-session label never matured.
        row=dict(decision_id=67,period='2025_recent',signal_date='2026-07-10',horizon=60,
                 endpoint=pd.NaT,heads_reason_counts={'missing_path':5},skew=dict(location_median=None))
        text=json.dumps(json_safe([row]),ensure_ascii=False,indent=2,allow_nan=False)
        self.assertIsNone(json.loads(text)[0]['endpoint'])
        self.assertIn('missing_path',text)
    def test_nested_nan_inside_contrast_block_is_nulled(self):
        self.assertEqual(json_safe(dict(a=[1.0,float('nan')],b=dict(c=float('-inf')))),
                         dict(a=[1.0,None],b=dict(c=None)))
    def test_no_nan_survives_a_real_contrast_row(self):
        hv=np.array([0.1,0.2,np.nan,0.3]);mv=np.array([0.05,0.9,0.4,0.2,0.1])
        r=compare_values(hv,mv);r['skew']=skew_robust_contrasts(hv,mv);r['endpoint']=pd.NaT
        json.dumps(json_safe(r),allow_nan=False)  # must not raise


class SkewRobustStatisticTests(unittest.TestCase):
    """The primary mean-minus-median is skew-sensitive; these pin the decomposition."""
    def test_identity_primary_equals_location_mean_plus_market_skew(self):
        rng=np.random.default_rng(7)
        for _ in range(200):
            hv=rng.normal(0,.1,size=rng.integers(3,60));mv=np.r_[rng.normal(0,.1,size=40),rng.normal(.5,.4,size=5)]
            primary=compare_values(hv,mv)['conditional_difference']
            s=skew_robust_contrasts(hv,mv)
            self.assertAlmostEqual(primary,s['location_mean']+s['market_skew'],places=12)
    def test_market_right_tail_is_separable_not_implied(self):
        # The primary is heads MEAN minus market MEDIAN, so it embeds the market's
        # own average-vs-typical gap. Build a market with a pinned median and an
        # inflated right tail and show the three quantities separate cleanly.
        # market = 200 zeros + one 100 -> median 0, mean 100/201.
        # heads = 0.5 flat -> mean .5, median .5.
        m=np.r_[np.zeros(200),100.];h=np.full(200,.5)
        s=skew_robust_contrasts(h,m)
        self.assertAlmostEqual(s['market_median'],0.0,places=12)
        self.assertAlmostEqual(s['location_median'],.5,places=12)          # typical beats typical
        self.assertAlmostEqual(s['location_mean'],.5-100/201,places=12)   # average is dragged down
        primary=compare_values(h,m)['conditional_difference']
        self.assertAlmostEqual(primary,.5,places=12)                      # primary = heads_mean - 0
        self.assertAlmostEqual(primary,s['location_mean']+s['market_skew'],places=12)
        # Now make the pool EXACTLY average: identical 201-element samples on both
        # sides, so heads mean == market mean exactly. A pool with zero genuine
        # edge still shows a large primary whenever the market is right-skewed.
        # This is the misreading the decomposition exists to prevent.
        m3=np.r_[np.zeros(200),100.]
        h3=m3.copy()
        s3=skew_robust_contrasts(h3,m3)
        self.assertAlmostEqual(s3['heads_mean'],s3['market_mean'],places=12)   # zero real edge
        self.assertAlmostEqual(s3['location_mean'],0.0,places=12)
        self.assertAlmostEqual(s3['location_median'],0.0,places=12)
        self.assertGreater(s3['market_skew'],.4)                                # all tail
        primary3=compare_values(h3,m3)['conditional_difference']
        self.assertGreater(primary3,.4)                                         # yet primary looks big
        self.assertAlmostEqual(primary3,s3['location_mean']+s3['market_skew'],places=12)
    def test_location_median_invariant_to_outlier_size(self):
        h=np.full(200,.5)
        a=skew_robust_contrasts(h,np.r_[np.zeros(200),100.])['location_median']
        b=skew_robust_contrasts(h,np.r_[np.zeros(200),10000.])['location_median']
        self.assertAlmostEqual(a,b,places=12)
        # location_mean is NOT invariant: it is the one that tracks the tail.
        am=skew_robust_contrasts(h,np.r_[np.zeros(200),100.])['location_mean']
        bm=skew_robust_contrasts(h,np.r_[np.zeros(200),10000.])['location_mean']
        self.assertLess(bm,am)
    def test_location_median_is_skew_free_when_both_sides_share_a_tail(self):
        rng=np.random.default_rng(11);base=rng.normal(0,.05,size=50)
        self.assertAlmostEqual(skew_robust_contrasts(base,base)['location_median'],0.0,places=12)
        self.assertAlmostEqual(skew_robust_contrasts(base,base)['location_mean'],0.0,places=12)
    def test_known_values_hand_checked(self):
        s=skew_robust_contrasts([0.,1.,2.],[0.,0.,9.])
        self.assertAlmostEqual(s['heads_median'],1.0);self.assertAlmostEqual(s['market_median'],0.0)
        self.assertAlmostEqual(s['location_median'],1.0)
        self.assertAlmostEqual(s['location_mean'],1.0-3.0)
        self.assertAlmostEqual(s['market_skew'],3.0)
    def test_empty_side_yields_none_not_crash(self):
        s=skew_robust_contrasts([np.nan],[0.1,0.2])
        self.assertTrue(all(v is None for v in s.values()))


if __name__=='__main__':unittest.main()
