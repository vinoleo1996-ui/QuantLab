import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from fixtures_v2 import dataset
from quantlab.strategy.opportunity import opportunity_features, normalized_features, forward_labels, OpportunityModel
from quantlab.strategy.crowding import exit_risk
from quantlab.strategy.allocation import allocate
from quantlab.config import CONFIG


class OpportunityCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds=dataset(days=580,stocks=24)
        cls.raw=opportunity_features(cls.ds)
        cls.features=normalized_features(cls.raw)
        cls.labels=forward_labels(cls.ds)

    def test_future_data_does_not_change_features_or_fitted_predictions(self):
        d=self.ds.calendar[500]
        cut=deepcopy(self.ds);cut.manifest['as_of']=str(d.date())
        cut.panels={c:p.loc[:d] for c,p in cut.panels.items()};cut.benchmark=cut.benchmark.loc[:d]
        past=normalized_features(opportunity_features(cut))
        pd.testing.assert_frame_equal(self.features.loc[:d],past)
        train=self.ds.calendar[320:450]
        model=OpportunityModel().fit(self.features,self.labels,train,str(self.ds.calendar[475]))
        f=model.predict(self.features,d)
        self.assertEqual(f.status,'ok')
        pd.testing.assert_series_equal(f.expected_returns,model.predict(past,d).expected_returns)
        self.assertLess(pd.Timestamp(f.training_cutoff),pd.Timestamp(str(d.date())+' 18:00').tz_localize('Asia/Shanghai'))

    def test_frozen_model_roundtrip_and_tamper(self):
        import tempfile,json
        from pathlib import Path
        from quantlab.strategy.opportunity import save_model,load_model
        model=OpportunityModel().fit(self.features,self.labels,self.ds.calendar[320:450],str(self.ds.calendar[475]))
        with tempfile.TemporaryDirectory() as tmp:
            path=save_model(model,Path(tmp)/'model.json');restored=load_model(path)
            d=self.ds.calendar[500]
            pd.testing.assert_series_equal(model.predict(self.features,d).expected_returns,restored.predict(self.features,d).expected_returns)
            payload=json.loads(path.read_text());payload['coef'][0]+=1;path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'checksum_mismatch'):load_model(path)

    def test_labels_end_after_twenty_open_intervals(self):
        d=self.ds.calendar[100];c='S000';p=self.ds.panels[c]
        row=self.labels.loc[(d,c)]
        self.assertEqual(row.label_end,self.ds.calendar[121])
        self.assertAlmostEqual(row.target,p.open.iloc[121]/p.open.iloc[101]-1)

    def test_unknown_industry_is_invalid_not_current_backfill(self):
        ds=deepcopy(self.ds);ds.instruments['sector']='UNKNOWN'
        raw=opportunity_features(ds)
        self.assertTrue(raw.dropna().empty)
        records=self.ds.instruments.set_index('instrument_id',drop=False).copy();records['sector']='UNKNOWN'
        ret=pd.DataFrame({c:p.close.pct_change() for c,p in self.ds.panels.items()}).dropna().tail(60)
        w,status=allocate(ret,records,pd.DataFrame(1.,index=['UNKNOWN'],columns=ret.columns),CONFIG)
        self.assertEqual(status,'industry_coverage_insufficient')

    def test_shared_downside_returns_form_one_capped_group(self):
        ds=self.ds;codes=list(ds.panels)[:3]
        close=pd.DataFrame({c:ds.benchmark.close for c in codes})
        turnover=close*0+1;amount=close*0+1e8
        risk=exit_risk(close,turnover,amount,ds.benchmark.close,close.index[-1],codes)
        self.assertEqual(len(set(risk.groups.values())),1)
        cfg=deepcopy(CONFIG);cfg.research.max_cluster_weight=.1
        records=ds.instruments.set_index('instrument_id',drop=False)
        exp=pd.get_dummies(records.loc[codes].sector,dtype=float).T
        weights,status=allocate(close.pct_change().tail(100),records,exp,cfg,expected=pd.Series(.2,index=codes),risk_snapshot=risk)
        self.assertEqual(status,'ok');self.assertLessEqual(weights.sum(),.100001)


if __name__=='__main__':unittest.main()
