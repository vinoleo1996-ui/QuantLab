import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

PATH=Path(__file__).resolve().parents[1]/'quantlab/strategy/nonlinear_expert.py'


class NonlinearExpertTests(unittest.TestCase):
    def api(self):
        self.assertTrue(PATH.exists(),'Nonlinear expert not implemented')
        if 'nonlinear_under_test' not in sys.modules:
            spec=importlib.util.spec_from_file_location('nonlinear_under_test',PATH)
            mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        return sys.modules['nonlinear_under_test']

    def frame(self):
        api=self.api();rng=np.random.default_rng(3);rows=[]
        for day,end in [('2019-01-31','2019-03-01'),('2019-02-28','2019-04-01'),('2019-03-29','2019-05-01')]:
            for i in range(80):
                x=rng.normal(size=11)
                r=dict(zip(api.FEATURES,x));r.update(date=day,instrument_id=str(i),
                    label_available_at=end+' 18:00:00+08:00',return_20=float(x[0]*x[1]+x[2]),coverage_20='observed')
                rows.append(r)
        return pd.DataFrame(rows)

    def test_only_matured_cohorts_and_no_train_predictions(self):
        api=self.api();p=api.EnsemblePolicy(cohorts=2,num_models=2,rounds=8,min_leaf=5)
        m=api.MonthlyEnsemble('plain',p).fit(self.frame(),'2019-04-02 18:00:00+08:00')
        self.assertEqual(m.evidence['cohort_dates'],['2019-01-31','2019-02-28'])
        self.assertLess(pd.Timestamp(m.evidence['latest_label_available_at']),pd.Timestamp('2019-04-02 18:00:00+08:00'))
        with self.assertRaisesRegex(ValueError,'prediction_before_training_cutoff'):
            m.predict(self.frame().iloc[:4],as_of='2019-03-01 18:00:00+08:00')

    def test_future_labels_cannot_change_model_and_seed_is_reproducible(self):
        api=self.api();p=api.EnsemblePolicy(cohorts=2,num_models=2,rounds=8,min_leaf=5)
        f=self.frame();changed=f.copy();changed.loc[changed.date=='2019-03-29','return_20']*=1000
        a=api.MonthlyEnsemble('double',p).fit(f,'2019-04-02 18:00:00+08:00')
        b=api.MonthlyEnsemble('double',p).fit(changed,'2019-04-02 18:00:00+08:00')
        x=f.iloc[:20].copy()
        pd.testing.assert_series_equal(a.predict(x,as_of='2019-04-03 18:00:00+08:00'),b.predict(x,as_of='2019-04-03 18:00:00+08:00'))
        self.assertEqual(a.evidence['sub_features'],b.evidence['sub_features'])

    def test_shuffling_stays_within_date_and_weights_are_date_balanced(self):
        api=self.api();dates=np.array(['a']*3+['b']*7)
        x=np.r_[np.arange(3),100+np.arange(7)]
        y=api.shuffle_within_dates(x,dates,np.random.default_rng(1))
        self.assertEqual(set(y[:3]),set(x[:3]));self.assertEqual(set(y[3:]),set(x[3:]))
        w=api.balance_dates(np.arange(1.,11.),dates)
        self.assertAlmostEqual(w[:3].sum(),w[3:].sum())

    def test_prediction_responds_to_features(self):
        api=self.api();p=api.EnsemblePolicy(cohorts=2,num_models=2,rounds=15,min_leaf=5)
        f=self.frame();m=api.MonthlyEnsemble('plain',p).fit(f,'2019-04-02 18:00:00+08:00')
        x=f.iloc[:40].copy();a=m.predict(x,as_of='2019-04-03 18:00:00+08:00')
        x.loc[:,api.FEATURES[2]]=-x[api.FEATURES[2]]*4
        b=m.predict(x,as_of='2019-04-03 18:00:00+08:00')
        self.assertFalse(a.equals(b))

if __name__=='__main__':unittest.main()
