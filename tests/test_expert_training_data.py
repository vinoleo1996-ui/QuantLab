import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class TrainingDataTests(unittest.TestCase):
    def api(self):
        self.assertTrue((ROOT/'scripts/prepare_expert_inputs.py').exists(),
                        'Chronological feature/label builder is missing')
        import prepare_expert_inputs as api
        return api

    def test_labels_need_exact_endpoint_and_respect_gap(self):
        api = self.api()
        dates = pd.bdate_range('2020-01-01', periods=25)
        close = pd.DataFrame({'a':np.arange(100.,125.), 'b':np.arange(100.,125.)},index=dates)
        close.loc[dates[20], 'b'] = np.nan
        values, status, endpoint = api.forward_labels(close, dates, dates[0], ['a','b'], [],20)
        self.assertAlmostEqual(values['a'], .2)
        self.assertTrue(pd.isna(values['b']))
        self.assertEqual(status['b'],'missing_endpoint')
        self.assertEqual(endpoint,dates[20])
        values, status, _ = api.forward_labels(close, dates, dates[0], ['a'],
            [{'date':str(dates[10].date()),'instrument_id':'a','reason':'unresolved_action'}],20)
        self.assertTrue(pd.isna(values['a']))
        self.assertEqual(status['a'],'accounting_gap')

    def test_future_append_does_not_change_matured_label(self):
        api = self.api()
        dates=pd.bdate_range('2020-01-01',periods=30)
        p=pd.DataFrame({'a':np.arange(100.,130.)},index=dates)
        a=api.forward_labels(p.iloc[:25],dates[:25],dates[0],['a'],[],20)
        p.iloc[25:]=10000
        b=api.forward_labels(p,dates,dates[0],['a'],[],20)
        pd.testing.assert_series_equal(a[0],b[0]);self.assertEqual(a[2],b[2])

    def test_unmatured_label_not_filled(self):
        api=self.api();d=pd.bdate_range('2020-01-01',periods=10)
        y,s,end=api.forward_labels(pd.DataFrame({'a':100.},index=d),d,d[0],['a'],[],20)
        self.assertIsNone(end);self.assertTrue(y.isna().all())
        self.assertTrue(s.eq('not_matured').all())

if __name__=='__main__':unittest.main()
