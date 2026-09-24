import unittest
import tempfile
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.config import CONFIG
from quantlab.strategy.allocation import allocate


class CashOpportunity(unittest.TestCase):
    def test_negative_expected_returns_choose_cash(self):
        ds = dataset(days=100)
        cfg = deepcopy(CONFIG)
        returns = pd.DataFrame({c:p.close.pct_change() for c,p in ds.panels.items()}).dropna()
        records = ds.instruments.set_index('instrument_id')
        exp = pd.get_dummies(records.sector, dtype=float).T
        w, status = allocate(returns, records, exp, cfg, expected=pd.Series(-.03,index=returns.columns))
        self.assertEqual(status, 'ok')
        self.assertLess(w.sum(), 1e-5)

    def test_forecast_perturbation_changes_order(self):
        from quantlab.backtest.engine import Engine
        from quantlab.strategy.opportunity import Forecast
        ds=dataset(days=100); cfg=deepcopy(CONFIG)
        cfg.start_date=str(ds.calendar[60].date());cfg.universe.a_min_listing_days=1
        cfg.research.universe_mode='pit_liquidity'
        d=ds.calendar[64]
        def run(code):
            f=Forecast(str(d),20,pd.Series({c:.2 if c==code else -.1 for c in ds.panels}),
                       'ok',[],str(ds.calendar[60]),'fixture-v1')
            return Engine(cfg).run(ds,strategy='opportunity',forecasts={d:f},review_override=[d])
        a,b=run('S000'),run('S001')
        self.assertTrue(any(f['instrument_id']=='S000' and f['side']=='buy' for f in a['fills']))
        self.assertTrue(any(f['instrument_id']=='S001' and f['side']=='buy' for f in b['fills']))
        self.assertNotEqual(a['fills'],b['fills'])


class ResearchIntegrity(unittest.TestCase):
    def test_coverage_requires_measured_matching_artifacts(self):
        from quantlab.data.qualification import qualification,coverage_input_fingerprint,REQUIRED
        import hashlib
        ds=dataset();ds.manifest['source']='vendor'
        with tempfile.TemporaryDirectory() as d:
            ds.evidence_root=Path(d);refs={}
            for field in (*REQUIRED,'industry_history'):
                doc=dict(kind='data_coverage',field=field,status='passed',input_fingerprint=coverage_input_fingerprint(ds),
                    coverage=dict(start=str(ds.calendar[0]),end=str(ds.calendar[-1]),instruments=sorted(ds.panels)),
                    measurements=dict(expected_observations=10000,actual_observations=10000,unresolved_required_gaps=0),
                    method_version='crosscheck-v1',source_urls=['https://provider.example/specification'])
                p=Path(d)/(field+'.json');p.write_text(json.dumps(doc))
                refs[field]=dict(source_url=doc['source_urls'][0],verification_artifact=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            ds.manifest['coverage_evidence']=refs
            self.assertEqual(qualification(ds)['status'],'qualified_for_daily_research')
            p=Path(d)/'raw_prices.json';doc=json.loads(p.read_text());doc['measurements']={}
            p.write_text(json.dumps(doc));refs['raw_prices']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertIn('coverage_artifact_invalid:raw_prices:measurements_missing',qualification(ds)['reasons'])

    def test_registry_counts_configuration_not_folds_and_outcomes_immutable(self):
        from quantlab.research import ExperimentRegistry
        with tempfile.TemporaryDirectory() as d:
            r=ExperimentRegistry(Path(d)/'trials.jsonl')
            a=r.register('h',{'alpha':1},'data','fold1','v1')
            b=r.register('h',{'alpha':1},'data','fold2','v1')
            self.assertEqual(a,b)
            r.record(a,{'fold':'fold1','net_excess':.01})
            with self.assertRaisesRegex(ValueError,'already_recorded'):
                r.record(a,{'fold':'fold1','net_excess':.02})

    def test_purge_uses_real_label_end(self):
        from quantlab.research import purge_labels
        ix=pd.bdate_range('2024-01-01',periods=5)
        ends=pd.Series([ix[1],ix[2],ix[4],pd.NaT,pd.NaT],index=ix)
        kept=purge_labels(ix,ends,ix[4])
        self.assertEqual(list(kept),list(ix[:2]))

    def test_boolean_gate_is_not_accepted_as_evidence(self):
        from quantlab.evidence import verify_bundle
        r=verify_bundle({'accounting_passed':True}, Path('.'))
        self.assertFalse(r['verified'])
        self.assertTrue(r['reasons'])


if __name__=='__main__':unittest.main()

class Attribution(unittest.TestCase):
    def test_contributions_reconcile_ledger_and_costs(self):
        from quantlab.backtest.engine import Engine
        from quantlab.report.attribution import instrument_returns
        ds=dataset();cfg=deepcopy(CONFIG);cfg.start_date=str(ds.calendar[20].date())
        r=Engine(cfg).run(ds,strategy='schedule',target_schedule={ds.calendar[21]:{'S000':.05},ds.calendar[40]:{'S001':.05}})
        contributions=instrument_returns(r)
        pd.testing.assert_series_equal(contributions.sum(axis=1),r['daily_return'],check_names=False,atol=1e-12,rtol=1e-10)

class SolverFailure(unittest.TestCase):
    def test_failed_optimizer_cannot_add_risk_via_starting_guess(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        ds=dataset(days=100);records=ds.instruments.set_index('instrument_id',drop=False)
        returns=pd.DataFrame({c:p.close.pct_change() for c,p in ds.panels.items()}).dropna()
        exp=pd.get_dummies(records.sector,dtype=float).T
        with patch('quantlab.strategy.allocation.minimize',return_value=SimpleNamespace(success=False,x=None)):
            w,status=allocate(returns,records,exp,CONFIG,expected=pd.Series(.2,index=returns.columns))
        self.assertEqual(status,'solver_failed_or_infeasible');self.assertEqual(w.sum(),0.)
