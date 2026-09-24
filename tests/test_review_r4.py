import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import pandas as pd

from fixtures_v2 import dataset
from quantlab.config import CONFIG


class DeliveryContracts(unittest.TestCase):
    def test_public_quote_fallback_preserves_per_row_source_and_time(self):
        from unittest.mock import patch
        from quantlab.data.realtime import batch_quotes
        def row(c):return dict(code=c,name=c,price=10.,pre_close=10.,open=10.,high=10.,low=10.,volume=100,amount=1000)
        with patch('quantlab.data.realtime._sina',return_value=([row('600000')],'sina-time')),patch('quantlab.data.realtime._tencent',return_value=([row('000001')],'tencent-time')) as fallback:
            q=batch_quotes(['600000','000001'])
            fallback.assert_called_once_with(['000001'])
            self.assertEqual(q.loc['600000','source'],'sina')
            self.assertEqual(q.loc['000001','time'],'tencent-time')

    def test_legacy_control_respects_reduced_risk_budget(self):
        from quantlab.strategy.allocation import allocate
        ds=dataset(days=100);records=ds.instruments.set_index('instrument_id')
        r=pd.DataFrame({c:p.close.pct_change() for c,p in ds.panels.items()}).dropna()
        w,status=allocate(r,records,pd.get_dummies(records.sector,dtype=float).T,CONFIG,gross=.3,vol_scale=.75)
        self.assertEqual(status,'ok');self.assertLessEqual(w.sum(),.300001)

    def test_synthetic_dataset_is_not_qualified_data(self):
        from quantlab.data.qualification import qualification
        result=qualification(dataset())
        self.assertEqual(result['status'],'blocked_data')
        self.assertIn('synthetic_test_fixture',result['reasons'])

    def test_signal_direction_comes_from_pending_quantity(self):
        from quantlab.cli import signal_payload
        result=dict(status='research_unvalidated',account=[dict(nav=20000,quantities={'S000':100},weights={'S000':.05})],
            trades=[dict(signal_date='2024-01-01',exec_date='2024-01-02',weights={'S000':.1})],
            orders=[dict(order_id='O1',instrument_id='S000',side='sell',quantity=100,remaining=100,
                         status='pending',signal_date='2024-01-01',reason='risk_exit',reject_reason='limit_down')])
        payload=signal_payload(result)
        self.assertEqual(payload['trades'][0][3],'清仓')
        self.assertEqual(payload['trades'][0][1],0.)

    def test_run_collision_and_tampered_pickle_rejected(self):
        from quantlab.backtest.engine import blocked_result
        from quantlab.report.runs import persist_run,load_run
        with tempfile.TemporaryDirectory() as tmp:
            r=blocked_result(['fixture']);folder=persist_run(r,Path(tmp))
            self.assertEqual(load_run(folder)['run_id'],r['run_id'])
            changed=deepcopy(r);changed['orders']=[{'unexpected':True}]
            with self.assertRaisesRegex(ValueError,'run_identity_collision'):persist_run(changed,Path(tmp))
            (folder/'result.pkl').write_bytes(b'untrusted')
            with self.assertRaisesRegex(ValueError,'artifact_checksum'):load_run(folder)

    def test_mandate_identity_ignores_data_cache_location(self):
        from quantlab.mandate import Mandate
        a=deepcopy(CONFIG);b=deepcopy(CONFIG);b.data.cache_dir=Path('/some/other/cache')
        self.assertEqual(Mandate.from_config(a).version,Mandate.from_config(b).version)

    def test_manual_resume_preserves_control_high_water_mark(self):
        from quantlab.strategy.risk import RiskManager
        r=RiskManager();r.evaluate(100.,date='2024-01-01');r.evaluate(79.,date='2024-01-02')
        r.approve_resume(approved_by='reviewer')
        result=r.evaluate(79.,date='2024-01-03')
        self.assertAlmostEqual(result['control_drawdown'],-.21)
        self.assertEqual(result['level'],4)


class ForwardRecovery(unittest.TestCase):
    def test_recorded_recovery_matches_continuous_replay(self):
        from quantlab.forward import ForwardBook
        from quantlab.backtest.engine import Engine
        from quantlab.strategy.opportunity import Forecast
        ds=dataset(days=80);cfg=deepcopy(CONFIG);cfg.research.universe_mode='pit_liquidity'
        d=ds.calendar[64];end=ds.calendar[68]
        f=Forecast(str(d),20,pd.Series({c:.2 if c=='S000' else -.1 for c in ds.panels}),
                   'ok',[],str(ds.calendar[60]),'fixture')
        expected=Engine(cfg).run(ds,strategy='opportunity',forecasts={d:f},start=d,end=end)
        with tempfile.TemporaryDirectory() as tmp:
            book=ForwardBook(Path(tmp)/'book.sqlite3','fixture',cfg)
            ds.manifest['as_of']=str(d.date());book.step(ds,f,now=str(d.date())+' 18:01',allow_test_fixture=True)
            ds.manifest['as_of']=str(end.date())
            result=book.recover(ds,{},now=str(end.date())+' 18:02',allow_test_fixture=True)
            self.assertTrue(result[-1]['recovered_session'])
            self.assertAlmostEqual(result[-1]['account']['nav'],expected['account'][-1]['nav'])
            self.assertEqual(result[-1]['account']['quantities'],expected['account'][-1]['quantities'])
            self.assertEqual(book.reconcile([])['status'],'no_actual_fills')
