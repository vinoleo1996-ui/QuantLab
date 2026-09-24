import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
import pandas as pd
from fixtures_v2 import dataset
from quantlab.forward import ForwardBook
from quantlab.strategy.opportunity import Forecast
from quantlab.config import CONFIG


class ForwardJournal(unittest.TestCase):
    def test_legacy_policy_book_runs_without_forecast_and_rejects_policy_change(self):
        ds=dataset(days=80);cfg=deepcopy(CONFIG);cfg.research.universe_mode='pit_liquidity'
        d=ds.calendar[64];ds.manifest['as_of']=str(d.date())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.sqlite3'
            book=ForwardBook(path,'legacy-frozen',cfg,strategy='legacy_repaired',risk_policy='P2')
            result=book.step(ds,now=str(d.date())+' 18:01',allow_test_fixture=True)
            self.assertEqual(result['risk_budget']['policy_id'],'P2')
            self.assertFalse(result['executable'])
            with self.assertRaisesRegex(ValueError,'forward_policy_changed'):
                ForwardBook(path,'legacy-frozen',cfg,strategy='legacy_repaired',risk_policy='P1')

    def test_restart_idempotency_revision_and_tamper(self):
        ds=dataset(days=80);cfg=deepcopy(CONFIG);cfg.research.universe_mode='pit_liquidity'
        d=ds.calendar[64];ds.manifest['as_of']=str(d.date())
        f=Forecast(str(d),20,pd.Series({c:.2 if c=='S000' else -.1 for c in ds.panels}),'ok',[],str(ds.calendar[60]),'fixture')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'forward.sqlite3';a=ForwardBook(path,'fixture',cfg)
            x=a.step(ds,f,now=str(d.date())+' 18:01',allow_test_fixture=True)
            b=ForwardBook(path,'fixture',cfg)
            self.assertEqual(x,b.step(ds,f,now=str(d.date())+' 18:02',allow_test_fixture=True))
            next_day=ds.calendar[65];ds.manifest['as_of']=str(next_day.date())
            y=b.step(ds,now=str(next_day.date())+' 18:01',allow_test_fixture=True)
            self.assertEqual(b.status()['sessions'],2)
            self.assertTrue(y['fills'])
            actual=dict(fill_id='F1',order_id=y['fills'][0]['order_id'],date=y['fills'][0]['date'],instrument_id='S000',side='buy',quantity=y['fills'][0]['quantity'],price=y['fills'][0]['price'],fees=y['fills'][0]['fees'])
            self.assertEqual(b.reconcile([actual])['status'],'reconciled_fills_only')
            self.assertEqual(len(b.reconcile([actual])['comparisons']),1)
            with b.connect() as db:db.execute("UPDATE sessions SET payload='{}' WHERE date=?",(str(d.date()),))
            with self.assertRaisesRegex(ValueError,'integrity_failed'):b.verify()

if __name__=='__main__':unittest.main()
