from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from fixtures_v2 import dataset
from quantlab.backtest.ledger import Ledger,Lot
from quantlab.data.contracts import dump_dataset,load_dataset
from quantlab.data.market import panels_for_research


class CategoricalActionDates(unittest.TestCase):
    def test_storage_categories_do_not_change_validation_features_or_cashflows(self):
        base=dataset(days=30,stocks=2)
        record,ex,pay=base.calendar[5:8]
        date_fields=('record_date','ex_date','pay_date','share_date','available_at')
        events=pd.DataFrame([dict(action_id='D1',instrument_id='S000',
            record_date=str(record.date()),ex_date=str(ex.date()),pay_date=str(pay.date()),share_date=str(pay.date()),
            available_at=str(base.calendar[0].date())+' 18:00:00',cash_per_share=1.,share_ratio=0.,
            source='fixture',revision='1',quality_status='ok',tax_policy='net_fixed')])
        encoded=events.copy()
        for column in date_fields:encoded[column]=pd.Categorical(encoded[column],ordered=False)
        plain=replace(base,actions=events)
        categorical=replace(base,actions=encoded)
        self.assertEqual(categorical.validate(),plain.validate())
        self.assertTrue(all(isinstance(encoded[c].dtype,pd.CategoricalDtype) for c in date_fields))
        for a,b in zip(panels_for_research(plain),panels_for_research(categorical)):
            pd.testing.assert_frame_equal(a,b)
        accounts=[]
        for ds in (plain,categorical):
            account=Ledger(1000.,slippage=0)
            account.cash=0.;account.lots={'S000':[Lot(100,0,10.,0)]}
            account.close(record,5,{'S000':10.})
            account.corporate_actions(ex,6,ds.actions,ds.calendar)
            account.corporate_actions(pay,7,ds.actions,ds.calendar)
            accounts.append(account)
            self.assertEqual(account.cash,100.)
            self.assertEqual(account.quantity('S000'),100)
            self.assertEqual(account.nav(),1000.)
        self.assertEqual(accounts[0].events,accounts[1].events)
        # Exercise the actual dictionary-encoded Parquet loading path too.
        with tempfile.TemporaryDirectory() as temp:
            plain.actions=encoded
            manifest=dump_dataset(plain,temp)
            loaded=load_dataset(manifest)
            self.assertEqual(loaded.validate(),[])
            self.assertEqual(loaded.actions.available_at.iloc[0],events.available_at.iloc[0])


if __name__=='__main__':unittest.main()
