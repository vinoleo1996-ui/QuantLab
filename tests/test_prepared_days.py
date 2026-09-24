import unittest

import pandas as pd

from fixtures_v2 import dataset
from quantlab.backtest.engine import _PreparedDays
from quantlab.config import Config


class PreparedRuleHistoryTests(unittest.TestCase):
    def test_lifecycle_revisions_preserve_rules_but_future_fee_changes_do_not_leak(self):
        for changed_fee in (False,True):
            with self.subTest(changed_fee=changed_fee):
                ds=dataset(days=10,stocks=2)
                original=ds.instruments.iloc[0].copy()
                later=original.copy()
                later['effective_at']=str(ds.calendar[5].date())
                later['available_at']=str(ds.calendar[5].date())+' 00:00:00'
                later['active']=False
                if changed_fee:later['commission']=.002
                ds.instruments=pd.concat([ds.instruments,pd.DataFrame([later])],ignore_index=True)
                owner=_PreparedDays(ds,Config(),pd.DataFrame(),ds.calendar)
                code=original.instrument_id
                self.assertEqual(owner.day(ds.calendar[4])['rules'][code].commission,original.commission)
                self.assertEqual(owner.day(ds.calendar[6])['rules'][code].commission,
                                 .002 if changed_fee else original.commission)
