import unittest
import pandas as pd
from quantlab.backtest.tax import dividend_tax_rate
from quantlab.backtest.ledger import Ledger,Lot
from test_ledger import RULE,bar


class DividendTax(unittest.TestCase):
    def test_calendar_month_and_year_boundaries(self):
        self.assertEqual(dividend_tax_rate('2024-01-31','2024-02-29','2024-02-10'),.2)
        self.assertEqual(dividend_tax_rate('2024-01-31','2024-03-01','2024-02-10'),.1)
        self.assertEqual(dividend_tax_rate('2023-01-01','2024-01-02','2023-02-01'),0)
        self.assertEqual(dividend_tax_rate('2012-01-01','2014-01-02','2013-02-01'),.05)

    def test_selling_collects_dividend_tax_without_creating_value(self):
        a=Ledger(100000,slippage=0,cash_buffer=0,participation=1)
        a.cash=90000;a.marks={'A':10.};a.lots={'A':[Lot(1000,0,10,0,'2024-01-01')]}
        a.close('2024-01-02',1,{'A':10.})
        actions=pd.DataFrame([dict(action_id='D1',instrument_id='A',record_date='2024-01-02',ex_date='2024-01-03',
          pay_date='2024-01-03',share_date='2024-01-03',available_at='2024-01-01',cash_per_share=1.,share_ratio=0.,tax_policy='gross_cn_individual')])
        a.corporate_actions('2024-01-03',2,actions,pd.bdate_range('2024-01-01',periods=10))
        a.close('2024-01-03',2,{'A':9.});a.queue_targets({},'2024-01-03','exit',{'A':RULE})
        a.execute('2024-01-04',3,{'A':bar('2024-01-04',9,ref=9)},{'A':RULE},{'A':1e8})
        self.assertEqual(a.fills[-1]['dividend_tax'],200.)
        self.assertAlmostEqual(a.nav(),100000-a.fills[-1]['fees'])
        self.assertEqual(a.quantity('A'),0)

if __name__=='__main__':unittest.main()

class ExecutionBoundaries(unittest.TestCase):
    def test_slippage_respects_computed_exchange_limit(self):
        a=Ledger(100000,slippage=.03,cash_buffer=0,participation=1)
        a.mark('2024-01-01',{'A':10.})
        a.queue_targets({'A':.5},'2024-01-01','test',{'A':RULE})
        b=bar('2024-01-02',10.98);b.pop('limit_up');b.pop('limit_down')
        a.execute('2024-01-02',1,{'A':b},{'A':RULE},{'A':1e8})
        self.assertFalse(a.fills)
        self.assertEqual(a.orders[0].reject_reason,'slippage_reaches_price_limit')
