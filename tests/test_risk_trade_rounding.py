import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from quantlab.backtest.engine import Engine
from quantlab.backtest.ledger import Ledger, Lot
from quantlab.strategy.risk_trades import round_risk_sales
from test_ledger import RULE
from test_dynamic_engine import fixture, context


class RiskTradeRoundingTests(unittest.TestCase):
    def test_unknown_rules_remain_fixed_exposure_and_nan_covariance_is_rejected(self):
        matrix=pd.DataFrame([[1.,1.]],index=['gross'],columns=['A','B'])
        shares,proof=round_risk_sales({'A':100,'B':900},{'A':1.,'B':1.},1000.,
            {'A':.05,'B':.45},{'A':RULE},linear_limits=[(matrix,pd.Series({'gross':.5}))])
        self.assertEqual(shares,{'A':0,'B':900})
        self.assertEqual(proof['projected_gross'],.9)
        self.assertFalse(proof['projected_constraints_passed'])
        self.assertEqual(proof['unresolved_fixed_holdings'],['B'])
        for covariance in (pd.DataFrame([[np.nan]],index=['A'],columns=['A']),
                           pd.DataFrame([[-.1]],index=['A'],columns=['A'])):
            with self.assertRaisesRegex(ValueError,'invalid_risk_rounding_covariance'):
                round_risk_sales({'A':100},{'A':10.},2000.,{'A':.5},{'A':RULE},
                                 covariance=covariance,volatility_limit=.01)

    def test_small_volatility_excess_needs_one_lot_not_one_per_stock(self):
        codes=[str(i) for i in range(20)]
        q={c:1500 for c in codes};marks={c:20. for c in codes}
        rules={c:replace(RULE,instrument_id=c) for c in codes}
        cov=pd.DataFrame((.180002/.96)**2,index=codes,columns=codes)
        target={c:.048*.18/.180002 for c in codes}
        shares,proof=round_risk_sales(q,marks,625000.,target,rules,covariance=cov,volatility_limit=.18)
        self.assertEqual(sum(q[c]-shares[c] for c in codes),100)
        self.assertLessEqual(proof['projected_annualized_vol'],.18)
        self.assertEqual(proof['additional_lots'],1)

    def test_group_and_gross_budgets_are_rechecked_after_rounding(self):
        codes=['A','B','C'];q={c:1000 for c in codes};marks={c:10. for c in codes}
        rules={c:replace(RULE,instrument_id=c) for c in codes}
        matrix=pd.DataFrame([[1.,1.,0.],[1.,1.,1.]],index=['sector','gross'],columns=codes)
        targets={'A':.199,'B':.199,'C':.2}
        shares,proof=round_risk_sales(q,marks,50000.,targets,rules,
            linear_limits=[(matrix,pd.Series({'sector':.398,'gross':.598}))])
        w=pd.Series(shares)*10/50000
        self.assertTrue((matrix@w<=pd.Series({'sector':.398,'gross':.598})+1e-10).all())
        self.assertEqual(sum(q[c]-shares[c] for c in codes),100)
        self.assertTrue(proof['projected_constraints_passed'])

    def test_exact_risk_quantity_preserves_bonus_odd_lot_and_pending_order(self):
        ledger=Ledger(100000.,min_trade_value=0.)
        ledger.cash=81500.;ledger.marks={'A':10.};ledger.lots={'A':[Lot(1850,0,10.,0)]}
        ledger.queue_targets({'A':.175},'2024-01-02','risk_exit',{'A':RULE},target_quantities={'A':1750})
        self.assertEqual([o.quantity for o in ledger.orders],[100])
        shares,_=round_risk_sales(ledger.quantities(),ledger.marks,ledger.nav(),{'A':.175},
                                 {'A':RULE},reserved={'A':100})
        ledger.queue_targets({'A':.175},'2024-01-03','risk_exit',{'A':RULE},target_quantities=shares)
        self.assertEqual([o.quantity for o in ledger.orders],[100])
        self.assertEqual(ledger.quantity('A'),1850)
        self.assertEqual(ledger.cash,81500.)

    def test_engine_small_risk_excess_submits_only_the_necessary_lot(self):
        ds,cfg=fixture();dates=ds.calendar
        rng=np.random.default_rng(781)
        shared=20*np.cumprod(1+rng.normal(0,.010,350))
        day=dates[322];price=shared[321]*(1+.09621543366490615)
        for panel in ds.panels.values():
            previous=np.r_[shared[0],shared[:-1]]
            panel['close']=shared;panel['open']=previous;panel['reference_price']=previous
            panel['limit_up'],panel['limit_down']=previous*1.1,previous*.9
            panel.loc[day,'close']=price
            panel.loc[dates[323],['open','close','reference_price','limit_up','limit_down']]=[price,price,price,price*1.1,price*.9]
        run=Engine(cfg).run(ds,risk_policy='P1',prepared=context(ds,cfg),start=dates[320],
            end=dates[323],review_override=[dates[320]],crowding=False)
        sold=[f for f in run['fills'] if f['side']=='sell' and f['signal_date']==str(day.date())]
        self.assertEqual(sum(f['quantity'] for f in sold),100)
        self.assertEqual(len(sold),1)
        trace=next(t for t in run['signal_trace'] if t['date']==str(day.date()))
        self.assertLessEqual(trace['risk_reduction']['lot_rounding']['projected_annualized_vol'],.18)
        self.assertTrue(all(s['cash']>=0 for s in run['account']))
