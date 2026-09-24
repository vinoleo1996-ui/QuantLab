import unittest
from copy import deepcopy
import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.config import CONFIG
from quantlab.backtest.engine import Engine
from quantlab.strategy.legacy import allocate_legacy, legacy_config
from quantlab.strategy.crowding import RiskSnapshot


class LegacyReplay(unittest.TestCase):
    def test_maximum_capital_and_volatility_ceiling_are_both_real(self):
        cfg=deepcopy(CONFIG);codes=[str(i) for i in range(20)]
        returns=pd.DataFrame(np.random.default_rng(13).normal(0,.005,(120,20)),columns=codes)
        groups=RiskSnapshot('2024-01-01',{c:c for c in codes},pd.Series(0.,index=codes),pd.Series(0.,index=codes),[])
        exposures=pd.DataFrame({c:{str(i%4):1.} for i,c in enumerate(codes)}).fillna(0.)
        w,status,detail=allocate_legacy(returns,cfg,selected=codes,exposures=exposures,risk_snapshot=groups)
        self.assertEqual(status,'ok',detail)
        self.assertAlmostEqual(w.sum(),.98,places=6)
        risky,status,detail=allocate_legacy(returns*40,cfg,selected=codes,exposures=exposures,risk_snapshot=groups)
        self.assertEqual(status,'ok',detail)
        self.assertLess(risky.sum(),.4)
        self.assertLessEqual(detail['target_vol'],cfg.portfolio.target_vol+1e-6)
        smaller=deepcopy(cfg);smaller.research.max_stock_weight=.02
        limited,status,detail=allocate_legacy(returns,smaller,selected=codes,exposures=exposures,risk_snapshot=groups)
        self.assertAlmostEqual(limited.sum(),.4,places=6)

    def test_feature_changes_reach_orders_and_future_quotes_do_not_rewrite_history(self):
        from quantlab.backtest.engine import prepare
        from quantlab.strategy.decision import review_dates
        ds=dataset(days=340,stocks=28)
        cfg=legacy_config(CONFIG);cfg.start_date=str(ds.calendar[280].date())
        cfg.capital.a_share_positions=8
        day=next(d for d in review_dates(ds.calendar) if d>=pd.Timestamp(cfg.start_date))
        a=Engine(cfg).run(ds,start=day,end=ds.calendar[ds.calendar.get_loc(day)+2])
        changed=deepcopy(ds)
        # Exchange two entire causal price/amount/turnover histories. This changes
        # actual factor inputs, not a mock score or the forecast distribution.
        selected=set(a['trades'][0]['weights'])
        inside=next(c for c in sorted(selected) if a['trades'][0]['weights'][c]>.001)
        outside=next(c for c in sorted(ds.panels) if c not in selected)
        changed.panels[inside],changed.panels[outside]=changed.panels[outside],changed.panels[inside]
        b=Engine(cfg).run(changed,start=day,end=ds.calendar[ds.calendar.get_loc(day)+2])
        self.assertNotEqual(a['signal_trace'][0]['scores'],b['signal_trace'][0]['scores'])
        self.assertNotEqual(a['trades'][0]['weights'],b['trades'][0]['weights'])
        self.assertNotEqual([f['instrument_id'] for f in a['fills']],[f['instrument_id'] for f in b['fills']])
        future=deepcopy(ds)
        for p in future.panels.values():p.loc[p.index>pd.Timestamp(a['as_of']),'close']*=2
        c=Engine(cfg).run(future,start=day,end=a['as_of'])
        self.assertEqual(a['fills'],c['fills'])
        self.assertEqual(a['trades'],c['trades'])
        self.assertTrue(a['capital_diagnostics']['decisions'][0]['filled_buy_notional']>0)

    def test_scoped_exception_does_not_disable_mandate(self):
        ds=dataset();cfg=deepcopy(CONFIG)
        with self.assertRaisesRegex(ValueError,'scoped_free_research'):
            Engine(cfg).run(ds,industry_policy='omit_for_diagnostic')
        ds.manifest['source']='free_sample';cfg.research.max_stock_weight=.2
        with self.assertRaisesRegex(ValueError,'mandate_security_limit'):
            Engine(cfg).run(ds,data_mode='free_research',industry_policy='omit_for_diagnostic')

    def test_old_strategy_runs_without_forecasts_and_industry_policy_reaches_fills(self):
        ds = dataset(days=340, stocks=32)
        ds.manifest['source'] = 'free_sample'
        ds.instruments['sector'] = 'UNKNOWN'
        cfg = deepcopy(CONFIG)
        cfg.start_date = str(ds.calendar[280].date())
        cfg.research.feature_profile = 'price_volume'
        cfg.research.universe_mode = 'pit_liquidity'
        args = dict(strategy='legacy_repaired', data_mode='free_research')
        capped = Engine(cfg).run(ds, **args)
        open_group = Engine(cfg).run(ds, industry_policy='omit_for_diagnostic', **args)
        self.assertTrue(capped['fills'])
        self.assertTrue(open_group['fills'])
        self.assertLessEqual(max(sum(t['weights'].values()) for t in capped['trades']), .250001)
        self.assertGreater(max(sum(a['weights'].values()) for a in open_group['account']), .50)
        self.assertEqual(open_group['industry_policy'], 'omit_for_diagnostic')
        self.assertNotEqual(capped['run_id'], open_group['run_id'])
        self.assertFalse(open_group['promotion_approved'])
        self.assertTrue(all(o['industry_policy']=='omit_for_diagnostic' for o in open_group['orders']))
        self.assertTrue(all(pd.Timestamp(f['date'])>pd.Timestamp(f['signal_date']) for f in capped['fills']))
        self.assertTrue(all(max(a['weights'].values(), default=0)<=.08 for a in open_group['account']))


if __name__ == '__main__':
    unittest.main()
