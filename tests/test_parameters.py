import unittest
from copy import deepcopy
import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.config import CONFIG
from quantlab.config_validation import validate_config
from quantlab.strategy.risk import RiskManager
from quantlab.strategy.decision import FactorDecisionService, review_dates
from quantlab.factors.engine import FactorEngine
from quantlab.data.market import universe_at, panels_for_research
from quantlab.backtest.ledger import Ledger
from quantlab.research import walk_forward_splits, promotion_gate, block_bootstrap_ci


class Parameters(unittest.TestCase):
    def test_inactive_settings_cannot_silently_change(self):
        cfg = deepcopy(CONFIG); cfg.research.ridge_alpha += 1
        with self.assertRaisesRegex(ValueError,"inactive_config_field"): validate_config(cfg)
        cfg = deepcopy(CONFIG);cfg.factors.neutralize_by_sector=True
        with self.assertRaisesRegex(ValueError,"inactive_config_field"): validate_config(cfg)

    def test_risk_threshold_changes_action(self):
        cfg = deepcopy(CONFIG)
        a = RiskManager(cfg.risk);a.evaluate(100)
        self.assertEqual(a.evaluate(89)["level"],1)
        cfg.risk.drawdown_alert=.12
        b = RiskManager(cfg.risk);b.evaluate(100)
        self.assertEqual(b.evaluate(89)["level"],0)

    def test_factor_window_changes_scores(self):
        rng=np.random.default_rng(3)
        dates=pd.bdate_range('2023-01-01',periods=330)
        close=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0,.01,(330,12)),axis=0)),index=dates)
        amount=close*1e6;turnover=close/100
        c=deepcopy(CONFIG.factors)
        first=FactorEngine('a',cfg=c).compute(close,amount,turnover)
        c.momentum_short=35
        second=FactorEngine('a',cfg=c).compute(close,amount,turnover)
        self.assertGreater((first['momentum_3m'].iloc[-1]-second['momentum_3m'].iloc[-1]).abs().sum(),0)

    def test_calendar_frequency_changes_decisions(self):
        dates=pd.bdate_range('2023-01-01',periods=90)
        self.assertGreater(len(review_dates(dates,'weekly')),len(review_dates(dates,'monthly')))
        self.assertLess(len(review_dates(dates,'every_n_weeks',4)),len(review_dates(dates,'every_n_weeks',2)))

    def test_liquidity_threshold_changes_universe(self):
        ds=dataset();cfg=deepcopy(CONFIG);cfg.universe.a_min_listing_days=1
        _,amount,_=panels_for_research(ds)
        a,_=universe_at(ds,ds.calendar[50],amount,cfg)
        cfg.universe.a_min_avg_amount_wan=20000
        b,_=universe_at(ds,ds.calendar[50],amount,cfg)
        self.assertTrue(a);self.assertFalse(b)

    def test_purge_boundary_and_insufficient_power(self):
        dates=pd.bdate_range('2018-01-01','2026-01-01')
        folds=walk_forward_splits(dates)
        self.assertGreaterEqual(len(folds),3)
        for f in folds:
            train_end=dates.get_loc(f['train'][-1]);val_start=dates.get_loc(f['validation'][0])
            self.assertGreater(val_start-train_end,CONFIG.research.horizon)
        self.assertFalse(promotion_gate({})['promotion_approved'])
        ci=block_bootstrap_ci(np.tile([.01,-.01],100),samples=100)
        self.assertLessEqual(ci['lower'],0)


if __name__=='__main__':unittest.main()
