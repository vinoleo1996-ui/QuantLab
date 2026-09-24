"""Account-level policy checks; deterministic scores isolate risk and execution."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.backtest.engine import Engine, prepare
from quantlab.config import Config
from quantlab.strategy.decision import ScoreDecision
from quantlab.strategy.legacy import legacy_config


class FixedRanking:
    ic = pd.DataFrame()

    def score(self, date, eligible):
        codes = sorted(eligible)
        scores = pd.Series(np.arange(len(codes), 0, -1, dtype=float), index=codes)
        return ScoreDecision(str(date), 'ok', scores, trace={'scores': scores.to_dict()})


def fixture():
    ds = dataset(days=350, stocks=20)
    ds.instruments['sector'] = ds.instruments.instrument_id
    market_returns = np.r_[np.tile([.001, -.001], 150), np.full(50, .002)]
    ds.benchmark['close'] = 100*np.exp(np.cumsum(market_returns))
    cfg = Config()
    cfg.start_date = str(ds.calendar[320].date())
    return ds, cfg


def context(ds, cfg):
    prepared = prepare(ds, legacy_config(cfg))
    prepared['legacy_scorer'] = FixedRanking()
    return prepared


class DynamicEngineTests(unittest.TestCase):
    def test_natural_cash_weight_drift_does_not_sell_a_lot_from_every_stock(self):
        ds, cfg = fixture()
        dates = ds.calendar
        for panel in ds.panels.values():
            close = np.full(len(panel), 20.)
            close[321:] = 20.*1.01**np.arange(1, len(panel)-320)
            previous = np.r_[close[0], close[:-1]]
            panel['close'], panel['open'], panel['reference_price'] = close, previous, previous
            panel['limit_up'], panel['limit_down'] = previous*1.1, previous*.9
        start = dates[320]
        result = Engine(cfg).run(ds, risk_policy='P1', prepared=context(ds, cfg),
            start=start, end=dates[325], review_override=[start], crowding=False)
        self.assertTrue(result['fills'])
        self.assertTrue(all(row['level'] == 0 for row in result['risk_history']))
        self.assertFalse(any(fill['side']=='sell' for fill in result['fills']))
        bought = result['account'][1]['quantities']
        self.assertTrue(all(row['quantities']==bought for row in result['account'][1:]))
        self.assertLess(result['account'][-1]['cash']/result['account'][-1]['nav'], .02)
        self.assertGreater(result['risk_history'][-1]['cash_reserve_shortfall'], 0.)

    def test_orders_and_fills_carry_the_signal_budget_under_execution_delay(self):
        ds, cfg = fixture()
        day = ds.calendar[320]
        result = Engine(cfg).run(ds, risk_policy='P1', prepared=context(ds, cfg),
            start=day, end=ds.calendar[325], review_override=[day], execution_delay=1,
            crowding=False)
        self.assertTrue(result['fills'])
        for fill in result['fills']:
            self.assertEqual(fill['risk_policy'], 'P1')
            self.assertEqual(fill['risk_budget']['policy_id'], 'P1')
            self.assertGreater(pd.Timestamp(fill['date']), pd.Timestamp(fill['signal_date']))
        self.assertTrue(all(a['cash'] >= 0 for a in result['account']))

    def test_forward_append_updates_market_history_without_resetting_state(self):
        ds, cfg = fixture()
        start, split, end = ds.calendar[320], ds.calendar[328], ds.calendar[333]
        args = dict(risk_policy='P2', crowding=False, review_override=[start])
        direct = Engine(cfg).run(ds, prepared=context(ds, cfg), start=start, end=end, **args)
        prefix = deepcopy(ds)
        prefix.panels = {c: p.loc[:split].copy() for c, p in prefix.panels.items()}
        prefix.benchmark = prefix.benchmark.loc[:split].copy()
        first = Engine(cfg).run(prefix, prepared=context(prefix, cfg), start=start, end=split, **args)
        resumed = Engine(cfg).run(ds, prepared=context(ds, cfg), start=ds.calendar[329], end=end,
                                 initial_state=first['_runtime'], **args)
        pd.testing.assert_series_equal(direct['nav'], resumed['nav'])
        self.assertEqual(direct['fills'], resumed['fills'])
        expected = {pd.Timestamp(x['date']): x['risk_budget'] for x in direct['risk_history']}
        for row in resumed['risk_history']:
            self.assertEqual(row['risk_budget'], expected[pd.Timestamp(row['date'])])
        self.assertIs(resumed['_runtime']['risk'], resumed['_runtime']['controller'].risk_manager)

    def test_limit_down_exit_survives_25_percent_halt_without_fictional_cash(self):
        ds, cfg = fixture()
        dates = ds.calendar
        for panel in ds.panels.values():
            prior = panel.loc[dates[323], 'close']
            for i in range(324, 329):
                px = prior*.9
                panel.loc[dates[i], ['open', 'close', 'reference_price', 'limit_down', 'limit_up']] = [px, px, prior, px, prior*1.1]
                prior = px
        start = dates[320]
        result = Engine(cfg).run(ds, risk_policy='P1', prepared=context(ds, cfg),
            start=start, end=dates[328], review_override=[start, dates[327]], crowding=False)
        self.assertTrue(result['fills'])
        self.assertEqual(result['risk_history'][-1]['level'], 4)
        self.assertTrue(result['risk_history'][-1]['manual_review'])
        pending = [o for o in result['orders'] if o['side']=='sell' and o['reason']=='risk_exit'
                   and o['status'] in ('pending', 'partial')]
        self.assertTrue(pending)
        self.assertTrue(all(o['remaining'] > 0 for o in pending))
        self.assertFalse(any(f['side']=='buy' and pd.Timestamp(f['date'])>=dates[324]
                             for f in result['fills']))
        self.assertTrue(all(a['cash'] >= 0 for a in result['account']))
        self.assertLess(result['nav'].iloc[-1], .75)


if __name__ == '__main__':
    unittest.main()
