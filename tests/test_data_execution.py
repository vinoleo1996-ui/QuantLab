import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from fixtures_v2 import dataset
from quantlab.config import CONFIG
from quantlab.data.store import DataStore
from quantlab.data.sources import DataFeed
from quantlab.backtest.engine import Engine, prepare
from quantlab.data.market import panels_for_research
from quantlab.strategy.decision import FactorDecisionService


class DataExecution(unittest.TestCase):
    def test_benchmark_calls_index_adapter_and_persists_identity(self):
        class Adapter:
            def stock_zh_index_daily_em(self, symbol):
                assert symbol == 'sh000300'
                return pd.DataFrame({'date':['2024-01-01','2024-01-02'],'close':[3500.,3510.]})
        with tempfile.TemporaryDirectory() as d:
            feed=DataFeed(DataStore(Path(d)))
            with patch.object(feed,'_ak',return_value=Adapter()), patch.object(feed,'_a_daily',side_effect=AssertionError('stock endpoint forbidden')):
                bench=feed.fetch_benchmark('a')
                self.assertEqual(bench.attrs['asset_type'],'index')
                self.assertEqual(bench.attrs['instrument_id'],'XSHG.INDEX.000300')

    def test_cached_daily_is_extended_and_not_called_success_on_stale(self):
        class Adapter:
            def stock_zh_a_hist(self, **kwargs):
                return pd.DataFrame({'日期':['2024-01-03'],'开盘':[12.],'收盘':[12.],'成交额':[1e8]})
        with tempfile.TemporaryDirectory() as d:
            store=DataStore(Path(d))
            old=pd.DataFrame({'open':[10.],'close':[10.],'amount':[1e8]},index=pd.to_datetime(['2024-01-01']))
            old.attrs.update(source='akshare',price_basis='raw')
            store.save(old,'a','daily_raw_A')
            feed=DataFeed(store)
            with patch.object(feed,'_ak',return_value=Adapter()),patch.object(feed,'_sleep'):
                result=feed.fetch_daily('a','A',expected_session='2024-01-03')
            self.assertEqual(len(result),2)
            self.assertEqual(result.attrs['freshness'],'current')
            with patch.object(feed,'_ak',side_effect=RuntimeError('offline')), patch.object(feed,'_baostock_daily',return_value=None), patch.object(feed,'_east_daily',return_value=None), patch.object(feed,'_sina_daily',return_value=None):
                result=feed.fetch_daily('a','A',expected_session='2024-01-04')
            self.assertEqual(result.attrs['freshness'],'stale')
            self.assertEqual(result.attrs['update_error'],'all_sources_failed')

    def test_cash_slippage_participation_and_order_floor_change_fills(self):
        ds=dataset();cfg=deepcopy(CONFIG);cfg.start_date=str(ds.calendar[20].date())
        schedule={ds.calendar[21]:{'S000':.05}}
        base=Engine(cfg).run(ds,strategy='schedule',target_schedule=schedule)
        q=base['fills'][0]['quantity']
        more=deepcopy(cfg);more.capital.init_cash*=2
        self.assertGreater(Engine(more).run(ds,strategy='schedule',target_schedule=schedule)['fills'][0]['quantity'],q)
        small=deepcopy(cfg);small.research.participation=1e-5
        limited=Engine(small).run(ds,strategy='schedule',target_schedule=schedule)
        self.assertLess(sum(x['quantity'] for x in limited['fills'] if x['side']=='buy'),q)
        floor=deepcopy(cfg);floor.cost.min_trade_value=1e9
        self.assertFalse(Engine(floor).run(ds,strategy='schedule',target_schedule=schedule)['fills'])
        expensive=deepcopy(cfg);expensive.cost.a_slippage=.01
        changed=Engine(expensive).run(ds,strategy='schedule',target_schedule=schedule)
        self.assertGreater(changed['fills'][0]['price'],base['fills'][0]['price'])

    def test_factor_prefix_invariance(self):
        ds=dataset(days=340,stocks=12)
        close,amount,turnover=panels_for_research(ds)
        full=FactorDecisionService(close,amount,turnover,ds.calendar)
        date=close.index[310]
        prefix=FactorDecisionService(close.loc[:date],amount.loc[:date],turnover.loc[:date],ds.calendar)
        pd.testing.assert_series_equal(full.score(date).scores,prefix.score(date).scores)


if __name__=='__main__':unittest.main()
