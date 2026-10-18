import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import _legacy_p5_helpers as S
import p5_snapshot as production


class Dataset:
    def __init__(self, panels):
        self.panels = panels


def test_settle_orders_conserves_cash_and_shares():
    orders = [
        {'code': 'sh.600000', 'side': 'buy', 'shares': 100},
        {'code': 'sh.600001', 'side': 'sell', 'shares': 50},
    ]
    cash, holdings, fills, unfilled = S.settle_orders(
        orders, {'sh.600000': 10.0, 'sh.600001': 20.0}, 10_000.0, {'sh.600001': 100})
    buy_value = 100 * 10.0 * (1 + S.SLIPPAGE)
    sell_value = 50 * 20.0 * (1 - S.SLIPPAGE)
    expected = 10_000.0 - buy_value - S.settlement_fee(buy_value, 'buy') \
        + sell_value - S.settlement_fee(sell_value, 'sell')
    assert cash == pytest.approx(expected)
    assert holdings == {'sh.600000': 100, 'sh.600001': 50}
    assert len(fills) == 2 and not unfilled


def test_settle_orders_rejects_negative_cash_and_missing_holding():
    cash, holdings, fills, unfilled = S.settle_orders(
        [{'code': 'sh.600000', 'side': 'buy', 'shares': 1000},
         {'code': 'sh.600001', 'side': 'sell', 'shares': 100}],
        {'sh.600000': 10.0, 'sh.600001': 20.0}, 100.0, {})
    assert cash == 100.0 and holdings == {}
    assert {row['reason'] for row in unfilled} == {'cash_limit', 'no_holding'}


def test_mark_snapshot_nav_exposures_and_industry():
    day = pd.Timestamp('2026-09-30')
    panel = pd.DataFrame({'close': [10.0]}, index=[day])
    dataset = Dataset({'sh.600000': panel})
    etf_close = pd.DataFrame({'sh518880': [4.0], 'sh511260': [5.0], 'sh513100': [1.0]}, index=[day])
    scores = pd.DataFrame([{'date': day, 'code': 'sh.600000', 'industry': '银行'}])
    mark = S.mark_snapshot({'sh.600000': 100, 'sh518880': 1000}, 5_000.0,
                           dataset, etf_close, day, scores)
    assert mark['nav'] == pytest.approx(10_000.0)
    assert mark['stock_exposure'] == pytest.approx(0.10)
    assert mark['etf_exposure'] == pytest.approx(0.40)
    assert mark['cash_weight'] == pytest.approx(0.50)
    assert mark['industry_weights'] == {'银行': pytest.approx(0.10)}
    assert mark['stock_names'] == 1


def test_stock_targets_reports_below_one_lot():
    plan = {'sh.600000': 0.97, 'sh.600001': 0.0}
    targets, skipped = S.stock_targets(plan, 800_000.0, {'sh.600000': 6000.0, 'sh.600001': 10.0}, {})
    assert targets['sh.600000'] == 0
    assert any(row['reason'] == 'below_one_lot' for row in skipped)


def test_etf_targets_pauses_nasdaq_and_splits_overflow():
    marks = {'gold': 4.0, 'treasury': 5.0, 'nasdaq': 1.0}
    shares, meta = S.etf_targets(800_000.0, marks, {}, premium_pause=True)
    assert meta['paused'] is True
    assert shares['nasdaq'] == 0
    assert meta['target_value']['nasdaq'] == 0
    assert meta['target_value']['gold'] == pytest.approx(0.375 * 0.35 * 800_000.0 + 0.25 * 0.35 * 800_000.0 / 2)


def test_rebalance_orders_are_whole_lots_and_within_cash():
    plan = {'sh.600000': 0.485, 'sh.600001': 0.485}
    marks = {'sh.600000': 10.0, 'sh.600001': 20.0}
    etf = {'gold': 4.0, 'treasury': 5.0, 'nasdaq': 1.0}
    orders, skipped, meta, cash_after = S.build_rebalance_orders(
        plan, {}, 800_000.0, 800_000.0, marks, etf, {'sh.600000': 1e9, 'sh.600001': 1e9},
        rebalance_stocks=True, rebalance_etf=True, premium_pause=False)
    assert cash_after >= 0
    buys = [o for o in orders if o['side'] == 'buy']
    assert buys and all(o['shares'] % 100 == 0 for o in buys)
    assert sum(o['value'] for o in buys) < 800_000.0
    assert {o['leg'] for o in buys} == {'stock', 'etf'}


def test_six_month_check_triggers_below_minus_three_percent():
    history = [{'date': '2026-03-31', 'nav': 100.0}, {'date': '2026-09-30', 'nav': 100.0}]
    alloc = pd.Series([100.0, 104.0], index=pd.to_datetime(['2026-03-31', '2026-09-30']))
    check = production.six_month_check(history, alloc, pd.Timestamp('2026-09-30'), 100.0)
    assert check['status'] == 'ok'
    assert check['relative_return'] == pytest.approx(-0.04)
    assert check['stop_check_triggered'] is True


def test_six_month_check_requires_history():
    check = production.six_month_check([], pd.Series(dtype=float), pd.Timestamp('2026-09-30'), 100.0)
    assert check['status'] == 'insufficient_history'
