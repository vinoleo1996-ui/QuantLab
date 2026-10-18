import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import p5_base_lot_replay as L
from quantlab import p4_alloc as A
from quantlab.backtest.ledger import Ledger
from quantlab.data.contracts import InstrumentRules


def rule(lot=100):
    return InstrumentRules(instrument_id='x', asset_type='stock', buy_lot=lot, sell_lot=lot,
                           sell_delay=1, price_tick=0.01, commission=0.0003,
                           min_commission=5.0, limit_ratio=0.10)


def test_high_priced_name_that_cannot_fill_one_lot_is_reported():
    ledger = Ledger(cash=10_400.0)
    ledger.marks = {'600000': 1700.0}
    ledger.queue_targets({'600000': 0.97}, '2026-01-05', 'rebalance',
                         {'600000': rule()})
    assert ledger.orders == []
    events = [e for e in ledger.events if e.get('reason') == 'below_one_lot']
    assert len(events) == 1
    assert events[0]['instrument_id'] == '600000'
    assert events[0]['type'] == 'order_not_submitted'


def test_affordable_name_buys_whole_board_lots():
    ledger = Ledger(cash=10_400.0)
    ledger.marks = {'600000': 10.0}
    ledger.queue_targets({'600000': 0.97}, '2026-01-05', 'rebalance',
                         {'600000': rule()})
    assert len(ledger.orders) == 1
    order = ledger.orders[0]
    assert order.side == 'buy'
    assert order.quantity == 1000
    assert order.quantity % 100 == 0


def _price_bundle(prices, sessions, missing_day=None, missing_name=None):
    raw = prices.reindex(sessions).copy()
    if missing_day is not None:
        raw.loc[pd.Timestamp(missing_day), missing_name] = np.nan
    mark = raw.ffill()
    missing = raw.isna()
    missing_dates = [dict(code=L.ETF_CODES[name], name=name, date=str(pd.Timestamp(day).date()))
                     for day in sessions for name in raw.columns if bool(missing.at[day, name])]
    return dict(raw=raw, mark=mark, missing=missing,
                missing_days=int(missing.any(axis=1).sum()), missing_dates=missing_dates)


def test_g0_sleeve_trades_only_quarter_ends_in_whole_lots(monkeypatch):
    sessions = pd.bdate_range('2018-01-02', '2019-12-31')
    prices = pd.DataFrame({'gold': np.linspace(3.0, 3.4, len(sessions)),
                           'treasury': np.linspace(4.0, 4.3, len(sessions)),
                           'nasdaq': np.linspace(1.0, 1.2, len(sessions))}, index=sessions)
    premium = pd.DataFrame(0.0, index=sessions, columns=['nasdaq', 'sp500'])
    premium.attrs.update(basis='adjusted', masked={}, fallback_days={})
    monkeypatch.setattr(L, 'etf_prices', lambda s: _price_bundle(prices, s))
    monkeypatch.setattr(L.R, 'qdii_premium_frame', lambda s, basis='adjusted': premium.reindex(s))
    nav, meta = L.run_g0_lot_sleeve(sessions, 280_000.0, cost_mult=1.0)
    assert len(nav) == len(sessions)
    assert meta['final_cash'] >= 0
    assert meta['trades'] > 0
    assert not meta['blocked']
    ends = set(A.quarter_ends(sessions))
    assert {pd.Timestamp(t['date']) for t in meta['trade_log']} <= ends
    assert all(t['shares'] % 100 == 0 for t in meta['trade_log'])
    assert all(v % 100 == 0 for v in meta['final_shares'].values())


def test_etf_missing_non_rebalance_marks_stale_without_trade(monkeypatch):
    sessions = pd.bdate_range('2018-01-02', '2019-12-31')
    prices = pd.DataFrame({'gold': np.linspace(3.0, 3.4, len(sessions)),
                           'treasury': np.linspace(4.0, 4.3, len(sessions)),
                           'nasdaq': np.linspace(1.0, 1.2, len(sessions))}, index=sessions)
    ends = set(A.quarter_ends(sessions))
    missing_day = next(day for day in sessions if day not in ends)
    premium = pd.DataFrame(0.0, index=sessions, columns=['nasdaq', 'sp500'])
    premium.attrs.update(basis='adjusted', masked={}, fallback_days={})
    monkeypatch.setattr(L, 'etf_prices',
                        lambda s: _price_bundle(prices, s, missing_day, 'treasury'))
    monkeypatch.setattr(L.R, 'qdii_premium_frame', lambda s, basis='adjusted': premium.reindex(s))
    _, meta = L.run_g0_lot_sleeve(sessions, 280_000.0, cost_mult=1.0)
    assert not meta['blocked']
    assert meta['missing_on_rebalance'] is False
    assert meta['stale_mark_days'] == [str(missing_day.date())]
    assert all(pd.Timestamp(t['date']) != missing_day for t in meta['trade_log'])


def test_etf_missing_on_rebalance_blocks_without_ffill_execution(monkeypatch):
    sessions = pd.bdate_range('2018-01-02', '2019-12-31')
    prices = pd.DataFrame({'gold': np.linspace(3.0, 3.4, len(sessions)),
                           'treasury': np.linspace(4.0, 4.3, len(sessions)),
                           'nasdaq': np.linspace(1.0, 1.2, len(sessions))}, index=sessions)
    rebalance_day = pd.Timestamp(sorted(A.quarter_ends(sessions))[0])
    premium = pd.DataFrame(0.0, index=sessions, columns=['nasdaq', 'sp500'])
    premium.attrs.update(basis='adjusted', masked={}, fallback_days={})
    monkeypatch.setattr(L, 'etf_prices',
                        lambda s: _price_bundle(prices, s, rebalance_day, 'gold'))
    monkeypatch.setattr(L.R, 'qdii_premium_frame', lambda s, basis='adjusted': premium.reindex(s))
    _, meta = L.run_g0_lot_sleeve(sessions, 280_000.0, cost_mult=1.0)
    assert meta['blocked'] is True
    assert meta['missing_on_rebalance'] is True
    assert meta['blocked_dates'] == [str(rebalance_day.date())]
    assert meta['stale_mark_days'] == []
    assert all(pd.Timestamp(t['date']) != rebalance_day for t in meta['trade_log'])


def test_known_etf_missing_days_are_raw_gaps_not_ffilled_execution():
    sessions = pd.DatetimeIndex(['2018-08-16', '2018-08-17', '2018-08-20',
                                 '2022-01-12', '2022-01-13', '2022-01-14'])
    bundle = L.etf_prices(sessions)
    assert np.isnan(bundle['raw'].loc['2018-08-17', 'treasury'])
    assert np.isnan(bundle['raw'].loc['2022-01-13', 'nasdaq'])
    assert bundle['mark'].loc['2018-08-17', 'treasury'] == bundle['raw'].loc['2018-08-16', 'treasury']
    assert bundle['mark'].loc['2022-01-13', 'nasdaq'] == bundle['raw'].loc['2022-01-12', 'nasdaq']
    missing = {(row['date'], row['name']) for row in bundle['missing_dates']}
    assert ('2018-08-17', 'treasury') in missing
    assert ('2022-01-13', 'nasdaq') in missing


def test_blend_reports_exposure_identity_and_quarterly_rule():
    sessions = pd.bdate_range('2018-01-02', '2019-12-31')
    rng = np.random.default_rng(7)
    stock_nav = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.01, len(sessions))), index=sessions)
    etf_nav = pd.Series(np.cumprod(1 + rng.normal(0.0001, 0.004, len(sessions))), index=sessions)
    stock_exposure = pd.Series(0.92, index=sessions)
    etf_cash = pd.Series(0.04, index=sessions)
    nav, trace, meta = L.run_one(stock_nav, stock_exposure, etf_nav, etf_cash, cost=1.0)
    assert len(nav) == len(sessions)
    assert trace['rebalance'] == 'quarter'
    assert meta['stock_exposure_mean'] == pytest.approx(0.92 * trace['weights']['equity'].mean(),
                                                        abs=1e-9)
    assert meta['stock_exposure_mean'] + meta['etf_exposure_mean'] + meta['cash_mean'] == pytest.approx(1.0)
