import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import _legacy_p5_helpers as L
import p5_snapshot as S
from quantlab.backtest.ledger import Ledger, Lot, trade_permission
from quantlab.data.contracts import InstrumentRules

RULE = InstrumentRules('sh.600000', 'stock', 100, 100, 1, 0.01, 0.0003, 5.0, 0.10)


def bar(day, price=10.0, status='trading', ref=10.0):
    return {'open': price, 'close': price, 'open_status': status,
            'open_status_available_at': f'{day} 09:25:00',
            'reference_price': ref, 'limit_up': ref * 1.1, 'limit_down': ref * 0.9}


def test_stock_buy_is_only_sellable_after_the_t1_session():
    ledger = Ledger(100_000, slippage=0, cash_buffer=0, participation=1)
    ledger.marks = {'sh.600000': 10.0}
    ledger.queue_targets({'sh.600000': 0.5}, '2026-09-22', 'rebalance', {'sh.600000': RULE})
    ledger.execute('2026-09-23', 1, {'sh.600000': bar('2026-09-23')},
                   {'sh.600000': RULE}, {'sh.600000': 1e9})
    assert ledger.quantity('sh.600000') == 5_000
    assert ledger.available('sh.600000', 1) == 0
    assert ledger.available('sh.600000', 2) == 5_000


def test_limit_bands_block_buy_at_limit_up_and_sell_at_limit_down():
    assert trade_permission(bar('2026-09-23', 11.0), RULE, 'buy', '2026-09-23') == (
        False, 'limit_up')
    assert trade_permission(bar('2026-09-23', 9.0), RULE, 'sell', '2026-09-23') == (
        False, 'limit_down')


def test_participation_caps_a_fill_to_one_percent_of_prior_amount():
    ledger = Ledger(100_000, slippage=0, cash_buffer=0, participation=0.01)
    ledger.marks = {'sh.600000': 10.0}
    ledger.queue_targets({'sh.600000': 0.9}, '2026-09-22', 'rebalance', {'sh.600000': RULE})
    ledger.execute('2026-09-23', 1, {'sh.600000': bar('2026-09-23')},
                   {'sh.600000': RULE}, {'sh.600000': 1_000_000.0})
    assert ledger.quantity('sh.600000') == 1_000
    assert ledger.orders[0].status == 'expired'
    assert ledger.events[-1]['causes'] == ['liquidity_limit']


def test_dividend_and_split_conserve_nav_and_settle_taxable_cash():
    ledger = Ledger(100_000, slippage=0, cash_buffer=0)
    ledger.lots = {'sh.600000': [Lot(1_000, 0, 10.0, 0, '2024-01-02')]}
    ledger.marks = {'sh.600000': 10.0}
    ledger.close('2024-01-02', 0, {'sh.600000': 10.0})
    events = pd.DataFrame([dict(
        action_id='A1', instrument_id='sh.600000', record_date='2024-01-02',
        ex_date='2024-01-03', pay_date='2024-01-05', share_date='2024-01-04',
        available_at='2024-01-02', cash_per_share=1.0, share_ratio=1.0,
        action_type='distribution', tax_policy='net_fixed')])
    calendar = pd.bdate_range('2024-01-02', periods=5)
    ledger.corporate_actions('2024-01-03', 1, events, calendar)
    ledger.close('2024-01-03', 1, {'sh.600000': 4.5})
    assert ledger.quantity('sh.600000') == 2_000
    assert ledger.nav() == pytest.approx(110_000.0)
    ledger.corporate_actions('2024-01-05', 3, events, calendar)
    assert ledger.cash == pytest.approx(101_000.0)
    assert ledger.nav() == pytest.approx(110_000.0)


def test_delisting_liquidates_at_last_close_without_transaction_fees():
    ledger = Ledger(cash=0.0, delistings=pd.DataFrame(
        {'code': ['sh.600000'], 'delist_date': [pd.Timestamp('2026-09-25')]}))
    ledger.lots = {'sh.600000': [Lot(1_000, 0, 10.0, 0)]}
    assert ledger.settle_delistings('2026-09-25', {'sh.600000': 12.0}) == ['sh.600000']
    assert ledger.cash == pytest.approx(12_000.0)
    assert ledger.fills[-1]['fees'] == 0.0



def order(code, side, remaining, status, signal_date='2026-08-28'):
    return {'instrument_id': code, 'side': side, 'remaining': remaining,
            'status': status, 'signal_date': signal_date, 'reason': 'stable_book_rebalance'}


def test_pending_orders_only_surfaces_unfilled_remaining_shares():
    stock = {'orders': [
        order('sh.600000', 'buy', 100, 'pending'),
        order('sh.600001', 'sell', 0, 'filled'),
        order('sh.600002', 'buy', 50, 'partial'),
        order('sh.600003', 'buy', 100, 'expired'),
        order('sh.600004', 'sell', 0, 'cancelled'),
    ]}
    rows = S.pending_orders(stock)
    assert [(row['code'], row['side'], row['shares'], row['status']) for row in rows] == [
        ('sh.600000', 'buy', 100, 'pending'),
        ('sh.600002', 'buy', 50, 'partial')]
    assert rows[0]['signal_date'] == '2026-08-28'
    assert rows[0]['reason'] == 'stable_book_rebalance'


@pytest.fixture
def blocked_replay(monkeypatch):
    calendar = pd.bdate_range('2026-09-01', '2026-09-30')
    monkeypatch.setattr(S, 'frozen_schedule', lambda *a, **k: {pd.Timestamp('2026-08-28'): {}})
    monkeypatch.setattr(S.p4_ledger, 'to_targets', lambda schedule: [])
    monkeypatch.setattr(S.p4_ledger, 'load_dataset',
                        lambda codes, manifest_path=None: SimpleNamespace(calendar=calendar))
    monkeypatch.setattr(S.p4_ledger.book, 'MANIFEST', ROOT / 'dummy_manifest.json')
    monkeypatch.setattr(S, 'sha256_file', lambda path: 'test-hash')

    def forbidden(*a, **k):
        raise AssertionError('replay must not run when the as-of date is not a complete session')
    monkeypatch.setattr(S.p4_ledger, 'replay', forbidden)
    return calendar


def test_replay_book_blocks_after_the_complete_session(blocked_replay):
    book = S.replay_book('2026-10-01')
    assert book['status'] == 'blocked'
    assert book['reason'] == 'as_of_beyond_complete_session'
    assert book['orders'] == [] and book['nav'] is None
    assert book['identity']['last_complete_session'] == '2026-09-30'
    assert book['identity']['requested_as_of'] == '2026-10-01'


def test_replay_book_blocks_non_sessions(blocked_replay):
    book = S.replay_book('2026-09-27')
    assert book['status'] == 'blocked'
    assert book['reason'] == 'as_of_not_a_session'
    assert book['orders'] == [] and book['nav'] is None


def test_snapshot_fee_matches_the_ledger_for_a_stock():
    rule = InstrumentRules('sh.600000', 'stock', 100, 100, 1, 0.01, 0.0003, 5.0, 0.10)
    ledger = Ledger()
    for side in ('buy', 'sell'):
        commission, stamp = ledger.fee('2024-01-02', rule, side, 50_000.0)
        assert L.settlement_fee(50_000.0, side) == pytest.approx(commission + stamp)


def test_snapshot_fee_uses_the_minimum_commission():
    assert L.settlement_fee(1_000.0, 'buy') == pytest.approx(5.0)
    assert L.settlement_fee(1_000.0, 'sell') == pytest.approx(5.5)


PRESTART_PANEL = ROOT / 'data/smallcap_v4/20261018T012723Z/close.parquet'


@pytest.mark.skipif(not PRESTART_PANEL.exists(), reason='field panel not present')
def test_prestart_populates_next_grid_beyond_the_requested_date(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'OUT', tmp_path / 'p5_forward')
    monkeypatch.setattr(S.p4_ledger, 'load_dataset',
                        lambda codes, manifest_path=None:
                        SimpleNamespace(calendar=pd.bdate_range('2018-01-02', '2026-09-30')))
    snapshot = S.run_prestart('2026-10-30')
    assert snapshot['calendar_provisional'] is True
    assert snapshot['data_status'] == 'not_available'
    assert snapshot['orders'] == []
    assert snapshot['phases']['0']['last_grid'] == '2026-09-28'
    assert snapshot['phases']['0']['next_grid'] == '2026-11-02'
    assert snapshot['phases']['15']['last_grid'] == '2026-10-26'
    assert snapshot['phases']['15']['next_grid'] == '2026-11-23'
    assert snapshot['reason'] == 'no_new_signal'
