import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import p5_snapshot as S


def fake_nav():
    index = pd.bdate_range('2026-09-15', periods=6)
    return pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=index)


def fake_book(status='paper_simulated', reason='ok'):
    nav = fake_nav()
    stock_value, etf_value, cash_value = 10_000.0, 6_500.0, 1_000.0
    total = stock_value + etf_value + cash_value
    return {
        'status': status, 'reason': reason, 'nav': total, 'nav_norm': nav,
        'identity': {'requested_as_of': '2026-09-22', 'last_complete_session': '2026-09-22',
                     'manifest': 'output/account_2025_20260927/extension/dataset/manifest.json',
                     'manifest_sha256': 'manifest-hash', 'scores_sha256': 'scores-hash'},
        'orders': [], 'stock': {'nav': nav, 'summary': {
            'data_quality_ok': True, 'missing_panel_count': 0,
            'suspension_coverage_complete': True, 'unexplained_stale_held_max_days': 0,
            'unsettled_delisting': False, 'stale_held_max_days': 0,
            'suspended_held_max_days': 0, 'raw_uncovered_stale_codes': [],
            'cagr': 0.13, 'max_drawdown': -0.20}, 'fills': []},
        'etf_nav': nav, 'etf_meta': {'final_cash': 1_000.0, 'final_shares': {
            'gold': 1_000, 'treasury': 500, 'nasdaq': 0}, 'blocked': False,
            'missing_on_rebalance': [], 'price_missing_days': [], 'trade_log': []},
        'combined_weights': {'equity': 0.65, 'g0': 0.35},
        'stock_sleeve_nav': 680_000.0, 'etf_sleeve_nav': 350_000.0,
        'stock_scale': 1.0, 'etf_scale': 1.0,
        'stock_holdings': {'sh.600000': 1_000},
        'stock_marks': {'sh.600000': 10.0},
        'etf_holdings': {'sh518880': 1_000, 'sh511260': 500},
        'etf_marks': pd.Series({'gold': 4.0, 'treasury': 5.0, 'nasdaq': 1.0}),
        'sleeve_cash': 1_000.0, 'stock_value': stock_value, 'etf_value': etf_value,
        'cash_value': cash_value, 'stock_exposure': stock_value / total,
        'etf_exposure': etf_value / total, 'cash_weight': cash_value / total,
        'cagr': 0.13, 'mdd': -0.20, 'last_signal': '2026-08-28',
        'calendar': pd.bdate_range('2026-09-21', periods=10),
        'last_grid': '2026-08-28', 'next_grid': '2026-09-28',
        'signal_today': False, 'quarter_key': '2026Q3',
        'etf_trades_today': [], 'blend_meta': {},
    }


def args_for(as_of, state=None, broker=False, holdings=None, cash=None):
    return Namespace(as_of=as_of, capital=800_000.0, cost=1.0, state=state,
                     holdings=holdings, cash=cash, broker_confirmed=broker,
                     dry_run=False, no_orders=False, parity=False, prestart=None)


@pytest.fixture
def snapshot_paths(tmp_path, monkeypatch):
    out = tmp_path / 'p5_forward'
    paper = out / 'state_paper.json'
    broker = out / 'state_broker.json'
    monkeypatch.setattr(S, 'OUT', out)
    monkeypatch.setattr(S, 'PAPER_STATE', paper)
    monkeypatch.setattr(S, 'BROKER_STATE', broker)
    monkeypatch.setattr(S, 'alloc60_series',
                        lambda stock_nav, etf_nav: (fake_nav(), 'alloc-hash',
                                                     {'equity_share': 0.60}))
    return out, paper, broker


def test_blocked_snapshot_writes_no_state(snapshot_paths, monkeypatch):
    out, paper, _ = snapshot_paths
    blocked = {'status': 'blocked', 'reason': 'as_of_beyond_complete_session',
               'identity': {'last_complete_session': '2026-09-22'}, 'orders': [], 'nav': None}
    monkeypatch.setattr(S, 'replay_book', lambda *a, **k: blocked)
    S.run(args_for('2026-10-30'))
    snapshot = json.loads((out / '2026-10-30/snapshot.json').read_text())
    assert snapshot['status'] == 'blocked'
    assert snapshot['orders'] == [] and snapshot['nav'] is None
    assert snapshot['as_of'] is None
    assert snapshot['reason'] == 'as_of_beyond_complete_session'
    assert not paper.exists()


def test_data_quality_block_writes_snapshot_without_advancing_state(snapshot_paths, monkeypatch):
    out, paper, _ = snapshot_paths
    monkeypatch.setattr(S, 'replay_book',
                        lambda *a, **k: fake_book('blocked', 'stock_data_quality'))
    S.run(args_for('2026-09-22'))
    snapshot = json.loads((out / '2026-09-22/snapshot.json').read_text())
    assert snapshot['status'] == 'blocked'
    assert snapshot['orders'] == []
    assert not paper.exists()


def test_paper_state_is_idempotent_for_the_same_as_of(snapshot_paths, monkeypatch):
    out, paper, _ = snapshot_paths
    monkeypatch.setattr(S, 'replay_book', lambda *a, **k: fake_book())
    S.run(args_for('2026-09-22'))
    S.run(args_for('2026-09-22'))
    state = json.loads(paper.read_text())
    assert [row['date'] for row in state['history']] == ['2026-09-22']
    history = pd.read_csv(out / 'history.csv')
    assert history.date.tolist() == ['2026-09-22']
    snapshot = json.loads((out / '2026-09-22/snapshot.json').read_text())
    assert snapshot['status'] == 'paper_simulated'
    assert snapshot['nav'] == pytest.approx(17_500.0)
    assert snapshot['stock_names'] == 1


def test_broker_run_does_not_touch_paper_state(snapshot_paths, monkeypatch, tmp_path):
    out, paper, broker = snapshot_paths
    monkeypatch.setattr(S, 'replay_book', lambda *a, **k: fake_book())
    S.run(args_for('2026-09-22'))
    paper_before = paper.read_bytes()
    holdings = tmp_path / 'holdings.csv'
    pd.DataFrame({'code': ['sh.600000', 'sh518880'], 'shares': [1_000, 500]}).to_csv(
        holdings, index=False)
    S.run(args_for('2026-09-23', broker=True, holdings=str(holdings), cash=250_000.0))
    assert paper.read_bytes() == paper_before
    broker_state = json.loads(broker.read_text())
    assert broker_state['status'] == 'broker_confirmed'
    assert broker_state['holdings'] == {'sh.600000': 1_000, 'sh518880': 500}
    assert broker_state['cash'] == pytest.approx(250_000.0)
    snapshot = json.loads((out / '2026-09-23/snapshot.json').read_text())
    assert snapshot['broker'] == {'holdings': broker_state['holdings'], 'cash': 250_000.0}
    assert snapshot['state_path'] == str(broker)


def test_broker_requires_imported_holdings_and_cash(snapshot_paths, monkeypatch):
    monkeypatch.setattr(S, 'replay_book', lambda *a, **k: fake_book())
    with pytest.raises(SystemExit):
        S.run(args_for('2026-09-22', broker=True, holdings=None, cash=250_000.0))


def test_write_history_replaces_the_same_date(tmp_path):
    path = tmp_path / 'history.csv'
    S.write_history(path, {'date': '2026-09-22', 'nav': 100.0})
    S.write_history(path, {'date': '2026-09-22', 'nav': 101.0})
    frame = pd.read_csv(path)
    assert frame.date.tolist() == ['2026-09-22']
    assert frame.nav.tolist() == [101.0]


def test_state_lock_is_exclusive_and_released(tmp_path):
    path = tmp_path / 'state.json'
    with S.state_lock(path):
        with pytest.raises(SystemExit):
            with S.state_lock(path):
                pass
    with S.state_lock(path):
        pass
