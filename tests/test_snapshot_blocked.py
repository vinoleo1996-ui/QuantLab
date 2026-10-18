"""A blocked snapshot may not publish a NAV, and may not run the six-month stop check on one."""
import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
S = importlib.import_module('p5_snapshot')


def fake_book(tmp_path):
    identity = {'requested_as_of': '2026-09-22', 'last_complete_session': '2026-09-30',
                'manifest': 'x/manifest.json', 'manifest_sha256': 'm' * 64,
                'scores_sha256': 's' * 64, 'schedule_last_signal': '2026-08-28',
                'stock_capital': 520_000.0, 'etf_capital': 280_000.0,
                'stock_gross': 0.97, 'cost_mult': 1.0,
                'book_config_hash': 'b' * 64, 'quarter_basis': 'calendar'}
    return {'status': 'blocked', 'reason': 'stock_data_quality', 'identity': identity,
            'orders': [{'code': 'sh.600000', 'leg': 'stock', 'side': 'buy', 'shares': 100}],
            'nav': None, 'nav_norm': pd.Series([1.0], index=[pd.Timestamp('2026-09-22')]),
            'stock': {'nav': pd.Series([1.0], index=[pd.Timestamp('2026-09-22')]),
                      'summary': {'data_quality_ok': False, 'missing_panel_count': 0,
                                  'suspension_coverage_complete': False,
                                  'unexplained_stale_held_max_days': 30, 'unsettled_delisting': False,
                                  'stale_held_max_days': 30, 'suspended_held_max_days': 0,
                                  'raw_uncovered_stale_codes': ['sh.600000'], 'cagr': 0.13,
                                  'max_drawdown': -0.2},
                      'fills': [], 'orders': [], 'account': [{'quantities': {}, 'marks': {},
                                                              'cash': 0.0, 'receivables': []}]},
            'etf_nav': pd.Series([1.0], index=[pd.Timestamp('2026-09-22')]),
            'etf_meta': {'final_cash': 0.0, 'final_shares': {}, 'trade_log': [],
                         'cash_ratio': pd.Series([0.0], index=[pd.Timestamp('2026-09-22')]),
                         'blocked': False, 'missing_on_rebalance': False,
                         'price_missing_days': 0,
                         'price_missing_dates': [], 'stale_mark_days': [],
                         'blocked_dates': [], 'premium_masked': {}, 'cash_ratio_mean': 0.0},
            'combined_weights': {}, 'trace': pd.DataFrame(), 'stock_sleeve_nav': 1.0,
            'etf_sleeve_nav': 1.0, 'stock_scale': 1.0, 'etf_scale': 1.0, 'stock_holdings': {},
            'stock_marks': {}, 'etf_holdings': {},
            'etf_marks': pd.Series({'sh518880': 1.0, 'sh511260': 1.0, 'sh513100': 1.0}),
            'sleeve_cash': 0.0, 'stock_value': 0.0, 'etf_value': 0.0, 'cash_value': 0.0,
            'stock_exposure': 0.0, 'etf_exposure': 0.0, 'cash_weight': 0.0,
            'cagr': 0.13, 'mdd': -0.2, 'last_signal': '2026-08-28',
            'calendar': pd.DatetimeIndex([pd.Timestamp('2026-09-22')]),
            'last_grid': '2026-09-22', 'next_grid': None, 'signal_today': True,
            'calendar_mismatch': None, 'quarter_basis': 'calendar', 'quarter_key': '2026Q3',
            'etf_trades_today': [], 'blend_meta': {}}


def args_for(tmp_path):
    return argparse.Namespace(as_of='2026-09-22', capital=800_000.0, cost=1.0, dry_run=True,
                              prestart=None, parity=False, broker_confirmed=False, holdings=None,
                              cash=None, state=str(tmp_path / 'state_paper.json'), no_orders=False)


def test_blocked_snapshot_publishes_no_nav_and_no_orders(tmp_path, monkeypatch):
    book = fake_book(tmp_path)
    monkeypatch.setattr(S, 'replay_book', lambda *a, **k: book)
    monkeypatch.setattr(S, 'OUT', tmp_path / 'p5_forward')
    monkeypatch.setattr(S, 'alloc60_series',
                        lambda etf_nav, calendar: (pd.Series([1.0, 1.0], index=etf_nav.index[:1]
                                                             .append(etf_nav.index[:1])),
                                                   'hash', {'config': 1}))
    S.run(args_for(tmp_path))
    snapshot = __import__('json').loads(
        (tmp_path / 'p5_forward' / 'dryrun' / '2026-09-22' / 'snapshot.json').read_text())
    assert snapshot['status'] == 'blocked'
    assert snapshot['nav'] is None
    assert snapshot['orders'] == []
    assert snapshot['data_quality']['unexplained_stale_held_max_days'] == 30
    assert snapshot['alloc60']['six_month']['stop_check_triggered'] is False
    assert not (tmp_path / 'state_paper.json').exists()
