"""ALLOC60 is the preregistered stop-loss benchmark, so its legs may not be swapped for ours.

`docs/prereg/P5_FORWARD.md:43` pins it to the frozen 90% gross POOL50 ledger with 60% equity and
monthly rebalancing. ``alloc60_series`` was feeding it the replayed 97% gross ``POOL50_G97`` sleeve,
i.e. a benchmark built from the same book it is supposed to test against.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
S = importlib.import_module('p5_snapshot')

FROZEN_EQUITY = ROOT / 'output/p4_alpha_20261012/ledger/nav_POOL50_p0_s1_cost1.csv'
# the ledger calendar, not business days: exchange holidays are not sessions
SESSIONS = pd.read_csv(FROZEN_EQUITY, index_col=0, parse_dates=True).index


def test_equity_leg_is_the_frozen_ninety_percent_ledger_not_the_replayed_book(monkeypatch):
    captured = {}

    def fake_allocate(equity_nav, etf_returns, equity_share, **kwargs):
        captured['equity_nav'] = equity_nav
        captured['share'] = equity_share
        captured['kwargs'] = kwargs
        return pd.Series(1.0, index=equity_nav.index), {'weights': pd.DataFrame()}

    monkeypatch.setattr(S.A, 'allocate', fake_allocate)
    etf_nav = pd.Series(1.0, index=SESSIONS)
    S.alloc60_series(etf_nav, SESSIONS)
    frozen = pd.read_csv(FROZEN_EQUITY, index_col=0, parse_dates=True)['nav']
    used = captured['equity_nav']
    common = frozen.index.intersection(used.index)
    assert len(common) > 2000
    assert float(np.max(np.abs(used.reindex(common).to_numpy() /
                              frozen.reindex(common).to_numpy() - 1))) < 1e-12
    assert captured['share'] == 0.60
    assert captured['kwargs']['rebalance'] == 'month'


def test_month_ends_come_from_the_calendar_so_a_mid_month_window_has_no_pseudo_rebalance(monkeypatch):
    captured = {}

    def fake_allocate(equity_nav, etf_returns, equity_share, **kwargs):
        captured['kwargs'] = kwargs
        return pd.Series(1.0, index=equity_nav.index), {'weights': pd.DataFrame()}

    monkeypatch.setattr(S.A, 'allocate', fake_allocate)
    calendar = SESSIONS.append(pd.DatetimeIndex(['2026-09-23', '2026-09-24']))
    S.alloc60_series(pd.Series(1.0, index=SESSIONS), calendar)
    assert captured['kwargs']['basis'] == 'calendar'
    assert list(captured['kwargs']['calendar']) == list(calendar)


def test_config_hash_records_the_benchmark_identity(monkeypatch):
    monkeypatch.setattr(S.A, 'allocate', lambda equity_nav, etf_returns, share, **kwargs:
                        (pd.Series(1.0, index=equity_nav.index), {'weights': pd.DataFrame()}))
    nav, digest, config = S.alloc60_series(pd.Series(1.0, index=SESSIONS), SESSIONS)
    assert config['equity'] == 'frozen POOL50 90% ledger nav_POOL50_p0_s1_cost1.csv'
    assert config['rebalance'] == 'month' and config['equity_share'] == 0.60
    assert config['frozen_equity_sha256']
    assert len(digest) == 64
