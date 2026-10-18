import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import p5_base_orders as O


def test_execution_session_is_the_next_exchange_session():
    calendar = pd.bdate_range('2026-09-25', periods=4)
    assert O.execution_session(calendar, '2026-09-25') == pd.Timestamp('2026-09-28')
    assert O.execution_session(calendar, '2026-09-25', '2026-09-28') == pd.Timestamp('2026-09-28')


def test_execution_session_rejects_a_mismatched_pin():
    calendar = pd.bdate_range('2026-09-25', periods=4)
    with pytest.raises(SystemExit, match='execution_date_not_next_session:2026-09-28'):
        O.execution_session(calendar, '2026-09-25', '2026-09-29')


def test_execution_session_requires_a_later_session():
    with pytest.raises(SystemExit, match='no_next_session'):
        O.execution_session(pd.bdate_range('2026-09-25', periods=1), '2026-09-25')


def test_stale_grid_signal_cannot_produce_an_order_list():
    with pytest.raises(SystemExit, match='stale_signal:2026-09-28:requested_2026-10-22'):
        O.require_signal_day('2026-09-28', '2026-10-22')
    assert O.require_signal_day('2026-09-28', '2026-09-28') == pd.Timestamp('2026-09-28')
