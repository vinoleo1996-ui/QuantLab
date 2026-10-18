"""Quarter/month rebalance dates must be decided by the exchange calendar, not by the window.

A replay window that ends inside a quarter used to treat its own final session as that quarter's
end, so the frozen 2026-09-22 book rebalanced on a day that is not a real quarter end, and a
forward replay produced a different date for the same quarter as the requested date advanced.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab import p4_alloc as A

ROOT = Path(__file__).resolve().parents[1]
CALENDAR = pd.bdate_range('2018-01-02', '2026-12-31')
Q3_END = pd.Timestamp('2026-09-30')
FROZEN_SLEEVE = ROOT / 'output/p5_base_lot_20261019/nav/cap800000_cost1_POOL50_G97_p0_s1.csv'
ETF_BARS = ROOT / 'data/smallcap_assets/etf_bars.parquet'


def window(end):
    return CALENDAR[CALENDAR <= pd.Timestamp(end)]


def _lot_replay():
    sys.path.insert(0, str(ROOT / 'scripts'))
    return importlib.import_module('p5_base_lot_replay')


def test_window_basis_marks_a_mid_quarter_final_day_as_quarter_end():
    sessions = window('2026-09-22')
    assert pd.Timestamp('2026-09-22') in A.quarter_ends(sessions)


def test_calendar_basis_keeps_the_quarter_end_fixed_as_the_window_grows():
    ends = {end: A.quarter_ends(window(end), basis='calendar', calendar=CALENDAR)
            for end in ('2026-09-22', '2026-10-15', '2026-10-30')}
    assert pd.Timestamp('2026-09-22') not in ends['2026-09-22']
    assert Q3_END not in ends['2026-09-22']
    assert Q3_END in ends['2026-10-15'] and Q3_END in ends['2026-10-30']
    # a shorter window is a strict prefix: previously chosen dates never move
    assert ends['2026-09-22'] < ends['2026-10-15']
    assert ends['2026-10-15'] == ends['2026-10-30']


def test_calendar_basis_is_refused_without_a_calendar_covering_the_window():
    with pytest.raises(ValueError, match='calendar_required'):
        A.quarter_ends(window('2026-09-22'), basis='calendar')
    short = CALENDAR[CALENDAR <= pd.Timestamp('2026-09-22')]
    with pytest.raises(ValueError, match='calendar_too_short'):
        A.quarter_ends(window('2026-09-22'), basis='calendar', calendar=short)


def test_rebalance_ends_forwards_basis_to_both_frequencies():
    sessions = window('2026-09-22')
    assert (A.rebalance_ends(sessions, 'quarter', basis='calendar', calendar=CALENDAR)
            == A.quarter_ends(sessions, basis='calendar', calendar=CALENDAR))
    assert Q3_END in A.rebalance_ends(window('2026-10-15'), 'month', basis='calendar', calendar=CALENDAR)
    with pytest.raises(ValueError, match='unknown_basis'):
        A.quarter_ends(sessions, basis='whatever', calendar=CALENDAR)


@pytest.mark.skipif(not (FROZEN_SLEEVE.exists() and ETF_BARS.exists()),
                    reason='frozen sleeve series or ETF bars not available')
def test_window_basis_reproduces_the_frozen_etf_sleeve_and_calendar_removes_the_extra_trade():
    B = _lot_replay()
    sessions = pd.DatetimeIndex(pd.read_csv(FROZEN_SLEEVE, index_col=0, parse_dates=True).index)
    reference = pd.read_csv(FROZEN_SLEEVE, index_col=0, parse_dates=True)['etf']
    # The real exchange calendar runs past the frozen window, so 09-22 is not a quarter end there.
    bars_calendar = pd.DatetimeIndex(sorted(pd.read_parquet(ETF_BARS, columns=['date'])
                                            .date.drop_duplicates()))
    assert bars_calendar.max() > sessions.max()

    legacy, legacy_meta = B.run_g0_lot_sleeve(sessions, 280_000.0, 1.0)
    assert float(np.max(np.abs(legacy.to_numpy() - reference.to_numpy()))) < 1e-6
    legacy_dates = {trade['date'] for trade in legacy_meta['trade_log']}
    assert str(pd.Timestamp('2026-09-22').date()) in legacy_dates

    pinned, pinned_meta = B.run_g0_lot_sleeve(sessions, 280_000.0, 1.0, 'calendar', bars_calendar)
    pinned_dates = {trade['date'] for trade in pinned_meta['trade_log']}
    assert str(pd.Timestamp('2026-09-22').date()) not in pinned_dates
    assert len(pinned_dates) < len(legacy_dates)
    assert float(pinned.iloc[-1]) != float(legacy.iloc[-1])
