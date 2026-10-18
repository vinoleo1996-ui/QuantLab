"""The snapshot reports the signal dates that produced its orders, and flags any disagreement.

``replay_book`` builds orders from the frozen score-day schedule (``days[phase // 5::4]`` of the
score book) while ``decision_grid`` walks the production calendar. When those two disagree the
snapshot used to report one and trade the other.
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
S = importlib.import_module('p5_snapshot')

CAL = pd.bdate_range('2026-01-01', '2026-12-31')
PROD = S.decision_grid(CAL, CAL[0], CAL[-1], 0, 20)


def test_a_schedule_that_lags_the_production_grid_is_reported_as_mismatch():
    out = S.grid_report({day: {} for day in PROD[:-1]}, CAL, PROD[-1], CAL[-1], 0)
    assert out['last_grid'] == str(PROD[-2].date())
    assert out['signal_today'] is False
    assert out['calendar_mismatch'] == {
        'frozen_schedule_last_signal': str(PROD[-2].date()),
        'decision_grid_last_signal': str(PROD[-1].date())}


def test_an_aligned_schedule_reports_the_signal_day_without_mismatch():
    out = S.grid_report({day: {} for day in PROD}, CAL, PROD[-1], CAL[-1], 0)
    assert out['calendar_mismatch'] is None
    assert out['signal_today'] is True
    assert out['last_grid'] == str(PROD[-1].date())


def test_next_signal_comes_from_the_schedule_not_from_the_grid_step():
    schedule = {PROD[0]: {}, PROD[2]: {}}
    out = S.grid_report(schedule, CAL, PROD[1], CAL[-1], 0)
    assert out['last_grid'] == str(PROD[0].date())
    assert out['next_grid'] == str(PROD[2].date())
    assert out['calendar_mismatch']['decision_grid_last_signal'] == str(PROD[1].date())


def test_an_empty_schedule_reports_nothing_and_still_flags_the_grid():
    out = S.grid_report({}, CAL, PROD[0], CAL[-1], 0)
    assert out == {'last_grid': None, 'next_grid': None, 'signal_today': False,
                   'calendar_mismatch': {'frozen_schedule_last_signal': None,
                                         'decision_grid_last_signal': str(PROD[0].date())}}
