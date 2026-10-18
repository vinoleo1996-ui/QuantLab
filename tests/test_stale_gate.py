"""A held name whose bars stop at the end of the window is a supply gap, not an endless suspension.

``sh.600293`` has a last local bar of 2026-09-28, no row in ``delistings.parquet`` and no bar from
the external daily source. Verified raw rows can show ``tradestatus=0`` for every session after that,
so the old classifier accepted the run as "suspended" and the ledger kept marking the position at the
last close with no liquidation and no gate failure.
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
stable_book = importlib.import_module('stable_book')

from quantlab.data.suspensions import classify_run

CODE = 'sh.600293'
WINDOW_END = pd.Timestamp('2026-09-30')
LONG_RUN = pd.bdate_range(end=WINDOW_END, periods=20)


class Evidence:
    """Every session in the run carries a verified suspension record."""

    def get(self, code):
        return {'complete': True}

    def session_status(self, code, day):
        return 'suspended'


def test_an_open_ended_run_without_a_delisting_record_is_a_supply_gap():
    result = classify_run(CODE, LONG_RUN, Evidence(), None, window_end=WINDOW_END)
    assert result['status'] == 'unexplained'
    assert result['reason'] == 'open_ended_supply_gap'
    assert result['days'] == 20


def test_a_run_that_resumes_before_the_window_end_is_still_a_suspension():
    assert classify_run(CODE, LONG_RUN[:-3], Evidence(), None,
                        window_end=WINDOW_END)['status'] == 'suspended'


def test_a_short_open_ended_run_is_not_yet_evidence_of_delisting():
    assert classify_run(CODE, LONG_RUN[-3:], Evidence(), None,
                        window_end=WINDOW_END)['status'] == 'suspended'


def test_a_known_delisting_date_after_the_run_keeps_the_suspension():
    assert classify_run(CODE, LONG_RUN, Evidence(), pd.Timestamp('2026-10-09'),
                        window_end=WINDOW_END)['status'] == 'suspended'


def test_data_quality_fails_closed_on_the_supply_gap():
    sessions = pd.bdate_range('2026-08-01', WINDOW_END)
    snapshots = [{'date': str(day.date()), 'quantities': {CODE: 100},
                  'stale': [CODE] if day >= LONG_RUN[0] else []} for day in sessions]
    report = stable_book.data_quality(snapshots, [], [], True,
                                      suspension_evidence=Evidence(), delistings={})
    assert report['data_quality_ok'] is False
    assert report['unexplained_stale_held_codes'][CODE] == 20
    assert report['unexplained_runs'][0]['reason'] == 'open_ended_supply_gap'
