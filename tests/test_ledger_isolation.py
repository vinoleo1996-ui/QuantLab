"""Account parameters must reach the ledger as arguments, never through shared module globals.

``p4_ledger.replay`` used to swap ``stable_book.GROSS/CASH_BUFFER/INITIAL_CASH`` in a context
manager. Two replays with different capital or gross therefore overwrote each other, which is a
live hazard while a second session runs in the same working tree.
"""
import threading
import time

import pytest

from quantlab import p4_ledger

TARGETS = [{'signal_date': '2024-01-02', 'weights': {'sh.600000': 0.5},
            'active': ['sh.600000'], 'sell': []}]
SENTINELS = (p4_ledger.book.GROSS, p4_ledger.book.CASH_BUFFER, p4_ledger.book.INITIAL_CASH)


def spy(monkeypatch, hold=0.0):
    seen = []

    def fake_replay_account(dataset, targets, **kwargs):
        seen.append(dict(kwargs, observed=(p4_ledger.book.GROSS, p4_ledger.book.CASH_BUFFER,
                                           p4_ledger.book.INITIAL_CASH)))
        if hold:
            time.sleep(hold)
        return {'summary': {}}

    monkeypatch.setattr(p4_ledger.book, 'replay_account', fake_replay_account)
    return seen


def run(capital, **kwargs):
    return p4_ledger.replay(TARGETS, start='2024-01-02', end='2024-01-03', dataset=object(),
                            capital_cny=capital, **kwargs)


def test_replay_passes_capital_gross_and_buffer_as_arguments(monkeypatch):
    seen = spy(monkeypatch)
    run(800_000.0, gross=0.97)
    assert seen[0]['capital'] == 800_000.0
    assert seen[0]['gross'] == 0.97
    # the buffer is derived inside the account replay, from the gross it was handed
    assert seen[0]['cash_buffer'] is None
    assert p4_ledger.book.cash_buffer_for(0.97) == pytest.approx(0.03)
    assert seen[0]['observed'] == SENTINELS


def test_defaults_are_forwarded_as_none_and_never_write_globals(monkeypatch):
    seen = spy(monkeypatch)
    run(800_000.0)
    assert seen[0]['gross'] is None
    assert p4_ledger.book.GROSS == SENTINELS[0]
    assert p4_ledger.book.CASH_BUFFER == SENTINELS[1]
    assert p4_ledger.book.INITIAL_CASH == SENTINELS[2]


def test_two_threads_with_different_accounts_do_not_see_each_other(monkeypatch):
    seen = spy(monkeypatch, hold=0.05)
    threads = [threading.Thread(target=run, args=(capital,), kwargs=dict(gross=0.97))
               for capital in (800_000.0, 600_000.0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(row['capital'] for row in seen) == [600_000.0, 800_000.0]
    assert all(row['observed'] == SENTINELS for row in seen)
