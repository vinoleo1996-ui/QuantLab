"""Parity of the forward engine against both quarter-end bases.

The frozen ``20261019`` book was replayed with the window basis, which treats a window ending
mid-quarter as a quarter end. The forward engine now pins quarter ends to the exchange calendar
(B1), so it must reproduce the re-run ``20261018_r1`` book exactly and stay inside tolerance of the
legacy window book.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import p5_snapshot as S

FROZEN = ROOT / 'output/p5_base_lot_20261018/runs/cap800000_cost1.csv'
LEGACY_NAV = ROOT / 'output/p5_base_lot_20261019/nav/cap800000_cost1_POOL50_G97_p0_s1.csv'
CALENDAR_NAV = ROOT / 'output/p5_base_lot_20261018_r1/nav/cap800000_cost1_POOL50_G97_p0_s1.csv'

# measured, not chosen: the window basis rebalances the ETF sleeve one session early
LEGACY_CAGR, LEGACY_MDD, LEGACY_NAV_VALUE = 0.13524805022069, -0.2376993952926168, 2_418_156.87393393
CALENDAR_CAGR, CALENDAR_MDD, CALENDAR_NAV_VALUE = (0.13524866642670252, -0.23769939529261686,
                                                   2_418_168.3195654587)


@pytest.mark.skipif(not (LEGACY_NAV.exists() and CALENDAR_NAV.exists()),
                    reason='p0_s1 evidence for one of the two bases is not present')
def test_replay_matches_the_calendar_basis_book_and_stays_inside_tolerance_of_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'OUT', tmp_path / 'p5_forward')
    report = S.run_parity()
    assert report['quarter_basis'] == 'calendar'
    assert report['within_tolerance'] is True
    assert abs(report['delta']['cagr_pp']) <= 0.20
    assert abs(report['delta']['nav_relative']) <= 0.002
    assert abs(report['delta']['mdd_pp']) <= 0.20
    # the engine reproduces the re-run calendar-basis book to the last digit
    assert report['replayed']['cagr'] == pytest.approx(CALENDAR_CAGR, abs=1e-10)
    assert report['replayed']['mdd'] == pytest.approx(CALENDAR_MDD, abs=1e-10)
    assert report['replayed']['nav'] == pytest.approx(CALENDAR_NAV_VALUE, abs=0.01)
    # and the legacy window book is unchanged as a historical reference
    assert report['legacy_window_reference']['cagr'] == pytest.approx(LEGACY_CAGR, abs=1e-10)
    assert report['legacy_window_reference']['nav'] == pytest.approx(LEGACY_NAV_VALUE, abs=0.01)
    assert abs(report['delta']['cagr_pp']) < 0.001, 'the two bases must differ only numerically'
    assert report['not_independent_validation'] is True
    assert report['promotion_approved'] is False
    written = json.loads((tmp_path / 'p5_forward/parity/2026-09-22/snapshot.json').read_text())
    assert written['within_tolerance'] is True
    assert written['quarter_basis'] == 'calendar'
