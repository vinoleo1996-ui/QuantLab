from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import p5_g0_alloc as G


pytestmark = pytest.mark.skipif(
    not (G.ALPHA / 'nav_POOL50_p0_s1_cost1.csv').exists()
    or not (G.GROSS / 'nav_POOL50_G97_p0_s1_cost1.csv').exists()
    or not (ROOT / 'data/smallcap_assets/etf_bars.parquet').exists(),
    reason='frozen P5_G0 ledgers or ETF bars are unavailable',
)


@pytest.fixture(scope='module')
def cells():
    return {
        'g0_60_c1': G.run(G.ALPHA, 'POOL50_', 0.60, 0.23, 1, 'g0', 'quarter'),
        'g0_60_c2': G.run(G.ALPHA, 'POOL50_', 0.60, 0.23, 2, 'g0', 'quarter'),
        'g97_65_c1': G.run(G.GROSS, 'POOL50_G97_', 0.65, 0.23, 1, 'g0', 'quarter'),
        'g97_65_c2': G.run(G.GROSS, 'POOL50_G97_', 0.65, 0.23, 2, 'g0', 'quarter'),
        'g97_70_noline_c1': G.run(G.GROSS, 'POOL50_G97_', 0.70, 1.0, 1, 'g0', 'quarter'),
    }


def test_g0_60_quarterly_reproduces_report(cells):
    cell = cells['g0_60_c1']
    assert cell['n'] == 8
    assert cell['cagr'] == pytest.approx(0.1270, abs=2e-4)
    assert cell['mdd'] == pytest.approx(-0.1911, abs=2e-4)
    assert all(t == 0 for t in cell['triggers'])


def test_g97_65_quarterly_reproduces_report(cells):
    cell = cells['g97_65_c1']
    assert cell['n'] == 8
    assert cell['cagr'] == pytest.approx(0.1319, abs=2e-4)
    assert cell['mdd'] == pytest.approx(-0.2278, abs=2e-4)
    assert all(t == 0 for t in cell['triggers'])


def test_g97_65_cost2_quarterly_reproduces_report(cells):
    cell = cells['g97_65_c2']
    assert cell['cagr'] == pytest.approx(0.1233, abs=2e-4)
    assert cell['mdd'] == pytest.approx(-0.2320, abs=2e-4)


def test_g97_70_noline_quarterly_reproduces_report(cells):
    cell = cells['g97_70_noline_c1']
    assert cell['cagr'] == pytest.approx(0.1317, abs=2e-4)
    assert cell['mdd'] == pytest.approx(-0.2471, abs=2e-4)


def test_decision_uses_the_same_frequency(cells):
    named = {
        'G97_65_23_q_c1': cells['g97_65_c1'],
        'G97_70_noline_q_c1': cells['g97_70_noline_c1'],
    }
    decision = G.decide(named, 'quarter')
    assert decision['selected'] == 'G97_65_23_q'
    assert 'does not beat' in decision['reason']
