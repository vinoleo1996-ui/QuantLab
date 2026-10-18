import numpy as np
import pandas as pd
import pytest

from quantlab import p4_alloc as A

DAYS = pd.bdate_range('2020-01-01', periods=400)


def etf(seed=0):
    rng = np.random.default_rng(seed)
    scales = {'gold': .01, 'nasdaq': .015, 'sp500': .012, 'treasury': .002}
    return pd.DataFrame({k: rng.normal(.0002, s, len(DAYS)) for k, s in scales.items()}, index=DAYS)


def test_quarter_and_month_have_different_rebalance_dates():
    months = A.rebalance_ends(DAYS, 'month')
    quarters = A.rebalance_ends(DAYS, 'quarter')
    assert len(months) == len(pd.Series(DAYS).groupby([DAYS.year, DAYS.month]).max())
    assert len(quarters) < len(months)
    assert quarters <= months


def test_unknown_rebalance_frequency_is_rejected():
    with pytest.raises(ValueError, match='unknown_rebalance'):
        A.rebalance_ends(DAYS, 'week')


def test_frequency_changes_nav_when_etf_returns_vary():
    equity = pd.Series(np.linspace(1.0, 1.2, len(DAYS)), index=DAYS)
    returns = etf()
    monthly, _ = A.allocate(equity, returns, rebalance='month')
    quarterly, _ = A.allocate(equity, returns, rebalance='quarter')
    assert not np.isclose(monthly.iloc[-1], quarterly.iloc[-1])


def test_daily_drawdown_check_catches_a_mid_month_breach():
    equity = pd.Series(1.0, index=DAYS)
    # A 30% breach and full recovery both happen inside the first month.
    equity.iloc[10:12] = 0.70
    equity.iloc[12:20] = np.linspace(0.70, 1.0, 8)
    flat = pd.DataFrame(0.0, index=DAYS, columns=['gold', 'nasdaq', 'sp500', 'treasury'])
    _, monthly = A.allocate(equity, flat, rebalance='month', drawdown_check='rebalance')
    _, daily = A.allocate(equity, flat, rebalance='month', drawdown_check='daily')
    assert monthly['disaster_triggers'] == 0
    assert daily['disaster_triggers'] == 1


def test_qdii_overflow_split_is_explicit():
    names = ['equity', 'gold', 'treasury', 'nasdaq']
    target = np.array([0.65, 0.13125, 0.13125, 0.0875])
    current = np.array([0.65, 0.13125, 0.13125, 0.05])
    out, paused = A.pause_qdii(target, current, names, {'nasdaq': 0.10},
                               overflow={'gold': 0.5, 'treasury': 0.5})
    assert paused == ['nasdaq']
    assert out[names.index('nasdaq')] == pytest.approx(0.05)
    assert out[names.index('gold')] == pytest.approx(0.13125 + 0.0375 / 2)
    assert out[names.index('treasury')] == pytest.approx(0.13125 + 0.0375 / 2)


def test_qdii_overflow_must_sum_to_one():
    with pytest.raises(ValueError, match='sum_to_one'):
        A.pause_qdii(np.ones(4), np.ones(4), ['equity', 'gold', 'treasury', 'nasdaq'],
                     {'nasdaq': 0.10}, overflow={'gold': 0.7, 'treasury': 0.7})
