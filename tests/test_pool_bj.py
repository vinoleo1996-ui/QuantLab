"""The Beijing exchange must never enter a pool built by ``pool_on`` itself.

``scripts/p4_alpha.py:85-86`` and ``scripts/p5_base_orders.py:136`` filter ``bj.`` at the call site,
but ``pool_on`` had no exclusion, so four other research paths (``p4_attrib.py:119``,
``smallcap_layers.py:57``, ``p1_m2.py:37``, ``p4_baselines.py:74``) could pool Beijing names.
"""
import pandas as pd

from quantlab.smallcap.universe import MIN_ADV, pool_on

DAY = pd.Timestamp('2024-01-02')
BJ = 'bj.832000'
SZ = ['sz.000001', 'sz.000002', 'sz.000003', 'sz.000004', 'sz.000005']


def features(codes, adv, cap):
    """One session wide with the codes across the columns, as ``rolling_features`` returns them."""
    index = pd.Index(codes)

    def frame(values):
        return pd.DataFrame([list(values)], index=[DAY], columns=index)

    return {'adv': frame([adv[code] for code in codes]),
            'history': frame([300] * len(codes)),
            'cap': frame([cap[code] for code in codes])}


def test_bj_name_ranked_into_the_bottom_band_is_still_excluded():
    codes = SZ + [BJ]
    # Beijing is the least liquid, so the bottom-40-percent cut would otherwise keep it.
    adv = {code: value for code, value in zip(codes, [9e7, 8e7, 7e7, 6e7, 5e7, MIN_ADV + 1])}
    cap = {code: 1e9 for code in codes}
    pool = pool_on(DAY, features(codes, adv, cap), 'adv', band='relative')
    assert BJ not in pool
    # five eligible names, ceil(0.6 * 5) = 3, so the last two ranks form the pool
    assert pool == ['sz.000004', 'sz.000005']


def test_bj_name_is_excluded_from_the_cap_ranked_pool_too():
    codes = SZ + [BJ]
    adv = {code: 5e7 for code in codes}
    cap = {code: (1e10 if code == BJ else 1e9) for code in codes}
    pool = pool_on(DAY, features(codes, adv, cap), 'cap', band='relative')
    assert BJ not in pool
    assert pool == ['sz.000004', 'sz.000005']
