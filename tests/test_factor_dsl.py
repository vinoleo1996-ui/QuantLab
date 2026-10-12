import unittest

import numpy as np
import pandas as pd

from quantlab.factor_lab.dsl import FormulaError, canonical, evaluate, formula_id, validate

DAYS = pd.bdate_range('2023-01-02', periods=120)


def panels(seed=0):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .02, (len(DAYS), 4)), axis=0)), index=DAYS, columns=list('abcd'))
    return {k: base * m for k, m in dict(open=1, high=1.01, low=.99, close=1, vwap=1, volume=1e6, amount=1e7,
                                         turnover=1, float_cap=1e9).items()} | {'ret': base.pct_change()}


class DslTests(unittest.TestCase):
    def test_rejects_anything_outside_the_whitelist(self):
        for bad in ['__import__("os")', 'close.shift(-1)', 'delay(close, -1)', 'delay(close, 0)', 'ts_mean(close, 300)',
                    'ts_mean(close, n)', 'foo(close)', 'close ** 2', 'lambda: close', '3.0', 'close[0]',
                    'ts_mean(close, 5, 6)', 'rank(close, 5)', 'ts_corr(close, volume, 3)']:
            with self.assertRaises(FormulaError, msg=bad):
                validate(bad)

    def test_formula_cannot_see_the_future(self):
        base = panels()
        changed = {k: v.copy() for k, v in base.items()}
        for v in changed.values():
            v.iloc[80:] *= 7.0
        text = 'rank(ts_corr(close, volume, 10)) - zscore(delta(log(close), 5)) + ts_rank(ret, 20) / ts_std(vwap, 15)'
        a, b = evaluate(text, base), evaluate(text, changed)
        pd.testing.assert_frame_equal(a.iloc[:80], b.iloc[:80])

    def test_canonical_form_dedupes_spacing(self):
        self.assertEqual(canonical('ts_mean( close ,5 )'), canonical('ts_mean(close, 5)'))
        self.assertEqual(formula_id('neg(ret)'), formula_id('neg( ret )'))


if __name__ == '__main__':
    unittest.main()
