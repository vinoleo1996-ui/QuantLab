import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.strategy.crowding import RiskSnapshot
from quantlab.strategy.legacy import allocate_legacy


class LegacySolverBoundary(unittest.TestCase):
    def test_maximum_gross_roundoff_does_not_block_feasible_allocation(self):
        cfg = deepcopy(CONFIG)
        cfg.portfolio.lw_shrinkage = True
        cfg.portfolio.target_vol = .14
        cfg.portfolio.weight_max_dev = .015
        cfg.research.max_stock_weight = .06
        cfg.research.max_cluster_weight = .30
        cfg.research.cash_buffer = .02
        codes = [str(i) for i in range(20)]
        returns = pd.DataFrame(
            np.random.default_rng(23).normal(0, .06, (120, 20)),
            columns=codes,
        )
        groups = RiskSnapshot(
            '2020-03-31',
            {code: 'common' if i < 12 else code
             for i, code in enumerate(codes)},
            pd.Series(0., index=codes), pd.Series(0., index=codes), [],
        )

        # Twelve names share a 30% ceiling; each of the eight others is at
        # most 6.5% of invested capital: g <= .30 + 8 * .065 * g = .625.
        # This synthetic case reproduces the real two-stage failure: stage 1
        # returns g=.6250000000002238, which stage 2 cannot impose exactly.
        weights, status, detail = allocate_legacy(
            returns, cfg, selected=codes, risk_snapshot=groups,
        )

        self.assertEqual(status, 'ok', detail)
        self.assertAlmostEqual(weights.sum(), .625, places=10)
        self.assertLessEqual(detail['feasible_gross'], .625 + 1e-12)
        self.assertLessEqual(weights.iloc[:12].sum(), .30 + 1e-10)
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights <= .06 + 1e-10).all())
        self.assertTrue((weights <= .065 * weights.sum() + 1e-10).all())
        self.assertLessEqual(detail['target_vol'], .14 + 1e-8)


if __name__ == '__main__':
    unittest.main()
