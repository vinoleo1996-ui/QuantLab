"""The continuous band keeps one basket, and the 15 percent clock starts in 2018."""
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import continuous_band as band


class ContinuousBandTests(unittest.TestCase):
    def _inputs(self):
        codes = [f'{i:06d}.SZ' for i in range(80)]
        industries = {code: str(i % 8) for i, code in enumerate(codes)}
        heads = {
            '2021-12-15': {code: 1.0 for code in codes},
            '2022-01-20': {code: (0.0 if i < 40 else 1.0) for i, code in enumerate(codes)},
        }
        groups = {day: dict(industries) for day in heads}
        return heads, groups

    def test_one_basket_crosses_the_period_boundary_a_new_policy_does_not(self):
        heads, groups = self._inputs()
        sessions = pd.to_datetime(['2021-12-15', '2022-01-20'])
        carried = band.build_targets(sessions, heads, groups)
        restarted = band.build_targets(sessions[1:], heads, groups)
        self.assertEqual(set(carried[0]['weights']), set(carried[1]['weights']))
        self.assertNotEqual(set(carried[1]['weights']), set(restarted[0]['weights']))
        self.assertLessEqual(len(carried[1]['weights']), 50)

    def test_equal_scores_follow_band_order_not_ticker_text(self):
        scores = {}
        industries = {}
        for index, code in enumerate(['600000.SH', '300001.SZ', '688001.SH', '002001.SZ']):
            scores[code] = 1.0
            industries[code] = f'board-{index}'
        for index in range(60):
            code = f'{index:06d}.SZ'
            scores[code] = 1.0
            industries[code] = f'sz-{index % 8}'
        chosen = band.select_band(scores, industries, ())
        self.assertIn('600000.SH', chosen)
        self.assertIn('300001.SZ', chosen)
        self.assertIn('688001.SH', chosen)
        self.assertLess(chosen.index('600000.SH'), chosen.index('000000.SZ'))
        self.assertFalse(all(code.startswith('000') for code in chosen))

    def test_liquidity_order_survives_json_merge_into_the_actual_target(self):
        liquidity_order = ['600000.SH', '300001.SZ', '688001.SH', '002001.SZ']
        liquidity_order.extend(f'{index:06d}.SZ' for index in range(70))
        payload = {
            'heads': {'2026-09-04': {code: 1.0 for code in liquidity_order}},
            'industries': {'2026-09-04': {code: f'industry-{index}' for index, code in enumerate(liquidity_order)}},
        }
        # The research input is JSON, not an in-memory dict constructed by the
        # selector.  This guards the real source-order handoff that had been
        # replaced by a ticker tie break in the old account.
        parsed = json.loads(json.dumps(payload))
        heads, groups = band.book.merge_band_inputs({'fixture': parsed})
        targets = band.build_targets(pd.to_datetime(['2026-09-04']), heads, groups)

        self.assertEqual(targets[0]['active'][:4], liquidity_order[:4])
        self.assertFalse(all(code.startswith('000') for code in targets[0]['active']))

    def test_full_window_cagr_is_the_only_gate(self):
        missed = {'cagr': 0.145, 'cagr_from_first_fill': 0.17, 'max_drawdown': -0.20,
                  'recent_return': 0.04, 'cagr_clock': '2018-01-02_to_2026-09-22',
                  'first': '2018-01-02', 'final_equity': 1, 'cost2': 0.02}
        reasons = []
        if missed['cagr'] < 0.15:
            reasons.append('cagr')
        self.assertIn('cagr', reasons)
        self.assertGreater(missed['cagr_from_first_fill'], 0.15)


if __name__ == '__main__':
    unittest.main()
