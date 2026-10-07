"""Behavior of the staggered two-leg book. No market download and no production mandate."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import stable_book as book
from tests.fixtures_v2 import dataset


class StableBookTests(unittest.TestCase):
    def test_prepare_requires_an_explicitly_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "archived-research"
            output.mkdir()
            (output / "protocol.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "research_output_not_empty"):
                book.prepare_fresh_output(output)

            fresh = Path(temporary) / "new-research"
            self.assertEqual(book.prepare_fresh_output(fresh), fresh)
            self.assertTrue(fresh.is_dir())

    def test_archived_protocol_source_identity_must_still_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "selector.py"
            source.write_text("rank = input_order\n")
            protocol = {"sources": {str(source): book.sha256(source)}}
            book.verify_protocol_sources(protocol)

            source.write_text("rank = ticker_order\n")
            with self.assertRaisesRegex(ValueError, "protocol_source_changed"):
                book.verify_protocol_sources(protocol)

    def test_explicit_output_rejects_a_path_outside_the_research_output_root(self):
        original = book.OUT
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "research_output_must_be_below_output_root"):
                    book.set_explicit_output(Path(temporary) / "outside")
        finally:
            book.OUT = original

    def test_overlap_mismatch_stops_before_any_target(self):
        left = {'heads': {'2024-01-02': {'000001.SZ': 1.0}}, 'industries': {'2024-01-02': {'000001.SZ': 'bank'}}}
        right = {'heads': {'2024-01-02': {'000001.SZ': 0.0}}, 'industries': {'2024-01-02': {'000001.SZ': 'bank'}}}
        with self.assertRaises(ValueError):
            book.merge_band_inputs({'a': left, 'b': right})

    def test_identical_overlap_is_kept_once(self):
        row = {'heads': {'2024-01-02': {'000001.SZ': 1.0}}, 'industries': {'2024-01-02': {'000001.SZ': 'bank'}}}
        extra = {'heads': {'2024-02-02': {'000002.SZ': 0.0}}, 'industries': {'2024-02-02': {'000002.SZ': 'other'}}}
        heads, industries = book.merge_band_inputs({'a': row, 'b': row, 'c': extra})
        self.assertEqual(set(heads), {'2024-01-02', '2024-02-02'})
        self.assertEqual(industries['2024-01-02']['000001.SZ'], 'bank')

    def test_band_sleeve_keeps_rank_150_and_industry_cap(self):
        codes = [f'{i:06d}.SZ' for i in range(200)]
        scores = {code: 1.0 for code in codes}
        industries = {code: 'one' if i < 40 else str(i) for i, code in enumerate(codes)}
        current = [codes[150], codes[149]]
        chosen = book.select_band_sleeve(scores, industries, current, locked=set(), size=7, retention_rank=150, slots=6)
        self.assertIn(codes[149], chosen)
        self.assertNotIn(codes[150], chosen)
        self.assertEqual(sum(industries[code] == 'one' for code in chosen), 6)
        self.assertEqual(len(chosen), 7)
        self.assertEqual(book.industry_slots(), 6)

    def test_band_sleeve_preserves_liquidity_input_order_for_equal_scores(self):
        liquidity_order = ['600000.SH', '300001.SZ', '688001.SH', '002001.SZ']
        scores = {code: 1.0 for code in liquidity_order}
        industries = {code: f'industry-{index}' for index, code in enumerate(liquidity_order)}

        chosen = book.select_band_sleeve(
            scores,
            industries,
            current=(),
            locked=set(),
            size=len(liquidity_order),
            retention_rank=150,
            slots=1,
        )

        self.assertEqual(chosen, tuple(liquidity_order))

    def test_band_sleeve_keeps_json_head_order_through_canonicalization(self):
        liquidity_order = ['600000.SH', '300001.SZ', '688001.SH', '002001.SZ']
        payload = {
            'heads': {'2026-09-04': {code: 1.0 for code in liquidity_order}},
            'industries': {'2026-09-04': {code: f'industry-{index}' for index, code in enumerate(liquidity_order)}},
        }
        heads, industries = book._band_maps({'fixture': json.loads(json.dumps(payload))})
        chosen = book.select_band_sleeve(
            heads['2026-09-04'],
            industries['2026-09-04'],
            current=(),
            locked=set(),
            size=len(liquidity_order),
            retention_rank=150,
            slots=1,
        )

        self.assertEqual(chosen, tuple(book.canon(code) for code in liquidity_order))

    def test_liquid_sleeve_uses_amount_order_and_skips_locked_names(self):
        ranked = [f'S{i}' for i in range(10)]
        chosen = book.select_liquid_sleeve(ranked, current=['S9'], locked={'S0', 'S1'}, size=3)
        self.assertEqual(chosen, ('S2', 'S3', 'S4'))

    def test_five_phase_clocks_are_four_sessions_apart(self):
        sessions = pd.bdate_range('2018-01-02', periods=120)
        first = book.update_schedule(sessions, 0)
        second = book.update_schedule(sessions, 4)
        self.assertEqual(book.PHASES, (0, 4, 8, 12, 16))
        self.assertEqual(first[0][0], sessions[0])
        self.assertEqual(second[0][0], sessions[4])
        self.assertEqual([book.update_schedule(sessions, phase)[0][0] for phase in book.PHASES],
                         [sessions[phase] for phase in book.PHASES])
        self.assertEqual([sleeve for _day, sleeve in first[:5]], [0, 1, 2, 3, 4])
        heads = {str(sessions[0].date()): {f'{i:06d}.SZ': 1.0 for i in range(80)}}
        industries = {str(sessions[0].date()): {f'{i:06d}.SZ': str(i % 20) for i in range(80)}}
        liquid = {str(day.date()): [f'L{i:03d}' for i in range(40)] for day in sessions}
        targets = book.build_targets(sessions[:21], heads, industries, liquid, 0)
        self.assertEqual([row['sleeve'] for row in targets], [0, 1])
        first_names = set(targets[0]['weights'])
        self.assertTrue(first_names <= set(targets[1]['weights']))
        self.assertEqual(targets[1]['active'], sorted(set(targets[1]['weights']) - first_names))
        self.assertLessEqual(len(targets[1]['weights']), 50)
        self.assertLessEqual(sum(targets[1]['weights'].values()), 0.90 + 1e-9)

    def test_full_cycle_holds_35_and_15_without_using_returns(self):
        sessions = pd.bdate_range('2018-01-02', periods=100)
        day = str(sessions[0].date())
        heads = {day: {f'{i:06d}.SZ': float(i % 2) for i in range(400)}}
        industries = {day: {f'{i:06d}.SZ': str(i % 30) for i in range(400)}}
        liquid = {str(item.date()): [f'L{i:03d}' for i in range(40)] for item in sessions}
        targets = book.build_targets(sessions[:80], heads, industries, liquid, 0)
        self.assertEqual(len(targets), 4)
        last = targets[-1]['weights']
        band = [code for code in last if not code.startswith('L')]
        liquid_names = [code for code in last if code.startswith('L')]
        self.assertEqual(len(band), 28)
        self.assertEqual(len(liquid_names), 12)
        self.assertAlmostEqual(sum(last[code] for code in band), book.LEG_A * 4 / 5, places=10)
        self.assertAlmostEqual(sum(last[code] for code in liquid_names), book.LEG_B * 4 / 5, places=10)

    def test_missing_band_snapshot_leaves_that_leg_in_cash(self):
        sessions = pd.bdate_range('2018-01-02', periods=20)
        liquid = {str(sessions[0].date()): ['L000', 'L001', 'L002', 'L003']}
        targets = book.build_targets(sessions[:1], {}, {}, liquid, 0)
        self.assertEqual(set(targets[0]['weights']), {'L000', 'L001', 'L002'})
        self.assertAlmostEqual(sum(targets[0]['weights'].values()), book.LEG_B / 5)

    def test_execution_weights_do_not_trade_a_sleeve_that_is_not_due(self):
        weights, scale = book.execution_weights(
            {'S000': 0.1, 'S001': 0.1, 'S002': 0.1},
            active=['S002'], sell=[], quantities={'S000': 1000, 'S001': 500},
            marks={'S000': 10.0, 'S001': 20.0, 'S002': 10.0}, nav=100000.0)
        self.assertEqual(scale, 1.0)
        self.assertAlmostEqual(weights['S000'], 0.1)
        self.assertAlmostEqual(weights['S001'], 0.1)
        self.assertAlmostEqual(weights['S002'], 0.1)

    def test_share_lock_survives_the_next_sleeve_fill(self):
        ds = dataset(days=40, stocks=4)
        days = list(ds.calendar[:40])
        targets = [
            {'signal_date': str(days[25].date()), 'weights': {'S000': 0.10, 'S001': 0.10},
             'active': ['S000', 'S001'], 'sell': [], 'sleeve': 0},
            {'signal_date': str(days[32].date()), 'weights': {'S000': 0.10, 'S001': 0.10, 'S002': 0.10},
             'active': ['S002'], 'sell': [], 'sleeve': 1},
        ]
        result = book.replay_account(ds, targets, start=days[0], end=days[-1])
        self.assertTrue(result['summary']['account_checks_passed'])
        quantities = {pd.Timestamp(state['date']): state['quantities'] for state in result['account']}
        before = quantities[days[31]]
        after = quantities[days[33]]
        self.assertGreater(before.get('S000', 0), 0)
        self.assertEqual(before['S000'], after['S000'])
        self.assertEqual(before['S001'], after['S001'])
        self.assertGreater(after.get('S002', 0), 0)
        self.assertLessEqual(result['summary']['max_holdings'], 50)

    def _five(self, cagrs, drawdown=-0.20, recent=0.04):
        names = book.phase_names()
        return {name: {'cagr': cagr, 'max_drawdown': drawdown, 'recent_return': recent}
                for name, cagr in zip(names, cagrs)}

    def test_gate_uses_median_and_range_not_the_best_phase(self):
        names = book.phase_names()
        cost2 = {name: {'cagr': 0.01} for name in names}
        wide = book.judge(self._five([0.16, 0.16, 0.16, 0.16, 0.30]), cost2)
        self.assertEqual(wide['decision'], 'combination_did_not_reach_stable_15')
        self.assertIn('phase_range', wide['reasons'])
        self.assertNotIn('median_cagr', wide['reasons'])
        low = book.judge(self._five([0.12, 0.13, 0.14, 0.15, 0.16]), cost2)
        self.assertIn('median_cagr', low['reasons'])
        broken = self._five([0.16, 0.17, 0.18, 0.19, 0.20])
        broken['phase0']['max_drawdown'] = -0.40
        self.assertIn('phase0_drawdown', book.judge(broken, cost2)['reasons'])
        passed = book.judge(self._five([0.15, 0.16, 0.17, 0.18, 0.19]), cost2)
        self.assertEqual(passed['decision'], 'research_candidate_not_promoted')
        self.assertAlmostEqual(passed['median_cagr'], 0.17)
        self.assertFalse(passed['promotion_approved'])
        weak_cost = dict(cost2)
        weak_cost['phase16'] = {'cagr': -0.01}
        doubled = book.judge(self._five([0.16, 0.16, 0.17, 0.17, 0.18]), weak_cost)
        self.assertEqual(doubled['decision'], 'not_retained_cost_double_not_positive')

    def test_qualification_is_the_existing_liquidity_screen(self):
        cfg = book.liquidity_config()
        self.assertEqual(cfg.research.universe_mode, 'pit_liquidity')
        self.assertEqual(cfg.universe.a_min_avg_amount_wan, 500)
        self.assertEqual(cfg.universe.a_min_listing_days, 60)
        self.assertTrue(cfg.universe.a_exclude_st)


if __name__ == '__main__':
    unittest.main()
