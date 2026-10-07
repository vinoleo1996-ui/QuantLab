"""Research-only cadence contracts; frozen Engine executes in a fresh process."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import continuity_cadence as cadence


class CadenceTests(unittest.TestCase):
    def test_paired_evidence_refuses_missing_sessions_and_labels_rising_day_differences(self):
        import pandas as pd
        dates = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06'])
        runs = {'qP2_monthly': {'daily': pd.Series([.01, -.02, 0.], index=dates)},
                'cadence13': {'daily': pd.Series([.005, -.01, .001], index=dates)}}
        differences, descriptive = cadence.paired_evidence(runs)
        groups = descriptive['cadence13-minus-monthly']['groups']
        self.assertEqual(groups['baseline_rising_days']['days'], 1)
        self.assertAlmostEqual(groups['baseline_rising_days']['mean_daily_net_difference'], -.005)
        self.assertAlmostEqual(differences.iloc[1, 0], .01)
        runs['cadence13']['daily'] = runs['cadence13']['daily'].iloc[1:]
        with self.assertRaisesRegex(ValueError, 'paired_return_calendar_mismatch'):
            cadence.paired_evidence(runs)

    def test_result_identity_rejects_wrong_policy_or_holding_variant(self):
        import copy
        config = {'capital': {'init_cash': 600000, 'min_holding_period_days': 20}}
        protocol = {'effective_config': config}
        result = dict(config=config, data_version=cadence.DATA_VERSION, candidate='quality_pb',
                      risk_policy='P2', allocation_policy='legacy', data_mode='free_research',
                      industry_policy='enforced', strategy='legacy_repaired', promotion_approved=False,
                      status='unverified_data_gaps')
        cadence.validate_result(result, protocol, 'qP2_monthly')
        with self.assertRaisesRegex(ValueError, 'result_config_mismatch'):
            cadence.validate_result(result, protocol, 'qP2_monthly_hold90')
        altered = copy.deepcopy(result)
        altered['risk_policy'] = 'P0'
        with self.assertRaisesRegex(ValueError, 'result_identity_mismatch'):
            cadence.validate_result(altered, protocol, 'qP2_monthly')

    def test_review_audit_explains_preemption_but_flags_missing_or_extra_decisions(self):
        run = dict(trades=[dict(signal_date='2020-01-03', reason='risk_exit'),
                           dict(signal_date='2020-02-01', reason='legacy_repaired')],
                   signal_trace=[dict(date='2020-01-10', status='warmup', reasons=['history'])],
                   account=[dict(date=d) for d in ('2020-01-03', '2020-01-10', '2020-01-17')])
        audit = cadence.review_audit(run, ['2020-01-03', '2020-01-10', '2020-01-17'])
        self.assertEqual(audit['accounted_count'], 2)
        self.assertEqual(audit['unexplained_dates'], ['2020-01-17'])
        self.assertEqual(audit['unexpected_ordinary_dates'], ['2020-02-01'])
        self.assertEqual(audit['deviations'][0]['reason'], 'risk_exit_preempted_ordinary_review')

    def test_attempt_reservation_rejects_replay_even_after_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run13'
            cadence.reserve_attempt(path)
            with self.assertRaisesRegex(ValueError, 'existing_run_attempt'):
                cadence.reserve_attempt(path)

    def test_net_metrics_use_initial_cash_and_prior_stage_boundary_without_cost_addback(self):
        import pandas as pd
        snapshots = [dict(date=d, nav=n, cash=n / 2, weights={'S': .5}, quantities={'S': 1})
                     for d, n in [('2020-01-02', 594000), ('2021-12-31', 660000),
                                  ('2022-01-04', 660000), ('2023-12-29', 528000),
                                  ('2024-01-02', 528000), ('2025-12-31', 633600),
                                  ('2026-01-05', 633600), ('2026-06-30', 696960)]]
        run = dict(account=snapshots, config={'capital': {'init_cash': 600000}},
                   nav=pd.Series([s['nav'] / 600000 for s in snapshots],
                                 index=pd.to_datetime([s['date'] for s in snapshots])),
                   fills=[dict(reason='legacy_repaired', fees=10, slippage_cost=20, notional=1000),
                          dict(reason='risk_exit', fees=5, slippage_cost=7, notional=200)],
                   status='unverified_data_gaps', data_issues=[{'reason': 'fixture_gap'}])
        summary, daily = cadence.normalized_metrics(run)
        self.assertAlmostEqual(summary['total_net_return'], .1616)
        self.assertAlmostEqual(daily.iloc[0], -.01)
        self.assertAlmostEqual(summary['stages']['2022-23']['net_return'], -.2)
        self.assertAlmostEqual(summary['stages']['2026H1']['net_return'], .1)
        self.assertAlmostEqual(summary['max_drawdown'], -.2)
        self.assertEqual(summary['worst_stage'], '2022-23')
        self.assertEqual(summary['costs']['ordinary']['fees'], 10)
        self.assertEqual(summary['costs']['risk_exit']['slippage'], 7)
        self.assertEqual(summary['fees'], 15)
        self.assertFalse(summary['promotion_approved'])
        self.assertNotIn('gross_cagr', summary)

    def test_frozen_protocol_cannot_change_and_tampered_pickle_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = root / 'protocol.json'
            cadence.freeze_json(protocol, {'harness': 'v1'})
            with self.assertRaisesRegex(ValueError, 'immutable_output_exists'):
                cadence.freeze_json(protocol, {'harness': 'v2'})
            self.assertIn('v1', protocol.read_text())
            artifact = root / 'result.pkl'
            artifact.write_bytes(b'not-a-pickle')
            with self.assertRaisesRegex(ValueError, 'artifact_checksum_mismatch'):
                cadence.checked_pickle(artifact, '0' * 64)

    def test_real_frozen_engine_keeps_monthly_preparation_and_overrides_reviews(self):
        program = r'''
import sys
sys.path.insert(0, 'scripts')
import continuity_cadence as c
c.load_frozen()
sys.path.insert(0, str(c.FROZEN / 'tests'))
from fixtures_v2 import dataset
from quantlab.backtest.engine import Engine, prepare
from quantlab.strategy.decision import review_dates
from copy import deepcopy
import pandas as pd
ds = dataset(days=430, stocks=20)
# Free-research admission rejects the reserved TEST_FIXTURE provider. This
# explicitly named synthetic provider exercises that mode, with no real data.
ds.manifest['source'] = 'synthetic_cadence_contract_test'
ds.instruments['sector'] = ds.instruments.instrument_id
ds.financials = pd.DataFrame([dict(instrument_id=code, available_at='2022-01-01',
    effective_at='2022-01-01', source='TEST_FIXTURE', revision='1', quality_status='ok',
    metric='net_profit', value=1e8, announced_on='2022-01-01', report_period='2021-12-31') for code in ds.panels])
for panel in ds.panels.values():
    panel['pb_mrq'] = 1.
cfg = c.effective_config()
start, end = ds.calendar[280], ds.calendar[425]
bad = deepcopy(cfg)
bad.capital.rebalance_rule = 'every_n_weeks'
bad.capital.rebalance_n_weeks = 13
try:
    Engine(bad).run_dataset(ds, strategy='legacy_repaired', risk_policy='P2',
        prepared=prepare(ds, bad), start=start, end=end)
except ValueError as exc:
    assert str(exc) == 'prepared_context_version_mismatch', str(exc)
else:
    raise AssertionError('old mismatch not reproduced')
result, audit = c.replay13(ds, cfg, start=start, end=end)
assert 'config' in result, (result['status'], result['reasons'])
assert result['config']['capital']['rebalance_rule'] == 'monthly'
assert result['config']['capital']['rebalance_n_weeks'] == 4
assert list(result['ic_matrix'].index) == review_dates(ds.calendar, 'monthly')
expected = [str(d.date()) for d in review_dates(ds.calendar, 'every_n_weeks', 13) if start <= d <= end]
assert audit['requested_dates'] == expected
assert audit['requested_count'] >= 2
assert audit['accounted_count'] == len(expected), audit
assert not audit['unexpected_ordinary_dates'], audit
assert not audit['unexplained_dates'], audit
assert result['account'] and result['risk_policy'] == 'P2'
print('legacy mismatch reproduced; monthly IC retained; 13-week review dates accounted')
'''
        completed = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == '__main__':
    unittest.main()
