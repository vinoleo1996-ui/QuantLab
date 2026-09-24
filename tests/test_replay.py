import unittest
from copy import deepcopy
from pathlib import Path
import tempfile

import pandas as pd

from fixtures_v2 import dataset
from quantlab.config import CONFIG
from quantlab.backtest.engine import Engine, prepare
from quantlab.data.contracts import asof_records
from quantlab.data.store import DataStore
from quantlab.cli import signal_payload
from quantlab.report.metrics import performance_table
from quantlab.strategy.decision import review_dates


class ReplayContracts(unittest.TestCase):
    def test_schedule_replay_and_outgoing_signals(self):
        ds = dataset(); cfg = deepcopy(CONFIG)
        cfg.universe.a_min_listing_days = 1
        cfg.start_date = str(ds.calendar[20].date())
        schedule = {ds.calendar[21]: {"S000": .05}, ds.calendar[40]: {"S001": .05}, ds.calendar[69]: {}}
        result = Engine(cfg).run(ds, strategy="schedule", target_schedule=schedule)
        self.assertEqual(result["status"], "research_unvalidated")
        self.assertTrue(result["fills"])
        self.assertTrue(all(pd.Timestamp(f["date"]) > pd.Timestamp(f["signal_date"]) for f in result["fills"]))
        self.assertGreater(result["account"][-1]["quantities"]["S001"], 0)
        payload = signal_payload(result)
        self.assertTrue(any(row[0]=="S001" and row[1]==0 for row in payload["trades"]))
        self.assertFalse(payload["executable"])
        self.assertTrue(all(a["cash"]>=0 for a in result["account"]))

    def test_no_signal_keeps_pnl_and_missing_quote_invalidates_metrics(self):
        ds = dataset(); cfg = deepcopy(CONFIG); cfg.start_date = str(ds.calendar[20].date())
        ds.panels["S000"].loc[ds.calendar[40], "close"] = float("nan")
        result = Engine(cfg).run(ds, strategy="schedule", target_schedule={ds.calendar[21]: {"S000": .05}})
        self.assertEqual(result["status"], "unverified_data_gaps")
        self.assertNotEqual(result["nav"].iloc[-1], result["nav"].loc[ds.calendar[22]])

    def test_bare_legacy_panels_blocked(self):
        result = Engine().run(dataset().panels)
        self.assertEqual(result["status"], "blocked_data")
        self.assertIsNone(result["metrics"])

    def test_future_rows_do_not_change_past_fills(self):
        ds = dataset(); cfg = deepcopy(CONFIG); cfg.start_date = str(ds.calendar[20].date())
        schedule = {ds.calendar[21]: {"S000": .05}, ds.calendar[35]: {"S001": .04}}
        full = Engine(cfg).run(ds, strategy="schedule", target_schedule=schedule)
        cut = Engine(cfg).run(ds, strategy="schedule", target_schedule=schedule, end=ds.calendar[45])
        pd.testing.assert_series_equal(full["nav"].loc[:ds.calendar[45]], cut["nav"])

    def test_actual_calendar_last_bar_not_assumed_period_end(self):
        cal = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"])
        self.assertEqual(review_dates(cal,"monthly"),[pd.Timestamp("2024-01-31")])
        self.assertEqual(review_dates(cal,"weekly"),[pd.Timestamp("2024-02-02")])

    def test_announcement_revision_availability(self):
        h = dataset().instruments.iloc[:1].copy()
        correction = h.copy(); correction["available_at"]="2024-01-01"; correction["sector"]="new"
        h = pd.concat([h, correction])
        self.assertEqual(asof_records(h,"2023-01-01").iloc[0].sector,"sector0")
        self.assertEqual(asof_records(h,"2024-01-02").iloc[0].sector,"new")

    def test_saved_and_monitored_run_identity_agrees(self):
        from quantlab.cli import run_backtest
        from quantlab.strategy.monitor import daily_expert_pulse
        with tempfile.TemporaryDirectory() as d:
            missing = str(Path(d)/"missing.json")
            result = run_backtest(save=False,manifest=missing)
            pulse = daily_expert_pulse(manifest=missing)
            self.assertEqual(result["run_id"],pulse["run_id"])
            self.assertFalse(pulse["executable"])

    def test_first_day_fees_are_in_total_return(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        r = pd.Series([-.01, 0., 0.],index=idx)
        self.assertAlmostEqual(performance_table((1+r).cumprod(),r)["total_return"],-.01)


class CacheContracts(unittest.TestCase):
    def test_source_degradation_rejected_versions_and_root_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            store = DataStore(Path(d))
            data = pd.DataFrame({"close": [10.,11.],"amount":[1e8,1e8]}, index=pd.bdate_range("2024-01-01",periods=2))
            data.attrs.update(source="complete",price_basis="raw")
            store.save(data,"a","daily_raw_A")
            bad = data.copy(); bad["amount"]=float("nan");bad.attrs["source"]="incomplete"
            with self.assertRaisesRegex(ValueError,"source_quality_regression"):store.save(bad,"a","daily_raw_A")
            self.assertEqual(store.load("a","daily_raw_A").amount.notna().sum(),2)
            self.assertTrue(list((Path(d)/"a_stock"/"versions").rglob("*.json")))
            self.assertTrue(store._path("a","daily_raw_A").is_relative_to(Path(d)))


if __name__ == "__main__": unittest.main()
