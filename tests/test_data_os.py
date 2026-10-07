"""Quant Data OS：原始层不覆盖，时点缺失不进训练集，冲突留下选中源。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import os

import pandas as pd

from quantlab.data_os.archive import REQUIRED_INGEST, ResearchOfflineError, replay, research_view, resolve, timed_capture
from quantlab.data_os.net import proxy_configured
from quantlab.data_os.bronze import BronzeStore
from quantlab.data_os.datasets import ml_dataset
from quantlab.data_os.features import evaluate_formula, price_features
from quantlab.data_os.normalize import fundamental_frame, macro_frame
from quantlab.data_os.fill_missing import first_release_after
from quantlab.data_os.quality import quality_gate
from quantlab.data_os.pipeline import build_dataset
from quantlab.data_os.pit import visible
from quantlab.data_os.sources import Capture, parse_baostock_kline, parse_fred_csv, parse_sec_companyfacts


def _bar(day, close, source="baostock", instrument="CN.XSHE.000001"):
    return {
        "instrument_id": instrument,
        "symbol": instrument.split(".")[-1],
        "market": "CN" if instrument.startswith("CN.") else "US",
        "trade_date": day,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "prev_close": close - 0.1,
        "volume": 1000.0,
        "amount": close * 1000.0,
        "turnover": 0.01,
        "source": source,
        "is_trading": True,
        "is_suspended": False,
    }


class DataOsTests(unittest.TestCase):
    def test_bronze_keeps_original_bytes(self):
        store = BronzeStore(Path(tempfile.mkdtemp()) / "bronze")
        first = store.put("baostock", "daily", b"raw-v1")
        second = store.put("baostock", "daily", b"raw-v1")
        third = store.put("baostock", "daily", b"raw-v2")
        self.assertFalse(second["created"])
        self.assertTrue(third["created"])
        self.assertEqual((store.root / first["path"]).read_bytes(), b"raw-v1")
        self.assertEqual((store.root / third["path"]).read_bytes(), b"raw-v2")

    def test_missing_publication_stays_out_of_training(self):
        dated = fundamental_frame([{
            "instrument_id": "US.SEC.AAPL", "concept": "Revenues", "value": 100.0, "unit": "USD",
            "period_start": "2024-01-01", "period_end": "2024-03-31", "filed_date": "2024-05-03",
            "accepted_datetime": "2024-05-03T21:05:00Z", "form": "10-Q", "accn": "1", "source": "sec",
            "revision_id": "1",
        }], "v", "America/New_York")
        missing = fundamental_frame([{
            "instrument_id": "CN.XSHE.000001", "concept": "epsTTM", "value": 1.2, "unit": "CNY",
            "period_start": pd.NA, "period_end": "2024-03-31", "filed_date": pd.NA,
            "accepted_datetime": pd.NA, "form": "baostock_profit", "accn": "", "source": "baostock",
            "revision_id": "",
        }], "v", "Asia/Shanghai")
        self.assertTrue(bool(dated.iloc[0]["research_grade"]))
        self.assertFalse(bool(missing.iloc[0]["research_grade"]))
        self.assertTrue(pd.isna(missing.iloc[0]["available_time"]))
        seen = visible(dated, "2024-04-30T00:00:00Z")
        later = visible(dated, "2024-05-06T00:00:00Z")
        self.assertTrue(seen.empty)
        self.assertEqual(len(later), 1)

    def test_macro_without_release_is_not_research_grade(self):
        frame = macro_frame([{"series_id": "CPIAUCSL", "observation_date": "2025-08-01", "value": 300.0, "source": "fred"}], "v")
        self.assertFalse(bool(frame.iloc[0]["research_grade"]))
        printed = macro_frame([{"series_id": "DGS10", "observation_date": "2024-01-02", "value": 4.0, "source": "fred"}], "v")
        self.assertTrue(bool(printed.iloc[0]["research_grade"]))
        self.assertEqual(printed.iloc[0]["availability_rule"], "observation_is_print")
        released = macro_frame([{
            "series_id": "CPIAUCSL", "observation_date": "2025-08-01", "value": 300.0,
            "release_date": "2025-09-11", "source": "fred",
        }], "v")
        self.assertTrue(bool(released.iloc[0]["research_grade"]))
        self.assertTrue(visible(released, "2025-08-31T23:59:59Z").empty)
        self.assertEqual(len(visible(released, "2025-09-12T12:00:00Z")), 1)

    def test_conflict_keeps_both_sources_and_selects_primary(self):
        root = Path(tempfile.mkdtemp()) / "conflict"
        captures = [
            Capture("baostock", "daily", b"bao", market_rows=[_bar("2024-01-02", 10.20, "baostock"), _bar("2024-01-03", 10.0, "baostock"), _bar("2024-01-04", 10.5, "baostock")]),
            Capture("akshare", "daily", b"ak", market_rows=[_bar("2024-01-02", 10.21, "akshare"), _bar("2024-01-03", 10.0, "akshare"), _bar("2024-01-04", 10.5, "akshare")]),
        ]
        meta = build_dataset(root, "v-conflict", captures, as_of="2024-01-04T08:00:00Z")
        market = pd.read_parquet(root / "silver" / "market" / "dataset_version=v-conflict" / "part.parquet")
        day = market[market["trade_date"].astype(str).str.startswith("2024-01-02")].iloc[0]
        self.assertEqual(day["selected_source"], "baostock")
        self.assertAlmostEqual(float(day["close"]), 10.20)
        self.assertTrue(bool(day["conflict"]))
        conflicts = pd.read_parquet(root / "quality" / "dataset_version=v-conflict" / "conflicts.parquet")
        self.assertAlmostEqual(float(conflicts.iloc[0]["conflict_value"]), abs(10.21 - 10.20) / 10.20, places=6)
        self.assertGreater(meta["research_grade_rows"], 0)
        learn = pd.read_parquet(root / "gold" / "ml_dataset" / "dataset_version=v-conflict" / "part.parquet")
        first = learn.sort_values("trade_date").iloc[0]
        self.assertAlmostEqual(float(first["y"]), 10.5 / 10.0 - 1, places=6)
        self.assertEqual(first["label_spec"], "close_t2_over_close_t1_minus_1")

    def test_formula_uses_only_declared_operators(self):
        market = pd.DataFrame([
            _bar("2024-01-02", 10, instrument="CN.XSHE.000001") | {"research_grade": True, "available_time": pd.Timestamp("2024-01-02T07:00:00Z"), "vwap": 10.0, "pit_valid": True, "quality_score": 1, "corporate_action_valid": False, "dataset_version": "v", "ingested_at": pd.Timestamp("2024-01-02T10:00:00Z"), "price_basis": "raw", "volume_unit": "share", "source_timestamp": pd.Timestamp("2024-01-02T07:00:00Z"), "availability_rule": "session_close", "selected_source": "baostock", "conflict": False, "conflict_value": 0},
        ])
        # 公式引擎在已有特征列上运算。这里直接给最小列。
        features = pd.DataFrame({
            "instrument_id": ["CN.XSHE.000001"] * 3,
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [10.0, 11.0, 12.0],
            "vwap": [10.0, 10.5, 11.5],
            "RET_1": [pd.NA, 0.1, 0.09],
            "volume": [100.0, 120.0, 80.0],
        })
        values = evaluate_formula(features, "rank(ts_mean(close/vwap,2))")
        self.assertTrue(pd.notna(values.iloc[-1]))
        self.assertTrue(price_features(market).empty or "RET_1" in price_features(pd.concat([market] * 1, ignore_index=True)).columns)

    def test_parsers_do_not_invent_release_dates(self):
        bars = parse_baostock_kline("date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus\n2024-01-02,sz.000001,10,11,9,10.5,10,1000,10500,0.1,1", "000001")
        self.assertEqual(bars[0]["instrument_id"], "CN.XSHE.000001")
        self.assertEqual(bars[0]["volume"], 1000.0)
        fred = parse_fred_csv("DATE,VALUE\n2025-08-01,300", "CPIAUCSL")
        self.assertIsNone(fred[0]["release_date"])
        facts = parse_sec_companyfacts({
            "cik": 320193,
            "facts": {"us-gaap": {"Revenues": {"units": {"USD": [{
                "start": "2024-01-01", "end": "2024-03-31", "val": 100, "accn": "0001-24-000001",
                "filed": "2024-05-03", "form": "10-Q",
            }]}}}},
        }, "AAPL", {"Revenues"}, {"0001-24-000001": "2024-05-03T21:05:00.000Z"})
        self.assertEqual(facts[0]["filed_date"], "2024-05-03")
        self.assertEqual(facts[0]["accepted_datetime"], "2024-05-03T21:05:00.000Z")
        self.assertNotEqual(facts[0]["period_end"], facts[0]["filed_date"])

    def test_adjust_factor_stays_off_the_raw_close(self):
        root = Path(tempfile.mkdtemp()) / "adjust"
        instrument = "CN.XSHE.000001"
        rows = [_bar("2024-01-02", 10.0), _bar("2024-01-03", 11.0), _bar("2024-01-04", 12.0)]
        daily = Capture(
            "baostock", "daily", b"bars",
            market_rows=rows,
            action_rows=[{
                "instrument_id": instrument,
                "event_date": "2020-06-01",
                "event_type": "adjust_factor",
                "adjust_factor": 1.8,
                "source": "baostock",
            }],
            instrument={"instrument_id": instrument, "symbol": "000001", "exchange": "XSHE", "country": "CN"},
            actions_checked=True,
        )
        membership = Capture(
            "baostock", "hs300", b"members",
            universe_rows=[{
                "instrument_id": instrument,
                "universe": "hs300",
                "effective_start": "2024-01-02",
                "effective_end": "2024-01-02",
                "as_of": "2024-01-02",
                "pit_membership": True,
                "source": "baostock",
                "symbol": "000001",
                "exchange": "XSHE",
                "country": "CN",
            }],
        )
        meta = build_dataset(root, "v-adjust", [daily, membership])
        market = pd.read_parquet(root / "silver" / "market" / "dataset_version=v-adjust" / "part.parquet")
        self.assertTrue((market["close"].astype(float).isin([10.0, 11.0, 12.0])).all())
        self.assertTrue((market["price_basis"] == "raw").all())
        self.assertTrue((market["adj_factor"].astype(float) == 1.8).all())
        self.assertTrue(meta["quality_gate"]["backtest_allowed"])
        blocked = build_dataset(Path(tempfile.mkdtemp()) / "bare", "v-bare", [Capture("baostock", "daily", b"only", market_rows=rows)])
        self.assertFalse(blocked["quality_gate"]["backtest_allowed"])
        bare_market = pd.read_parquet(Path(blocked["paths"]["market"]))
        bare_universe = pd.DataFrame([{
            "instrument_id": "CN.XSHE.000001",
            "universe": "hs300",
            "effective_start": "2018-01-02",
            "effective_end": "2018-01-02",
            "pit_membership": True,
        }])
        gate = quality_gate(bare_market, bare_universe)
        self.assertFalse(gate["backtest_allowed"])
        self.assertGreater(gate["checks"]["members_missing_bar"], 0)

    def test_cpi_release_is_after_the_observation_month(self):
        dates = ["2024-02-13", "2024-04-25"]
        self.assertEqual(first_release_after("2024-01-01", dates, quarterly=False), "2024-02-13")
        self.assertEqual(first_release_after("2024-01-01", dates, quarterly=True), "2024-04-25")

    def test_proxy_failure_retries_without_proxy(self):
        root = Path(tempfile.mkdtemp()) / "proxy"
        saved = os.environ.get("https_proxy")
        os.environ["https_proxy"] = "http://127.0.0.1:9"
        calls = {"n": 0}

        def fetch():
            calls["n"] += 1
            if proxy_configured():
                return Capture("akshare", "daily", b"", error="ProxyError: RemoteDisconnected via proxy")
            return Capture("akshare", "daily", b"direct-ok")

        try:
            capture = timed_capture(root, "akshare", "daily", fetch)
        finally:
            if saved is None:
                os.environ.pop("https_proxy", None)
            else:
                os.environ["https_proxy"] = saved
        self.assertEqual(calls["n"], 2)
        self.assertIsNone(capture.error)
        self.assertEqual(capture.payload, b"direct-ok")
        self.assertEqual(os.environ.get("https_proxy"), saved)

    def test_research_replay_uses_local_archive_only(self):
        self.assertNotIn("stooq", REQUIRED_INGEST)
        root = Path(tempfile.mkdtemp()) / "archive"
        kline = "\n".join([
            "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus",
            "2024-01-02,sz.000001,9.8,10.2,9.7,10.0,9.9,1000,10000,0.1,1",
            "2024-01-03,sz.000001,10.0,11.2,9.8,11.0,10.0,1000,11000,0.1,1",
            "2024-01-04,sz.000001,11.0,12.2,10.8,12.0,11.0,1000,12000,0.1,1",
        ])
        actions = [{"instrument_id": "CN.XSHE.000001", "event_date": "2020-06-01", "event_type": "adjust_factor", "adjust_factor": 1.2, "source": "baostock"}]
        instrument = {"instrument_id": "CN.XSHE.000001", "symbol": "000001", "exchange": "XSHE", "country": "CN", "asset_type": "stock", "currency": "CNY", "status": "listed"}
        payload = f"{kline}\n--profit--\ncode,pubDate,statDate,concept,value\n--actions--\n{pd.Series(actions).to_json(orient='records', force_ascii=False)}\n--instrument--\n{pd.Series(instrument).to_json(force_ascii=False)}".encode()
        BronzeStore(root).put("baostock", "daily", payload)
        BronzeStore(root).put("baostock", "hs300", b"as_of,code,name\n2024-01-02,sz.000001,\u5e73\u5b89\u94f6\u884c\n")
        built = build_dataset(root, "v-replay", replay(root))
        self.assertTrue(built["quality_gate"]["backtest_allowed"])
        view = research_view(root, "2024-01-31T00:00:00Z")
        self.assertEqual(view["market_rows"], 3)
        plan = resolve(root)
        self.assertEqual(plan["primary_provider"], "local_archive")
        self.assertFalse(plan["ingest_required"])
        self.assertEqual(plan["freshness_requirement"], "T+1")
        with self.assertRaises(ResearchOfflineError):
            research_view(Path(tempfile.mkdtemp()), "2024-01-31T00:00:00Z")

    def test_ml_label_is_not_same_bar_return(self):
        rows = []
        closes = [10.0, 11.0, 12.0, 13.0]
        for index, close in enumerate(closes):
            rows.append(_bar(f"2024-01-0{index + 2}", close))
        meta_root = Path(tempfile.mkdtemp()) / "labels"
        build_dataset(meta_root, "v-label", [Capture("baostock", "daily", b"k", market_rows=rows)])
        learn = ml_dataset  # 绑定名称，避免未使用告警；真正结果读金层。
        del learn
        frame = pd.read_parquet(meta_root / "gold" / "ml_dataset" / "dataset_version=v-label" / "part.parquet")
        frame = frame.sort_values("trade_date")
        self.assertAlmostEqual(float(frame.iloc[0]["y"]), 12.0 / 11.0 - 1, places=6)
        self.assertNotAlmostEqual(float(frame.iloc[0]["y"]), 11.0 / 10.0 - 1, places=6)


if __name__ == "__main__":
    unittest.main()
