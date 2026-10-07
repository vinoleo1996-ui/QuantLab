import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import quantlab.webapp.app as webapp


class PortfolioHubTests(unittest.TestCase):
    def setUp(self):
        webapp.app.config.update(TESTING=True)
        self.client = webapp.app.test_client()

    def test_portfolio_shell_is_separate_from_research_hub(self):
        response = self.client.get("/portfolio")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('/static/portfolio_terminal.css', html)
        self.assertIn('/static/portfolio_terminal_adapter.js', html)
        self.assertIn('/static/portfolio_terminal.js', html)
        self.assertLess(
            html.index('/static/portfolio_terminal_adapter.js'),
            html.index('/static/portfolio_terminal.js'),
        )
        self.assertNotIn('portfolio_hub.js', html)
        self.assertNotIn('dashboard.js', html)

    def test_terminal_keeps_position_details_and_sort_keys_visible(self):
        """The recovered terminal must expose the ledger fields users need to inspect."""

        static_root = Path(webapp.app.static_folder or "")
        terminal = (static_root / "portfolio_terminal.js").read_text(encoding="utf-8")
        shell = (static_root / "portfolio_terminal.html").read_text(encoding="utf-8")

        self.assertIn("statChip('持仓数量'", terminal)
        self.assertIn('data-qty=', terminal)
        self.assertIn('data-key="since"', shell)
        self.assertIn('data-key="pnl"', shell)
        self.assertIn('data-key="mv"', shell)
        self.assertIn('id="enablePublicQuotes"', shell)
        self.assertIn('id="symbolSearch"', shell)
        self.assertIn('id="holdPnl"', shell)
        self.assertIn('data-overlay="trend"', shell)
        self.assertIn('data-overlay="ma"', shell)
        self.assertIn('data-overlay="hs"', shell)
        self.assertIn('data-freq="year"', shell)
        self.assertIn('近15年', shell)
        self.assertNotIn('开盘后最新价会更新最后一根日K', shell)
        self.assertIn('let publicQuotesEnabled = true;', terminal)
        self.assertIn('updatePublicQuoteControls(publicQuotesEnabled);', terminal)
        self.assertIn('class="tab active" data-pane="holdings-pane"', shell)
        self.assertIn('<div class="pane active" id="holdings-pane">', shell)
        self.assertNotIn('window.confirm(', terminal)
        self.assertNotIn("statChip('所属板块', esc(sector))", terminal)

    def test_terminal_nav_and_series_read_the_grok_paper_feed_not_latest(self):
        with patch.object(webapp, "_load", side_effect=AssertionError("latest.json must not be read")):
            nav_response = self.client.get("/api/portfolio/terminal/nav")
            series_response = self.client.get("/api/portfolio/terminal/series")

        self.assertEqual(nav_response.status_code, 200)
        self.assertEqual(series_response.status_code, 200)
        nav = nav_response.get_json()
        series = series_response.get_json()
        self.assertEqual(nav["status"], "paper_simulation_unverified")
        self.assertTrue(nav["interpretable"])
        self.assertIn("metrics", nav)
        self.assertTrue(series["available"])
        self.assertEqual(len(series["dates"]), len(series["nav"]))
        self.assertEqual(len(series["nav"]), len(series["drawdown"]))

    def test_terminal_kline_uses_frequency_and_terminal_bar_shape(self):
        frame = pd.DataFrame([
            {"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 1000.0, "amount": 10000.0},
            {"open": 10.5, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 1200.0, "amount": 13200.0},
            {"open": 11.0, "high": 13.0, "low": 10.5, "close": 12.5, "volume": 1500.0, "amount": 18750.0},
            {"open": 12.5, "high": 14.0, "low": 12.0, "close": 13.0, "volume": 1700.0, "amount": 22100.0},
        ], index=pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-09-28"]))
        with patch.object(webapp, "_load_public_kline", return_value=(frame, "eastmoney", [])):
            response = self.client.get("/api/portfolio/kline?code=600000&freq=year&bars=0")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["frequency"], "year")
        self.assertEqual(len(payload["bars"]), 2)
        self.assertEqual(set(payload["bars"][-1]), {"d", "o", "c", "l", "h", "v", "a"})
        self.assertEqual(payload["bars"][-1]["o"], 11.0)
        self.assertEqual(payload["bars"][-1]["c"], 13.0)
        self.assertEqual(payload["adjust"], "raw")
        self.assertEqual(payload["adjustment"], "raw")
        self.assertEqual(payload["quote_state"], "closed")

    def test_intraday_quote_state_requires_a_fresh_current_session_timestamp(self):
        tz = ZoneInfo("Asia/Shanghai")
        now = datetime(2026, 9, 28, 14, 56, tzinfo=tz)
        self.assertEqual(webapp._quote_state_from_timestamp("2026-09-28 14:52", now=now), "trading")
        self.assertEqual(webapp._quote_state_from_timestamp("2026-09-28 14:40", now=now), "delayed")
        self.assertEqual(
            webapp._quote_state_from_timestamp("2026-09-28 14:59", now=datetime(2026, 9, 28, 15, 1, tzinfo=tz)),
            "closed",
        )
        minute = {
            "rows": [{"t": "2026-09-28 14:40", "p": 10.0, "v": 200.0, "pre": 9.8}],
            "pre_close": 9.8,
            "last": 10.0,
            "quote_time": "2026-09-28 14:40",
        }
        with patch.object(webapp, "_fetch_tencent_intraday", return_value=minute), patch.object(
            webapp,
            "_current_shanghai_time",
            return_value=now,
        ):
            payload = self.client.get("/api/portfolio/intraday?code=600905").get_json()
        self.assertEqual(payload["quote_state"], "delayed")

    def test_terminal_kline_marks_the_fixed_local_panel_as_stale_fallback(self):
        frame = pd.DataFrame([
            {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 1000.0, "amount": 10200.0},
            {"open": 10.2, "high": 10.8, "low": 10.0, "close": 10.7, "volume": 1100.0, "amount": 11770.0},
        ], index=pd.to_datetime(["2026-06-29", "2026-06-30"]))
        with patch.object(webapp, "_load_public_kline", return_value=(pd.DataFrame(), None, ["public:timeout"])), patch.object(
            webapp,
            "_load_local_market_daily",
            return_value=frame,
        ):
            response = self.client.get("/api/portfolio/kline?code=600000&freq=day&bars=120")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"], "local_panel_unverified")
        self.assertEqual(payload["source_kind"], "local_panel")
        self.assertEqual(payload["adjust"], "none")
        self.assertEqual(payload["as_of"], "2026-06-30")

    def test_terminal_kline_labels_an_incomplete_local_year_bar_with_its_real_as_of_date(self):
        """A local fallback must not make a June cutoff look like a completed year."""

        frame = pd.DataFrame([
            {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 1000.0, "amount": 10200.0},
            {"open": 10.2, "high": 11.0, "low": 10.0, "close": 10.8, "volume": 1200.0, "amount": 12960.0},
            {"open": 10.8, "high": 11.4, "low": 10.4, "close": 11.0, "volume": 1400.0, "amount": 15400.0},
            {"open": 11.0, "high": 11.8, "low": 10.6, "close": 11.5, "volume": 1600.0, "amount": 18400.0},
        ], index=pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-06-30"]))
        with patch.object(webapp, "_load_public_kline", return_value=(pd.DataFrame(), None, ["public:timeout"])), patch.object(
            webapp,
            "_load_local_market_daily",
            return_value=frame,
        ):
            response = self.client.get("/api/portfolio/kline?code=600000&freq=year&bars=0")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "local_panel_unverified")
        self.assertEqual(payload["as_of"], "2026-06-30")
        self.assertEqual(payload["bars"][-1]["d"], "2026-06-30")

    def test_terminal_supplemental_industry_top3_fails_closed(self):
        observed = {
            "available": False,
            "code": "600905",
            "sector": {"name": "电力", "board_code": "BK0428", "constituent_count": 4},
            "industry_top3_1m": {
                "status": "unavailable",
                "top": [],
                "message": "有成分缺少近 20 个交易日收盘，不发布 Top 3。",
            },
        }
        with patch.object(webapp, "cached_observe", return_value=observed):
            response = self.client.get("/api/portfolio/industry-top3?code=sh.600905")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["sector"]["name"], "电力")
        top3 = payload["industry_top3_1m"]
        self.assertEqual(top3["status"], "unavailable")
        self.assertEqual(top3["top"], [])
        self.assertIn("不发布 Top 3", top3["message"])

    def test_tencent_day_parser_reads_index_bars(self):
        payload = {"data": {"sh000001": {"day": [
            ["2026-09-29", "3800", "3810", "3820", "3790", "100", {}, "0", "1000"],
            ["2026-09-30", "3810", "3838", "3840", "3805", "120", {}, "0", "1100"],
        ]}}}
        frame = webapp._parse_tencent_day_bars(payload, "sh000001")
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(float(frame.iloc[-1]["close"]), 3838.0)
        self.assertAlmostEqual(float(frame.iloc[-1]["amount"]), 1100.0 * 10000.0)
        self.assertEqual(webapp._public_kline_adjustment("tencent_daily"), "none")

    def test_shanghai_index_is_not_routed_as_the_shenzhen_stock(self):
        self.assertEqual(webapp._listed_identity("sh000001"), ("000001", "sh", True))
        self.assertEqual(webapp._listed_identity("000001"), ("000001", "sz", False))
        self.assertEqual(webapp._tencent_symbol("000001", "sh"), "sh000001")
        self.assertEqual(webapp._tencent_symbol("000001"), "sz000001")
        self.assertEqual(webapp._tencent_symbol("399006", "sz"), "sz399006")

    def test_symbol_search_returns_only_the_desk_rows(self):
        with patch.object(webapp, "search_a_shares", return_value=[{"code": "600519", "name": "贵州茅台", "market": "沪A"}]):
            payload = self.client.get("/api/portfolio/search?q=茅台").get_json()

        self.assertEqual(payload["rows"][0]["code"], "600519")
        self.assertEqual(payload["query"], "茅台")

    def test_holdings_keeps_all_fifty_grok_rows_when_research_latest_is_blocked(self):
        with patch.object(webapp, "_load", side_effect=AssertionError("latest.json must not be read")), patch.object(
            webapp,
            "_overlay_paper_quotes",
            side_effect=lambda rows: rows,
        ) as overlay:
            response = self.client.get("/api/portfolio/holdings")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertFalse(payload["public_quotes_enabled"])
        overlay.assert_not_called()
        self.assertEqual(len(payload["rows"]), 50)
        self.assertEqual(payload["summary"]["holdings_count"], 50)
        row = payload["rows"][0]
        for field in (
            "industry", "sector", "board_name", "cost", "quantity", "market_value",
            "unrealized_pnl", "return_since_entry", "pct_chg", "live_quote",
            "quote_state", "quote_time",
        ):
            self.assertIn(field, row)
        self.assertEqual(row["quote_state"], "ledger_mark")

    def test_holdings_fetches_public_quotes_only_with_explicit_request(self):
        with patch.object(webapp, "_overlay_paper_quotes", side_effect=lambda rows: rows) as overlay:
            response = self.client.get("/api/portfolio/holdings?live=1")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertTrue(payload["public_quotes_enabled"])
        overlay.assert_called_once()

    def test_terminal_read_only_routes_ignore_refresh_and_latest(self):
        with patch.object(webapp, "_load", side_effect=AssertionError("latest.json must not be read")):
            signals = self.client.get("/api/portfolio/terminal/signals?refresh=1").get_json()
            experts = self.client.get("/api/portfolio/terminal/experts?refresh=1").get_json()
            risk = self.client.get("/api/portfolio/terminal/risk").get_json()

        self.assertEqual(signals["status"], "paper_simulation_unverified")
        self.assertEqual(signals["trades"], [])
        self.assertEqual(experts["status"], "paper_simulation_unverified")
        self.assertEqual(experts["experts"], {})
        self.assertFalse(risk["available"])

    def test_terminal_public_observation_endpoint_shapes(self):
        minute = {
            "rows": [{"t": "2026-09-29 09:30", "p": 10.0, "v": 200.0, "pre": 9.8}],
            "pre_close": 9.8,
            "last": 10.0,
            "quote_time": "2026-09-29 09:30",
        }
        forecast_html = """
        <table><tr><th>指标</th><th>2025A</th><th>2026E</th></tr>
        <tr><td>营业收入(元)</td><td>100</td><td>200</td></tr></table>
        """
        with patch.object(webapp, "_fetch_tencent_intraday", return_value=minute), patch.object(
            webapp,
            "_fetch_tencent_indices",
            return_value=[{"name": "上证指数", "price": 3000.0, "pre_close": 2990.0, "pct_chg": 0.003344}],
        ), patch.object(webapp, "_fetch_ths_revenue_html", return_value=forecast_html):
            intraday = self.client.get("/api/portfolio/intraday?code=600905").get_json()
            indices = self.client.get("/api/portfolio/terminal/indices").get_json()
            revenue = self.client.get("/api/portfolio/revenue-forecast?code=600905").get_json()

        self.assertTrue(intraday["available"])
        self.assertEqual(intraday["rows"][0]["p"], 10.0)
        self.assertEqual(indices["rows"][0]["name"], "上证指数")
        forecast = revenue["revenue_forecast"]
        self.assertEqual(forecast["status"], "available")
        self.assertEqual(forecast["metric"], "营业收入(元)")
        self.assertEqual(forecast["forecasts"], [{"year": 2026, "value": 200.0}])

    def test_tencent_intraday_uses_the_provider_qt_previous_close_when_minute_rows_omit_it(self):
        class _Response:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        raw = b'''{"data":{"sh600905":{"data":{"date":"20260929","data":["0930 10.00 200"]},"qt":{"sh600905":["", "sample", "600905", "10.00", "9.80"]}}}}'''
        with patch.object(webapp, "urlopen", return_value=_Response(raw)):
            result = webapp._fetch_tencent_intraday("600905")

        self.assertEqual(result["pre_close"], 9.8)
        self.assertEqual(result["last"], 10.0)

    def test_holdings_response_exposes_mark_vs_public_quote_and_pnl(self):
        rows = [{
            "code": "600000",
            "name": "浦发银行",
            "quantity": 100,
            "weight": 0.25,
            "price": 10.0,
            "live_quote": 10.5,
            "pre_close": 10.2,
            "pct_chg": 10.5 / 10.2 - 1,
            "cost": 9.0,
            "quote_time": "2026-09-29 14:35:00",
            "quote_source": "test",
            "as_of": "2026-09-28",
        }]
        account = {
            "status": "research_unvalidated",
            "cash": 1000.0,
            "receivables": 0.0,
            "nav_mark": 2000.0,
            "account_as_of": "2026-09-28",
        }
        with patch.object(webapp, "_portfolio_snapshot", return_value=(rows, account, None)), patch.object(
            webapp,
            "_current_shanghai_time",
            return_value=datetime(2026, 9, 29, 15, 35, tzinfo=ZoneInfo("Asia/Shanghai")),
        ):
            response = self.client.get("/api/portfolio/holdings")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"], "available")
        row = payload["rows"][0]
        self.assertEqual(row["mark_price"], 10.0)
        self.assertEqual(row["current_price"], 10.5)
        self.assertTrue(row["price_is_public"])
        self.assertEqual(row["quote_state"], "closed")
        self.assertAlmostEqual(row["unrealized_pnl"], 150.0)
        self.assertAlmostEqual(row["day_pnl"], 30.0)
        self.assertAlmostEqual(payload["summary"]["market_value"], 1050.0)

    def test_pre_open_public_quote_never_becomes_a_fake_zero_day_return(self):
        rows = [{
            "code": "600000", "name": "浦发银行", "quantity": 100, "weight": 0.25,
            "price": 10.0, "live_quote": 10.2, "pre_close": 10.2, "cost": 9.0,
            "quote_time": "2026-09-29 09:05:00", "quote_source": "test", "as_of": "2026-09-28",
        }]
        account = {
            "status": "paper_simulation_unverified", "displayable": True,
            "cash": 1000.0, "receivables": 0.0, "nav_mark": 2000.0,
            "account_as_of": "2026-09-28",
        }
        with patch.object(webapp, "_portfolio_snapshot", return_value=(rows, account, None)), patch.object(
            webapp,
            "_current_shanghai_time",
            return_value=datetime(2026, 9, 29, 9, 10, tzinfo=ZoneInfo("Asia/Shanghai")),
        ):
            payload = self.client.get("/api/portfolio/holdings").get_json()

        self.assertEqual(payload["rows"][0]["quote_state"], "pre_open")
        self.assertTrue(payload["rows"][0]["price_is_public"])
        self.assertFalse(payload["summary"]["day_quote_complete"])
        self.assertIsNone(payload["today"])
        self.assertEqual(payload["summary"]["quote_state_counts"], {"pre_open": 1})

    def test_one_delayed_intraday_quote_blocks_a_portfolio_realtime_day_return(self):
        rows = [
            {
                "code": "600000", "name": "样本一", "quantity": 100, "weight": 0.5,
                "price": 10.0, "live_quote": 10.5, "pre_close": 10.0, "cost": 9.0,
                "quote_time": "2026-09-29 14:55:00", "quote_source": "test", "as_of": "2026-09-28",
            },
            {
                "code": "600001", "name": "样本二", "quantity": 100, "weight": 0.5,
                "price": 8.0, "live_quote": 8.4, "pre_close": 8.0, "cost": 7.0,
                "quote_time": "2026-09-29 14:00:00", "quote_source": "test", "as_of": "2026-09-28",
            },
        ]
        account = {
            "status": "paper_simulation_unverified", "displayable": True,
            "cash": 1000.0, "receivables": 0.0, "nav_mark": 3000.0,
            "account_as_of": "2026-09-28",
        }
        with patch.object(webapp, "_portfolio_snapshot", return_value=(rows, account, None)), patch.object(
            webapp,
            "_current_shanghai_time",
            return_value=datetime(2026, 9, 29, 14, 58, tzinfo=ZoneInfo("Asia/Shanghai")),
        ):
            payload = self.client.get("/api/portfolio/holdings").get_json()

        self.assertEqual([row["quote_state"] for row in payload["rows"]], ["trading", "delayed"])
        self.assertFalse(payload["summary"]["day_quote_complete"])
        self.assertIsNone(payload["today"])

    def test_morning_close_is_not_labelled_as_a_delayed_quote(self):
        lunch_time = datetime(2026, 9, 30, 11, 35, tzinfo=ZoneInfo("Asia/Shanghai"))

        self.assertEqual(
            webapp._quote_state_from_timestamp("2026-09-30 11:30:00", now=lunch_time),
            "lunch_break",
        )
        self.assertEqual(
            webapp._quote_state_from_timestamp("2026-09-30 11:20:00", now=lunch_time),
            "delayed",
        )
        self.assertEqual(webapp._quote_state_label("lunch_break"), "午间休市")

    def test_lunch_break_quotes_value_the_book_and_publish_today(self):
        """The 11:30 print is today's price during the lunch break, not a missing quote."""

        rows = [
            {
                "code": "600000", "name": "样本一", "quantity": 100, "weight": 0.5,
                "price": 10.0, "live_quote": 10.5, "pre_close": 10.0, "cost": 9.0,
                "quote_time": "2026-09-30 11:30:00", "quote_source": "sina", "as_of": "2026-09-28",
            },
            {
                "code": "600001", "name": "样本二", "quantity": 100, "weight": 0.5,
                "price": 8.0, "live_quote": 7.6, "pre_close": 8.0, "cost": 7.0,
                "quote_time": "2026-09-30 11:30:00", "quote_source": "sina", "as_of": "2026-09-28",
            },
        ]
        account = {
            "status": "paper_simulation_unverified", "displayable": True,
            "cash": 1000.0, "receivables": 0.0, "nav_mark": 2800.0,
            "account_as_of": "2026-09-28",
        }
        lunch = datetime(2026, 9, 30, 11, 50, tzinfo=ZoneInfo("Asia/Shanghai"))
        with patch.object(webapp, "_current_shanghai_time", return_value=lunch), patch.object(
            webapp, "_portfolio_snapshot", return_value=(rows, account, None)
        ):
            payload = self.client.get("/api/portfolio/holdings?live=1").get_json()

        self.assertEqual([row["quote_state"] for row in payload["rows"]], ["lunch_break", "lunch_break"])
        self.assertTrue(all(row["price_is_public"] for row in payload["rows"]))
        self.assertEqual(payload["rows"][0]["current_price"], 10.5)
        self.assertEqual(payload["rows"][0]["live_quote"], 10.5)
        self.assertEqual(payload["summary"]["quote_count"], 2)
        self.assertTrue(payload["summary"]["day_quote_complete"])
        self.assertAlmostEqual(payload["summary"]["day_pnl"], 10.0)
        self.assertAlmostEqual(payload["today"], 10.0 / 2800.0)
        self.assertEqual(payload["summary"]["quote_state_counts"], {"lunch_break": 2})

    def test_stale_morning_quote_during_lunch_does_not_complete_today(self):
        rows = [
            {
                "code": "600000", "name": "样本一", "quantity": 100, "weight": 0.5,
                "price": 10.0, "live_quote": 10.5, "pre_close": 10.0, "cost": 9.0,
                "quote_time": "2026-09-30 11:30:00", "quote_source": "sina", "as_of": "2026-09-28",
            },
            {
                "code": "600001", "name": "样本二", "quantity": 100, "weight": 0.5,
                "price": 8.0, "live_quote": 8.2, "pre_close": 8.0, "cost": 7.0,
                "quote_time": "2026-09-30 11:20:00", "quote_source": "sina", "as_of": "2026-09-28",
            },
        ]
        account = {
            "status": "paper_simulation_unverified", "displayable": True,
            "cash": 1000.0, "receivables": 0.0, "nav_mark": 2800.0,
            "account_as_of": "2026-09-28",
        }
        lunch = datetime(2026, 9, 30, 11, 50, tzinfo=ZoneInfo("Asia/Shanghai"))
        with patch.object(webapp, "_current_shanghai_time", return_value=lunch), patch.object(
            webapp, "_portfolio_snapshot", return_value=(rows, account, None)
        ):
            payload = self.client.get("/api/portfolio/holdings?live=1").get_json()

        self.assertEqual([row["quote_state"] for row in payload["rows"]], ["lunch_break", "delayed"])
        self.assertEqual(payload["rows"][0]["current_price"], 10.5)
        self.assertFalse(payload["summary"]["day_quote_complete"])
        self.assertIsNone(payload["today"])

    def test_previous_session_close_never_mixes_into_an_intraday_portfolio_return(self):
        """A prior-day close is observable, but is not a current-session quote."""

        rows = [
            {
                "code": "600000", "name": "样本一", "quantity": 100, "weight": 0.5,
                "price": 10.0, "live_quote": 10.5, "pre_close": 10.0, "cost": 9.0,
                "quote_time": "2026-09-29 14:55:00", "quote_source": "test", "as_of": "2026-09-28",
            },
            {
                "code": "600001", "name": "样本二", "quantity": 100, "weight": 0.5,
                "price": 8.0, "live_quote": 8.4, "pre_close": 8.0, "cost": 7.0,
                "quote_time": "2026-09-28 15:00:00", "quote_source": "test", "as_of": "2026-09-28",
            },
        ]
        account = {
            "status": "paper_simulation_unverified", "displayable": True,
            "cash": 1000.0, "receivables": 0.0, "nav_mark": 3000.0,
            "account_as_of": "2026-09-28",
        }
        with patch.object(webapp, "_portfolio_snapshot", return_value=(rows, account, None)), patch.object(
            webapp,
            "_current_shanghai_time",
            return_value=datetime(2026, 9, 29, 14, 58, tzinfo=ZoneInfo("Asia/Shanghai")),
        ):
            payload = self.client.get("/api/portfolio/holdings").get_json()

        self.assertEqual([row["quote_state"] for row in payload["rows"]], ["trading", "closed"])
        self.assertFalse(payload["summary"]["day_quote_complete"])
        self.assertIsNone(payload["today"])

    def test_public_price_without_an_explicit_timestamp_is_not_used_as_live_valuation(self):
        rows = [{
            "code": "600000", "name": "浦发银行", "quantity": 100, "weight": 0.25,
            "price": 10.0, "live_quote": 10.5, "pre_close": 10.2, "cost": 9.0,
            "quote_time": None, "quote_source": "test", "as_of": "2026-09-28",
        }]
        account = {
            "status": "paper_simulation_unverified", "displayable": True,
            "cash": 1000.0, "receivables": 0.0, "nav_mark": 2000.0,
            "account_as_of": "2026-09-28",
        }
        with patch.object(webapp, "_portfolio_snapshot", return_value=(rows, account, None)):
            payload = self.client.get("/api/portfolio/holdings").get_json()

        row = payload["rows"][0]
        self.assertEqual(row["quote_state"], "timestamp_missing")
        self.assertFalse(row["price_is_public"])
        self.assertEqual(row["current_price"], 10.0)
        self.assertEqual(payload["summary"]["quote_count"], 0)

    def test_partial_public_quotes_never_publish_a_partial_portfolio_day_return(self):
        rows = [
            {
                "code": "600000", "name": "浦发银行", "quantity": 100, "weight": 0.5,
                "price": 10.0, "live_quote": 10.5, "pre_close": 10.2, "cost": 9.0,
                "quote_time": "2026-09-29 14:35:00", "quote_source": "test", "as_of": "2026-09-28",
            },
            {
                "code": "600001", "name": "样本二", "quantity": 100, "weight": 0.5,
                "price": 8.0, "live_quote": 8.2, "pre_close": None, "cost": 7.0,
                "quote_time": "2026-09-29 14:35:00", "quote_source": "test", "as_of": "2026-09-28",
            },
        ]
        account = {
            "status": "paper_simulation_unverified", "displayable": True,
            "cash": 1000.0, "receivables": 0.0, "nav_mark": 3000.0,
            "account_as_of": "2026-09-28",
        }
        closed_session = datetime(2026, 9, 29, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        with patch.object(webapp, "_current_shanghai_time", return_value=closed_session), patch.object(
            webapp, "_portfolio_snapshot", return_value=(rows, account, None)
        ):
            payload = self.client.get("/api/portfolio/holdings").get_json()

        summary = payload["summary"]
        self.assertEqual(summary["quote_count"], 2)
        self.assertEqual(summary["day_quote_count"], 1)
        self.assertFalse(summary["day_quote_complete"])
        self.assertIsNone(summary["day_pnl"])
        self.assertIsNone(payload["today"])

    def test_holdings_do_not_relabel_build_time_industry_as_sector(self):
        rows = [{
            "code": "sh.600000",
            "name": "浦发银行",
            "quantity": 100,
            "weight": 0.25,
            "price": 10.0,
            "cost": 9.0,
            "industry": "C68 货币金融服务",
            "build_time_industry": "C68 货币金融服务",
            "board": "SH main",
            "as_of": "2026-09-28",
        }]
        account = {
            "status": "paper_simulation_unverified",
            "displayable": True,
            "cash": 1000.0,
            "receivables": 0.0,
            "nav_mark": 2000.0,
            "account_as_of": "2026-09-28",
        }
        with patch.object(webapp, "_portfolio_snapshot", return_value=(rows, account, None)):
            payload = self.client.get("/api/portfolio/holdings").get_json()

        row = payload["rows"][0]
        self.assertEqual(row["industry"], "C68 货币金融服务")
        self.assertIsNone(row["sector"])
        self.assertEqual(row["board_name"], "SH main")

    def test_complete_unvalidated_simulation_ledger_remains_displayable(self):
        rows = [{
            "code": "sh.600905",
            "name": "厦门港务",
            "quantity": 7700,
            "weight": 0.05,
            "price": 3.7,
            "live_quote": 3.8,
            "pre_close": 3.75,
            "cost": 3.6,
            "entry_date": "2026-09-01",
            "as_of": "2026-09-22",
        }]
        account = {
            "status": "unverified_data_gaps",
            "displayable": True,
            "cash": 1000.0,
            "receivables": 0.0,
            "nav_mark": 30000.0,
            "account_as_of": "2026-09-22",
            "source_kind": "pinned_simulation_ledger",
        }
        with patch.object(webapp, "_portfolio_snapshot", return_value=(rows, account, None)):
            response = self.client.get("/api/portfolio/holdings")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"], "available")
        self.assertEqual(payload["account_status"], "unverified_data_gaps")
        self.assertFalse(payload["legacy_snapshot"])
        self.assertEqual(payload["rows"][0]["code"], "sh.600905")
        self.assertEqual(payload["rows"][0]["quantity"], 7700.0)

    def test_blocked_account_is_visible_as_empty_not_fabricated(self):
        with patch.object(webapp, "_portfolio_snapshot", return_value=([], {"status": "blocked_data"}, None)):
            response = self.client.get("/api/portfolio/holdings")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["status"], "blocked_data")
        self.assertEqual(payload["rows"], [])
        self.assertTrue(any("历史回放权重" in item for item in payload["caveats"]))

    def test_blocked_account_never_falls_back_to_unverified_weight_history(self):
        with patch.object(webapp, "_portfolio_snapshot", return_value=([], {"status": "blocked_data"}, None)):
            response = self.client.get("/api/portfolio/holdings")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["status"], "blocked_data")
        self.assertEqual(payload["account_status"], "blocked_data")
        self.assertFalse(payload["legacy_snapshot"])
        self.assertEqual(payload["rows"], [])
        self.assertTrue(any("历史回放权重" in item for item in payload["caveats"]))

    def test_kline_rejects_non_security_code_without_network(self):
        with patch.object(webapp, "_load_public_kline") as loader:
            response = self.client.get("/api/portfolio/kline?code=bad")
        self.assertEqual(response.status_code, 400)
        loader.assert_not_called()

    def test_kline_response_is_daily_ohlcv_and_marks_public_data_unverified(self):
        frame = pd.DataFrame([
            {"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 1000.0, "amount": 10000.0},
            {"open": 10.5, "high": 10.8, "low": 10.0, "close": 10.2, "volume": 1200.0, "amount": 12200.0},
        ], index=pd.to_datetime(["2026-09-28", "2026-09-29"]))
        with patch.object(webapp, "_load_public_kline", return_value=(frame, "test", [])):
            response = self.client.get("/api/portfolio/kline?code=600000&range=1m")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"], "public_quote_unverified")
        self.assertEqual(len(payload["candles"]), 2)
        self.assertEqual(payload["candles"][-1]["date"], "2026-09-29")
        self.assertEqual(payload["source"], "test")

    def test_kline_maps_canonical_ledger_code_to_public_code(self):
        frame = pd.DataFrame([
            {"open": 3.7, "high": 3.9, "low": 3.6, "close": 3.8, "volume": 1000.0, "amount": 3800.0},
        ], index=pd.to_datetime(["2026-09-29"]))
        with patch.object(webapp, "_load_public_kline", return_value=(frame, "test", [])) as loader:
            response = self.client.get("/api/portfolio/kline?code=sh.600905&range=1m")
        self.assertEqual(response.status_code, 200)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0], "600905")
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["code"], "600905")

    def test_kline_does_not_fall_back_to_unverified_local_cache(self):
        with patch.object(webapp, "_load_public_kline", return_value=(pd.DataFrame(), None, ["eastmoney:TimeoutError"])):
            response = self.client.get("/api/portfolio/kline?code=600000&range=1m")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["status"], "unavailable")
        self.assertIsNone(payload["source"])


if __name__ == "__main__":
    unittest.main()
