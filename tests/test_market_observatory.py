"""Focused contracts for the read-only market-observatory helpers.

These tests use only in-memory fixtures.  Public quote, THS, and filesystem
access are deliberately outside this unit-test surface.
"""

from __future__ import annotations

import unittest

import pandas as pd

from quantlab.portfolio.market_observatory import (
    local_panel_fallback,
    parse_ths_revenue_forecast,
    resample_ohlcv,
    top_month_returns,
    validate_period,
)


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [12.0, 13.0, 15.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [11.0, 12.0, 14.0, 12.5],
            "volume": [100.0, 200.0, 300.0, 400.0],
            "amount": [1000.0, 2000.0, 3000.0, 4000.0],
        },
        index=pd.to_datetime(["2025-12-29", "2025-12-30", "2026-01-02", "2026-01-05"]),
    )


class MarketObservatoryTests(unittest.TestCase):
    def test_period_validation_is_strict(self):
        self.assertEqual(validate_period("day"), "day")
        self.assertEqual(validate_period("week"), "week")
        self.assertEqual(validate_period("month"), "month")
        self.assertEqual(validate_period("year"), "year")
        for invalid in ("Day", "quarter", "1m", "", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "unsupported_market_period"):
                    validate_period(invalid)

    def test_daily_week_month_and_year_ohlcv_use_market_aggregations(self):
        daily = _daily()
        pd.testing.assert_frame_equal(resample_ohlcv(daily, "day"), daily)

        weekly = resample_ohlcv(daily, "week")
        self.assertEqual(list(weekly.index.strftime("%Y-%m-%d")), ["2026-01-02", "2026-01-09"])
        self.assertEqual(weekly.iloc[0].to_dict(), {
            "open": 10.0, "high": 15.0, "low": 9.0, "close": 14.0,
            "volume": 600.0, "amount": 6000.0,
        })

        monthly = resample_ohlcv(daily, "month")
        self.assertEqual(list(monthly.index.strftime("%Y-%m-%d")), ["2025-12-31", "2026-01-31"])
        self.assertEqual(monthly.iloc[0].to_dict(), {
            "open": 10.0, "high": 13.0, "low": 9.0, "close": 12.0,
            "volume": 300.0, "amount": 3000.0,
        })

        yearly = resample_ohlcv(daily, "year")
        self.assertEqual(list(yearly.index.strftime("%Y-%m-%d")), ["2025-12-31", "2026-12-31"])
        self.assertEqual(yearly.iloc[1].to_dict(), {
            "open": 12.0, "high": 15.0, "low": 11.0, "close": 12.5,
            "volume": 700.0, "amount": 7000.0,
        })

    def test_resample_rejects_noncanonical_daily_panel(self):
        incomplete = _daily().drop(columns="amount")
        with self.assertRaisesRegex(ValueError, "canonical_ohlcv_missing_columns"):
            resample_ohlcv(incomplete, "year")

    def test_local_panel_fallback_binds_cutoff_and_source(self):
        frame, metadata = local_panel_fallback(
            _daily(),
            period="week",
            cutoff="2026-01-02",
            source="paper_local_cache",
        )
        self.assertTrue(metadata["available"])
        self.assertEqual(metadata["status"], "local_panel_unverified")
        self.assertEqual(metadata["source"], "paper_local_cache")
        self.assertEqual(metadata["source_kind"], "local_panel")
        self.assertEqual(metadata["cutoff"], "2026-01-02")
        self.assertEqual(metadata["as_of"], "2026-01-02")
        self.assertEqual(list(frame.index.strftime("%Y-%m-%d")), ["2026-01-02"])

    def test_local_panel_fallback_never_invents_an_empty_series(self):
        frame, metadata = local_panel_fallback(
            None,
            period="month",
            cutoff="2026-01-31",
            source="paper_local_cache",
        )
        self.assertFalse(metadata["available"])
        self.assertEqual(metadata["status"], "unavailable")
        self.assertEqual(metadata["reason"], "local_panel_missing")
        self.assertTrue(frame.empty)

    def test_ths_revenue_parser_returns_only_forecast_year_average_cells(self):
        html = """
        <table>
          <thead><tr><th>指标</th><th>2024A</th><th>2025E</th><th>2026年预测</th></tr></thead>
          <tbody>
            <tr><td>营业收入(元)</td><td>800</td><td>1,234.50</td><td>2,000</td></tr>
            <tr><td>净利润(元)</td><td>10</td><td>12</td><td>15</td></tr>
          </tbody>
        </table>
        """
        forecast = parse_ths_revenue_forecast(html)
        self.assertTrue(forecast["available"])
        self.assertEqual(forecast["status"], "available")
        self.assertEqual(forecast["metric"], "营业收入(元)")
        self.assertEqual(forecast["forecasts"], [
            {"year": 2025, "value": 1234.5},
            {"year": 2026, "value": 2000.0},
        ])

    def test_ths_revenue_parser_refuses_reported_or_different_revenue_rows(self):
        html = """
        <table>
          <tr><th>指标</th><th>2024A</th><th>2025A</th></tr>
          <tr><td>营业总收入(元)</td><td>100</td><td>120</td></tr>
          <tr><td>营业收入(元)</td><td>90</td><td>110</td></tr>
        </table>
        """
        forecast = parse_ths_revenue_forecast(html)
        self.assertFalse(forecast["available"])
        self.assertEqual(forecast["status"], "unavailable")
        self.assertEqual(forecast["forecasts"], [])
        self.assertEqual(forecast["message"], "暂无机构预测")
        self.assertEqual(forecast["reason"], "forecast_revenue_not_found")

    def test_month_return_ranking_is_deterministic_and_requires_complete_inputs(self):
        constituents = [
            {"code": "000002", "name": "乙"},
            {"code": "000001", "name": "甲"},
            {"code": "000003", "name": "丙"},
            {"code": "000004", "name": "丁"},
        ]
        ranking = top_month_returns(
            constituents,
            {"000001": 0.10, "000002": 0.10, "000003": 0.05, "000004": -0.02},
        )
        self.assertTrue(ranking["available"])
        self.assertEqual(ranking["window_trading_days"], 20)
        self.assertEqual([row["code"] for row in ranking["top"]], ["000001", "000002", "000003"])
        self.assertEqual([row["rank"] for row in ranking["top"]], [1, 2, 3])

        no_constituents = top_month_returns([], {})
        self.assertFalse(no_constituents["available"])
        self.assertEqual(no_constituents["top"], [])
        self.assertEqual(no_constituents["reason"], "industry_constituents_missing")

        incomplete = top_month_returns(constituents, {"000001": 0.1}, constituents_complete=False)
        self.assertFalse(incomplete["available"])
        self.assertEqual(incomplete["top"], [])
        self.assertEqual(incomplete["reason"], "industry_constituents_incomplete")


if __name__ == "__main__":
    unittest.main()
