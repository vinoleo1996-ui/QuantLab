"""Regression coverage for the optional public quote adapter.

These tests use a captured-shape Sina response only.  They never contact a
market-data provider or turn an accounting mark into a quote.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import quantlab.data.realtime as realtime
from quantlab.data.realtime import batch_quotes


class _UrlResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _sina_record(symbol: str, name: str, *, price: str, pre_close: str,
                 date: str, clock: str) -> str:
    fields = [""] * 33
    fields[0] = name
    fields[1] = "10.00"
    fields[2] = pre_close
    fields[3] = price
    fields[4] = "10.20"
    fields[5] = "9.80"
    fields[8] = "12300"
    fields[9] = "123456.00"
    fields[30] = date
    fields[31] = clock
    fields[32] = "00"
    return f'var hq_str_{symbol}="{",".join(fields)}";'


class _TextResponse:
    def __init__(self, text: str):
        self.text = text


class _RequestsLikeResponse:
    """Small requests-compatible fixture whose text follows ``encoding``."""

    def __init__(self, payload: bytes):
        self.content = payload
        self.encoding = "ISO-8859-1"

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "ISO-8859-1", errors="replace")

    def raise_for_status(self) -> None:
        return None


def _tencent_record(symbol: str, name: str, code: str, *, price: str,
                    pre_close: str, timestamp: str) -> str:
    """A minimal captured-shape Tencent quote record."""

    fields = [""] * 38
    fields[1] = name
    fields[2] = code
    fields[3] = price
    fields[4] = pre_close
    fields[5] = price
    fields[6] = "1000"
    fields[30] = timestamp
    fields[33] = price
    fields[34] = pre_close
    fields[37] = "100"
    return f'v_{symbol}="{"~".join(fields)}";'


class RealtimeQuoteTests(unittest.TestCase):
    def test_requests_path_decodes_provider_payload_as_gbk(self):
        response = _RequestsLikeResponse("上证指数".encode("gbk"))

        class _Requests:
            @staticmethod
            def get(*_args, **_kwargs):
                return response

        with patch.dict(sys.modules, {"requests": _Requests}):
            decoded = realtime._get("https://example.invalid/quote")

        self.assertEqual(decoded.text, "上证指数")

    def test_stdlib_gbk_fallback_parses_every_sina_record_and_quote_timestamp(self):
        """Missing optional ``requests`` must not silently blank public quotes."""
        payload = (
            _sina_record("sh601956", "东贝集团", price="10.10", pre_close="10.00",
                         date="20260929", clock="14:59:58")
            + _sina_record("sh600905", "三峡能源", price="20.20", pre_close="20.00",
                           date="20260929", clock="15:00:02")
        ).encode("gbk")

        with patch.dict(sys.modules, {"requests": None}):
            with patch("urllib.request.urlopen", return_value=_UrlResponse(payload)) as urlopen:
                quotes = batch_quotes(["601956", "600905"])

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(list(quotes.index), ["601956", "600905"])
        self.assertEqual(quotes.loc["601956", "name"], "东贝集团")
        self.assertAlmostEqual(quotes.loc["600905", "pct_chg"], 0.01)
        self.assertEqual(quotes.loc["601956", "time"], "20260929 14:59:58")
        self.assertEqual(quotes.loc["600905", "time"], "20260929 15:00:02")
        self.assertTrue((quotes["source"] == "sina").all())

    def test_tencent_fallback_keeps_each_securitys_own_quote_timestamp(self):
        """A batch fallback must not copy one ticker's timestamp onto every row."""
        fallback_rows = [
            {
                "code": "601956", "name": "东贝集团", "price": 5.07, "pre_close": 5.08,
                "open": 5.06, "high": 5.12, "low": 5.02, "volume": 100.0, "amount": 1000.0,
                "time": "2026-09-29 16:14:57",
            },
            {
                "code": "002374", "name": "中锐股份", "price": 3.30, "pre_close": 3.27,
                "open": 3.28, "high": 3.34, "low": 3.26, "volume": 200.0, "amount": 2000.0,
                "time": "2026-09-29 16:14:12",
            },
        ]
        with patch.object(realtime, "_sina", return_value=([], None)), patch.object(
            realtime,
            "_tencent",
            return_value=(fallback_rows, "2026-09-29 16:14:57"),
        ):
            quotes = batch_quotes(["601956", "002374"])

        self.assertEqual(quotes.loc["601956", "time"], "2026-09-29 16:14:57")
        self.assertEqual(quotes.loc["002374", "time"], "2026-09-29 16:14:12")

    def test_index_quotes_use_explicit_market_symbols_and_validate_them(self):
        payload = "\n".join([
            _tencent_record("sh000001", "上证指数", "000001", price="3200.10", pre_close="3190.10",
                            timestamp="20260930150000"),
            _tencent_record("sz399001", "深证成指", "399001", price="10000.10", pre_close="9900.10",
                            timestamp="20260930150000"),
            _tencent_record("sz399006", "创业板指", "399006", price="2000.10", pre_close="1990.10",
                            timestamp="20260930150000"),
        ])
        with patch.object(realtime, "_get", return_value=_TextResponse(payload)) as request:
            quotes = realtime.batch_index_quotes()

        self.assertIn("q=sh000001,sz399001,sz399006", request.call_args.args[0])
        self.assertEqual(list(quotes.index), ["000001", "399001", "399006"])
        self.assertEqual(quotes.loc["000001", "name"], "上证指数")
        self.assertAlmostEqual(quotes.loc["000001", "price"], 3200.10)

        wrong_symbol = _tencent_record("sz000001", "平安银行", "000001", price="11.35", pre_close="11.20",
                                       timestamp="20260930150000")
        with patch.object(realtime, "_get", return_value=_TextResponse(wrong_symbol)):
            rejected = realtime.batch_index_quotes()

        self.assertNotIn("000001", rejected.index)

    def test_tencent_stock_parser_reads_each_semicolon_delimited_record(self):
        payload = (
            _tencent_record("sh600000", "浦发银行", "600000", price="10.10", pre_close="10.00",
                            timestamp="20260930150000")
            + _tencent_record("sz000001", "平安银行", "000001", price="11.35", pre_close="11.20",
                              timestamp="20260930150001")
        )
        with patch.object(realtime, "_get", return_value=_TextResponse(payload)):
            rows, _ = realtime._tencent(["600000", "000001"])

        self.assertEqual([row["code"] for row in rows], ["600000", "000001"])
        self.assertEqual(rows[1]["name"], "平安银行")
