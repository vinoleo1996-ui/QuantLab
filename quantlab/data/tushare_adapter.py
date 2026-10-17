"""Dual-source Tushare adapter: datahubco for bulk market data, promax for single-stock queries.

This adapter provides a unified interface over the two data sources supplied by the user:
- datahubco (http://datahubco.com): high-limit, good for full-market daily snapshots
- promax (https://pcd.mobcvb.cn/tushare/pro): HTTPS, supports suspension, adj_factor, limit prices, ETF data

Usage:
    from quantlab.data.tushare_adapter import TushareAdapter
    api = TushareAdapter()
    df = api.daily_trade_date('20261016')          # full-market snapshot via datahubco
    df = api.daily('000001.SZ', '20260101', '20261017')  # single-stock history via promax
    susp = api.suspend('000616.SZ', '20180101', '20181231')  # suspension records via promax
"""
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Credentials — read from environment variables; never hard-code in source.
# Set DATAHUB_API_KEY and PCD_API_KEY before use.
# ---------------------------------------------------------------------------
DATAHUBCO_BASE = "http://datahubco.com/app-api/openapi/v1/tushare"
DATAHUBCO_KEY = os.getenv("DATAHUB_API_KEY", "")

PROMAX_BASE = "https://pcd.mobcvb.cn/tushare/pro"
PROMAX_KEY = os.getenv("PCD_API_KEY", "")

# Frequency guard for promax (seconds between requests)
PROMAX_MIN_INTERVAL = 1.0

# Retry config
MAX_RETRIES = 2
BACKOFF_BASE = 1.0


class TushareAdapter:
    """Unified adapter with automatic source selection and rate-limit handling."""

    def __init__(self):
        self._last_promax_ts = 0.0
        if not DATAHUBCO_KEY:
            raise RuntimeError("DATAHUB_API_KEY environment variable not set")
        if not PROMAX_KEY:
            raise RuntimeError("PCD_API_KEY environment variable not set")

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _get(self, base: str, key: str, api: str, params: dict | None = None) -> dict:
        """Raw GET with retry/backoff."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        url = f"{base}/{api}"
        headers = {"X-API-Key": key}
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                if resp.status_code == 429:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                body = resp.json()
                if body.get("code", 0) not in (0, None):
                    raise RuntimeError(f"upstream code={body['code']}: {body.get('msg', '')}")
                return body
            except requests.exceptions.RequestException:
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE * (2 ** attempt))
                    continue
                raise
        raise RuntimeError("max_retries_exceeded")

    def _promax(self, api: str, **params) -> dict:
        """Promax call with frequency guard."""
        elapsed = time.time() - self._last_promax_ts
        if elapsed < PROMAX_MIN_INTERVAL:
            time.sleep(PROMAX_MIN_INTERVAL - elapsed)
        result = self._get(PROMAX_BASE, PROMAX_KEY, api, params)
        self._last_promax_ts = time.time()
        return result

    def _datahubco(self, api: str, **params) -> dict:
        """Datahubco call (no frequency guard needed for our volume)."""
        return self._get(DATAHUBCO_BASE, DATAHUBCO_KEY, api, params)

    @staticmethod
    def _to_df(body: dict, columns: list[str] | None = None) -> pd.DataFrame:
        """Convert Tushare response body to DataFrame."""
        data = body.get("data", {})
        items = data.get("items", [])
        fields = data.get("fields", columns)
        if not items:
            return pd.DataFrame(columns=fields or [])
        return pd.DataFrame(items, columns=fields)

    # ------------------------------------------------------------------
    # Daily market data
    # ------------------------------------------------------------------
    def daily_trade_date(self, trade_date: str, fields: str | None = None) -> pd.DataFrame:
        """Full-market snapshot for a single trade date via datahubco.

        Returns ~5,500 rows (all stocks with trading on that day).
        Columns: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        """
        cols = fields or "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        all_items: list[list[Any]] = []
        offset = 0
        limit = 5000
        while True:
            body = self._datahubco("daily", trade_date=trade_date, limit=limit, offset=offset, fields=cols)
            items = body.get("data", {}).get("items", [])
            count = body.get("data", {}).get("count", len(items))
            if not items:
                break
            all_items.extend(items)
            if offset + len(items) >= count:
                break
            offset += len(items)
        df = pd.DataFrame(all_items, columns=cols.split(","))
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        for c in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def daily(self, ts_code: str, start_date: str, end_date: str, fields: str | None = None) -> pd.DataFrame:
        """Single-stock daily history via promax (or datahubco if promax fails)."""
        cols = fields or "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        try:
            body = self._promax("daily", ts_code=ts_code, start_date=start_date, end_date=end_date, fields=cols)
        except RuntimeError:
            # Fallback to datahubco
            body = self._datahubco("daily", ts_code=ts_code, start_date=start_date, end_date=end_date, fields=cols)
        df = self._to_df(body, cols.split(","))
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        for c in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Suspension / trading status
    # ------------------------------------------------------------------
    def suspend(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Suspension records via promax.

        Columns: ts_code, trade_date, suspend_timing, suspend_type
        suspend_type: 'S' = suspended, 'R' = resumed
        """
        body = self._promax("suspend_d", ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = self._to_df(body)
        if not df.empty and "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return df

    def is_suspended(self, ts_code: str, trade_date: str) -> bool | None:
        """Quick check: is a stock suspended on a specific date?"""
        df = self.suspend(ts_code, trade_date, trade_date)
        if df.empty:
            return None  # unknown
        return bool((df["suspend_type"] == "S").any())

    # ------------------------------------------------------------------
    # Adjustment factor (for total-return / back-adjusted prices)
    # ------------------------------------------------------------------
    def adj_factor(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Adjustment factor via promax.

        Columns: ts_code, trade_date, adj_factor
        """
        body = self._promax("adj_factor", ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = self._to_df(body)
        if not df.empty and "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Limit prices
    # ------------------------------------------------------------------
    def limit_price(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Daily limit-up/limit-down prices via promax.

        Columns: trade_date, ts_code, up_limit, down_limit
        """
        body = self._promax("stk_limit", ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = self._to_df(body)
        if not df.empty and "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            for c in ["up_limit", "down_limit"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # ETF data
    # ------------------------------------------------------------------
    def etf_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """ETF daily bars via promax.

        Columns: same as daily (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
        """
        body = self._promax("fund_daily", ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = self._to_df(body)
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        for c in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def etf_nav(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """ETF NAV via promax.

        Columns: ts_code, ann_date, nav_date, unit_nav, accum_nav, accum_div, net_asset, adj_nav, update_flag
        """
        body = self._promax("fund_nav", ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = self._to_df(body)
        if not df.empty and "nav_date" in df.columns:
            df["nav_date"] = pd.to_datetime(df["nav_date"], format="%Y%m%d")
            df["unit_nav"] = pd.to_numeric(df["unit_nav"], errors="coerce")
        return df

    def etf_premium(self, ts_code: str, trade_date: str) -> float | None:
        """QDII premium = market_price / NAV - 1."""
        df_price = self.etf_daily(ts_code, trade_date, trade_date)
        df_nav = self.etf_nav(ts_code, trade_date, trade_date)
        if df_price.empty or df_nav.empty:
            return None
        price = df_price["close"].iloc[0]
        nav = df_nav["unit_nav"].iloc[0]
        if pd.isna(price) or pd.isna(nav) or nav == 0:
            return None
        return float(price / nav - 1.0)

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------
    def trade_cal(self, exchange: str = "SSE", start_date: str = "20140101", end_date: str = "20261231") -> pd.DataFrame:
        """Trading calendar via promax.

        Columns: exchange, cal_date, is_open, pretrade_date
        """
        body = self._promax("trade_cal", exchange=exchange, start_date=start_date, end_date=end_date)
        df = self._to_df(body)
        if not df.empty and "cal_date" in df.columns:
            df["cal_date"] = pd.to_datetime(df["cal_date"], format="%Y%m%d")
            df["is_open"] = pd.to_numeric(df["is_open"], errors="coerce").astype(bool)
        return df

    # ------------------------------------------------------------------
    # Stock basic info (delisting, listing status)
    # ------------------------------------------------------------------
    def stock_basic(self, ts_code: str | None = None, exchange: str | None = None, list_status: str | None = None) -> pd.DataFrame:
        """Stock basic info via promax (single stock or exchange-scoped)."""
        params: dict[str, Any] = {}
        if ts_code:
            params["ts_code"] = ts_code
        if exchange:
            params["exchange"] = exchange
        if list_status:
            params["list_status"] = list_status
        body = self._promax("stock_basic", **params)
        return self._to_df(body)

    # ------------------------------------------------------------------
    # Pro_bar (back-adjusted OHLC)
    # ------------------------------------------------------------------
    def pro_bar(self, ts_code: str, start_date: str, end_date: str, freq: str = "D", asset: str = "E", adj: str = "hfq") -> pd.DataFrame:
        """Back-adjusted bars via promax.

        Columns: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        adj: 'qfq' (front) or 'hfq' (back)
        """
        body = self._promax("pro_bar", ts_code=ts_code, freq=freq, asset=asset, adj=adj,
                            start_date=start_date, end_date=end_date)
        df = self._to_df(body)
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        for c in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
