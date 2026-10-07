"""数据源适配层：AKShare(A股) / yfinance(美股) 及多级真实备份后端。

统一 DataFeed 契约，供上层(因子/策略/回测)透明使用：
- fetch_daily(market, symbol)  -> DataFrame(OHLCV + amount + turnover)
- fetch_candidates(market)     -> DataFrame(代码/名称/市值/成交额/行业…)
- fetch_fundamental(market, symbol) -> dict(财务指标尽力折算)
- fetch_benchmark(market)      -> DataFrame(指数/ETF 日线)

数据源标记: real(akshare/yfinance/baostock/腾讯/新浪/东财)。
说明: 所有真实后端同时失败时直接抛错——绝不回退到任何离线人造数据。
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from quantlab.config import CONFIG, DataSource
from quantlab.data.store import DataStore

US_SYMBOLS = None  # 占位：用于 SP500 常见池


class DataFeed:
    def __init__(self, store: DataStore | None = None, source: str | None = None):
        from quantlab.config_validation import validate_config
        validate_config(CONFIG)
        self.cfg: DataSource = CONFIG.data
        self.store = store or DataStore()
        self.source = source or "auto"
        self._used_source: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 查询器
    # ------------------------------------------------------------------
    def _ak(self):
        import akshare as ak
        return ak

    def fetch_industry_a(self) -> dict[str, str] | None:
        """全A真实行业映射 {code: 行业名}（东财行业板块成分，一次性拉取后缓存）。
        供因子行业中性化与组合行业约束使用；失败(网络/接口)时返回 None，上游按 NA 处理。"""
        key = "industries_a"
        cached = self.store.load("fund", key)
        if cached is not None and len(cached) and not self.cfg.reset_cache:
            return cached["industry"].to_dict()
        try:
            ak = self._ak()
            boards = ak.stock_board_industry_name_em()
            if boards is None or boards.empty:
                return None
            mapping: dict[str, str] = {}
            for b in boards["板块名称"].astype(str).tolist():
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=b)
                    for c in cons["代码"].astype(str):
                        mapping.setdefault(c.zfill(6), b)
                except Exception:  # noqa: BLE001
                    continue
            if not mapping:
                print("[data] 东财行业成分拉取为空")
                return None
            df = pd.DataFrame({"industry": pd.Series(mapping)})
            self.store.save(df, "fund", key)
            return mapping
        except Exception as e:  # noqa: BLE001
            print(f"[data] industry fetch failed ({e})")
            return None

    def _yf(self):
        import yfinance as yf
        return yf

    # ------------------------------------------------------------------
    # 候选池
    # ------------------------------------------------------------------
    def fetch_candidates(self, market: str) -> pd.DataFrame:
        if market == "a":
            return self._a_candidates()
        if market == "us":
            return self._us_candidates()
        raise ValueError(market)

    def _a_candidates(self) -> pd.DataFrame:
        key = "candidates_a"
        cached = self.store.load("a", key)
        fetched = cached.attrs.get("fetched_at") if cached is not None else None
        fresh = fetched is not None and pd.Timestamp(fetched).date() == pd.Timestamp.now(tz="UTC").date()
        if cached is not None and fresh and not self.cfg.reset_cache:
            return cached
        df = None
        if self.source in ("auto", "akshare"):
            try:
                ak = self._ak()
                spot = ak.stock_zh_a_spot_em()
                spot = spot.rename(columns={
                    "代码": "code", "名称": "name", "最新价": "price",
                    "总市值": "mktcap", "流通市值": "float_mktcap",
                    "成交额": "amount", "换手率": "turnover", "涨跌幅": "pct_chg",
                })
                need = ["code", "name", "price", "mktcap", "float_mktcap", "amount", "turnover", "pct_chg"]
                spot = spot[[c for c in need if c in spot.columns]].copy()
                spot["mktcap"] = pd.to_numeric(spot["mktcap"], errors="coerce")
                spot["float_mktcap"] = pd.to_numeric(spot["float_mktcap"], errors="coerce")  # already yuan
                spot["amount"] = pd.to_numeric(spot["amount"], errors="coerce")
                spot["turnover"] = pd.to_numeric(spot["turnover"], errors="coerce")
                include = self._a_filter(spot)
                df = spot[include].copy()
                # 真实行业映射(供行业中性化/行业约束)；失败则 NA
                if "sector" not in df.columns:
                    ind = self.fetch_industry_a()
                    if ind:
                        df["sector"] = df["code"].astype(str).map(ind).fillna("NA")
                    else:
                        df["sector"] = "NA"
                df["source"] = "akshare"
            except Exception as e:  # noqa: BLE001
                print(f"[data] AKShare candidates failed ({e}); trying backup backends")
        if df is None or df.empty:
            try:
                df = self._sina_spot()
                if df is not None and not df.empty:
                    print("[data] sina backend OK")
            except Exception as e:  # noqa: BLE001
                print(f"[data] sina candidates failed ({e})")
        if df is None or df.empty:
            try:
                df = self._east_spot()
                if df is not None and not df.empty:
                    print("[data] eastmoney backend OK")
            except Exception as e:  # noqa: BLE001
                print(f"[data] eastmoney backend candidates failed ({e})")
        if df is None or df.empty:
            raise RuntimeError("[data] A股候选池所有真实后端(akshare/sina/eastmoney)均失败，停止并拒绝人造回退")
        df = df.set_index("code", drop=False)
        self.store.save(df, "a", key)
        return df

    def _a_filter(self, spot: pd.DataFrame) -> pd.Series:
        u = CONFIG.universe
        incl = pd.Series(True, index=spot.index)
        if u.a_exclude_st:
            incl &= ~spot["name"].astype(str).str.contains("ST|退", regex=True)
        float_cap = spot.get("float_mktcap")
        if float_cap is not None:
            incl &= float_cap.fillna(0) >= u.a_min_mktcap_yi * 1e8
        amt = spot.get("amount")
        if amt is not None:
            incl &= amt.fillna(0) >= u.a_min_avg_amount_wan * 1e4
        price = spot.get("price")
        if price is not None:
            incl &= price > 0
        return incl

    def _us_candidates(self):
        raise RuntimeError("US stock selection/execution is disabled; use timestamped context datasets for background research")

    # ------------------------------------------------------------------
    # 日线
    # ------------------------------------------------------------------
    def fetch_daily(self, market: str, symbol: str, *, refresh=False, expected_session=None) -> pd.DataFrame | None:
        self._refresh = refresh
        self._expected_session = expected_session
        if market == "a":
            return self._a_daily(symbol)
        if market == "us":
            return self._us_daily(symbol)
        raise ValueError(market)

    def _a_daily(self, symbol: str) -> pd.DataFrame | None:
        key = f"daily_raw_{symbol}" if not self.cfg.a_daily_adjust else f"daily_{self.cfg.a_daily_adjust}_{symbol}"
        cached = self.store.load("a", key)
        expected = getattr(self, "_expected_session", None)
        if expected is None:
            now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
            expected = now.normalize() if now.hour >= 18 else now.normalize() - pd.offsets.BDay()
            if expected.weekday() >= 5: expected = expected - pd.offsets.BDay()
        expected = pd.Timestamp(expected)
        if cached is not None and len(cached) and cached.index[-1] >= expected and not self.cfg.reset_cache and not getattr(self, "_refresh", False):
            self._used_source[symbol] = cached.attrs.get("source", "cache")
            return cached
        self._request_start = (cached.index[-1] - pd.Timedelta(days=7)) if cached is not None and len(cached) and not self.cfg.reset_cache else pd.Timestamp("today") - pd.DateOffset(years=self.cfg.history_years)
        df = None
        if self.source in ("auto", "akshare"):
            try:
                ak = self._ak()
                end = pd.Timestamp("today").strftime("%Y%m%d")
                start = self._request_start.strftime("%Y%m%d")
                raw = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                         start_date=start, end_date=end,
                                         adjust=self.cfg.a_daily_adjust)
                if raw is None or raw.empty:
                    raise ValueError("empty")
                raw = raw.rename(columns={
                    "日期": "datetime", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "换手率": "turnover", "涨跌幅": "pct_chg",
                })
                df = raw[[c for c in ["datetime", "open", "close", "high", "low", "volume", "amount", "turnover", "pct_chg"] if c in raw.columns]].copy()
                for c in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = self._normalize_daily(df)
                df.attrs["source"] = "akshare"
                self._sleep()
            except Exception as e:  # noqa: BLE001
                print(f"[data] AKShare daily {symbol} failed ({e})")
        if df is None or df.empty:
            try:
                df = self._baostock_daily(symbol)
            except Exception as e:  # noqa: BLE001
                print(f"[data] baostock daily {symbol} failed ({e})")
        if df is None or df.empty:
            try:
                df = self._east_daily(symbol)
            except Exception as e:  # noqa: BLE001
                print(f"[data] eastmoney daily {symbol} failed ({e})")
        if df is None or df.empty:
            try:
                df = self._sina_daily(symbol)
            except Exception as e:  # noqa: BLE001
                print(f"[data] backup daily {symbol} failed ({e})")
        if df is not None and len(df):
            df.attrs["price_basis"] = "raw" if not self.cfg.a_daily_adjust else self.cfg.a_daily_adjust
            df.attrs["last_requested_session"] = str(expected.date())
            df["source"] = df.attrs.get("source", "unknown")
            df["revision"] = pd.Timestamp.now(tz="UTC").isoformat()
            df["available_at"] = df["revision"]  # observed now, never invented historical availability
            df["quality_status"] = "unverified_public_feed"
            df['open_status']='unknown'
            df['open_status_available_at']=pd.NaT
            try:
                self.store.save(df, "a", key)
                df = self.store.load("a", key)
            except ValueError as exc:
                if cached is None: raise
                import warnings
                warnings.warn('Cache update rejected; preserved previous version: '+str(exc),RuntimeWarning)
                df = cached.copy()
                df.attrs["update_error"] = str(exc)
        elif cached is not None:
            df = cached.copy()
            df.attrs["update_error"] = "all_sources_failed"
        if df is not None:
            df.attrs["freshness"] = "current" if df.index[-1] >= expected else "stale"
            self._used_source[symbol] = df.attrs.get("source", "unknown")
        return df

    def _us_daily(self, symbol: str) -> pd.DataFrame | None:
        key=f'daily_yfinance_adjusted_{symbol}'
        cached = self.store.load("us", key)
        if cached is not None and len(cached) and not self.cfg.reset_cache:
            self._used_source[symbol] = cached.attrs.get("source", "cache")
            return cached
        df = None
        if self.source in ("auto", "yfinance"):
            try:
                yf = self._yf()
                end = pd.Timestamp("today")
                start = end - pd.DateOffset(years=self.cfg.history_years)
                raw = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                                  auto_adjust=True, progress=False)
                if raw is None or raw.empty:
                    raise ValueError("empty")
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                         "Close": "close", "Volume": "volume"})
                df = self._normalize_daily(df)
                if "volume" in df.columns:
                    df["turnover"] = df["volume"] / df["volume"].rolling(20).mean()
                    df["amount"] = df["close"] * df["volume"]
                df.attrs["source"] = "yfinance"
                df.attrs.update(price_basis='yfinance_adjusted',quality_status='unverified_context_only',amount_unit='USD_estimate')
            except Exception as e:  # noqa: BLE001
                print(f"[data] yfinance daily {symbol} failed ({e})")
        if df is None or df.empty:
            return None
        if df is not None and len(df):
            self.store.save(df, "us", key)
        return df

    def _normalize_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)
            df = df[df["datetime"].notna()].set_index("datetime").sort_index()
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if "pct_chg" not in df.columns and "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change(fill_method=None)
        if "turnover" in df.columns:
            df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # 财务
    # ------------------------------------------------------------------
    def fetch_fundamental(self, market: str, symbol: str) -> pd.DataFrame | None:
        key = f"fund_{symbol}"
        cached = self.store.load("fund", key)
        if cached is not None and len(cached):
            return cached
        df = None
        if market == "a" and self.source in ("auto", "akshare"):
            try:
                ak = self._ak()
                raw = ak.stock_financial_analysis_indicator(symbol=symbol)
                # Report period is not publication time: research loader rejects this as PIT. 
                if raw is None or raw.empty:
                    raise ValueError("empty")
                raw = raw.rename(columns={
                    "日期": "datetime", "净利润": "net_profit_y", "净资产收益率": "roe",
                    "销售净利率": "net_margin", "销售毛利率": "gross_margin",
                    "每股收益": "eps", "每股净资产": "bps",
                })
                keep = [c for c in ["datetime", "net_profit_y", "roe", "net_margin", "gross_margin", "eps", "bps"] if c in raw.columns]
                df = raw[keep].copy()
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
                df = df[~df.index.duplicated(keep="last")]
                df.attrs["source"] = "akshare"
                self._sleep()
            except Exception as e:  # noqa: BLE001
                print(f"[data] fund {symbol} failed ({e})")
        if (df is None or df.empty) and market == "us" and self.source in ("auto", "yfinance"):
            try:
                yf = self._yf()
                t = yf.Ticker(symbol)
                info = t.get_info() or {}
                if not info:
                    raise ValueError("empty info")
                df = pd.DataFrame([{
                    "datetime": pd.Timestamp("today"),
                    "roe": info.get("returnOnEquity"),
                    "gross_margin": info.get("grossMargins"),
                    "eps": info.get("trailingEps"),
                    "eps_ttm": info.get("trailingEps"),
                    "pe": info.get("trailingPE"),
                    "pb": info.get("priceToBook"),
                    "fcf_yield": info.get("freeCashflow") and abs(info.get("freeCashflow", 0)) / max(info.get("marketCap", 1), 1),
                }])
                df = df.set_index("datetime")
                df.attrs["source"] = "yfinance"
            except Exception as e:  # noqa: BLE001
                print(f"[data] us fund {symbol} failed ({e})")
        if df is None or df.empty:
            return None
        df.attrs["pit_status"] = "unverified_report_period_only"
        self.store.save(df, "fund", key)
        return df

    # ------------------------------------------------------------------
    # 基准
    # ------------------------------------------------------------------
    def fetch_benchmark(self, market: str) -> pd.DataFrame | None:
        if market != "a":
            raise RuntimeError("US execution benchmark is not enabled")
        symbol = str(CONFIG.benchmark_a).zfill(6)
        key = f"bench_index_sh{symbol}"
        cached = self.store.load("bench", key)
        try:
            raw = self._ak().stock_zh_index_daily_em(symbol=f"sh{symbol}")
            if raw is None or raw.empty: raise ValueError("empty index")
            raw = raw.rename(columns={"date": "datetime"})
            df = self._normalize_daily(raw)
            df.attrs.update(source="akshare_index", instrument_id=f"XSHG.INDEX.{symbol}",
                            asset_type="index", price_basis="index_price")
            self.store.save(df, "bench", key)
            return self.store.load("bench", key)
        except Exception as exc:
            if cached is not None and cached.attrs.get("asset_type") == "index":
                cached.attrs["update_error"] = str(exc)
                cached.attrs["freshness"] = "unverified"
                return cached
            raise RuntimeError(f"index benchmark unavailable: {symbol}; {exc}") from exc

    def _sina_spot(self) -> pd.DataFrame | None:
        """新浪后端：全A实时快照(候选池)。"""
        from quantlab.data import sina as sn
        df = sn.fetch_spot_candidates()
        if df is None or df.empty:
            return None
        df = df.copy()
        include = self._a_filter(df)
        return df[include].copy()

    def _sina_daily(self, symbol: str) -> pd.DataFrame | None:
        """腾讯后端：前复权日线。"""
        from quantlab.data import sina as sn
        df = sn.fetch_kline(symbol, start=str(getattr(self, "_request_start", pd.Timestamp("today") - pd.DateOffset(years=self.cfg.history_years)).date()))
        if df is None or df.empty:
            return None
        df = df.copy()
        df["amount"] = float("nan")
        df["turnover"] = float("nan")
        df["pct_chg"] = df["close"].pct_change()
        return self._normalize_daily(df)

    def _baostock_daily(self, symbol: str) -> pd.DataFrame | None:
        """Baostock 后端：全A前复权日线（含换手/涨跌幅/isST）。"""
        import baostock as bs
        cfg = CONFIG.data
        if str(symbol).startswith(('4', '8', '92')):
            return None  # baostock 不提供北交所行情
        sym = f"{'sh' if str(symbol).startswith(('6', '5', '9')) else 'sz'}.{symbol}"
        if not getattr(self, "_bs_logged_in", False):
            lg = bs.login()
            self._bs_logged_in = (lg.error_code == '0')
            if not self._bs_logged_in:
                return None
        start = getattr(self, "_request_start", pd.Timestamp("today") - pd.DateOffset(years=cfg.history_years)).strftime("%Y-%m-%d")
        end = pd.Timestamp("today").strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(
            sym, "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            start_date=start, end_date=end, frequency="d", adjustflag="3" if not cfg.a_daily_adjust else "1" if cfg.a_daily_adjust == "hfq" else "2")
        if rs is None or rs.error_code != '0':
            return None
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=rs.fields)
        for c in ("open", "high", "low", "close", "volume", "amount", "turn", "pctChg"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.rename(columns={"date": "datetime", "turn": "turnover", "pctChg": "pct_chg"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df["code"] = symbol
        df.attrs["source"] = "baostock"
        return self._normalize_daily(df)

    def _east_spot(self) -> pd.DataFrame | None:
        """东财 curl_cffi 后端：全A实时快照(候选池)。"""
        from quantlab.data import eastmoney as em
        df = em.fetch_spot_candidates()
        if df is None or df.empty:
            return None
        include = self._a_filter(df)
        return df[include].copy()

    def _east_daily(self, symbol: str) -> pd.DataFrame | None:
        """东财 curl_cffi 后端：历史K线(前复权)。"""
        from quantlab.data import eastmoney as em
        df = em.fetch_kline(symbol, start=getattr(self, "_request_start", pd.Timestamp("today") - pd.DateOffset(years=self.cfg.history_years)).strftime("%Y%m%d"))
        if df is None or df.empty:
            return None
        return self._normalize_daily(df)

    def _sleep(self) -> None:
        time.sleep(self.cfg.rate_limiter_sec)

    def source_of(self, symbol: str) -> str:
        return self._used_source.get(symbol, "unknown")


if __name__ == "__main__":
    feed = DataFeed()
    c = feed.fetch_candidates("a")
    print("candidates:", c.shape, list(c.columns))
    sym = c["code"].iloc[0]
    px = feed.fetch_daily("a", str(sym))
    print("daily:", sym, px.shape if px is not None else None, px.attrs.get("source") if px is not None else None)
