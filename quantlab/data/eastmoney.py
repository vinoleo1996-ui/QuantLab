"""东方财富直连后端（基于 curl_cffi，规避 requests 被 WAF 指纹封锁）。

为什么需要它：
- 部分网络下 akshare 依赖的 requests/urllib3 TLS 指纹(JA3)被东财 WAF 直接断开
  (RemoteDisconnected)；而 curl_cffi impersonate=chrome 携带浏览器指纹可通过。
- 本模块提供 候选池(全A实时快照) 与 历史K线(前复权) 两条东财接口的 curl_cffi 直连，
  作为 akshare 的 fallback；两者都失败再由上层回退合成数据。

接口：
  fetch_spot_candidates() -> DataFrame
  fetch_kline(code)       -> DataFrame(datetime/open/close/high/low/...)
"""

from __future__ import annotations

import time

import pandas as pd

from curl_cffi import requests as creq

from quantlab.config import CONFIG

_UT = "bd1d9ddb04089700cf9c27f6f7426281"
_TIMEOUT = 10
_BASE = "https://push2.eastmoney.com/api/qt/clist/get"
_KBASE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
# 集合：深主板+创业板+科创+主板A(沪)+北交所，即全A
_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81,s:2048"
_FIELDS = "f2,f3,f5,f6,f8,f9,f12,f14,f20,f21,f23,f18,f16,f15,f17"


def _get(url: str) -> creq.Response:
    r = creq.get(url, timeout=_TIMEOUT, impersonate="chrome",
                 headers={"Referer": "https://quote.eastmoney.com/"})
    return r


def fetch_spot_candidates() -> pd.DataFrame | None:
    """东财全A实时快照(分页)。失败返回 None。"""
    rows: list[dict] = []
    total = 0
    pn = 1
    fails = 0
    for _ in range(120):
        url = (f"{_BASE}?pn={pn}&pz=100&po=1&np=1&ut={_UT}&fltt=2&invt=2"
               f"&fid=f6&fs={_FS}&fields={_FIELDS}")
        try:
            data = _get(url).json()
        except Exception:
            fails += 1
            if fails >= 3:
                return None
            time.sleep(1.0)
            continue
        fails = 0
        blob = data.get("data") or {}
        if not blob or not blob.get("diff"):
            break
        if pn == 1:
            total = int(blob.get("total") or 0)
        rows.extend(blob["diff"])
        if total and pn * 100 >= total:
            break
        pn += 1
        time.sleep(0.25)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["f12"] = df["f12"].astype(str).str.zfill(6)
    df = df.rename(columns={
        "f2": "price", "f3": "pct_chg", "f5": "volume", "f6": "amount",
        "f8": "turnover", "f9": "pe", "f12": "code", "f14": "name",
        "f20": "mktcap", "f21": "float_mktcap", "f15": "high", "f16": "low",
        "f17": "open", "f18": "pre_close",
    })
    for c in ("price", "pct_chg", "volume", "amount", "turnover", "pe",
              "mktcap", "float_mktcap", "high", "low", "open", "pre_close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in ("code", "name", "price", "open", "high", "low", "pre_close",
                        "mktcap", "float_mktcap", "amount", "turnover", "pct_chg", "pe") if c in df.columns]
    df = df[keep]
    df["float_mktcap"] = df["float_mktcap"].fillna(df["mktcap"])
    df["mktcap"] = df["mktcap"]
    df["source"] = "eastmoney"
    return df.set_index("code", drop=False)


def _secid(code: str) -> str:
    c = str(code)
    return f"1.{c}" if c.startswith(("5", "6", "9")) else f"0.{c}"


def fetch_kline(code: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    """东财历史K线(101=日线, fqt=2 后复权, 历史序列稳定)。返回标准 OHLCV 面板。"""
    cfg = CONFIG.data
    beg = (start or (pd.Timestamp("today") - pd.DateOffset(years=cfg.history_years)).strftime("%Y%m%d"))
    end_d = end or pd.Timestamp("today").strftime("%Y%m%d")
    url = (f"{_KBASE}?secid={_secid(code)}&klt=101&fqt={0 if not cfg.a_daily_adjust else 2 if cfg.a_daily_adjust == "hfq" else 1}&beg={beg}&end={end_d}"
           f"&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61")
    try:
        blob = _get(url).json()
    except Exception:
        return None
    data = (blob or {}).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return None
    recs = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 8:
            continue
        recs.append({
            "datetime": pd.Timestamp(parts[0]), "open": float(parts[1]),
            "close": float(parts[2]), "high": float(parts[3]), "low": float(parts[4]),
            "volume": float(parts[5]), "amount": float(parts[6]),
            "turnover": float(parts[10]) if len(parts) > 10 else float("nan"),
            "pct_chg": float(parts[8]) if len(parts) > 8 else float("nan"),
        })
    if not recs:
        return None
    df = pd.DataFrame(recs).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["code"] = code
    df.attrs["source"] = "eastmoney"
    return df