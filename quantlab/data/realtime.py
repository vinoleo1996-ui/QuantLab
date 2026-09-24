"""实时行情查询：新浪批量日线 + 腾讯盘中 fallback。

均为免 token 直连接口：
- 新浪  hq.sinajs.cn/list=sh600519,sz000001,...
- 腾讯  qt.gtimg.cn/q=sh600519,sz000001,...

返回统一 DataFrame: code/name/price/pre_close/open/high/low/volume/amount/time/source
"""

from __future__ import annotations

import time

import pandas as pd
import requests

_TIMEOUT = 8
_HDR = {"Referer": "https://finance.sina.com.cn/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/124.0 Safari/537.36"}
_SINA = "https://hq.sinajs.cn/list="
_TX = "https://qt.gtimg.cn/q="


def _sym(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("92", "43", "83", "87")):
        return f"bj{c}"  # 北交所: 920xxx 新号段 + 43/83/87 老三板号段
    if c.startswith(("6", "5", "9", "900")):
        return f"sh{c}"
    return f"sz{c}"


def _sina(codes: list[str]) -> tuple[list[dict], str | None]:
    body = ",".join(_sym(c) for c in codes)
    r = requests.get(_SINA + body, headers=_HDR, timeout=_TIMEOUT)
    r.raise_for_status()
    rows = []
    time_ = None
    for line in r.text.strip().splitlines():
        if "hq_str_" not in line:
            continue
        sym = line.split("hq_str_")[1].split("=")[0]
        payload = line.split('="')[1].rstrip('"')
        f = payload.split(",")
        if len(f) < 10 or not f[0]:
            continue
        def num(i: int) -> float:
            try:
                return float(f[i])
            except (ValueError, IndexError):
                return float("nan")
        if f[-3] and f[-2] and time_ is None:
            time_ = f"{f[-3]} {f[-2]}"
        rows.append({
            "code": sym[2:], "name": f[0], "open": num(1), "pre_close": num(2),
            "price": num(3), "high": num(4), "low": num(5),
            "volume": num(8), "amount": num(9),
        })
    return rows, time_


def _tencent(codes: list[str]) -> tuple[list[dict], str | None]:
    body = ",".join(_sym(c) for c in codes)
    r = requests.get(_TX + body, headers=_HDR, timeout=_TIMEOUT)
    r.raise_for_status()
    rows = []
    time_ = None
    for line in r.text.strip().splitlines():
        if "v_" not in line:
            continue
        sym = line.split("v_")[1].split("=")[0]
        f = line.split('"')[1].split("~")
        if len(f) < 8 or not f[1]:
            continue
        def num(i: int) -> float:
            try:
                return float(f[i])
            except (ValueError, IndexError):
                return float("nan")
        if len(f) > 32 and f[30] and f[31] and time_ is None:
            time_ = f"{f[30]} {f[31]}"
        rows.append({
            "code": f[2] or sym[2:], "name": f[1], "price": num(3), "pre_close": num(4),
            "open": num(5), "high": num(33) if len(f) > 33 else float("nan"),
            "low": num(34) if len(f) > 34 else float("nan"),
            "volume": num(6), "amount": num(37) if len(f) > 37 else float("nan"),
        })
    return rows, time_


def batch_quotes(codes: list[str]) -> pd.DataFrame:
    """批量实时报价。新浪失败自动切腾讯。"""
    codes = [str(c).zfill(6) for c in codes if str(c)]
    if not codes:
        return pd.DataFrame(columns=["code", "name", "price", "pre_close", "open",
                                     "high", "low", "volume", "amount", "pct_chg", "time", "source"])
    rows, t = [], None
    # 新浪
    try:
        rows, t = _sina(codes)
    except Exception:  # noqa: BLE001
        rows, t = [], None
    src = "sina"
    if len(rows) < len(codes):
        try:
            r2, t2 = _tencent(codes)
        except Exception:  # noqa: BLE001
            r2, t2 = [], None
        if r2:
            rows = r2
            src = "tencent"
            t = t or t2
            time.sleep(0.25)
    if not rows:
        return pd.DataFrame(columns=["code", "name", "price", "pre_close", "open",
                                     "high", "low", "volume", "amount", "pct_chg", "time", "source"])
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c in ("price", "pre_close", "open", "high", "low", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pct_chg"] = (df["price"] / df["pre_close"] - 1.0).replace([float("inf"), -float("inf")], float("nan"))
    df["time"] = t or ""
    df["source"] = src
    return df.set_index("code", drop=False)


def quote_one(code: str) -> dict:
    df = batch_quotes([code])
    if df.empty:
        return {"code": code, "name": "", "price": None, "pct_chg": None, "time": ""}
    row = df.iloc[0].to_dict()
    return row