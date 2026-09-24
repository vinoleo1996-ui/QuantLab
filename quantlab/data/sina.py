"""Public Sina candidate snapshots and Tencent K-lines. These are unverified public feeds; successful requests do not establish historical membership, opening tradability or PIT availability."""

from __future__ import annotations

import re
import time

import pandas as pd
import requests

from quantlab.config import CONFIG

_TIMEOUT = 10
_HDR = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
}
_SPOT = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
_QTEN = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_START_COL = {
    "open": 1, "close": 2, "high": 3, "low": 4, "volume": 5,
}


def fetch_spot_candidates() -> pd.DataFrame | None:
    """新浪全A快照(分页)。返回候选表；失败返回 None。"""
    rows: list[dict] = []
    page = 1
    for _ in range(40):
        try:
            r = requests.get(_SPOT, params={
                "page": page, "num": 80, "sort": "amount", "asc": 0, "node": "hs_a",
            }, headers=_HDR, timeout=_TIMEOUT)
            r.raise_for_status()
            part = r.json()
        except Exception as exc:
            raise RuntimeError(f'sina_candidate_page_failed:{page}; incomplete snapshot rejected') from exc
        if not part:
            break
        rows.extend(part)
        if len(part) < 80:
            break
        page += 1
        time.sleep(0.2)
    if len(rows) < 50:
        return None
    df = pd.DataFrame(rows)
    for c in ("trade", "settlement", "changepercent", "turnoverratio", "amount",
              "mktcap", "nmc", "volume", "high", "low", "open"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"trade": "price", "code": "code", "name": "name",
                            "changepercent": "pct_chg", "turnoverratio": "turnover",
                            "mktcap": "mktcap", "nmc": "float_mktcap",
                            "amount": "amount", "volume": "volume"})
    keep = [c for c in ("code", "name", "price", "open", "high", "low", "pre_close",
                        "mktcap", "float_mktcap", "amount", "volume", "turnover", "pct_chg") if c in df.columns]
    df = df[keep]
    if "pre_close" not in df.columns and "settlement" in df.columns:
        df["pre_close"] = pd.to_numeric(df["settlement"], errors="coerce")
        keep.append("pre_close")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["mktcap"] = df["mktcap"].fillna(0) * 1e4
    df["float_mktcap"] = df["float_mktcap"].fillna(df["mktcap"]).fillna(0) * 1e4
    df["source"] = "sina"
    return df.set_index("code", drop=False)


def _sym(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("92", "43", "83", "87")):
        return f"bj{c}"
    if c.startswith(("6", "5", "9", "900")):
        return f"sh{c}"
    return f"sz{c}"


def fetch_kline(code: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    """腾讯前复权日线，按需分多段拉取直至覆盖所需历史。"""
    cfg = CONFIG.data
    need_start = start or (pd.Timestamp("today") - pd.DateOffset(years=cfg.history_years))
    if isinstance(need_start, str):
        need_start = pd.Timestamp(need_start)
    need_start = need_start.strftime("%Y-%m-%d")
    end_d = end or pd.Timestamp("today").strftime("%Y-%m-%d")
    sym = _sym(code)
    rows: list[list] = []
    for _ in range(8):
        beg = (pd.Timestamp(end_d) - pd.DateOffset(days=820)).strftime("%Y-%m-%d")
        try:
            r = requests.get(_QTEN, params={
                "param": f"{sym},day,{beg},{end_d},800,{cfg.a_daily_adjust}",
            }, headers=_HDR, timeout=_TIMEOUT)
            r.raise_for_status()
            blob = r.json()
            series = ((blob.get("data") or {}).get(sym) or {})
            bars = series.get((cfg.a_daily_adjust or "") + "day") or []
        except Exception:
            time.sleep(1.0)
            continue
        if bars:
            rows = bars + rows
            first = pd.Timestamp(bars[0][0]).strftime("%Y-%m-%d")
            if first <= need_start:
                break
            end_d = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            break
        time.sleep(0.2)
    if not rows:
        return None
    recs = []
    seen = set()
    for bar in rows:
        d = bar[0]
        if d in seen:
            continue
        seen.add(d)
        ts = pd.Timestamp(d)
        if ts < pd.Timestamp(need_start):
            continue
        def _num(i: int):
            try:
                return float(bar[i])
            except (IndexError, TypeError, ValueError):
                return float("nan")
        recs.append({
            "datetime": ts, "open": _num(1), "close": _num(2), "high": _num(3),
            "low": _num(4), "volume": _num(5),
        })
    if not recs:
        return None
    df = pd.DataFrame(recs).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["code"] = code
    df.attrs["source"] = "tencent"
    return df
