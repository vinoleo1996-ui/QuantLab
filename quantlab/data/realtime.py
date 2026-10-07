"""实时行情查询：新浪批量日线 + 腾讯盘中 fallback。

均为免 token 直连接口：
- 新浪  hq.sinajs.cn/list=sh600519,sz000001,...
- 腾讯  qt.gtimg.cn/q=sh600519,sz000001,...

返回统一 DataFrame: code/name/price/pre_close/open/high/low/volume/amount/time/source
"""

from __future__ import annotations

import time

import pandas as pd

_TIMEOUT = 8
_HDR = {"Referer": "https://finance.sina.com.cn/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/124.0 Safari/537.36"}
_SINA = "https://hq.sinajs.cn/list="
_TX = "https://qt.gtimg.cn/q="


def _get(url):
    try:
        import requests  # Optional public-data dependency, unused by offline research.
    except ImportError:
        requests = None
    if requests is not None:
        response = requests.get(url, headers=_HDR, timeout=_TIMEOUT)
        response.raise_for_status()  # Includes 429: finite failure, never an endless same-page retry.
        # Both supported public quote endpoints publish GBK payloads without a
        # dependable charset header.  Requests otherwise defaults to
        # ISO-8859-1, corrupting Chinese instrument names and defeating the
        # explicit index name contract below.
        response.encoding = "gbk"
        return response
    import urllib.request
    req = urllib.request.Request(url, headers=_HDR)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        raw = resp.read()
    text = raw.decode("gbk", errors="replace")

    class _Resp:
        pass

    out = _Resp()
    out.text = text
    return out


def _sym(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("92", "43", "83", "87")):
        return f"bj{c}"  # 北交所: 920xxx 新号段 + 43/83/87 老三板号段
    if c.startswith(("6", "5", "9", "900")):
        return f"sh{c}"
    return f"sz{c}"


def _sina(codes: list[str]) -> tuple[list[dict], str | None]:
    body = ",".join(_sym(c) for c in codes)
    r = _get(_SINA + body)
    rows = []
    time_ = None
    # Sina may concatenate multiple ``var hq_str_*`` assignments on one line.
    # Splitting only on line breaks would drop every record after the first,
    # then incorrectly trigger a second-provider request for supposedly
    # missing holdings.
    for line in r.text.strip().split(";"):
        if "hq_str_" not in line:
            continue
        try:
            sym = line.split("hq_str_", 1)[1].split("=", 1)[0].strip()
            payload = line.split('="', 1)[1].rstrip('"')
        except IndexError:
            continue
        f = payload.split(",")
        if len(f) < 10 or not f[0]:
            continue
        def num(i: int) -> float:
            try:
                return float(f[i])
            except (ValueError, IndexError):
                return float("nan")
        quote_time = f"{f[30]} {f[31]}" if len(f) > 31 and f[30] and f[31] else None
        if quote_time and time_ is None:
            time_ = quote_time
        rows.append({
            "code": sym[2:], "name": f[0], "open": num(1), "pre_close": num(2),
            "price": num(3), "high": num(4), "low": num(5),
            "volume": num(8), "amount": num(9), "time": quote_time,
        })
    return rows, time_


def _tencent(codes: list[str]) -> tuple[list[dict], str | None]:
    body = ",".join(_sym(c) for c in codes)
    r = _get(_TX + body)
    rows = []
    time_ = None
    # Tencent commonly concatenates quote assignments with semicolons on one
    # line.  Parse every assignment before deciding that a requested ticker is
    # missing and needs a second provider.
    for line in r.text.replace(";", "\n").strip().splitlines():
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
        quote_time = ""
        if len(f) > 30 and f[30]:
            parsed_time = pd.to_datetime(f[30], format="%Y%m%d%H%M%S", errors="coerce")
            if not pd.isna(parsed_time):
                quote_time = str(parsed_time)
        if quote_time and time_ is None:
            time_ = quote_time
        rows.append({
            "code": f[2] or sym[2:], "name": f[1], "price": num(3), "pre_close": num(4),
            "open": num(5), "high": num(33) if len(f) > 33 else float("nan"),
            "low": num(34) if len(f) > 34 else float("nan"),
            "volume": num(6)*100, "amount": num(37)*10000 if len(f) > 37 else float("nan"),
            "time": quote_time,
        })
    return rows, time_


# Index identifiers are not interchangeable with stock codes.  In particular,
# ``000001`` means Ping An Bank under the stock resolver but means the SSE
# Composite only with Tencent's explicit ``sh000001`` market symbol.
_INDEX_QUOTE_CONTRACTS = (
    ("sh000001", "000001", "上证指数"),
    ("sz399001", "399001", "深证成指"),
    ("sz399006", "399006", "创业板指"),
)


def _tencent_explicit_symbols(symbols: list[str]) -> tuple[list[dict], str | None]:
    """Read Tencent quotes by already-qualified market symbol.

    This intentionally does not call :func:`_sym`: callers such as the index
    strip need an exchange-qualified contract rather than stock-code routing.
    """

    expected = set(symbols)
    r = _get(_TX + ",".join(symbols))
    rows: list[dict] = []
    time_: str | None = None
    for line in r.text.replace(";", "\n").splitlines():
        if "v_" not in line or '"' not in line:
            continue
        try:
            symbol = line.split("v_", 1)[1].split("=", 1)[0].strip()
            fields = line.split('"', 1)[1].split('"', 1)[0].split("~")
        except IndexError:
            continue
        if symbol not in expected or len(fields) < 8 or not fields[1]:
            continue

        def num(index: int) -> float:
            try:
                return float(fields[index])
            except (ValueError, IndexError):
                return float("nan")

        quote_time = ""
        if len(fields) > 30 and fields[30]:
            parsed_time = pd.to_datetime(fields[30], format="%Y%m%d%H%M%S", errors="coerce")
            if not pd.isna(parsed_time):
                quote_time = str(parsed_time)
        if quote_time and time_ is None:
            time_ = quote_time
        rows.append({
            "symbol": symbol, "code": fields[2] or symbol[2:], "name": fields[1],
            "price": num(3), "pre_close": num(4), "open": num(5),
            "high": num(33) if len(fields) > 33 else float("nan"),
            "low": num(34) if len(fields) > 34 else float("nan"),
            "volume": num(6) * 100, "amount": num(37) * 10000 if len(fields) > 37 else float("nan"),
            "time": quote_time,
        })
    return rows, time_


def batch_index_quotes() -> pd.DataFrame:
    """Return only Tencent quotes that satisfy the index symbol/name contract.

    A missing, mismatched, stale, or malformed record remains absent.  The UI
    will render that entry as unavailable instead of relabelling a stock quote
    as an index or falling back to a local mark.
    """

    symbols = [symbol for symbol, _code, _name in _INDEX_QUOTE_CONTRACTS]
    contracts = {symbol: {"code": code, "name": name}
                 for symbol, code, name in _INDEX_QUOTE_CONTRACTS}
    try:
        raw_rows, fallback_time = _tencent_explicit_symbols(symbols)
    except Exception:  # noqa: BLE001 - public observations fail closed.
        raw_rows, fallback_time = [], None
    accepted: list[dict] = []
    for row in raw_rows:
        contract = contracts.get(str(row.get("symbol") or ""))
        if contract is None:
            continue
        code = str(row.get("code") or "").zfill(6)
        name = str(row.get("name") or "").strip()
        price = pd.to_numeric(row.get("price"), errors="coerce")
        pre_close = pd.to_numeric(row.get("pre_close"), errors="coerce")
        if code != contract["code"] or name != contract["name"]:
            continue
        if pd.isna(price) or pd.isna(pre_close) or price <= 0 or pre_close <= 0:
            continue
        accepted.append({
            **row, "code": code, "name": name, "source": "tencent_index",
            "time": row.get("time") or fallback_time or "",
        })
    if not accepted:
        return pd.DataFrame(columns=["code", "name", "price", "pre_close", "open", "high", "low",
                                     "volume", "amount", "pct_chg", "time", "source"])
    quotes = pd.DataFrame(accepted)
    quotes["pct_chg"] = (quotes["price"] / quotes["pre_close"] - 1.0).replace(
        [float("inf"), -float("inf")], float("nan"))
    return quotes.drop_duplicates("code", keep="first").set_index("code", drop=False)


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
    rows=[dict(row,source='sina',time=row.get('time') or t or '') for row in rows]
    if len(rows) < len(codes):
        try:
            missing=[c for c in codes if c not in {r['code'] for r in rows}]
            r2, t2 = _tencent(missing)
        except Exception:  # noqa: BLE001
            r2, t2 = [], None
        if r2:
            # Tencent returns a timestamp for each symbol.  Do not attach the
            # first record's time to its peers: that would make a delayed row
            # look fresh in the portfolio UI.
            rows.extend(dict(row, source="tencent", time=row.get("time") or t2 or "") for row in r2)
    if not rows:
        return pd.DataFrame(columns=["code", "name", "price", "pre_close", "open",
                                     "high", "low", "volume", "amount", "pct_chg", "time", "source"])
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c in ("price", "pre_close", "open", "high", "low", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pct_chg"] = (df["price"] / df["pre_close"] - 1.0).replace([float("inf"), -float("inf")], float("nan"))
    df=df.drop_duplicates('code',keep='first')
    return df.set_index("code", drop=False)


def quote_one(code: str) -> dict:
    df = batch_quotes([code])
    if df.empty:
        return {"code": code, "name": "", "price": None, "pct_chg": None, "time": ""}
    row = df.iloc[0].to_dict()
    return row
