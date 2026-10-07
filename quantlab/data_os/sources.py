"""免费源适配。每个适配器返回原始字节和标准化行，失败记在 errors 里，不改用别的日期顶上。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

import pandas as pd

from quantlab.data_os.identity import a_share_identity, us_identity


SEC_USER_AGENT = "QuantLab data_os research@localhost"


@dataclass
class Capture:
    source: str
    dataset: str
    payload: bytes
    market_rows: list[dict] = field(default_factory=list)
    fundamental_rows: list[dict] = field(default_factory=list)
    action_rows: list[dict] = field(default_factory=list)
    macro_rows: list[dict] = field(default_factory=list)
    universe_rows: list[dict] = field(default_factory=list)
    instrument: dict | None = None
    actions_checked: bool = False
    error: str | None = None


def _json_rows(rows: list[dict]) -> list[dict]:
    dumped = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
                item[key] = None
            elif isinstance(value, (int, float, str)):
                item[key] = value
            else:
                item[key] = str(value)
        dumped.append(item)
    return dumped


def _num(value):
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if pd.isna(number):
        return pd.NA
    return number


def parse_baostock_kline(text: str, symbol: str) -> list[dict]:
    identity = a_share_identity(symbol)
    rows = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 10 or parts[0] in {"date", "code"} or not parts[0][:4].isdigit():
            continue
        # date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus
        rows.append({
            "instrument_id": identity["instrument_id"],
            "symbol": identity["symbol"],
            "market": "CN",
            "trade_date": parts[0],
            "open": _num(parts[2]),
            "high": _num(parts[3]),
            "low": _num(parts[4]),
            "close": _num(parts[5]),
            "prev_close": _num(parts[6]),
            "volume": _num(parts[7]),
            "amount": _num(parts[8]),
            "turnover": _num(parts[9]) if len(parts) > 9 else pd.NA,
            "source": "baostock",
            "is_suspended": parts[10] == "0" if len(parts) > 10 else False,
            "is_trading": parts[10] != "0" if len(parts) > 10 else True,
        })
    return rows


def parse_baostock_profit(text: str, symbol: str) -> list[dict]:
    identity = a_share_identity(symbol)
    rows = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 4 or parts[0] in {"code", "pubDate"} or not parts[1][:4].isdigit():
            continue
        # code,pubDate,statDate, then named trailing pairs are not available in csv mode.
        # Expected headerless export: code,pubDate,statDate,concept,value
        if len(parts) >= 5:
            rows.append({
                "instrument_id": identity["instrument_id"],
                "concept": parts[3],
                "value": _num(parts[4]),
                "unit": "vendor",
                "period_start": pd.NA,
                "period_end": parts[2],
                "filed_date": parts[1],
                "accepted_datetime": pd.NA,
                "form": "baostock_profit",
                "accn": "",
                "source": "baostock",
                "revision_id": parts[1],
            })
    return rows


def _ak_date(value) -> str:
    if isinstance(value, (int, float)) and not pd.isna(value) and float(value) > 10**11:
        return pd.to_datetime(float(value), unit="ms", utc=True).date().isoformat()
    text = str(value)
    if text.isdigit() and len(text) >= 12:
        return pd.to_datetime(int(text), unit="ms", utc=True).date().isoformat()
    return text[:10]


def parse_akshare_hist(frame: pd.DataFrame, symbol: str) -> list[dict]:
    identity = a_share_identity(symbol)
    rows = []
    for _, item in frame.iterrows():
        volume = _num(item.get("成交量"))
        rows.append({
            "instrument_id": identity["instrument_id"],
            "symbol": identity["symbol"],
            "market": "CN",
            "trade_date": _ak_date(item.get("日期")),
            "open": _num(item.get("开盘")),
            "high": _num(item.get("最高")),
            "low": _num(item.get("最低")),
            "close": _num(item.get("收盘")),
            "prev_close": pd.NA,
            "volume": float(volume) * 100 if pd.notna(volume) else pd.NA,
            "amount": _num(item.get("成交额")),
            "turnover": _num(item.get("换手率")),
            "source": "akshare",
        })
    return rows


def parse_tencent_kline(payload: dict, symbol: str) -> list[dict]:
    identity = a_share_identity(symbol)
    data = payload.get("data") or {}
    node = data.get(identity["tencent_code"]) or {}
    series = node.get("day") or node.get("qfqday") or []
    rows = []
    for item in series:
        if len(item) < 6:
            continue
        volume = _num(item[5])
        rows.append({
            "instrument_id": identity["instrument_id"],
            "symbol": identity["symbol"],
            "market": "CN",
            "trade_date": item[0],
            "open": _num(item[1]),
            "close": _num(item[2]),
            "high": _num(item[3]),
            "low": _num(item[4]),
            "volume": float(volume) * 100 if pd.notna(volume) else pd.NA,
            "amount": pd.NA,
            "prev_close": pd.NA,
            "turnover": pd.NA,
            "source": "tencent",
            "price_basis": "raw" if "day" in node else "unknown",
        })
    return rows


def parse_stooq(text: str, symbol: str) -> list[dict]:
    identity = us_identity(symbol)
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 6 or parts[0].lower() == "date":
            continue
        rows.append({
            "instrument_id": identity["instrument_id"],
            "symbol": identity["symbol"],
            "market": "US",
            "trade_date": parts[0],
            "open": _num(parts[1]),
            "high": _num(parts[2]),
            "low": _num(parts[3]),
            "close": _num(parts[4]),
            "volume": _num(parts[5]),
            "amount": pd.NA,
            "prev_close": pd.NA,
            "turnover": pd.NA,
            "source": "stooq",
        })
    return rows


def parse_yahoo_chart(payload: dict, symbol: str) -> tuple[list[dict], list[dict]]:
    identity = us_identity(symbol, payload.get("exchange") or "US")
    chart = payload["chart"]["result"][0]
    stamps = chart["timestamp"]
    quote = chart["indicators"]["quote"][0]
    market_rows = []
    for index, stamp in enumerate(stamps):
        day = pd.to_datetime(stamp, unit="s", utc=True).tz_convert("America/New_York").date().isoformat()
        market_rows.append({
            "instrument_id": identity["instrument_id"],
            "symbol": identity["symbol"],
            "market": "US",
            "trade_date": day,
            "open": _num(quote["open"][index]),
            "high": _num(quote["high"][index]),
            "low": _num(quote["low"][index]),
            "close": _num(quote["close"][index]),
            "volume": _num(quote["volume"][index]),
            "amount": pd.NA,
            "prev_close": pd.NA,
            "turnover": pd.NA,
            "source": "yahoo",
        })
    actions = []
    events = chart.get("events") or {}
    for kind, mapping in (("split", events.get("splits") or {}), ("dividend", events.get("dividends") or {})):
        for item in mapping.values():
            day = pd.to_datetime(item["date"], unit="s", utc=True).tz_convert("America/New_York").date().isoformat()
            actions.append({
                "instrument_id": identity["instrument_id"],
                "event_date": day,
                "event_type": kind,
                "split_ratio": item.get("splitRatio", pd.NA) if kind == "split" else pd.NA,
                "cash_dividend": item.get("amount", pd.NA) if kind == "dividend" else pd.NA,
                "stock_dividend": pd.NA,
                "rights_ratio": pd.NA,
                "effective_date": day,
                "announcement_date": pd.NA,
                "source": "yahoo",
            })
    return market_rows, actions


def parse_sec_companyfacts(payload: dict, symbol: str, concepts: set[str], accepted: dict[str, str]) -> list[dict]:
    identity = us_identity(symbol, "SEC")
    identity["cik"] = str(payload.get("cik", "")).zfill(10)
    rows = []
    facts = payload.get("facts") or {}
    for taxonomy in facts.values():
        for concept, body in taxonomy.items():
            if concepts and concept not in concepts:
                continue
            units = body.get("units") or {}
            for unit, items in units.items():
                for item in items:
                    accn = item.get("accn") or ""
                    rows.append({
                        "instrument_id": identity["instrument_id"],
                        "concept": concept,
                        "value": _num(item.get("val")),
                        "unit": unit,
                        "period_start": item.get("start"),
                        "period_end": item.get("end"),
                        "filed_date": item.get("filed"),
                        "accepted_datetime": accepted.get(accn) or accepted.get(accn.replace("-", "")),
                        "form": item.get("form"),
                        "accn": accn,
                        "source": "sec",
                        "revision_id": accn,
                    })
    return rows


def parse_fred_csv(text: str, series_id: str, releases: dict[str, str] | None = None) -> list[dict]:
    rows = []
    releases = releases or {}
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2 or parts[1] in {".", ""}:
            continue
        observation = parts[0]
        inline_release = parts[2].strip() if len(parts) > 2 and parts[2].strip() not in {"", "."} else None
        rows.append({
            "series_id": series_id,
            "observation_date": observation,
            "value": _num(parts[1]),
            "release_date": inline_release or releases.get(observation),
            "source": "fred",
        })
    return rows


def fetch_bytes(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
    from quantlab.data_os.net import fetch_bytes as _fetch
    return _fetch(url, headers=headers, timeout=timeout)


def capture_fred(series_id: str) -> Capture:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        payload = fetch_bytes(url)
        return Capture("fred", series_id, payload, macro_rows=parse_fred_csv(payload.decode("utf-8"), series_id))
    except Exception as exc:  # noqa: BLE001
        return Capture("fred", series_id, b"", error=str(exc))


def capture_stooq(symbol: str) -> Capture:
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    try:
        payload = fetch_bytes(url)
        rows = parse_stooq(payload.decode("utf-8"), symbol)
        return Capture("stooq", "daily", payload, market_rows=rows, instrument=us_identity(symbol))
    except Exception as exc:  # noqa: BLE001
        return Capture("stooq", "daily", b"", error=str(exc))


def capture_tencent(symbol: str, start: str, end: str) -> Capture:
    identity = a_share_identity(symbol)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={identity['tencent_code']},day,{start},{end},640,"
    try:
        payload = fetch_bytes(url)
        body = json.loads(payload.decode("utf-8"))
        return Capture("tencent", "daily", payload, market_rows=parse_tencent_kline(body, symbol), instrument=identity)
    except Exception as exc:  # noqa: BLE001
        return Capture("tencent", "daily", b"", error=str(exc))


def capture_yahoo(symbol: str, start: str, end: str) -> Capture:
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        history = ticker.history(start=start, end=end, auto_adjust=False, actions=True)
        payload = history.reset_index().to_json(date_format="iso", orient="records").encode("utf-8")
        identity = us_identity(symbol, str(getattr(ticker, "fast_info", {}).get("exchange", "US")) if False else "US")
        rows = []
        for stamp, item in history.iterrows():
            day = pd.Timestamp(stamp).tz_convert("America/New_York").date().isoformat() if getattr(stamp, "tzinfo", None) else pd.Timestamp(stamp).date().isoformat()
            rows.append({
                "instrument_id": identity["instrument_id"],
                "symbol": identity["symbol"],
                "market": "US",
                "trade_date": day,
                "open": _num(item.get("Open")),
                "high": _num(item.get("High")),
                "low": _num(item.get("Low")),
                "close": _num(item.get("Close")),
                "volume": _num(item.get("Volume")),
                "amount": pd.NA,
                "prev_close": pd.NA,
                "turnover": pd.NA,
                "source": "yahoo",
            })
        actions = []
        if "Dividends" in history.columns:
            for stamp, value in history["Dividends"].items():
                if value and float(value) > 0:
                    day = pd.Timestamp(stamp).date().isoformat()
                    actions.append({
                        "instrument_id": identity["instrument_id"],
                        "event_date": day,
                        "event_type": "dividend",
                        "split_ratio": pd.NA,
                        "cash_dividend": float(value),
                        "stock_dividend": pd.NA,
                        "rights_ratio": pd.NA,
                        "effective_date": day,
                        "announcement_date": pd.NA,
                        "source": "yahoo",
                    })
        return Capture("yahoo", "daily", payload, market_rows=rows, action_rows=actions, instrument=identity)
    except Exception as exc:  # noqa: BLE001
        return Capture("yahoo", "daily", b"", error=str(exc))


def _absorb_sec_accessions(accepted: dict, node: dict) -> None:
    recent = node.get("filings", {}).get("recent", node)
    for accn, moment in zip(recent.get("accessionNumber", []), recent.get("acceptanceDateTime", [])):
        accepted[str(accn)] = moment
        accepted[str(accn).replace("-", "")] = moment


def _sec_accepted_map(submissions: dict, headers: dict) -> dict:
    accepted: dict[str, str] = {}
    _absorb_sec_accessions(accepted, submissions)
    for item in submissions.get("filings", {}).get("files", []) or []:
        name = item.get("name")
        if not name:
            continue
        extra = json.loads(fetch_bytes(f"https://data.sec.gov/submissions/{name}", headers=headers).decode("utf-8"))
        _absorb_sec_accessions(accepted, extra)
    return accepted


def capture_sec(symbol: str, cik: str, concepts: set[str]) -> Capture:
    cik_text = str(cik).zfill(10)
    headers = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}
    try:
        facts_payload = fetch_bytes(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_text}.json", headers=headers)
        submissions_payload = fetch_bytes(f"https://data.sec.gov/submissions/CIK{cik_text}.json", headers=headers)
        submissions = json.loads(submissions_payload.decode("utf-8"))
        accepted = _sec_accepted_map(submissions, headers)
        facts = json.loads(facts_payload.decode("utf-8"))
        rows = parse_sec_companyfacts(facts, symbol, concepts, accepted)
        identity = us_identity(symbol, "SEC")
        identity["cik"] = cik_text
        payload = facts_payload + b"\n" + submissions_payload
        return Capture("sec", "companyfacts", payload, fundamental_rows=rows, instrument=identity)
    except Exception as exc:  # noqa: BLE001
        return Capture("sec", "companyfacts", b"", error=str(exc))


def _akshare_sina_rows(frame: pd.DataFrame, symbol: str) -> list[dict]:
    identity = a_share_identity(symbol)
    rows = []
    for _, item in frame.iterrows():
        turnover = _num(item.get("turnover"))
        rows.append({
            "instrument_id": identity["instrument_id"],
            "symbol": identity["symbol"],
            "market": "CN",
            "trade_date": str(item.get("date"))[:10],
            "open": _num(item.get("open")),
            "high": _num(item.get("high")),
            "low": _num(item.get("low")),
            "close": _num(item.get("close")),
            "prev_close": pd.NA,
            "volume": _num(item.get("volume")),
            "amount": _num(item.get("amount")),
            "turnover": float(turnover) * 100 if pd.notna(turnover) else pd.NA,
            "source": "akshare",
        })
    return rows


def capture_akshare(symbol: str, start: str, end: str) -> Capture:
    identity = a_share_identity(symbol)
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")
    errors = []
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        return Capture("akshare", "daily", b"", error=str(exc))
    try:
        frame = ak.stock_zh_a_hist(symbol=identity["akshare_code"], period="daily", start_date=start_compact, end_date=end_compact, adjust="")
        if frame is not None and not frame.empty:
            payload = frame.to_json(orient="records", force_ascii=False).encode("utf-8")
            return Capture("akshare", "daily", payload, market_rows=parse_akshare_hist(frame, symbol), instrument=identity)
        errors.append("eastmoney: empty")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"eastmoney: {exc}")
    try:
        frame = ak.stock_zh_a_daily(symbol=identity["tencent_code"], start_date=start_compact, end_date=end_compact, adjust="")
        if frame is not None and not frame.empty:
            records = json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))
            payload = json.dumps({"symbol": symbol, "vendor": "sina", "rows": records}, ensure_ascii=False).encode("utf-8")
            return Capture("akshare", "daily", payload, market_rows=_akshare_sina_rows(frame, symbol), instrument=identity)
        errors.append("sina: empty")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina: {exc}")
    return Capture("akshare", "daily", b"", error="; ".join(errors))


def capture_baostock(symbol: str, start: str, end: str) -> Capture:
    try:
        import baostock as bs
        identity = a_share_identity(symbol)
        login = bs.login()
        if getattr(login, "error_code", "1") != "0":
            return Capture("baostock", "daily", b"", error=getattr(login, "error_msg", "login failed"))
        try:
            result = bs.query_history_k_data_plus(
                identity["baostock_code"],
                "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus",
                start_date=start, end_date=end, frequency="d", adjustflag="3",
            )
            lines = [",".join(result.fields)]
            while result.error_code == "0" and result.next():
                lines.append(",".join(result.get_row_data()))
            text = "\n".join(lines)
            profit_lines = ["code,pubDate,statDate,concept,value"]
            statement_fns = (
                bs.query_profit_data,
                bs.query_growth_data,
                bs.query_balance_data,
                bs.query_cash_flow_data,
                bs.query_dupont_data,
            )
            for year in range(int(start[:4]), int(end[:4]) + 1):
                for quarter in (2, 4):
                    for fetch in statement_fns:
                        result_set = fetch(code=identity["baostock_code"], year=year, quarter=quarter)
                        while result_set.error_code == "0" and result_set.next():
                            mapped = dict(zip(result_set.fields, result_set.get_row_data()))
                            for concept, value in mapped.items():
                                if concept in {"code", "pubDate", "statDate"} or value in (None, ""):
                                    continue
                                profit_lines.append(",".join([
                                    mapped.get("code", identity["baostock_code"]),
                                    mapped.get("pubDate", ""),
                                    mapped.get("statDate", ""),
                                    concept,
                                    value,
                                ]))
            actions = _baostock_actions(bs, identity)
            basic = bs.query_stock_basic(code=identity["baostock_code"])
            if basic.error_code == "0" and basic.next():
                info = dict(zip(basic.fields, basic.get_row_data()))
                identity["name"] = info.get("code_name") or pd.NA
                identity["list_date"] = info.get("ipoDate") or pd.NA
                identity["delist_date"] = info.get("outDate") or pd.NA
                identity["status"] = "delisted" if info.get("outDate") else "listed"
            profit_text = "\n".join(profit_lines)
        finally:
            bs.logout()
        payload = "\n".join([
            text,
            "--profit--",
            profit_text,
            "--actions--",
            json.dumps(_json_rows(actions), ensure_ascii=False),
            "--instrument--",
            json.dumps(_json_rows([identity])[0], ensure_ascii=False),
        ]).encode("utf-8")
        return Capture(
            "baostock", "daily", payload,
            market_rows=parse_baostock_kline(text, symbol),
            fundamental_rows=parse_baostock_profit(profit_text, symbol),
            action_rows=actions,
            instrument=identity,
            actions_checked=True,
        )
    except Exception as exc:  # noqa: BLE001
        return Capture("baostock", "daily", b"", error=str(exc))


def _baostock_actions(bs, identity: dict, years: range | None = None) -> list[dict]:
    rows = []
    for year in years or range(2018, 2027):
        dividend = bs.query_dividend_data(code=identity["baostock_code"], year=str(year), yearType="report")
        while dividend.error_code == "0" and dividend.next():
            item = dict(zip(dividend.fields, dividend.get_row_data()))
            announced = item.get("dividPlanAnnounceDate") or item.get("dividPreNoticeDate") or ""
            if not item.get("dividOperateDate") and not announced:
                continue
            rows.append({
                "instrument_id": identity["instrument_id"],
                "event_date": item.get("dividOperateDate") or announced,
                "event_type": "dividend",
                "split_ratio": pd.NA,
                "cash_dividend": _num(item.get("dividCashPsBeforeTax")),
                "stock_dividend": _num(item.get("dividStocksPs")),
                "rights_ratio": pd.NA,
                "adjust_factor": pd.NA,
                "effective_date": item.get("dividOperateDate") or pd.NA,
                "announcement_date": announced or pd.NA,
                "source": "baostock",
            })
    factor = bs.query_adjust_factor(code=identity["baostock_code"], start_date="2000-01-01", end_date="2026-12-31")
    while factor.error_code == "0" and factor.next():
        item = dict(zip(factor.fields, factor.get_row_data()))
        rows.append({
            "instrument_id": identity["instrument_id"],
            "event_date": item.get("dividOperateDate"),
            "event_type": "adjust_factor",
            "split_ratio": pd.NA,
            "cash_dividend": pd.NA,
            "stock_dividend": pd.NA,
            "rights_ratio": pd.NA,
            "adjust_factor": _num(item.get("backAdjustFactor")),
            "effective_date": item.get("dividOperateDate"),
            "announcement_date": pd.NA,
            "source": "baostock",
        })
    return rows


def capture_hs300(dates: list[str]) -> Capture:
    """按历史日期取沪深300成分。成分日就是 as_of，不用今天的名单回填。"""
    try:
        import baostock as bs
        login = bs.login()
        if getattr(login, "error_code", "1") != "0":
            return Capture("baostock", "hs300", b"", error=getattr(login, "error_msg", "login failed"))
        lines = ["as_of,code,name"]
        universe_rows = []
        try:
            for day in dates:
                result = bs.query_hs300_stocks(date=day)
                if result.error_code != "0":
                    raise RuntimeError(result.error_msg)
                while result.error_code == "0" and result.next():
                    update_date, code, name = result.get_row_data()[:3]
                    identity = a_share_identity(code)
                    lines.append(",".join([day, code, name]))
                    universe_rows.append({
                        "instrument_id": identity["instrument_id"],
                        "universe": "hs300",
                        "effective_start": day,
                        "effective_end": day,
                        "as_of": update_date or day,
                        "pit_membership": True,
                        "source": "baostock",
                        "name": name,
                        "symbol": identity["symbol"],
                        "exchange": identity["exchange"],
                        "country": "CN",
                        "currency": "CNY",
                        "asset_type": "stock",
                    })
        finally:
            bs.logout()
        payload = "\n".join(lines).encode("utf-8")
        return Capture("baostock", "hs300", payload, universe_rows=universe_rows)
    except Exception as exc:  # noqa: BLE001
        return Capture("baostock", "hs300", b"", error=str(exc))
