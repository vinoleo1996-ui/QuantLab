"""Read-only A-share search and industry-board context for the observation terminal.

The interaction follows a trading desk: type a name, see the narrow industry
board, then the board's own leaders.  Membership comes from Eastmoney's
published industry-board list.  The 20-day move is each member's own last 21
daily closes from Tencent.  Nothing here reads the paper ledger, ranks the 50
holdings as if they were a sector, or changes a research result.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from quantlab.portfolio.market_observatory import top_month_returns

_EASTMONEY_UT = "bd1d9ddb04089700cf9c27f6f7426281"
_SUGGEST = "https://searchapi.eastmoney.com/api/suggest/get"
_SLIST = "https://push2.eastmoney.com/api/qt/slist/get"
_MEMBERS = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_KLINE = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newkline/newkline"
_MAX_PRICED_MEMBERS = 80
_MIN_MEMBERS = 3
_WINDOW = 20

_REASON_TEXT = {
    "industry_constituents_missing": "没有读到该板块的成分。",
    "industry_constituents_invalid": "板块成分有重复或无效代码，不发布排名。",
    "industry_constituents_incomplete": "板块成分不完整，不发布排名。",
    "industry_constituents_insufficient": "板块成分不足 3 只，不发布 Top 3。",
    "month_return_coverage_incomplete": "有成分缺少近 20 个交易日收盘，不发布 Top 3。",
}


def _get(url: str, referer: str, timeout: float = 12) -> bytes:
    request = Request(url, headers={"Referer": referer, "User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed public hosts.
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _json_bytes(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", "replace")
    start = text.find("{")
    if start < 0:
        return {}
    payload = json.loads(text[start:])
    return payload if isinstance(payload, dict) else {}


class DeskClient:
    """Small public-source client.  Tests substitute this object."""

    def suggest(self, query: str) -> dict[str, Any]:
        url = _SUGGEST + "?" + urlencode({
            "input": query, "type": "14", "token": "D43BF722C8E33BDC906FB84D85E326E8", "count": "8",
        })
        return _json_bytes(_get(url, "https://quote.eastmoney.com/"))

    def boards(self, secid: str) -> dict[str, Any]:
        url = _SLIST + "?" + urlencode({
            "secid": secid, "spt": "3", "pi": "0", "pz": "30", "po": "1", "np": "1",
            "ut": _EASTMONEY_UT, "fltt": "2", "invt": "2",
            "fields": "f3,f12,f14,f19,f128,f140",
        })
        return _json_bytes(_get(url, "https://quote.eastmoney.com/"))

    def members(self, board_code: str, page: int, page_size: int) -> dict[str, Any]:
        url = _MEMBERS + "?" + urlencode({
            "reportName": "RPT_F10_CORETHEME_BOARDTYPE",
            "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,BOARD_NAME,BOARD_CODE",
            "pageSize": str(page_size),
            "pageNumber": str(page),
            "filter": f'(BOARD_CODE="{board_code}")',
        })
        return _json_bytes(_get(url, "https://data.eastmoney.com/"))

    def daily_closes(self, code: str) -> list[tuple[str, float]]:
        symbol = _tencent_symbol(code)
        url = _KLINE + "?" + urlencode({"param": f"{symbol},day,,,40,"})
        payload = _json_bytes(_get(url, "https://gu.qq.com/"))
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        node = data.get(symbol) if isinstance(data.get(symbol), dict) else {}
        bars = node.get("day") if isinstance(node.get("day"), list) else []
        closes: list[tuple[str, float]] = []
        for bar in bars:
            if not isinstance(bar, (list, tuple)) or len(bar) < 3:
                continue
            try:
                closes.append((str(bar[0])[:10], float(bar[2])))
            except (TypeError, ValueError):
                continue
        return closes


def _tencent_symbol(code: str) -> str:
    if code.startswith(("4", "8", "92")):
        return f"bj{code}"
    return f"sh{code}" if code.startswith(("5", "6", "9")) else f"sz{code}"


def _secids(code: str) -> list[str]:
    if code.startswith(("4", "8", "92")):
        return [f"0.{code}"]
    primary = f"1.{code}" if code.startswith(("5", "6", "9")) else f"0.{code}"
    secondary = f"0.{code}" if primary.startswith("1.") else f"1.{code}"
    return [primary, secondary]


def _public_code(value: object) -> str | None:
    text = str(value or "").strip()
    digits = "".join(character for character in text if character.isdigit())
    return digits[-6:] if len(digits) >= 6 else None


def parse_suggest(payload: object) -> list[dict[str, str]]:
    """Keep A-share hits only.  Funds, indices and bonds are not quote targets here."""

    table = payload.get("QuotationCodeTable") if isinstance(payload, dict) else None
    rows = table.get("Data") if isinstance(table, dict) else None
    if not isinstance(rows, list):
        return []
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _public_code(row.get("Code"))
        classify = str(row.get("Classify") or "")
        kind = str(row.get("SecurityTypeName") or "")
        if code is None or code in seen or classify != "AStock":
            continue
        if kind and "A" not in kind and "板" not in kind:
            continue
        seen.add(code)
        found.append({
            "code": code,
            "name": str(row.get("Name") or code),
            "pinyin": str(row.get("PinYin") or ""),
            "market": kind or "A股",
        })
    return found[:8]


def _percent_points(value: object) -> float | None:
    """Eastmoney board percent arrives as points, so -0.52 means -0.52%."""

    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number / 100.0


def _leader_code(value: object) -> str | None:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    if not digits or len(digits) > 6:
        return None
    return digits.zfill(6)


def parse_industry_boards(payload: object) -> list[dict[str, Any]]:
    """Eastmoney marks an industry board with f19 == 2.  Concepts and regions stay out."""

    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("diff") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    boards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            board_kind = int(row.get("f19"))
        except (TypeError, ValueError):
            continue
        if board_kind != 2:
            continue
        raw = str(row.get("f12") or "").strip().upper()
        numeric = raw[2:] if raw.startswith("BK") else raw
        if not numeric.isdigit() or numeric in seen:
            continue
        seen.add(numeric)
        leader_code = _leader_code(row.get("f140"))
        leader_name = str(row.get("f128") or "").strip()
        boards.append({
            "board_code": f"BK{numeric}",
            "numeric_code": numeric,
            "name": str(row.get("f14") or f"BK{numeric}"),
            "today_pct": _percent_points(row.get("f3")),
            "today_leader_code": leader_code,
            "today_leader_name": leader_name or None,
        })
    return boards


def parse_members(payload: object) -> tuple[list[dict[str, str]], int | None]:
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        return [], None
    try:
        count = int(result.get("count"))
    except (TypeError, ValueError):
        count = None
    rows = result.get("data") if isinstance(result.get("data"), list) else []
    members: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _public_code(row.get("SECURITY_CODE"))
        if code is None:
            continue
        members.append({"code": code, "name": str(row.get("SECURITY_NAME_ABBR") or code)})
    return members, count


def twenty_day_return(closes: list[tuple[str, float]]) -> tuple[float, str] | None:
    """Last close over the close 20 sessions earlier.  Requires 21 real prints."""

    usable = [(day, price) for day, price in closes if price > 0 and len(day) >= 10]
    if len(usable) < _WINDOW + 1:
        return None
    last_day, last_price = usable[-1]
    base_price = usable[-(_WINDOW + 1)][1]
    if base_price <= 0:
        return None
    return last_price / base_price - 1.0, last_day


def session_return(closes: list[tuple[str, float]]) -> float | None:
    """Latest close against the previous close.  This is the board's follow-through, not a limit-up count."""

    usable = [(day, price) for day, price in closes if price > 0 and len(day) >= 10]
    if len(usable) < 2 or usable[-2][1] <= 0:
        return None
    return usable[-1][1] / usable[-2][1] - 1.0


def _board_snapshot(board: dict[str, Any]) -> dict[str, Any]:
    leader = None
    if board.get("today_leader_code") and board.get("today_leader_name"):
        leader = {"code": board["today_leader_code"], "name": board["today_leader_name"]}
    return {
        "name": board["name"],
        "board_code": board["board_code"],
        "today_pct": board.get("today_pct"),
        "today_leader": leader,
    }


def search_a_shares(query: str, client: DeskClient | None = None) -> list[dict[str, str]]:
    text = " ".join(str(query or "").split())
    if not text or len(text) > 32:
        return []
    source = client or DeskClient()
    try:
        payload = source.suggest(text)
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return []
    return parse_suggest(payload)


def _member_count(client: DeskClient, numeric_code: str) -> int | None:
    try:
        payload = client.members(numeric_code, 1, 1)
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    _rows, count = parse_members(payload)
    return count


def _all_members(client: DeskClient, numeric_code: str, count: int) -> list[dict[str, str]] | None:
    page_size = 50
    collected: list[dict[str, str]] = []
    page = 1
    while len(collected) < count and page <= 3:
        try:
            payload = client.members(numeric_code, page, page_size)
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
            return None
        rows, reported = parse_members(payload)
        if reported is not None and reported != count:
            return None
        if not rows:
            break
        collected.extend(rows)
        page += 1
    if len(collected) != count:
        return None
    return collected


def _closes_for(client: DeskClient, code: str) -> tuple[str, list[tuple[str, float]]]:
    try:
        return code, client.daily_closes(code)
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return code, []


def _unavailable(code: str, message: str, sector: dict[str, Any] | None = None,
                 context: dict[str, Any] | None = None) -> dict[str, Any]:
    ranking = {
        "available": False,
        "status": "unavailable",
        "metric": "20_trading_day_return",
        "window_trading_days": _WINDOW,
        "top": [],
        "source": "eastmoney_industry_board",
        "message": message,
        "sector_name": (sector or {}).get("name"),
    }
    return {
        "available": False,
        "code": code,
        "sector": sector,
        "context": context,
        "industry_top3_1m": ranking,
    }


def _follow_context(code: str, boards: list[dict[str, Any]], focus: dict[str, Any],
                    members: list[dict[str, str]] | None,
                    closes: dict[str, list[tuple[str, float]]],
                    returns: dict[str, float]) -> dict[str, Any]:
    """Where this stock sits in its industry chain, and whether the board is following.

    Breadth is the share of members up on the last close.  It is not a limit-up
    ladder and it is not published unless every member has that close.
    """

    chain = [_board_snapshot(board) for board in boards if board["board_code"] != focus["board_code"]]
    context: dict[str, Any] = {"focus": _board_snapshot(focus), "chain": chain}
    if not members:
        return context
    sessions = {item["code"]: session_return(closes.get(item["code"], [])) for item in members}
    if any(value is None for value in sessions.values()):
        return context
    values = {key: value for key, value in sessions.items() if value is not None}
    up = sum(value > 0 for value in values.values())
    down = sum(value < 0 for value in values.values())
    context["breadth"] = {
        "up": up,
        "down": down,
        "flat": len(values) - up - down,
        "total": len(values),
    }
    context["equal_weight_today"] = sum(values.values()) / len(values)
    if code in values:
        context["selected_today"] = values[code]
        context["versus_board"] = values[code] - context["equal_weight_today"]
    if code in returns and len(returns) == len(members):
        ordered = sorted(returns, key=lambda item: (-returns[item], item))
        context["selected_rank_20d"] = ordered.index(code) + 1
        context["selected_count"] = len(ordered)
    return context


def observe_public_industry(code: str, client: DeskClient | None = None) -> dict[str, Any]:
    """Resolve the narrowest priced industry board and rank its full membership."""

    source = client or DeskClient()
    boards: list[dict[str, Any]] = []
    for secid in _secids(code):
        try:
            boards = parse_industry_boards(source.boards(secid))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
            boards = []
        if boards:
            break
    if not boards:
        return _unavailable(code, "没有读到东财行业板块；不会用持仓名单或证监会门类代替。")

    sized: list[tuple[int, dict[str, Any]]] = []
    for board in boards:
        count = _member_count(source, board["numeric_code"])
        if count is not None:
            sized.append((count, board))
    eligible = [(count, board) for count, board in sized if count >= _MIN_MEMBERS]
    if not eligible:
        name = boards[0]["name"]
        sector = {"name": name, "board_code": boards[0]["board_code"], "source": "eastmoney_industry_board",
                  "constituent_count": sized[0][0] if sized else 0}
        return _unavailable(code, f"行业板块「{name}」成分不足，不发布 Top 3。", sector,
                            _follow_context(code, boards, boards[0], None, {}, {}))

    count, board = min(eligible, key=lambda item: (item[0], item[1]["board_code"]))
    sector = {
        "name": board["name"],
        "board_code": board["board_code"],
        "source": "eastmoney_industry_board",
        "constituent_count": count,
    }
    if count > _MAX_PRICED_MEMBERS:
        return _unavailable(
            code,
            f"东财行业板块「{board['name']}」有 {count} 只成分。一次点选不拉取超过 {_MAX_PRICED_MEMBERS} 只的行情，所以不发布 Top 3。",
            sector,
            _follow_context(code, boards, board, None, {}, {}),
        )
    members = _all_members(source, board["numeric_code"], count)
    if not members:
        return _unavailable(code, f"行业板块「{board['name']}」的成分列表不完整，不发布 Top 3。", sector,
                            _follow_context(code, boards, board, None, {}, {}))

    returns: dict[str, float] = {}
    closes_by_code: dict[str, list[tuple[str, float]]] = {}
    as_of = None
    pending = list(members)
    # A parallel burst sometimes drops one Tencent response.  Retry the misses
    # once, one at a time, before treating the board as incomplete.
    for attempt in (1, 2):
        if not pending:
            break
        workers = 1 if attempt == 2 else min(4, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            priced = list(pool.map(lambda item: _closes_for(source, item["code"]), pending))
        pending = []
        for member_code, member_closes in priced:
            closes_by_code[member_code] = member_closes
            parsed = twenty_day_return(member_closes)
            if parsed is None:
                pending.append(next(item for item in members if item["code"] == member_code))
                continue
            value, day = parsed
            returns[member_code] = value
            if as_of is None or day > as_of:
                as_of = day
    context = _follow_context(code, boards, board, members, closes_by_code, returns)
    ranking = top_month_returns(
        [{"code": item["code"], "name": item["name"]} for item in members],
        [{"code": key, "return_20d": value} for key, value in returns.items()],
        constituents_complete=True,
    )
    if not ranking.get("available"):
        reason = str(ranking.get("reason") or "")
        ranking["message"] = _REASON_TEXT.get(reason, "板块近 20 日排名暂不可用。")
        ranking["sector_name"] = board["name"]
        ranking["source"] = "eastmoney_industry_board"
        return {"available": False, "code": code, "sector": sector, "context": context, "industry_top3_1m": ranking}
    ranking["message"] = (
        f"东财行业板块「{board['name']}」，{count} 只成分全部计入近 20 个交易日涨幅。"
    )
    ranking["sector_name"] = board["name"]
    ranking["source"] = "eastmoney_board_members+tencent_daily_close"
    ranking["as_of"] = as_of
    return {"available": True, "code": code, "sector": sector, "context": context, "industry_top3_1m": ranking}


def cached_observe(code: str, clock: Callable[[], float] | None = None) -> dict[str, Any]:
    """Keep one successful board read briefly, so selecting the same name does not refetch."""

    import time

    now = (clock or time.monotonic)()
    cached = _OBSERVE_CACHE.get(code)
    if cached and now - cached[0] < 90 and cached[1].get("sector"):
        return cached[1]
    observed = observe_public_industry(code)
    if observed.get("sector"):
        _OBSERVE_CACHE[code] = (now, observed)
    return observed


_OBSERVE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
