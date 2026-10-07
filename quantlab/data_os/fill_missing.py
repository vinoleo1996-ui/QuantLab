"""补设计里缺的、而且免费源能核对公布日的数据。"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

import pandas as pd

from quantlab.data_os.bronze import BronzeStore
from quantlab.data_os.identity import a_share_identity
from quantlab.data_os.sources import fetch_bytes, parse_fred_csv

# ALFRED release id。名称不对就不用这份发布日。
ALFRED_RELEASES = {
    "CPIAUCSL": (10, "Consumer Price Index"),
    "UNRATE": (50, "Employment Situation"),
    "INDPRO": (13, "Industrial Production"),
    "GDP": (53, "Gross Domestic Product"),
    "FEDFUNDS": (18, "H.15 Selected Interest Rates"),
}
SAME_DAY = ("DGS2", "DGS10", "VIXCLS")
NEXT_DAY = ()
QUARTERLY = {"GDP"}


def alfred_dates(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) == 10 and line[4] == "-" and line[7] == "-" and line.replace("-", "").isdigit():
            found.append(line)
    return sorted(set(found))


def first_release_after(observation_date: str, dates: list[str], quarterly: bool = False) -> str | None:
    period_start = pd.Timestamp(observation_date)
    period_end = (period_start + pd.offsets.QuarterEnd(0)) if quarterly else (period_start + pd.offsets.MonthEnd(0))
    for day in dates:
        if pd.Timestamp(day) > period_end:
            return day
    return None


def _fred_table(series_id: str, dates: list[str] | None = None) -> tuple[bytes, list[dict]]:
    raw = fetch_bytes(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}").decode("utf-8")
    rows = parse_fred_csv(raw, series_id)
    quarterly = series_id in QUARTERLY
    lines = ["observation_date,value,release_date"]
    for row in rows:
        release = row.get("release_date")
        if series_id in SAME_DAY:
            release = row["observation_date"]
        elif series_id in NEXT_DAY:
            release = (pd.Timestamp(row["observation_date"]) + pd.Timedelta(days=1)).date().isoformat()
        elif dates:
            release = first_release_after(str(row["observation_date"])[:10], dates, quarterly)
        row["release_date"] = release
        lines.append(",".join([
            str(row["observation_date"])[:10],
            "" if row["value"] is None or (isinstance(row["value"], float) and pd.isna(row["value"])) else str(row["value"]),
            "" if not release else release,
        ]))
    return "\n".join(lines).encode("utf-8"), rows


def ingest_macro(root: str | Path) -> dict:
    root = Path(root)
    store = BronzeStore(root)
    stored = []
    for series_id, (release_id, title) in ALFRED_RELEASES.items():
        page = fetch_bytes(f"https://alfred.stlouisfed.org/release/downloaddates?rid={release_id}&ff=txt").decode("utf-8", "replace")
        if title not in page.splitlines()[1]:
            # 标题在正文前部，不在第二行时整页检索。
            if title not in page[:500]:
                stored.append({"series": series_id, "error": f"ALFRED title mismatch rid={release_id}"})
                continue
        dates = alfred_dates(page)
        payload, rows = _fred_table(series_id, dates)
        store.put("fred", series_id, payload, {"kind": "fred_with_release", "series": series_id})
        stored.append({"series": series_id, "rows": len(rows), "releases": len(dates)})
    for series_id in SAME_DAY + NEXT_DAY:
        payload, rows = _fred_table(series_id)
        store.put("fred", series_id, payload, {"kind": "fred_with_release", "series": series_id})
        stored.append({"series": series_id, "rows": len(rows)})
    return {"stored": stored}


def ingest_a_share_snapshots(root: str | Path, dates: list[str] | None = None) -> dict:
    """中证500、上证50、全市场名单和行业。行业是当前分类，不回填到历史时点。"""
    import baostock as bs
    root = Path(root)
    dates = dates or ["2018-01-02", "2024-01-02"]
    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        return {"error": getattr(login, "error_msg", "login failed")}
    try:
        universe_rows = []
        for day in dates:
            for universe, fetch in (
                ("zz500", lambda day=day: bs.query_zz500_stocks(date=day)),
                ("sz50", lambda day=day: bs.query_sz50_stocks(date=day)),
                ("all_a", lambda day=day: bs.query_all_stock(day=day)),
            ):
                result = fetch()
                if result.error_code != "0":
                    universe_rows.append({"universe": universe, "as_of": day, "error": result.error_msg})
                    continue
                while result.error_code == "0" and result.next():
                    data = result.get_row_data()
                    code = data[1] if universe != "all_a" else data[0]
                    name = data[2] if universe != "all_a" and len(data) > 2 else (data[2] if len(data) > 2 else "")
                    if universe == "all_a":
                        # code, tradeStatus, code_name
                        code, _, name = data[0], data[1], data[2] if len(data) > 2 else ""
                    identity = a_share_identity(code)
                    universe_rows.append({
                        "instrument_id": identity["instrument_id"],
                        "universe": universe,
                        "effective_start": day,
                        "effective_end": day,
                        "as_of": day,
                        "pit_membership": True,
                        "source": "baostock",
                        "name": name,
                        "symbol": identity["symbol"],
                        "exchange": identity["exchange"],
                        "country": "CN",
                    })
        industry = bs.query_stock_industry()
        industry_rows = []
        while industry.error_code == "0" and industry.next():
            mapped = dict(zip(industry.fields, industry.get_row_data()))
            if not mapped.get("code") or not mapped.get("industry"):
                continue
            identity = a_share_identity(mapped["code"])
            industry_rows.append({
                "instrument_id": identity["instrument_id"],
                "universe": "csrc_industry",
                "effective_start": mapped.get("updateDate"),
                "effective_end": mapped.get("updateDate"),
                "as_of": mapped.get("updateDate"),
                "pit_membership": False,
                "source": "baostock",
                "name": mapped.get("industry"),
                "symbol": identity["symbol"],
                "exchange": identity["exchange"],
                "country": "CN",
            })
        universe_rows.extend(industry_rows)
    finally:
        bs.logout()
    payload = json.dumps({"universe": universe_rows}, ensure_ascii=False).encode("utf-8")
    BronzeStore(root).put("baostock", "snapshots", payload, {"kind": "index_and_industry"})
    counts = {}
    for row in universe_rows:
        if "instrument_id" in row:
            counts.setdefault(row["universe"], 0)
            counts[row["universe"]] += 1
    return {"universe_rows": counts, "industry_rows": sum(1 for row in universe_rows if row.get("universe") == "csrc_industry")}
