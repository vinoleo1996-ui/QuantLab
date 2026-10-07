"""采集 → 青铜层 → 银层 → 质量/冲突 → 金层数据集。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from quantlab.data_os.bronze import BronzeStore
from quantlab.data_os.conflict import resolve_market
from quantlab.data_os.datasets import agent_dataset, alpha_dataset, ml_dataset
from quantlab.data_os.identity import alias_frame, master_frame
from quantlab.data_os.normalize import action_frame, apply_adjust_factors, fundamental_frame, macro_frame, market_frame
from quantlab.data_os.quality import quality_gate
from quantlab.data_os.schema import ALIAS_COLUMNS, INSTRUMENT_COLUMNS
from quantlab.data_os.silver import write_quality, write_table
from quantlab.data_os.sources import Capture


DEFAULT_FORMULAS = {
    "alpha_001": "rank(ts_mean(close/vwap,20))",
    "alpha_002": "rank(-corr(RET_1,volume,20))",
}


def build_dataset(root: str | Path, version: str, captures: list[Capture], as_of=None, formulas: dict | None = None) -> dict:
    root = Path(root)
    bronze = BronzeStore(root)
    errors = []
    market_rows, fund_cn, fund_us, action_rows, macro_rows = [], [], [], [], []
    instruments = []
    aliases = []
    covered_actions = set()
    pit_universe = []
    for capture in captures:
        if capture.payload:
            bronze.put(capture.source, capture.dataset, capture.payload, {"error": capture.error})
        if capture.error:
            errors.append({"source": capture.source, "dataset": capture.dataset, "error": capture.error})
        market_rows.extend(capture.market_rows)
        for row in capture.fundamental_rows:
            (fund_us if str(row.get("instrument_id", "")).startswith("US.") else fund_cn).append(row)
        action_rows.extend(capture.action_rows)
        macro_rows.extend(capture.macro_rows)
        pit_universe.extend(capture.universe_rows)
        if capture.actions_checked and capture.instrument:
            covered_actions.add(capture.instrument["instrument_id"])
        if capture.instrument:
            instruments.append(capture.instrument)
            aliases.append({
                "instrument_id": capture.instrument["instrument_id"],
                "symbol": capture.instrument["symbol"],
                "effective_start": capture.instrument.get("list_date"),
                "effective_end": capture.instrument.get("delist_date"),
                "source": capture.source,
            })
    for row in market_rows:
        row["corporate_action_valid"] = row["instrument_id"] in covered_actions
    market_rows = apply_adjust_factors(market_rows, action_rows)
    resolved, conflicts = resolve_market([pd.DataFrame(market_rows)] if market_rows else [])
    market = market_frame(resolved.to_dict("records"), version) if not resolved.empty else pd.DataFrame()
    fund_parts = []
    if fund_cn:
        fund_parts.append(fundamental_frame(fund_cn, version, "Asia/Shanghai"))
    if fund_us:
        fund_parts.append(fundamental_frame(fund_us, version, "America/New_York"))
    fundamentals = pd.concat(fund_parts, ignore_index=True) if fund_parts else pd.DataFrame()
    actions = action_frame(action_rows, version, "Asia/Shanghai") if action_rows else pd.DataFrame()
    macro = macro_frame(macro_rows, version) if macro_rows else pd.DataFrame()
    master = master_frame(instruments) if instruments else pd.DataFrame(columns=INSTRUMENT_COLUMNS)
    aliases_frame = alias_frame(aliases) if aliases else pd.DataFrame(columns=ALIAS_COLUMNS)
    seen = {row["instrument_id"] for row in instruments}
    for instrument_id in sorted(set(market["instrument_id"]) if not market.empty else []):
        if instrument_id in seen:
            continue
        country, exchange, symbol = instrument_id.split(".", 2)
        instruments.append({
            "instrument_id": instrument_id, "symbol": symbol, "exchange": exchange,
            "country": country, "asset_type": "stock", "currency": "CNY" if country == "CN" else "USD",
            "status": "listed",
        })
        seen.add(instrument_id)
    master = master_frame(instruments) if instruments else master

    universe_rows = []
    for row in pit_universe:
        item = dict(row)
        item["dataset_version"] = version
        item["pit_membership"] = True
        universe_rows.append(item)
        if item["instrument_id"] not in seen:
            instruments.append({
                "instrument_id": item["instrument_id"],
                "symbol": item.get("symbol") or item["instrument_id"].split(".")[-1],
                "exchange": item.get("exchange") or item["instrument_id"].split(".")[1],
                "country": item.get("country") or item["instrument_id"].split(".")[0],
                "asset_type": "stock",
                "currency": item.get("currency") or "CNY",
                "name": item.get("name"),
                "status": "listed",
            })
            seen.add(item["instrument_id"])
    if not market.empty:
        for instrument_id, group in market.groupby("instrument_id"):
            start = pd.to_datetime(group["trade_date"]).min().date().isoformat()
            universe_rows.append({
                "instrument_id": instrument_id,
                "universe": "ingested",
                "effective_start": start,
                "effective_end": pd.NA,
                "as_of": pd.NA,
                "pit_membership": False,
                "source": "data_os",
                "dataset_version": version,
            })
    master = master_frame(instruments) if instruments else master
    universe = pd.DataFrame(universe_rows)
    gate = quality_gate(market, universe)
    paths = {
        "instruments": write_table(root, "instruments", master, version),
        "market": write_table(root, "market", market, version),
        "corporate_actions": write_table(root, "corporate_actions", actions, version),
        "fundamentals": write_table(root, "fundamentals", fundamentals, version),
        "universe": write_table(root, "universe", universe, version),
        "macro": write_table(root, "macro", macro, version),
    }
    write_quality(root, "conflicts", conflicts if not conflicts.empty else pd.DataFrame(columns=["instrument_id", "trade_date", "field"]), version)
    write_quality(root, "lineage", bronze.lineage(), version)
    if not market.empty:
        grade = market.loc[~market["research_grade"].astype(bool), ["instrument_id", "trade_date", "source", "quality_score", "pit_valid"]]
    else:
        grade = pd.DataFrame()
    write_quality(root, "rejected_market", grade, version)

    learn = ml_dataset(root, version, market, fundamentals, macro) if not market.empty else pd.DataFrame()
    alpha_dataset(root, version, learn, formulas or DEFAULT_FORMULAS)
    moment = as_of or (market["available_time"].max() if not market.empty else datetime.now(timezone.utc))
    agent_dataset(root, version, market, fundamentals, macro, moment)
    meta = {
        "dataset_version": version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "market_rows": int(len(market)),
        "research_grade_rows": int(market["research_grade"].sum()) if not market.empty else 0,
        "fundamental_rows": int(len(fundamentals)),
        "fundamental_research_grade": int(fundamentals["research_grade"].sum()) if not fundamentals.empty else 0,
        "macro_rows": int(len(macro)),
        "macro_research_grade": int(macro["research_grade"].sum()) if not macro.empty else 0,
        "pit_membership_rows": int(universe["pit_membership"].astype(bool).sum()) if not universe.empty and "pit_membership" in universe.columns else 0,
        "quality_gate": gate,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    meta_path = root / "metadata" / f"dataset_version={version}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _duckdb(root, version, paths["market"])
    return meta


def _duckdb(root: Path, version: str, market_path: Path) -> None:
    import duckdb
    database = root / "metadata" / f"{version}.duckdb"
    connection = duckdb.connect(str(database))
    try:
        connection.execute("CREATE TABLE market AS SELECT * FROM read_parquet(?)", [str(market_path)])
    finally:
        connection.close()


def live_captures(a_symbol: str, us_symbol: str, cik: str, start: str, end: str, root: str | Path = "data/openquant") -> list[Capture]:
    """采集层。Stooq 不再作为无密钥免费源。AKShare 失败只记健康度，不阻断归档。"""
    import os
    from quantlab.data_os.archive import timed_capture
    from quantlab.data_os.sources import capture_akshare, capture_baostock, capture_fred, capture_hs300, capture_sec, capture_stooq, capture_tencent, capture_yahoo
    concepts = {"Revenues", "NetIncomeLoss", "EarningsPerShareDiluted"}
    root = Path(root)
    jobs = [
        ("baostock", "hs300", lambda: capture_hs300(["2018-01-02", "2024-01-02"])),
        ("baostock", "daily", lambda: capture_baostock(a_symbol, start, end)),
        ("akshare", "daily", lambda: capture_akshare(a_symbol, start, end)),
        ("tencent", "daily", lambda: capture_tencent(a_symbol, start, end)),
        ("yahoo", "daily", lambda: capture_yahoo(us_symbol, start, end)),
        ("sec", "companyfacts", lambda: capture_sec(us_symbol, cik, concepts)),
        ("fred", "DGS10", lambda: capture_fred("DGS10")),
        ("fred", "CPIAUCSL", lambda: capture_fred("CPIAUCSL")),
    ]
    if os.environ.get("STOOQ_API_KEY"):
        jobs.append(("stooq", "daily", lambda: capture_stooq(us_symbol)))
    return [timed_capture(root, source, dataset, function) for source, dataset, function in jobs]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="构建 OpenQuant 数据集")
    parser.add_argument("--root", default="data/openquant")
    parser.add_argument("--hs300-history", action="store_true", help="补齐两份沪深300成分的日线、分红和复权")
    parser.add_argument("--live", action="store_true", help="采集外网并写入原始归档")
    parser.add_argument("--replay", action="store_true", help="只从本地原始归档重建，不访问外网")
    parser.add_argument("--research", action="store_true", help="只读本地银层")
    parser.add_argument("--as-of", default="2024-01-31T00:00:00Z")
    parser.add_argument("--a-symbol", default="000001")
    parser.add_argument("--us-symbol", default="AAPL")
    parser.add_argument("--cik", default="0000320193")
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-02-29")
    args = parser.parse_args()
    if args.hs300_history:
        from quantlab.data_os.hs300_history import ingest
        print(json.dumps(ingest(args.root, args.start, args.end), ensure_ascii=False, indent=2))
        return
    if args.research:
        from quantlab.data_os.archive import research_view
        view = research_view(args.root, args.as_of)
        view.pop("market")
        view.pop("fundamentals")
        view.pop("macro")
        view.pop("universe")
        print(json.dumps(view, ensure_ascii=False, indent=2))
        return
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.live:
        captures = live_captures(args.a_symbol, args.us_symbol, args.cik, args.start, args.end, args.root)
    elif args.replay:
        from quantlab.data_os.archive import replay
        captures = replay(args.root)
    else:
        raise SystemExit("采集用 --live。从原始归档重建用 --replay。研究读取用 --research。")
    if not captures:
        raise SystemExit("没有可重建的原始归档。")
    meta = build_dataset(args.root, version, captures)
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
