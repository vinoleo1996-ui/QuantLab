"""本地归档是研究时的数据源。采集可以访问外网，研究读取不可以。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd

from quantlab.data_os.bronze import BronzeStore
from quantlab.data_os.identity import a_share_identity, us_identity
from quantlab.data_os.pit import visible
from quantlab.data_os.sources import (
    Capture,
    _akshare_sina_rows,
    parse_akshare_hist,
    parse_baostock_kline,
    parse_baostock_profit,
    parse_fred_csv,
    parse_sec_companyfacts,
    parse_tencent_kline,
)


FRESHNESS = {
    "market_daily": "T+1",
    "financial_statement": "event_driven",
    "macro": "release_driven",
    "news": "15m",
}

# Stooq 的 CSV 接口已不再是无密钥免费源，不能放进默认采集。
OPTIONAL_PROVIDERS = ("akshare", "stooq")
REQUIRED_INGEST = ("baostock", "tencent", "yahoo", "sec", "fred")


class ResearchOfflineError(RuntimeError):
    pass


def record_health(root: Path, source: str, dataset: str, ok: bool, seconds: float, error: str | None) -> None:
    path = Path(root) / "metadata" / "provider_health.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "source": source,
        "dataset": dataset,
        "ok": ok,
        "seconds": round(seconds, 3),
        "error": error,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def provider_health(root: Path) -> pd.DataFrame:
    path = Path(root) / "metadata" / "provider_health.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["source", "dataset", "ok", "seconds", "error", "recorded_at"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def timed_capture(root: Path, source: str, dataset: str, function) -> Capture:
    from quantlab.data_os.net import direct_network, should_retry_direct
    started = time.perf_counter()
    capture = function()
    if should_retry_direct(capture.error):
        with direct_network():
            capture = function()
    record_health(root, source, dataset, capture.error is None and bool(capture.payload), time.perf_counter() - started, capture.error)
    return capture


def latest_version(root: Path) -> str | None:
    files = sorted(Path(root).glob("metadata/dataset_version=*.json"))
    if not files:
        return None
    return files[-1].stem.split("=", 1)[1]


def resolve(root: Path, dataset: str = "market_daily") -> dict:
    version = latest_version(root)
    local = _table(root, version, "market") if version and dataset == "market_daily" else pd.DataFrame()
    if dataset == "macro":
        local = _table(root, version, "macro") if version else pd.DataFrame()
    if dataset == "financial_statement":
        local = _table(root, version, "fundamentals") if version else pd.DataFrame()
    present = local is not None and not local.empty
    fallback = {
        "market_daily": ["baostock", "tencent"],
        "financial_statement": ["sec", "baostock"],
        "macro": ["fred"],
    }.get(dataset, [])
    return {
        "primary_provider": "local_archive" if present else None,
        "fallback_providers": [] if present else fallback,
        "optional_providers": list(OPTIONAL_PROVIDERS),
        "cache_policy": "prefer_local",
        "freshness_requirement": FRESHNESS.get(dataset, "T+1"),
        "quality_threshold": 0.95,
        "ingest_required": not present,
        "dataset_version": version,
    }


def research_view(root: Path, as_of) -> dict:
    """只读本地银层。没有归档就停止，不访问外网。"""
    version = latest_version(root)
    if version is None:
        raise ResearchOfflineError("没有本地归档。研究运行不访问外网，需要先单独做采集。")
    market = visible(_table(root, version, "market"), as_of)
    fundamentals = visible(_table(root, version, "fundamentals"), as_of)
    macro = visible(_table(root, version, "macro"), as_of)
    universe = _table(root, version, "universe")
    if not universe.empty and "as_of" in universe.columns:
        as_of_time = pd.Timestamp(as_of)
        if as_of_time.tzinfo is None:
            as_of_time = as_of_time.tz_localize("UTC")
        snaps = pd.to_datetime(universe["as_of"], utc=True, errors="coerce")
        universe = universe.loc[snaps.isna() | (snaps <= as_of_time)].copy()
    return {
        "dataset_version": version,
        "as_of": str(as_of),
        "market_rows": int(len(market)),
        "fundamental_rows": int(len(fundamentals)),
        "macro_rows": int(len(macro)),
        "universe_rows": int(len(universe)),
        "market": market,
        "fundamentals": fundamentals,
        "macro": macro,
        "universe": universe,
    }


def replay(root: Path) -> list[Capture]:
    """从已保存的原始字节重放，不打开外网。"""
    store = BronzeStore(root)
    if not store.manifest_path.exists():
        return []
    captures = []
    seen = set()
    for line in store.manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        digest = record["sha256"]
        if digest in seen or not record.get("created", True):
            continue
        seen.add(digest)
        payload = (store.root / record["path"]).read_bytes()
        if payload != payload[:0] and __import__("hashlib").sha256(payload).hexdigest() != digest:
            raise ResearchOfflineError(f"原始字节和清单哈希不一致: {record['path']}")
        try:
            parsed = _replay_one(record["source"], record["dataset"], payload)
        except Exception as exc:  # noqa: BLE001
            parsed = Capture(record["source"], record["dataset"], payload, error=str(exc))
        if parsed is not None:
            captures.append(parsed)
    return captures


def _table(root: Path, version: str | None, name: str) -> pd.DataFrame:
    if not version:
        return pd.DataFrame()
    path = Path(root) / "silver" / name / f"dataset_version={version}" / "part.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _replay_one(source: str, dataset: str, payload: bytes) -> Capture | None:
    if source == "baostock" and dataset == "daily":
        return _replay_baostock_daily(payload)
    if source == "baostock" and dataset == "hs300":
        return _replay_hs300(payload)
    if source == "fred":
        text = payload.decode("utf-8")
        return Capture("fred", dataset, payload, macro_rows=parse_fred_csv(text, dataset))
    if source == "baostock" and dataset == "snapshots":
        return _replay_snapshots(payload)
    if source == "tencent":
        body = json.loads(payload.decode("utf-8"))
        symbol = next(iter((body.get("data") or {})), "sz000001")
        return Capture("tencent", "daily", payload, market_rows=parse_tencent_kline(body, symbol[-6:]))
    if source == "yahoo":
        return _replay_yahoo(payload)
    if source == "sec" and dataset == "companyfacts":
        return _replay_sec(payload)
    if source == "akshare":
        body = json.loads(payload.decode("utf-8"))
        if isinstance(body, dict) and body.get("vendor") == "sina":
            frame = pd.DataFrame(body["rows"])
            return Capture("akshare", "daily", payload, market_rows=_akshare_sina_rows(frame, body["symbol"]))
        rows = body
        frame = pd.DataFrame(rows)
        symbol = str(rows[0].get("股票代码", "000001")).zfill(6)
        return Capture("akshare", "daily", payload, market_rows=parse_akshare_hist(frame, symbol))
    return None


def _replay_baostock_daily(payload: bytes) -> Capture:
    text = payload.decode("utf-8")
    kline, rest = text.split("\n--profit--\n", 1) if "\n--profit--\n" in text else (text, "")
    profit, actions_text, instrument_text = rest, "", ""
    if "\n--actions--\n" in rest:
        profit, tail = rest.split("\n--actions--\n", 1)
        if "\n--instrument--\n" in tail:
            actions_text, instrument_text = tail.split("\n--instrument--\n", 1)
        else:
            actions_text = tail
    symbol = "000001"
    for line in kline.splitlines():
        parts = line.split(",")
        if parts and parts[0][:4].isdigit() and len(parts) > 1:
            symbol = parts[1].split(".")[-1]
            break
    actions = json.loads(actions_text) if actions_text.strip() else []
    instrument = json.loads(instrument_text) if instrument_text.strip() else a_share_identity(symbol)
    return Capture(
        "baostock", "daily", payload,
        market_rows=parse_baostock_kline(kline, symbol),
        fundamental_rows=parse_baostock_profit(profit, symbol) if profit.strip() else [],
        action_rows=actions,
        instrument=instrument,
        actions_checked="\n--actions--\n" in text,
    )


def _replay_yahoo(payload: bytes) -> Capture:
    rows = json.loads(payload.decode("utf-8"))
    identity = us_identity("AAPL")
    market_rows = []
    actions = []
    for item in rows:
        day = str(item.get("Date", ""))[:10]
        market_rows.append({
            "instrument_id": identity["instrument_id"],
            "symbol": identity["symbol"],
            "market": "US",
            "trade_date": day,
            "open": item.get("Open"),
            "high": item.get("High"),
            "low": item.get("Low"),
            "close": item.get("Close"),
            "volume": item.get("Volume"),
            "amount": None,
            "prev_close": None,
            "turnover": None,
            "source": "yahoo",
            "price_basis": "raw",
        })
        dividend = item.get("Dividends") or 0
        if dividend and float(dividend) > 0:
            actions.append({
                "instrument_id": identity["instrument_id"],
                "event_date": day,
                "event_type": "dividend",
                "cash_dividend": float(dividend),
                "effective_date": day,
                "source": "yahoo",
            })
    return Capture("yahoo", "daily", payload, market_rows=market_rows, action_rows=actions, instrument=identity)


def _replay_sec(payload: bytes) -> Capture:
    text = payload.decode("utf-8")
    decoder = json.JSONDecoder()
    facts, index = decoder.raw_decode(text)
    submissions, _ = decoder.raw_decode(text[index:].lstrip())
    accepted = {}
    recent = submissions.get("filings", {}).get("recent", {})
    for accn, moment in zip(recent.get("accessionNumber", []), recent.get("acceptanceDateTime", [])):
        accepted[str(accn)] = moment
        accepted[str(accn).replace("-", "")] = moment
    symbol = "AAPL"
    concepts = {"Revenues", "NetIncomeLoss", "EarningsPerShareDiluted"}
    rows = parse_sec_companyfacts(facts, symbol, concepts, accepted)
    identity = us_identity(symbol, "SEC")
    identity["cik"] = str(facts.get("cik", "")).zfill(10)
    return Capture("sec", "companyfacts", payload, fundamental_rows=rows, instrument=identity)


def _replay_snapshots(payload: bytes) -> Capture:
    body = json.loads(payload.decode("utf-8"))
    universe_rows = [row for row in body.get("universe", []) if row.get("instrument_id")]
    fundamental_rows = []
    for row in body.get("industry", []):
        if row.get("instrument_id") and row.get("value"):
            fundamental_rows.append(row)
    return Capture("baostock", "snapshots", payload, universe_rows=universe_rows, fundamental_rows=fundamental_rows)


def _replay_hs300(payload: bytes) -> Capture:
    rows = []
    for line in payload.decode("utf-8").splitlines()[1:]:
        day, code, name = line.split(",", 2)
        identity = a_share_identity(code)
        rows.append({
            "instrument_id": identity["instrument_id"],
            "universe": "hs300",
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
    return Capture("baostock", "hs300", payload, universe_rows=rows)
