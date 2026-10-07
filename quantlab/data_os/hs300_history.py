"""补齐两份沪深300成分的日线、分红和复权。已写入原始归档的股票会跳过。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantlab.data_os.archive import latest_version
from quantlab.data_os.bronze import BronzeStore
from quantlab.data_os.identity import a_share_identity
from quantlab.data_os.net import direct_network, should_retry_direct
from quantlab.data_os.sources import capture_baostock


def member_ids(root: Path) -> list[str]:
    version = latest_version(root)
    if not version:
        raise FileNotFoundError("没有本地成分归档")
    frame = pd.read_parquet(root / "silver" / "universe" / f"dataset_version={version}" / "part.parquet")
    members = frame[frame["pit_membership"].astype(bool) & (frame["universe"] == "hs300")]
    return sorted(set(members["instrument_id"]))


def _identity(instrument_id: str) -> dict:
    country, exchange, code = instrument_id.split(".", 2)
    prefix = {"XSHG": "sh", "XSHE": "sz", "BJSE": "bj"}[exchange]
    identity = a_share_identity(f"{prefix}{code}")
    if identity["instrument_id"] != instrument_id or country != "CN":
        raise ValueError(instrument_id)
    return identity


def _done(root: Path, symbol: str, start: str, end: str) -> bool:
    path = root / "metadata" / "bronze_manifest.jsonl"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        meta = record.get("meta") or {}
        if meta.get("kind") == "hs300_history" and meta.get("symbol") == symbol and meta.get("start") == start and meta.get("end") == end and record.get("created"):
            return True
    return False


def ingest(root: str | Path, start: str = "2018-01-02", end: str = "2024-01-31") -> dict:
    root = Path(root)
    store = BronzeStore(root)
    identities = [_identity(item) for item in member_ids(root)]
    ok = skipped = failed = 0
    failures = []
    for index, identity in enumerate(identities, start=1):
        symbol = identity["symbol"]
        if _done(root, symbol, start, end):
            skipped += 1
            continue
        capture = capture_baostock(symbol, start, end)
        if should_retry_direct(capture.error):
            with direct_network():
                capture = capture_baostock(symbol, start, end)
        if capture.error or not capture.payload:
            failed += 1
            failures.append({"symbol": symbol, "error": capture.error})
            print(f"{index}/{len(identities)} {symbol} FAIL {capture.error}", flush=True)
            continue
        store.put("baostock", "daily", capture.payload, {"kind": "hs300_history", "symbol": symbol, "start": start, "end": end})
        ok += 1
        if ok % 20 == 0 or index == len(identities):
            print(f"{index}/{len(identities)} stored {ok} skipped {skipped} failed {failed}", flush=True)
    return {"members": len(identities), "stored": ok, "skipped": skipped, "failed": failed, "failures": failures[:30]}
