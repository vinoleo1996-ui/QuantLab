"""Download the QuantDB 253-session U0 seed and build the pool only on the due day.

The API key comes from QUANTDB_API_KEY.  Partition bytes stay in the gitignored
ledger.  This does not score themes and does not run an account.
"""

from __future__ import annotations

import http.client
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantlab.data.quantdb_u0_seed import (  # noqa: E402
    SEED_SESSION_COUNT,
    QuantDBPartition,
    assemble_quantdb_u0_rows,
    build_quantdb_u0_snapshot,
    resolve_forward_anchor,
)
from quantlab.research_signal_ledger import ResearchSignalLedger  # noqa: E402


BASE = "https://www.quantdb.cn"
OUTPUT = ROOT / "output" / "quantdb_theme_graph_20260930"
LEDGER_ID = "quantdb-theme-graph-20260930"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(url: str, key: str | None) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        opener = urllib.request.build_opener(_NoRedirect)
        current = url
        headers = {"Accept": "application/json"}
        if key:
            headers["X-API-Key"] = key
        try:
            for _hop in range(4):
                request = urllib.request.Request(current, headers=headers)
                try:
                    with opener.open(request, timeout=120) as response:
                        return response.read()
                except urllib.error.HTTPError as exc:
                    if exc.code not in {301, 302, 303, 307, 308}:
                        raise SystemExit(f"quantdb_http_{exc.code}") from exc
                    location = exc.headers.get("Location")
                    if not location:
                        raise SystemExit("quantdb_redirect_missing")
                    current = urllib.parse.urljoin(current, location)
                    headers = {}
            raise SystemExit("quantdb_redirect_limit")
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
            last_error = exc
            import time
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"quantdb_connection_failed:{type(last_error).__name__}")


def _url(path: str, params: dict[str, str]) -> str:
    return BASE + path + "?" + urllib.parse.urlencode(params)


def _stored_trade_dates(ledger_path: Path) -> set[str]:
    if not ledger_path.exists():
        return set()
    import sqlite3
    database = sqlite3.connect(ledger_path)
    found = set()
    for (body_json,) in database.execute("SELECT body_json FROM events WHERE kind='receipt' AND status='accepted'"):
        body = json.loads(body_json)
        metadata = body.get("metadata") or {}
        if metadata.get("dataset") == "daily_unadjusted":
            trade_date = (metadata.get("request_parameters") or {}).get("trade_date")
            if isinstance(trade_date, str):
                found.add(trade_date)
    database.close()
    return found


def _instrument_id(symbol: str) -> str:
    number, separator, market = str(symbol).upper().partition(".")
    if separator != "." or len(number) != 6 or not number.isdigit() or market not in {"SH", "SZ", "BJ"}:
        raise ValueError("invalid_symbol")
    if number.startswith(("4", "8", "92", "43", "83", "87")):
        prefix = "bj."
    elif number.startswith(("5", "6", "9")):
        prefix = "sh."
    else:
        prefix = "sz."
    return prefix + number


def _build_snapshot(ledger: ResearchSignalLedger, window: list[str], open_days: list[str], anchor: pd.Timestamp) -> dict[str, object]:
    """Build U0 once the anchor session itself is inside the 253-day seed."""
    import sqlite3
    database = sqlite3.connect(OUTPUT / "ledger" / "ledger.sqlite3")
    blobs: dict[str, str] = {}
    stock_blob = None
    for event_id, body_json in database.execute(
        "SELECT event_id, body_json FROM events WHERE kind='receipt' AND status='accepted'"
    ):
        body = json.loads(body_json)
        metadata = body.get("metadata") or {}
        if metadata.get("dataset") == "daily_unadjusted":
            trade_date = (metadata.get("request_parameters") or {}).get("trade_date")
            if trade_date:
                blobs[str(trade_date)] = body["blob_path"]
        if metadata.get("dataset") == "stock_list" and stock_blob is None:
            stock_blob = body["blob_path"]
    database.close()
    if stock_blob is None or any(day.replace("-", "") not in blobs for day in window):
        return {"u0_built": False, "reason": "seed_blob_missing"}
    stock_rows = json.loads((OUTPUT / "ledger" / stock_blob).read_text())["data"]
    identity = pd.DataFrame(
        {
            "instrument_id": [_instrument_id(row["symbol"]) for row in stock_rows],
            "is_st": [bool(row.get("is_st")) for row in stock_rows],
            "is_delisted": [bool(row.get("is_delisted")) for row in stock_rows],
            "name": [str(row.get("name") or "") for row in stock_rows],
        }
    ).drop_duplicates("instrument_id").set_index("instrument_id")
    partitions = []
    manifest_rows = []
    for day in window:
        trade_date = day.replace("-", "")
        raw = (OUTPUT / "ledger" / blobs[trade_date]).read_bytes()
        frame = pd.read_parquet(io.BytesIO(raw))
        frame = pd.DataFrame(
            {
                "instrument_id": frame["symbol"].map(_instrument_id),
                "amount": frame["amount"],
                "close": frame["close"],
            }
        )
        partitions.append(QuantDBPartition(pd.Timestamp(day), frame, trade_date, ""))
        manifest_rows.append({"trade_date": trade_date, "byte_count": len(raw)})
    captured_at = pd.Timestamp.now(tz="Asia/Shanghai")
    manifest_raw = json.dumps({"partitions": manifest_rows}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    recorded = ledger.record_receipt(
        source_id="quantdb.u0-seed",
        source_family="quantdb",
        origin_id="quantdb:u0-seed:" + window[-1],
        canonical_url="https://www.quantdb.cn/api/v1/data/download",
        raw=manifest_raw,
        content_type="application/json",
        published_at=captured_at,
        first_captured_at=captured_at,
        available_at=captured_at,
        availability_basis="first_capture_of_quantdb_seed_manifest",
        metadata={"dataset": "u0_seed_manifest", "response_id": "u0-seed-" + window[-1].replace("-", ""), "request_parameters": {}},
    )
    evidence = ledger.export_receipt_evidence(recorded["event_id"])
    rows = assemble_quantdb_u0_rows(
        partitions=partitions,
        identity=identity,
        provenance={
            "source_receipt_id": evidence["canonical_receipt_id"],
            "source_receipt_hash": evidence["raw"]["sha256"],
            "source_published_at": captured_at,
            "source_first_captured_at": captured_at,
            "source_available_at": evidence["receipt"]["available_at"],
        },
    )
    snapshot = build_quantdb_u0_snapshot(
        ledger=ledger,
        rows=rows,
        session=anchor,
        decision_at=pd.Timestamp(anchor.date()).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=18),
        cadence_anchor=anchor,
        exchange_sessions=open_days,
    )
    return {
        "u0_built": snapshot.status == "ok",
        "u0_status": snapshot.status,
        "u0_reason": snapshot.abstain_reason,
        "u0_member_count": len(snapshot.members),
        "u0_membership_hash": snapshot.membership_hash,
    }


def main() -> None:
    key = os.environ.get("QUANTDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("QUANTDB_API_KEY_required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calendar_raw = _request(_url("/api/v1/data/calendar", {"start_date": "2016-01-01", "end_date": "2026-12-31"}), key)
    calendar = json.loads(calendar_raw)
    open_days = []
    for row in calendar.get("data") or []:
        if int(row.get("is_open", 1)) != 1:
            continue
        text = str(row.get("trade_date") or "")[:10]
        if len(text) == 10:
            open_days.append(text)
    anchor = resolve_forward_anchor(open_days)
    manifest = json.loads(_request(_url(
        "/api/v1/data/download/manifest",
        {"category_id": "1", "sub_category": "daily_unadjusted", "layout": "v2"},
    ), key))
    by_date: dict[str, str] = {}
    for item in manifest.get("files") or []:
        path = str(item.get("relative_path") or "")
        if "dt=" not in path:
            continue
        trade_date = path.split("dt=", 1)[1][:8]
        if trade_date.isdigit() and len(trade_date) == 8:
            by_date[trade_date] = str(item["key"])
    available = [day for day in open_days if day.replace("-", "") in by_date]
    target = anchor.strftime("%Y-%m-%d") if anchor is not None else None
    if target in available:
        end = available.index(target)
        window = available[end - SEED_SESSION_COUNT + 1 : end + 1]
    else:
        window = available[-SEED_SESSION_COUNT + 1 :]
    ledger = ResearchSignalLedger(OUTPUT / "ledger", ledger_id=LEDGER_ID)
    stored = _stored_trade_dates(OUTPUT / "ledger" / "ledger.sqlite3")
    missing = [day for day in window if day.replace("-", "") not in stored]
    fetched_at = pd.Timestamp.now(tz="Asia/Shanghai")

    def _download(day: str) -> tuple[str, bytes]:
        trade_date = day.replace("-", "")
        body = _request(_url("/api/v1/data/download", {
            "category_id": "1",
            "sub_category": "daily_unadjusted",
            "layout": "v2",
            "object_key": by_date[trade_date],
        }), key)
        return trade_date, body

    downloaded = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_download, day) for day in missing]
        for future in as_completed(futures):
            trade_date, body = future.result()
            ledger.record_receipt(
                source_id="quantdb.u0-seed",
                source_family="quantdb",
                origin_id="quantdb:daily-seed:" + trade_date,
                canonical_url=_url("/api/v1/data/download", {
                    "category_id": "1",
                    "sub_category": "daily_unadjusted",
                    "layout": "v2",
                    "trade_date": trade_date,
                }),
                raw=body,
                content_type="application/vnd.apache.parquet",
                published_at=fetched_at,
                first_captured_at=fetched_at,
                available_at=fetched_at,
                availability_basis="first_capture_of_quantdb_response_no_vendor_publish_time",
                metadata={
                    "dataset": "daily_unadjusted",
                    "response_id": "daily-seed-" + trade_date,
                    "request_parameters": {
                        "category_id": "1",
                        "sub_category": "daily_unadjusted",
                        "trade_date": trade_date,
                    },
                },
            )
            downloaded += 1
    build = {"u0_built": False, "reason": "anchor_session_not_in_vendor_calendar"}
    if anchor is not None and len(window) == SEED_SESSION_COUNT and window[-1] == anchor.strftime("%Y-%m-%d"):
        build = _build_snapshot(ledger, window, open_days, anchor)
    summary = {
        "anchor_rule": "first_open_session_on_or_after_2026-09-30",
        "anchor_session": None if anchor is None else str(anchor.date()),
        "vendor_last_open_session": open_days[-1] if open_days else None,
        "seed_window_first": window[0] if window else None,
        "seed_window_last": window[-1] if window else None,
        "seed_window_count": len(window),
        "downloaded_partitions": downloaded,
        "promotion_approved": False,
        "not_an_account_result": True,
        **build,
    }
    (OUTPUT / "u0_seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
