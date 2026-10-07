"""Capture one QuantDB concept-graph slice and try the frozen graph scorer.

The API key is read from QUANTDB_API_KEY.  Raw responses stay in the gitignored
output ledger.  This script does not build U0, does not train, and does not
run an account.  A score is possible only when the ledger write is at or
before that session's 18:00 cutoff and a real U0 snapshot is supplied.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantlab.data.quantdb_theme_market_adapter import (  # noqa: E402
    QuantDBRawResponse,
    capture_quantdb_theme_graph_v1,
)
from quantlab.research_signal_ledger import ResearchSignalLedger  # noqa: E402


BASE = "https://www.quantdb.cn"
OUTPUT = ROOT / "output" / "quantdb_theme_graph_20260930"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(url: str, key: str | None) -> tuple[bytes, str]:
    opener = urllib.request.build_opener(_NoRedirect)
    current = url
    headers = {"Accept": "application/json"}
    if key:
        headers["X-API-Key"] = key
    for _hop in range(4):
        request = urllib.request.Request(current, headers=headers)
        try:
            with opener.open(request, timeout=120) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                return body, content_type
        except urllib.error.HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                detail = exc.read()[:180]
                raise SystemExit(f"quantdb_http_{exc.code}:{detail!r}") from exc
            location = exc.headers.get("Location")
            if not location:
                raise SystemExit("quantdb_redirect_missing")
            current = urllib.parse.urljoin(current, location)
            headers = {}
    raise SystemExit("quantdb_redirect_limit")


def _json_url(path: str, params: dict[str, str]) -> str:
    return BASE + path + "?" + urllib.parse.urlencode(params)


def _parse_daily_manifest(raw: bytes) -> list[dict[str, object]]:
    """Return dated V2 partitions without persisting the raw manifest."""
    try:
        document = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SystemExit("quantdb_daily_manifest_invalid") from exc
    files = document.get("files") if isinstance(document, dict) else None
    if not isinstance(files, list):
        raise SystemExit("quantdb_daily_manifest_files_missing")
    by_date: dict[str, dict[str, object]] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("relative_path") or "")
        marker = "dt="
        if marker not in path:
            continue
        trade_date = path.split(marker, 1)[1][:8]
        object_key = item.get("key")
        if len(trade_date) != 8 or not trade_date.isdigit() or not isinstance(object_key, str) or not object_key:
            continue
        size = item.get("size", item.get("size_bytes"))
        by_date[trade_date] = {"trade_date": trade_date, "object_key": object_key, "size": size}
    return [by_date[day] for day in sorted(by_date)]


def _runtime_key(*, key_stdin: bool) -> str:
    key = getpass.getpass("QuantDB API Key: ").strip() if key_stdin else os.environ.get("QUANTDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("QUANTDB_API_KEY_required")
    return key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key-stdin",
        action="store_true",
        help="read the key once from a no-echo terminal prompt instead of an environment variable",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="only calculate the latest 253-session download size; do not write a ledger or download a partition",
    )
    args = parser.parse_args(argv)
    key = _runtime_key(key_stdin=args.key_stdin)
    manifest_url = _json_url(
        "/api/v1/data/download/manifest",
        {"category_id": "1", "sub_category": "daily_unadjusted", "layout": "v2"},
    )
    manifest_raw, _ = _request(manifest_url, key)
    partitions = _parse_daily_manifest(manifest_raw)
    if len(partitions) < 253:
        raise SystemExit("quantdb_daily_manifest_under_253_sessions")
    latest_253 = partitions[-253:]
    if args.manifest_only:
        sizes = [item["size"] for item in latest_253]
        known_sizes = all(isinstance(size, int) and size >= 0 for size in sizes)
        print(json.dumps({
            "status": "preflight_only",
            "dataset": "daily_unadjusted",
            "session_count": len(latest_253),
            "first_session": latest_253[0]["trade_date"],
            "last_session": latest_253[-1]["trade_date"],
            "total_bytes": sum(sizes) if known_sizes else None,
            "all_partition_sizes_available": known_sizes,
            "raw_or_ledger_written": False,
            "promotion_approved": False,
        }, ensure_ascii=False, sort_keys=True))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calendar_url = _json_url("/api/v1/data/calendar", {"start_date": "2016-01-01", "end_date": "2026-12-31"})
    stock_url = _json_url("/api/v1/data/stock-list", {"limit": "20000"})
    concept_url = _json_url(
        "/api/v1/data/preview",
        {"category_id": "2", "sub_category": "sector_concept", "limit": "200000"},
    )
    calendar_raw, _ = _request(calendar_url, key)
    stock_raw, _ = _request(stock_url, key)
    concept_raw, _ = _request(concept_url, key)
    calendar = json.loads(calendar_raw)
    open_days = [
        str(row["trade_date"])[:10].replace("-", "")
        for row in calendar.get("data") or []
        if int(row.get("is_open", 1)) == 1
    ]
    by_date = {str(item["trade_date"]): str(item["object_key"]) for item in partitions}
    available = [day for day in open_days if day in by_date]
    if len(available) < 3:
        raise SystemExit("quantdb_daily_partitions_incomplete")
    reference, prior, decision = available[-3:]
    daily_payloads = []
    for trade_date in (reference, prior, decision):
        download_url = _json_url(
            "/api/v1/data/download",
            {
                "category_id": "1",
                "sub_category": "daily_unadjusted",
                "layout": "v2",
                "object_key": by_date[trade_date],
            },
        )
        body, _content_type = _request(download_url, key)
        daily_payloads.append((trade_date, by_date[trade_date], body, download_url))
    fetched_at = pd.Timestamp.now(tz="Asia/Shanghai")
    responses = [
        QuantDBRawResponse(
            "daily_manifest", "daily-manifest", {"category_id": "1", "sub_category": "daily_unadjusted", "layout": "v2"},
            manifest_url, manifest_raw, "application/json", fetched_at, fetched_at, fetched_at,
            "first_capture_of_quantdb_response_no_vendor_publish_time",
        ),
        QuantDBRawResponse(
            "trade_calendar", "trade-calendar", {"start_date": "2016-01-01", "end_date": "2026-12-31"},
            calendar_url, calendar_raw, "application/json", fetched_at, fetched_at, fetched_at,
            "first_capture_of_quantdb_response_no_vendor_publish_time",
        ),
        QuantDBRawResponse(
            "stock_list", "stock-list", {"limit": "20000"},
            stock_url, stock_raw, "application/json", fetched_at, fetched_at, fetched_at,
            "first_capture_of_quantdb_response_no_vendor_publish_time",
        ),
        QuantDBRawResponse(
            "sector_concept", "sector-concept", {"category_id": "2", "sub_category": "sector_concept"},
            concept_url, concept_raw, "application/json", fetched_at, fetched_at, fetched_at,
            "first_capture_of_quantdb_response_no_vendor_publish_time",
        ),
    ]
    for trade_date, object_key, body, download_url in daily_payloads:
        responses.append(QuantDBRawResponse(
            "daily_unadjusted",
            "daily-" + trade_date,
            {"category_id": "1", "sub_category": "daily_unadjusted", "trade_date": trade_date, "object_key": object_key},
            download_url,
            body,
            "application/vnd.apache.parquet",
            fetched_at,
            fetched_at,
            fetched_at,
            "first_capture_of_quantdb_response_no_vendor_publish_time",
        ))
    ledger = ResearchSignalLedger(OUTPUT / "ledger", ledger_id="quantdb-theme-graph-20260930")
    captured = capture_quantdb_theme_graph_v1(
        ledger=ledger,
        u0=None,
        decision_session=pd.Timestamp(decision),
        prior_session=pd.Timestamp(prior),
        decision_at=pd.Timestamp(decision).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=18),
        responses=responses,
    )
    summary = {
        "adapter_version": captured.adapter_version,
        "status": captured.status,
        "abstain_reason": captured.abstain_reason,
        "decision_session": str(captured.decision_session),
        "prior_session": f"{prior[:4]}-{prior[4:6]}-{prior[6:]}",
        "reference_session": f"{reference[:4]}-{reference[4:6]}-{reference[6:]}",
        "graph_called": captured.graph_result is not None,
        "u0_built": False,
        "next_blocker_after_clock": "forward_u0_requires_253_price_sessions_and_60_adv_observations",
        "coverage": dict(captured.coverage),
        "receipt_count": len(captured.receipt_ids),
        "promotion_approved": False,
        "not_an_account_result": True,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
