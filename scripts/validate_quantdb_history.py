"""Validate U0 and the concept graph on the last completed session.

This is a historical mechanism check.  It is not a forward decision: the
partition bytes were captured on 2026-09-30, and the concept file has no
as-of date.  A replay clock lets the frozen builder and scorer run against
2026-09-29.  The summary says so.  Nothing here is an account result.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantlab.data.exchange_rules import limit_ratio  # noqa: E402
from quantlab.data.quantdb_theme_market_adapter import (  # noqa: E402
    QuantDBRawResponse,
    _listing_session,
    capture_quantdb_theme_graph_v1,
)
from quantlab.data.quantdb_u0_seed import (  # noqa: E402
    SEED_SESSION_COUNT,
    QuantDBPartition,
    assemble_quantdb_u0_rows,
    build_quantdb_u0_snapshot,
)
from quantlab.research_signal_ledger import ResearchSignalLedger  # noqa: E402
from quantlab.research_universe import (  # noqa: E402
    U0,
    UniverseSnapshot,
    forward_membership_sha256,
)


SOURCE = ROOT / "output" / "quantdb_theme_graph_20260930" / "ledger"
SOURCE_ID = "quantdb-theme-graph-20260930"
OUT = ROOT / "output" / "quantdb_historical_validation_20260929"
DECISION = pd.Timestamp("2026-09-29")
PRIOR = pd.Timestamp("2026-09-28")
REPLAY_CAPTURE = pd.Timestamp("2026-09-29 17:30:00+08:00")
REPLAY_RECORD = pd.Timestamp("2026-09-29 17:40:00+08:00")
DECISION_AT = pd.Timestamp("2026-09-29 18:00:00+08:00")
BASIS = "historical_replay_of_bytes_captured_2026-09-30_not_a_forward_receipt"


def _instrument_id(symbol: str) -> str:
    number, separator, market = str(symbol).upper().partition(".")
    if separator != "." or len(number) != 6 or not number.isdigit() or market not in {"SH", "SZ", "BJ"}:
        raise ValueError("invalid_symbol")
    if number.startswith(("4", "8", "92", "43", "83", "87")):
        return "bj." + number
    if number.startswith(("5", "6", "9")):
        return "sh." + number
    return "sz." + number


def _blobs() -> dict[str, dict[str, str]]:
    database = sqlite3.connect(SOURCE / "ledger.sqlite3")
    found: dict[str, dict[str, str]] = {}
    for body_json, in database.execute(
        "SELECT body_json FROM events WHERE kind='receipt' AND status='accepted'"
    ):
        body = json.loads(body_json)
        metadata = body.get("metadata") or {}
        dataset = metadata.get("dataset")
        if dataset == "daily_unadjusted":
            trade_date = (metadata.get("request_parameters") or {}).get("trade_date")
            if trade_date:
                found[str(trade_date)] = body
        elif dataset in {"trade_calendar", "stock_list", "sector_concept"} and dataset not in found:
            found[dataset] = body
    database.close()
    return found


def _ensure_missing_day(blobs: dict[str, dict[str, str]]) -> None:
    if "20250912" in blobs:
        return
    key = os.environ.get("QUANTDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("missing_20250912_and_QUANTDB_API_KEY")
    manifest_url = "https://www.quantdb.cn/api/v1/data/download/manifest?" + urllib.parse.urlencode(
        {"category_id": "1", "sub_category": "daily_unadjusted", "layout": "v2"}
    )
    request = urllib.request.Request(manifest_url, headers={"X-API-Key": key, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        files = json.loads(response.read()).get("files") or []
    object_key = next(item["key"] for item in files if "dt=20250912" in str(item.get("relative_path") or ""))

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    current = "https://www.quantdb.cn/api/v1/data/download?" + urllib.parse.urlencode(
        {"category_id": "1", "sub_category": "daily_unadjusted", "layout": "v2", "object_key": object_key}
    )
    headers = {"X-API-Key": key, "Accept": "application/json"}
    for _hop in range(4):
        outgoing = urllib.request.Request(current, headers=headers)
        try:
            with opener.open(outgoing, timeout=90) as response:
                raw = response.read()
                break
        except urllib.error.HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            current = urllib.parse.urljoin(current, exc.headers["Location"])
            headers = {}
    ledger = ResearchSignalLedger(SOURCE, ledger_id=SOURCE_ID)
    recorded = ledger.record_receipt(
        source_id="quantdb.u0-seed",
        source_family="quantdb",
        origin_id="quantdb:daily-seed:20250912",
        canonical_url="https://www.quantdb.cn/api/v1/data/download?category_id=1&sub_category=daily_unadjusted&layout=v2&trade_date=20250912",
        raw=raw,
        content_type="application/vnd.apache.parquet",
        published_at=pd.Timestamp.now(tz="Asia/Shanghai"),
        first_captured_at=pd.Timestamp.now(tz="Asia/Shanghai"),
        available_at=pd.Timestamp.now(tz="Asia/Shanghai"),
        availability_basis="first_capture_of_quantdb_response_no_vendor_publish_time",
        metadata={
            "dataset": "daily_unadjusted",
            "response_id": "daily-seed-20250912",
            "request_parameters": {"category_id": "1", "sub_category": "daily_unadjusted", "trade_date": "20250912"},
        },
    )
    evidence = ledger.export_receipt_evidence(recorded["event_id"])
    blobs["20250912"] = evidence["receipt"]["body"]


def _load_partition(spec: tuple[str, str]) -> tuple[str, pd.DataFrame]:
    """Read one saved partition in a worker process."""
    day, path = spec
    frame = pd.read_parquet(path)
    return day, pd.DataFrame(
        {
            "instrument_id": frame["symbol"].map(_instrument_id),
            "amount": pd.to_numeric(frame["amount"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
        }
    )


def _response(dataset: str, response_id: str, parameters: dict[str, str], raw: bytes, content_type: str, path: str) -> QuantDBRawResponse:
    return QuantDBRawResponse(
        dataset=dataset,
        response_id=response_id,
        request_parameters=parameters,
        canonical_url="https://www.quantdb.cn" + path,
        raw=raw,
        content_type=content_type,
        published_at=REPLAY_CAPTURE,
        first_captured_at=REPLAY_CAPTURE,
        available_at=REPLAY_CAPTURE,
        availability_basis=BASIS,
    )


def _limit_gap(snapshot, partitions, identity, open_days) -> tuple[dict[str, int], tuple[str, ...]]:
    calendar = pd.DatetimeIndex(pd.to_datetime(open_days))
    by_session = {pd.Timestamp(item.session).normalize(): item.frame.set_index("instrument_id") for item in partitions}
    decision = pd.Timestamp("2026-09-29")
    prior = pd.Timestamp("2026-09-28")
    reference = pd.Timestamp("2026-09-24")
    reasons: dict[str, int] = {}
    priceable: list[str] = []
    for code in snapshot.members:
        reason = None
        if code not in identity.index or pd.isna(identity.loc[code, "list_date"] if "list_date" in identity.columns else identity.loc[code].get("list_date", pd.NaT)):
            reason = "missing_list_date"
        else:
            for session, anchor in ((prior, reference), (decision, prior)):
                if code not in by_session[session].index or code not in by_session[anchor].index:
                    reason = "missing_bar"
                    break
                listing = _listing_session(pd.Timestamp(identity.loc[code, "list_date"]), session, calendar)
                ratio = limit_ratio(code, session, bool(identity.loc[code, "is_st"]), listing)
                if ratio is None:
                    number = code.split(".")[-1]
                    if number.startswith(("4", "8", "92", "43", "83", "87")):
                        reason = "beijing_board_outside_limit_rule"
                    elif listing is None or (isinstance(listing, int) and listing < 6):
                        reason = "listing_session_unknown_or_new"
                    else:
                        reason = "limit_rule_undefined"
                    break
        if reason is None:
            priceable.append(str(code))
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    return reasons, tuple(priceable)


def main() -> None:
    blobs = _blobs()
    _ensure_missing_day(blobs)
    calendar = json.loads((SOURCE / blobs["trade_calendar"]["blob_path"]).read_text())
    open_days = [str(row["trade_date"])[:10] for row in calendar["data"] if int(row.get("is_open", 1)) == 1]
    end = open_days.index("2026-09-29")
    window = open_days[end - SEED_SESSION_COUNT + 1 : end + 1]
    if len(window) != SEED_SESSION_COUNT or any(day.replace("-", "") not in blobs for day in window):
        raise SystemExit("historical_window_incomplete:" + ",".join(day for day in window if day.replace("-", "") not in blobs))
    specs = [
        (day, str(SOURCE / blobs[day.replace("-", "")]["blob_path"]))
        for day in window
    ]
    workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(_load_partition, specs, chunksize=16))
    order = {day: index for index, day in enumerate(window)}
    loaded.sort(key=lambda item: order[item[0]])
    partitions = [QuantDBPartition(pd.Timestamp(day), frame, day, "") for day, frame in loaded]
    stock_raw = (SOURCE / blobs["stock_list"]["blob_path"]).read_bytes()
    concept_raw = (SOURCE / blobs["sector_concept"]["blob_path"]).read_bytes()
    calendar_raw = (SOURCE / blobs["trade_calendar"]["blob_path"]).read_bytes()
    stock_rows = json.loads(stock_raw)["data"]
    identity = pd.DataFrame(
        {
            "instrument_id": [_instrument_id(row["symbol"]) for row in stock_rows],
            "is_st": [bool(row.get("is_st")) for row in stock_rows],
            "is_delisted": [bool(row.get("is_delisted")) for row in stock_rows],
            "name": [str(row.get("name") or "") for row in stock_rows],
            "list_date": [pd.to_datetime(row["list_date"], errors="coerce") if str(row.get("list_date")) not in {"", "0", "None"} else pd.NaT for row in stock_rows],
        }
    ).drop_duplicates("instrument_id").set_index("instrument_id")
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = ResearchSignalLedger(OUT / "ledger", ledger_id="quantdb-historical-validation-20260929", now_fn=lambda: REPLAY_RECORD)
    manifest = json.dumps(
        {"validation_session": "2026-09-29", "sessions": [day.replace("-", "") for day in window]},
        sort_keys=True,
    ).encode("utf-8")
    recorded = ledger.record_receipt(
        source_id="quantdb.u0-seed",
        source_family="quantdb",
        origin_id="quantdb:historical-validation:20260929",
        canonical_url="https://www.quantdb.cn/api/v1/data/download",
        raw=manifest,
        content_type="application/json",
        published_at=REPLAY_CAPTURE,
        first_captured_at=REPLAY_CAPTURE,
        available_at=REPLAY_CAPTURE,
        availability_basis=BASIS,
        metadata={"dataset": "u0_seed_manifest", "response_id": "historical-20260929", "request_parameters": {}},
    )
    evidence = ledger.export_receipt_evidence(recorded["event_id"])
    rows = assemble_quantdb_u0_rows(
        partitions=partitions,
        identity=identity,
        provenance={
            "source_receipt_id": evidence["canonical_receipt_id"],
            "source_receipt_hash": evidence["raw"]["sha256"],
            "source_published_at": REPLAY_CAPTURE,
            "source_first_captured_at": REPLAY_CAPTURE,
            "source_available_at": evidence["receipt"]["available_at"],
        },
    )
    snapshot = build_quantdb_u0_snapshot(
        ledger=ledger,
        rows=rows,
        session=DECISION,
        decision_at=DECISION_AT,
        cadence_anchor=DECISION,
        exchange_sessions=open_days,
    )
    summary: dict[str, object] = {
        "historical_validation": True,
        "forward_decision": False,
        "validation_session": "2026-09-29",
        "window_first": window[0],
        "window_last": window[-1],
        "window_count": len(window),
        "replay_clock": str(REPLAY_RECORD),
        "actual_blob_capture_date": "2026-09-30",
        "concept_snapshot": "captured_2026-09-30_no_history_date",
        "u0_status": snapshot.status,
        "u0_reason": snapshot.abstain_reason,
        "u0_member_count": len(snapshot.members),
        "u0_membership_hash": snapshot.membership_hash,
        "promotion_approved": False,
        "not_an_account_result": True,
    }
    if snapshot.status == "ok":
        gap, priceable = _limit_gap(snapshot, partitions, identity, open_days)
        summary["limit_gap_inside_u0"] = gap
        summary["priceable_u0_member_count"] = len(priceable)
        daily_responses = []
        for day in ("2026-09-24", "2026-09-28", "2026-09-29"):
            trade_date = day.replace("-", "")
            raw = (SOURCE / blobs[trade_date]["blob_path"]).read_bytes()
            daily_responses.append(_response(
                "daily_unadjusted",
                "daily-" + trade_date,
                {"category_id": "1", "sub_category": "daily_unadjusted", "trade_date": trade_date},
                raw,
                "application/vnd.apache.parquet",
                "/api/v1/data/download",
            ))
        captured = capture_quantdb_theme_graph_v1(
            ledger=ledger,
            u0=snapshot,
            decision_session=DECISION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
            responses=(
                _response("trade_calendar", "calendar", {"start_date": "2016-01-01", "end_date": "2026-12-31"}, calendar_raw, "application/json", "/api/v1/data/calendar"),
                _response("stock_list", "stocks", {"limit": "20000"}, stock_raw, "application/json", "/api/v1/data/stock-list"),
                _response("sector_concept", "concepts", {"category_id": "2", "sub_category": "sector_concept"}, concept_raw, "application/json", "/api/v1/data/preview"),
                _response(
                    "daily_manifest",
                    "daily-manifest",
                    {"category_id": "1", "sub_category": "daily_unadjusted", "layout": "v2"},
                    json.dumps({"files": [{"trade_date": day.replace("-", "")} for day in ("2026-09-24", "2026-09-28", "2026-09-29")]}).encode("utf-8"),
                    "application/json",
                    "/api/v1/data/download/manifest",
                ),
                *daily_responses,
            ),
        )
        scores = None if captured.graph_result is None else captured.graph_result.stock_scores
        summary.update(
            {
                "graph_status": captured.status,
                "graph_reason": captured.abstain_reason,
                "graph_universe": "full_u0_unpriceable_names_excluded",
                "adapter_coverage": dict(captured.coverage),
                "scored_names": 0 if scores is None else int(scores.notna().sum()),
                "positive_score_names": 0 if scores is None else int((scores > 0).sum()),
                "score_median": None if scores is None or scores.empty else float(scores.median()),
                "eligible_themes": None if captured.graph_result is None else int(captured.graph_result.coverage.get("eligible_themes", 0)),
            }
        )
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
