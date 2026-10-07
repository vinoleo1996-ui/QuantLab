"""Historical theme-graph account simulation on already completed sessions.

The concept map is the current QuantDB snapshot, applied to every past
decision.  This is a labeled replay, not a forward decision and not an
adoptable performance.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantlab.backtest.ledger import Ledger  # noqa: E402
from quantlab.data.contracts import InstrumentRules  # noqa: E402
from quantlab.data.exchange_rules import limit_prices, limit_ratio  # noqa: E402
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
from quantlab.theme_carrier_graph import order_new_entries  # noqa: E402
import importlib.util  # noqa: E402

_seed_spec = importlib.util.spec_from_file_location(
    "capture_quantdb_u0_seed", ROOT / "scripts" / "capture_quantdb_u0_seed.py"
)
_seed = importlib.util.module_from_spec(_seed_spec)
_seed_spec.loader.exec_module(_seed)
LEDGER_ID = _seed.LEDGER_ID
SOURCE_ROOT = _seed.OUTPUT
_request = _seed._request
_stored_trade_dates = _seed._stored_trade_dates
_url = _seed._url


WINDOW_START = "2025-09-12"
WINDOW_END = "2026-09-29"
HOLDINGS = 50
OUT = ROOT / "output" / "theme_history_account_20260929"
LEDGER_RUN_ID = "theme-history-account-20260929"
SOURCE = SOURCE_ROOT / "ledger"
SEGMENTS = (
    ("2018_2021", "2018-01-02", "2021-12-31"),
    ("2022_2025", "2022-01-01", "2025-12-31"),
    ("2025_to_20260922", "2025-01-02", "2026-09-22"),
)


def rebalance_codes(targets: tuple[str, ...] | list[str], quantities: list[str]) -> tuple[str, ...]:
    """Targets plus names still held, so a departure can be sold."""
    chosen = list(targets)
    seen = set(chosen)
    for code in quantities:
        if code not in seen:
            chosen.append(code)
            seen.add(code)
    return tuple(chosen)


def book_targets(held: list[str], ranked: list[str], *, limit: int = HOLDINGS) -> tuple[str, ...]:
    """Keep names still in the ranked pool, then fill openings from ``ranked``."""
    rank = {code: index for index, code in enumerate(ranked)}
    pool = set(ranked)
    kept = [code for code in held if code in pool]
    kept.sort(key=lambda code: rank[code])
    chosen = kept[:limit]
    chosen_set = set(chosen)
    for code in ranked:
        if len(chosen) >= limit:
            break
        if code not in chosen_set:
            chosen.append(code)
            chosen_set.add(code)
    return tuple(chosen)


def performance(nav: list[float], dates: list[str]) -> dict[str, float | None]:
    if len(nav) < 2 or nav[0] <= 0:
        return {"cagr": None, "max_drawdown": None}
    wealth = np.asarray(nav, dtype=float)
    years = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 1 / 365.25)
    peak = np.maximum.accumulate(wealth)
    return {
        "cagr": float((wealth[-1] / wealth[0]) ** (1 / years) - 1),
        "max_drawdown": float((wealth / peak - 1).min()),
    }


def slice_performance(nav: list[float], dates: list[str], start: str, end: str) -> dict[str, float | None]:
    chosen = [(value, day) for value, day in zip(nav, dates) if start <= day <= end]
    if len(chosen) < 2:
        return {"cagr": None, "max_drawdown": None, "start": None, "end": None}
    values, days = zip(*chosen)
    result = performance(list(values), list(days))
    result["start"] = days[0]
    result["end"] = days[-1]
    return result


def candidate_decision(full_cagr: float | None, max_drawdown: float | None, segment_cagrs: list[float | None]) -> dict[str, bool]:
    """A full-window CAGR of at least 12% stays, including results above 18%."""
    retained = (
        full_cagr is not None
        and full_cagr >= 0.12
        and max_drawdown is not None
        and max_drawdown >= -0.32
        and bool(segment_cagrs)
        and all(value is not None and value > 0 for value in segment_cagrs)
    )
    return {
        "candidate_retained": retained,
        "promotion_approved": False,
        "adoptable_performance": False,
    }


def _instrument_id(symbol: str) -> str:
    number, separator, market = str(symbol).upper().partition(".")
    if separator != "." or len(number) != 6 or not number.isdigit() or market not in {"SH", "SZ", "BJ"}:
        raise ValueError("invalid_symbol")
    if number.startswith(("4", "8", "92", "43", "83", "87")):
        return "bj." + number
    if number.startswith(("5", "6", "9")):
        return "sh." + number
    return "sz." + number


def _load_partition(spec: tuple[str, str]) -> tuple[str, pd.DataFrame]:
    day, path = spec
    frame = pd.read_parquet(path)
    return day, pd.DataFrame(
        {
            "instrument_id": frame["symbol"].map(_instrument_id),
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "amount": pd.to_numeric(frame["amount"], errors="coerce"),
        }
    )


def _rule(code: str, ratio: float) -> InstrumentRules:
    return InstrumentRules(
        instrument_id=code,
        asset_type="stock",
        buy_lot=100,
        sell_lot=1,
        sell_delay=1,
        price_tick=0.01,
        commission=0.00025,
        min_commission=5.0,
        limit_ratio=ratio,
    )


def derived_open_bar(*, day: str, open_price: float, reference: float, ratio: float, is_st: bool, listing_session: int) -> dict:
    """Daily-open assumption the research ledger will execute, with the 09:25 status time."""
    up, down = limit_prices(reference, ratio, 0.01)
    return {
        "open": float(open_price),
        "reference_price": float(reference),
        "limit_up": up,
        "limit_down": down,
        "open_status": "derived",
        "open_status_basis": "daily_open_assumption",
        "rule_version": "cn_daily_v1",
        "is_st_at_open": bool(is_st),
        "listing_session_lower_bound": int(listing_session),
        "open_status_available_at": str(pd.Timestamp(day).date()) + " 09:25:00",
    }


def _download_missing(needed: list[str]) -> int:
    stored = _stored_trade_dates(SOURCE / "ledger.sqlite3")
    missing = [day for day in needed if day.replace("-", "") not in stored]
    if not missing:
        return 0
    key = os.environ.get("QUANTDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("lookback_missing_and_QUANTDB_API_KEY_required")
    manifest = json.loads(_request(_url(
        "/api/v1/data/download/manifest",
        {"category_id": "1", "sub_category": "daily_unadjusted", "layout": "v2"},
    ), key))
    by_date = {}
    for item in manifest.get("files") or []:
        path = str(item.get("relative_path") or "")
        if "dt=" not in path:
            continue
        trade_date = path.split("dt=", 1)[1][:8]
        if len(trade_date) == 8 and trade_date.isdigit():
            by_date[trade_date] = str(item["key"])
    ledger = ResearchSignalLedger(SOURCE, ledger_id=LEDGER_ID)
    fetched_at = pd.Timestamp.now(tz="Asia/Shanghai")

    def _one(day: str) -> tuple[str, bytes]:
        trade_date = day.replace("-", "")
        body = _request(_url("/api/v1/data/download", {
            "category_id": "1",
            "sub_category": "daily_unadjusted",
            "layout": "v2",
            "object_key": by_date[trade_date],
        }), key)
        return trade_date, body

    downloaded = 0
    jobs = [day for day in missing if day.replace("-", "") in by_date]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one, day) for day in jobs]
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
                availability_basis="historical_lookback_seed_not_a_forward_receipt",
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
            if downloaded % 40 == 0:
                print(f"downloaded {downloaded}/{len(jobs)}", flush=True)
    return downloaded


def _raw_response(dataset, response_id, parameters, raw, content_type, path, when):
    return QuantDBRawResponse(
        dataset=dataset,
        response_id=response_id,
        request_parameters=parameters,
        canonical_url="https://www.quantdb.cn" + path,
        raw=raw,
        content_type=content_type,
        published_at=when,
        first_captured_at=when,
        available_at=when,
        availability_basis="historical_replay_of_current_theme_snapshot",
    )


def main() -> None:
    import sqlite3
    database = sqlite3.connect(SOURCE / "ledger.sqlite3")
    blobs = {"trade_calendar": None, "stock_list": None, "sector_concept": None}
    paths = {}
    for (body_json,) in database.execute("SELECT body_json FROM events WHERE kind='receipt' AND status='accepted'"):
        body = json.loads(body_json)
        metadata = body.get("metadata") or {}
        dataset = metadata.get("dataset")
        if dataset in blobs and blobs[dataset] is None:
            blobs[dataset] = SOURCE / body["blob_path"]
        if dataset == "daily_unadjusted":
            trade_date = (metadata.get("request_parameters") or {}).get("trade_date")
            if trade_date:
                paths[str(trade_date)] = SOURCE / body["blob_path"]
    database.close()
    calendar = json.loads(blobs["trade_calendar"].read_text())
    open_days = [str(row["trade_date"])[:10] for row in calendar["data"] if int(row.get("is_open", 1)) == 1]
    start = open_days.index(WINDOW_START)
    end = open_days.index(WINDOW_END)
    needed = open_days[start - (SEED_SESSION_COUNT - 1) : end + 1]
    downloaded = _download_missing(needed)
    paths = _refresh_paths(paths)
    missing = [day for day in needed if day.replace("-", "") not in paths]
    if missing:
        raise SystemExit("lookback_still_missing:" + str(len(missing)))
    specs = [(day, str(paths[day.replace("-", "")])) for day in needed]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        loaded = list(pool.map(_load_partition, specs, chunksize=16))
    panels = {day: frame for day, frame in loaded}
    closes, opens, amounts = {}, {}, {}
    for day, frame in panels.items():
        indexed = frame.drop_duplicates("instrument_id").set_index("instrument_id")
        closes[day] = indexed["close"]
        opens[day] = indexed["open"]
        amounts[day] = indexed["amount"]
    stock_raw = blobs["stock_list"].read_bytes()
    concept_raw = blobs["sector_concept"].read_bytes()
    calendar_raw = blobs["trade_calendar"].read_bytes()
    stock_rows = json.loads(stock_raw)["data"]
    identity = pd.DataFrame(
        {
            "instrument_id": [_instrument_id(row["symbol"]) for row in stock_rows],
            "is_st": [bool(row.get("is_st")) for row in stock_rows],
            "is_delisted": [bool(row.get("is_delisted")) for row in stock_rows],
            "name": [str(row.get("name") or "") for row in stock_rows],
            "list_date": [
                pd.to_datetime(row["list_date"], errors="coerce")
                if str(row.get("list_date")) not in {"", "0", "None"} else pd.NaT
                for row in stock_rows
            ],
        }
    ).drop_duplicates("instrument_id").set_index("instrument_id")
    calendar_index = pd.DatetimeIndex(pd.to_datetime(open_days))
    decisions = {open_days[index] for index in range(start, end, 20)}
    clock = {"now": pd.Timestamp(WINDOW_START + " 17:40:00+08:00")}
    research = ResearchSignalLedger(
        OUT / "research-ledger", ledger_id=LEDGER_RUN_ID, now_fn=lambda: clock["now"]
    )
    books = {
        "baseline": Ledger(600_000.0, slippage=0.003, allow_open_assumption=True),
        "theme": Ledger(600_000.0, slippage=0.003, allow_open_assumption=True),
    }
    held = {name: [] for name in books}
    rules_cache = {}
    nav = {name: [] for name in books}
    dates = []
    decision_rows = []
    pending = False

    def ensure_rule(code, ratio):
        return rules_cache.setdefault(code, _rule(code, ratio))

    OUT.mkdir(parents=True, exist_ok=True)
    day_pos = {day: index for index, day in enumerate(open_days)}
    cache_path = OUT / "decision_cache.jsonl"
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["date"]] = row
    for day in open_days[start : end + 1]:
        session = day_pos[day]
        previous = open_days[session - 1]
        for name, ledger in books.items():
            if pending:
                bars, rules, adv = {}, {}, {}
                panel = closes[day]
                panel_open = opens[day]
                prior_panel = closes[previous]
                codes = set(held[name]) | set(ledger.quantities())
                for code in codes:
                    if code not in panel.index or code not in prior_panel.index or code not in identity.index:
                        continue
                    info = identity.loc[code]
                    if isinstance(info, pd.DataFrame):
                        info = info.iloc[0]
                    if pd.isna(info.list_date):
                        continue
                    listing = _listing_session(pd.Timestamp(info.list_date), pd.Timestamp(day), calendar_index)
                    ratio = limit_ratio(code, day, bool(info.is_st), listing)
                    if ratio is None or listing is None:
                        continue
                    reference = float(prior_panel.loc[code])
                    bars[code] = derived_open_bar(
                        day=day,
                        open_price=float(panel_open.loc[code]),
                        reference=reference,
                        ratio=float(ratio),
                        is_st=bool(info.is_st),
                        listing_session=int(listing),
                    )
                    rules[code] = ensure_rule(code, ratio)
                    history = open_days[max(0, session - 20) : session]
                    amount_values = []
                    for item in history:
                        series = amounts[item]
                        if code in series.index and np.isfinite(series.loc[code]):
                            amount_values.append(float(series.loc[code]))
                    adv[code] = float(np.mean(amount_values) * 10000) if amount_values else np.nan
                ledger.execute(day, session, bars, rules, adv)
            ledger.close(day, session, closes[day].dropna().astype(float).to_dict())
            nav[name].append(float(ledger.nav()))
        dates.append(day)
        pending = False
        if day not in decisions:
            continue
        cached_row = cached.get(day)
        if cached_row is not None and cached_row.get("graph_status") != "ok":
            decision_rows.append({"date": day, "u0_status": cached_row.get("u0_status"), "reason": cached_row.get("reason")})
            print(day, cached_row.get("u0_status"), cached_row.get("reason"), flush=True)
            continue
        if cached_row is None:
            window = open_days[session - SEED_SESSION_COUNT + 1 : session + 1]
            partitions = [QuantDBPartition(pd.Timestamp(item), panels[item], item, "") for item in window]
            when = pd.Timestamp(day + " 17:30:00+08:00")
            clock["now"] = pd.Timestamp(day + " 17:40:00+08:00")
            recorded = research.record_receipt(
                source_id="quantdb.u0-seed",
                source_family="quantdb",
                origin_id="quantdb:theme-history-u0:" + day,
                canonical_url="https://www.quantdb.cn/api/v1/data/download",
                raw=json.dumps({"sessions": [item.replace("-", "") for item in window]}).encode(),
                content_type="application/json",
                published_at=when,
                first_captured_at=when,
                available_at=when,
                availability_basis="historical_replay_not_a_forward_receipt",
                metadata={"dataset": "u0_seed_manifest", "response_id": "u0-" + day, "request_parameters": {}},
            )
            evidence = research.export_receipt_evidence(recorded["event_id"])
            snapshot = build_quantdb_u0_snapshot(
                ledger=research,
                rows=assemble_quantdb_u0_rows(
                    partitions=partitions,
                    identity=identity,
                    provenance={
                        "source_receipt_id": evidence["canonical_receipt_id"],
                        "source_receipt_hash": evidence["raw"]["sha256"],
                        "source_published_at": when,
                        "source_first_captured_at": when,
                        "source_available_at": evidence["receipt"]["available_at"],
                    },
                ),
                session=pd.Timestamp(day),
                decision_at=pd.Timestamp(day + " 18:00:00+08:00"),
                cadence_anchor=pd.Timestamp(WINDOW_START),
                exchange_sessions=open_days,
            )
            if snapshot.status != "ok":
                failed = {"date": day, "u0_status": snapshot.status, "reason": snapshot.abstain_reason, "graph_status": snapshot.status}
                decision_rows.append({"date": day, "u0_status": snapshot.status, "reason": snapshot.abstain_reason})
                with cache_path.open("a", encoding="utf-8") as cache_file:
                    cache_file.write(json.dumps(failed, ensure_ascii=False) + "\n")
                print(day, snapshot.status, snapshot.abstain_reason, flush=True)
                continue
            prior = open_days[session - 1]
            daily_responses = []
            for item in (open_days[session - 2], prior, day):
                trade_date = item.replace("-", "")
                daily_responses.append(_raw_response(
                    "daily_unadjusted", "daily-" + trade_date + "-" + day,
                    {"category_id": "1", "sub_category": "daily_unadjusted", "trade_date": trade_date},
                    paths[trade_date].read_bytes(),
                    "application/vnd.apache.parquet",
                    "/api/v1/data/download",
                    when,
                ))
            captured = capture_quantdb_theme_graph_v1(
                ledger=research,
                u0=snapshot,
                decision_session=pd.Timestamp(day),
                prior_session=pd.Timestamp(prior),
                decision_at=pd.Timestamp(day + " 18:00:00+08:00"),
                responses=(
                    _raw_response("trade_calendar", "calendar-" + day, {}, calendar_raw, "application/json", "/api/v1/data/calendar", when),
                    _raw_response("stock_list", "stocks-" + day, {}, stock_raw, "application/json", "/api/v1/data/stock-list", when),
                    _raw_response("sector_concept", "concepts-" + day, {"category_id": "2"}, concept_raw, "application/json", "/api/v1/data/preview", when),
                    _raw_response("daily_manifest", "manifest-" + day, {"layout": "v2"}, b'{"files":[]}', "application/json", "/api/v1/data/download/manifest", when),
                    *daily_responses,
                ),
            )
            priceable = []
            if captured.status == "ok" and captured.graph_result is not None:
                priceable = list(captured.graph_result.stock_scores.index)
            liquidity = sorted(priceable, key=lambda code: snapshot.rank_by_instrument.get(code, 10**9))
            theme_new = liquidity
            if captured.status == "ok" and captured.graph_result is not None:
                theme_new = list(order_new_entries(
                    baseline_candidate_order=liquidity,
                    retained_holdings=held["theme"],
                    result=captured.graph_result,
                ).new_entry_order)
            cached_row = {
                "date": day,
                "graph_status": captured.status,
                "u0_members": len(snapshot.members),
                "excluded": (captured.coverage or {}).get("excluded_member_count"),
                "scored": (captured.coverage or {}).get("scored_member_count"),
                "liquidity": liquidity,
                "theme_new": theme_new,
            }
            with cache_path.open("a", encoding="utf-8") as cache_file:
                cache_file.write(json.dumps(cached_row, ensure_ascii=False) + "\n")
        else:
            liquidity = list(cached_row["liquidity"])
            theme_new = list(cached_row["theme_new"])
        retained_theme = [code for code in held["theme"] if code in set(liquidity)]
        rank = {code: index for index, code in enumerate(liquidity)}
        retained_theme.sort(key=lambda code: rank[code])
        retained_theme = retained_theme[:HOLDINGS]
        targets = {
            "baseline": book_targets(held["baseline"], liquidity),
            "theme": tuple((retained_theme + [code for code in theme_new if code not in set(retained_theme)])[:HOLDINGS]),
        }
        for name, ledger in books.items():
            held[name] = list(targets[name])
            target_rules = {}
            for code in rebalance_codes(targets[name], list(ledger.quantities())):
                info = identity.loc[code] if code in identity.index else None
                if isinstance(info, pd.DataFrame):
                    info = info.iloc[0]
                listing = 60
                ratio = 0.1
                if info is not None and not pd.isna(info.list_date):
                    listing_value = _listing_session(pd.Timestamp(info.list_date), pd.Timestamp(day), calendar_index)
                    ratio_value = limit_ratio(code, day, bool(info.is_st), listing_value)
                    if listing_value is not None and ratio_value is not None:
                        listing = int(listing_value)
                        ratio = float(ratio_value)
                target_rules[code] = ensure_rule(code, ratio)
            if not targets[name]:
                continue
            ledger.queue_targets({code: 0.98 / len(targets[name]) for code in targets[name]}, day, "rebalance", target_rules)
        pending = True
        decision_rows.append({
            "date": day,
            "u0_members": cached_row.get("u0_members"),
            "excluded": cached_row.get("excluded"),
            "scored": cached_row.get("scored"),
            "graph_status": cached_row.get("graph_status"),
        })
        overlap = len(set(targets["baseline"]) & set(targets["theme"]))
        decision_rows[-1]["entry_disagreement"] = len(set(targets["baseline"]) ^ set(targets["theme"]))
        decision_rows[-1]["shared_names"] = overlap
        print(
            day, cached_row.get("graph_status"), "held", {name: len(held[name]) for name in held},
            "fills", {name: len(ledger.fills) for name, ledger in books.items()},
            "disagree", decision_rows[-1]["entry_disagreement"],
            flush=True,
        )

    summary = {
        "historical_simulation_with_current_theme_snapshot": True,
        "forward_decision": False,
        "promotion_approved": False,
        "not_an_account_result": True,
        "adoptable_performance": False,
        "window": [WINDOW_START, WINDOW_END],
        "downloaded_lookback_partitions": downloaded,
        "stored_daily_partitions": len(paths),
        "stored_daily_span": [min(paths), max(paths)] if paths else [],
        "decisions": [row["date"] for row in decision_rows],
        "baseline": performance(nav["baseline"], dates),
        "theme": performance(nav["theme"], dates),
        "segments": {
            name: {
                label: slice_performance(nav[name], dates, start, end)
                for label, start, end in SEGMENTS
            }
            for name in books
        },
        "ending_nav": {name: books[name].nav() for name in books},
        "fills": {name: len(books[name].fills) for name in books},
        "orders": {name: len(books[name].orders) for name in books},
        "execution_rejections": {
            name: {
                str(reason): int(count)
                for reason, count in pd.Series(
                    [event.get("reason") for event in books[name].events if event.get("type") == "execution_rejected"]
                ).value_counts().items()
            }
            for name in books
        },
    }
    if WINDOW_START == "2018-01-02":
        summary["candidate"] = {
            name: candidate_decision(
                summary[name]["cagr"],
                summary[name]["max_drawdown"],
                [summary["segments"][name][label]["cagr"] for label, _start, _end in SEGMENTS],
            )
            for name in books
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "decisions.json").write_text(json.dumps(decision_rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _refresh_paths(paths):
    return _blob_paths()


def _blob_paths():
    import sqlite3
    database = sqlite3.connect(SOURCE / "ledger.sqlite3")
    found = {}
    for (body_json,) in database.execute("SELECT body_json FROM events WHERE kind='receipt' AND status='accepted'"):
        body = json.loads(body_json)
        metadata = body.get("metadata") or {}
        if metadata.get("dataset") != "daily_unadjusted":
            continue
        trade_date = (metadata.get("request_parameters") or {}).get("trade_date")
        if trade_date and body.get("blob_path"):
            found[str(trade_date)] = SOURCE / body["blob_path"]
    database.close()
    return found


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=WINDOW_START)
    parser.add_argument("--end", default=WINDOW_END)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    WINDOW_START = args.start
    WINDOW_END = args.end
    if args.out:
        OUT = Path(args.out)
        if not OUT.is_absolute():
            OUT = ROOT / OUT
        LEDGER_RUN_ID = "theme-history-" + args.start.replace("-", "") + "-" + args.end.replace("-", "")
    main()
