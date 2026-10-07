"""No-account liquidity-rank audit for the broad basket.

Financial tiers stay out: local ``available_at`` is empty, and announcement
dates are not a substitute.  This file does not build holdings, replay a
ledger, or report a return.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import json

import pandas as pd

import extend_2018_band as band
import stable_book as book

OUT = ROOT / "output" / "liquidity_rank_audit_20260930"
FINANCIALS = (
    ROOT / "data/local_market/1c85fd9aceaa03c3c4863bec/financials.parquet",
    ROOT / "data/local_market/fa3fd23e8e6824fb619bda46/financials.parquet",
)


def financial_available_at_counts(paths=FINANCIALS) -> list[dict]:
    counts = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["available_at"])
        empty = int(frame["available_at"].isna().sum())
        counts.append({
            "path": str(path.relative_to(ROOT)),
            "rows": int(len(frame)),
            "available_at_empty": empty,
            "available_at_present": int(len(frame) - empty),
        })
    return counts


def cadence_days(calendar, *, anchor, start, end, step=20) -> tuple[pd.Timestamp, ...]:
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize()
    anchor = pd.Timestamp(anchor).normalize()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    position = int(calendar.get_loc(anchor))
    days = []
    step_count = 0
    while position - step * step_count >= 0:
        day = pd.Timestamp(calendar[position - step * step_count])
        if day < start:
            break
        days.append(day)
        step_count += 1
    days = list(reversed(days))
    step_count = 1
    while position + step * step_count < len(calendar):
        day = pd.Timestamp(calendar[position + step * step_count])
        if day > end:
            break
        days.append(day)
        step_count += 1
    return tuple(days)


def band_audit(observations: list[tuple[float, str]]) -> tuple[list[dict], dict]:
    """Rank one day.  Financial fields stay empty and no name is a holding."""
    ordered = sorted(observations, key=lambda item: (-item[0], item[1]))
    if len(ordered) < band.RANK_START:
        return [], {
            "status": "not_bought",
            "qualified": len(ordered),
            "in_band": 0,
            "rank_order_ok": True,
            "financial_tier_used": False,
        }
    records = []
    selection_order = 0
    for rank, (amount, symbol) in enumerate(ordered, start=1):
        if rank < band.RANK_START:
            continue
        if rank > band.RANK_END:
            break
        in_band = amount >= band.MIN_ADV
        if in_band:
            selection_order += 1
        records.append({
            "symbol": symbol,
            "adv_60": amount,
            "liquidity_rank": rank,
            "financial_score": None,
            "financial_available_at": None,
            "financial_tier_used": False,
            "selection_order": selection_order if in_band else None,
            "in_liquidity_band": in_band,
            "portfolio_selected": False,
        })
    ranks = [row["liquidity_rank"] for row in records if row["in_liquidity_band"]]
    monotonic = all(later >= earlier for earlier, later in zip(ranks, ranks[1:]))
    return records, {
        "status": "ok" if monotonic else "rank_order_invalid",
        "qualified": len(ordered),
        "in_band": len(ranks),
        "rank_order_ok": monotonic,
        "financial_tier_used": False,
    }


def main() -> None:
    financials = financial_available_at_counts()
    if any(item["available_at_present"] for item in financials):
        raise SystemExit("financial_available_at_unexpectedly_present")
    manifest = json.loads(book.MANIFEST.read_text())
    calendar = pd.DatetimeIndex(book._dated(pd.read_parquet(
        book.MANIFEST.parent / manifest["calendar"]["path"])).session)
    days = cadence_days(calendar, anchor=band.ANCHOR, start=book.START, end=book.END)
    print(json.dumps({"stage": "days", "count": len(days), "first": str(days[0].date()), "last": str(days[-1].date())}), flush=True)
    history = band.load_amount_history(end=book.END)
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "decision_audit.jsonl"
    day_rows = []
    with audit_path.open("w", encoding="utf-8") as handle:
        for day in days:
            observations = []
            for code, (dates, values) in history.items():
                amount = band.amount_on(dates, values, day)
                if amount is not None:
                    observations.append((amount, code))
            records, status = band_audit(observations)
            day_key = str(day.date())
            day_rows.append({"date": day_key, **status, "financial_available_at": None})
            for record in records:
                handle.write(json.dumps({"date": day_key, **record}, ensure_ascii=False) + "\n")
            print(json.dumps({"date": day_key, **status}), flush=True)
    invalid = [row["date"] for row in day_rows if row["status"] == "rank_order_invalid"]
    summary = {
        "account_replay_started": False,
        "portfolio_constructed": False,
        "not_a_return": True,
        "not_a_baseline": True,
        "adoptable_performance": False,
        "promotion_approved": False,
        "financial_tier_status": "blocked_data",
        "financial_tier_reason": "available_at_empty",
        "financials": financials,
        "window": [str(book.START.date()), str(book.END.date())],
        "cadence_anchor": str(pd.Timestamp(band.ANCHOR).date()),
        "decision_days": len(day_rows),
        "days_not_bought": sum(row["status"] == "not_bought" for row in day_rows),
        "days_rank_order_ok": sum(row["status"] == "ok" for row in day_rows),
        "rank_order_invalid_days": invalid,
        "phase_accounts": "not_started",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
