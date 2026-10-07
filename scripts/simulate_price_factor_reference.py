"""Reorder only new entries of the 50-name reference by three price facts.

The facts are the stock's own yesterday limit-up, broken board, and advance.
No concept snapshot is used.  Relative strength stays out because it needs a
theme group.  This is a research replay, not an adoptable performance.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import stable_book as book
from quantlab.data.exchange_rules import limit_prices
from quantlab.strategy.expert_router import select_reference_basket

OUT = ROOT / "output" / "price_factor_reference_20260930"
HOLDINGS = 50
RETENTION_RANK = 150
INDUSTRY_SLOTS = 15
GROSS = 0.90


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "simulate_theme_history", ROOT / "scripts" / "simulate_theme_history.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _limit(row, reference: float) -> float | None:
    quoted = row.get("limit_up")
    if quoted is not None and np.isfinite(quoted) and float(quoted) > 0:
        return float(quoted)
    ratio = row.get("limit_ratio_at_open")
    if ratio is None or not np.isfinite(ratio) or not np.isfinite(reference) or reference <= 0:
        return None
    upper, _lower = limit_prices(float(reference), float(ratio), 0.01)
    return float(upper) if np.isfinite(upper) else None


def bar_components(current, previous) -> dict[str, float] | None:
    """One name's three price facts.  Missing limits abstain that name."""
    prior_close = float(previous.get("close"))
    current_close = float(current.get("close"))
    current_high = float(current.get("high"))
    if not all(np.isfinite(value) and value > 0 for value in (prior_close, current_close, current_high)):
        return None
    prior_limit = _limit(previous, float(previous.get("reference_price", np.nan)))
    current_limit = _limit(current, float(current.get("reference_price", prior_close)))
    if prior_limit is None or current_limit is None:
        return None
    tick = 0.01
    return {
        "prior_limit_up_premium": 1.0 if prior_close >= prior_limit - tick / 2 else 0.0,
        "broken_board_rate": 1.0 if current_high >= current_limit - tick / 2 and current_close < current_limit - tick / 2 else 0.0,
        "advancing_breadth": 1.0 if current_close > prior_close else 0.0,
    }


def price_factor_priorities(components: dict[str, dict[str, float]]) -> dict[str, float]:
    """Equal-weight percentile.  A broken board ranks lower."""
    if len(components) < 2:
        return {code: 0.0 for code in components}
    frame = pd.DataFrame(components).T
    directions = {
        "prior_limit_up_premium": True,
        "broken_board_rate": False,
        "advancing_breadth": True,
    }
    scores = []
    for name, higher in directions.items():
        ranks = frame[name].rank(ascending=higher, method="average")
        scores.append((ranks - 1) / (len(frame) - 1))
    combined = sum(scores) / len(scores)
    return {code: float(combined.loc[code]) for code in combined.index}


def select_price_factor_book(heads, industries, current, priorities) -> tuple[str, ...]:
    return select_reference_basket(
        heads,
        current,
        industries,
        priorities=priorities,
        holdings=HOLDINGS,
        retention_rank=RETENTION_RANK,
        industry_slots=INDUSTRY_SLOTS,
        selection_policy="reference_retention",
    )


def _row(panel: pd.DataFrame, day: pd.Timestamp):
    if day not in panel.index:
        return None
    row = panel.loc[day]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def build_targets(sessions, heads, industries, panels) -> list[dict]:
    by_day = {pd.Timestamp(day).normalize(): day for day in heads}
    calendar = list(pd.DatetimeIndex(pd.to_datetime(sessions)).normalize())
    position = {day: index for index, day in enumerate(calendar)}
    basket: tuple[str, ...] = ()
    targets = []
    for session in calendar:
        key = by_day.get(session)
        if key is None:
            continue
        prior = calendar[position[session] - 1] if position[session] else None
        components = {}
        if prior is not None:
            for code in heads[key]:
                panel = panels.get(code)
                if panel is None:
                    continue
                current = _row(panel, session)
                previous = _row(panel, prior)
                if current is None or previous is None:
                    continue
                facts = bar_components(current, previous)
                if facts is not None:
                    components[code] = facts
        chosen = select_price_factor_book(heads[key], industries[key], basket, price_factor_priorities(components))
        previous_names = set(basket)
        basket = chosen
        weights = {code: GROSS / len(chosen) for code in chosen} if chosen else {}
        targets.append({
            "signal_date": str(session.date()),
            "weights": weights,
            "active": list(chosen),
            "sell": sorted(previous_names - set(chosen)),
            "sell_only": False,
            "max_exposure": GROSS,
            "sleeve": 0,
        })
    return targets


def _already_canonical(mapping) -> bool:
    return bool(mapping) and all(str(code).startswith(("sh.", "sz.", "bj.")) for code in mapping)


def _prepare_panel(frame: pd.DataFrame) -> pd.DataFrame:
    return book._dated(frame)


def main() -> None:
    history = _load_gate()
    parts = {name: json.loads(path.read_text()) for name, path in book.INPUTS.items()}
    heads, industries = book._band_maps(parts)
    manifest = json.loads(book.MANIFEST.read_text())
    root = book.MANIFEST.parent
    calendar = pd.DatetimeIndex(book._dated(pd.read_parquet(root / manifest["calendar"]["path"])).session)
    sessions = book.window_sessions(calendar)
    codes = sorted({code for cross in heads.values() for code in cross})
    panels = {}
    columns = ["close", "high", "limit_up", "reference_price", "limit_ratio_at_open"]
    for code in codes:
        spec = manifest["panels"].get(code)
        if spec is None:
            continue
        frame = _prepare_panel(pd.read_parquet(root / spec["path"], columns=columns))
        panels[code] = frame
    targets = build_targets(sessions, heads, industries, panels)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "targets.json").write_text(json.dumps({"targets": targets, "promotion_approved": False}) + "\n", encoding="utf-8")
    import io
    chosen = sorted({book.canon(code) for row in targets for code in list(row["weights"]) + row["sell"]})
    full = {}
    for code in chosen:
        spec = manifest["panels"].get(code)
        if spec is None:
            continue
        full[code] = book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, spec))))
    instruments = pd.read_parquet(io.BytesIO(book._read_checked(root, manifest["instruments"])))
    actions = pd.read_parquet(io.BytesIO(book._read_checked(root, manifest["actions"])))
    benchmark = book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, manifest["benchmark"]))))
    from quantlab.data.contracts import MarketDataset
    dataset = MarketDataset(full, instruments, calendar, actions, benchmark, manifest=manifest)
    errors = dataset.validate("free_research")
    if errors:
        payload = {"reason": "projection_invalid", "errors": errors, "promotion_approved": False, "adoptable_performance": False}
        (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return
    result = book.replay_account(dataset, OUT / "targets.json", start=book.START, end=book.END, cost_mult=1.0)
    nav = result["nav"]
    dates = [str(day.date()) for day in nav.index]
    values = [float(item) for item in nav.tolist()]
    segments = {
        label: history.slice_performance(values, dates, start, end)
        for label, start, end in history.SEGMENTS
    }
    full_stats = history.performance(values, dates)
    decision = history.candidate_decision(
        full_stats["cagr"],
        full_stats["max_drawdown"],
        [segments[label]["cagr"] for label, _start, _end in history.SEGMENTS],
    )
    summary = {
        "account": "price_factor_reference",
        "components": ["prior_limit_up_premium", "broken_board_rate", "advancing_breadth"],
        "relative_strength_used": False,
        "concept_snapshot_used": False,
        "historical_simulation_with_current_theme_snapshot": False,
        "forward_decision": False,
        "promotion_approved": False,
        "adoptable_performance": False,
        "candidate_retained": decision["candidate_retained"],
        "window": ["2018-01-02", "2026-09-22"],
        "performance": full_stats,
        "segments": segments,
        "ending_nav": float(nav.iloc[-1]),
        "fills": result["summary"]["fills"],
        "decisions": len(targets),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
