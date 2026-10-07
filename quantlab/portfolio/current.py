"""Read the current versioned simulation account without inventing holdings.

This adapter intentionally refuses weight-only replays and never scans output
directories for a plausible-looking account. A holdings source is displayable
only when the current versioned result contains a complete simulation ledger.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

import pandas as pd

from quantlab.config import OUTPUT_DIR


def _load_result() -> dict | None:
    """Read the versioned current result; never fall back to ``results.pkl``."""

    latest = OUTPUT_DIR / "runs" / "latest.json"
    if not latest.exists():
        return None
    try:
        run_id = json.loads(latest.read_text())["run_id"]
        from quantlab.report.runs import load_run

        return {"result": load_run(OUTPUT_DIR / "runs" / run_id), "source": "versioned_replay"}
    except (ValueError, KeyError, OSError):
        return {"result": {"status": "unverified_artifacts"}, "source": "unverified_artifacts"}


def _complete_account_state(state: object) -> bool:
    """Require exact ledger fields before rendering a position or P&L row."""

    if not isinstance(state, Mapping):
        return False
    fields = ("quantities", "weights", "marks", "cost_basis", "entry_dates")
    if any(not isinstance(state.get(field), Mapping) for field in fields):
        return False
    quantities = state["quantities"]
    if not quantities:
        return False
    for code, quantity in quantities.items():
        try:
            is_positive = float(quantity) > 0
        except (TypeError, ValueError):
            return False
        if not is_positive or any(code not in state[field] for field in fields[1:]):
            return False
    return bool(state.get("date"))


def _public_code(code: object) -> str | None:
    """Translate a ledger id to a six-digit public quote key without mutating it."""

    value = str(code or "").strip().lower()
    for pattern in (r"(?:sh|sz|bj)\.(\d{6})", r"(\d{6})\.(?:sh|sz|bj)", r"(\d{6})"):
        match = re.fullmatch(pattern, value)
        if match:
            return match.group(1)
    return None


def snapshot(with_quotes: bool = True) -> list[dict]:
    """Return the current complete simulated ledger, or an honest empty list."""

    data = _load_result()
    if data is None:
        return []
    result = data["result"]
    states = result.get("account") or []
    if result.get("status") != "research_unvalidated" or not states or not _complete_account_state(states[-1]):
        return []
    state = states[-1]
    quantities = state["quantities"]
    quote_keys = {code: _public_code(code) for code in quantities}
    quotes = pd.DataFrame()
    if with_quotes:
        public_codes = [code for code in quote_keys.values() if code]
        if public_codes:
            try:
                from quantlab.data.realtime import batch_quotes

                quotes = batch_quotes(public_codes)
            except (ImportError, OSError, ValueError):
                pass
    rows = []
    for code, quantity in quantities.items():
        public_code = quote_keys.get(code)
        quote = quotes.loc[public_code].to_dict() if public_code and public_code in quotes.index else {}
        mark = state["marks"].get(code)
        cost = state["cost_basis"].get(code)
        rows.append({
            "code": str(code),
            "public_code": public_code,
            "name": quote.get("name", str(code)),
            "quantity": quantity,
            "weight": state["weights"].get(code),
            "account_type": "simulation",
            "as_of": str(state["date"]),
            "run_id": result.get("run_id"),
            "cost": cost,
            "entry_date": state["entry_dates"].get(code),
            "price": mark,
            "live_quote": quote.get("price"),
            "quote_time": quote.get("time"),
            "quote_source": quote.get("source"),
            "pre_close": quote.get("pre_close"),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "volume": quote.get("volume"),
            "amount": quote.get("amount"),
            "pct_chg": quote.get("pct_chg"),
        })
    return rows


def summary() -> dict:
    """Return current account metadata without promoting research or execution."""

    data = _load_result()
    out = {
        "exec_date": "",
        "nav": None,
        "drawdown": None,
        "daily": None,
        "status": "blocked_data",
        "account_type": "simulation",
        "displayable": False,
        "source_kind": data.get("source") if data else None,
    }
    if data is None:
        return out
    result = data["result"]
    out["status"] = result.get("status", "unverified_artifacts")
    out["run_id"] = result.get("run_id")
    states = result.get("account") or []
    if out["status"] != "research_unvalidated" or not states or not _complete_account_state(states[-1]):
        return out
    state = states[-1]
    out["displayable"] = True
    out["cash"] = float(state["cash"]) if state.get("cash") is not None else None
    out["receivables"] = float(state["receivables"]) if state.get("receivables") is not None else 0.0
    out["nav_mark"] = float(state["nav"]) if state.get("nav") is not None else None
    out["account_as_of"] = str(state["date"])
    out["holdings_count"] = len(state["quantities"])
    nav = result.get("nav")
    if nav is not None and len(nav):
        out["nav"] = float(nav.iloc[-1])
        out["exec_date"] = str(nav.index[-1].date())
        drawdown = result.get("drawdown")
        if drawdown is not None and len(drawdown):
            out["drawdown"] = float(drawdown.iloc[-1])
    daily_return = result.get("daily_return")
    if daily_return is not None and len(daily_return) >= 2:
        out["daily"] = float(daily_return.iloc[-1])
    trades = result.get("trades") or []
    last_trade = trades[-1] if trades else {}
    out["regime_trend"] = last_trade.get("regime", {}).get("trend")
    out["gross_scale"] = float(last_trade.get("risk", {}).get("gross_scale", 1.0)) if trades else 1.0
    out["risk_reasons"] = list(last_trade.get("risk", {}).get("reasons", []))
    return out


def ytd_return() -> float | None:
    """Return the current ledger's year-to-date return only when it is displayable."""

    data = _load_result()
    if data is None:
        return None
    result = data["result"]
    states = result.get("account") or []
    if result.get("status") != "research_unvalidated" or not states or not _complete_account_state(states[-1]):
        return None
    nav = result.get("nav")
    if nav is None or len(nav) < 2:
        return None
    nav = pd.Series(nav).sort_index()
    year_start = pd.Timestamp(year=nav.index[-1].year, month=1, day=1)
    base = nav[nav.index < year_start]
    if not len(base):
        return None
    opening_nav = float(base.iloc[-1])
    return float(nav.iloc[-1] / opening_nav - 1.0) if opening_nav else None
