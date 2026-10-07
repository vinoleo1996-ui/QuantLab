"""Fail-closed adapter for the fixed Grok Bot paper-account ledger.

This module deliberately reads only the four named artifacts below.  It never
looks at QuantLab's output directory, ``latest.json``, weights, or replay
pickles.  The ledger is a paper simulation, not a broker account or a
research-approved strategy; callers must preserve that boundary in their UI.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any


SNAPSHOT_NAME = "grok_paper_ledger"
PAPER_STATUS = "paper_simulation_unverified"
POSITION_COUNT = 50
DEFAULT_LEDGER_ROOT = Path.home() / "grokbot_isolated" / "paper" / "ledger"

_FILES = {
    "current_positions": "current_positions.csv",
    "current_state": "current_state.json",
    "initial_positions": "initial_positions_2026-09-22.csv",
    "initial_state": "initial_state_2026-09-22.json",
}
_CODE = re.compile(r"(?:sh|sz|bj)\.\d{6}")
_CURRENT_FIELDS = (
    "code",
    "name",
    "implied_shares",
    "original_shares",
    "last_price",
    "market_value",
    "weight",
    "cost_basis_total",
    "unrealized_pnl",
    "entry_date",
    "entry_price",
)
_INITIAL_FIELDS = (
    "code",
    "name",
    "board",
    "industry_csrc2012",
    "shares",
    "entry_date",
    "entry_price",
    "market_value",
    "cost_basis_total",
)


class LedgerValidationError(ValueError):
    """An externally supplied ledger is missing, malformed, or inconsistent."""


def _ledger_root(root: Path | str | None) -> Path:
    return Path(DEFAULT_LEDGER_ROOT if root is None else root).expanduser()


def _provenance(root: Path) -> dict[str, Any]:
    return {
        "source_kind": SNAPSHOT_NAME,
        "ledger_root": str(root),
        "files": {
            key: {"path": str(root / filename), "sha256": None}
            for key, filename in _FILES.items()
        },
    }


def _invalid(root: Path, provenance: dict[str, Any], reason: str) -> dict[str, Any]:
    """Return no positions when any part of the named ledger cannot be trusted."""

    summary = {
        "snapshot_name": SNAPSHOT_NAME,
        "source_kind": SNAPSHOT_NAME,
        "status": "unavailable",
        "account_type": "simulation",
        "displayable": False,
        "holdings_count": 0,
    }
    return {
        "snapshot_name": SNAPSHOT_NAME,
        "source_kind": SNAPSHOT_NAME,
        "status": "unavailable",
        "account_type": "simulation",
        "displayable": False,
        "as_of": None,
        "rows": [],
        "summary": summary,
        "provenance": provenance,
        "source_paths": {key: value["path"] for key, value in provenance["files"].items()},
        "source_sha256": {key: value["sha256"] for key, value in provenance["files"].items()},
        "errors": [reason],
    }


def _read_sources(root: Path, provenance: dict[str, Any]) -> dict[str, bytes]:
    raw: dict[str, bytes] = {}
    for key, filename in _FILES.items():
        path = root / filename
        try:
            value = path.read_bytes()
        except FileNotFoundError as exc:
            raise LedgerValidationError(f"source_missing:{key}") from exc
        except OSError as exc:
            raise LedgerValidationError(f"source_unreadable:{key}") from exc
        if not value:
            raise LedgerValidationError(f"source_empty:{key}")
        provenance["files"][key]["sha256"] = hashlib.sha256(value).hexdigest()
        raw[key] = value
    return raw


def _csv_rows(raw: bytes, *, label: str, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8").lstrip("\ufeff")
    except UnicodeDecodeError as exc:
        raise LedgerValidationError(f"{label}_decode") from exc
    try:
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames
        if not fieldnames or any(not isinstance(name, str) or not name for name in fieldnames):
            raise LedgerValidationError(f"{label}_schema:header")
        if len(set(fieldnames)) != len(fieldnames) or any(field not in fieldnames for field in fields):
            raise LedgerValidationError(f"{label}_schema:required_fields")
        rows = list(reader)
    except csv.Error as exc:
        raise LedgerValidationError(f"{label}_csv") from exc
    if any(None in row for row in rows):
        raise LedgerValidationError(f"{label}_schema:column_count")
    if len(rows) != POSITION_COUNT:
        raise LedgerValidationError(f"{label}_count:{len(rows)}")
    return rows


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerValidationError(f"{label}_decode") from exc
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise LedgerValidationError(f"{label}_json") from exc
    if not isinstance(value, Mapping):
        raise LedgerValidationError(f"{label}_schema:object")
    return dict(value)


def _require_fields(
    record: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    label: str,
    allow_blank: tuple[str, ...] = (),
) -> None:
    missing = [field for field in fields if field not in record or record[field] is None]
    if missing:
        raise LedgerValidationError(f"{label}_schema:missing_{missing[0]}")
    blank = [
        field
        for field in fields
        if field not in allow_blank and isinstance(record[field], str) and not record[field].strip()
    ]
    if blank:
        raise LedgerValidationError(f"{label}_schema:blank_{blank[0]}")


def _number(value: object, *, label: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise LedgerValidationError(f"{label}_schema:number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LedgerValidationError(f"{label}_schema:number") from exc
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        raise LedgerValidationError(f"{label}_schema:number")
    return number


def _date(value: object, *, label: str) -> tuple[str, date]:
    if not isinstance(value, str):
        raise LedgerValidationError(f"{label}_schema:date")
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError as exc:
        raise LedgerValidationError(f"{label}_schema:date") from exc
    return parsed.isoformat(), parsed


def _code(value: object, *, label: str) -> str:
    code = str(value or "").strip().lower()
    if not _CODE.fullmatch(code):
        raise LedgerValidationError(f"{label}_schema:code")
    return code


def _close(left: float, right: float, *, label: str) -> None:
    # Ledger amounts are stored to cents, while some cost values retain more precision.
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=0.02):
        raise LedgerValidationError(f"{label}_inconsistent")


def _current_position(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    _require_fields(record, _CURRENT_FIELDS, label=label)
    code = _code(record["code"], label=label)
    quantity = _number(record["original_shares"], label=f"{label}:original_shares", positive=True)
    implied_shares = _number(record["implied_shares"], label=f"{label}:implied_shares", positive=True)
    _close(quantity, implied_shares, label=f"{label}:share_count")
    mark = _number(record["last_price"], label=f"{label}:last_price", positive=True)
    market_value = _number(record["market_value"], label=f"{label}:market_value", positive=True)
    _close(market_value, quantity * mark, label=f"{label}:market_value")
    cost_basis_total = _number(record["cost_basis_total"], label=f"{label}:cost_basis_total", positive=True)
    unrealized_pnl = _number(record["unrealized_pnl"], label=f"{label}:unrealized_pnl")
    _close(unrealized_pnl, market_value - cost_basis_total, label=f"{label}:unrealized_pnl")
    entry_date, _ = _date(record["entry_date"], label=f"{label}:entry_date")
    entry_price = _number(record["entry_price"], label=f"{label}:entry_price", positive=True)
    weight = _number(record["weight"], label=f"{label}:weight", nonnegative=True)
    name = str(record["name"]).strip()
    if not name:
        raise LedgerValidationError(f"{label}_schema:name")
    return {
        "code": code,
        "name": name,
        "quantity": quantity,
        "mark": mark,
        "market_value": market_value,
        "weight": weight,
        "cost_basis_total": cost_basis_total,
        "entry_date": entry_date,
        "entry_price": entry_price,
    }


def _initial_position(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    # One source record currently has no CSRC label.  The field itself must be
    # present, but an absent label is returned as explicit missing metadata,
    # rather than inventing an industry or hiding all 50 holdings.
    _require_fields(record, _INITIAL_FIELDS, label=label, allow_blank=("industry_csrc2012",))
    code = _code(record["code"], label=label)
    quantity = _number(record["shares"], label=f"{label}:shares", positive=True)
    entry_price = _number(record["entry_price"], label=f"{label}:entry_price", positive=True)
    market_value = _number(record["market_value"], label=f"{label}:market_value", positive=True)
    _close(market_value, quantity * entry_price, label=f"{label}:market_value")
    cost_basis_total = _number(record["cost_basis_total"], label=f"{label}:cost_basis_total", positive=True)
    entry_date, _ = _date(record["entry_date"], label=f"{label}:entry_date")
    name = str(record["name"]).strip()
    board = str(record["board"]).strip()
    industry = str(record["industry_csrc2012"]).strip()
    if not name or not board:
        raise LedgerValidationError(f"{label}_schema:identity")
    return {
        "code": code,
        "name": name,
        "quantity": quantity,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "market_value": market_value,
        "cost_basis_total": cost_basis_total,
        "board": board,
        "build_time_industry": industry or None,
        "build_time_industry_status": "recorded" if industry else "missing_in_build_ledger",
    }


def _unique_by_code(positions: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed = {position["code"]: position for position in positions}
    if len(indexed) != POSITION_COUNT:
        raise LedgerValidationError(f"{label}_schema:duplicate_code")
    return indexed


def _require_position_list(state: Mapping[str, Any], *, label: str, parser) -> list[dict[str, Any]]:
    positions = state.get("positions")
    if not isinstance(positions, list):
        raise LedgerValidationError(f"{label}_schema:positions")
    if len(positions) != POSITION_COUNT:
        raise LedgerValidationError(f"{label}_count:{len(positions)}")
    parsed: list[dict[str, Any]] = []
    for index, position in enumerate(positions):
        if not isinstance(position, Mapping):
            raise LedgerValidationError(f"{label}_schema:position_{index}")
        parsed.append(parser(position, label=f"{label}:position_{index}"))
    _unique_by_code(parsed, label=label)
    return parsed


def _require_exact_count(value: object, *, label: str) -> None:
    count = _number(value, label=label, positive=True)
    if not count.is_integer() or int(count) != POSITION_COUNT:
        raise LedgerValidationError(f"{label}_count:{value}")


def _current_state(state: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("account", "as_of", "nav", "cash", "invested", "n_positions", "positions")
    _require_fields(state, fields, label="current_state")
    account = str(state["account"]).strip()
    if not account:
        raise LedgerValidationError("current_state_schema:account")
    as_of, as_of_date = _date(state["as_of"], label="current_state:as_of")
    nav = _number(state["nav"], label="current_state:nav", positive=True)
    cash = _number(state["cash"], label="current_state:cash", nonnegative=True)
    invested = _number(state["invested"], label="current_state:invested", positive=True)
    _close(nav, cash + invested, label="current_state:nav")
    _require_exact_count(state["n_positions"], label="current_state:n_positions")
    positions = _require_position_list(state, label="current_state", parser=_current_position)
    _close(sum(position["market_value"] for position in positions), invested, label="current_state:invested")
    return {
        "account": account,
        "as_of": as_of,
        "as_of_date": as_of_date,
        "nav": nav,
        "cash": cash,
        "invested": invested,
        "positions": positions,
    }


def _initial_state(state: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "account",
        "as_of",
        "capital",
        "cash",
        "invested_value",
        "nav",
        "n_positions",
        "N",
        "positions",
        "source",
        "source_sha256",
    )
    _require_fields(state, fields, label="initial_state")
    account = str(state["account"]).strip()
    if not account:
        raise LedgerValidationError("initial_state_schema:account")
    as_of, as_of_date = _date(state["as_of"], label="initial_state:as_of")
    capital = _number(state["capital"], label="initial_state:capital", positive=True)
    cash = _number(state["cash"], label="initial_state:cash", nonnegative=True)
    invested = _number(state["invested_value"], label="initial_state:invested_value", positive=True)
    nav = _number(state["nav"], label="initial_state:nav", positive=True)
    _close(nav, cash + invested, label="initial_state:nav")
    _require_exact_count(state["n_positions"], label="initial_state:n_positions")
    _require_exact_count(state["N"], label="initial_state:N")
    positions = _require_position_list(state, label="initial_state", parser=_initial_position)
    _close(sum(position["market_value"] for position in positions), invested, label="initial_state:invested_value")
    source = str(state["source"]).strip()
    declared_sha256 = str(state["source_sha256"]).strip().lower()
    if not source or not re.fullmatch(r"[0-9a-f]{64}", declared_sha256):
        raise LedgerValidationError("initial_state_schema:source_provenance")
    return {
        "account": account,
        "as_of": as_of,
        "as_of_date": as_of_date,
        "capital": capital,
        "cash": cash,
        "invested": invested,
        "nav": nav,
        "positions": positions,
        "declared_source": source,
        "declared_source_sha256": declared_sha256,
    }


def _same_positions(left: Mapping[str, dict[str, Any]], right: Mapping[str, dict[str, Any]], *, label: str) -> None:
    if set(left) != set(right):
        raise LedgerValidationError(f"{label}_codes_mismatch")
    for code, first in left.items():
        second = right[code]
        for field in ("quantity", "market_value", "cost_basis_total", "entry_price"):
            _close(first[field], second[field], label=f"{label}:{code}:{field}")
        if first["name"] != second["name"] or first["entry_date"] != second["entry_date"]:
            raise LedgerValidationError(f"{label}_identity_mismatch:{code}")


def _build_snapshot(
    *,
    current_rows: list[dict[str, Any]],
    current_state: dict[str, Any],
    initial_rows: list[dict[str, Any]],
    initial_state: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    current_csv = _unique_by_code(current_rows, label="current_positions")
    current_json = _unique_by_code(current_state["positions"], label="current_state")
    initial_csv = _unique_by_code(initial_rows, label="initial_positions")
    initial_json = _unique_by_code(initial_state["positions"], label="initial_state")
    _same_positions(current_csv, current_json, label="current_ledger")
    _same_positions(initial_csv, initial_json, label="initial_ledger")
    if current_state["account"] != initial_state["account"]:
        raise LedgerValidationError("ledger_account_mismatch")
    if initial_state["as_of_date"] > current_state["as_of_date"]:
        raise LedgerValidationError("ledger_date_order_invalid")
    if set(current_csv) != set(initial_csv):
        raise LedgerValidationError("ledger_build_time_join_missing")

    rows: list[dict[str, Any]] = []
    for position in current_rows:
        initial = initial_csv[position["code"]]
        # This paper ledger is an initial-build snapshot.  Requiring these fields
        # to reconcile prevents silently displaying a different portfolio.
        _close(position["quantity"], initial["quantity"], label=f"ledger_quantity:{position['code']}")
        _close(position["cost_basis_total"], initial["cost_basis_total"], label=f"ledger_cost:{position['code']}")
        if position["entry_date"] != initial["entry_date"]:
            raise LedgerValidationError(f"ledger_entry_date_mismatch:{position['code']}")
        rows.append({
            "code": position["code"],
            "public_code": position["code"].split(".", 1)[1],
            "name": position["name"],
            "build_time_name": initial["name"],
            "quantity": position["quantity"],
            "weight": position["weight"],
            "account_type": "simulation",
            "account_status": PAPER_STATUS,
            "source_kind": SNAPSHOT_NAME,
            "as_of": current_state["as_of"],
            "run_id": current_state["account"],
            "ledger_id": current_state["account"],
            "cost": position["cost_basis_total"] / position["quantity"],
            "cost_basis_total": position["cost_basis_total"],
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "price": position["mark"],
            "mark": position["mark"],
            "mark_source": SNAPSHOT_NAME,
            "mark_as_of": current_state["as_of"],
            "market_value": position["market_value"],
            "build_time_industry": initial["build_time_industry"],
            "industry": initial["build_time_industry"],
            "build_time_industry_status": initial["build_time_industry_status"],
            "industry_as_of": initial_state["as_of"],
            "board": initial["board"],
            # A caller may overlay a separately acquired public quote.  This
            # adapter never turns an accounting mark into a fake real-time quote.
            "live_quote": None,
            "quote_time": None,
            "quote_source": None,
            "pre_close": None,
            "open": None,
            "high": None,
            "low": None,
            "volume": None,
            "amount": None,
            "pct_chg": None,
        })

    provenance["declared_initial_source"] = {
        "path": initial_state["declared_source"],
        "sha256": initial_state["declared_source_sha256"],
    }
    summary = {
        "snapshot_name": SNAPSHOT_NAME,
        "source_kind": SNAPSHOT_NAME,
        "status": PAPER_STATUS,
        "account_type": "simulation",
        "displayable": True,
        "account": current_state["account"],
        "run_id": current_state["account"],
        "account_as_of": current_state["as_of"],
        "build_as_of": initial_state["as_of"],
        "cash": current_state["cash"],
        "invested": current_state["invested"],
        "nav_mark": current_state["nav"],
        "initial_capital": initial_state["capital"],
        "holdings_count": POSITION_COUNT,
    }
    return {
        "snapshot_name": SNAPSHOT_NAME,
        "source_kind": SNAPSHOT_NAME,
        "status": PAPER_STATUS,
        "account_type": "simulation",
        "displayable": True,
        "as_of": current_state["as_of"],
        "rows": rows,
        "summary": summary,
        "provenance": provenance,
        "source_paths": {key: value["path"] for key, value in provenance["files"].items()},
        "source_sha256": {key: value["sha256"] for key, value in provenance["files"].items()},
        "errors": [],
    }


def load_grok_paper_ledger(root: Path | str | None = None) -> dict[str, Any]:
    """Load only the complete, named Grok paper ledger, or return an empty state.

    ``root`` is an injectable ledger directory for tests.  The production
    default is exactly ``~/grokbot_isolated/paper/ledger``.  No directory scan,
    replay result, or weight-only fallback is permitted.
    """

    ledger_root = _ledger_root(root)
    provenance = _provenance(ledger_root)
    try:
        raw = _read_sources(ledger_root, provenance)
        current_rows = [
            _current_position(row, label=f"current_positions:row_{index}")
            for index, row in enumerate(_csv_rows(raw["current_positions"], label="current_positions", fields=_CURRENT_FIELDS))
        ]
        initial_rows = [
            _initial_position(row, label=f"initial_positions:row_{index}")
            for index, row in enumerate(_csv_rows(raw["initial_positions"], label="initial_positions", fields=_INITIAL_FIELDS))
        ]
        current_state = _current_state(_json_object(raw["current_state"], label="current_state"))
        initial_state = _initial_state(_json_object(raw["initial_state"], label="initial_state"))
        return _build_snapshot(
            current_rows=current_rows,
            current_state=current_state,
            initial_rows=initial_rows,
            initial_state=initial_state,
            provenance=provenance,
        )
    except LedgerValidationError as exc:
        return _invalid(ledger_root, provenance, str(exc))
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as exc:
        return _invalid(ledger_root, provenance, f"ledger_invalid:{type(exc).__name__}")


def snapshot(root: Path | str | None = None) -> list[dict[str, Any]]:
    """Compatibility helper returning rows only from a validated Grok ledger."""

    return load_grok_paper_ledger(root)["rows"]


def summary(root: Path | str | None = None) -> dict[str, Any]:
    """Compatibility helper returning the paper-ledger account boundary."""

    return load_grok_paper_ledger(root)["summary"]
