"""Read-only check of a QuantDB L2 preview against the frozen 14-field contract.

A matching column name does not qualify a field.  Training stays closed until
each key has a declared unit, an ``available_at`` field, and its own raw
receipt.  This module does not download a dataset or fit a model.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from quantlab.data.l2_contract import L2_FEATURE_KEYS


def assess_l2_preview(
    *,
    columns: Sequence[str],
    preview_response_sha256: str | None = None,
) -> dict[str, object]:
    """Compare one preview header with the frozen QuantMind keys."""
    names = tuple(str(column) for column in columns)
    present = tuple(key for key in L2_FEATURE_KEYS if key in names)
    missing = tuple(key for key in L2_FEATURE_KEYS if key not in names)
    reasons: list[str] = []
    if missing:
        reasons.append("l2_fields_missing")
    if "available_at" not in names:
        reasons.append("available_at_not_declared")
    reasons.append("per_field_raw_receipt_missing")
    reasons.append("units_not_declared")
    return {
        "status": "not_qualified",
        "training_allowed": False,
        "account_result": False,
        "sample_count": 0,
        "frozen_field_count": len(L2_FEATURE_KEYS),
        "present_fields": present,
        "missing_fields": missing,
        "preview_response_sha256": preview_response_sha256,
        "reasons": tuple(reasons),
    }


def preview_column_names(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Read column names from a preview object without keeping row values."""
    columns = payload.get("columns")
    if isinstance(columns, (list, tuple)) and columns:
        return tuple(str(column) for column in columns)
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], Mapping):
        return tuple(str(column) for column in data[0])
    return ()
