"""同一天多个源的收盘价不一致时，留下选中源、偏差和冲突标记。"""

from __future__ import annotations

import pandas as pd


MARKET_PRIORITY = {
    "baostock": 0,
    "yahoo": 0,
    "sec": 0,
    "fred": 0,
    "akshare": 1,
    "stooq": 1,
    "tencent": 2,
    "eastmoney": 3,
}


def resolve_market(frames: list[pd.DataFrame], tolerance: float = 1e-4) -> tuple[pd.DataFrame, pd.DataFrame]:
    stacked = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)
    if stacked.empty:
        return stacked, pd.DataFrame(columns=["instrument_id", "trade_date", "field", "selected_source", "conflict_value"])
    stacked["priority"] = stacked["source"].map(lambda source: MARKET_PRIORITY.get(str(source), 9))
    chosen_rows = []
    conflicts = []
    keys = ["instrument_id", "trade_date"]
    for key, group in stacked.groupby(keys, dropna=False):
        group = group.sort_values(["priority", "source"])
        valid = group[group.apply(_price_ok, axis=1)]
        pool = valid if not valid.empty else group
        selected = pool.iloc[0]
        closes = []
        for _, other in group.iterrows():
            if other["source"] == selected["source"] or pd.isna(other.get("close")) or pd.isna(selected.get("close")):
                continue
            base = float(selected["close"])
            deviation = abs(float(other["close"]) - base) / abs(base) if base else 1.0
            if deviation > tolerance:
                closes.append(deviation)
                conflicts.append({
                    "instrument_id": key[0],
                    "trade_date": key[1],
                    "field": "close",
                    "selected_source": selected["source"],
                    "other_source": other["source"],
                    "selected_value": base,
                    "other_value": float(other["close"]),
                    "conflict_value": deviation,
                })
        row = selected.drop(labels=["priority"]).copy()
        row["selected_source"] = selected["source"]
        row["conflict"] = bool(closes)
        row["conflict_value"] = max(closes) if closes else 0.0
        chosen_rows.append(row)
    return pd.DataFrame(chosen_rows), pd.DataFrame(conflicts)


def _price_ok(row: pd.Series) -> bool:
    try:
        high, low = float(row["high"]), float(row["low"])
        open_, close = float(row["open"]), float(row["close"])
        volume = float(row["volume"])
    except (TypeError, ValueError):
        return False
    return volume >= 0 and high >= max(open_, close, low) and low <= min(open_, close, high) and close > 0
