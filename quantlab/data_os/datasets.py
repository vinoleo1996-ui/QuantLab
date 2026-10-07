"""金层四套数据集。训练集只接收 research_grade 的行，标签与特征分开。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantlab.data_os.features import evaluate_formula, price_features
from quantlab.data_os.pit import visible


def _version_path(root: Path, name: str, version: str) -> Path:
    path = Path(root) / "gold" / name / f"dataset_version={version}" / "part.parquet"
    if path.exists():
        raise FileExistsError(f"金层版本已存在，拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_parquet(path, index=False)
    return path


def research_base(root: Path, version: str, tables: dict[str, pd.DataFrame]) -> Path:
    pieces = []
    for name, frame in tables.items():
        if frame is None or frame.empty:
            continue
        marked = frame.copy()
        marked["table_name"] = name
        pieces.append(marked)
    payload = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return _write(payload, _version_path(root, "research_base", version))


def ml_dataset(root: Path, version: str, market: pd.DataFrame, fundamentals: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    features = price_features(market)
    if features.empty:
        frame = pd.DataFrame()
        _write(frame, _version_path(root, "ml_dataset", version))
        return frame
    features = _attach_asof(features, fundamentals, "concept", "value", "fund_")
    features = _attach_macro(features, macro)
    countries = market[["instrument_id", "symbol"]].drop_duplicates()
    # 市场由 instrument_id 前缀决定。A 股标签用 t+1 到 t+2，避免把 t 日收盘当成可成交价。
    features["market"] = features["instrument_id"].str.slice(0, 2)
    grouped = features.groupby("instrument_id", group_keys=False)
    entry = grouped["close"].shift(-1)
    later = grouped["close"].shift(-2)
    features["y"] = later / entry - 1
    us = features["market"] == "US"
    features.loc[us, "y"] = grouped["close"].shift(-1) / features["close"] - 1
    features.loc[us, "label_spec"] = "close_t1_over_close_t_minus_1"
    features.loc[~us, "label_spec"] = "close_t2_over_close_t1_minus_1"
    del countries
    learn = features.dropna(subset=["y"])
    _write(learn, _version_path(root, "ml_dataset", version))
    return learn


def alpha_dataset(root: Path, version: str, features: pd.DataFrame, formulas: dict[str, str]) -> pd.DataFrame:
    rows = []
    if not features.empty:
        forward = features.groupby("instrument_id")["y"].transform(lambda series: series) if "y" in features.columns else pd.Series(pd.NA, index=features.index)
        for alpha_id, formula in formulas.items():
            values = evaluate_formula(features, formula)
            for index, value in values.items():
                rows.append({
                    "instrument_id": features.at[index, "instrument_id"],
                    "trade_date": features.at[index, "trade_date"],
                    "alpha_id": alpha_id,
                    "alpha_value": value,
                    "formula": formula,
                    "forward_return": forward.at[index] if len(forward) else pd.NA,
                })
    frame = pd.DataFrame(rows)
    _write(frame, _version_path(root, "alpha_dataset", version))
    return frame


def agent_dataset(root: Path, version: str, market: pd.DataFrame, fundamentals: pd.DataFrame, macro: pd.DataFrame, as_of) -> pd.DataFrame:
    prices = visible(market[market["research_grade"].astype(bool)], as_of) if not market.empty else market
    facts = visible(fundamentals[fundamentals["research_grade"].astype(bool)], as_of) if fundamentals is not None and not fundamentals.empty else fundamentals
    macros = visible(macro[macro["research_grade"].astype(bool)], as_of) if macro is not None and not macro.empty else macro
    rows = []
    if not prices.empty:
        last = prices.sort_values("trade_date").groupby("instrument_id").tail(1)
        for _, row in last.iterrows():
            own_facts = facts[facts["instrument_id"] == row["instrument_id"]] if facts is not None and not facts.empty else pd.DataFrame()
            rows.append({
                "as_of": str(pd.Timestamp(as_of)),
                "instrument_id": row["instrument_id"],
                "price_context": json.dumps({"trade_date": str(row["trade_date"]), "close": row["close"], "source": row["source"]}, default=str),
                "fundamental_context": json.dumps(own_facts[["concept", "value", "available_time"]].assign(available_time=lambda item: item["available_time"].astype(str)).to_dict("records"), default=str) if not own_facts.empty else "[]",
                "macro_context": json.dumps(macros.assign(available_time=lambda item: item["available_time"].astype(str), observation_date=lambda item: item["observation_date"].astype(str))[["series_id", "observation_date", "value", "available_time"]].to_dict("records"), default=str) if macros is not None and not macros.empty else "[]",
                "news_context": "[]",
                "agent_observation": None,
                "decision": None,
                "outcome": None,
            })
    frame = pd.DataFrame(rows)
    _write(frame, _version_path(root, "agent_dataset", version))
    return frame


def _attach_asof(features: pd.DataFrame, facts: pd.DataFrame, key: str, value: str, prefix: str) -> pd.DataFrame:
    if facts is None or facts.empty:
        return features
    usable = facts[facts["research_grade"].astype(bool)].copy()
    if usable.empty:
        return features
    usable["available_time"] = pd.to_datetime(usable["available_time"], utc=True)
    features = features.copy()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    pieces = [features]
    for concept, group in usable.groupby(key):
        column = f"{prefix}{concept}"
        group = group.sort_values("available_time")[["instrument_id", "available_time", value]].rename(columns={value: column})
        merged = []
        for instrument_id, bars in features.groupby("instrument_id"):
            facts_one = group[group["instrument_id"] == instrument_id]
            bars = bars.sort_values("available_time")
            if facts_one.empty:
                bars[column] = pd.NA
                merged.append(bars)
                continue
            joined = pd.merge_asof(bars, facts_one.drop(columns=["instrument_id"]), on="available_time", direction="backward")
            merged.append(joined)
        features = pd.concat(merged, ignore_index=True)
        pieces = [features]
    return pieces[0]


def _attach_macro(features: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    if macro is None or macro.empty:
        return features
    usable = macro[macro["research_grade"].astype(bool)].copy()
    if usable.empty:
        return features
    usable["available_time"] = pd.to_datetime(usable["available_time"], utc=True)
    features = features.copy()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = features.sort_values("available_time")
    for series_id, group in usable.groupby("series_id"):
        column = f"macro_{series_id}"
        group = group.sort_values("available_time")[["available_time", "value"]].rename(columns={"value": column})
        features = pd.merge_asof(features, group, on="available_time", direction="backward")
    return features
