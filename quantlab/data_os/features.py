"""特征只用当时及更早的行情。标签不在这一层计算。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


RET_WINDOWS = (1, 2, 5, 10, 20, 60, 120, 250)
MA_WINDOWS = (5, 10, 20, 60, 120, 250)


def _by_instrument(frame: pd.DataFrame):
    return frame.groupby("instrument_id", group_keys=False)


def price_features(market: pd.DataFrame) -> pd.DataFrame:
    grade = market[market["research_grade"].astype(bool)].copy()
    if grade.empty:
        return pd.DataFrame(columns=["instrument_id", "trade_date"])
    grade["trade_date"] = pd.to_datetime(grade["trade_date"])
    grade = grade.sort_values(["instrument_id", "trade_date"])
    grouped = _by_instrument(grade)
    out = grade[["instrument_id", "trade_date", "available_time", "close", "high", "low", "volume", "amount", "vwap", "turnover"]].copy()
    for window in RET_WINDOWS:
        previous = grouped["close"].shift(window)
        out[f"RET_{window}"] = grade["close"] / previous - 1
    for window in MA_WINDOWS:
        out[f"MA{window}"] = grouped["close"].transform(lambda series: series.rolling(window, min_periods=window).mean())
    out["EMA12"] = grouped["close"].transform(lambda series: series.ewm(span=12, adjust=False).mean())
    out["EMA26"] = grouped["close"].transform(lambda series: series.ewm(span=26, adjust=False).mean())
    out["MACD"] = out["EMA12"] - out["EMA26"]
    for window in (5, 10, 20):
        out[f"STD{window}"] = grouped["close"].transform(lambda series: series.rolling(window, min_periods=window).std())
    previous_close = grouped["close"].shift(1)
    true_range = pd.concat([
        (grade["high"] - grade["low"]),
        (grade["high"] - previous_close).abs(),
        (grade["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR14"] = true_range.groupby(grade["instrument_id"]).transform(lambda series: series.rolling(14, min_periods=14).mean())
    ratio_hl = (grade["high"] / grade["low"]).where(grade["low"] > 0)
    log_hl = pd.Series(np.log(ratio_hl.astype(float)), index=grade.index)
    out["PARKINSON20"] = (log_hl ** 2).groupby(grade["instrument_id"]).transform(
        lambda series: (series.rolling(20, min_periods=20).mean() / (4 * math.log(2))).pow(0.5)
    )
    ratio_co = (grade["close"] / grade["open"]).where(grade["open"] > 0)
    log_co = pd.Series(np.log(ratio_co.astype(float)), index=grade.index)
    gk = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2
    out["GARMAN_KLASS20"] = gk.groupby(grade["instrument_id"]).transform(
        lambda series: series.rolling(20, min_periods=20).mean().pow(0.5)
    )
    volume_ma = grouped["volume"].transform(lambda series: series.rolling(20, min_periods=20).mean())
    volume_std = grouped["volume"].transform(lambda series: series.rolling(20, min_periods=20).std())
    out["VOL_RATIO"] = grade["volume"] / volume_ma
    out["VOL_ZSCORE"] = (grade["volume"] - volume_ma) / volume_std
    vwap = pd.to_numeric(grade["vwap"], errors="coerce")
    out["VWAP_DISTANCE"] = grade["close"] / vwap.where(vwap > 0) - 1
    amount_ma = grouped["amount"].transform(lambda series: series.rolling(20, min_periods=20).mean())
    amount_std = grouped["amount"].transform(lambda series: series.rolling(20, min_periods=20).std())
    out["AMOUNT_MA20"] = amount_ma
    out["AMOUNT_ZSCORE"] = (grade["amount"] - amount_ma) / amount_std
    direction = grouped["close"].diff().map(lambda value: 0.0 if pd.isna(value) else (1.0 if value > 0 else (-1.0 if value < 0 else 0.0)))
    out["OBV"] = (direction * grade["volume"]).groupby(grade["instrument_id"]).cumsum()
    ret1 = out["RET_1"].abs()
    out["ILLIQUIDITY"] = ret1 / grade["amount"].where(grade["amount"] > 0)
    out["TURNOVER"] = grade["turnover"]
    out["AMOUNT"] = grade["amount"]
    for column in ("RET_1", "VOL_RATIO", "TURNOVER"):
        if column in out.columns:
            out[f"{column}_RANK"] = out.groupby("trade_date")[column].rank(pct=True)
    return out


def evaluate_formula(features: pd.DataFrame, formula: str) -> pd.Series:
    """只接受 rank / ts_mean / corr / 除法，不执行任意代码。"""
    series = _parse("".join(formula.split()), features)
    if not isinstance(series, pd.Series):
        raise ValueError(f"公式没有产出序列: {formula}")
    return series


def _parse(text: str, features: pd.DataFrame):
    if text in features.columns:
        return features[text]
    if text.startswith("-"):
        return -_parse(text[1:], features)
    if text.startswith("rank(") and text.endswith(")"):
        return _parse(text[5:-1], features).groupby(features["trade_date"]).rank(pct=True)
    if text.startswith("ts_mean(") and text.endswith(")"):
        body, window = _split_last_arg(text[8:-1])
        return _parse(body, features).groupby(features["instrument_id"]).transform(
            lambda series: series.rolling(int(window), min_periods=int(window)).mean()
        )
    if text.startswith("corr(") and text.endswith(")"):
        left, rest = _split_first_arg(text[5:-1])
        right, window = _split_last_arg(rest)
        return _rolling_corr(_parse(left, features), _parse(right, features), features["instrument_id"], int(window))
    if "/" in text and _top_level(text, "/"):
        left, right = _split_top(text, "/")
        denominator = _parse(right, features)
        denominator = denominator.where(pd.to_numeric(denominator, errors="coerce") != 0)
        return _parse(left, features) / denominator
    raise ValueError(f"不支持的公式: {text}")


def _rolling_corr(left: pd.Series, right: pd.Series, instrument: pd.Series, window: int) -> pd.Series:
    out = pd.Series(index=left.index, dtype="float64")
    for _, index in left.groupby(instrument).groups.items():
        out.loc[index] = left.loc[index].rolling(window, min_periods=window).corr(right.loc[index])
    return out


def _top_level(text: str, operator: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == operator and depth == 0:
            return True
    return False


def _split_top(text: str, operator: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == operator and depth == 0:
            return text[:index], text[index + 1:]
    raise ValueError(text)


def _split_last_arg(text: str) -> tuple[str, str]:
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
        elif char == "," and depth == 0:
            return text[:index], text[index + 1:]
    raise ValueError(text)


def _split_first_arg(text: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            return text[:index], text[index + 1:]
    raise ValueError(text)
