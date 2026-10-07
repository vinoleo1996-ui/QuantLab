"""质量分是完整、及时、一致、来源、时点五项的乘积。时点缺失则整行不能进训练集。"""

from __future__ import annotations

import pandas as pd


PRICE_FIELDS = ["open", "high", "low", "close", "volume"]


def ohlc_consistent(row: pd.Series) -> bool:
    try:
        open_, high, low, close = (float(row[name]) for name in ("open", "high", "low", "close"))
        volume = float(row["volume"])
    except (TypeError, ValueError):
        return False
    if min(open_, high, low, close) <= 0 or volume < 0:
        return False
    return high + 1e-9 >= max(open_, close, low) and low - 1e-9 <= min(open_, close, high)


def quality_score(completeness: float, timeliness: float, consistency: float, provenance: float, pit: float) -> float:
    parts = [completeness, timeliness, consistency, provenance, pit]
    if any(part < 0 or part > 1 for part in parts):
        raise ValueError("质量分分量必须在 0 到 1")
    score = 1.0
    for part in parts:
        score *= part
    return float(score)


def annotate_market(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    numbers = out[PRICE_FIELDS].apply(pd.to_numeric, errors="coerce")
    completeness = numbers.notna().sum(axis=1) / len(PRICE_FIELDS)
    high_enough = numbers["high"] + 1e-9 >= numbers[["open", "close", "low"]].max(axis=1)
    low_enough = numbers["low"] - 1e-9 <= numbers[["open", "close", "high"]].min(axis=1)
    consistency = (numbers.min(axis=1) > 0) & (numbers["volume"] >= 0) & high_enough & low_enough
    pit = out["available_time"].notna()
    if "pit_valid" in out.columns:
        pit = pit & out["pit_valid"].map(lambda value: bool(value) if pd.notna(value) else False)
    provenance = out["source"].notna() & out["ingested_at"].notna() if "ingested_at" in out.columns else out["source"].notna()
    score = completeness * pit.astype(float) * consistency.astype(float) * provenance.astype(float) * pit.astype(float)
    out["pit_valid"] = pit.to_numpy()
    out["quality_score"] = score.to_numpy()
    out["research_grade"] = (pit & consistency & (completeness == 1) & provenance).to_numpy()
    return out


def quality_gate(market: pd.DataFrame, universe: pd.DataFrame) -> dict:
    """回测许可只在原始价、时点和当时成分同时成立时打开。美股免费行情不假装有 CRSP。"""
    if market is None or market.empty:
        return {"backtest_allowed": False, "reason": "no_market_rows", "checks": {}}
    duplicate = int(market.duplicated(["instrument_id", "trade_date"]).sum())
    china = market[market["instrument_id"].astype(str).str.startswith("CN.")]
    united = market[market["instrument_id"].astype(str).str.startswith("US.")]
    raw_prices = bool(china.empty or (china["price_basis"] == "raw").all())
    timed = bool(china.empty or china["available_time"].notna().all())
    actions = bool(china.empty or china["corporate_action_valid"].astype(bool).all())
    membership = pd.DataFrame()
    if universe is not None and not universe.empty and "pit_membership" in universe.columns:
        membership = universe[universe["pit_membership"].astype(bool) & (universe["universe"] == "hs300")]
    missing = []
    missing_bars = []
    missing_actions = []
    if not china.empty:
        covered = set(membership["instrument_id"]) if not membership.empty else set()
        missing = sorted(set(china["instrument_id"]) - covered)
    if not membership.empty:
        present = set()
        if not china.empty:
            present = set(zip(china["instrument_id"], pd.to_datetime(china["trade_date"]).dt.strftime("%Y-%m-%d")))
            checked = set(china.loc[china["corporate_action_valid"].astype(bool), "instrument_id"])
        else:
            checked = set()
        for _, row in membership.iterrows():
            day = str(row["effective_start"])[:10]
            if (row["instrument_id"], day) not in present:
                missing_bars.append(f"{row['instrument_id']}:{day}")
        missing_actions = sorted(set(membership["instrument_id"]) - checked)
    checks = {
        "duplicate_bars": duplicate,
        "china_raw_price": raw_prices,
        "china_available_time": timed,
        "china_corporate_actions_checked": actions and not missing_actions,
        "china_missing_pit_membership": missing,
        "members_missing_bar": len(missing_bars),
        "members_missing_bar_sample": missing_bars[:20],
        "members_missing_actions": len(missing_actions),
        "us_survivorship_free": False,
        "us_rows": int(len(united)),
    }
    allowed = (
        duplicate == 0 and raw_prices and timed and actions and not missing and not china.empty
        and not missing_bars and not missing_actions
    )
    return {"backtest_allowed": allowed, "reason": "passed" if allowed else "quality_gate", "checks": checks}


def annotate_fact(frame: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    out = frame.copy()
    scores, grades, pits = [], [], []
    for _, row in out.iterrows():
        present = [pd.notna(row.get(name)) for name in required]
        completeness = sum(present) / len(required) if required else 1.0
        pit = bool(row.get("pit_valid")) and pd.notna(row.get("available_time"))
        provenance = 1.0 if row.get("source") else 0.0
        score = quality_score(completeness, 1.0 if pit else 0.0, 1.0, provenance, 1.0 if pit else 0.0)
        scores.append(score)
        pits.append(pit)
        grades.append(bool(pit and completeness == 1.0 and provenance == 1.0 and pd.notna(row.get("value"))))
    out["pit_valid"] = pits
    out["quality_score"] = scores
    out["research_grade"] = grades
    return out
