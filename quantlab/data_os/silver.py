"""银层：每个 dataset_version 单独落盘，不回写旧版本。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.data_os.schema import (
    ACTION_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    INSTRUMENT_COLUMNS,
    MACRO_COLUMNS,
    MARKET_COLUMNS,
    UNIVERSE_COLUMNS,
    require_columns,
)


TABLES = {
    "instruments": INSTRUMENT_COLUMNS,
    "market": MARKET_COLUMNS,
    "corporate_actions": ACTION_COLUMNS,
    "fundamentals": FUNDAMENTAL_COLUMNS,
    "universe": UNIVERSE_COLUMNS,
    "macro": MACRO_COLUMNS,
}


def write_table(root: Path, name: str, frame: pd.DataFrame, version: str) -> Path:
    columns = TABLES[name]
    if frame is None or frame.empty:
        payload = pd.DataFrame(columns=columns)
    else:
        payload = require_columns(frame, columns, name)
    path = Path(root) / "silver" / name / f"dataset_version={version}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"银层版本已存在，拒绝覆盖: {path}")
    payload.to_parquet(path, index=False)
    return path


def write_quality(root: Path, name: str, frame: pd.DataFrame, version: str) -> Path:
    path = Path(root) / "quality" / f"dataset_version={version}" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"质量报告已存在，拒绝覆盖: {path}")
    frame.to_parquet(path, index=False)
    return path
