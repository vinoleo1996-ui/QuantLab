"""本地 parquet 缓存与数据登记。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.config import A_STOCK_DIR, FUND_DIR, US_STOCK_DIR, CONFIG


class DataStore:
    """按 (market, kind) 分目录缓存的 parquet 存储。

    - market: a | us | bench | fund
    - key: 标的代码或数据表名
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else CONFIG.data.cache_dir.parent
        self.dirs = {
            "a": self.root / "a_stock" if root is not None else CONFIG.data.cache_dir,
            "us": self.root / "us_stock",
            "fund": self.root / "fundamentals",
            "bench": self.root / "bench",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def _path(self, market: str, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.dirs.get(market, self.root / market) / f"{safe}_{key.replace('/', '_')}.parquet"

    def save(self, df: pd.DataFrame, market: str, key: str) -> None:
        if df is None or df.empty:
            return
        path = self._path(market, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        old = self.load(market, key)
        incoming = df.copy()
        if old is not None and key.startswith(("daily_", "bench_")):
            if old.attrs.get("price_basis") != incoming.attrs.get("price_basis"):
                raise ValueError("price_basis_change_requires_new_cache_key")
            overlap = old.index.intersection(incoming.index)
            for field in ("open", "close", "amount", "volume", "turnover"):
                if field in old:
                    new = incoming[field].reindex(overlap) if field in incoming else pd.Series(float("nan"), index=overlap)
                    if (old[field].reindex(overlap).notna() & new.isna()).any():
                        raise ValueError(f"source_quality_regression:{key}:{field}")
            attrs = incoming.attrs.copy()
            incoming = pd.concat([old, incoming]).loc[lambda x: ~x.index.duplicated(keep="last")].sort_index()
            incoming.attrs = attrs
        if "source" not in incoming.attrs:
            incoming.attrs["source"] = "unknown"
        incoming.attrs["fetched_at"] = datetime.now(timezone.utc).isoformat()
        versions = path.parent / "versions" / key
        versions.mkdir(parents=True, exist_ok=True)
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archive = versions / (digest + ".parquet")
            if not archive.exists(): archive.write_bytes(path.read_bytes())
        fd, name = tempfile.mkstemp(dir=path.parent, suffix=".parquet")
        os.close(fd)
        try:
            incoming.to_parquet(name)
            os.replace(name, path)
        finally:
            if Path(name).exists(): Path(name).unlink()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (versions / (digest + ".parquet")).write_bytes(path.read_bytes())
        record = {"sha256": digest, "source": incoming.attrs["source"],
                  "fetched_at": incoming.attrs["fetched_at"], "rows": len(incoming),
                  "last_event": str(incoming.index[-1]), "price_basis": incoming.attrs.get("price_basis", "unknown")}
        (versions / (digest + ".json")).write_text(json.dumps(record, indent=2))

    def load(self, market: str, key: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
        p = self._path(market, key)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
            df = df.set_index("datetime").sort_index()
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.set_index("date").sort_index()
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return df

    def exists(self, market: str, key: str) -> bool:
        return self._path(market, key).exists()

    def last_date(self, market: str, key: str) -> pd.Timestamp | None:
        df = self.load(market, key)
        return None if df is None or df.empty else df.index[-1]

    def list_keys(self, market: str) -> list[str]:
        d = self.dirs.get(market)
        if d is None:
            return []
        return [p.name.split("_", 1)[1].replace(".parquet", "") for p in d.glob("*.parquet")]