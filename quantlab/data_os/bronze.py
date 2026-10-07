"""青铜层：原始响应按内容哈希追加，已存在的字节不再写一遍。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BronzeStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.manifest_path = self.root / "metadata" / "bronze_manifest.jsonl"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, source: str, dataset: str, payload: bytes, meta: dict | None = None) -> dict:
        digest = hashlib.sha256(payload).hexdigest()
        directory = self.raw / source / dataset
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest}.bin"
        created = not path.exists()
        if created:
            path.write_bytes(payload)
        record = {
            "source": source,
            "dataset": dataset,
            "sha256": digest,
            "bytes": len(payload),
            "path": str(path.relative_to(self.root)),
            "created": created,
            "ingested_at": _now(),
            "meta": meta or {},
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def lineage(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            return pd.DataFrame(columns=["source", "dataset", "sha256", "bytes", "path", "created", "ingested_at"])
        rows = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                item.pop("meta", None)
                rows.append(item)
        return pd.DataFrame(rows)
