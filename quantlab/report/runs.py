"""Immutable run artifacts, including exact benchmark, inputs, code and environment."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path

from quantlab.config import ROOT, OUTPUT_DIR


def provenance():
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "quantlab").rglob("*.py"))}
    packages = {x.metadata["Name"]: x.version for x in importlib.metadata.distributions() if x.metadata.get("Name")}
    return {"code_sha256": code, "python": platform.python_version(), "packages": packages}


def finalize_run_identity(result):
    prov = provenance()
    digest = hashlib.sha256(json.dumps(prov["code_sha256"], sort_keys=True).encode()).hexdigest()[:12]
    if result.get("code_version") is None:
        result["run_id"] += "-" + digest
        result["code_version"] = digest
    elif result["code_version"] != digest:
        raise ValueError("cannot_resave_result_under_different_source_version")
    return prov


def persist_run(result, root=None):
    root = Path(root or OUTPUT_DIR / "runs")
    prov = finalize_run_identity(result)
    folder = root / result["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "result.pkl").exists():
        (folder / "result.pkl").write_bytes(pickle.dumps(result))
        (folder / "provenance.json").write_text(json.dumps(prov, indent=2))
        (folder / "metadata.json").write_text(json.dumps({k: result.get(k) for k in
            ("run_id", "data_version", "model_version", "as_of", "status", "reasons", "config", "promotion_approved")},
            default=str, ensure_ascii=False, indent=2))
        bench = result.get("benchmark")
        if bench is not None: bench.to_parquet(folder / "benchmark.parquet")
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps({"run_id": result["run_id"], "status": result["status"]}, indent=2))
    return folder
