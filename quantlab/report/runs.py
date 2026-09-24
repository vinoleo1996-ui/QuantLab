"""Immutable run artifacts, including exact benchmark, inputs, code and environment."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
import os
import tempfile
import shutil
from pathlib import Path

from quantlab.config import ROOT, OUTPUT_DIR
from quantlab.data.contracts import fingerprint


def provenance():
    paths=list((ROOT/'quantlab').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    packages={x.metadata['Name']:x.version for x in importlib.metadata.distributions() if x.metadata.get('Name')}
    direct={}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith(('#','-')):continue
        name=line.split('>=')[0].split('==')[0].strip()
        if name:
            try:direct[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:direct[name]='not_installed_optional_backend'
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True))
    except (OSError,subprocess.CalledProcessError):commit=None;dirty=True
    lock=ROOT/'requirements-research.lock'
    return dict(code_sha256=code,python=platform.python_version(),packages=packages,direct_dependencies=direct,
                dependency_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
                git_commit=commit,git_dirty=dirty)


def finalize_run_identity(result):
    prov = provenance()
    digest = hashlib.sha256(json.dumps(prov["code_sha256"], sort_keys=True).encode()).hexdigest()[:12]
    if result.get("code_version") is None:
        result["run_id"] += "-" + digest
        result["code_version"] = digest
    elif result["code_version"] != digest:
        raise ValueError("cannot_resave_result_under_different_source_version")
    return prov


def _result_fingerprint(result):
    return fingerprint({k:result.get(k) for k in ('run_id','data_version','code_version','model_version','status','as_of',
        'reasons','config','account','orders','fills','trades','signal_trace','allocation_issues','risk_history')})


def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.json')
    try:
        with os.fdopen(fd,'w') as out:
            json.dump(data,out,default=str,ensure_ascii=False,indent=2);out.flush();os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if Path(name).exists():Path(name).unlink()


def load_run(folder):
    """Integrity checked local artifacts only; never load imported/untrusted pickles."""
    folder=Path(folder)
    manifest=json.loads((folder/'artifacts.json').read_text())
    for name,digest in manifest['sha256'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('artifact_checksum_mismatch:'+name)
    result=pickle.loads((folder/'result.pkl').read_bytes())
    if _result_fingerprint(result)!=manifest['result_fingerprint']:raise ValueError('artifact_result_mismatch')
    return result


def persist_run(result, root=None):
    root = Path(root or OUTPUT_DIR / "runs")
    prov = finalize_run_identity(result)
    folder = root / result["run_id"]
    root.mkdir(parents=True,exist_ok=True)
    if folder.exists():
        old=load_run(folder)
        if _result_fingerprint(old)!=_result_fingerprint(result):raise ValueError('run_identity_collision')
    else:
        staging=Path(tempfile.mkdtemp(prefix='.run-',dir=root))
        (staging / "result.pkl").write_bytes(pickle.dumps(result))
        (staging / "provenance.json").write_text(json.dumps(prov, indent=2))
        (staging / "metadata.json").write_text(json.dumps({k: result.get(k) for k in
            ("run_id", "data_version", "model_version", "as_of", "status", "reasons", "config", "promotion_approved")},
            default=str, ensure_ascii=False, indent=2))
        bench = result.get("benchmark")
        if bench is not None: bench.to_parquet(staging / "benchmark.parquet")
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir()}
        atomic_json(staging/'artifacts.json',dict(result_fingerprint=_result_fingerprint(result),sha256=hashes))
        try:staging.rename(folder)
        except OSError:
            shutil.rmtree(staging)
            if not folder.exists():raise
            if _result_fingerprint(load_run(folder))!=_result_fingerprint(result):raise ValueError('run_identity_collision')
    atomic_json(root/'latest.json',dict(run_id=result['run_id'],status=result['status']))
    return folder
