"""Predeclared experiment limits, purged splits, uncertainty and forward evidence gates."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.data.contracts import fingerprint, timestamp


def walk_forward_splits(calendar, cfg=None, label_ends=None):
    cfg = cfg or CONFIG.research
    dates = pd.DatetimeIndex(calendar)
    if dates.empty: return []
    folds = []
    origin = dates[0]
    while True:
        val_start = origin + pd.DateOffset(months=cfg.train_months)
        test_start = val_start + pd.DateOffset(months=cfg.validation_months)
        test_end = test_start + pd.DateOffset(months=cfg.test_months)
        if test_end > dates[-1] + pd.Timedelta(days=1): break
        train = dates[(dates>=origin)&(dates<val_start)]
        valid = dates[(dates>=val_start)&(dates<test_start)]
        test = dates[(dates>=test_start)&(dates<test_end)]
        # Labels consume next-session entry plus horizon sessions; purge both boundaries.
        if label_ends is None:
            train, valid = train[:-(cfg.horizon+1)], valid[:-(cfg.horizon+1)]
        else:
            train=purge_labels(train,label_ends,val_start)
            valid=purge_labels(valid,label_ends,test_start)
        if len(train) and len(valid) and len(test): folds.append(dict(train=train, validation=valid, test=test))
        origin += pd.DateOffset(months=cfg.test_months)
    return folds


def purge_labels(dates, label_ends, boundary):
    """Keep only labels actually complete strictly before the next partition."""
    dates=pd.DatetimeIndex(dates)
    ends=pd.to_datetime(label_ends.reindex(dates))
    return dates[ends.notna() & (ends < pd.Timestamp(boundary))]


class ExperimentRegistry:
    def __init__(self, path, cfg=None):
        import sqlite3
        self.path=Path(path).with_suffix('.sqlite3')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.cfg=cfg or CONFIG.research
        with sqlite3.connect(self.path) as db:
            db.executescript('CREATE TABLE IF NOT EXISTS trials (id TEXT PRIMARY KEY, body TEXT NOT NULL);'
                             'CREATE TABLE IF NOT EXISTS folds (id TEXT, fold TEXT, PRIMARY KEY(id,fold));'
                             'CREATE TABLE IF NOT EXISTS outcomes (id TEXT, fold TEXT, body TEXT NOT NULL, PRIMARY KEY(id,fold));')

    def register(self,hypothesis,config,data_version,split,model_version):
        import sqlite3
        body=dict(hypothesis=hypothesis,config=config,data_version=data_version,model_version=model_version)
        identity=fingerprint(body)
        with sqlite3.connect(self.path,timeout=30) as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM trials WHERE id=?',(identity,)).fetchone():
                if db.execute('SELECT COUNT(*) FROM trials').fetchone()[0]>=self.cfg.max_trials:
                    raise ValueError('model_search_budget_exhausted')
                db.execute('INSERT INTO trials VALUES (?,?)',(identity,json.dumps(body,default=str)))
            db.execute('INSERT OR IGNORE INTO folds VALUES (?,?)',(identity,str(split)))
        return identity

    def record(self,trial_id,outcome):
        import sqlite3
        fold=str(outcome.get('fold','default'))
        with sqlite3.connect(self.path,timeout=30) as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM trials WHERE id=?',(trial_id,)).fetchone(): raise ValueError('unregistered_trial')
            if db.execute('SELECT 1 FROM outcomes WHERE id=? AND fold=?',(trial_id,fold)).fetchone(): raise ValueError('outcome_already_recorded')
            db.execute('INSERT INTO outcomes VALUES (?,?,?)',(trial_id,fold,json.dumps(outcome,default=str)))


def block_bootstrap_ci(excess_returns, seed=None, block=20, samples=2000):
    r = np.asarray(pd.Series(excess_returns).dropna(), dtype=float)
    if len(r)<block*3: return dict(status="insufficient_data", lower=None, upper=None)
    rng = np.random.default_rng(CONFIG.seed if seed is None else seed)
    estimates = []
    for _ in range(samples):
        starts = rng.integers(0, len(r)-block+1, size=int(np.ceil(len(r)/block)))
        sample = np.concatenate([r[i:i+block] for i in starts])[:len(r)]
        estimates.append(float(sample.mean()*252))
    return dict(status="estimated", lower=float(np.quantile(estimates,.025)), upper=float(np.quantile(estimates,.975)),
                block=block, samples=samples, estimator="annualized_mean_daily_excess")


def promotion_gate(evidence, cfg=None):
    cfg = cfg or CONFIG.research
    if 'artifacts' in evidence:
        from quantlab.evidence import verify_bundle
        return verify_bundle(evidence, evidence.get('root', '.'))
    reasons = ['verified_evidence_bundle_required']
    for name in ("accounting_passed", "temporal_passed", "pit_coverage_passed", "stress_passed", "execution_calibrated"):
        if evidence.get(name) is not True: reasons.append(name+"_missing_or_failed")
    fold_excess = evidence.get("fold_net_excess", [])
    if len(fold_excess)<3: reasons.append("too_few_nonoverlapping_folds")
    elif np.mean(np.array(fold_excess)>0)<2/3: reasons.append("inconsistent_incremental_returns")
    ci = evidence.get("incremental_ci", {})
    if ci.get("lower") is None or ci["lower"]<=0: reasons.append("incremental_value_not_proven")
    if evidence.get("concentration_passed") is not True: reasons.append("concentration_unverified")
    if evidence.get("forward_months", 0)<cfg.forward_months: reasons.append("forward_record_too_short")
    if evidence.get("forward_model_unchanged") is not True: reasons.append("forward_version_not_frozen")
    return dict(status="unproven" if reasons else "eligible_for_human_review", reasons=reasons,
                promotion_approved=False, statistical_power="Few six-month windows can leave economically meaningful effects indistinguishable from noise.")


def record_forward(path, payload, now=None):
    """Append-only forward journal; historical replay cannot manufacture forward evidence."""
    now = timestamp(now or pd.Timestamp.now(tz="Asia/Shanghai"))
    if payload.get("status") != "research_unvalidated": raise ValueError("unverified_run")
    as_of = timestamp(payload["as_of"])
    if as_of.date()!=now.date() or as_of>now: raise ValueError("forward_record_must_be_current")
    if not payload.get("model_version") or not payload.get("data_version"): raise ValueError("missing_version")
    path = Path(path)
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    if any(x["as_of"]==payload["as_of"] and x["model_version"]==payload["model_version"] for x in rows):
        raise ValueError("forward_record_already_exists")
    row = dict(payload, recorded_at=now.isoformat(), previous_hash=rows[-1]["hash"] if rows else None)
    row["hash"] = fingerprint(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f: f.write(json.dumps(row, default=str, ensure_ascii=False)+"\n")
    return row
