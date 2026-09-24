"""Predeclared experiment limits, purged splits, uncertainty and forward evidence gates."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.data.contracts import fingerprint, timestamp


def walk_forward_splits(calendar, cfg=None):
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
        train, valid = train[:-(cfg.horizon+1)], valid[:-(cfg.horizon+1)]
        if len(train) and len(valid) and len(test): folds.append(dict(train=train, validation=valid, test=test))
        origin += pd.DateOffset(months=cfg.test_months)
    return folds


class ExperimentRegistry:
    def __init__(self, path, cfg=None):
        self.path = Path(path)
        self.cfg = cfg or CONFIG.research

    def register(self, hypothesis, config, data_version, split, model_version):
        row = dict(hypothesis=hypothesis, config=config, data_version=data_version,
                   split=split, model_version=model_version, registered_at=pd.Timestamp.now(tz="UTC").isoformat())
        rows = [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
        identity = fingerprint({k: v for k,v in row.items() if k!="registered_at"})
        if any(x["id"]==identity for x in rows): return identity
        if len(rows)>=self.cfg.max_trials: raise ValueError("model_search_budget_exhausted")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f: f.write(json.dumps(dict(id=identity, **row), default=str)+"\n")
        return identity

    def record(self, trial_id, outcome):
        rows = [json.loads(x) for x in self.path.read_text().splitlines()]
        if not any(x["id"]==trial_id for x in rows): raise ValueError("unregistered_trial")
        out = self.path.with_name(self.path.stem+"_outcomes.jsonl")
        with out.open("a") as f: f.write(json.dumps(dict(id=trial_id, outcome=outcome), default=str)+"\n")


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
    reasons = []
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
