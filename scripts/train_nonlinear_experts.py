"""Build only chronological out-of-training predictions for the approved study."""
from __future__ import annotations
import argparse
import fcntl
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd
import continuity_cadence as shared

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/expert_selection_20260927'
MODULE=ROOT/'quantlab/strategy/nonlinear_expert.py'
spec=importlib.util.spec_from_file_location('study_nonlinear',MODULE)
nonlinear=importlib.util.module_from_spec(spec);sys.modules[spec.name]=nonlinear;spec.loader.exec_module(nonlinear)


def _train_locked(kind):
    from dataclasses import asdict
    import lightgbm
    diagnostic=OUT/'random_diagnostic/summary.json'
    if not diagnostic.exists(): raise ValueError('reference_random_diagnostic_not_completed')
    features=json.loads((OUT/'features_complete.json').read_text())
    for entry in features['files'].values():shared.verify_file(entry['path'],entry['sha256'])
    shared.verify_file(features['router_scores']['path'],features['router_scores']['sha256'])
    source_sha=shared.sha256(MODULE)
    p=nonlinear.EnsemblePolicy()
    folder=OUT/'models'/kind
    protocol=dict(kind=kind,source_sha256=source_sha,runner_sha256=shared.sha256(__file__),
        features_complete_sha256=shared.sha256(OUT/'features_complete.json'),
        diagnostic_sha256=shared.sha256(diagnostic),policy=asdict(p),lightgbm=lightgbm.__version__,
        created_at=datetime.now(timezone.utc).isoformat(),
        targets='all original router monthly cohorts plus completed July/August2026; no parameter selection',
        fallback='entire decision uses five-expert R on missing candidate predictions; no shrinking universe')
    if (folder/'protocol.json').exists():
        previous=json.loads((folder/'protocol.json').read_text())
        for k in ('source_sha256','runner_sha256','features_complete_sha256','diagnostic_sha256','policy','lightgbm'):
            # JSON makes tuples into lists.
            if json.dumps(previous[k],sort_keys=True)!=json.dumps(protocol[k],sort_keys=True):
                raise ValueError('training_resume_identity_changed:'+k)
    else:shared.freeze_json(folder/'protocol.json',protocol)
    if (folder/'complete.json').exists():
        done=json.loads((folder/'complete.json').read_text())
        shared.verify_file(folder/'protocol.json',done['protocol_sha256'])
        for entry in done['predictions'].values():
            dest=Path(entry['path'])
            shared.verify_file(dest/'complete.json',entry['complete_sha256'])
            for name,digest in json.loads((dest/'complete.json').read_text())['files'].items():
                shared.verify_file(dest/name,digest)
        print(json.dumps(dict(stage='already_completed_verified',kind=kind,months=len(done['predictions']))),flush=True)
        return
    all_features=pd.concat([pd.read_parquet(e['path']) for e in features['files'].values()],ignore_index=True)
    scored=pd.read_parquet(features['router_scores']['path'])
    outputs={};start=time.monotonic()
    for i,(day,frame) in enumerate(scored.groupby('date',sort=True)):
        dest=folder/day
        if (dest/'complete.json').exists():
            complete=json.loads((dest/'complete.json').read_text())
            for name,digest in complete['files'].items():shared.verify_file(dest/name,digest)
            outputs[day]=dict(path=str(dest.resolve()),complete_sha256=shared.sha256(dest/'complete.json'))
            continue
        if dest.exists():
            # Preserve an interrupted publication for audit; a retry owns a new
            # empty destination and never overwrites model bytes from an attempt.
            abandoned=folder/'incomplete_attempts'/f'{day}-{uuid.uuid4().hex}'
            abandoned.parent.mkdir(parents=True,exist_ok=True)
            dest.rename(abandoned)
        as_of=nonlinear.cutoff(day)+pd.Timedelta(hours=18)
        began=time.monotonic()
        model=nonlinear.MonthlyEnsemble(kind,p)
        try:
            model.fit(all_features,as_of)
            values=model.predict(frame.set_index('instrument_id'),as_of=as_of)
            if values.nunique()<2:raise ValueError('constant_nonlinear_prediction')
        except ValueError as exc:
            if str(exc) not in ('insufficient_matured_training_cohorts','missing_prediction_features','constant_nonlinear_prediction'):
                raise
            evidence=dict(status='unavailable',reason=str(exc),as_of=str(as_of),kind=kind)
            shared.freeze_json(dest/'evidence.json',evidence)
            shared.freeze_json(dest/'complete.json',dict(status='unavailable',files={'evidence.json':shared.sha256(dest/'evidence.json')}))
        else:
            prediction=values.rename('prediction').rename_axis('instrument_id').reset_index()
            prediction['date']=day
            shared.publish(dest/'predictions.parquet',lambda h:prediction.to_parquet(h,index=False))
            for j,boost in enumerate(model.models):
                shared.publish(dest/f'model{j}.txt',lambda h,m=boost:h.write(m.model_to_string().encode()))
            shared.freeze_json(dest/'evidence.json',dict(model.evidence,status='ok',
                training_feature_manifest_sha256=protocol['features_complete_sha256'],
                prediction_source=features['router_scores'],model_source_sha256=source_sha))
            names=['predictions.parquet','evidence.json']+[f'model{j}.txt' for j in range(p.num_models)]
            shared.freeze_json(dest/'complete.json',dict(status='ok',files={n:shared.sha256(dest/n) for n in names}))
        outputs[day]=dict(path=str(dest.resolve()),complete_sha256=shared.sha256(dest/'complete.json'))
        status=dict(stage='monthly_predictions',kind=kind,completed=i+1,total=scored.date.nunique(),date=day,
                    seconds_this_month=time.monotonic()-began,elapsed_seconds=time.monotonic()-start,
                    at=datetime.now(timezone.utc).isoformat())
        shared.freeze_json(folder/'status.json',status,immutable=False);print(json.dumps(status),flush=True)
        shared.verify_file(MODULE,source_sha)
    shared.freeze_json(folder/'complete.json',dict(predictions=outputs,protocol_sha256=shared.sha256(folder/'protocol.json')))
    shared.freeze_json(folder/'status.json',dict(stage='completed',kind=kind,months=len(outputs)),immutable=False)


def train(kind):
    folder=OUT/'models'/kind
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('training_already_running:'+kind) from exc
        _train_locked(kind)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['plain','double'])
    train(parser.parse_args().kind)
