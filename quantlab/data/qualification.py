"""Purpose-specific data acceptance; provider assertions alone do not certify history."""
from pathlib import Path
import json
import hashlib
import pandas as pd
from quantlab.data.contracts import load_dataset, fingerprint, frame_hash, timestamp

REQUIRED=('security_history','raw_prices','corporate_actions','calendar','benchmark','opening_rules')


def coverage_input_fingerprint(ds):
    """Exclude the manifest to avoid circular hashes with its evidence references."""
    return fingerprint(dict(panels={c:frame_hash(p) for c,p in sorted(ds.panels.items())},
        instruments=frame_hash(ds.instruments),actions=frame_hash(ds.actions),calendar=list(ds.calendar),
        benchmark=frame_hash(ds.benchmark),exposures=frame_hash(ds.exposures)))


def _verify_coverage(ds, field, ref):
    root=Path(ds.evidence_root).resolve() if ds.evidence_root else None
    if root is None:raise ValueError('artifact_root_missing')
    path=(root/ref['verification_artifact']).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('artifact_path_invalid')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('checksum_mismatch')
    doc=json.loads(raw)
    if doc.get('kind')!='data_coverage' or doc.get('field')!=field or doc.get('status')!='passed':raise ValueError('coverage_not_passed')
    if doc.get('input_fingerprint')!=coverage_input_fingerprint(ds):raise ValueError('input_mismatch')
    scope=doc.get('coverage',{})
    start,end=timestamp(scope.get('start')),timestamp(scope.get('end'))
    if pd.isna(start) or pd.isna(end) or start>timestamp(ds.calendar[0]) or end<timestamp(ds.manifest['as_of']):raise ValueError('insufficient_interval')
    if sorted(scope.get('instruments',[]))!=sorted(ds.panels):raise ValueError('instrument_scope_mismatch')
    m=doc.get('measurements',{})
    if not m:raise ValueError('measurements_missing')
    minimum={'security_history':len(ds.instruments),'raw_prices':sum(map(len,ds.panels.values())),
             'opening_rules':sum(map(len,ds.panels.values())),'calendar':len(ds.calendar),
             'benchmark':len(ds.benchmark),'corporate_actions':len(ds.actions),'industry_history':len(ds.instruments)}[field]
    expected,actual=m.get('expected_observations'),m.get('actual_observations')
    if not isinstance(expected,int) or isinstance(expected,bool) or expected<minimum or expected!=actual or m.get('unresolved_required_gaps')!=0:
        raise ValueError('coverage_measurements_failed')
    if not doc.get('method_version') or ref['source_url'] not in doc.get('source_urls',[]):raise ValueError('method_or_source_missing')


def qualification(ds, *, require_industry=True):
    reasons=ds.validate();m=ds.manifest;coverage=m.get('coverage_evidence',{})
    if m.get('universe_coverage')!='historical':reasons.append('historical_universe_required')
    if m.get('source')=='TEST_FIXTURE':reasons.append('synthetic_test_fixture')
    for field in (*REQUIRED, *(('industry_history',) if require_industry else ())):
        evidence=coverage.get(field,{})
        if not evidence.get('source_url') or not evidence.get('verification_artifact') or not evidence.get('sha256'):
            reasons.append('coverage_evidence_missing:'+field)
        else:
            try:_verify_coverage(ds,field,evidence)
            except (ValueError,KeyError,TypeError,OSError) as exc:reasons.append('coverage_artifact_invalid:'+field+':'+str(exc))
    missing_sectors=ds.instruments.sector.isna()|ds.instruments.sector.isin(['UNKNOWN','NA','']) if 'sector' in ds.instruments else pd.Series([True])
    if 'asset_type' in ds.instruments:missing_sectors=missing_sectors & ds.instruments.asset_type.eq('stock')
    if require_industry and missing_sectors.any():reasons.append('historical_industry_incomplete')
    statuses={};freshness={}
    for code,p in ds.panels.items():
        statuses[code]=dict(rows=len(p),unknown_open=int((p.open_status!='trading').sum()) if 'open_status' in p else len(p),
                            bad_quality=int((p.quality_status!='ok').sum()) if 'quality_status' in p else len(p))
        freshness[code]=str(p.index[-1]) if len(p) else None
    return dict(status='blocked_data' if reasons else 'qualified_for_daily_research',reasons=sorted(set(reasons)),
                data_version=ds.version,coverage=statuses,latest_dates=freshness,
                opening_capacity='unverified_until_minute_or_actual_fill_calibration',
                announcement_model='disabled_pending_PIT_incremental_evidence',promotion_approved=False)


def audit_manifest(path):
    path=Path(path)
    if not path.exists():return dict(status='blocked_data',reasons=['dataset_manifest_missing'],path=str(path),promotion_approved=False)
    try:
        return qualification(load_dataset(path))
    except (ValueError,KeyError,OSError,AttributeError) as exc:return dict(status='blocked_data',reasons=[str(exc)],promotion_approved=False)
