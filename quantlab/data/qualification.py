"""Purpose-specific data acceptance; provider assertions alone do not certify history."""
from pathlib import Path
import json
import hashlib
import pandas as pd
from quantlab.data.contracts import load_dataset, fingerprint, frame_hash, timestamp
from quantlab.data.industry import resolve_industries

REQUIRED=('security_history','raw_prices','corporate_actions','calendar','benchmark','opening_rules')


def coverage_input_fingerprint(ds):
    """Exclude the manifest to avoid circular hashes with its evidence references."""
    return fingerprint(dict(panels={c:frame_hash(p) for c,p in sorted(ds.panels.items())},
        instruments=frame_hash(ds.instruments),actions=frame_hash(ds.actions),calendar=list(ds.calendar),
        benchmark=frame_hash(ds.benchmark),exposures=frame_hash(ds.exposures),
        industry_records=frame_hash(ds.industry_records),financials=frame_hash(ds.financials)))


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


def research_assessment(ds, *, require_industry=False):
    """Admit an explicitly limited experiment, without asserting provider completeness."""
    reasons=ds.validate('free_research')
    if ds.manifest.get('source')=='TEST_FIXTURE':reasons.append('synthetic_test_fixture')
    if ds.manifest.get('universe_coverage')!='historical':reasons.append('historical_universe_required')
    if not ds.panels:reasons.append('empty_universe')
    for code,p in ds.panels.items():
        for field in ('amount','turnover'):
            if field not in p:reasons.append('price_volume_missing:'+code+':'+field)
        if 'amount' in p and (p.amount.dropna()<0).any():reasons.append('negative_amount:'+code)
    resolved=resolve_industries(ds,ds.manifest['as_of'])
    sectors=resolved.sector.reindex(ds.panels) if not resolved.empty else pd.Series('UNKNOWN',index=list(ds.panels))
    industry=bool(len(sectors)) and not (sectors.isna()|sectors.isin(['UNKNOWN','NA',''])).any()
    # A conditional feature experiment can use valid industries while unknown
    # names remain ineligible for those features. Strict qualification above
    # continues to require complete historical evidence.
    industry_partial=bool((sectors.notna()&~sectors.isin(['UNKNOWN','NA',''])).any())
    if require_industry and not industry_partial:reasons.append('historical_industry_incomplete')
    warnings=['public_data_revisions_not_independently_verified','opening_capacity_unverified']
    if not industry:warnings.append('industry_disabled_unknown_exposure_capped')
    if ds.manifest.get('actions_coverage')!='complete':warnings.append('accounting_coverage_unverified')
    if ds.manifest.get('execution_gaps'):warnings.append('opening_rule_conflicts:'+str(len(ds.manifest['execution_gaps'])))
    if any(p.get('open_status',pd.Series(dtype=str)).eq('derived').any() for p in ds.panels.values()):
        warnings.append('daily_open_execution_is_explicit_assumption')
    industry_mode=ds.manifest.get('industry_mode','verified_pit')
    if industry_mode=='effective_date_proxy':warnings.append('industry_effective_dates_are_not_publication_evidence')
    if ds.manifest.get('feature_price_basis')=='historical_adjustment_snapshot':
        warnings.append('feature_prices_use_current_historical_adjustment_snapshot')
    coverage=dict(price_rows=sum(len(p) for p in ds.panels.values()),
        turnover_rows=sum(int(p.get('turnover',pd.Series(dtype=float)).notna().sum()) for p in ds.panels.values()),
        observed_state_rows=sum(int(p.get('is_st_at_open',pd.Series(dtype=float)).notna().sum()) for p in ds.panels.values()),
        securities=len(ds.panels),industry_known=int((sectors.notna()&~sectors.isin(['UNKNOWN','NA',''])).sum()))
    if coverage['turnover_rows']<coverage['price_rows']:warnings.append('legacy_turnover_factor_coverage_incomplete')
    if coverage['observed_state_rows']<coverage['price_rows'] and ds.manifest.get('source')=='local_market_snapshot':
        warnings.append('missing_security_states_do_not_allow_new_positions')
    return dict(status='blocked_data' if reasons else 'research_ready',reasons=sorted(set(reasons)),
        warnings=warnings,data_version=ds.version,promotion_approved=False,
        execution_gaps=ds.manifest.get('execution_gaps',[]),
        modules=dict(price_volume=not reasons,industry=industry,industry_partial=coverage['industry_known']>0,
                     industry_verified=industry and industry_mode=='verified_pit',
                     legacy_turnover_complete=coverage['turnover_rows']==coverage['price_rows'],
                     announcements=False,execution_capacity=False),coverage=coverage,
        accounting_verified=ds.manifest.get('actions_coverage')=='complete' and not ds.manifest.get('data_gaps'))


def audit_manifest(path, profile="strict", require_industry=True):
    path=Path(path)
    if not path.exists():return dict(status='blocked_data',reasons=['dataset_manifest_missing'],path=str(path),promotion_approved=False)
    try:
        ds=load_dataset(path,profile)
        return research_assessment(ds,require_industry=require_industry) if profile=='free_research' else qualification(ds,require_industry=require_industry)
    except (ValueError,KeyError,OSError,AttributeError) as exc:return dict(status='blocked_data',reasons=[str(exc)],promotion_approved=False)
