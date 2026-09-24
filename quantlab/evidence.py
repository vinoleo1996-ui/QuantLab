"""Content-addressed research evidence. File existence or caller booleans are insufficient."""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd

ROLES=('accounting','temporal','data','stress','oos','execution','forward')


def verify_bundle(bundle, root):
    root=Path(root).resolve(); reasons=[]; documents={}
    identity=bundle.get('identity',{})
    if not all(identity.get(k) for k in ('data_version','model_version','code_version','mandate_version')):
        reasons.append('missing_evidence_identity')
    for role in ROLES:
        ref=bundle.get('artifacts',{}).get(role)
        if not isinstance(ref,dict): reasons.append(role+'_missing'); continue
        try:
            path=(root/ref['path']).resolve()
            if not path.is_relative_to(root): raise ValueError('path_escape')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=ref.get('sha256'): raise ValueError('checksum_mismatch')
            doc=json.loads(raw)
            if doc.get('kind')!=role or doc.get('identity')!=identity: raise ValueError('identity_mismatch')
            coverage=doc.get('coverage',{})
            start,end=pd.Timestamp(coverage['start']),pd.Timestamp(coverage['end'])
            if pd.isna(start) or pd.isna(end) or start>end: raise ValueError('invalid_coverage')
            if role!='forward' and (str(coverage.get('start'))!=str(bundle.get('coverage',{}).get('start')) or
                                    str(coverage.get('end'))!=str(bundle.get('coverage',{}).get('end'))):
                raise ValueError('coverage_mismatch')
            if doc.get('status')!='passed': raise ValueError('not_passed')
            # Require measured details; a naked status or boolean cannot promote a model.
            detail=doc.get('measurements',{})
            if not detail: raise ValueError('measurements_missing')
            if role=='accounting':
                rows=detail.get('accounts',[])
                if not rows or any(not all(math.isfinite(float(r.get(k,0))) for k in ('cash','nav','securities','receivables','payables')) or r['cash']<0 or abs(r['nav']-r['cash']-r['securities']-r.get('receivables',0)+r.get('payables',0))>.02 for r in rows):
                    raise ValueError('accounting_failed')
            if role=='temporal' and (detail.get('comparisons',0)<1 or detail.get('mismatches',1)!=0): raise ValueError('temporal_failed')
            if role=='data' and (detail.get('unresolved_required_gaps',1)!=0 or not detail.get('field_sources')): raise ValueError('data_failed')
            if role=='stress' and (len(detail.get('scenarios',[]))<8 or any(not s.get('constraints_passed') for s in detail['scenarios'])): raise ValueError('stress_failed')
            if role=='oos':
                folds=detail.get('folds',[])
                if len(folds)<3: raise ValueError('too_few_folds')
                ordered=sorted(folds,key=lambda f:f['start'])
                if any(pd.Timestamp(a['end'])>=pd.Timestamp(b['start']) for a,b in zip(ordered,ordered[1:])): raise ValueError('overlapping_tests')
                if sum(f['net_excess']>0 for f in folds)/len(folds)<2/3: raise ValueError('inconsistent_oos')
                if any(not math.isfinite(float(detail.get(k,{}).get('lower',-1))) or detail.get(k,{}).get('lower',-1)<=0 for k in ('ci20','ci60')): raise ValueError('increment_not_proven')
                if not detail.get('concentration_checks') or not all(detail['concentration_checks'].values()): raise ValueError('concentration_failed')
            if role=='execution' and (detail.get('matched_observations',0)<1 or not detail.get('calibration_version')): raise ValueError('uncalibrated_execution')
            if role=='forward' and (end<start+pd.DateOffset(months=6) or detail.get('missing_sessions',1)!=0 or detail.get('version_changes',1)!=0 or not detail.get('journal_hash')):
                raise ValueError('incomplete_forward_record')
            documents[role]=doc
        except (KeyError,ValueError,TypeError,OSError) as exc: reasons.append(role+':'+str(exc))
    return dict(verified=not reasons,reasons=reasons,documents=documents,promotion_approved=False,
                status='eligible_for_human_review' if not reasons else 'unproven')


def file_reference(path, root):
    path=Path(path);return dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def export_research_bundle(run,ds,summary,root):
    """Publish measured research roles. Missing temporal/execution/forward roles stay missing."""
    from quantlab.data.qualification import qualification
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    identity={k:run.get(k) for k in ('data_version','model_version','code_version','mandate_version')}
    accounts=run.get('account',[])
    coverage=dict(start=str(accounts[0]['date']),end=str(accounts[-1]['date'])) if accounts else {}
    refs={}
    def save(role,measurements,passed):
        doc=dict(kind=role,identity=identity,coverage=coverage,measurements=measurements,status='passed' if passed else 'unproven')
        p=root/(role+'.json');p.write_text(json.dumps(doc,default=str,ensure_ascii=False,indent=2));refs[role]=file_reference(p,root)
    measured=[dict(cash=a['cash'],nav=a['nav'],securities=sum(a['quantities'][c]*a['marks'][c] for c in a['quantities']),
                   receivables=a['receivables'],payables=0.) for a in accounts]
    save('accounting',dict(accounts=measured),bool(measured))
    data=qualification(ds)
    save('data',dict(unresolved_required_gaps=len(data['reasons']),field_sources=ds.manifest.get('coverage_evidence',{})),
         data['status']=='qualified_for_daily_research')
    save('stress',dict(scenarios=[dict(name=k,**v) for k,v in summary.get('stress',{}).items()]),bool(summary.get('stress')))
    folds=[dict(start=f['start'],end=f['end'],net_excess=x) for f,x in zip(summary.get('selection',[]),summary.get('fold_net_excess',[]))]
    save('oos',dict(folds=folds,ci20=summary.get('ci20'),ci60=summary.get('ci60'),
                   concentration_checks=summary.get('concentration',{}).get('checks',{})),
         summary.get('status')!='fixture_test_only' and summary.get('preflight_passed',False))
    bundle=dict(identity=identity,coverage=coverage,artifacts=refs,source='measured_research_only',promotion_approved=False)
    (root/'bundle.json').write_text(json.dumps(bundle,default=str,indent=2));return bundle
