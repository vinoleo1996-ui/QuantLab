"""Frozen reference-pool selection diagnostic; never invokes an account or model."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/Users/agiuser/Desktop/QuantLab')
OUT=ROOT/'output/expert_selection_20260927/random_diagnostic'
DATA=ROOT/'output/account_2025_20260927/extension/dataset'
SEED=20260927

def sample_basket(heads, industries, n, rng, matched=None):
    """Permutation within financial tiers then industry cap; or exact strata SRSWOR.

    Does not accept labels, so unavailable future outcomes cannot alter eligibility.
    Primary is a randomized sequential priority mechanism, not uniform over all
    feasible constrained subsets. Matched arm samples uniformly within each cell.
    """
    codes=sorted(heads)
    if matched is not None:
        counts=Counter((heads[c],industries[c]) for c in matched)
        chosen=[]
        for key,count in sorted(counts.items()):
            pool=[c for c in codes if (heads[c],industries[c])==key]
            if len(pool)<count:raise ValueError('infeasible_stratum')
            chosen.extend(rng.choice(pool,count,replace=False).tolist())
        return chosen
    chosen=[];used=Counter()
    for tier in sorted(set(heads.values()),reverse=True):
        pool=[c for c in codes if heads[c]==tier]
        for c in rng.permutation(pool):
            if used[industries[c]]<15:
                chosen.append(str(c));used[industries[c]]+=1
                if len(chosen)==n:return chosen
    raise ValueError('insufficient_industry_capacity')

def forward_label(prices, calendar, start, horizon, period_end, gaps):
    start=pd.Timestamp(start); i=calendar.get_indexer([start])[0]
    if i<0:raise ValueError('signal_not_session')
    if i+horizon>=len(calendar):return None,'immature',None
    end=calendar[i+horizon]
    if end>pd.Timestamp(period_end):return None,'period_end',str(end.date())
    if any(start<pd.Timestamp(d)<=end for d in gaps):return None,'known_gap',str(end.date())
    v=prices.reindex(calendar[i:i+horizon+1]).to_numpy(float)
    if not np.isfinite(v).all() or (v<=0).any():return None,'missing_path',str(end.date())
    return float(v[-1]/v[0]-1),'observed',str(end.date())

def block_indices(n, length, rng):
    starts=rng.integers(0,n,size=int(np.ceil(n/length)))
    return ((starts[:,None]+np.arange(length))%n).reshape(-1)[:n]

def bootstrap(values, calendar, horizon, seed):
    sparse=pd.Series(np.nan,index=calendar)
    for d,x in values:sparse.loc[pd.Timestamp(d)]=x
    a=sparse.to_numpy();rng=np.random.default_rng(seed);means=[]
    for _ in range(10000):
        x=a[block_indices(len(a),horizon,rng)];x=x[np.isfinite(x)]
        if len(x):means.append(float(x.mean()))
    return dict(mean_pp=float(np.nanmean(a)*100) if np.isfinite(a).any() else None,
                ci_family95_pp=(np.quantile(means,[.05/(2*8),1-.05/(2*8)])*100).tolist() if means else None,
                bootstrap_draws=10000,valid_bootstrap_draws=len(means),calendar_sessions=len(a),block_sessions=horizon,
                observed_dates=int(np.isfinite(a).sum()),effective_blocks_upper_bound=min(int(np.isfinite(a).sum()),len(a)/horizon),
                inference='conditional_complete_basket_dates; low_power; not_independent_OOS')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    periods={'2022_2025':('2025-12-31','reference_sustainability_20260927/2022_2025/decision_inputs.json'),
             '2025_recent':('2026-09-22','reference_recent_20260927/inputs.json')}
    sources=[Path(__file__),Path(__file__).parents[1]/'tests/test_router_random_diagnostic.py',DATA/'manifest.json',DATA/manifest['calendar']['path']]
    specs=[]
    for period,(end,path) in periods.items():
        inp=ROOT/'output'/path;tgt=ROOT/f'output/expert_router_20260927/{period}/conditional_router_targets.json';sources += [inp,tgt]
        inputs=json.loads(inp.read_text()); targets=json.loads(tgt.read_text()); traces={x['signal_date']:x for x in targets['trace']}
        for t in targets['targets']:
            d=t['signal_date'];info=traces[d]['information_date'];h=inputs['heads'][info];s=inputs['industries'][info];selected=sorted(t['weights'])
            assert info<=d and len(selected)==len(set(selected)) and set(selected)<=set(h)
            s={c:s.get(c,'unclassified') for c in h}
            specs.append(dict(period=period,signal_date=d,information_date=info,period_end=end,heads=h,industries=s,selected=selected,retained_count=traces[d]['retained_count']))
    allcodes=sorted(set(c for x in specs for c in x['heads']))
    entries={c:manifest['panels'].get(c.split('.')[1].lower()+'.'+c.split('.')[0]) for c in allcodes}
    identities={str(p):sha(p) for p in sources}
    # Verify/freeze every panel hash before outcome calculation, not just the manifest.
    for c,e in entries.items():
        if e:
            p=DATA/e['path'];digest=sha(p)
            if digest!=e['sha256']:raise ValueError('panel_hash_mismatch:'+c)
            identities[str(p)]=digest
    if sha(DATA/manifest['calendar']['path'])!=manifest['calendar']['sha256']:raise ValueError('calendar_hash_mismatch')
    freeze(OUT/'protocol.json',dict(created_at=datetime.now(timezone.utc).isoformat(),question='Does frozen router selection beat score-blind selection within its reference pool?',
        decision='Inform approved new-expert/retention comparisons; inconclusive CI does not cancel approved experiments.',
        account_conditions='60万元、最多50只、只多、无杠杆; offline equal-weight signal-close proxy only; no account replay',
        source_hashes=identities,seed=SEED,draws=1000,bootstrap_draws=10000,horizons=[20,60],family_tests=8,
        controls=['financial-tier random priority with industry cap 15','exact observed tier x industry counts'],
        missing='No filtering during membership. Whole basket unknown if any label unknown. Both arms excluded for each affected paired draw; report conditional means and full-known fractions.',
        bounds='Nonnegative price implies return>=-1; missing upper bound is unbounded. Lower basket=(known sum - missing count)/original count. No finite unconditional difference bounds when both sides have missing labels.',
        temporal='Circular moving blocks of 20/60 exchange sessions on sparse date observations; dates never treated as consecutive sessions. Bonferroni 95% family intervals for 8 tests.',
        limits=['30bp costs excluded','no cash, execution, total-return, or alpha/beta identification','primary includes retention differences','exact strata matches exposures, not retention or all characteristics','historical adjusted feature price/PIT data unverified','recent overlaps long period; family adjustment does not create independence']))
    rng=np.random.default_rng(SEED);members={}
    for j,x in enumerate(specs):
        codes=sorted(x['heads']); lookup={c:i for i,c in enumerate(codes)}
        for control in ['primary','strata']:
            arr=np.array([[lookup[c] for c in sample_basket(x['heads'],x['industries'],len(x['selected']),rng,x['selected'] if control=='strata' else None)] for _ in range(1000)],dtype=np.int32)
            assert all(len(set(row))==len(x['selected']) for row in arr)
            members[f'{j}_{control}']=arr
    freeze(OUT/'frozen_decisions.json',specs)
    memberpath=OUT/'memberships.npz'
    with memberpath.open('xb') as f:np.savez_compressed(f,**members)
    freeze(OUT/'membership_identity.json',{'sha256':sha(memberpath),'decisions_sha256':sha(OUT/'frozen_decisions.json'),'protocol_sha256':sha(OUT/'protocol.json'),'stage':'memberships_frozen_before_outcomes'})
    print('Membership frozen for',len(specs),'decisions;',len(allcodes),'panels',flush=True)
    cal=pd.DatetimeIndex(pd.read_parquet(DATA/manifest['calendar']['path']).session).tz_localize(None).normalize()
    gaps=defaultdict(list)
    for g in manifest.get('data_gaps',[])+manifest.get('accounting_gaps',[]):
        if g.get('instrument_id') and g.get('date'):gaps[g['instrument_id']].append(g['date'])
    requests=defaultdict(list)
    for j,x in enumerate(specs):
        for c in x['heads']:requests[c].append(j)
    labels={};reasons={};endpoints={}
    for k,(c,js) in enumerate(requests.items()):
        e=entries[c];code=c.split('.')[1].lower()+'.'+c.split('.')[0]
        if e:
            p=pd.read_parquet(DATA/e['path'],columns=['feature_close','available_at','quality_status']);p.index=pd.DatetimeIndex(p.index).tz_localize(None).normalize()
            av=pd.to_datetime(p.available_at,errors='coerce');cut=pd.Series(p.index+pd.Timedelta(hours=18),index=p.index)
            prices=p.feature_close.where(av.notna() & av.le(cut) & p.quality_status.eq('ok'))
        else:prices=pd.Series(dtype=float)
        for j in js:
            x=specs[j]
            for h in (20,60):
                v,r,end=forward_label(prices,cal,x['signal_date'],h,x['period_end'],gaps[code]);labels[j,c,h]=v;reasons[j,c,h]=r;endpoints[j,h]=end
        if (k+1)%1000==0:print('Labels',k+1,flush=True)
    rows=[];labelrows=[]
    for j,x in enumerate(specs):
        codes=sorted(x['heads']);lookup={c:i for i,c in enumerate(codes)};sel=[lookup[c] for c in x['selected']];n=len(sel)
        for h in (20,60):
            y=np.array([labels[j,c,h] if labels[j,c,h] is not None else np.nan for c in codes]);sy=y[sel];sk=np.isfinite(sy).all();sm=float(sy.mean()) if sk else None
            for i,c in enumerate(codes):labelrows.append(dict(period=x['period'],signal_date=x['signal_date'],instrument=c,horizon=h,value=labels[j,c,h],reason=reasons[j,c,h],selected=c in x['selected']))
            for control in ['primary','strata']:
                a=members[f'{j}_{control}'];ay=y[a];known=np.isfinite(ay).all(axis=1);means=ay[known].mean(axis=1);pairs=means if sk else np.array([])
                def strata(cs):return [{'tier':tier,'industry':ind,'count':count} for (tier,ind),count in sorted(Counter((x['heads'][c],x['industries'][c]) for c in cs).items())]
                rows.append(dict(period=x['period'],signal_date=x['signal_date'],information_date=x['information_date'],horizon=h,endpoint=endpoints[j,h],control=control,
                    n=n,pool_n=len(codes),retained_count=x['retained_count'],selected_strata=strata(x['selected']),pool_strata=strata(codes),
                    selected_unknown=int((~np.isfinite(sy)).sum()),pool_missing_reasons=dict(Counter(reasons[j,c,h] for c in codes)),
                    random_unknown_names_total=int((~np.isfinite(ay)).sum()),random_full_known_fraction=float(known.mean()),paired_draws=len(pairs),
                    selected_mean=sm,random_conditional_mean=float(pairs.mean()) if len(pairs) else None,
                    selected_minus_random=float(sm-pairs.mean()) if len(pairs) else None,
                    selected_conditional_percentile=float(100*(np.sum(pairs<sm)+.5*np.sum(pairs==sm))/len(pairs)) if len(pairs) else None,
                    selected_lower_bound=float((np.nansum(sy)-np.sum(~np.isfinite(sy)))/n),selected_upper_bound=sm if sk else 'unbounded',
                    random_average_lower_bound=float(np.mean((np.nansum(ay,axis=1)-np.sum(~np.isfinite(ay),axis=1))/n)),
                    random_average_upper_bound=float(ay.mean()) if np.isfinite(ay).all() else 'unbounded'))
    freeze(OUT/'per_date.json',clean(rows))
    pd.DataFrame(labelrows).to_parquet(OUT/'labels.parquet',index=False)
    summaries=[]
    for period,(end,_) in periods.items():
        first=min(x['signal_date'] for x in specs if x['period']==period);calendar=cal[(cal>=pd.Timestamp(first))&(cal<=pd.Timestamp(end))]
        for h in (20,60):
            for control in ['primary','strata']:
                sub=[r for r in rows if r['period']==period and r['horizon']==h and r['control']==control];finite=[r for r in sub if r['selected_minus_random'] is not None]
                stat=bootstrap([(r['signal_date'],r['selected_minus_random']) for r in finite],calendar,h,SEED+h+len(summaries))
                stat.update(period=period,horizon=h,control=control,total_decisions=len(sub),selected_complete_decisions=sum(r['selected_unknown']==0 for r in sub),
                    random_full_known_fraction_all_dates=float(np.mean([r['random_full_known_fraction'] for r in sub])),
                    random_full_known_fraction_paired_dates=float(np.mean([r['random_full_known_fraction'] for r in finite])) if finite else None,
                    median_conditional_percentile=float(np.median([r['selected_conditional_percentile'] for r in finite])) if finite else None,
                    unconditional_effect='not_identified_due_to_missing_outcomes_and_retention_confound')
                summaries.append(stat)
    freeze(OUT/'summary.json',clean(dict(status='completed_conditional_diagnostic',promotion_approved=False,account_replayed=False,model_trained=False,comparisons=summaries,
        old_audit_findings=['draw_stats uses at most 8 duplicate retries, therefore no replacement is not guaranteed','ok=coverage observed precedes both sampling and selected/top-N construction; horizon-specific outcome selection changes baskets','bootstrap draws IID date indices despite overlapping 20/60-session labels','fixed N20 is compared with variable observed selected count; missing selected names renormalized','breadth top N is formed after outcome availability filtering; cannot bound new-expert or width benefit'])))
    freeze(OUT/'result_hashes.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(clean(summaries),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':run()
