"""Frozen heads versus historical quoted-stock market price diagnostic; no accounts."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scripts.router_random_diagnostic import forward_label
from scripts.reference_selection_diagnostic import bootstrap_comparisons

ROOT=Path('/Users/agiuser/Desktop/QuantLab')
DATA=ROOT/'output/account_2025_20260927/extension/dataset'
SEED=20260927

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def freeze(p,value):
    with Path(p).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def json_safe(value):
    """Map NaT/NaN to None so a missing label endpoint serialises as JSON null.

    Crash 2026-09-27T18:04: the final decision dates (2026-07-10/08-07/09-04)
    have no matured 60-session label, so endpoint is NaT and allow_nan=False
    rejected the whole per_date.json. A missing endpoint is a real data state,
    not a numeric result; it must serialise as null and stay countable.
    """
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value,float) and not np.isfinite(value):return None
    if value is pd.NaT:return None
    return value

def local_time(x):
    t=pd.Timestamp(x)
    if pd.isna(t):return pd.NaT
    return t.tz_localize('Asia/Shanghai') if t.tzinfo is None else t.tz_convert('Asia/Shanghai')

def market_members(records, quotes, signal):
    """quotes contains ONLY signal-date rows. No forward labels accepted."""
    cutoff=local_time(signal).normalize()+pd.Timedelta(hours=18)
    r=records.drop_duplicates().copy()
    r['_av']=r.available_at.map(local_time);r['_ef']=r.effective_at.map(local_time)
    r=r[r._av.le(cutoff)&r._ef.le(cutoff)]
    if r.duplicated(['instrument_id','_av','_ef']).any():raise ValueError('ambiguous_instrument_state')
    r=r.sort_values(['_ef','_av','revision']).drop_duplicates('instrument_id',keep='last')
    states=r.set_index('instrument_id'); members=[];reasons=Counter()
    for code in sorted(set(records.instrument_id)|set(quotes)):
        if code not in states.index:reason='no_asof_state'
        elif states.loc[code,'asset_type']!='stock':reason='not_stock'
        elif not bool(states.loc[code,'active']):reason='inactive'
        elif code not in quotes:reason='no_signal_quote'
        else:
            q=quotes[code];av=local_time(q['available_at']);v=q['feature_close']
            if pd.isna(av) or av>cutoff:reason='late_or_unknown_quote'
            elif q['quality_status']!='ok':reason='bad_quote_quality'
            elif not np.isfinite(v) or v<=0:reason='invalid_quote'
            else:reason='included';members.append(code)
        reasons[reason]+=1
    return members,dict(reasons)

def valid_prices(frame):
    p=frame.copy();p.index=pd.DatetimeIndex(p.index).tz_localize(None).normalize()
    if not p.index.is_unique:raise ValueError('duplicate_price_session')
    # Local source timestamps may be naive or timezone aware.
    av=p.available_at.map(local_time)
    cuts=pd.Series(pd.DatetimeIndex(p.index).tz_localize('Asia/Shanghai')+pd.Timedelta(hours=18),index=p.index)
    return p.feature_close.where(av.notna()&av.le(cuts)&p.quality_status.eq('ok'))

def summarize_values(values, statistic):
    a=np.array(values,dtype=float);known=a[np.isfinite(a)];n=len(a);missing=n-len(known)
    if not n:raise ValueError('empty_membership')
    fn=np.mean if statistic=='mean' else np.median
    lower=float(fn(np.r_[known,np.full(missing,-1.)]))
    upper=float(fn(np.r_[known,np.full(missing,np.inf)]))
    return dict(statistic=statistic,requested=n,observed=len(known),missing=missing,coverage=len(known)/n,
                conditional=float(fn(known)) if len(known) else None,lower_bound=lower,
                upper_bound=upper if np.isfinite(upper) else 'unbounded')

def compare_values(heads, market):
    h=summarize_values(heads,'mean');m=summarize_values(market,'median')
    return dict(heads=h,market=m,conditional_difference=h['conditional']-m['conditional'] if h['conditional'] is not None and m['conditional'] is not None else None,
                lower_bound=h['lower_bound']-m['upper_bound'] if m['upper_bound']!='unbounded' else '-unbounded',
                upper_bound=h['upper_bound']-m['lower_bound'] if h['upper_bound']!='unbounded' else 'unbounded',
                heads_median=summarize_values(heads,'median'),market_mean=summarize_values(market,'mean'))

def skew_robust_contrasts(heads, market):
    """Pre-registered secondary statistics that do not depend on right skew.

    The primary statistic is heads MEAN minus market MEDIAN, which the protocol
    itself flags as 'sensitive to right skew, not portfolio alpha'. These two
    are the decomposition that keeps that honest, both on the same conditional
    observed values and the same original denominators:
      location_median = heads median - market median  (skew-free on both sides)
      location_mean   = heads mean   - market mean    (skew-matched)
    Identity: primary = location_mean + (market mean - market median), so the
    market's own right tail is separable and countable rather than implied.
    """
    a=np.array(heads,dtype=float);b=np.array(market,dtype=float)
    ka=a[np.isfinite(a)];kb=b[np.isfinite(b)]
    if not len(ka) or not len(kb):
        return dict(location_median=None,location_mean=None,market_skew=None,heads_median=None,
                    market_median=None,heads_mean=None,market_mean=None)
    hm,bm=float(np.median(ka)),float(np.median(kb))
    ha,ba=float(np.mean(ka)),float(np.mean(kb))
    return dict(location_median=hm-bm,location_mean=ha-ba,market_skew=ba-bm,
                heads_median=hm,market_median=bm,heads_mean=ha,market_mean=ba)

def identifier(c):
    number,exchange=c.split('.');return exchange.lower()+'.'+number

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    manifest_path=DATA/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if sha(manifest_path)!='4c2f52a03edb6e79963bb6a889fe69087efd1c466a2017b22be253f94a98136f':raise ValueError('manifest_identity_changed')
    source=Path(__file__).resolve();base=source.parents[1]
    decision_path=ROOT/'output/expert_selection_20260927/random_diagnostic/frozen_decisions.json'
    paths=[source,base/'tests/test_reference_pool_market_diagnostic.py',base/'scripts/router_random_diagnostic.py',base/'scripts/reference_selection_diagnostic.py',manifest_path,decision_path,
           ROOT/'output/reference_selection_20260927/PLAN.md',ROOT/'output/reference_selection_20260927/AMENDMENT.md']
    for key in ['instruments','calendar']:
        e=manifest[key];p=DATA/e['path']
        if sha(p)!=e['sha256']:raise ValueError(key+'_hash_mismatch')
        paths.append(p)
    identities={str(p):sha(p) for p in paths}
    assert len(manifest['panels'])==6040
    for code,e in manifest['panels'].items():
        p=DATA/e['path'];digest=sha(p)
        if digest!=e['sha256']:raise ValueError('panel_hash_mismatch:'+code)
        identities[str(p)]=digest
    specs=json.loads(decision_path.read_text());assert len(specs)==70
    freeze(out/'protocol.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=identities,panel_count=6040,seed=SEED,horizons=[20,60],family_tests=12,bootstrap_draws=10000,
        unresolved_question='Does archived heads price behavior differ from local historical active quoted-stock distribution?',
        decision='Inform pool investigation only; no financial-tier causal effect, strategy adoption or account replay.',
        account_conditions='600000 CNY; at most 50; long-only no leverage. This diagnostic has no capital, costs or execution.',
        membership='As-of effective AND available instrument state stock/active; valid positive same-day quote available by local 18:00. No future coverage/survival filters.',
        statistic='PRIMARY heads observed equal-weight mean minus market observed median; original memberships preserved. PRE-REGISTERED SECONDARIES on the same conditional observed values and same original denominators: location_median = heads median minus market median (skew-free both sides), location_mean = heads mean minus market mean (skew-matched), plus market_own_skew = market mean minus market median. Identity primary = location_mean + market_own_skew, so the broad market right tail is separable and countable rather than implied. Secondaries are reported alongside the primary, never instead of it.',
        crash_and_fix='Attempt 1 (2026-09-27T18:04) crashed in freeze(per_date.json) with allow_nan=False: the final decision dates 2026-07-10/2026-08-07/2026-09-04 have no matured 60-session label so endpoint is NaT. Fix maps NaT/NaN to JSON null; a missing endpoint is a real data state, stays countable via missing_endpoint_dates, and is not imputed.',
        bootstrap='sparse circular calendar-session blocks, 20/60 sessions, 10000 resamples; four tests join eight sibling tests, family=12',
        bounds='Missing positive-price returns in [-1,+infinity); mean/median bounds use original denominator. No imputation in estimates.',
        limitations=['historical lifecycle/listing/BSE proxies, not strict PIT','heads both financial tiers and SH/SZ/BJ','known-data conditional prices, not total returns, net returns or index','mean-minus-median sensitive to right skew, not portfolio alpha','overlapping periods and viewed history not OOS']))
    (out/'source').mkdir()
    for p in paths[:4]:shutil.copyfile(p,out/'source'/p.name)
    cal=pd.DatetimeIndex(pd.read_parquet(DATA/manifest['calendar']['path']).session).tz_localize(None).normalize()
    if not cal.is_unique or not cal.is_monotonic_increasing:raise ValueError('invalid_calendar')
    records=pd.read_parquet(DATA/manifest['instruments']['path'])
    days=sorted(set(x['signal_date'] for x in specs));quotes={d:{} for d in days}
    for i,(code,e) in enumerate(manifest['panels'].items()):
        # Predicate limits the data read to signal rows before global membership freeze.
        p=pd.read_parquet(DATA/e['path'],columns=['feature_close','available_at','quality_status'],filters=[('event_time','in',[pd.Timestamp(d) for d in days])])
        if not p.index.is_unique:raise ValueError('duplicate_signal_quote')
        for date,q in p.iterrows():quotes[str(pd.Timestamp(date).date())][code]=q.to_dict()
        if (i+1)%1000==0:print('signal-only quote panels',i+1,flush=True)
    market={d:market_members(records,quotes[d],d) for d in days};frozen=[]
    for j,x in enumerate(specs):
        assert x['information_date']<=x['signal_date']
        heads=sorted(identifier(c) for c in x['heads']);members,reasons=market[x['signal_date']]
        frozen.append(dict(decision_id=j,period=x['period'],signal_date=x['signal_date'],information_date=x['information_date'],period_end=x['period_end'],heads=heads,heads_tiers={identifier(c):v for c,v in x['heads'].items()},market=members,membership_reasons=reasons,
            heads_exchanges=dict(Counter(c.split('.')[0] for c in heads)),market_exchanges=dict(Counter(c.split('.')[0] for c in members))))
    freeze(out/'memberships.json',frozen)
    freeze(out/'membership_identity.json',dict(created_at=datetime.now(timezone.utc).isoformat(),stage='global_memberships_frozen_before_full_price_reads',membership_sha256=sha(out/'memberships.json'),protocol_sha256=sha(out/'protocol.json')))
    print('All 70 memberships frozen; now reading full price paths',flush=True)
    requests=defaultdict(list);gaps=defaultdict(list)
    for x in frozen:
        for c in sorted(set(x['heads'])|set(x['market'])):requests[c].append(x)
    for g in manifest.get('data_gaps',[])+manifest.get('accounting_gaps',[]):
        if g.get('instrument_id') and g.get('date'):gaps[g['instrument_id']].append(g['date'])
    labelrows=[]
    for i,(code,rs) in enumerate(sorted(requests.items())):
        e=manifest['panels'].get(code)
        prices=valid_prices(pd.read_parquet(DATA/e['path'],columns=['feature_close','available_at','quality_status'])) if e else pd.Series(dtype=float)
        for x in rs:
            for h in [20,60]:
                v,reason,end=forward_label(prices,cal,x['signal_date'],h,x['period_end'],gaps[code])
                labelrows.append(dict(decision_id=x['decision_id'],instrument_id=code,horizon=h,value=v,reason=reason,endpoint=end))
        if (i+1)%500==0:print('full-path label panels',i+1,flush=True)
    labels=pd.DataFrame(labelrows);labels.to_parquet(out/'labels.parquet',index=False)
    rows=[]
    for x in frozen:
        for h in [20,60]:
            ls=labels[(labels.decision_id==x['decision_id'])&(labels.horizon==h)].set_index('instrument_id')
            hs=ls.loc[x['heads']];ms=ls.loc[x['market']]
            hv=hs.value.to_numpy();mv=ms.value.to_numpy()
            result=compare_values(hv,mv)
            result.update(skew=skew_robust_contrasts(hv,mv),
                          decision_id=x['decision_id'],period=x['period'],signal_date=x['signal_date'],information_date=x['information_date'],horizon=h,endpoint=ls.endpoint.iloc[0],heads_reason_counts=dict(Counter(hs.reason)),market_reason_counts=dict(Counter(ms.reason)))
            rows.append(result)
    freeze(out/'per_date.json',json_safe(rows));summary=[]
    for period in ['2022_2025','2025_recent']:
        ps=[x for x in frozen if x['period']==period];pc=cal[(cal>=pd.Timestamp(ps[0]['signal_date']))&(cal<=pd.Timestamp(ps[0]['period_end']))]
        for h in [20,60]:
            rr=[r for r in rows if r['period']==period and r['horizon']==h];values=[(r['signal_date'],r['conditional_difference']) for r in rr if r['conditional_difference'] is not None]
            stat=bootstrap_comparisons(values,pc,h,SEED,family_tests=12);ci=stat['ci_family95_pp']
            stat.update(period=period,horizon=h,inference='conditional_observed_heads_mean_minus_market_median; low_power; viewed_overlapping_history',total_dates=len(rr),heads_complete_dates=sum(r['heads']['missing']==0 for r in rr),market_complete_dates=sum(r['market']['missing']==0 for r in rr),
                heads_requested=sum(r['heads']['requested'] for r in rr),heads_observed=sum(r['heads']['observed'] for r in rr),market_requested=sum(r['market']['requested'] for r in rr),market_observed=sum(r['market']['observed'] for r in rr),
                heads_reason_counts=dict(sum((Counter(r['heads_reason_counts']) for r in rr),Counter())),market_reason_counts=dict(sum((Counter(r['market_reason_counts']) for r in rr),Counter())),
                missing_endpoint_dates=sum(r['endpoint'] is None or pd.isna(r['endpoint']) for r in rr),
                unusable_dates=len(rr)-len(values),
                judgment='insufficient' if ci is None else 'positive_conditional' if ci[0]>0 else 'negative_conditional' if ci[1]<0 else 'inconclusive',promotion_approved=False)
            # Pre-registered skew-robust secondaries. Same family=12, same
            # calendar blocks, same dates. Reported alongside the primary, never
            # instead of it.
            sec={}
            for key in ['location_median','location_mean']:
                vv=[(r['signal_date'],r['skew'][key]) for r in rr if r['skew'][key] is not None]
                s2=bootstrap_comparisons(vv,pc,h,SEED,family_tests=12);c2=s2['ci_family95_pp']
                s2.update(observed_dates=s2['observed_dates'],
                          judgment='insufficient' if c2 is None else 'positive_conditional' if c2[0]>0 else 'negative_conditional' if c2[1]<0 else 'inconclusive',
                          primary_at_this_cell=stat['mean_pp'],
                          market_own_skew_pp=float(np.mean([r['skew']['market_skew'] for r in rr if r['skew']['market_skew'] is not None])*100) if any(r['skew']['market_skew'] is not None for r in rr) else None)
                sec[key]=s2
            stat['skew_robust']=sec
            summary.append(stat)
    freeze(out/'summary.json',json_safe(summary))
    # Recheck all input bytes after outcomes to detect concurrent modification.
    changed=[p for p,d in identities.items() if sha(p)!=d]
    if changed:raise ValueError('inputs_changed:'+str(changed))
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='run.log']
    freeze(out/'complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),all_input_hashes_rechecked=len(identities),changed_inputs=[],outputs={str(p.relative_to(out)):sha(p) for p in files},label_rows=len(labels),per_date_rows=len(rows),promotion_approved=False))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();run(args.out)
