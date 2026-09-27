"""Frozen R/S and R/A1/A2 account study, using the existing reference ledger.

Preparation freezes inputs before making targets. The RS stage can finish while
nonlinear models train; the all stage reuses RS accounts and adds only A1/A2.
This is a conditional research experiment, never a production entry point.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import wraps
import fcntl
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import continuity_cadence as shared

# The managed checkout shares output through a symlink. Resolve before any
# frozen imports so the existing no-mixed-code guard compares physical paths.
shared.FROZEN = shared.FROZEN.resolve()
import reference_expert_router as base
from quantlab.data.contracts import decision_time, timestamp

reference, replay, router = base.reference, base.replay, base.router
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/expert_selection_20260927'
PERIODS = base.PERIODS
ARMS = ('R', 'S', 'A1', 'A2')
POLICIES = dict(R='reference_retention', S='score_consistent_retention',
                A1='reference_retention', A2='reference_retention')
VERSION = 'expert-selection-study-v1'
PARENT_VERSION = '6a53f151a1e545857f6925173891ac91a1e06a88bb0de8a7ef34ac6fcd9b7deb'


def write(path, value):
    shared.freeze_json(path, value)


def progress(stage, **kwargs):
    value = dict(stage=stage, at=datetime.now(timezone.utc).isoformat(), **kwargs)
    shared.freeze_json(OUT/'study_status.json', value, immutable=False)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def exclusive(function):
    """One writer across both stages; the OS releases a crashed process lock."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        path=OUT/'study'/'writer.lock'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a+') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('study_writer_already_running') from exc
            handle.seek(0);handle.truncate()
            json.dump(dict(pid=os.getpid(),action=function.__name__,at=datetime.now(timezone.utc).isoformat()),handle)
            handle.flush();os.fsync(handle.fileno())
            try:return function(*args,**kwargs)
            finally:fcntl.flock(handle,fcntl.LOCK_UN)
    return wrapped


def symbol(code):
    parts = str(code).split('.')
    return parts[1]+'.'+parts[0].upper() if len(parts)==2 and parts[0] in ('sz','sh','bj') else str(code)


class ExpandedExperts:
    """Six-expert route with complete-cohort exclusion and whole-decision R fallback.

    ``predictions`` contains only previously generated training-out forecasts,
    one artifact per signal date. No fitted training values enter this adapter.
    Existing five-expert finite membership is decided before sixth coverage.
    """
    def __init__(self, scored, calendar, benchmark, predictions, *, policy=router.RouterPolicy()):
        self.baseline = base.ArchivedExperts(scored, calendar, benchmark, policy=policy)
        self.policy = policy
        self.predictions = {pd.Timestamp(d): v for d,v in predictions.items()}
        self.columns = self.baseline.columns + ['score_nonlinear']
        self.blocks, self.status = {}, {}
        for day, original in self.baseline.blocks.items():
            frame = original.copy()
            frame['score_nonlinear'] = np.nan
            entry = self.predictions.get(day)
            reason = 'missing_nonlinear_snapshot'
            if entry is not None:
                evidence = entry['evidence']
                reason = evidence.get('reason', 'nonlinear_training_unavailable')
                if evidence.get('status') == 'ok':
                    cutoff = timestamp(evidence['as_of'])
                    latest = timestamp(evidence['latest_label_available_at'])
                    train_dates = [pd.Timestamp(d).normalize() for d in evidence['cohort_dates']]
                    if cutoff != decision_time(day) or latest > cutoff or not train_dates or max(train_dates) >= day:
                        raise ValueError('nonlinear_prediction_looks_ahead:'+str(day.date()))
                    values = pd.Series(entry['predictions'], dtype=float)
                    if values.index.has_duplicates:
                        raise ValueError('duplicate_nonlinear_prediction')
                    known = values.dropna()
                    if not np.isfinite(known).all() or not known.between(0,1).all():
                        raise ValueError('invalid_nonlinear_rank')
                    frame['score_nonlinear'] = values.reindex(frame.index)
                    required = np.isfinite(original[self.baseline.columns]).all(axis=1)
                    complete = np.isfinite(frame.loc[required, 'score_nonlinear']).all()
                    reason = 'ok' if complete else 'missing_nonlinear_common_members'
            self.blocks[day], self.status[day] = frame, reason
        cohorts = []
        for old in self.baseline.model.cohorts:
            day = old.signal_date.tz_convert('Asia/Shanghai').tz_localize(None).normalize()
            frame = self.blocks[day]
            if self.status[day] != 'ok':
                # Keep the original requested/observed counts. Do not let a
                # partially present sixth prediction make coverage look better.
                cohort = replace(old, experts=tuple(self.columns), gram=np.zeros((6,6)),
                                 cross=np.zeros(6), status=self.status[day])
            else:
                cohort = router.cohort_from_outcomes(old.signal_date, old.available_at,
                    old.state, frame[self.columns],
                    frame.return_20.where(frame.coverage_20.eq('observed')), policy)
                if cohort.requested != old.requested:
                    raise ValueError('sixth_expert_changed_common_membership')
            cohorts.append(cohort)
        self.model = router.ExpertRouter(cohorts, policy)

    def predict(self, day):
        original, meta = self.baseline.predict(day, 'conditional_router')
        def fallback(reason, **extra):
            return original, dict(meta, status='fallback_R', fallback_reason=reason,
                                  baseline_status=meta['status'], **extra)
        if meta['status'] != 'ok':
            return fallback('R_'+meta['status'])
        date = pd.Timestamp(meta['score_date'])
        if self.status[date] != 'ok':
            return fallback(self.status[date])
        decision = self.model.decide(decision_time(day), meta['state'], self.blocks[date][self.columns])
        if decision.status != 'ok':
            return fallback(decision.status, nonlinear_learning=decision.evidence)
        evidence = self.predictions[date]['evidence']
        return ({symbol(c):float(v) for c,v in decision.scores.items()},
            dict(meta, status='ok', weights=decision.weights, learning=decision.evidence,
                 nonlinear_model=dict(version=evidence['version'], train_cutoff=evidence['as_of'],
                     last_label_available_at=evidence['latest_label_available_at'],
                     source_date=str(date.date()))))


def build_targets(inputs, sessions, provider, arm):
    """Common original target cadence, separate retention policy, explicit trace."""
    if arm not in ARMS:
        raise ValueError('unknown_study_arm')
    days = [str(pd.Timestamp(d).date()) for d in sessions]
    if not days or days != sorted(set(days)):
        raise ValueError('invalid_target_sessions')
    heads, groups = inputs['heads'], inputs['industries']
    if set(heads) != set(groups):
        raise ValueError('reference_group_dates_mismatch')
    earlier = sorted(d for d in heads if d < days[0])
    seed = earlier[-1] if earlier else None
    basket, targets, traces = (), [], []
    for day in days:
        source = day if day in heads else seed if day == days[0] else None
        if source is None:
            continue
        priorities, evidence = provider(source)
        original = router.select_reference_basket(heads[source], basket, groups[source])
        selected = router.select_reference_basket(heads[source], basket, groups[source],
            priorities=priorities, selection_policy=POLICIES[arm])
        before, after = set(basket), set(selected)
        retained, additions, removals = sorted(before & after), sorted(after-before), sorted(before-after)
        reason = {c: dict(action=('retain' if c in before else 'admit'),
            financial_tier=heads[source][c], score=priorities.get(c),
            industry=groups[source].get(c,'unclassified'),
            reason=('selected_by_'+POLICIES[arm] if c in priorities else 'original_missing_score_slot'))
            for c in selected}
        reason.update({c:dict(action='remove', reason=('outside_current_candidates' if c not in heads[source]
                       else 'retention_or_rank_or_industry_slot'), score=priorities.get(c)) for c in removals})
        traces.append(dict(signal_date=day, information_date=source, warm_start=source<day,
            arm=arm, selection_policy=POLICIES[arm], head_count=len(heads[source]),
            scored_candidates=sum(c in priorities for c in heads[source]), selected=list(selected),
            previous_target=list(basket), retained=retained, retained_count=len(retained),
            additions=additions, removals=removals, original_with_same_prior_basket=list(original),
            changed_slots=len(after-set(original)), reasons=reason, router=evidence))
        basket = selected
        targets.append(dict(signal_date=day, weights={c:.9/len(basket) for c in sorted(basket)},
                            sell_only=False, max_exposure=.9))
    return targets, traces


def load_predictions(kind):
    root = OUT/'models'/kind
    complete = json.loads((root/'complete.json').read_text())
    shared.verify_file(root/'protocol.json', complete['protocol_sha256'])
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol['kind'] != kind or protocol['features_complete_sha256'] != shared.sha256(OUT/'features_complete.json'):
        raise ValueError('nonlinear_model_input_identity_changed')
    result, sources = {}, {str(root/'complete.json'):shared.sha256(root/'complete.json'),
                          str(root/'protocol.json'):complete['protocol_sha256']}
    for date, item in complete['predictions'].items():
        folder = Path(item['path'])
        shared.verify_file(folder/'complete.json',item['complete_sha256'])
        sources[str(folder/'complete.json')] = item['complete_sha256']
        month = json.loads((folder/'complete.json').read_text())
        for name, digest in month['files'].items():
            shared.verify_file(folder/name,digest)
            sources[str(folder/name)] = digest
        evidence = json.loads((folder/'evidence.json').read_text())
        if evidence.get('status') != month['status']:
            raise ValueError('prediction_status_mismatch')
        values = pd.Series(dtype=float)
        if month['status'] == 'ok':
            frame = pd.read_parquet(folder/'predictions.parquet')
            if set(frame.date.astype(str)) != {date}:
                raise ValueError('prediction_vintage_mismatch')
            values = frame.set_index('instrument_id').prediction
        result[pd.Timestamp(date)] = dict(evidence=evidence, predictions=values)
    return result, sources


def source_files():
    return [Path(__file__), ROOT/'quantlab/strategy/expert_router.py', Path(base.__file__),
        Path(shared.__file__), Path(reference.__file__), Path(replay.__file__),
        ROOT/'tests/test_expert_selection_study.py', OUT/'PLAN.md',
        OUT/'features_complete.json', OUT/'features_protocol.json', reference.MANIFEST,
        base.OUT/'prepared.json'] + [reference.input_path(p) for p in PERIODS] + [
        base.OUT/p/name for p in PERIODS for name in
        ('conditional_router_targets.json','conditional_router/complete.json')]


def stage_dir(stage):
    if stage not in ('rs','all'):
        raise ValueError('unknown_study_stage')
    return OUT/'study'/stage


@exclusive
def prepare(stage='rs'):
    dest = stage_dir(stage)
    if (dest/'prepared.json').exists():
        verify(stage)
        progress('preparation_already_complete', study_stage=stage)
        return
    if stage == 'all':
        verify('rs')
    feature = json.loads((OUT/'features_complete.json').read_text())
    shared.verify_file(OUT/'features_protocol.json',feature['protocol_sha256'])
    feature_protocol=json.loads((OUT/'features_protocol.json').read_text())
    shared.verify_file(reference.MANIFEST,feature_protocol['manifest_sha256'])
    scores = Path(feature['router_scores']['path'])
    shared.verify_file(scores, feature['router_scores']['sha256'])
    files = source_files()+[scores]
    sources = {str(p):shared.sha256(p) for p in files}
    predictions = {}
    if stage == 'all':
        for arm,kind in (('A1','plain'),('A2','double')):
            predictions[arm], extra = load_predictions(kind)
            sources.update(extra)
    old_protocol = base.OUT/'protocol.json'
    sources[str(old_protocol)] = shared.sha256(old_protocol)
    old = json.loads(old_protocol.read_text())
    for entry in old['baselines'].values():
        for name,digest in entry['files'].items():
            shared.verify_file(Path(entry['path'])/name,digest)
    arms = ['R','S'] if stage=='rs' else list(ARMS)
    protocol = dict(version=VERSION, stage=stage, arms=arms, periods=PERIODS,
        created_at=datetime.now(timezone.utc).isoformat(),
        unresolved=['Q5_new_nonlinear_information','Q5_score_retention_mismatch'],
        hypothesis='S tests unified retention independently; A1/A2 test nonlinear prediction and adaptive ensemble independently. No combined arm.',
        result_changes='No net increment closes frozen configuration; positive but uncertain increment stays shadow; replacement requires every predeclared profit, risk, evidence, stress, data and forward gate.',
        sources=sources, baselines=old['baselines'], frozen_source=shared.source_identity(),
        parent_data_version=PARENT_VERSION, profile=asdict(replay.ReplayProfile.common(50)),
        router_policy=asdict(router.RouterPolicy()), policies=POLICIES,
        account='60w,50cap,90pct,20session original phase,150buffer,15industry seats; original PB/financial tiers/ADV; no P2 execution',
        data='74 old score dates reproduced exactly; July/Aug2026 added commonly; no unfinished September score',
        sixth_coverage='All original five-expert finite members required; unavailable cohort excluded without shrinking membership; whole decision fallback R',
        training='Monthly prior36 matured cohorts; 20session price-return rank; only training-out forecasts enter historical six-expert routing',
        statistic=dict(samples=10000,seed=20260927,blocks=[20,60],confidence=.975,
            correction='max absolute t within period; two periods Bonferroni; overlapping windows not independent',
            pairs='A1,A2,S each minus R/reference50/reference100; A2 minus A1',
            dsr='208 known lower bound only; incomplete historical search; sensitivity not pass'),
        adoption=dict(long_cagr=.15,max_drawdown=.25,positive_annual_increments=3,
            years=[2022,2023,2024,2025],positive_net=True,both_blocks=True,
            cost2='must remain profitable',stress='liquidity0.3,5exit-session locks,execution delay',
            data='unverified',forward='six months frozen required'),
        dependencies={m.__name__:m.__version__ for m in (np,pd,__import__('scipy'))},
        promotion_approved=False)
    if (dest/'protocol.json').exists():
        prior = json.loads((dest/'protocol.json').read_text())
        for key in set(protocol)-{'created_at'}:
            if shared.digest_json(prior[key]) != shared.digest_json(protocol[key]):
                raise ValueError('preparation_resume_identity_changed:'+key)
    else:
        write(dest/'protocol.json',protocol)
        for p in files:
            if p.suffix=='.py':
                shared.publish(dest/'source'/p.relative_to(ROOT), lambda h,p=p:h.write(p.read_bytes()))
    m = json.loads(reference.MANIFEST.read_text())
    if m.get('data_version',m.get('version')) not in (None,PARENT_VERSION):
        raise ValueError('parent_data_version_changed')
    cal = pd.DatetimeIndex(reference.read_table(reference.MANIFEST.parent,m['calendar']).session)
    bench = base.dated(reference.read_table(reference.MANIFEST.parent,m['benchmark']))
    scored = pd.read_parquet(scores)
    baseline = base.ArchivedExperts(scored,cal,bench)
    providers = {arm:ExpandedExperts(scored,cal,bench,p) for arm,p in predictions.items()}
    target_files, reused_rs = {}, []
    for period,(start,end) in PERIODS.items():
        days = cal[(cal>=start)&(cal<=end)]
        inputs = json.loads(reference.input_path(period).read_text())
        plain,_ = build_targets(inputs,days,lambda d:({},{}),'R')
        if plain != reference.build_targets(inputs,days,50):
            raise ValueError('reference_target_parity_failed:'+period)
        for arm in arms:
            if stage=='all' and arm in ('R','S'):
                path = stage_dir('rs')/period/(arm+'_targets.json')
                target_files[str(path)] = shared.sha256(path); reused_rs.append(str(path))
                continue
            provider = (providers[arm].predict if arm in providers else
                        lambda d:baseline.predict(d,'conditional_router'))
            targets,trace = build_targets(inputs,days,provider,arm)
            path = dest/period/(arm+'_targets.json')
            payload = dict(targets=targets,trace=trace,arm=arm,selection_policy=POLICIES[arm],
                model_version=VERSION,study_protocol_sha256=shared.sha256(dest/'protocol.json'))
            if path.exists():
                if json.loads(path.read_text()) != payload:
                    raise ValueError('prepared_targets_changed:'+str(path))
            else:
                write(path,payload)
            target_files[str(path)] = shared.sha256(path)
    write(dest/'prepared.json',dict(targets=target_files,reused_rs=reused_rs,
        protocol_sha256=shared.sha256(dest/'protocol.json'),original_target_parity=True))
    progress('prepared',study_stage=stage,target_files=len(target_files))


def verify(stage):
    dest = stage_dir(stage)
    protocol = json.loads((dest/'protocol.json').read_text())
    prepared = json.loads((dest/'prepared.json').read_text())
    shared.verify_file(dest/'protocol.json',prepared['protocol_sha256'])
    for path,digest in {**protocol['sources'],**prepared['targets']}.items():
        shared.verify_file(path,digest)
    if shared.source_identity()!=protocol['frozen_source']:
        raise ValueError('frozen_ledger_changed')
    for entry in protocol['baselines'].values():
        for name,digest in entry['files'].items():
            shared.verify_file(Path(entry['path'])/name,digest)
    return protocol,prepared


def target_path(stage,period,arm):
    origin = 'rs' if arm in ('R','S') else stage
    return stage_dir(origin)/period/(arm+'_targets.json')


def account_folder(stage,period,arm):
    origin = 'rs' if arm in ('R','S') else stage
    return stage_dir(origin)/period/arm


def execution_identity(targets,period,protocol):
    return dict(targets=shared.digest_json(targets),profile=protocol['profile'],period=list(PERIODS[period]),
        parent_manifest=shared.sha256(reference.MANIFEST),ledger=protocol['frozen_source']['code_digest'],
        replay=shared.sha256(replay.__file__),cost_mult=1.)


def can_reuse(wanted, existing):
    return shared.digest_json(wanted)==shared.digest_json(existing)


def verify_complete(folder, identity=None):
    complete=json.loads((folder/'complete.json').read_text())
    for name,digest in complete.items():
        shared.verify_file(folder/name,digest)
    if identity is not None:
        previous=json.loads((folder/'identity.json').read_text())
        if not can_reuse(identity,previous):
            raise ValueError('account_resume_identity_changed:'+str(folder))
    return complete


def old_reusable(period,targets,protocol):
    """Prove old R target and execution identity, rather than trusting its label."""
    folder=base.OUT/period/'conditional_router'
    complete=verify_complete(folder)
    oldp=json.loads((base.OUT/'protocol.json').read_text())
    oldprep=json.loads((base.OUT/'prepared.json').read_text())
    shared.verify_file(base.OUT/'protocol.json',oldprep['protocol_sha256'])
    oldpath=base.OUT/period/'conditional_router_targets.json'
    # Old protocol paths refer to canonical checkout, whose bytes are shared.
    expected=next((v for k,v in oldprep['targets'].items() if Path(k).name==oldpath.name
                   and Path(k).parent.name==period),None)
    if expected is None:return None
    shared.verify_file(oldpath,expected)
    oldtargets=json.loads(oldpath.read_text())['targets']
    oldresult=shared.checked_pickle(folder/'result.pkl',complete['result.pkl'])
    if oldresult['protocol']['target_schedule_sha256'] != replay.fingerprint(replay.read_targets({'targets':oldtargets})[1]):
        raise ValueError('old_account_target_identity_mismatch')
    oldmanifest=next(v for k,v in oldp['sources'].items() if k.endswith('extension/dataset/manifest.json'))
    oldreplay=next(v for k,v in oldp['sources'].items() if k.endswith('/reference_challenge_replay.py'))
    oldidentity=dict(targets=shared.digest_json(oldtargets),profile=oldresult['protocol']['profile'],
        period=list(oldp['periods'][period]),parent_manifest=oldmanifest,
        ledger=oldp['frozen_source']['code_digest'],replay=oldreplay,
        cost_mult=oldresult['protocol']['cost_multiplier'])
    if oldresult.get('parent_data_version')!=PARENT_VERSION:return None
    return (folder,oldresult) if can_reuse(execution_identity(targets,period,protocol),oldidentity) else None


def preserve_partial(folder, identity, payload):
    """Keep interruption evidence; reuse a fully written result if it matches."""
    if not folder.exists() or not any(folder.iterdir()):return None,None
    archive=folder.with_name(folder.name+'.interrupted.'+str(time.time_ns()))
    folder.rename(archive)
    progress('partial_account_preserved',path=str(archive))
    result=None
    if (archive/'result.pkl').exists():
        try:
            candidate=shared.checked_pickle(archive/'result.pkl',shared.sha256(archive/'result.pkl'))
            p=candidate['protocol']
            matches=(candidate['study_protocol_sha256']==payload['study_protocol_sha256'] and
                candidate['parent_data_version']==PARENT_VERSION and p['profile']==identity['profile'] and
                [str(p['start']),str(p['end'])]==identity['period'] and p['cost_multiplier']==identity['cost_mult'] and
                p['target_schedule_sha256']==replay.fingerprint(replay.read_targets(payload)[1]))
            if matches:result=candidate
        except (ValueError,KeyError,EOFError,pickle.UnpicklingError,TypeError):
            # Truncated local output remains archived. No claim of a completed
            # account; execution must resume by rerunning that one account.
            pass
    return archive,result


def projection(stage,prepared):
    """Load target-name execution panels, with restart-safe immutable evidence."""
    from quantlab.data.contracts import MarketDataset
    m=json.loads(reference.MANIFEST.read_text())
    codes=sorted({replay.canonical_code(c) for path in prepared['targets']
        for target in json.loads(Path(path).read_text())['targets'] for c in target['weights']})
    if set(codes)-set(m['panels']):raise ValueError('target_execution_panels_missing')
    read=lambda entry:reference.read_table(reference.MANIFEST.parent,entry)
    panels={c:base.dated(read(m['panels'][c])) for c in codes}
    records,actions=read(m['instruments']),read(m['actions'])
    manifest=dict(m,panels={c:m['panels'][c] for c in codes},execution_projection=dict(
        parent_manifest_sha256=shared.sha256(reference.MANIFEST),codes=codes))
    ds=MarketDataset(panels,records[records.instrument_id.isin(codes)].copy(),
        pd.DatetimeIndex(read(m['calendar']).session),actions[actions.instrument_id.isin(codes)].copy(),
        base.dated(read(m['benchmark'])),manifest=manifest)
    errors=ds.validate('free_research')
    if errors:raise ValueError(str(errors))
    path=stage_dir(stage)/'execution_projection.json'
    evidence=dict(codes=codes,panels=len(codes),rows=sum(map(len,panels.values())),version=ds.version,
                  role='execution-only target projection; historical candidate pool unchanged')
    if path.exists():
        if json.loads(path.read_text())!=evidence:raise ValueError('projection_identity_changed')
    else:write(path,evidence)
    return ds


def diagnostics(result, trace, returns):
    """Descriptive holdings/selection attribution; no additive causal claim."""
    nav=pd.Series({pd.Timestamp(x['date']):float(x['nav']) for x in result['account']})
    initial=float(result['config']['capital']['init_cash'])
    high, peak, duration, episodes = initial, nav.index[0], 0, []
    for day,value in nav.items():
        if value>=high:
            if duration:episodes.append(dict(peak=str(peak.date()),recovery=str(day.date()),
                                           sessions=duration,recovered=True))
            high,peak,duration=value,day,0
        else:duration+=1
    if duration:episodes.append(dict(peak=str(peak.date()),recovery=None,sessions=duration,recovered=False))
    halves={f'{year}H{half}':float((1+g).prod()-1)
        for (year,half),g in returns.groupby([returns.index.year,(returns.index.month>6).astype(int)+1])}
    shifts=[]
    traces={t['signal_date']:t for t in trace}
    for row in result['signal_trace']:
        t=traces[row['date']]
        actual=set(row['actual_weights']); targets={replay.canonical_code(c) for c in t['selected']}
        shifts.append(dict(date=row['date'],target_count=len(targets),actual_count=len(actual),
            target_not_held=sorted(targets-actual),held_not_target=sorted(actual-targets),
            timing='signal close before future fills',orders=len(row['orders'])))
    # Attribute daily marked position PnL by latest selection role, leaving
    # fees, fills, corporate actions and receivables in a disclosed residual.
    events=sorted((pd.Timestamp(t['signal_date']),t) for t in trace)
    roles={};event_pos=0; pnl=Counter(); counts=Counter()
    account=result['account']
    for previous,current in zip(account,account[1:]):
        day=pd.Timestamp(current['date'])
        while event_pos<len(events) and events[event_pos][0]<day:
            _,t=events[event_pos]
            roles={replay.canonical_code(c):('retained' if c in t['retained'] else 'replacement') for c in t['selected']}
            roles.update({replay.canonical_code(c):'pending_removed' for c in t['removals']})
            event_pos+=1
        attributed=0.
        for code,q in previous['quantities'].items():
            if code in previous['marks'] and code in current['marks']:
                value=q*(current['marks'][code]-previous['marks'][code])
                role=roles.get(code,'other_actual_holding');pnl[role]+=value;counts[role]+=1;attributed+=value
        pnl['fills_actions_fees_receivables_residual']+=current['nav']-previous['nav']-attributed
    return dict(recovery_episodes=episodes,max_underwater_sessions=max((e['sessions'] for e in episodes),default=0),
        unrecovered_at_end=bool(episodes and not episodes[-1]['recovered']),half_year_returns=halves,
        worst_half_year=min(halves.items(),key=lambda x:x[1]) if halves else None,
        target_actual=shifts,selection_mark_pnl=dict(pnl),position_days=dict(counts),
        attribution_limit='Previous-share close mark change by latest target role, not causal replacement gain. Raw ex-action mark jumps remain in role terms and offset in residual. No missing-dividend correction.',
        target_semantics='previous TARGET basket, not assumed actual fills')


@exclusive
def run(stage='rs'):
    protocol,prepared=verify(stage)
    ds=None
    for period,(start,end) in PERIODS.items():
        for arm in protocol['arms']:
            path=target_path(stage,period,arm)
            payload=json.loads(path.read_text())
            identity=execution_identity(payload['targets'],period,protocol)
            folder=account_folder(stage,period,arm)
            if (folder/'complete.json').exists():
                verify_complete(folder,identity);progress('reused_completed_account',period=period,arm=arm);continue
            partial,recovered=preserve_partial(folder,identity,payload)
            reuse=((partial,recovered) if recovered is not None else
                   old_reusable(period,payload['targets'],protocol) if arm=='R' else None)
            if reuse:
                original,result=reuse
                progress('reusing_verified_old_R',period=period,source=str(original))
            else:
                if ds is None:
                    progress('loading_execution_projection',study_stage=stage)
                    ds=projection(stage,prepared)
                progress('running_account',period=period,arm=arm)
                result=replay.replay_reference(ds,path,start=start,end=end,profile=replay.ReplayProfile.common(50))
            result.update(strategy=arm,model_version=VERSION,parent_data_version=PARENT_VERSION,
                study_protocol_sha256=payload['study_protocol_sha256'],decision_evidence=payload['trace'],
                selection_policy=POLICIES[arm],reused_account=str(reuse[0]) if reuse else None)
            for order in result['orders']:
                order['candidate']=arm;order['selection_policy']=POLICIES[arm]
                order['study_protocol_sha256']=payload['study_protocol_sha256']
            summary,returns=reference.metrics(result,period)
            extra=diagnostics(result,payload['trace'],returns)
            summary.update(max_underwater_sessions=extra['max_underwater_sessions'],
                unrecovered_at_end=extra['unrecovered_at_end'],worst_half_year=extra['worst_half_year'])
            shared.publish(folder/'result.pkl',lambda h:pickle.dump(result,h,pickle.HIGHEST_PROTOCOL))
            write(folder/'summary.json',summary);write(folder/'identity.json',identity)
            write(folder/'diagnostics.json',extra)
            shared.publish(folder/'daily_returns.csv',lambda h:h.write(returns.rename('daily_return').to_csv(index_label='date').encode()))
            names=('result.pkl','summary.json','daily_returns.csv','identity.json','diagnostics.json')
            write(folder/'complete.json',{n:shared.sha256(folder/n) for n in names})
            progress('account_completed',period=period,arm=arm,cagr=summary['cagr'],net_profit=summary['net_profit'])
    verify(stage)
    progress('accounts_complete',study_stage=stage)


@exclusive
def compare(stage='rs'):
    from quantlab.research_statistics import simultaneous_block_ci, family_sharpe
    protocol,_=verify(stage)
    result_path=stage_dir(stage)/'comparison.json'
    identities={}
    for period in PERIODS:
        for arm in protocol['arms']:
            folder=account_folder(stage,period,arm)
            verify_complete(folder)
            identities[str(folder/'complete.json')]=shared.sha256(folder/'complete.json')
    comparison_identity=dict(protocol_sha256=shared.sha256(stage_dir(stage)/'protocol.json'),
                             completed_accounts=identities)
    if result_path.exists():
        existing=json.loads(result_path.read_text())
        if existing.get('comparison_identity')!=comparison_identity:
            raise ValueError('comparison_resume_identity_changed')
        progress('comparison_already_complete',study_stage=stage)
        return existing
    output=dict(stage=stage,periods={},promotion_approved=False,comparison_identity=comparison_identity,
        status='interim_RS_descriptive_no_final_family_inference' if stage=='rs' else 'conditional_completed',
        overlap='Recent2025 account overlaps long account; not independent out of sample',
        unverified=['strict PIT and full actions','stress','data qualification','six-month frozen forward'])
    for period in PERIODS:
        folders={a:account_folder(stage,period,a) for a in protocol['arms']}
        folders.update({f'reference{n}':Path(protocol['baselines'][f'{period}/reference{n}']['path']) for n in (50,100)})
        summaries,returns={},{}
        for arm,folder in folders.items():
            verify_complete(folder)
            summaries[arm]=json.loads((folder/'summary.json').read_text())
            returns[arm]=pd.read_csv(folder/'daily_returns.csv',index_col='date',parse_dates=True).daily_return
        if not all(r.index.equals(returns['R'].index) for r in returns.values()):
            raise ValueError('comparison_calendar_mismatch')
        pairs={f'{a}_minus_{b}':(a,b) for a in protocol['arms'] if a!='R' for b in ('R','reference50','reference100')}
        if stage=='all':pairs['A2_minus_A1']=('A2','A1')
        excess=pd.DataFrame({k:returns[a]-returns[b] for k,(a,b) in pairs.items()})
        intervals={str(block):simultaneous_block_ci(excess,block=block,samples=10000,
            seed=20260927,confidence=.975) for block in (20,60)}
        annual={str(y):{a:float((1+r.loc[str(y)]).prod()-1) for a,r in returns.items()}
                for y in sorted(set(returns['R'].index.year))}
        behavior={}
        for a in protocol['arms']:
            trace=json.loads(target_path(stage,period,a).read_text())['trace']
            weights=[t['router']['weights'] for t in trace if t['router'].get('weights')]
            behavior[a]=dict(decisions=len(trace),retained=sum(t['retained_count'] for t in trace),
                admissions=sum(len(t['additions']) for t in trace),removals=sum(len(t['removals']) for t in trace),
                changed_decisions=sum(t['changed_slots']>0 for t in trace),
                prediction_status=dict(Counter(t['router']['status'] for t in trace)),
                fallbacks=dict(Counter(t['router'].get('fallback_reason') for t in trace if t['router'].get('fallback_reason'))),
                mean_weights=pd.DataFrame(weights).mean().to_dict() if weights else {},
                mean_score_coverage=float(np.mean([t['scored_candidates']/max(t['head_count'],1) for t in trace])))
        output['periods'][period]=dict(arms=summaries,annual_returns=annual,intervals=intervals,behavior=behavior,
            DSR_sensitivity_only=family_sharpe({a:dict(daily_return=r) for a,r in returns.items()},
                registered_trials=208,historical_search_complete=False),
            DSR_warning='208 is a known lower-bound sensitivity, incomplete search history and family dispersion; no DSR adoption claim')
    verdict={}
    for a in protocol['arms']:
        if a=='R':continue
        long=output['periods']['2022_2025']; recent=output['periods']['2025_recent']
        positive=any(p['arms'][a]['net_profit']>p['arms']['R']['net_profit'] for p in (long,recent))
        checks=dict(net_positive=all(p['arms'][a]['net_profit']>0 for p in (long,recent)),
            long_cagr15=long['arms'][a]['cagr']>=.15,
            drawdown25=all(p['arms'][a]['max_drawdown']>=-.25 for p in (long,recent)),
            three_of_four_each_control=all(sum(long['annual_returns'][str(y)][a]>long['annual_returns'][str(y)][b]
                for y in (2022,2023,2024,2025))>=3 for b in ('R','reference50','reference100')),
            both_block_intervals=all(ci['status']=='estimated' and
                all(ci['comparisons'][f'{a}_minus_{b}']['simultaneous_lower']>0 for b in ('R','reference50','reference100'))
                for p in (long,recent) for ci in p['intervals'].values()))
        verdict[a]=dict(checks=checks,requires_registered_stress=positive,
            decision=('interim_wait_for_complete_family' if stage=='rs' else
                      'close_frozen_configuration_no_parameter_rescue' if not positive else
                      'shadow_improvement_evidence_incomplete' if not all(checks.values()) else
                      'numeric_gates_pass_pending_stress_data_and_forward'),promotion_approved=False)
    output['candidates']=verdict
    write(result_path,output)
    progress('comparison_complete',study_stage=stage,candidates=verdict)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','compare'])
    parser.add_argument('--stage',choices=['rs','all'],default='rs')
    args=parser.parse_args()
    globals()[args.action](args.stage)
