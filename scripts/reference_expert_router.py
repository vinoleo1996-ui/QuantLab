"""One frozen algorithm challenger, on the existing reference account ledger."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import continuity_cadence as shared
import reference_challenge as reference  # imports and verifies frozen QuantLab first
import reference_challenge_replay as replay
import numpy as np
import pandas as pd

ROOT = shared.ROOT
MODULE = ROOT / 'quantlab/strategy/expert_router.py'
spec = importlib.util.spec_from_file_location('conditional_expert_router', MODULE)
router = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = router
spec.loader.exec_module(router)
OUT = ROOT / 'output/expert_router_20260927'
SCORES = ROOT / 'output/score_mechanism_20260927/run2/score_rank_membership.parquet'
SCORE_HASH = '48fa232c31f9fcd6d8a4a62f009188cc2d1dff4e5eaa20fc6058872b5b6e9f34'
EXPERTS = ['momentum', 'value', 'lowvol_quality', 'flow', 'composite']
PERIODS = {p: reference.PERIODS[p] for p in ('2022_2025', '2025_recent')}
ARMS = ('original_experts', 'conditional_router')


def build_routed_targets(inputs, sessions, provider):
    """Keep the reference cadence, financial tiers, retention and sector slots.

    Warm-start selection uses only the last reference snapshot before launch,
    including that snapshot's expert vintage. It is submitted on launch day.
    The previous TARGET basket is retained, as in the frozen reference.
    """
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
        selected = router.select_reference_basket(heads[source], basket, groups[source], priorities=priorities)
        ranking = sorted(heads[source], key=lambda c: (-heads[source][c], c))
        retained = set(basket) & set(ranking[:150])
        traces.append(dict(signal_date=day, information_date=source, warm_start=source < day,
            head_count=len(heads[source]), scored_candidates=sum(c in priorities for c in heads[source]),
            retained_count=len(retained), selected=list(selected),
            original_with_same_prior_basket=list(original),
            additions=sorted(set(selected)-set(basket)), removals=sorted(set(basket)-set(selected)),
            changed_slots=len(set(selected)-set(original)), router=evidence))
        basket = selected
        targets.append(dict(signal_date=day, weights={c: .9/len(basket) for c in sorted(basket)},
                            sell_only=False, max_exposure=.9))
    return targets, traces


class ArchivedExperts:
    """As-of adapter for already frozen monthly scores and matured rank labels."""
    def __init__(self, scored, calendar, benchmark, *, policy=router.RouterPolicy()):
        from quantlab.config import CONFIG
        from quantlab.strategy.risk_policy import PolicyController
        from quantlab.data.contracts import decision_time, timestamp
        self.calendar = pd.DatetimeIndex(calendar)
        if self.calendar.has_duplicates or not self.calendar.is_monotonic_increasing:
            raise ValueError('invalid_expert_calendar')
        if scored.duplicated(['date', 'instrument_id']).any():
            raise ValueError('duplicate_archived_expert_prediction')
        benchmark = benchmark.sort_index().copy()
        known = [pd.notna(row.available_at) and timestamp(row.available_at) <= decision_time(day)
                 and row.quality_status == 'ok' for day, row in benchmark.iterrows()]
        self.market = PolicyController('P2', CONFIG, benchmark.close.where(known))
        self.blocks = {pd.Timestamp(day): frame.set_index('instrument_id').sort_index()
                       for day, frame in scored.groupby('date')}
        self.dates = sorted(self.blocks)
        self.columns = ['score_'+e for e in EXPERTS]
        cohorts = []
        for day, frame in sorted(self.blocks.items()):
            position = self.calendar.get_indexer([day])[0]
            if position < 0:
                raise ValueError('expert_date_outside_calendar')
            if position+20 >= len(self.calendar):
                continue
            outcomes = frame.return_20.where(frame.coverage_20.eq('observed'))
            state = self.state(day)
            cohorts.append(router.cohort_from_outcomes(decision_time(day),
                decision_time(self.calendar[position+20]), state, frame[self.columns], outcomes, policy))
        self.model = router.ExpertRouter(cohorts, policy)

    def state(self, day):
        context = self.market.market_snapshot(day)
        return context['desired_state'] if context['ready'] else 'unknown'

    def predict(self, day, arm):
        from quantlab.data.contracts import decision_time
        if arm not in ARMS:
            raise ValueError('unregistered_expert_arm')
        day = pd.Timestamp(day)
        prior = [d for d in self.dates if d <= day]
        if not prior:
            return {}, dict(status='no_expert_snapshot')
        date = prior[-1]
        positions = self.calendar.get_indexer([date, day])
        if min(positions) < 0:
            raise ValueError('expert_prediction_outside_calendar')
        age = int(positions[1]-positions[0])
        meta = dict(score_date=str(date.date()), as_of=str(day.date()), age_sessions=age,
                    state=self.state(day), arm=arm)
        if age > 31:
            return {}, dict(meta, status='stale_expert_snapshot')
        frame = self.blocks[date]
        if arm == 'original_experts':
            scores = frame.score_mixture.where(np.isfinite(frame[self.columns]).all(axis=1)).dropna()
            meta['status'] = 'ok' if len(scores) else 'missing_predictions'
        else:
            decision = self.model.decide(decision_time(day), meta['state'], frame[self.columns])
            scores = decision.scores
            meta.update(status=decision.status, weights=decision.weights, learning=decision.evidence)
        def symbol(code):
            exchange, number = code.split('.') if '.' in code else ('', code)
            return number+'.'+exchange.upper() if exchange in ('sh','sz','bj') else code
        return {symbol(c):float(v) for c,v in scores.items()}, meta


def write(path, value):
    shared.freeze_json(path, value)


def progress(stage, **kwargs):
    value = dict(stage=stage, at=datetime.now(timezone.utc).isoformat(), **kwargs)
    shared.freeze_json(OUT/'status.json', value, immutable=False)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def dated(frame):
    if 'event_time' in frame:
        frame = frame.set_index('event_time')
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    return frame.sort_index()


def prepare():
    """Freeze hypothesis, source, data and policy before generating candidates."""
    shared.verify_file(SCORES, SCORE_HASH)
    sources = [Path(__file__), MODULE, Path(shared.__file__), Path(reference.__file__),
               Path(replay.__file__), ROOT/'tests/test_reference_expert_router.py',
               ROOT/'tests/test_expert_router.py', reference.MANIFEST, SCORES,
               OUT/'PLAN.md', OUT/'external_sources.json', OUT/'test_results.txt']
    sources += [reference.input_path(p) for p in PERIODS]
    baseline = {}
    for period in PERIODS:
        for n in (50, 100):
            folder = reference.OUT/period/f'common{n}_cost1'
            complete = json.loads((folder/'complete.json').read_text())
            for name, digest in complete.items():
                shared.verify_file(folder/name, digest)
            baseline[f'{period}/reference{n}'] = dict(path=str(folder), files=complete)
            sources += [folder/'complete.json']
    versions = {m.__name__: m.__version__ for m in (np, pd, __import__('scipy'))}
    protocol = dict(version='reference-expert-router-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        question='Q5: Can a context-conditioned, jointly fitted expert blend improve NEW admissions in the profitable reference, beyond original expert sorting?',
        unresolved='Actual expert errors differ from the factor-IC proxy; previous pipeline-only work did not test a new algorithm against the profitable reference.',
        model=router.MODEL_VERSION, router_policy=asdict(router.RouterPolicy()),
        source_mechanism='Qlib TRA historical prediction error and context routing; original constrained linear implementation, not the published neural model or its reported returns',
        fixed_selection=dict(holdings=50, exposure=.9, retention_rank=150, industry_slots=15,
            priorities='Only new admission order within unchanged binary financial tiers. Unknown score positions unchanged.',
            cadence='original reference20-session snapshots', warm_start='latest prelaunch reference snapshot with expert information AS OF THAT SNAPSHOT; submit2024-12-31, execute2025-01-02 earliest'),
        archived_prediction='latest monthly snapshot <= reference information day; <=31 exchange sessions old; otherwise original reference order explicitly tagged. No7-9month scores manufactured.',
        training_label='archived20-session close-to-close adjusted PRICE return percentile proxy, not next-open tradeable total return; missing labels never filled. Mature only at endpoint18:00.',
        cohort_coverage='finite five-expert common sample; at least95pct observed labels and30 names; cohort weights date-equal; last12-to36 matured cohorts',
        risk='original reference standalone research permissions without P2; shared actual-share ledger,60w,90pct target,common30bp/source fees/lots/ADV1pct',
        missing_prediction='all fallback/partial coverage reported; fallback is original reference ranking, not an invented model forecast',
        arms=list(ARMS), new_account_runs=4, periods=PERIODS,
        baselines=baseline, frozen_source=shared.source_identity(), dependencies=versions,
        sources={str(p):shared.sha256(p) for p in sources},
        tests='9 targeted behavior tests before freeze; real synthetic orders and fills, maturity, future append, stale snapshot, original parity and seed vintage',
        parent_data_version='6a53f151a1e545857f6925173891ac91a1e06a88bb0de8a7ef34ac6fcd9b7deb',
        selection_gate=dict(net_positive=True, candidate_vs_original_and_reference50_and100_positive_both_periods=True,
            max_drawdown=.25, paired_intervals='20/60-day moving blocks,4000 resamples,seed20260927;4 within-period comparisons max-|t|;97.5pct intervals per period for2-period family',
            failure_action='close this frozen candidate priority; no parameter rescue',
            success_action='shadow only, then registered execution stress and cost2; never automatic adoption'),
        limitations=['all historical data seen; recent overlaps2022-2025, not independent OOS',
            'original labels are price-return proxies and existing data/PIT/industry/action gaps remain',
            'archive starts2020; no credible earlier2018-2021 test for this algorithm',
            'market state is trend/volatility context, not real money-flow measurement',
            'historical search count incomplete; no reset and no claim of DSR-certified alpha'],
        promotion_approved=False)
    write(OUT/'protocol.json', protocol)
    # Save executable source bytes as well as hashes; no copy of mutable data.
    for p in sources:
        if p.suffix == '.py':
            shared.publish(OUT/'source'/p.relative_to(ROOT), lambda h, p=p: h.write(p.read_bytes()))
    m = json.loads(reference.MANIFEST.read_text())
    cal = pd.DatetimeIndex(reference.read_table(reference.MANIFEST.parent, m['calendar']).session)
    bench = dated(reference.read_table(reference.MANIFEST.parent, m['benchmark']))
    scored = pd.read_parquet(SCORES, columns=['date','instrument_id','coverage_20','return_20',
        'score_mixture']+['score_'+e for e in EXPERTS])
    provider = ArchivedExperts(scored, cal, bench)
    cohort_log = [dict(signal_date=str(r.signal_date), available_at=str(r.available_at), state=r.state,
        requested=r.requested, observed=r.observed, status=r.status) for r in provider.model.cohorts]
    write(OUT/'training_cohorts.json', cohort_log)
    files = {}
    for period, (start, end) in PERIODS.items():
        days = cal[(cal >= start)&(cal <= end)]
        inputs = json.loads(reference.input_path(period).read_text())
        baseline_targets = reference.build_targets(inputs, days, 50)
        no_overlay, _ = build_routed_targets(inputs, days, lambda _: ({}, {}))
        if no_overlay != baseline_targets:
            raise ValueError('reference_target_parity_failed:'+period)
        for arm in ARMS:
            targets, trace = build_routed_targets(inputs, days, lambda day: provider.predict(day, arm))
            path = OUT/period/(arm+'_targets.json')
            write(path, dict(targets=targets, trace=trace, model_version=router.MODEL_VERSION,
                study_protocol_sha256=shared.sha256(OUT/'protocol.json')))
            files[str(path)] = shared.sha256(path)
    write(OUT/'prepared.json', dict(targets=files, protocol_sha256=shared.sha256(OUT/'protocol.json'),
        cohort_sha256=shared.sha256(OUT/'training_cohorts.json'), original_target_parity=True))
    progress('prepared', target_files=len(files))


def verify():
    protocol = json.loads((OUT/'protocol.json').read_text())
    prepared = json.loads((OUT/'prepared.json').read_text())
    shared.verify_file(OUT/'protocol.json', prepared['protocol_sha256'])
    for p, digest in {**protocol['sources'], **prepared['targets']}.items():
        shared.verify_file(p, digest)
    for entry in protocol['baselines'].values():
        for name, digest in entry['files'].items():
            shared.verify_file(Path(entry['path'])/name, digest)
    if shared.source_identity() != protocol['frozen_source']:
        raise ValueError('frozen_ledger_changed')
    return protocol, prepared


def projection(prepared):
    from quantlab.data.contracts import MarketDataset
    m = json.loads(reference.MANIFEST.read_text())
    codes = sorted({replay.canonical_code(c) for path in prepared['targets']
        for target in json.loads(Path(path).read_text())['targets'] for c in target['weights']})
    if set(codes)-set(m['panels']):
        raise ValueError('target_execution_panels_missing')
    read = lambda entry: reference.read_table(reference.MANIFEST.parent, entry)
    panels = {c:dated(read(m['panels'][c])) for c in codes}
    records, actions = read(m['instruments']), read(m['actions'])
    manifest = dict(m, panels={c:m['panels'][c] for c in codes}, execution_projection=dict(
        parent_manifest_sha256=shared.sha256(reference.MANIFEST), codes=codes))
    ds = MarketDataset(panels, records[records.instrument_id.isin(codes)].copy(),
        pd.DatetimeIndex(read(m['calendar']).session), actions[actions.instrument_id.isin(codes)].copy(),
        dated(read(m['benchmark'])), manifest=manifest)
    errors = ds.validate('free_research')
    if errors:
        raise ValueError(str(errors))
    write(OUT/'execution_projection.json', dict(codes=codes, panels=len(codes),
        rows=sum(len(x) for x in panels.values()), version=ds.version,
        role='execution projection of all frozen target names, NOT the historical selection pool'))
    return ds


def run():
    protocol, prepared = verify()
    write(OUT/'started.json', dict(at=datetime.now(timezone.utc).isoformat(),
                                  protocol_sha256=prepared['protocol_sha256']))
    progress('loading_execution_projection')
    ds = projection(prepared)
    for period, (start, end) in PERIODS.items():
        for arm in ARMS:
            progress('running_account', period=period, arm=arm)
            path = OUT/period/(arm+'_targets.json')
            result = replay.replay_reference(ds, path, start=start, end=end,
                                             profile=replay.ReplayProfile.common(50))
            result.update(strategy=arm, model_version=router.MODEL_VERSION if arm=='conditional_router'
                          else 'frozen_original_expert_mixture',
                          study_protocol_sha256=prepared['protocol_sha256'],
                          parent_data_version=protocol['parent_data_version'])
            result['decision_evidence'] = json.loads(path.read_text())['trace']
            # Attach identity to each order without changing any execution field.
            for order in result['orders']:
                order['candidate'] = arm
            summary, returns = reference.metrics(result, period)
            folder = OUT/period/arm
            shared.publish(folder/'result.pkl', lambda h: pickle.dump(result,h,pickle.HIGHEST_PROTOCOL))
            write(folder/'summary.json', summary)
            shared.publish(folder/'daily_returns.csv',lambda h:h.write(returns.rename('daily_return').to_csv(index_label='date').encode()))
            write(folder/'complete.json',{n:shared.sha256(folder/n) for n in ('result.pkl','summary.json','daily_returns.csv')})
            progress('account_completed', period=period, arm=arm, cagr=summary['cagr'], net_profit=summary['net_profit'])
    verify()
    progress('accounts_complete', runs=4)


def compare():
    from quantlab.research_statistics import simultaneous_block_ci, family_sharpe
    protocol, prepared = verify()
    result = dict(periods={}, promotion_approved=False, historical_search_incomplete=True,
                  overlap='2025recent overlaps2022-2025; no independent new OOS')
    numeric_pass = True
    for period in PERIODS:
        folders = {arm:OUT/period/arm for arm in ARMS}
        folders.update({f'reference{n}':Path(protocol['baselines'][f'{period}/reference{n}']['path']) for n in (50,100)})
        summaries, returns = {}, {}
        for name, folder in folders.items():
            complete = json.loads((folder/'complete.json').read_text())
            for filename, digest in complete.items():
                shared.verify_file(folder/filename,digest)
            summaries[name] = json.loads((folder/'summary.json').read_text())
            returns[name] = pd.read_csv(folder/'daily_returns.csv', index_col='date', parse_dates=True).daily_return
        if not all(r.index.equals(returns['reference50'].index) for r in returns.values()):
            raise ValueError('comparison_calendar_mismatch')
        pairs = {'original_minus50':('original_experts','reference50'),
                 'router_minus_original':('conditional_router','original_experts'),
                 'router_minus50':('conditional_router','reference50'),
                 'router_minus100':('conditional_router','reference100')}
        excess = pd.DataFrame({k:returns[a]-returns[b] for k,(a,b) in pairs.items()})
        intervals = {str(block):simultaneous_block_ci(excess,block=block,samples=4000,seed=20260927,
                                                    confidence=.975) for block in (20,60)}
        behavior = {}
        for arm in ARMS:
            trace=json.loads((OUT/period/(arm+'_targets.json')).read_text())['trace']
            weights=[t['router']['weights'] for t in trace if t['router'].get('weights')]
            behavior[arm] = dict(decisions=len(trace), changed_decisions=sum(t['changed_slots']>0 for t in trace),
                changed_slots=sum(t['changed_slots'] for t in trace),
                prediction_status=dict(Counter(t['router']['status'] for t in trace)),
                mean_candidate_score_coverage=float(np.mean([t['scored_candidates']/t['head_count'] for t in trace])),
                mean_weights=pd.DataFrame(weights).mean().to_dict() if weights else {})
        candidate = summaries['conditional_router']
        gate = (candidate['net_profit']>0 and candidate['max_drawdown']>=-.25 and
            all(candidate['net_profit']>summaries[c]['net_profit'] for c in ('original_experts','reference50','reference100')) and
            all(ci['status']=='estimated' and all(ci['comparisons'][c]['simultaneous_lower']>0
                for c in ('router_minus_original','router_minus50','router_minus100')) for ci in intervals.values()))
        numeric_pass = numeric_pass and gate
        annual = {str(y):{k:float((1+v.loc[str(y)]).prod()-1) for k,v in returns.items()}
                  for y in sorted(set(returns['reference50'].index.year))}
        dsr = family_sharpe({k:dict(daily_return=v) for k,v in returns.items()}, registered_trials=205,
                           historical_search_complete=False)
        result['periods'][period]=dict(arms=summaries, behavior=behavior, intervals=intervals, annual_returns=annual,
            numeric_gate=bool(gate), DSR_sensitivity_only=dsr,
            DSR_warning='205 is the known203-family plus2 lower-bound sensitivity, not a certified full search count; dispersion only covers these4 paths, not all trials; not an adoption gate')
    result['decision']='retain_shadow_pending_stress' if numeric_pass else 'not_proven_close_frozen_router_priority_no_parameter_rescue'
    result['unverified']=['strict PIT/full actions', 'independent holdout', 'cost doubled', 'actual exit stress', 'production risk permission', 'six-month frozen forward']
    write(OUT/'comparison.json',result)
    progress('completed_conditional', decision=result['decision'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','compare'])
    args=parser.parse_args()
    globals()[args.action]()
