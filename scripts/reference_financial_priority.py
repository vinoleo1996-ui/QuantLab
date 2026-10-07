"""Coverage-matched financial-priority test; frozen strategy code is untouched."""
from __future__ import annotations
import math
from collections import Counter
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT/'output/reference_sustainability_20260927'
OUT = ROOT/'output/reference_financial_priority_20260927'
spec = importlib.util.spec_from_file_location('reference_order_helpers', ROOT/'scripts/reference_sustainability.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def financial_state(profit, pb):
    kp, kb = finite(profit), finite(pb)
    if (kp and profit < 0) or (kb and not 0 < pb <= 3):
        return 'fail'
    return 'pass' if kp and kb else 'unknown'


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def common_inputs(inputs):
    snapshots = {}
    for symbol, day, profit, pb in inputs['financial_snapshots']:
        key = (day, symbol)
        if key in snapshots:
            raise ValueError('duplicate_financial_snapshot')
        snapshots[key] = (profit, pb)
    heads, groups, coverage = {}, {}, []
    for day, cross in sorted(inputs['heads'].items()):
        keep, counts = {}, Counter()
        for symbol, old_score in cross.items():
            if (day, symbol) not in snapshots:
                raise ValueError('missing_archived_snapshot')
            profit, pb = snapshots[day, symbol]
            state = financial_state(profit, pb)
            counts[state] += 1
            if float(state == 'pass') != old_score:
                raise ValueError('archived_financial_score_mismatch')
            if finite(profit) and finite(pb):
                keep[symbol] = float(state == 'pass')
        heads[day] = keep
        groups[day] = {s: inputs['industries'][day][s] for s in keep}
        coverage.append(dict(date=day, candidates=len(cross), known_both=len(keep),
            known_fraction=len(keep)/len(cross) if cross else None,
            passing=counts['pass'], observed_fail=counts['fail'], unknown=counts['unknown'],
            excluded_incomplete=len(cross)-len(keep)))
    return heads, groups, coverage


def make_pair(heads, industries, sessions):
    return {
        'financial': prior.make_targets(heads, industries, sessions, neutral=True),
        'neutral': prior.make_targets({d:dict.fromkeys(v,0.) for d,v in heads.items()},
                                      industries, sessions, neutral=True),
    }


def state(stage, **values):
    row=dict(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **values)
    prior.write(OUT/'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def verify(protocol):
    for p,h in protocol['sources'].items():
        if prior.sha(p)!=h: raise ValueError('study_source_changed:'+p)
    prior.verify(json.loads((PREVIOUS/'protocol.json').read_text()))
    prepared=OUT/'prepared.json'
    if prepared.exists():
        for p,h in json.loads(prepared.read_text()).items():
            if prior.sha(p)!=h: raise ValueError('prepared_input_changed:'+p)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((PREVIOUS/'protocol.json').read_text())
    prior.verify(old)
    paths=[Path(__file__), ROOT/'tests/test_reference_financial_priority.py', OUT/'PLAN.md',
           PREVIOUS/'protocol.json', prior.MANIFEST]
    for period in prior.PERIODS:
        paths += [PREVIOUS/period/'decision_inputs.json',PREVIOUS/period/'daily_returns.csv']
    protocol=dict(version='reference-financial-priority-common-coverage-v1',
        question='Q4-b: binary profit/PB preference versus no numeric preference in the same observable universe',
        seed=prior.SEED, periods=prior.PERIODS, configurations=2, new_account_runs=4,
        common_eligibility='original historic candidates AND finite profit AND finite PB at each decision',
        missing='exclude from both arms before ranking; retain unknown and known-fail counts separately',
        arms={'financial':'binary preference (profit>=0 and 0<PB<=3), fixed hash ties',
              'neutral':'all scores zero, same fixed hash ties'},
        unchanged='500000 cash; 100-name cap; 90% target; original decision dates; 150 retention; 15 industry slots; no new stops; same 5bp cash ledger',
        metric='financial minus neutral, daily net return mean x252; calendar CAGR reported separately',
        intervals='2000 moving block bootstrap samples; 20/60 sessions; per-period pointwise 95% and two-period Bonferroni 95%',
        decision='Both periods positive increment and positive block lower bounds: retain financial-priority evidence; mixed direction: close stable-increment claim; positive but intervals cross zero: unproven. Never auto-promote.',
        sources={str(p):prior.sha(p) for p in paths},
        data_version='5403b0902a7ca7737c9cc11cfc1c2a67e32934a3e5264265713a713b554eb1a2',
        qualifications=['retrospective_not_blind','archived_announced_on_query_not_certified_revision_PIT',
            'observation_coverage_not_economic_pass','no_change_of_seed_or_threshold_after_results'],
        promotion_approved=False)
    p=OUT/'protocol.json'
    if p.exists():
        if json.loads(p.read_text())!=protocol: raise ValueError('immutable_protocol_mismatch')
    else: prior.write(p,protocol)
    for period in prior.PERIODS:
        inputs=json.loads((PREVIOUS/period/'decision_inputs.json').read_text())
        heads,industries,coverage=common_inputs(inputs)
        with (PREVIOUS/period/'daily_returns.csv').open() as stream:
            sessions=[row['date'][:10] for row in csv.DictReader(stream)]
        pair=make_pair(heads,industries,sessions)
        if [(r['signal_date'],len(r['weights'])) for r in pair['financial']] != [(r['signal_date'],len(r['weights'])) for r in pair['neutral']]:
            raise ValueError('target_count_not_matched')
        directory=OUT/period
        composition={}
        for arm,targets in pair.items():
            observations=[]
            for t in targets:
                day=t['signal_date']; w=t['weights']
                if not set(w)<=set(heads[day]): raise ValueError('ineligible_target')
                counts=Counter(industries[day][s] for s in w)
                if len(w)>100 or sum(w.values())>.9+1e-10 or max(counts.values(),default=0)>15:
                    raise ValueError('target_permission_violation')
                observations.append(dict(date=day, holdings=len(w), pass_share=sum(heads[day][s] for s in w)/len(w) if w else None,
                    target_gross=sum(w.values()),exchange=dict(Counter(s.split('.')[-1] for s in w))))
            composition[arm]=observations
            prior.write(directory/arm/'targets.json',dict(targets=targets,
                targets_sha256=hashlib.sha256(json.dumps(targets,sort_keys=True).encode()).hexdigest(),
                source_commit='c01bef11dc708f8d50cc26148f3741f9508741f5',
                data_identity=old['current_catalog'],study_protocol_sha256=prior.sha(p)))
        prior.write(directory/'coverage.json',dict(dates=coverage,composition=composition,
            unknown_removed_in_both=True,prior_full_universe_result_is_not_comparator=True))
    files=[OUT/p/'coverage.json' for p in prior.PERIODS]
    files += [OUT/p/a/'targets.json' for p in prior.PERIODS for a in ('financial','neutral')]
    prepared={str(p):prior.sha(p) for p in files}
    if (OUT/'prepared.json').exists() and json.loads((OUT/'prepared.json').read_text())!=prepared:
        raise ValueError('prepared_artifacts_changed')
    prior.write(OUT/'prepared.json',prepared)
    verify(protocol)
    state('prepared',configurations=2,account_runs=4)


def run_accounts(protocol):
    prior.imports()
    from quantlab.data.contracts import load_dataset
    from quantlab.reference_cash_replay import replay_reference
    state('loading_dataset')
    ds=load_dataset(prior.MANIFEST,profile='free_research')
    for period,(start,end) in prior.PERIODS.items():
        for arm in ('financial','neutral'):
            directory=OUT/period/arm
            if (directory/'complete.json').exists():
                marker=json.loads((directory/'complete.json').read_text())
                for name in ('result.pkl','targets.json','summary.json'):
                    if prior.sha(directory/name)!=marker[name]: raise ValueError('completed_artifact_changed')
                continue
            with (directory/'attempt.json').open('x') as stream:
                json.dump(dict(started_at=datetime.now(timezone.utc).isoformat()),stream)
            verify(protocol)
            state('replaying',period=period,arm=arm)
            result=replay_reference(ds,directory/'targets.json',start=start,end=end)
            if result['data_version']!=protocol['data_version']: raise ValueError('data_version_changed')
            result['study']=dict(protocol_sha256=prior.sha(OUT/'protocol.json'),arm=arm,
                                 role='coverage_matched_financial_priority')
            summary,returns=prior.metrics(result)
            with (directory/'result.pkl').open('xb') as stream:pickle.dump(result,stream,pickle.HIGHEST_PROTOCOL)
            returns.rename('daily_return').to_csv(directory/'daily_returns.csv',index_label='date')
            prior.write(directory/'summary.json',dict(summary,data_version=result['data_version']))
            verify(protocol)
            prior.write(directory/'complete.json',{name:prior.sha(directory/name) for name in ('result.pkl','targets.json','summary.json')})
            state('account_completed',period=period,arm=arm,net_profit=summary['net_profit'],cagr=summary['cagr'])
            del result


def compare(protocol):
    import numpy as np
    import pandas as pd
    periods={}
    for period in prior.PERIODS:
        summary={arm:json.loads((OUT/period/arm/'summary.json').read_text()) for arm in ('financial','neutral')}
        returns={arm:pd.read_csv(OUT/period/arm/'daily_returns.csv',index_col='date',parse_dates=True).daily_return for arm in summary}
        if not returns['financial'].index.equals(returns['neutral'].index): raise ValueError('paired_calendar_mismatch')
        diff=returns['financial']-returns['neutral']; intervals={}
        for block in (20,60):
            rng=np.random.default_rng(20260927+block)
            starts=rng.integers(0,len(diff)-block+1,size=(2000,int(np.ceil(len(diff)/block))))
            ix=(starts[:,:,None]+np.arange(block)).reshape(2000,-1)[:,:len(diff)]
            means=diff.to_numpy()[ix].mean(axis=1)*252
            intervals[str(block)]=dict(pointwise95=np.quantile(means,[.025,.975]).tolist(),
                                       two_period_family95=np.quantile(means,[.0125,.9875]).tolist())
        pd.DataFrame(dict(**returns,difference=diff)).to_csv(OUT/period/'paired_returns.csv',index_label='date')
        periods[period]=dict(**summary,paired_annualized_mean_difference=float(diff.mean()*252),
            cagr_difference=summary['financial']['cagr']-summary['neutral']['cagr'],intervals=intervals,
            yearly_pair_difference={str(y):float(g.mean()*252) for y,g in diff.groupby(diff.index.year)})
    positive=all(p['paired_annualized_mean_difference']>0 and p['financial']['net_profit']>p['neutral']['net_profit'] for p in periods.values())
    supported=positive and all(v['two_period_family95'][0]>0 for p in periods.values() for v in p['intervals'].values())
    report=dict(status='completed_conditional',periods=periods,
        conclusion='retain_financial_increment_evidence' if supported else 'positive_but_unproven' if positive else 'stable_financial_increment_not_supported',
        limitations=['coverage_matched_historical_proxy_not_full_universe','both_periods_retrospective',
            'single_hash_not_random_distribution','2_period_family_not_full_historical_search_correction',
            'same_revision_corporate_action_and_execution_gaps','no_50_name_adaptation_no_new_stress_no_forward'],
        promotion_approved=False,protocol_sha256=prior.sha(OUT/'protocol.json'))
    prior.write(OUT/'comparison.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','compare'])
    args=parser.parse_args()
    if args.action=='prepare':prepare();return
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    if args.action=='run':run_accounts(protocol)
    report=compare(protocol);verify(protocol)
    state('completed_conditional',conclusion=report['conclusion'],promotion_approved=False)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        state('technical_failure',error=repr(exc));raise
