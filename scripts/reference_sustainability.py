"""One registered tie-order perturbation of the archived 100-stock reference.

Frozen policy/account code is imported, never edited. A single deterministic
security priority is tested in both already-seen historical periods.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT/'output/reference1882'
ETF = REF/'frozen_source'
QL = ROOT/'output/frozen_source/6157c64'
OUT = ROOT/'output/reference_sustainability_20260927'
MANIFEST = ROOT/'data/local_market/1c85fd9aceaa03c3c4863bec/manifest.json'
BASE = REF/'cash_replay/6157c64/1c85fd9aceaa03c3'
PERIODS = {'2018_2021':('2018-01-01','2021-12-31'),
           '2022_2025':('2022-01-01','2025-12-31')}
SEED = 'reference-code-neutral-v1:20260927:'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def state(stage, **values):
    record = dict(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **values)
    write(OUT/'status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def priority_scores(scores):
    """Preserve 0/1 classes; hash order alone resolves ties, with no outcome input."""
    if any(value not in (0., 1.) for value in scores.values()):
        raise ValueError('expected_binary_financial_scores')
    ranked = sorted(scores, key=lambda s:(hashlib.sha256((SEED+s).encode()).hexdigest(), s))
    n = len(ranked)
    return {s:float(scores[s])+.5*(n-i)/(n+1) for i,s in enumerate(ranked)}


def imports():
    for path in (QL, ETF):
        if str(path) not in sys.path: sys.path.insert(0, str(path))


def make_targets(heads, industries, sessions, *, neutral=False):
    imports()
    from a_share_quant.stock_only.broad_basket import BroadBasketPolicy
    from a_share_quant.daily_replay import ReplayDecisionState
    convert = lambda d: date.fromisoformat(str(d)[:10])
    scores = {convert(d): priority_scores(v) if neutral else v for d,v in heads.items()}
    groups = {convert(d):v for d,v in industries.items() if convert(d) in scores}
    days = [convert(d) for d in sessions]
    policy = BroadBasketPolicy(scores, groups, {days[0]:{}}, holdings=100, buffer=50,
                               drawdown_scaler=None)
    result = []
    for day in days:
        target = policy(ReplayDecisionState(day, 500000., 500000., ()))
        if target is not None:
            result.append(dict(signal_date=str(day), weights=dict(target.weights),
                               sell_only=target.sell_only, max_exposure=target.max_exposure))
    return result


def require_targets_equal(actual, archive):
    fields = ('signal_date','weights','sell_only','max_exposure')
    expected = [{k:r[k] for k in fields} for r in archive['targets']]
    if actual != expected: raise ValueError('original_target_reconstruction_mismatch')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), ROOT/'tests/test_reference_sustainability.py', OUT/'PLAN.md', MANIFEST,
             REF/'rebuild_protocol.json', ETF/'source_manifest.json']
    files += [REF/f'targets_{period}_cost1.json' for period in PERIODS]
    for period in PERIODS:
        summary = json.loads((BASE/(period+'_cost1')/'run_summary.json').read_text())
        files += [BASE/(period+'_cost1')/'run_summary.json',
                  Path(summary['artifact_directory'])/'result.pkl',
                  Path(summary['artifact_directory'])/'artifacts.json',REF/f'rebuild_{period}.json']
    files += sorted(QL.rglob('*.py'))
    identity = json.loads((REF/'rebuild_protocol.json').read_text())['current_catalog']
    stat = Path(identity['path']).stat()
    if (stat.st_size,stat.st_mtime_ns) != (identity['size'],identity['mtime_ns']):
        raise ValueError('reference_database_changed')
    if sha(identity['path']) != identity['sha256']: raise ValueError('reference_database_checksum')
    sources = json.loads((ETF/'source_manifest.json').read_text())['full_snapshot_sources']
    for p,h in sources.items():
        if sha(ETF/p) != h: raise ValueError('frozen_reference_source_changed:'+p)
    protocol = dict(version='reference-code-order-one-perturbation-v1', seed=SEED,
        periods=PERIODS, configuration_count=1, new_account_runs=2,
        only_policy_change='binary-score ties ordered by fixed security hash instead of code',
        sources={str(p):sha(p) for p in files}, reference_sources=sources,
        current_catalog=identity, historical_status='retrospective_not_blind',
        interpretation='One perturbation cannot prove priority independence or sustainability.',
        promotion_approved=False)
    target = OUT/'protocol.json'
    if target.exists():
        if json.loads(target.read_text()) != protocol: raise ValueError('protocol_already_frozen_differently')
    else: write(target, protocol)
    return protocol


def verify(protocol):
    for p,h in protocol['sources'].items():
        if sha(p) != h: raise ValueError('study_source_changed:'+p)
    for p,h in protocol['reference_sources'].items():
        if sha(ETF/p) != h: raise ValueError('frozen_reference_source_changed:'+p)
    identity = protocol['current_catalog']; stat=Path(identity['path']).stat()
    if (stat.st_size,stat.st_mtime_ns) != (identity['size'],identity['mtime_ns']):
        raise ValueError('reference_database_changed')


def prepare(period, protocol):
    imports()
    spec = importlib.util.spec_from_file_location('reference_original_runner',ETF/'scripts/run_growth_core.py')
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    from a_share_quant.stock_only.micro_filters import load_profit_pb_snapshots, quality_pass
    start,end = map(date.fromisoformat, PERIODS[period])
    catalog = Path(protocol['current_catalog']['path'])
    state('reconstructing_original_targets', period=period)
    rows = original._load_cross_sections(catalog, start-timedelta(days=400), end,20,2000,
        include_absolute=True, universe_offset=3000,minimum_adv=5000000.,exclude_st=False)
    scores,industries = original.build_signals(rows)
    base = scores['defensive_composite']
    # This branch of growth_core uses defensive_base=copy(base); none of the
    # growth/cross-market heads enters quality_filtered_equal. Verify archive.
    snapshots = load_profit_pb_snapshots(catalog, base)
    passing = quality_pass(snapshots, minimum_profit=0., maximum_pb=3.)
    full_heads = {d:{s:float(passing.get((s,d),False)) for s in cross} for d,cross in base.items()}
    signal_hash=hashlib.sha256(json.dumps({'quality_filtered_equal':{str(d):v for d,v in full_heads.items()}},sort_keys=True).encode()).hexdigest()
    rebuilt = json.loads((REF/f'rebuild_{period}.json').read_text())
    if signal_hash != rebuilt['signal_sha256']: raise ValueError('original_signal_hash_mismatch')
    heads = {d:v for d,v in full_heads.items() if d>=start}
    sessions = [d for d in original.trading_sessions(catalog) if start<=d<=end]
    expected=json.loads((REF/f'targets_{period}_cost1.json').read_text())
    baseline=make_targets(heads,industries,sessions)
    require_targets_equal(baseline,expected)
    variant=make_targets(heads,industries,sessions,neutral=True)
    require_targets_equal(variant,{'targets':make_targets(heads,industries,sessions,neutral=True)})
    # Future labelled columns are excluded from all selection paths.
    modified=[tuple(r[:9])+tuple(None for _ in r[9:]) for r in rows]
    if original.build_signals(modified)!= (scores,industries): raise ValueError('future_labels_affect_selection')
    directory=OUT/period;directory.mkdir(exist_ok=True)
    write(directory/'decision_inputs.json',dict(
        heads={str(d):v for d,v in heads.items()}, industries={str(d):industries[d] for d in heads},
        financial_snapshots=snapshots, scope='frozen historical proxy inputs, not newly qualified PIT'))
    def composition(records):
        symbols=[c for r in records for c in r['weights']]
        return dict(target_observations=len(symbols),unique_symbols=len(set(symbols)),
            exchange=dict(Counter(s.split('.')[-1] for s in symbols)),
            prefix=dict(Counter(s[:3] for s in symbols)))
    write(directory/'target_parity.json',dict(passed=True,original_signal_sha256=signal_hash,
        dates=len(baseline),future_labels_irrelevant=True,
        original=composition(baseline),neutral=composition(variant),
        average_target_overlap=sum(len(set(a['weights'])&set(b['weights']))/max(1,len(a['weights']))
            for a,b in zip(baseline,variant))/len(baseline)))
    payload=dict(targets=variant,targets_sha256=hashlib.sha256(json.dumps(variant,sort_keys=True).encode()).hexdigest(),
        source_commit='c01bef11dc708f8d50cc26148f3741f9508741f5',data_identity=protocol['current_catalog'],
        study_protocol_sha256=sha(OUT/'protocol.json'),role='reference_code_order_sensitivity')
    write(directory/'targets.json',payload)
    verify(protocol)
    return directory/'targets.json'


def metrics(run):
    import numpy as np
    import pandas as pd
    from quantlab.policy_comparison import account_checks
    from quantlab.report.research_diagnostics import account_diagnostics
    checks=account_checks(run)
    if not checks['passed']: raise ValueError('account_checks_failed:'+str(checks))
    account=pd.DataFrame(run['account']).set_index('date')
    account.index=pd.DatetimeIndex(account.index)
    nav=account.nav.astype(float);ret=nav.pct_change(fill_method=None);ret.iloc[0]=nav.iloc[0]/500000.-1
    if not np.allclose(ret,run['daily_return'],atol=1e-12,rtol=0): raise ValueError('daily_return_mismatch')
    if not np.allclose(nav/500000.,run['nav'],atol=1e-12,rtol=0): raise ValueError('nav_mismatch')
    if not np.allclose(account.cash+pd.Series([sum(s['weights'].values())*s['nav'] for s in run['account']],index=nav.index)
                       +account.get('receivables',0)-account.get('payables',0),nav,atol=.05,rtol=0):
        raise ValueError('cash_exposure_identity')
    pnl=nav.diff();pnl.iloc[0]=nav.iloc[0]-500000.
    by_year={str(y):float((1+g).prod()-1) for y,g in ret.groupby(ret.index.year)}
    avg_weights={}
    for w in account.weights:
        for c,v in w.items():avg_weights[c]=avg_weights.get(c,0)+v/len(account)
    out=dict(status=run['status'],net_profit=float(nav.iloc[-1]-500000.),
        cagr=float((nav.iloc[-1]/500000.)**(365.25/(nav.index[-1]-nav.index[0]).days)-1),
        max_drawdown=float((nav/nav.cummax().clip(lower=500000.)-1).min()),
        mean_gross=float(np.mean([sum(w.values()) for w in account.weights])),
        sessions=len(ret),yearly_returns=by_year,fees=float(sum(f['fees'] for f in run['fills'])),
        slippage=float(sum(f.get('slippage_cost',0) for f in run['fills'])),
        fills=len(run['fills']),first_invested=next((str(s['date']) for s in run['account'] if s['quantities']),None),
        best5_days_pnl=float(pnl.nlargest(5).sum()),
        pnl_excluding_best5_days_descriptive=float(pnl.sum()-pnl.nlargest(5).sum()),
        top10_average_security_weights=dict(sorted(avg_weights.items(),key=lambda p:-p[1])[:10]),
        rejection_counts=run['rejection_counts'],data_issues=len(run.get('data_issues',[])),
        missing_target_securities=run['missing_target_securities'],account_checks=checks,
        diagnostics=account_diagnostics(run),promotion_approved=False)
    return out,ret


def replay(period,protocol,ds):
    import pandas as pd
    from quantlab.reference_cash_replay import replay_reference
    from quantlab.data.contracts import fingerprint
    directory=OUT/period
    marker=directory/'attempt.json'
    with marker.open('x') as stream: json.dump({'started_at':datetime.now(timezone.utc).isoformat()},stream)
    state('replaying_cash_account',period=period)
    result=replay_reference(ds,directory/'targets.json',start=PERIODS[period][0],end=PERIODS[period][1])
    result['study_protocol_sha256']=sha(OUT/'protocol.json')
    result['protocol']=dict(result['protocol'],role='reference_code_order_sensitivity',priority_seed=SEED)
    result.update(strategy='reference_code_order_sensitivity',model_version='reference-code-order-v1')
    result['run_id']=fingerprint(dict(protocol=result['protocol'],data_version=result['data_version']))[:24]
    for order in result['orders']:
        order['candidate']='reference_code_order_sensitivity'
    for trace in result['signal_trace']:trace['reason']='frozen_reference_binary_score_hash_tiebreak'
    summary,returns=metrics(result)
    with (directory/'result.pkl').open('xb') as stream:pickle.dump(result,stream,pickle.HIGHEST_PROTOCOL)
    returns.rename('daily_return').to_csv(directory/'daily_returns.csv',index_label='date')
    pd.DataFrame(result['fills']).to_csv(directory/'fills.csv',index=False)
    write(directory/'summary.json',dict(summary,data_version=result['data_version'],run_id=result['run_id']))
    verify(protocol)
    write(directory/'complete.json',dict(status='completed_conditional',result_sha256=sha(directory/'result.pkl'),
        targets_sha256=sha(directory/'targets.json'),summary_sha256=sha(directory/'summary.json')))
    state('period_complete',period=period,cagr=summary['cagr'],max_drawdown=summary['max_drawdown'])


def compare(protocol):
    import numpy as np
    import pandas as pd
    reports={}
    for period in PERIODS:
        baseline_info=json.loads((BASE/(period+'_cost1')/'run_summary.json').read_text())
        p=Path(baseline_info['artifact_directory'])/'result.pkl'
        with p.open('rb') as stream:base=pickle.load(stream)
        baseline,original_returns=metrics(base)
        variant=json.loads((OUT/period/'summary.json').read_text())
        alt=pd.read_csv(OUT/period/'daily_returns.csv',index_col='date',parse_dates=True).daily_return
        if not alt.index.equals(original_returns.index): raise ValueError('paired_calendar_mismatch')
        diff=alt-original_returns
        blocks={}
        for block in (20,60):
            rng=np.random.default_rng(20260927+block)
            starts=rng.integers(0,len(diff)-block+1,size=(2000,int(np.ceil(len(diff)/block))))
            indices=(starts[:,:,None]+np.arange(block)).reshape(2000,-1)[:,:len(diff)]
            samples=diff.to_numpy()[indices].mean(axis=1)*252
            blocks[str(block)]=list(np.quantile(samples,[.025,.975]).astype(float))
        pd.DataFrame({'original':original_returns,'neutral':alt,'difference':diff}).to_csv(OUT/period/'paired_returns.csv',index_label='date')
        reports[period]=dict(original=baseline,neutral=variant,
            paired_annualized_mean_difference=float(diff.mean()*252),pointwise_block_intervals=blocks,
            cagr_difference=variant['cagr']-baseline['cagr'])
    survives=all(v['neutral']['cagr']>0 and v['neutral']['max_drawdown']>=-.25 for v in reports.values())
    strong=survives and all(v['neutral']['cagr']>=.15 for v in reports.values())
    report=dict(status='completed_conditional_code_order_test',periods=reports,
        both_periods_positive_within_25pct_drawdown=survives,both_periods_reach_15pct=strong,
        conclusion='code_order_perturbation_survives_but_sustainability_unproven' if survives else 'code_order_sensitivity_requires_explanation',
        limitations=['single_fixed_priority_not_distribution','retrospective_both_periods_not_blind',
            'code_order_change_also_changes_exchange_sector_and_turnover_exposure',
            'same_pit_revision_and_corporate_action_gaps','no_full_historical_search_DSR',
            '5bp_slippage_not_common_30bp','100_names_not_50','new_stress_and_forward_not_done'],
        promotion_approved=False,protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'comparison.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','compare'))
    args=parser.parse_args()
    if args.action=='prepare':
        protocol=freeze()
        for period in PERIODS:prepare(period,protocol)
        state('targets_prepared',periods=list(PERIODS))
    else:
        protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol);imports()
        if args.action=='run':
            from quantlab.data.contracts import load_dataset
            state('loading_common_dataset')
            ds=load_dataset(MANIFEST,profile='free_research')
            for period in PERIODS:replay(period,protocol,ds)
        report=compare(protocol)
        verify(protocol)
        state('completed_conditional',both_periods_reach_15pct=report['both_periods_reach_15pct'],
              promotion_approved=False)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        state('failed',error=repr(exc));raise
