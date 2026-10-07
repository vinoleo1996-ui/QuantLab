"""Fresh 2025 account for the archived reference; no main-strategy changes."""
from __future__ import annotations
import argparse
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/reference_recent_20260927'
spec=importlib.util.spec_from_file_location('reference_order_helpers',ROOT/'scripts/reference_sustainability.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)


def recent_schedule(heads, industries, sessions):
    # Keep one prior score for a fresh account. Earlier target retention history
    # is deliberately not inherited from the old 2022-start account.
    first=str(sessions[0])[:10]
    earlier=[str(d)[:10] for d in heads if str(d)[:10]<first]
    if not earlier: raise ValueError('no_prior_reference_signal')
    start=max(earlier)
    selected={str(d)[:10]:v for d,v in heads.items() if str(d)[:10]>=start}
    groups={d:industries[d] for d in selected}
    return prior.make_targets(selected,groups,sessions)


def verify(protocol):
    for path,expected in protocol['sources'].items():
        if prior.sha(path)!=expected:raise ValueError('reference_recent_source_changed:'+path)
    prior.verify(json.loads((prior.OUT/'protocol.json').read_text()))


def prepare(end):
    OUT.mkdir(parents=True,exist_ok=True)
    prior.imports()
    spec=importlib.util.spec_from_file_location('reference_original_runner',prior.ETF/'scripts/run_growth_core.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    from a_share_quant.stock_only.micro_filters import load_profit_pb_snapshots,quality_pass
    old=json.loads((prior.OUT/'protocol.json').read_text())
    sources=[Path(__file__),ROOT/'tests/test_reference_recent_account.py',OUT/'PLAN.md',prior.OUT/'protocol.json',
             prior.OUT/'2022_2025/decision_inputs.json']
    protocol=dict(version='reference-fresh-2025-v1',prepare_date='2024-12-31',performance_start='2025-01-02',end=end,
        initial_cash=500000.,holdings=100,target_gross=.9,order_policy='original_binary_score_code_order',
        warm_start='latest already published score snapshot; no old account or old target retention inherited',
        cadence_anchor='2020-11-27, same query anchor as original 2022-2025 study; every20 sessions',
        score_fields='profit>=0 and 0<PB<=3 gives binary soft preference; original unknown handling retained',
        comparison='separate reference, not identical capital/risk/holdings/cost to QuantLab main',
        sources={str(p):prior.sha(p) for p in sources},current_catalog=old['current_catalog'],
        promotion_approved=False)
    if (OUT/'protocol.json').exists():
        if json.loads((OUT/'protocol.json').read_text())!=protocol:raise ValueError('immutable_protocol_mismatch')
    else:prior.write(OUT/'protocol.json',protocol)
    verify(protocol)
    rows=original._load_cross_sections(Path(old['current_catalog']['path']),date(2020,11,27),date.fromisoformat(end),20,2000,
        include_absolute=True,universe_offset=3000,minimum_adv=5000000.,exclude_st=False)
    scores,groups=original.build_signals(rows)
    snapshots=load_profit_pb_snapshots(Path(old['current_catalog']['path']),scores['defensive_composite'])
    passing=quality_pass(snapshots,minimum_profit=0.,maximum_pb=3.)
    heads={str(d):{s:float(passing.get((s,d),False)) for s in cross} for d,cross in scores['defensive_composite'].items()}
    industries={str(d):cross for d,cross in groups.items()}
    previous=json.loads((prior.OUT/'2022_2025/decision_inputs.json').read_text())
    for day,values in previous['heads'].items():
        if heads.get(day)!=values or industries.get(day)!=previous['industries'][day]:
            raise ValueError('historical_head_changed:'+day)
    sessions=[str(d) for d in original.trading_sessions(Path(old['current_catalog']['path'])) if '2024-12-31'<=str(d)<=end]
    if not sessions or sessions[-1]!=end:raise ValueError('reference_calendar_does_not_cover_end')
    # Calendar cadence must still produce all decisions, even after the old
    # account's end. Missing query rows cannot silently suppress a review.
    calendar=[str(d) for d in original.trading_sessions(Path(old['current_catalog']['path'])) if '2020-11-27'<=str(d)<=end]
    expected={d for i,d in enumerate(calendar) if i%20==0 and d>='2025-01-02'}
    if not expected<=heads.keys():raise ValueError('missing_expected_reference_review')
    targets=recent_schedule(heads,industries,sessions)
    payload=dict(targets=targets,targets_sha256=hashlib.sha256(json.dumps(targets,sort_keys=True).encode()).hexdigest(),
        source_commit='c01bef11dc708f8d50cc26148f3741f9508741f5',data_identity=old['current_catalog'],
        study_protocol_sha256=prior.sha(OUT/'protocol.json'))
    prior.write(OUT/'targets.json',payload)
    used={max(d for d in heads if d<'2024-12-31')}|{d for d in heads if d>='2024-12-31'}
    prior.write(OUT/'inputs.json',dict(heads={d:heads[d] for d in sorted(used)},industries={d:industries[d] for d in sorted(used)},
        prior_2022_2025_heads_exact=True,expected_reviews=sorted(expected),sessions=sessions))
    prior.write(OUT/'prepared.json',{str(OUT/name):prior.sha(OUT/name) for name in ['targets.json','inputs.json']})
    verify(protocol)
    print(json.dumps(dict(stage='prepared',end=end,targets=len(targets),prior_head_parity=True)),flush=True)


def run(manifest):
    prior.imports()
    from quantlab.data.contracts import load_dataset
    from quantlab.reference_cash_replay import replay_reference
    import pandas as pd
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    for p,h in json.loads((OUT/'prepared.json').read_text()).items():
        if prior.sha(p)!=h:raise ValueError('reference_recent_prepared_changed')
    identity=dict(study_protocol_sha256=prior.sha(OUT/'protocol.json'),manifest=str(manifest.resolve()),
        manifest_sha256=prior.sha(manifest),started_at=datetime.now(timezone.utc).isoformat())
    with (OUT/'run_protocol.json').open('x') as stream:json.dump(identity,stream,indent=2)
    print('loading shared market dataset',flush=True)
    ds=load_dataset(manifest,profile='free_research')
    if ds.calendar.max()<pd.Timestamp(protocol['end']):raise ValueError('execution_data_ends_early')
    print('replaying fresh reference account',flush=True)
    result=replay_reference(ds,OUT/'targets.json',start=protocol['prepare_date'],end=protocol['end'])
    summary,returns=prior.metrics(result)
    before=[s for s in result['account'] if pd.Timestamp(s['date'])<pd.Timestamp(protocol['performance_start'])]
    if len(before)!=1 or before[0]['quantities'] or before[0]['nav']!=500000.:
        raise ValueError('reference_preparation_day_not_cash')
    if any(pd.Timestamp(f['date'])<pd.Timestamp(protocol['performance_start']) for f in result['fills']):
        raise ValueError('reference_traded_before_start')
    displayed=returns.loc[protocol['performance_start']:]
    summary.update(performance_start=str(displayed.index[0].date()),as_of=protocol['end'],initial_cash=500000.,
        account_preparation_date=protocol['prepare_date'],final_equity=500000.+summary['net_profit'],
        total_return=summary['net_profit']/500000.,cagr=((500000.+summary['net_profit'])/500000.)**(365.25/(displayed.index[-1]-displayed.index[0]).days)-1,
        mean_gross=sum(sum(s['weights'].values()) for s in result['account'][1:])/len(displayed),sessions=len(displayed),
        yearly_returns={str(y):float((1+g).prod()-1) for y,g in displayed.groupby(displayed.index.year)},
        data_version=result['data_version'],model_version='original-reference-fresh-account',executable=False)
    with (OUT/'result.pkl').open('xb') as stream:pickle.dump(result,stream,pickle.HIGHEST_PROTOCOL)
    displayed.rename('daily_return').to_csv(OUT/'daily_returns.csv',index_label='date')
    prior.write(OUT/'summary.json',summary)
    verify(protocol)
    if prior.sha(manifest)!=identity['manifest_sha256']:raise ValueError('execution_manifest_changed')
    prior.write(OUT/'complete.json',{name:prior.sha(OUT/name) for name in ['result.pkl','summary.json','daily_returns.csv','run_protocol.json','targets.json']})
    print(json.dumps(dict(stage='completed_conditional',net_profit=summary['net_profit'],cagr=summary['cagr'],end=protocol['end'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--end',default='2026-09-22');parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.end)
    else:
        if args.manifest is None:parser.error('--manifest is required for run')
        run(args.manifest)


if __name__=='__main__':main()
