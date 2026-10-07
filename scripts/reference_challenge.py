"""Frozen, finite reference challenge. No production-policy or catalog writes."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import pickle
import sys

import continuity_cadence as shared
import reference_challenge_replay as replay
import numpy as np
import pandas as pd
from quantlab.data.contracts import MarketDataset, fingerprint

ROOT = shared.ROOT
OUT = ROOT / 'output/reference_challenge_20260927/run2'
MANIFEST = ROOT / 'output/account_2025_20260927/extension/dataset/manifest.json'
ETF = ROOT / 'output/reference1882/frozen_source'
PERIODS = {'2018_2021': ('2018-01-01', '2021-12-31'),
           '2022_2025': ('2022-01-01', '2025-12-31'),
           '2025_recent': ('2024-12-31', '2026-09-22')}


def write(path, value):
    shared.freeze_json(path, value)


def progress(stage, **kwargs):
    record = dict(stage=stage, at=datetime.now(timezone.utc).isoformat(), **kwargs)
    shared.freeze_json(OUT/'status.json', record, immutable=False)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def input_path(period):
    if period == '2025_recent':
        return ROOT/'output/reference_recent_20260927/inputs.json'
    return ROOT/'output/reference_sustainability_20260927'/period/'decision_inputs.json'


def archive_path(period):
    return (ROOT/'output/reference_recent_20260927/targets.json' if period == '2025_recent'
            else ROOT/f'output/reference1882/targets_{period}_cost1.json')


def build_targets(inputs, sessions, holdings):
    """Only the count changes; absolute retention rank 150 and 15 sector slots stay."""
    if holdings not in (50, 100):
        raise ValueError('unregistered_holdings')
    if str(ETF) not in sys.path:
        sys.path.insert(0, str(ETF))
    from a_share_quant.stock_only.broad_basket import BroadBasketPolicy
    from a_share_quant.daily_replay import ReplayDecisionState
    cv = lambda x: date.fromisoformat(str(x)[:10])
    heads = {cv(d): v for d, v in inputs['heads'].items()}
    groups = {cv(d): v for d, v in inputs['industries'].items()}
    days = [cv(d) for d in sessions]
    policy = BroadBasketPolicy(heads, groups, {days[0]: {}}, holdings=holdings,
                               buffer=150-holdings, industry_cap_share=15/holdings,
                               drawdown_scaler=None)
    if policy.industry_slots != 15:
        raise ValueError('industry_slots_changed')
    targets = []
    for day in days:
        target = policy(ReplayDecisionState(day, 600000., 600000., ()))
        if target is not None:
            targets.append(dict(signal_date=str(day), weights=dict(target.weights),
                                sell_only=target.sell_only, max_exposure=target.max_exposure))
    return targets


def read_table(root, entry):
    path = (root/entry['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('source_path_escape')
    shared.verify_file(path, entry['sha256'])
    return pd.read_parquet(path)


def prepare():
    """Freeze targets and question before executing any new real account."""
    m = json.loads(MANIFEST.read_text())
    calendar = pd.DatetimeIndex(read_table(MANIFEST.parent, m['calendar']).session)
    sources = [Path(__file__), Path(replay.__file__), ROOT/'tests/test_reference_challenge.py',
               Path(shared.__file__), MANIFEST, OUT/'PLAN.md',
               ROOT/'output/reference_recent_20260927/complete.json',
               ROOT/'output/reference_recent_20260927/result.pkl',
               ROOT/'output/account_2025_20260927/FINAL_DELIVERY.json']
    sources += list((ETF/'a_share_quant').rglob('*.py'))
    sources += [input_path(p) for p in PERIODS] + [archive_path(p) for p in PERIODS]
    targets = {}
    for period, (start, end) in PERIODS.items():
        days = calendar[(calendar >= start) & (calendar <= end)]
        inputs = json.loads(input_path(period).read_text())
        for n in (100, 50):
            values = build_targets(inputs, days, n)
            if n == 100:
                archive = json.loads(archive_path(period).read_text())['targets']
                fields = ('signal_date', 'weights', 'sell_only', 'max_exposure')
                if values != [{k: t[k] for k in fields} for t in archive]:
                    raise ValueError('original_targets_not_reproduced:'+period)
            path = OUT/period/f'targets_{n}.json'
            write(path, dict(targets=values, source_commit='c01bef11dc708f8d50cc26148f3741f9508741f5',
                             role='frozen_reference_count_dependency'))
            targets[str(path)] = shared.sha256(path)
    protocol = dict(version='reference-challenge-v1', frozen_source=shared.source_identity(),
        created_at=datetime.now(timezone.utc).isoformat(), question='Does the reference advantage survive common capital/costs, and does limiting its count to 50 preserve or improve it?',
        open_questions=['Q4_reference_cost_and_breadth_dependency'],
        decision='Reference is mandatory profit comparator. Retain count50 as a shadow candidate only if net increment and risk hold across periods; no post-result parameter repair.',
        periods=PERIODS, initial_cash=600000, profiles={str(n):replay.asdict(replay.ReplayProfile.common(n)) for n in (100,50)},
        new_account_runs=9, verification_replay=1,
        arms=['common100_cost1','common50_cost1','common50_cost2'],
        fixed_selection=dict(retention_rank=150, industry_slots=15, target_exposure=.9,
                             score='original_binary_profit_pb_then_code', cadence='original_20_session_dates'),
        inherited_risk='standalone original reference, without P2; not authorized production risk',
        matched_to_main=['initial capital','one-way 30bp slippage','source commission/lot/minimum/settlement rules','1000 CNY minimum normal adjustment','prior20 ADV1pct'],
        remaining_differences=['reference universe/financial snapshot/selection/cadence/100 or50 holdings/risk',
            'reference renews fixed unfilled normal orders until next target; main expiry policy differs',
            'both matched cost profiles omit separate transfer fee; original reference charged it',
            'count50 retains absolute15 industry slots: max target industry exposure13.5pct becomes27pct',
            'retention cutoff150 unchanged; target retention is prior intended basket, not actual filled holdings'],
        selection_gate=dict(net_positive_all_periods=True, increment_positive_all_three_periods=True,
            max_drawdown=.25, cost2_positive_all_periods=True,
            paired_bootstrap='20 and60 day circular blocks; 4000 resamples; fixed seed20260927; family6 Bonferroni',
            data_execution_and_forward_validation='separate unresolved gates; promotion always false'),
        overlap='recent2025 overlaps2022_2025 and is not an independent third OOS period; all history has been seen',
        parent_data_version='6a53f151a1e545857f6925173891ac91a1e06a88bb0de8a7ef34ac6fcd9b7deb',
        execution_projection='Load the union of all predeclared target securities, preserving every stored row. Used only for fixed-target replay, never as the historical selection pool.',
        sources={str(p):shared.sha256(p) for p in sources},targets=targets,promotion_approved=False)
    write(OUT/'protocol.json', protocol)
    progress('prepared', target_files=len(targets), new_account_runs=9)


def verify(protocol):
    for p,h in dict(protocol['sources'], **protocol['targets']).items():
        shared.verify_file(p,h)
    current = shared.source_identity()
    if current != protocol['frozen_source']:
        raise ValueError('frozen_strategy_changed')


def projection(protocol):
    m = json.loads(MANIFEST.read_text())
    codes = sorted({replay.canonical_code(c) for p in protocol['targets']
                    for t in json.loads(Path(p).read_text())['targets'] for c in t['weights']})
    absent = set(codes)-set(m['panels'])
    if absent:
        raise ValueError('registered_target_panels_missing:'+str(absent))
    def dated(frame):
        if 'event_time' in frame:
            frame = frame.set_index('event_time')
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        return frame.sort_index()
    panels = {c:dated(read_table(MANIFEST.parent,m['panels'][c])) for c in codes}
    records=read_table(MANIFEST.parent,m['instruments'])
    actions=read_table(MANIFEST.parent,m['actions'])
    manifest=dict(m,panels={c:m['panels'][c] for c in codes},
                  execution_projection=dict(parent_manifest_sha256=shared.sha256(MANIFEST),codes=codes))
    ds=MarketDataset(panels,records[records.instrument_id.isin(codes)].copy(),
        pd.DatetimeIndex(read_table(MANIFEST.parent,m['calendar']).session),
        actions[actions.instrument_id.isin(codes)].copy(),dated(read_table(MANIFEST.parent,m['benchmark'])),manifest=manifest)
    errors=ds.validate('free_research')
    if errors:raise ValueError(str(errors))
    write(OUT/'execution_projection.json',dict(codes=codes,panels=len(codes),rows=sum(len(p) for p in panels.values()),
        parent_manifest_sha256=shared.sha256(MANIFEST),projection_version=ds.version,
        checksums={c:m['panels'][c]['sha256'] for c in codes},qualification='unchanged conditional data; not a new universe'))
    return ds


def metrics(result, period):
    from quantlab.policy_comparison import account_checks
    check=account_checks(result)
    if not check['passed']:raise ValueError(str(check))
    initial=result['config']['capital']['init_cash']
    account=result['account']
    nav=pd.Series({pd.Timestamp(s['date']):s['nav'] for s in account})
    ret=nav.pct_change(fill_method=None);ret.iloc[0]=nav.iloc[0]/initial-1
    if period=='2025_recent':
        if nav.index[0]!=pd.Timestamp('2024-12-31') or account[0]['quantities'] or nav.iloc[0]!=initial:
            raise ValueError('preparation_day_not_cash')
        ret=ret.loc['2025-01-02':]
    if not np.allclose(result['daily_return'].reindex(ret.index),ret,rtol=0,atol=1e-12):raise ValueError('return_mismatch')
    states=[s for s in account if pd.Timestamp(s['date']) in ret.index]
    stages={}
    for y,g in ret.groupby(ret.index.year):
        before=nav.loc[nav.index<g.index[0]]
        base=float(before.iloc[-1]) if len(before) else initial
        stages[str(y)]=dict(net_return=float((1+g).prod()-1),net_profit=float(nav[g.index[-1]]-base))
    totals=dict(initial_cash=initial,final_equity=float(nav.iloc[-1]),net_profit=float(nav.iloc[-1]-initial),
        total_return=float(nav.iloc[-1]/initial-1),cagr=float((nav.iloc[-1]/initial)**(365.25/(ret.index[-1]-ret.index[0]).days)-1),
        max_drawdown=float((nav/nav.cummax().clip(lower=initial)-1).min()),
        mean_gross=float(np.mean([sum(s['weights'].values()) for s in states])),
        max_stock_weight=max((max(s['weights'].values(),default=0.) for s in states),default=0.),
        max_holdings=max(len(s['quantities']) for s in states),
        fees=sum(f['fees'] for f in result['fills']),slippage=sum(f['slippage_cost'] for f in result['fills']),
        fills=len(result['fills']),sessions=len(ret),first_invested=next((str(s['date']) for s in states if s['quantities']),None),
        data_issues=len(result['data_issues']),stages=stages,account_checks=check,
        annualized_two_way_turnover=sum(f['notional'] for f in result['fills'])/float(nav.reindex(ret.index).mean())*252/len(ret),
        status=result['status'],promotion_approved=False)
    return totals,ret


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    write(OUT/'started.json',dict(at=datetime.now(timezone.utc).isoformat(),protocol_sha256=shared.sha256(OUT/'protocol.json')))
    progress('loading_fixed_execution_projection')
    ds=projection(protocol)
    # Compatibility gate before nine research runs; all money/share states and
    # fills must match the already independently reconciled recent reference.
    progress('checking_original_reference_parity')
    original=replay.replay_reference(ds,archive_path('2025_recent'),start='2024-12-31',end='2026-09-22')
    complete=json.loads((ROOT/'output/reference_recent_20260927/complete.json').read_text())
    old=shared.checked_pickle(ROOT/'output/reference_recent_20260927/result.pkl',complete['result.pkl'])
    errors=[]
    if len(original['account'])!=len(old['account']):errors.append('calendar_count')
    for a,b in zip(original['account'],old['account']):
        for key in ['date','nav','cash','quantities','marks','receivables','payables']:
            if a.get(key)!=b.get(key):errors.append('account_'+key)
    if original['fills']!=old['fills']:errors.append('fills')
    write(OUT/'original_parity.json',dict(passed=not errors,errors=sorted(set(errors)),states=len(original['account']),fills=len(original['fills'])))
    if errors:raise ValueError('original_replay_parity_failed')
    del original,old
    for period,(start,end) in PERIODS.items():
        for n,mult in ((100,1),(50,1),(50,2)):
            arm=f'common{n}_cost{mult}';folder=OUT/period/arm
            progress('running_account',period=period,arm=arm)
            result=replay.replay_reference(ds,OUT/period/f'targets_{n}.json',start=start,end=end,
                                           cost_mult=mult,profile=replay.ReplayProfile.common(n))
            result['study_protocol_sha256']=shared.sha256(OUT/'protocol.json')
            result['parent_data_version']=protocol['parent_data_version']
            summary,returns=metrics(result,period)
            shared.publish(folder/'result.pkl',lambda f:pickle.dump(result,f,pickle.HIGHEST_PROTOCOL))
            write(folder/'summary.json',summary)
            shared.publish(folder/'daily_returns.csv',lambda f:f.write(returns.rename('daily_return').to_csv(index_label='date').encode()))
            write(folder/'complete.json',{name:shared.sha256(folder/name) for name in ('result.pkl','summary.json','daily_returns.csv')})
            progress('account_completed',period=period,arm=arm,cagr=summary['cagr'],max_drawdown=summary['max_drawdown'])
    verify(protocol)
    progress('accounts_complete',runs=9,parity_passed=True)


def intervals(excess):
    x=np.asarray(excess,float);n=len(x);out={}
    for block in (20,60):
        rng=np.random.default_rng(20260927)
        offsets=np.arange(block);samples=[]
        for _ in range(4000):
            starts=rng.integers(0,n,size=int(np.ceil(n/block)))
            idx=((starts[:,None]+offsets)%n).ravel()[:n]
            samples.append(float(x[idx].mean()*252))
        out[str(block)]={'mean_annualized_daily_excess':float(x.mean()*252),
            'family6_interval':np.quantile(samples,[.05/12,1-.05/12]).tolist(),
            'nominal95_interval':np.quantile(samples,[.025,.975]).tolist()}
    return out


def compare():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    allresults={};passed=True
    for period in PERIODS:
        summaries={arm:json.loads((OUT/period/arm/'summary.json').read_text()) for arm in protocol['arms']}
        a=pd.read_csv(OUT/period/'common100_cost1/daily_returns.csv',index_col='date').daily_return
        b=pd.read_csv(OUT/period/'common50_cost1/daily_returns.csv',index_col='date').daily_return
        if not a.index.equals(b.index):raise ValueError('paired_calendar_mismatch')
        ci=intervals(b-a)
        gate=(summaries['common50_cost1']['net_profit']>0 and
              summaries['common50_cost2']['net_profit']>0 and
              summaries['common50_cost1']['max_drawdown']>=-.25 and
              summaries['common50_cost2']['max_drawdown']>=-.25 and
              summaries['common50_cost1']['net_profit']>summaries['common100_cost1']['net_profit'] and
              all(v['family6_interval'][0]>0 for v in ci.values()))
        passed=passed and gate
        allresults[period]=dict(arms=summaries,paired_count50_minus100=ci,numeric_gate=gate)
    decision='retain_shadow_requires_risk_data_validation' if passed else 'count50_not_proven_superior_no_parameter_rescue'
    write(OUT/'comparison.json',dict(periods=allresults,decision=decision,promotion_approved=False,
        no_independent_OOS=True,historical_search_count_not_reset=True,
        scope='Count dependency, not isolated stock selection alpha; recent overlaps older period.',
        outstanding=['P2/account permission adaptation','full company-action/PIT gates','real exit pressure','forward validation']))
    progress('completed_conditional',decision=decision)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','compare'])
    args=parser.parse_args();globals()[args.action]()
