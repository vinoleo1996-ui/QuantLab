"""Registered walk-forward research on one cash ledger. Missing evidence stops inference."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from quantlab.config import CONFIG
from quantlab.data.contracts import fingerprint
from quantlab.data.qualification import qualification
from quantlab.strategy.opportunity import opportunity_features,normalized_features,forward_labels,OpportunityModel,Forecast
from quantlab.strategy.decision import review_dates
from quantlab.backtest.engine import Engine,prepare,blocked_result
from quantlab.research import ExperimentRegistry,walk_forward_splits,block_bootstrap_ci
from quantlab.report.runs import persist_run,provenance
from quantlab.report.metrics import performance_table

DEFAULT_PROTOCOL=dict(version='opportunity-research-v1',seed=20260924,random_trials=1000,preflight_trials=20,
    candidates=[dict(kind='equal',alpha=1.),dict(kind='ridge',alpha=.1),dict(kind='ridge',alpha=1.),dict(kind='ridge',alpha=10.)],
    primary_metric='annualized_mean_daily_net_excess',diagnostics=['without_opportunity','without_crowding','without_buffer','monthly','horizon21','without_etfs','legacy_risk'],
    interpretation='retrospective; neither conditional random rank nor historical OOS proves future alpha')


def compare_returns(a,b):
    pair=pd.concat([a.rename('strategy'),b.rename('control')],axis=1).dropna()
    return pair.strategy-pair.control


def usable(run):
    return run.get('status')=='research_unvalidated' and run.get('run_health',{}).get('research_interpretable',False)


def summary(run):
    nav,ret=run['nav'],run['daily_return']
    valid=usable(run)
    fees=sum(f['fees']+f.get('slippage_cost',0) for f in run.get('fills',[]))
    turnover=sum(f['notional'] for f in run.get('fills',[]))/run.get('config',{}).get('capital',{}).get('init_cash',600000)
    return dict(status=run['status'],run_health=run.get('run_health',{}),metrics=performance_table(nav,ret) if valid else None,
                fees_and_slippage=fees,turnover_initial_capital=turnover,
                pending=sum(o['status'] in ('pending','partial') for o in run.get('orders',[])),
                rejected_or_expired=sum(bool(o['reject_reason']) for o in run.get('orders',[])))


def _forecasts(model,features,ds,cfg,dates,prepared):
    reviews=set(review_dates(ds.calendar,cfg.capital.rebalance_rule,cfg.capital.rebalance_n_weeks))
    return {d:model.predict(features,d,prepared['eligible'][d]) for d in dates if d in reviews and d in prepared['eligible']}


def run_suite(ds,root,protocol=None,cfg=None,allow_test_fixture=False):
    cfg=deepcopy(cfg or CONFIG);protocol=deepcopy(protocol or DEFAULT_PROTOCOL);root=Path(root)
    identity=fingerprint(dict(protocol=protocol,data=ds.version,config=asdict(cfg),code=provenance()["code_sha256"]))
    out=root/identity[:20];out.mkdir(parents=True,exist_ok=True)
    contract=out/'protocol.json'
    payload=dict(protocol=protocol,data_version=ds.version,config=asdict(cfg))
    if contract.exists() and json.loads(contract.read_text())!=json.loads(json.dumps(payload,default=str)):raise ValueError('protocol_immutable')
    if not contract.exists():contract.write_text(json.dumps(payload,default=str,indent=2))
    assessment=qualification(ds)
    fixture=ds.manifest.get('source')=='TEST_FIXTURE' and allow_test_fixture
    if assessment['status']!='qualified_for_daily_research' and not fixture:
        result=blocked_result(assessment['reasons']);persist_run(result,root=out/'runs')
        (out/'summary.json').write_text(json.dumps(assessment,default=str,indent=2));return assessment
    if protocol['random_trials']<200 and not fixture:raise ValueError('at_least_200_random_trials')
    if len(protocol['candidates'])+len(protocol.get('diagnostics',[]))>cfg.research.max_trials:raise ValueError('model_search_budget_exhausted')
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    registry=ExperimentRegistry(out/'registry',cfg.research)
    # Register all model/diagnostic definitions before observing any validation results.
    for spec in protocol['candidates']+[{ 'diagnostic':x} for x in protocol.get('diagnostics',[])]:
        registry.register('opportunity-module',spec,ds.version,'protocol','opportunity-v1')
    raw=opportunity_features(ds);features=normalized_features(raw);labels=forward_labels(ds,cfg.research.horizon)
    raw.to_parquet(out/'raw_features.parquet');features.to_parquet(out/'features.parquet');labels.to_parquet(out/'labels.parquet')
    ends=labels.groupby(level='date').label_end.max()
    calendar=ds.calendar[ds.calendar<=pd.Timestamp(ds.manifest['as_of']).normalize()]
    folds=walk_forward_splits(calendar,cfg.research,label_ends=ends)
    if not folds:
        result=dict(status='insufficient_sample',reasons=['complete_walk_forward_window_missing'],promotion_approved=False)
        (out/'summary.json').write_text(json.dumps(result,indent=2));return result
    prepared=prepare(ds,cfg);engine=Engine(cfg);all_forecasts={};selection=[];models=[]
    for k,fold in enumerate(folds):
        valid=fold['validation'];test=fold['test'];evaluated=[]
        control=engine.run(ds,strategy='vol_matched_equal',start=valid[0],end=valid[-1],prepared=prepared)
        for spec in protocol['candidates']:
            trial=registry.register('opportunity-module',spec,ds.version,str(k),'opportunity-v1')
            try:
                model=OpportunityModel(**spec,horizon=cfg.research.horizon).fit(features,labels,fold['train'],valid[0])
                forecast=_forecasts(model,features,ds,cfg,valid,prepared)
                run=engine.run(ds,strategy='opportunity',forecasts=forecast,start=valid[0],end=valid[-1],prepared=prepared)
                value=float(compare_returns(run['daily_return'],control['daily_return']).mean()*252) if usable(run) and usable(control) else None
                outcome=dict(fold=str(k),status='evaluated' if value is not None else 'invalid_run',net_excess=value,summary=summary(run))
                persist_run(run,root=out/'runs')
                if value is not None and np.isfinite(value):evaluated.append((value,spec))
            except (ValueError,KeyError) as exc:outcome=dict(fold=str(k),status='failed',reason=str(exc))
            registry.record(trial,outcome)
        if not evaluated:
            selection.append(dict(fold=k,status='no_valid_candidate'));continue
        best=max(evaluated,key=lambda x:x[0])[1]
        fit_dates=fold['train'].append(fold['validation'])
        model=OpportunityModel(**best,horizon=cfg.research.horizon).fit(features,labels,fit_dates,test[0])
        forecasts=_forecasts(model,features,ds,cfg,test,prepared);all_forecasts.update(forecasts)
        from quantlab.strategy.opportunity import save_model
        save_model(model,out/'models'/f'fold_{k}.json')
        models.append((fold,best,model))
        selection.append(dict(fold=k,status='selected',candidate=best,start=str(test[0]),end=str(test[-1]),model_version=model.model_version,training_cutoff=model.training_cutoff))
    (out/'selection.json').write_text(json.dumps(selection,default=str,indent=2))
    if len(models)!=len(folds):
        result=dict(status='unproven',reasons=['one_or_more_folds_have_no_valid_candidate'],selection=selection,promotion_approved=False)
        (out/'summary.json').write_text(json.dumps(result,indent=2));return result
    serial=[]
    for f in all_forecasts.values():
        row=asdict(f);row['expected_returns']=f.expected_returns.to_dict();serial.append(row)
    (out/'forecasts.json').write_text(json.dumps(dict(data_version=ds.version,forecasts=serial),default=str,indent=2))
    start,end=folds[0]['test'][0],folds[-1]['test'][-1]
    observed=engine.run(ds,strategy='opportunity',forecasts=all_forecasts,start=start,end=end,prepared=prepared)
    controls={name:engine.run(ds,strategy=name,start=start,end=end,prepared=prepared) for name in ('buy_hold','monthly_equal','vol_matched_equal','repaired_legacy')}
    for run in [observed,*controls.values()]:persist_run(run,root=out/'runs')
    paired=compare_returns(observed['daily_return'],controls['vol_matched_equal']['daily_return'])
    paired.to_csv(out/'paired_oos_excess.csv')
    random=[];rng=np.random.default_rng(protocol['seed']);healthy=usable(observed)
    for i in range(protocol['random_trials']):
        order=list(rng.permutation(sorted(ds.panels)))
        randomized={}
        for d,f in all_forecasts.items():
            names=[c for c in order if c in f.expected_returns.index]
            mu=pd.Series(sorted(f.expected_returns.reindex(names).values,reverse=True),index=names)
            randomized[d]=Forecast(f.as_of,f.horizon,mu,f.status,f.reasons,f.training_cutoff,f.model_version+'-random-'+str(i))
        run=engine.run(ds,strategy='opportunity',forecasts=randomized,start=start,end=end,prepared=prepared)
        healthy=healthy and usable(run)
        value=float(compare_returns(run['daily_return'],controls['vol_matched_equal']['daily_return']).mean()*252) if usable(run) else None
        row=dict(trial=i,priority=order,net_excess=value,**summary(run));random.append(row)
        run['daily_return'].to_csv(out/f'random_{i:04d}.csv')
        if i+1==protocol['preflight_trials'] and not healthy:break
    (out/'random_trials.json').write_text(json.dumps(random,default=str,indent=2))
    incremental=[float(paired.loc[f['test'][0]:f['test'][-1]].mean()*252) for f in folds]
    diagnostics={}
    for name in protocol.get('diagnostics',[]):
        trial=registry.register('opportunity-module',{'diagnostic':name},ds.version,'diagnostic','opportunity-v1')
        dc=deepcopy(cfg);forecasts=all_forecasts;kwargs={}
        if name=='without_opportunity':kwargs['strategy']='repaired_legacy'
        if name=='without_crowding':kwargs['crowding']=False
        if name=='without_buffer':
            dc.capital.min_holding_period_days=0;dc.research.entry_percentile=0.;dc.research.exit_reviews=1;dc.research.exit_percentile=1.
        if name=='monthly':dc.capital.rebalance_rule='monthly'
        if name=='legacy_risk':kwargs['risk_mode']='legacy_monthly'
        if name=='without_etfs':dc.research.max_etfs=0
        if name=='horizon21':dc.research.horizon=21
        dp=prepare(ds,dc)
        if name in ('monthly','horizon21'):
            forecasts={};dl=labels if dc.research.horizon==20 else forward_labels(ds,dc.research.horizon)
            for f,best,_ in models:
                model=OpportunityModel(**best,horizon=dc.research.horizon).fit(features,dl,f['train'].append(f['validation']),f['test'][0])
                forecasts.update(_forecasts(model,features,ds,dc,f['test'],dp))
        run=Engine(dc).run(ds,strategy=kwargs.pop('strategy','opportunity'),forecasts=forecasts,start=start,end=end,prepared=dp,**kwargs)
        diagnostics[name]=summary(run);registry.record(trial,dict(fold='diagnostic',**diagnostics[name]));persist_run(run,root=out/'runs')
    ci20=block_bootstrap_ci(paired,seed=protocol['seed'],block=20)
    ci60=block_bootstrap_ci(paired,seed=protocol['seed'],block=60)
    from quantlab.report.attribution import concentration_diagnostics
    concentration=concentration_diagnostics(observed,controls['vol_matched_equal'],ds)
    stresses=stress_suite(ds,cfg,all_forecasts,start,end,out/'stress')
    result=dict(status='fixture_test_only' if fixture else 'unproven',promotion_approved=False,
        observed=summary(observed),controls={k:summary(v) for k,v in controls.items()},
        benchmark=dict(identity=ds.manifest['benchmark_type'],return_basis=ds.manifest.get('benchmark_return_basis','unspecified')),
        benchmark_comparison=performance_table(observed['nav'],observed['daily_return'],ds.benchmark.close) if usable(observed) else None,
        fold_net_excess=incremental,ci20=ci20,ci60=ci60,random_trials_completed=len(random),
        preflight_passed=healthy,diagnostics=diagnostics,stress=stresses,concentration=concentration,
        mc_p_upper=(1+sum(x['net_excess']>=float(paired.mean()*252) for x in random))/(len(random)+1) if not fixture and healthy and len(random)==protocol['random_trials'] else None,
        reasons=['historical_research_not_forward_validation','execution_calibration_required','concentration_ablation_review_required']+([] if healthy else ['experiment_mechanics_failed']),
        selection=selection,artifact_directory=str(out))
    from quantlab.evidence import export_research_bundle,verify_bundle
    bundle=export_research_bundle(observed,ds,result,out/'evidence')
    gate=verify_bundle(bundle,out/'evidence')
    result['evidence_bundle']=str(out/'evidence'/'bundle.json')
    result['evidence_gate']=dict(status=gate['status'],reasons=gate['reasons'],verified=gate['verified'])
    (out/'summary.json').write_text(json.dumps(result,default=str,indent=2));return result


def stress_suite(ds,cfg,forecasts,start,end,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);cal=ds.calendar[(ds.calendar>=start)&(ds.calendar<=end)]
    engine=Engine(cfg);base=engine.run(ds,strategy='opportunity',forecasts=forecasts,start=start,end=end)
    adverse=base['daily_return'].rolling(5).sum().idxmin() if len(base['daily_return'])>=5 else cal[-1]
    loc=max(0,cal.get_indexer([adverse],method='pad')[0]-4)
    cases={'double_cost':dict(cost_mult=2.),'liquidity_minus_70pct':dict(liquidity_mult=.3),
           'five_sessions_locked':dict(blocked_sessions=list(cal[20:25])),
           'adverse_five_sessions_locked':dict(blocked_sessions=list(cal[loc:loc+5])),
           'halted':dict(blocked_sessions=list(cal[10:20])), 'execution_delay':dict(execution_delay=1),
           'missing_quotes':{},'opening_gap':{},'correlation_up':{}}
    rows={}
    for name,kw in cases.items():
        stressed=deepcopy(ds)
        if len(cal)>25 and name in ('missing_quotes','opening_gap','correlation_up'):
            day=cal[25]
            for c,p in stressed.panels.items():
                if day not in p.index:continue
                if name=='missing_quotes':p.loc[day,['open','close']]=np.nan
                elif name=='opening_gap':
                    p.loc[day:,'open']*=.9;p.loc[day:,'close']*=.9
                    for col in ('reference_price','limit_up','limit_down'):
                        if col in p:p.loc[p.index>day,col]*=.9
                else:
                    # Synthetic common down shock, explicitly a stress rather than history.
                    shocks=np.linspace(1.,.8,min(5,len(p.loc[day:])))
                    factor=pd.Series(.8,index=p.loc[day:].index);factor.iloc[:len(shocks)]=shocks
                    for col in ('open','close','reference_price','limit_up','limit_down'):
                        if col in p:p.loc[day:,col]*=factor
        run=engine.run(stressed,strategy='opportunity',forecasts=forecasts,start=start,end=end,**kw)
        rows[name]=dict(summary(run),constraints_passed=bool(run.get('account')) and all(a['cash']>=0 and all(q>=0 for q in a['quantities'].values()) for a in run.get('account',[])),
                        retrospective_scenario=name=='adverse_five_sessions_locked',frozen_signal_execution_stress=True)
        persist_run(run,root=root)
    return rows
