"""Bounded, retrospective return comparison; all arms use the same cash ledger.

Legacy/trend arms exchange only the security ranking of a common forecast
distribution. They are ranking ablations, not replicas of the old reported NAV.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.data.contracts import fingerprint, load_dataset
from quantlab.data.qualification import research_assessment
from quantlab.backtest.engine import Engine, prepare
from quantlab.strategy.decision import FactorDecisionService, review_dates
from quantlab.strategy.opportunity import (Forecast, OpportunityModel, forward_labels,
    normalized_features, opportunity_features, save_model)
from quantlab.research import ExperimentRegistry, walk_forward_splits, block_bootstrap_ci
from quantlab.report.runs import atomic_json, persist_run, provenance
from quantlab.report.metrics import performance_table


PROTOCOL = dict(version='return-comparison-v1', seed=20260924,
    cohort='120 initial-2014 mainboard securities, fixed hash priority; later delistings retained',
    split_origin='2016-07-01', frozen_train_start='2017-01-01', test_start='2020-01-01',
    train_months=36, validation_months=6, test_months=6, horizon=20,
    ridge_alphas=[.1, 1., 10.], entry_buffer=.005,
    primary='annualized mean of paired daily net returns',
    arms=['rolling_ridge','frozen_ridge','legacy_rank','trend_rank','rolling_buffer',
          'monthly_equal','vol_matched_equal'],
    matching='Forecast names intersect causal reference availability; ranking arms preserve every expected-return value. Account, execution, cash, fees and hard limits shared.',
    legacy='Current repaired FactorDecisionService with experts; 11 price/volume factors, no financial statements; current strict allocator, not historical full-investment optimizer.',
    trend='Mean percentile rank of 60/120-session price returns skipping last 5 sessions',
    test_policy='All arms reported; no automatic winner promotion or tuning using test results',
    interpretation='Retrospective conditional public-data evidence; not full-market, forward or qualified performance')


def rank_matched(forecast, scores, name):
    """Exchange security identities without changing the expected-return budget."""
    scores=scores.reindex(forecast.expected_returns.index)
    if scores.isna().any() or scores.nunique()<2:
        return replace(forecast,status='invalid',reasons=['reference_scores_missing_or_constant'])
    order=scores.sort_index().sort_values(ascending=False,kind='stable').index
    mu=pd.Series(np.sort(forecast.expected_returns.to_numpy())[::-1],index=order)
    return replace(forecast,expected_returns=mu,
        model_version=fingerprint(dict(base=forecast.model_version,ranking=name)),
        trace=dict(forecast.trace,reference=name,reference_scores=scores.to_dict(),
                   reference_kind='matched_distribution_ranking_ablation'))


def restrict_forecast(forecast, legacy, trend):
    common=forecast.expected_returns.index.intersection(legacy.dropna().index).intersection(trend.dropna().index)
    mu=forecast.expected_returns.reindex(common)
    ok=forecast.status=='ok' and len(mu)>1 and mu.nunique()>1 and legacy.reindex(common).nunique()>1 and trend.reindex(common).nunique()>1
    return replace(forecast,expected_returns=mu,status='ok' if ok else 'invalid',
        reasons=[] if ok else ['common_reference_universe_insufficient'],
        trace=dict(forecast.trace,original_forecast_count=len(forecast.expected_returns),common_forecast_count=len(mu)))


def reference_scores(ds, prepared, cfg, dates):
    legacy_cfg=deepcopy(cfg);legacy_cfg.factors.use_experts=True
    service=FactorDecisionService(prepared['close'],prepared['amount'],prepared['turnover'],ds.calendar,legacy_cfg)
    close=prepared['close']
    trend=((close.shift(5)/close.shift(65)-1).rank(axis=1,pct=True)+
           (close.shift(5)/close.shift(125)-1).rank(axis=1,pct=True))/2
    legacy={};simple={};trace=[]
    for d in dates:
        eligible=list(prepared['eligible'].get(d,[]))
        # Unknown rights must not become a spurious 271-session legacy trend.
        for gap in ds.manifest.get('data_gaps',[]):
            gd=pd.Timestamp(gap['date']);age=ds.calendar.searchsorted(d)-ds.calendar.searchsorted(gd)
            if age>=0 and (gap['reason']=='delisting_settlement_unverified' or age<=271):
                eligible=[c for c in eligible if c!=gap['instrument_id']]
        decision=service.score(d,eligible)
        legacy[d]=decision.scores
        simple[d]=trend.loc[d].reindex(eligible).dropna()
        trace.append(dict(date=str(d),status=decision.status,reasons=decision.reasons,**decision.trace))
    return legacy,simple,trace


def mechanically_usable(run):
    h=run.get('run_health',{})
    return bool(len(run.get('nav',[]))) and not any(h.get(k,1) for k in ('allocation_failures','forecast_failures','data_issue_count'))


def account_summary(run):
    account=run.get('account',[]);fills=run.get('fills',[])
    return dict(run_id=run['run_id'],status=run['status'],run_health=run.get('run_health',{}),
        mechanically_usable=mechanically_usable(run),metrics=None,
        conditional_metrics=performance_table(run['nav'],run['daily_return']) if account else None,
        fills=len(fills),fees=sum(f['fees'] for f in fills),slippage=sum(f['slippage_cost'] for f in fills),
        two_way_traded_notional=sum(f['notional'] for f in fills),
        mean_holdings=float(np.mean([len(a['quantities']) for a in account])) if account else None,
        pending_orders=sum(o['status'] in ('pending','partial') for o in run.get('orders',[])),
        accounting_identity_passed=bool(account) and all(a['cash']>=0 and abs(a['nav']-a['cash']-a['receivables']-sum(q*a['marks'][c] for c,q in a['quantities'].items()))<1e-6 for a in account),
        next_session_fills_only=all(pd.Timestamp(f['date'])>pd.Timestamp(f['signal_date']) for f in fills))


def paired_summary(a,b,folds,seed):
    pair=pd.concat({'a':a['daily_return'],'b':b['daily_return']},axis=1).dropna();excess=pair.a-pair.b
    segments=[dict(start=str(f['test'][0].date()),end=str(f['test'][-1].date()),
        annualized_net_excess=float(excess.loc[f['test'][0]:f['test'][-1]].mean()*252)) for f in folds]
    return dict(conditional=True,comparison_mechanically_usable=mechanically_usable(a) and mechanically_usable(b),
        annualized_mean_daily_net_excess=float(excess.mean()*252),
        relative_wealth=float((1+pair.a).prod()/(1+pair.b).prod()-1),
        positive_segments=sum(x['annualized_net_excess']>0 for x in segments),segments=segments,
        ci20=block_bootstrap_ci(excess,seed=seed,block=20),ci60=block_bootstrap_ci(excess,seed=seed,block=60))


def run_comparison(manifest, output):
    ds=load_dataset(manifest,profile='free_research');assessment=research_assessment(ds)
    if assessment['status']!='research_ready':return assessment
    cfg=deepcopy(CONFIG);cfg.seed=PROTOCOL['seed'];cfg.research.feature_profile='price_volume'
    cfg.research.universe_mode='pit_liquidity'
    before=provenance();identity=fingerprint(dict(protocol=PROTOCOL,data=ds.version,config=asdict(cfg),source=before))[:24]
    root=Path(output)/identity;root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():return json.loads((root/'summary.json').read_text())
    atomic_json(root/'protocol.json',dict(protocol=PROTOCOL,config=asdict(cfg),data_version=ds.version,assessment=assessment))
    atomic_json(root/'provenance.json',before)
    registry=ExperimentRegistry(root/'registry',cfg.research)
    specs=[dict(alpha=a) for a in PROTOCOL['ridge_alphas']]
    for spec in specs:registry.register('rolling-ridge',spec,ds.version,'protocol','comparison-v1')
    for name in PROTOCOL['arms'][1:]:registry.register('comparison-arm',{'arm':name},ds.version,'protocol','comparison-v1')
    print('[comparison] data preparation',flush=True)
    prepared=prepare(ds,cfg)
    raw=opportunity_features(ds,profile='price_volume');features=normalized_features(raw)
    labels=forward_labels(ds,20,allow_open_assumption=True)
    # Train only on the pool actually eligible at each historical decision.
    mask=[c in prepared['eligible'].get(d,[]) for d,c in features.index]
    features=features.loc[mask]
    raw.to_parquet(root/'raw_features.parquet');features.to_parquet(root/'features.parquet');labels.to_parquet(root/'labels.parquet')
    cal=ds.calendar[(ds.calendar>=pd.Timestamp(PROTOCOL['split_origin']))&(ds.calendar<=pd.Timestamp(ds.manifest['as_of']))]
    folds=walk_forward_splits(cal,cfg.research,label_ends=labels.groupby(level='date').label_end.max())
    if not folds:raise ValueError('insufficient_complete_walk_forward_windows')
    reviews=[d for d in review_dates(ds.calendar,'weekly') if folds[0]['validation'][0]<=d<=folds[-1]['test'][-1]]
    legacy,trend,traces=reference_scores(ds,prepared,cfg,reviews)
    atomic_json(root/'legacy_score_trace.json',traces)
    engine=Engine(cfg);rolling={};selections=[]
    def forecasts(model, dates):
        return {d:restrict_forecast(model.predict(features,d,prepared['eligible'][d]),legacy[d],trend[d])
                for d in dates if d in legacy}
    def replay(fc, **kwargs):
        return engine.run(ds,forecasts=fc,prepared=prepared,data_mode='free_research',**kwargs)
    for k,fold in enumerate(folds):
        print('[comparison] validation fold '+str(k+1)+'/'+str(len(folds)),flush=True)
        valid=fold['validation'];test=fold['test'];evaluated=[]
        control=replay(None,strategy='vol_matched_equal',start=valid[0],end=valid[-1])
        for spec in specs:
            trial=registry.register('rolling-ridge',spec,ds.version,str(k),'comparison-v1')
            try:
                model=OpportunityModel(**spec,profile='price_volume').fit(features,labels,fold['train'],valid[0])
                run=replay(forecasts(model,valid),start=valid[0],end=valid[-1])
                persist_run(run,root=root/'validation_runs')
                pair=pd.concat([run['daily_return'],control['daily_return']],axis=1).dropna()
                score=float((pair.iloc[:,0]-pair.iloc[:,1]).mean()*252) if mechanically_usable(run) and mechanically_usable(control) else None
                outcome=dict(fold=str(k),status='evaluated' if score is not None else 'unusable_account',net_excess=score,summary=account_summary(run))
                if score is not None:evaluated.append((score,-spec['alpha'],spec))
            except (ValueError,KeyError) as exc:outcome=dict(fold=str(k),status='failed',reason=str(exc))
            registry.record(trial,outcome)
        if not evaluated:
            atomic_json(root/'failure.json',dict(status='incomplete_comparison',fold=k,reason='no_mechanically_usable_validation_candidate',promotion_approved=False))
            raise ValueError('no_mechanically_usable_validation_candidate:'+str(k))
        best=max(evaluated,key=lambda x:x[:2])[2]
        model=OpportunityModel(**best,profile='price_volume').fit(features,labels,fold['train'].append(valid),test[0])
        save_model(model,root/'models'/('fold_'+str(k)+'.json'))
        rolling.update(forecasts(model,test))
        selections.append(dict(fold=k,alpha=best['alpha'],test_start=str(test[0]),test_end=str(test[-1]),training_cutoff=model.training_cutoff,model_version=model.model_version))
        atomic_json(root/'selection.json',selections)
    start,end=folds[0]['test'][0],folds[-1]['test'][-1]
    frozen=OpportunityModel(alpha=1.,profile='price_volume').fit(features,labels,ds.calendar[(ds.calendar>=pd.Timestamp(PROTOCOL['frozen_train_start']))&(ds.calendar<start)],start)
    save_model(frozen,root/'models'/'frozen.json')
    fc={'rolling_ridge':rolling,'frozen_ridge':forecasts(frozen,ds.calendar[(ds.calendar>=start)&(ds.calendar<=end)]),
        'legacy_rank':{d:rank_matched(f,legacy[d],'legacy_rank') for d,f in rolling.items()},
        'trend_rank':{d:rank_matched(f,trend[d],'trend_rank') for d,f in rolling.items()}}
    runs={}
    for name in PROTOCOL['arms']:
        print('[comparison] replay '+name,flush=True)
        if name=='rolling_buffer':
            buffered=deepcopy(cfg);buffered.research.entry_weight_buffer=PROTOCOL['entry_buffer']
            run=Engine(buffered).run(ds,forecasts=rolling,start=start,end=end,data_mode='free_research')
        elif name in fc:run=replay(fc[name],start=start,end=end)
        else:run=replay(None,strategy=name,start=start,end=end)
        persist_run(run,root=root/'runs');runs[name]=run
        run['daily_return'].to_csv(root/(name+'_returns.csv'))
        if name in fc:
            atomic_json(root/(name+'_forecasts.json'),dict(data_version=ds.version,forecasts=[asdict(f)|{'expected_returns':f.expected_returns.to_dict()} for f in fc[name].values()]))
        trial=registry.register('comparison-arm',{'arm':name},ds.version,'test','comparison-v1') if name!='rolling_ridge' else None
        if trial:registry.record(trial,dict(fold='test',**account_summary(run)))
    print('[comparison] pressure replays',flush=True)
    stresses={}
    for name,kw in [('double_cost',dict(cost_mult=2.)),('liquidity_minus_70pct',dict(liquidity_mult=.3)),
                    ('five_day_exit_lock',dict(blocked_sessions=list(ds.calendar[(ds.calendar>=start)&(ds.calendar<=end)][100:105]))),
                    ('one_session_delay',dict(execution_delay=1))]:
        run=replay(rolling,start=start,end=end,**kw);persist_run(run,root=root/'stress_runs')
        stresses[name]=account_summary(run)
    after=provenance()
    if before!=after:raise ValueError('source_or_environment_changed_during_comparison')
    result=dict(status='research_unvalidated',run_id=identity,data_version=ds.version,model_version='return-comparison-v1',as_of=str(end),
        git_commit=before['git_commit'],git_dirty=before['git_dirty'],promotion_approved=False,metrics=None,
        dataset_securities=len(ds.panels),data_assessment=assessment,selection=selections,
        runs={k:account_summary(v) for k,v in runs.items()},
        comparisons={name:paired_summary(runs[name],runs['rolling_ridge'],folds,PROTOCOL['seed']) for name in ('frozen_ridge','legacy_rank','trend_rank','rolling_buffer')},
        versus_equal=paired_summary(runs['rolling_ridge'],runs['vol_matched_equal'],folds,PROTOCOL['seed']),
        stress=stresses,artifact_directory=str(root),
        limitations=['retrospective_history_previously_examined','fixed_initial_cohort_not_full_dynamic_universe',
            'unknown_industry_group_capped_at_25pct','public_rights_and_opening_execution_unverified',
            'legacy_and_trend_are_ranking_ablations_not_old_strategy_replications',
            'double_cost_reoptimizes_portfolio_not_a_fixed_order_cost_attribution'])
    atomic_json(root/'summary.json',result)
    return result
