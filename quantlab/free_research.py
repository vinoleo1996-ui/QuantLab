"""First real-data replay with explicit assumptions, separate from promotion evidence."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd
from quantlab.config import CONFIG
from quantlab.data.contracts import load_dataset,fingerprint
from quantlab.data.qualification import research_assessment
from quantlab.backtest.engine import Engine,prepare
from quantlab.strategy.opportunity import opportunity_features,normalized_features,forward_labels,OpportunityModel,save_model
from quantlab.strategy.decision import review_dates
from quantlab.report.runs import persist_run,provenance,atomic_json
from quantlab.report.metrics import performance_table


def run_free_research(manifest,output,*,train_start='2017-01-01',test_start='2020-01-01'):
    ds=load_dataset(manifest,profile='free_research')
    assessment=research_assessment(ds)
    if assessment['status']!='research_ready':return assessment
    cfg=deepcopy(CONFIG);cfg.research.universe_mode='pit_liquidity';cfg.research.feature_profile='price_volume'
    cfg.start_date=test_start
    protocol=dict(version='first-free-real-replay-v1',train_start=train_start,test_start=test_start,
        candidate=dict(kind='ridge',alpha=1.,profile='price_volume',horizon=20),
        scope=ds.manifest.get('research_scope'),config=asdict(cfg),data_version=ds.version,
        note='One prespecified fit, no parameter selection. Retrospective engineering/research replay, not formal walk-forward OOS or alpha evidence.')
    before=provenance();identity=fingerprint(dict(protocol=protocol,code=before['code_sha256']))[:24]
    root=Path(output)/identity;root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():return json.loads((root/'summary.json').read_text())
    atomic_json(root/'protocol.json',protocol);atomic_json(root/'provenance.json',before)
    print('[free-research] extracting causal price/volume features',flush=True)
    raw=opportunity_features(ds,profile='price_volume');features=normalized_features(raw)
    labels=forward_labels(ds,20,allow_open_assumption=True)
    dates=ds.calendar[(ds.calendar>=pd.Timestamp(train_start))&(ds.calendar<pd.Timestamp(test_start))]
    model=OpportunityModel(**protocol['candidate']).fit(features,labels,dates,test_start)
    save_model(model,root/'model.json')
    raw.to_parquet(root/'raw_features.parquet');features.to_parquet(root/'features.parquet');labels.to_parquet(root/'labels.parquet')
    print('[free-research] preparing shared account inputs',flush=True)
    prepared=prepare(ds,cfg)
    reviews=review_dates(ds.calendar,cfg.capital.rebalance_rule,cfg.capital.rebalance_n_weeks)
    forecasts={d:model.predict(features,d,prepared['eligible'][d]) for d in reviews
               if d>=pd.Timestamp(test_start) and d in prepared['eligible']}
    serial=[]
    for f in forecasts.values():
        row=asdict(f);row['expected_returns']=f.expected_returns.to_dict();serial.append(row)
    atomic_json(root/'forecasts.json',dict(data_version=ds.version,forecasts=serial))
    engine=Engine(cfg);runs={}
    for name in ('opportunity','monthly_equal','buy_hold','opportunity_double_cost'):
        print('[free-research] replaying '+name,flush=True)
        run=engine.run(ds,strategy='opportunity' if name.startswith('opportunity') else name,forecasts=forecasts,
            prepared=prepared,data_mode='free_research',cost_mult=2. if name.endswith('double_cost') else 1.)
        persist_run(run,root=root/'runs');run['daily_return'].to_csv(root/(name+'_returns.csv'))
        run['nav'].to_csv(root/(name+'_nav.csv'))
        filled=run.get('fills',[]);health=run.get('run_health',{})
        runs[name]=dict(status=run['status'],run_id=run['run_id'],run_health=health,
            conditional_metrics=performance_table(run['nav'],run['daily_return']) if len(run['nav']) else None,
            fills=len(filled),fees=sum(f['fees'] for f in filled),slippage_cost=sum(f['slippage_cost'] for f in filled),
            rejected=sum(bool(o['reject_reason']) for o in run.get('orders',[])),
            unresolved_holdings={c:q for c,q in run.get('account',[{}])[-1].get('quantities',{}).items()
                if c in {g['instrument_id'] for g in ds.manifest.get('data_gaps',[])}},
            data_issues=run.get('data_issues',[]),allocation_issues=run.get('allocation_issues',[]))
    after=provenance()
    if before['code_sha256']!=after['code_sha256']:raise ValueError('source_changed_during_research')
    result=dict(status='research_unvalidated',promotion_approved=False,run_id=identity,data_version=ds.version,
        model_version=model.model_version,as_of=ds.manifest['as_of'],git_commit=before['git_commit'],git_dirty=before['git_dirty'],
        metrics=None,metric_kind='conditional_book_value_under_public_data_and_daily_open_assumptions',
        accounting_qualified=False,data_assessment=assessment,runs=runs,
        limitations=['fixed_initial_24_stock_cohort_not_full_market','industry_module_disabled_unknown_group_capped_at_25pct',
          'retired_positions_are_not_liquidated_at_last_price','unresolved_rights_are_not_bounded_by_last_quote',
          'no_formal_walk_forward_or_random_significance_claim','not_forward_or_executable_signals'],
        artifact_directory=str(root))
    atomic_json(root/'summary.json',result)
    return result
