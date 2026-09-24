"""Fixed six-arm repair attribution. No tuning, model selection or promotion."""
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import json
import tarfile
import numpy as np
import pandas as pd

from quantlab.config import CONFIG, ROOT
from quantlab.data.contracts import load_dataset, fingerprint
from quantlab.data.qualification import research_assessment
from quantlab.strategy.legacy import legacy_config
from quantlab.backtest.engine import Engine, prepare
from quantlab.report.runs import provenance, persist_run, load_run, atomic_json
from quantlab.report.metrics import performance_table
from quantlab.report.attribution import instrument_returns
from quantlab.research import block_bootstrap_ci, ExperimentRegistry

PROTOCOL = dict(version='legacy-repair-comparison-v2',seed=20260924,start='2020-01-02',end='2026-06-30',
    strategies=['legacy_repaired','monthly_equal','vol_matched_equal'],
    industry_policies=['enforced','omit_for_diagnostic'],
    selection='None. Fixed archived-factor baseline; no Ridge return scale or supervised fitting.',
    interpretation='Conditional retrospective public-data research. Industry exception is not a mandate change.',
    cost='Actual ledger commission/tax plus 30bp one-way slippage; no artificial zero-cost fills',
    controls='All use position/industry/downside/execution caps. Monthly equal has no drawdown/single-stock timing; volatility equal and legacy use daily risk.')


def comparison_protocol(dataset, *, diagnostic_sample=False):
    """Declare the input universe; a security count never proves market coverage."""
    scope = dataset.manifest.get('research_scope', {})
    declared = str(scope.get('scope', 'unspecified'))
    sampled = ('cohort' in declared or 'sample' in declared
               or scope.get('sample_size') is not None)
    if sampled and not diagnostic_sample:
        raise ValueError('diagnostic_sample_requires_explicit_flag; fixed samples cannot stand in for market research')
    return dict(PROTOCOL, diagnostic_sample=bool(diagnostic_sample), universe=dict(
        declared_scope=scope, securities_with_panels=len(dataset.panels),
        market_coverage='not_established',
        eligibility='historical listing age, active/ST status and prior available liquidity at each decision',
        sampling='explicit_diagnostic_sample' if diagnostic_sample else 'no_comparison_layer_sampling'))


def account_statistics(run):
    nav=run['nav'];ret=run['daily_return'];states=run['account'];capital=run['config']['capital']['init_cash']
    contributions=instrument_returns(run)
    prior_nav=nav.shift().fillna(1.)*capital
    stock_pnl=contributions.mul(prior_nav,axis=0).sum().sort_values(ascending=False)
    reconciliation=float((contributions.sum(axis=1)-ret).abs().max())
    half=[dict(period=f'{y}H{h}',net_return=float((1+x).prod()-1))
          for (y,h),x in ret.groupby([ret.index.year,(ret.index.month>6).astype(int)+1])]
    underwater=nav<nav.cummax().clip(lower=1.)-1e-10;longest=duration=0
    for v in underwater:duration=duration+1 if v else 0;longest=max(longest,duration)
    fills=run['fills'];fees=sum(f['fees'] for f in fills);slip=sum(f['slippage_cost'] for f in fills)
    sectors=Counter()
    for r in run['risk_history']:
        for key,value in r['actual_sector_exposure'].items():sectors[key]+=value/len(states)
    score_groups=[]
    traces=[t for t in run['signal_trace'] if t.get('scores') and t.get('target_weights')]
    for i,t in enumerate(traces):
        d=pd.Timestamp(t['date']);end=pd.Timestamp(traces[i+1]['date']) if i+1<len(traces) else ret.index[-1]
        pnl=contributions.loc[(contributions.index>d)&(contributions.index<=end)].mul(prior_nav,axis=0).sum()
        ranks=pd.Series(t['scores']).rank(pct=True)
        score_groups.append(dict(date=str(d.date()),next_decision=str(end.date()),stocks=[dict(instrument_id=c,
            score=t['scores'].get(c),percentile=ranks.get(c),target_weight=w,pnl_contribution=float(pnl.get(c,0)))
            for c,w in t['target_weights'].items() if w>1e-8]))
    return dict(run_id=run['run_id'],status=run['status'],industry_policy=run['industry_policy'],
        conditional_metrics=performance_table(nav,ret),metrics=None,promotion_approved=False,
        net_profit=float(states[-1]['nav']-capital),ending_equity=states[-1]['nav'],
        run_health=run['run_health'],mean_stock_weight=float(np.mean([sum(a['weights'].values()) for a in states])),
        mean_holdings=float(np.mean([len(a['quantities']) for a in states])),mean_industry_exposure=dict(sectors),
        eligible_universe_history=[dict(date=str(pd.Timestamp(x['date']).date()),eligible_count=x.get('eligible_count'))
                                   for x in run['risk_history']],
        fees=fees,slippage=slip,traded_notional=sum(f['notional'] for f in fills),
        traded_notional_over_mean_nav=sum(f['notional'] for f in fills)/float(np.mean([a['nav'] for a in states])),
        halfyears=half,positive_halfyears=sum(x['net_return']>0 for x in half),
        longest_underwater_sessions=longest,unrecovered_at_end=bool(underwater.iloc[-1]),
        stock_pnl=stock_pnl.to_dict(),best_five_daily_returns={str(d.date()):float(v) for d,v in ret.nlargest(5).items()},
        score_to_position_pnl=score_groups,attribution_reconciliation_error=reconciliation,
        capital_diagnostics=run['capital_diagnostics'],allocation_issues=run['allocation_issues'],data_issues=run['data_issues'])


def run_comparison(manifest, output, *, diagnostic_sample=False):
    ds=load_dataset(manifest,profile='free_research')
    protocol=comparison_protocol(ds,diagnostic_sample=diagnostic_sample)
    assessment=research_assessment(ds,require_industry=False)
    if assessment['status']!='research_ready':return assessment
    cfg=legacy_config(CONFIG);cfg.seed=PROTOCOL['seed'];cfg.research.feature_profile='price_volume';cfg.research.universe_mode='pit_liquidity'
    before=provenance();identity=fingerprint(dict(protocol=protocol,config=asdict(cfg),data=ds.version,code=before))[:24]
    root=Path(output)/identity;root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'protocol.json',dict(protocol=protocol,config=asdict(cfg),data_version=ds.version,provenance=before))
    if (root/'summary.json').exists():return json.loads((root/'summary.json').read_text())
    with tarfile.open(root/'source.tar.gz','w:gz') as archive:
        for file in sorted([*(ROOT/'quantlab').rglob('*.py'),*(ROOT/'scripts').glob('*.py'),*(ROOT/'tests').glob('*.py')]):
            archive.add(file,arcname=str(file.relative_to(ROOT)))
    registry=ExperimentRegistry(root/'registry')
    arms=[(s,p) for s in PROTOCOL['strategies'] for p in PROTOCOL['industry_policies']]
    registered={f'{s}__{p}':registry.register('legacy_repair_attribution',dict(strategy=s,industry_policy=p),ds.version,'fixed',PROTOCOL['version']) for s,p in arms}
    print('[legacy] preparing frozen data',identity,flush=True)
    prepared=prepare(ds,cfg);runs={};stats={}
    for strategy,policy in arms:
        name=f'{strategy}__{policy}';print('[legacy] replay',name,flush=True)
        run=Engine(cfg).run(ds,strategy=strategy,industry_policy=policy,prepared=prepared,
            start=PROTOCOL['start'],end=PROTOCOL['end'],data_mode='free_research')
        if provenance()!=before:raise ValueError('source_or_environment_changed_during_replay')
        folder=persist_run(run,root=root/'runs');runs[name]=run
        stats[name]=account_statistics(run)
        atomic_json(root/(name+'.json'),stats[name]);atomic_json(root/'progress.json',{k:{z:v[z] for z in ('run_id','net_profit','run_health','mean_stock_weight')} for k,v in stats.items()})
        registry.record(registered[name],dict(fold='fixed',status=run['status'],run_id=run['run_id'],net_profit=stats[name]['net_profit']))
        print('[legacy] done',name,stats[name]['net_profit'],run['run_health'],flush=True)
    paired={}
    for policy in PROTOCOL['industry_policies']:
        a=runs[f'legacy_repaired__{policy}']['daily_return']
        for control in ('monthly_equal','vol_matched_equal'):
            b=runs[f'{control}__{policy}']['daily_return'];x=a-b
            paired[f'{policy}__vs_{control}']=dict(conditional=True,annualized_mean_daily_net_excess=float(x.mean()*252),
                ci20=block_bootstrap_ci(x,seed=PROTOCOL['seed'],block=20),ci60=block_bootstrap_ci(x,seed=PROTOCOL['seed'],block=60))
    compact={name:{k:v for k,v in stat.items() if k not in ('score_to_position_pnl','capital_diagnostics','stock_pnl','best_five_daily_returns','data_issues','allocation_issues','eligible_universe_history')} for name,stat in stats.items()}
    summary=dict(run_id=identity,status='research_unvalidated',data_version=ds.version,model_version=PROTOCOL['version'],as_of=PROTOCOL['end'],
        metrics=None,promotion_approved=False,source_unchanged=provenance()==before,data_assessment=assessment,runs=compact,paired=paired,
        universe=protocol['universe'],diagnostic_sample=diagnostic_sample,
        limitations=['historical sample already examined','market-wide coverage has not been certified','unknown historical industries',
                      'public corporate action and delisting gaps retained','daily opening capacity unverified'],artifact_directory=str(root))
    atomic_json(root/'summary.json',summary)
    lines=['# 原策略修复：同数据范围六组回放','',
           f"输入证券数：{len(ds.panels)}；范围：{protocol['universe']['declared_scope']}；诊断样本：{diagnostic_sample}。逐日实际合格数量见各组JSON。", '',
           '条件账面结果；未证明稳定盈利，未批准实盘。行业诊断只关闭行业限制，其余权限保留。',
           '', '| 策略 | 行业政策 | 净利润(元) | 年化 | 回撤 | 平均仓位 | 费用及滑点(元) | 配权失败 | 持仓数据问题 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in stats.items():
        m=s['conditional_metrics'];health=s['run_health']
        lines.append(f"| {name.split('__')[0]} | {s['industry_policy']} | {s['net_profit']:,.2f} | {m['annual_return']:.2%} | {m['max_drawdown']:.2%} | {s['mean_stock_weight']:.2%} | {s['fees']+s['slippage']:,.2f} | {health['allocation_failures']} | {health['data_issue_count']} |")
    lines += ['', '每组JSON含半年收益、恢复时间、现金与订单去向、股票贡献及评分到持仓贡献。配权失败和持仓权益缺口使结果不能视作完整可验证绩效；行业归因在UNKNOWN覆盖下不可识别。',
              '', '所有策略固定参数，没有训练择优。不能把行业限制的解除称为选股Alpha，也不能将六组之一的最高收益直接升级为默认投资策略。',
              '', f'源码归档 source.tar.gz；数据版本 {ds.version}；运行前后源码及环境一致。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return summary
