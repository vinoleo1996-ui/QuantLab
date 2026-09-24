#!/usr/bin/env python3
"""Run common-ledger controls and operational stresses on a contracted dataset."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.config import CONFIG, OUTPUT_DIR
from quantlab.data.contracts import load_dataset
from quantlab.backtest.engine import Engine, prepare, blocked_result
from quantlab.report.runs import persist_run
from quantlab.report.html import build_html_report
from quantlab.report.metrics import performance_table
from quantlab.research import promotion_gate, walk_forward_splits


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/research/manifest.json')
    args=p.parse_args()
    if not Path(args.manifest).exists():
        result=blocked_result(['PIT_dataset_missing','raw_execution_prices_missing','verified_index_benchmark_missing'])
        folder=persist_run(result)
        (folder/'promotion_gate.json').write_text(json.dumps(promotion_gate({}),indent=2))
        print(folder);return
    ds=load_dataset(args.manifest)
    prepared=prepare(ds,CONFIG)
    result=Engine().run(ds,prepared=prepared)
    if result['nav'].empty:
        print(persist_run(result));return
    start=result['nav'].index[0]
    result['baselines']={}
    for name in ('buy_hold','monthly_equal','vol_matched_equal'):
        baseline=Engine().run(ds,strategy=name,start=start,prepared=prepared)
        result['baselines'][name]={k:baseline[k] for k in ('nav','daily_return','status','run_id','fills')}
    # Store index series as a price-index comparator, never mislabel it a tradable total-return account.
    result['index_comparison']={'type':ds.manifest['benchmark_type'],'series':ds.benchmark.copy()}
    stress={}
    cases={'double_cost':dict(cost_mult=2.),'liquidity_minus_70pct':dict(liquidity_mult=.3),
           'five_sessions_locked':dict(blocked_sessions=list(result['nav'].index[20:25])),
           'one_session_execution_delay':dict(execution_delay=1)}
    for name,kwargs in cases.items():
        run=Engine().run(ds,prepared=prepared,**kwargs)
        stress[name]={'status':run['status'],'metrics':performance_table(run['nav'],run['daily_return']),
                      'nonnegative_cash':all(x['cash']>=0 for x in run.get('account',[])),
                      'unfilled_orders':sum(x['status']!='filled' for x in run.get('orders',[]))}
    result['stress']=stress
    result['promotion_gate']=promotion_gate({})
    result['promotion_gate']['reasons'] += ['correlation_and_tail_replays_require_additional_scenarios',
                                          'model_OOS_outcomes_not_supplied']
    result['folds']=[{k:[str(x) for x in v] for k,v in f.items()} for f in walk_forward_splits(ds.calendar)]
    result['run_id'] += '-suite'
    folder=persist_run(result)
    build_html_report(result,out=folder/'report.html')
    print(folder)


if __name__=='__main__':main()
