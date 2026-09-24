"""Read the same dated decision output used by replay; no parallel scoring implementation."""
from pathlib import Path
import json
from quantlab.config import OUTPUT_DIR


def daily_expert_pulse(limit=100, manifest=None, strategy="legacy_repaired", forecasts=None, data_mode='strict',feature_profile=None,industry_policy='enforced'):
    from quantlab.cli import run_backtest, signal_payload
    result = run_backtest(save=False, manifest=manifest, strategy=strategy, forecasts=forecasts,data_mode=data_mode,feature_profile=feature_profile,industry_policy=industry_policy)
    from quantlab.report.runs import persist_run,atomic_json
    folder=persist_run(result)
    signal = signal_payload(result)
    trace = result.get("signal_trace", [])
    last = trace[-1] if trace else {}
    scores=last.get('predictions',last.get('scores',{}))
    scores=dict(sorted(scores.items(),key=lambda x:x[1],reverse=True)[:max(0,limit)])
    payload=dict(signal, date=result.get("as_of"), n_candidates=len(result.get("codes", [])),
                score_status=last.get("status", "blocked_data"), scores=scores,display_limit=limit,
                coverage={"note": "共用回测决策函数；非券商持仓", "cache_daily": len(result.get("codes", []))},
                expert_weights=last.get("expert_weights", {}), experts={}, new_names=[], held_dropped_out=[])
    atomic_json(folder/('monitor_'+str(limit)+'.json'),payload)
    return payload
