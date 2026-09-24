"""Read the same dated decision output used by replay; no parallel scoring implementation."""
from pathlib import Path
import json
from quantlab.config import OUTPUT_DIR


def daily_expert_pulse(limit=100, manifest=None):
    from quantlab.cli import run_backtest, signal_payload
    result = run_backtest(save=False, manifest=manifest)
    signal = signal_payload(result)
    trace = result.get("signal_trace", [])
    last = trace[-1] if trace else {}
    return dict(signal, date=result.get("as_of"), n_candidates=len(result.get("codes", [])),
                score_status=last.get("status", "blocked_data"), scores=last.get("scores", {}),
                coverage={"note": "共用回测决策函数；非券商持仓", "cache_daily": len(result.get("codes", []))},
                expert_weights=last.get("expert_weights", {}), experts={}, new_names=[], held_dropped_out=[])
