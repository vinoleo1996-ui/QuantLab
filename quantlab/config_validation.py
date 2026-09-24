"""Explicit config support. Dormant legacy/future knobs cannot silently change a run."""
from dataclasses import fields, is_dataclass
from quantlab.config import Config


ACTIVE = {
    "capital": {"init_cash", "target_positions", "a_share_positions", "rebalance_rule", "rebalance_n_weeks", "min_holding_period_days"},
    "universe": {"a_exclude_st", "a_min_mktcap_yi", "a_min_avg_amount_wan", "a_min_listing_days"},
    "data": {"cache_dir", "a_daily_adjust", "reset_cache", "history_years", "rate_limiter_sec"},
    "portfolio": {"target_vol", "vol_est_window", "max_sector_weight", "lw_shrinkage"},
    "cost": {"a_slippage", "min_trade_value"},
    "risk": {"stop_loss_single", "drawdown_alert", "drawdown_reduce", "drawdown_exit", "drawdown_halt", "vol_reduction_factor"},
    "research": {"model_version", "horizon", "train_months", "validation_months", "test_months", "max_trials",
                 "max_stock_weight", "max_etf_weight", "max_etfs", "max_cluster_weight", "downside_corr", "risk_aversion", "crowding_cost", "liquidity_cost", "cash_buffer", "participation", "entry_percentile",
                 "exit_percentile", "exit_reviews", "recovery_buffer", "recovery_sessions", "forward_months", "universe_mode"},
}


def config_fields(cfg):
    for group in fields(cfg):
        obj = getattr(cfg, group.name)
        if is_dataclass(obj):
            for f in fields(obj): yield group.name, f.name, getattr(obj, f.name)
        else: yield "", group.name, obj


def status(group, name):
    if group == "" and name in ("seed", "start_date", "benchmark_a"): return "active"
    if group == "factors": return "archived_factor_module_only"
    if name in ACTIVE.get(group, set()): return "active"
    return "reserved_reject_overrides"


def validate_config(cfg):
    default = {(g,n): v for g,n,v in config_fields(Config())}
    for group, name, value in config_fields(cfg):
        if status(group,name)!="active" and value != default[group,name]:
            raise ValueError(f"inactive_config_field:{group}.{name}; not implemented for this engine")
    if cfg.capital.rebalance_rule not in ("monthly", "weekly", "every_n_weeks"): raise ValueError("invalid_rebalance_rule")
    if cfg.research.model_version != "opportunity-ledger-v3": raise ValueError("unpromoted_model_version")
    if cfg.research.universe_mode not in ("pit_cap", "pit_liquidity", "snapshot_research"): raise ValueError("invalid_universe_mode")
    if cfg.capital.init_cash<=0 or cfg.capital.a_share_positions<1: raise ValueError("invalid_capital")
    if not 0<=cfg.research.cash_buffer<1 or not 0<cfg.research.participation<=1: raise ValueError("invalid_execution_limits")
    levels = [cfg.risk.drawdown_alert,cfg.risk.drawdown_reduce,cfg.risk.drawdown_exit,cfg.risk.drawdown_halt]
    if levels != sorted(set(levels)) or not 0<levels[0]<levels[-1]<1: raise ValueError("invalid_drawdown_levels")
    if cfg.factors.ic_window<24 or cfg.portfolio.vol_est_window<40: raise ValueError("insufficient_estimation_window")
    if cfg.research.recovery_sessions<1 or cfg.research.exit_reviews<1: raise ValueError("invalid_hysteresis")
    if min(cfg.research.risk_aversion,cfg.research.crowding_cost,cfg.research.liquidity_cost,cfg.factors.expert_prior_strength)<0 or cfg.factors.expert_temperature_floor<=0:
        raise ValueError('invalid_nonnegative_penalty')
    return True
