"""Daily cash/share replay shared by strategies and baselines.

Legacy price panels without execution/PIT contracts are rejected. Diagnostic proxy
runs require an explicit flag and are never eligible for promotion or live signals.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.data.contracts import (MarketDataset, InstrumentRules, asof_records, decision_time,
                                    fingerprint, timestamp)
from quantlab.data.market import panels_for_research, universe_at, exposure_matrix
from quantlab.backtest.ledger import Ledger, trade_permission
from quantlab.strategy.crowding import exit_risk
from quantlab.mandate import Mandate
from quantlab.strategy.decision import review_dates, FactorDecisionService
from quantlab.strategy.legacy import legacy_config, allocate_legacy
from quantlab.strategy.allocation import allocate
from quantlab.strategy.risk import RiskManager, Regime


def _limit_ratio(code):
    if str(code).startswith(("300", "301", "688", "689")): return .20
    if str(code).startswith(("43", "83", "87", "92")): return .30
    return .10


def limit_masks(open_, prev_close, amount=None, volume=None):
    """Legacy diagnostic helper only; opening replay uses dated market status/rules."""
    gap = open_ / prev_close - 1
    known = open_.gt(0) & prev_close.gt(0)
    if amount is not None:
        trading = amount.gt(0)
        if volume is not None: trading |= amount.isna() & volume.gt(0)
        known &= trading
    buys, sells = known.copy(), known.copy()
    for c in gap:
        buys[c] &= gap[c] < _limit_ratio(c) - .005
        sells[c] &= gap[c] > -_limit_ratio(c) + .005
    return buys, sells


def blocked_result(reasons, market="a"):
    return {"nav": pd.Series(dtype=float), "daily_return": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float), "trades": [], "weight_history": [],
            "ic_matrix": pd.DataFrame(), "factor_panels": {}, "codes": [], "market": market,
            "status": "blocked_data", "reasons": reasons, "metrics": None,
            "promotion_approved": False, "model_version": "opportunity-unconfigured", "data_version": None,
            "as_of": None, "run_id": fingerprint(reasons)[:20], "orders": [], "fills": []}


def prepare(dataset, cfg):
    close, amount, turnover = panels_for_research(dataset)
    # Capture raw and processed inputs in every persisted run, including the exact benchmark.
    records, rules, exposures, eligible, close_records = {}, {}, {}, {}, {}
    for d in close.index:
        morning = asof_records(dataset.instruments, timestamp(d) + pd.Timedelta(hours=9, minutes=25))
        records[d] = morning
        rules[d] = {}
        for c, row in morning.iterrows():
            try: rules[d][c] = InstrumentRules.from_record(row)
            except ValueError: pass
        exposures[d] = exposure_matrix(dataset, d, morning, list(morning.index),
                                       as_of=timestamp(d) + pd.Timedelta(hours=9, minutes=25))
        eligible[d], close_records[d] = universe_at(dataset, d, amount, cfg)
    return {"close": close, "amount": amount, "turnover": turnover,
            "version": dataset.version, "preparation_config": fingerprint(asdict(cfg)), "records": records, "rules": rules,
            "exposures": exposures, "eligible": eligible, "close_records": close_records,
            "bars": {d: {c: row.to_dict() for c, panel in dataset.panels.items()
                         if d in panel.index for _, row in panel.loc[[d]].iterrows()} for d in close.index}}



class Engine:
    def __init__(self, cfg=None, market="a"):
        self.cfg, self.market = cfg or CONFIG, market

    def run(self, panels, meta=None, bench=None, **kwargs):
        if isinstance(panels, MarketDataset): return self.run_dataset(panels, **kwargs)
        return blocked_result(["versioned_dataset_manifest_required", "raw_prices_and_corporate_actions_unverified",
                               "historical_universe_and_exchange_calendar_unverified"], self.market)

    def run_dataset(self, dataset, *, strategy="legacy_repaired",
                    diagnostic=False, risk_mode="daily", prepared=None, target_schedule=None,
                    start=None, end=None, cost_mult=1.0, liquidity_mult=1.0, blocked_sessions=(),
                    execution_delay=0, forecasts=None, review_override=None, crowding=True, initial_state=None, data_mode='strict', industry_policy='enforced'):
        if self.market != "a": return blocked_result(["us_execution_not_validated"], self.market)
        if risk_mode not in ("daily", "legacy_monthly"): raise ValueError("invalid_risk_policy")
        if data_mode not in ('strict','free_research'):raise ValueError('unknown_data_mode')
        if industry_policy not in ('enforced','omit_for_diagnostic'):raise ValueError('unknown_industry_policy')
        if industry_policy=='omit_for_diagnostic' and (data_mode!='free_research' or diagnostic):
            raise ValueError('industry_omission_requires_scoped_free_research')
        enforce_industry = industry_policy=='enforced'
        selected_cfg = legacy_config(self.cfg) if strategy=='legacy_repaired' else self.cfg
        assessment=None
        errors = dataset.validate(data_mode)
        if data_mode=='free_research':
            from quantlab.data.qualification import research_assessment
            assessment=research_assessment(dataset,require_industry=strategy=='opportunity' and selected_cfg.research.feature_profile=='price_volume_industry')
            if not enforce_industry:
                assessment['warnings']=[w for w in assessment['warnings'] if w!='industry_disabled_unknown_exposure_capped']+['industry_limits_explicitly_omitted_for_diagnostic']
            errors=assessment['reasons']
        if errors and not diagnostic: return blocked_result(errors)
        mandate = None if diagnostic else Mandate.from_config(selected_cfg)
        if mandate and not dataset.instruments.asset_type.isin(mandate.allowed_assets).all():
            return blocked_result(['asset_not_in_mandate'])
        cfg = selected_cfg if diagnostic else mandate.config()
        from quantlab.config_validation import validate_config
        validate_config(cfg)
        if cfg.research.enable_announcements:
            return blocked_result(["announcement_module_requires_PIT_and_incremental_evidence"])
        if strategy not in ("buy_hold", "monthly_equal", "vol_matched_equal", "schedule", "opportunity", "legacy_repaired"):
            raise ValueError("unknown_or_unpromoted_strategy")
        if strategy == "opportunity" and forecasts is None: return blocked_result(["frozen_forecasts_required"])
        if strategy!='opportunity' and forecasts:raise ValueError('forecasts_require_explicit_opportunity_strategy')
        if strategy == "schedule" and target_schedule is None: raise ValueError("target schedule required")
        p = prepared or prepare(dataset, cfg)
        if p["version"] != dataset.version or p.get("preparation_config") != fingerprint(asdict(cfg)):
            raise ValueError("prepared_context_version_mismatch")
        close, amount = p["close"], p["amount"]
        scorer = None
        if strategy=='legacy_repaired':
            scorer = p.get('legacy_scorer')
            if scorer is None:
                scorer = FactorDecisionService(close, amount, p['turnover'], dataset.calendar, cfg,dataset.manifest.get('data_gaps',[]))
                p['legacy_scorer'] = scorer
        constrained_strategy = strategy in ('opportunity','legacy_repaired','monthly_equal','vol_matched_equal')
        start = pd.Timestamp(start or cfg.start_date)
        end = pd.Timestamp(end or close.index[-1])
        dates = close.index[(close.index >= start) & (close.index <= end)]
        if dates.empty: return blocked_result(["empty_replay_window"])
        rule = "monthly" if strategy in ("buy_hold", "monthly_equal", "vol_matched_equal") else cfg.capital.rebalance_rule
        reviews = set(review_dates(dataset.calendar, rule, cfg.capital.rebalance_n_weeks))
        if review_override is not None: reviews = set(review_override)
        if "review_override" in p: reviews &= set(p["review_override"])
        if strategy in ("buy_hold", "monthly_equal", "vol_matched_equal") and len(dates): reviews.add(dates[0])
        account = Ledger(cfg.capital.init_cash, cfg.cost.a_slippage, cfg.research.cash_buffer,
                         cfg.research.participation, cost_mult, cfg.cost.min_trade_value,allow_open_assumption=data_mode=='free_research')
        risk = RiskManager(cfg.risk, cfg.research, cfg.portfolio.target_vol) if risk_mode == "daily" else LegacyMonthlyRisk(cfg)
        regime = Regime(dataset.benchmark.close) if not dataset.benchmark.empty else None
        states, decisions, traces, allocation_issues = [], [], [], []
        last_risk = dict(gross_scale=1., vol_scale=1., level=0, allow_new_stock=True, reasons=[], drawdown=0.)
        invested = False
        delayed = []
        outside_count = {}
        if initial_state is not None:
            if initial_state['mandate_version']!=(mandate.version if mandate else None):raise ValueError('forward_mandate_changed')
            account=initial_state['ledger'];risk=initial_state['risk'];last_risk=initial_state['last_risk']
            delayed=initial_state['delayed'];outside_count=initial_state['outside_count'];invested=initial_state['invested']
            if account.snapshots and dates[0]<=account.snapshots[-1]['date']:raise ValueError('forward_session_not_new')
        config_dump = asdict(cfg)
        run_id = fingerprint({"data": p["version"], "config": config_dump, "strategy": strategy,
                              "start": str(start), "end": str(end), "risk": risk_mode,"data_mode":data_mode,
                              "cost": cost_mult, "liquidity": liquidity_mult, "blocked": list(blocked_sessions),
                              "delay": execution_delay, "schedule": target_schedule,
                              "forecasts": {str(d):dict(model=f.model_version,training=f.training_cutoff,mu=f.expected_returns.to_dict()) for d,f in (forecasts or {}).items()},
                              "crowding":crowding,"reviews":review_override,"industry_policy":industry_policy,
                              "initial_account": None if initial_state is None else dict(cash=account.cash,lots={c:[asdict(l) for l in ls] for c,ls in account.lots.items()},risk=last_risk,orders=account.export()['orders'])})[:24]
        for d in dates:
            session = int(dataset.calendar.get_loc(d))
            for gap in dataset.manifest.get('data_gaps',[]):
                if pd.Timestamp(gap['date'])==d and account.quantity(gap['instrument_id']):
                    account.data_issues.append(dict(gap,held_shares=account.quantity(gap['instrument_id'])))
            bars = p["bars"].get(d, {})
            if d in blocked_sessions:
                bars = {c: dict(b, open_status="halted") for c, b in bars.items()}
            morning = p["records"][d]
            rules = p["rules"][d].copy()
            account.corporate_actions(d, session, dataset.actions, dataset.calendar)
            adv = amount.loc[amount.index < d].tail(20).mean().to_dict()
            adv = {c: v * liquidity_mult for c, v in adv.items()}
            exp = p["exposures"][d]
            if not enforce_industry: exp = exp.iloc[:0]
            execution_risk=None
            prior_dates=close.index[close.index<d]
            risk_codes=sorted(set(account.quantities())|{o.instrument_id for o in account.orders if o.status in ('pending','partial')})
            if constrained_strategy and crowding and len(prior_dates) and risk_codes:
                execution_risk=exit_risk(close,p['turnover'],amount,dataset.benchmark.close,prior_dates[-1],risk_codes,
                                         cfg.research.downside_corr,account.nav(),include_penalties=False)
            def can_buy(c, q, price, fees, ledger):
                if c not in morning.index or not bool(morning.loc[c].get("active", False)) or (cfg.universe.a_exclude_st and bool(morning.loc[c].get("is_st", True))): return False
                nav = ledger.nav() - fees
                vals = {k: ledger.quantity(k) * ledger.marks[k] for k in ledger.quantities()}
                vals[c] = vals.get(c, 0.) + q * price
                cap = cfg.research.max_etf_weight if rules[c].asset_type == "equity_etf" else cfg.research.max_stock_weight
                if vals[c] / nav > cap + 1e-8: return False
                if sum(vals.values()) / nav > min(last_risk["gross_scale"], 1-cfg.research.cash_buffer) + 1e-8: return False
                count_limit = len(dataset.panels) if strategy in ("buy_hold", "monthly_equal", "vol_matched_equal") else cfg.capital.a_share_positions
                if len([k for k in vals if vals[k]>0 and k in rules and rules[k].asset_type == "stock"]) > count_limit: return False
                if strategy not in ("buy_hold", "monthly_equal", "vol_matched_equal") and len(vals) > cfg.capital.target_positions: return False
                if len([k for k in vals if vals[k]>0 and k in rules and rules[k].asset_type == "equity_etf"]) > cfg.research.max_etfs: return False
                if execution_risk is not None:
                    cm=execution_risk.matrix(list(vals))
                    if (cm @ (pd.Series(vals)/nav)>cfg.research.max_cluster_weight+1e-8).any():return False
                weights = pd.Series(vals).reindex(exp.columns, fill_value=0) / nav
                return bool((exp @ weights <= cfg.portfolio.max_sector_weight + 1e-8).all())
            # Delayed intents preserve original signal dates, and cannot bypass a later halt.
            for intent in list(delayed):
                if intent[0] <= session:
                    if last_risk["gross_scale"] > 0:
                        account.queue_targets(intent[1], intent[2], intent[3], rules)
                    delayed.remove(intent)
            account.execute(d, session, bars, rules, adv, can_buy=can_buy)
            prices = {c: b.get("close", np.nan) for c, b in bars.items()
                      if pd.notna(b.get("available_at")) and timestamp(b["available_at"]) <= decision_time(d)
                      and b.get("quality_status") == "ok"}
            state = account.close(d, session, prices)
            records = p['close_records'][d]
            exp = exposure_matrix(dataset, d, records, list(records.index))
            observed_exposure = exp @ pd.Series(state['weights'], dtype=float).reindex(exp.columns,fill_value=0)
            if not enforce_industry: exp = exp.iloc[:0]
            old_level = last_risk["level"]
            if strategy not in ("buy_hold", "monthly_equal") and (risk_mode == "daily" or d in reviews):
                last_risk = risk.evaluate(state["nav"], regime.state(d)["multiplier"] if regime is not None else 1., date=d)
            exposure_now = exp @ pd.Series(state["weights"], dtype=float).reindex(exp.columns,fill_value=0)
            breaches = exposure_now[exposure_now > cfg.portfolio.max_sector_weight + 1e-6].to_dict()
            states.append(dict(date=d, **last_risk, industry_policy=industry_policy, actual_sector_exposure=observed_exposure.to_dict(),
                               unresolved_sector_breaches=breaches))
            eligible = p["eligible"][d]
            # Stop decisions are close-time intents, never fabricated fills.
            stops = [c for c in account.quantities() if account.cost_basis(c)>0 and
                     account.marks[c] / account.cost_basis(c) - 1 < cfg.risk.stop_loss_single]
            status_exits = {c: ('inactive_security' if not bool(records.loc[c].active) else 'st_security')
                for c in account.quantities() if c in records.index and
                (not bool(records.loc[c].active) or (cfg.universe.a_exclude_st and bool(records.loc[c].is_st)))}
            stops = sorted(set(stops) | set(status_exits))
            if strategy in ("buy_hold", "monthly_equal") or (risk_mode != "daily" and d not in reviews):
                stops = []; status_exits = {}
            states[-1]['eligibility_exit_reasons'] = status_exits
            gross_breach = (risk_mode == "daily" or d in reviews) and last_risk["gross_scale"] < 1 and sum(state["weights"].values()) > last_risk["gross_scale"] + 1e-4
            constraint_target=dict(state['weights'])
            if constrained_strategy and constraint_target:
                for c,w in list(constraint_target.items()):
                    cap=cfg.research.max_etf_weight if rules[c].asset_type=='equity_etf' else cfg.research.max_stock_weight
                    if w>cap+1e-6:constraint_target[c]=cap*.98
                matrices=[(exp,cfg.portfolio.max_sector_weight)]
                if crowding:
                    cr=exit_risk(close,p['turnover'],amount,dataset.benchmark.close,d,list(constraint_target),cfg.research.downside_corr,account.nav(),include_penalties=False)
                    matrices.append((cr.matrix(list(constraint_target)),cfg.research.max_cluster_weight))
                for matrix,cap in matrices:
                    for _,row in matrix.iterrows():
                        exposure=sum(row.get(c,0)*w for c,w in constraint_target.items())
                        if exposure>cap+1e-6:
                            for c in constraint_target:
                                if row.get(c,0)>0:constraint_target[c]*=cap*.98/exposure
            constraint_breach=any(constraint_target[c]<w-1e-8 for c,w in state['weights'].items())
            states[-1]['constraint_exit_required']=constraint_breach
            risk_exit = last_risk["level"] > old_level or bool(stops) or gross_breach or constraint_breach
            target = None
            reason = ""
            trace = {}
            if risk_exit:
                weights = account.weights()
                scale = min(1., last_risk["gross_scale"] / max(sum(weights.values()), 1e-12))
                if last_risk["level"] == 1: scale = min(scale, cfg.risk.vol_reduction_factor)
                target = {c: min(w * scale,constraint_target.get(c,w)) if c not in stops else 0. for c, w in weights.items()}
                reason = "risk_exit"
                account.cancel_buys(reason); delayed.clear()
            elif strategy == "schedule" and d in target_schedule:
                target, reason = target_schedule[d], "scheduled_target"
            elif strategy == 'legacy_repaired' and d in reviews:
                held = account.quantities()
                stocks = [c for c in eligible if c in rules and rules[c].asset_type=='stock']
                score = scorer.score(d, stocks)
                trace = dict(score.trace, reasons=score.reasons)
                if score.status!='ok':
                    account.cancel_buys('invalid_legacy_score')
                    traces.append(dict(date=str(d.date()),status=score.status,**trace));continue
                locked = {}
                for c in held:
                    permission = c in rules and trade_permission(bars.get(c),rules[c],'sell',d,allow_open_assumption=data_mode=='free_research')[0]
                    if (not permission or account.available(c,session)<held[c] or
                        account.holding_age(c,session)<cfg.capital.min_holding_period_days):locked[c]=state['weights'][c]
                pending_exits = {o.instrument_id for o in account.orders if o.reason=='risk_exit' and o.status in ('pending','partial')}
                desired = list(sorted(locked))
                for c in score.scores.index:
                    if c in pending_exits or c in desired:continue
                    if not last_risk.get('allow_new_stock',True) and c not in held:continue
                    if len(desired)>=cfg.capital.a_share_positions:break
                    desired.append(c)
                codes = sorted(set(desired)|set(held))
                rs = exit_risk(close,p['turnover'],amount,dataset.benchmark.close,d,codes,cfg.research.downside_corr,account.nav(),include_penalties=False) if crowding and codes else None
                exposures = exp.reindex(columns=codes,fill_value=0.)
                returns = close.loc[:d,codes].pct_change(fill_method=None).tail(cfg.portfolio.vol_est_window)
                x,status,capital = allocate_legacy(returns,cfg,selected=desired,previous=state['weights'],locked=locked,
                    exposures=exposures,risk_snapshot=rs,gross=last_risk['gross_scale'],vol_scale=last_risk['vol_scale'])
                trace.update(capital=capital,selected=desired,locked=locked,groups=rs.groups if rs else {})
                if status!='ok':
                    allocation_issues.append(dict(date=str(d.date()),reason=status,capital=capital))
                    traces.append(dict(date=str(d.date()),status='allocation_failed',**trace))
                    account.cancel_buys(status);continue
                target,reason = x.to_dict(),'legacy_repaired'
            elif strategy == 'opportunity' and d in reviews:
                forecast=(forecasts or {}).get(d)
                if forecast is None or forecast.status!='ok':
                    account.cancel_buys('invalid_forecast')
                    traces.append(dict(date=str(d),status='invalid',reasons=['forecast_missing_or_invalid']))
                    continue
                forecast.validate(d)
                if forecast.trace.get('feature_profile',cfg.research.feature_profile)!=cfg.research.feature_profile:
                    raise ValueError('forecast_feature_profile_mismatch')
                if forecast.horizon!=cfg.research.horizon:raise ValueError('forecast_horizon_mismatch')
                mu=forecast.expected_returns.reindex(eligible).dropna()
                ranks=mu.rank(pct=True)
                held=account.quantities()
                for c in held:outside_count[c]=outside_count.get(c,0)+1 if ranks.get(c,0)<cfg.research.exit_percentile else 0
                kept=[c for c in held if account.holding_age(c,session)<cfg.capital.min_holding_period_days or outside_count.get(c,0)<cfg.research.exit_reviews]
                new=list(ranks[ranks>=cfg.research.entry_percentile].sort_values(ascending=False,kind='stable').index)
                desired=[];stock_count=etf_count=0
                for c in dict.fromkeys(kept+new):
                    if c not in rules:continue
                    if rules[c].asset_type=='stock':
                        if stock_count>=cfg.capital.a_share_positions:continue
                        stock_count+=1
                    else:
                        if etf_count>=cfg.research.max_etfs:continue
                        etf_count+=1
                    desired.append(c)
                codes=sorted(set(desired)|set(held))
                if not codes:continue
                previous=pd.Series(account.weights(),dtype=float)
                expected=mu.reindex(codes,fill_value=-1.)
                allowed=set(desired) if last_risk.get('allow_new_stock',True) else {c for c in desired if rules[c].asset_type=='equity_etf'}
                locked={}
                for c in held:
                    permission,_=trade_permission(bars.get(c),rules[c],'sell',d,allow_open_assumption=data_mode=='free_research')
                    if (not permission or account.available(c,session)<held[c] or
                        account.holding_age(c,session)<cfg.capital.min_holding_period_days):locked[c]=previous[c]
                rs=exit_risk(close,p['turnover'],amount,dataset.benchmark.close,d,codes,cfg.research.downside_corr,account.nav()) if crowding else None
                returns=close.loc[:d,codes].pct_change(fill_method=None).tail(cfg.portfolio.vol_est_window)
                x,status=allocate(returns,records,exp.reindex(columns=codes,fill_value=0.),cfg,
                                  last_risk['gross_scale'],last_risk['vol_scale'],expected,previous,risk_snapshot=rs,
                                  locked=locked,nav=account.nav(),date=d,cost_mult=cost_mult,allowed_new=allowed,
                                  industry_policy=('disabled_diagnostic' if not enforce_industry else
                                      'unknown_capped' if cfg.research.feature_profile=='price_volume' else 'verified'))
                if status!='ok':
                    allocation_issues.append(dict(date=str(d),reason=status));account.cancel_buys(status);continue
                target=x.to_dict();reason='opportunity'
                trace=dict(forecast.trace,predictions=mu.to_dict(),training_cutoff=forecast.training_cutoff,
                           forecast_model_version=forecast.model_version,groups=rs.groups if rs else {},
                           risk_proxy_reasons=rs.reasons if rs else [],locked=locked)
            elif strategy in ("buy_hold", "monthly_equal", "vol_matched_equal") and d in reviews and not (strategy == "buy_hold" and invested):
                selected=[c for c in eligible if c in rules and c in close]
                if not selected:continue
                exposures=exp.reindex(columns=selected,fill_value=0.)
                x=pd.Series({c:min((1-cfg.research.cash_buffer)/len(selected),
                    cfg.research.max_etf_weight if rules[c].asset_type=='equity_etf' else cfg.research.max_stock_weight) for c in selected})
                sector=exposures@x
                if len(sector) and sector.max()>cfg.portfolio.max_sector_weight:x*=cfg.portfolio.max_sector_weight/sector.max()
                if constrained_strategy and crowding:
                    rs=exit_risk(close,p['turnover'],amount,dataset.benchmark.close,d,selected,cfg.research.downside_corr,account.nav(),include_penalties=False)
                    cluster=rs.matrix(selected)@x
                    if len(cluster) and cluster.max()>cfg.research.max_cluster_weight:x*=cfg.research.max_cluster_weight/cluster.max()
                if strategy=='vol_matched_equal':
                    returns=close.loc[:d,selected].pct_change(fill_method=None).tail(cfg.portfolio.vol_est_window).dropna()
                    if len(returns)<40:continue
                    vol=float((returns@x).std()*np.sqrt(252))
                    x*=min(1.,cfg.portfolio.target_vol*last_risk['vol_scale']/max(vol,1e-12))
                x*=min(1.,last_risk['gross_scale']/max(x.sum(),1e-12))
                if not last_risk.get('allow_new_stock',True):
                    x=pd.Series({c:min(w,account.weights().get(c,0.)) for c,w in x.items()})
                target,reason=x.to_dict(),strategy
            if target is not None:
                next_dates = dataset.calendar[dataset.calendar > d]
                exec_date = next_dates[execution_delay] if len(next_dates)>execution_delay else None
                if execution_delay:
                    delayed.append((session+1+execution_delay, target, d, reason))
                else: account.queue_targets(target, d, reason, rules)
                invested = True
                new_orders = [o for o in account.orders if o.signal_date==str(d.date())]
                for o in new_orders:o.industry_policy=industry_policy
                trace.update(industry_policy=industry_policy,capital=dict(trace.get('capital',{}),
                    target_gross=sum(target.values()),cash_before_orders=account.cash,nav=account.nav(),
                    submitted_notional=sum(o.quantity*account.marks[o.instrument_id] for o in new_orders),
                    order_ids=[o.order_id for o in new_orders]))
                decisions.append({"signal_date": str(d.date()), "exec_date": str(exec_date.date()) if exec_date is not None else None,
                                  "weights": target, "factor_weights": trace.get("factor_weights", {}),
                                  "experts": trace.get("expert_weights", {}), "risk": dict(last_risk),
                                  "regime": regime.state(d) if regime is not None else {}, "reason": reason})
                traces.append({"date": str(d.date()), "status": "ok", **trace, "target_weights": target,
                               "raw_close": {c: prices.get(c) for c in target}})
        nav = pd.Series({s["date"]: s["nav"] / cfg.capital.init_cash for s in account.snapshots}, name="nav")
        ret = nav.pct_change(fill_method=None)
        if len(ret): ret.iloc[0] = nav.iloc[0]-1
        dd = nav / nav.cummax().clip(lower=1.) - 1
        status = "diagnostic_proxy" if diagnostic else "research_unvalidated"
        if account.data_issues: status = "unverified_data_gaps"
        model_versions=sorted({f.model_version for f in (forecasts or {}).values()})
        reported_model=(model_versions[0] if len(model_versions)==1 else 'schedule-'+fingerprint(model_versions)[:20]) if strategy=='opportunity' else "baseline-"+strategy
        if scorer is not None:reported_model='legacy-repaired-'+fingerprint(dict(factors=asdict(cfg.factors),portfolio=asdict(cfg.portfolio)))[:16]
        for o in account.orders:o.industry_policy=industry_policy
        for f in account.fills:f['industry_policy']=industry_policy
        result = {"nav": nav, "daily_return": ret, "drawdown": dd, "trades": decisions,
                  "data_mode":data_mode,"data_assessment":assessment,"industry_policy":industry_policy,
                  "research_exceptions":[] if enforce_industry else ['industry_limit_omitted_for_diagnostic'],
                  "weight_history": [pd.Series(x["weights"], name=x["date"], dtype=float) for x in account.snapshots],
                  "factor_panels": {}, "ic_matrix": scorer.ic if scorer is not None else pd.DataFrame(), "codes": list(close), "market": self.market,
                  "run_id": run_id, "data_version": p["version"], "model_version": reported_model,
                  "as_of": str(end), "status": status, "reasons": errors, "promotion_approved": False,
                  "mandate_version": mandate.version if mandate else "diagnostic_unrestricted",
                  "config": config_dump, "benchmark": dataset.benchmark.copy(), "benchmark_manifest": dataset.manifest,
                  "risk_history": states, "signal_trace": traces, "allocation_issues": allocation_issues,
                  "pending_intents": delayed, "strategy": strategy, "risk_mode": risk_mode,
                  "input_panels": dataset.panels, "input_instruments": dataset.instruments,
                  "input_actions": dataset.actions, "calendar": dataset.calendar, **account.export()}
        result['_runtime']=dict(ledger=account,risk=risk,last_risk=last_risk,delayed=delayed,outside_count=outside_count,
                                invested=invested,mandate_version=mandate.version if mandate else None)
        forecast_failures=sum(t.get('status') in ('invalid','warmup') for t in traces)
        result['run_health']=dict(allocation_failures=len(allocation_issues),decision_count=len(decisions),forecast_failures=forecast_failures,
            data_issue_count=len(account.data_issues),mean_cash_weight=float(np.mean([a['cash']/a['nav'] for a in account.snapshots])),
            research_interpretable=not allocation_issues and not account.data_issues and not forecast_failures)
        if assessment is not None and not assessment['accounting_verified']:
            result['run_health']['research_interpretable']=False
            result['reasons']=result['reasons']+['accounting_coverage_unverified']
        from quantlab.report.capital import capital_report
        result['capital_diagnostics'] = capital_report(result)
        return result


def close_ret_only(c2c, top, t, window):
    past = c2c.loc[:t].tail(window)[top].dropna(how="all")
    return past if len(past)>=40 else None


class LegacyMonthlyRisk:
    """Frozen old control policy for attribution only, not a live risk manager."""
    def __init__(self, cfg):
        self.peak = 0.
        self.cfg = cfg
    def evaluate(self, value, regime_mult=1., date=None):
        self.peak = max(self.peak, value)
        dd = value/self.peak-1
        gross = .3 if dd < -.22 or regime_mult <= .55 else .7 if dd < -.15 else 1.
        vol = .75 if dd < -.10 else 1.
        level = 3 if gross <= .3 else 2 if gross < 1 else 1 if vol < 1 else 0
        return dict(drawdown=dd, gross_scale=gross, vol_scale=vol, level=level,
                    allow_new_stock=True, reasons=["frozen_legacy_monthly_policy"])
