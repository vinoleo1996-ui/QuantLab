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
from quantlab.strategy.risk_policy import PolicyController


def _known_bool(value, default=False):
    return default if pd.isna(value) else bool(value)


def _st_state(record, bar, default=False):
    value=(bar or {}).get('is_st_at_open') if record.get('state_basis')=='daily_observation_required' else record.get('is_st')
    return _known_bool(value,default)


def _group_cap(budget, name, kind, soft=False):
    unknown = 'UNKNOWN' in str(name).upper()
    if unknown:
        hard = getattr(budget, 'unknown_'+kind+'_weight')
        return max(0., hard-.02) if soft and budget.policy_id!='P0' else hard
    return getattr(budget, ('soft_' if soft else 'max_')+kind+'_weight')


def _portfolio_vol(close, date, weights, cfg):
    from sklearn.covariance import LedoitWolf
    names = [c for c,w in weights.items() if w>0 and c in close]
    if not names: return 0.
    r = close.loc[:date,names].pct_change(fill_method=None).tail(cfg.portfolio.vol_est_window).dropna()
    if len(r)<40: return None
    cov = LedoitWolf().fit(r).covariance_ if cfg.portfolio.lw_shrinkage else r.cov().to_numpy()
    w = np.array([weights[c] for c in names])
    return float(np.sqrt(max(0., w@cov@w)*252))


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


class _DayView:
    """Bounded dated execution data; never materialize millions of bar dicts."""
    def __init__(self, owner, key): self.owner,self.key=owner,key
    def __getitem__(self, date):
        value=self.owner.day(date)
        if self.key=='eligible' and self.key not in value:
            if date not in self.owner.eligibility_cache:
                eligible,records=universe_at(self.owner.dataset,date,self.owner.amount,self.owner.cfg)
                self.owner.eligibility_cache[date]=eligible
                value['close_records']=records
            value['eligible']=self.owner.eligibility_cache[date]
        if self.key=='exposures' and self.key not in value:
            value['exposures']=exposure_matrix(self.owner.dataset,date,value['records'],list(value['records'].index),
                                              as_of=timestamp(date)+pd.Timedelta(hours=9,minutes=25))
        return value[self.key]
    def get(self, date, default=None):
        return self[date] if date in self.owner.dates else default


class _PreparedDays:
    def __init__(self,dataset,cfg,amount,dates):
        self.dataset,self.cfg,self.amount,self.dates=dataset,cfg,amount,dates
        self.cache={}
        self.eligibility_cache={}
        self.rule_cache={}
        self.static_rules=None
        if not dataset.instruments.instrument_id.duplicated().any():
            self.static_rules={}
            for _,row in dataset.instruments.iterrows():
                try:self.static_rules[row.instrument_id]=InstrumentRules.from_record(row)
                except ValueError:pass
    def day(self,d):
        if d not in self.cache:
            ds=self.dataset
            morning=asof_records(ds.instruments,timestamp(d)+pd.Timedelta(hours=9,minutes=25)).copy()
            closing=asof_records(ds.instruments,decision_time(d)).copy()
            if self.static_rules is not None:
                rules={c:self.static_rules[c] for c in morning.index if c in self.static_rules}
            else:
                rules={}
                for c,row in morning.iterrows():
                    key=(c,str(row.get('effective_at')),str(row.get('available_at')),str(row.get('revision')))
                    if key not in self.rule_cache:
                        try: self.rule_cache[key]=InstrumentRules.from_record(row)
                        except ValueError: self.rule_cache[key]=None
                    if self.rule_cache[key] is not None: rules[c]=self.rule_cache[key]
            self.cache[d]=dict(records=morning,rules=rules,close_records=closing)
            if len(self.cache)>2: self.cache.pop(next(iter(self.cache)))
        return self.cache[d]


class _BarsView:
    def __init__(self,dataset): self.dataset=dataset
    def for_codes(self,date,codes):
        out={}
        for c in codes:
            panel=self.dataset.panels.get(c)
            if panel is not None and date in panel.index: out[c]=panel.loc[date].to_dict()
        return out
    def get(self,date,default=None): return self.for_codes(date,self.dataset.panels)


def prepare(dataset, cfg, shared=None):
    config=asdict(cfg)
    signature=fingerprint({k:v for k,v in config.items() if k!='universe'})
    version=dataset.version
    extra={}
    if shared is not None:
        if shared['version']!=version or shared.get('signal_configuration')!=signature:
            raise ValueError('shared_preparation_version_mismatch')
        close,amount,turnover=shared['close'],shared['amount'],shared['turnover']
        if 'legacy_scorer' in shared:extra['legacy_scorer']=shared['legacy_scorer']
        extra['score_cache']=shared.get('score_cache',{})
    else:
        close, amount, turnover = panels_for_research(dataset)
    extra['signal_configuration']=signature
    extra.setdefault('score_cache',{})
    if len(dataset.panels)>500:
        owner=_PreparedDays(dataset,cfg,amount,close.index)
        return dict(close=close,amount=amount,turnover=turnover,version=version,**extra,
                    preparation_config=fingerprint(asdict(cfg)),bars=_BarsView(dataset),
                    **{k:_DayView(owner,k) for k in ('records','rules','exposures','eligible','close_records')})
    # Capture raw and processed inputs in every persisted run, including the exact benchmark.
    records, rules, exposures, eligible, close_records = {}, {}, {}, {}, {}
    for d in close.index:
        morning = asof_records(dataset.instruments, timestamp(d) + pd.Timedelta(hours=9, minutes=25))
        records[d] = morning
        rules[d] = {}
        for c, row in morning.iterrows():
            if row.get('state_basis')=='daily_observation_required':
                panel=dataset.panels.get(c)
                morning.loc[c,'is_st']=panel.at[d,'is_st_at_open'] if panel is not None and d in panel.index and 'is_st_at_open' in panel else None
            try: rules[d][c] = InstrumentRules.from_record(row)
            except ValueError: pass
        exposures[d] = exposure_matrix(dataset, d, morning, list(morning.index),
                                       as_of=timestamp(d) + pd.Timedelta(hours=9, minutes=25))
        eligible[d], close_records[d] = universe_at(dataset, d, amount, cfg)
    return {"close": close, "amount": amount, "turnover": turnover,
            "version": version, **extra, "preparation_config": fingerprint(config), "records": records, "rules": rules,
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
                    execution_delay=0, forecasts=None, review_override=None, crowding=True, initial_state=None, data_mode='strict', industry_policy='enforced',
                    risk_policy='P0', candidate='baseline', allocation_policy='legacy'):
        if self.market != "a": return blocked_result(["us_execution_not_validated"], self.market)
        if risk_mode not in ("daily", "legacy_monthly"): raise ValueError("invalid_risk_policy")
        if risk_policy not in ('P0','P1','P2'): raise ValueError('unknown_risk_policy')
        if risk_policy!='P0' and (risk_mode!='daily' or diagnostic or strategy!='legacy_repaired'):
            raise ValueError('dynamic_policy_requires_mandated_legacy_daily_research')
        if candidate not in ('baseline','quality_pb','quality_coverage','industry_rotation'):
            raise ValueError('unknown_candidate')
        if allocation_policy not in ('legacy','equal'): raise ValueError('unknown_allocation_policy')
        if (candidate!='baseline' or allocation_policy!='legacy') and strategy!='legacy_repaired':
            raise ValueError('candidate_requires_legacy_strategy')
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
        mandate = None if diagnostic else Mandate.from_config(selected_cfg,risk_policy=risk_policy)
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
        def dated_score(day,stocks):
            # This cache belongs to a version-checked prepared snapshot. Public
            # FactorDecisionService remains uncached for mutable research probes.
            key=(str(day),fingerprint(stocks),day in reviews)
            cache=p.setdefault('score_cache',{})
            if key not in cache:
                value=scorer.score(day,stocks)
                if day not in reviews:
                    from dataclasses import replace
                    value=replace(value,trace={k:v for k,v in value.trace.items() if k!='standardized_features'})
                cache[key]=value
            return cache[key]
        account = Ledger(cfg.capital.init_cash, cfg.cost.a_slippage, cfg.research.cash_buffer,
                         cfg.research.participation, cost_mult, cfg.cost.min_trade_value,allow_open_assumption=data_mode=='free_research')
        risk = RiskManager(cfg.risk, cfg.research, cfg.portfolio.target_vol) if risk_mode == "daily" else LegacyMonthlyRisk(cfg)
        regime = Regime(dataset.benchmark.close) if not dataset.benchmark.empty else None
        controller = PolicyController(risk_policy,cfg,dataset.benchmark.close)
        budget = controller.evaluate(dates[0]-pd.Timedelta(days=1),cfg.capital.init_cash)
        states, decisions, traces, allocation_issues, observations = [], [], [], [], []
        observation_dates=set(review_dates(dataset.calendar,'weekly')) if scorer is not None else set()
        last_risk = dict(gross_scale=1., vol_scale=1., level=0, allow_new_stock=True, reasons=[], drawdown=0.)
        invested = False
        delayed = []
        outside_count = {}
        gaps_by_date={}
        for gap in dataset.manifest.get('data_gaps',[]):
            gaps_by_date.setdefault(pd.Timestamp(gap['date']),[]).append(gap)
        accounting_by_date={}
        for gap in dataset.manifest.get('accounting_gaps',[]):
            accounting_by_date.setdefault(pd.Timestamp(gap['date']),[]).append(gap)
        actions_by_date=dict(tuple(dataset.actions.groupby(pd.to_datetime(dataset.actions.ex_date).dt.normalize()))) if not dataset.actions.empty else {}
        if initial_state is not None:
            if initial_state['mandate_version']!=(mandate.version if mandate else None):raise ValueError('forward_mandate_changed')
            account=initial_state['ledger'];risk=initial_state['risk'];last_risk=initial_state['last_risk']
            delayed=initial_state['delayed'];outside_count=initial_state['outside_count'];invested=initial_state['invested']
            if initial_state.get('risk_policy','P0')!=risk_policy or initial_state.get('candidate','baseline')!=candidate or initial_state.get('allocation_policy','legacy')!=allocation_policy:
                raise ValueError('forward_strategy_policy_changed')
            controller=initial_state.get('controller',controller);budget=initial_state.get('budget',budget)
            if risk_mode=='daily':
                updated=dataset.benchmark.close.copy()
                shared=controller.benchmark.index.intersection(updated.index)
                completed=shared[shared<=pd.Timestamp(account.snapshots[-1]['date'])] if account.snapshots else shared
                if not controller.benchmark.reindex(completed).equals(updated.reindex(completed)):
                    raise ValueError('forward_benchmark_history_changed')
                controller.benchmark=updated
            if account.snapshots and dates[0]<=account.snapshots[-1]['date']:raise ValueError('forward_session_not_new')
        if risk_mode=='daily': risk=controller.risk_manager
        config_dump = asdict(cfg)
        run_id = fingerprint({"data": p["version"], "config": config_dump, "strategy": strategy,
                              "start": str(start), "end": str(end), "risk": risk_mode,"data_mode":data_mode,
                              "cost": cost_mult, "liquidity": liquidity_mult, "blocked": list(blocked_sessions),
                              "delay": execution_delay, "schedule": target_schedule,
                              "forecasts": {str(d):dict(model=f.model_version,training=f.training_cutoff,mu=f.expected_returns.to_dict()) for d,f in (forecasts or {}).items()},
                              "crowding":crowding,"reviews":review_override,"industry_policy":industry_policy,
                              "risk_policy":risk_policy,"policy_version":controller.version,"candidate":candidate,"allocation_policy":allocation_policy,
                              "initial_account": None if initial_state is None else dict(cash=account.cash,lots={c:[asdict(l) for l in ls] for c,ls in account.lots.items()},risk=last_risk,orders=account.export()['orders'],delayed=delayed)})[:24]
        for d in dates:
            session = int(dataset.calendar.get_loc(d))
            for gap in gaps_by_date.get(d,[]):
                if account.quantity(gap['instrument_id']):
                    account.data_issues.append(dict(gap,held_shares=account.quantity(gap['instrument_id'])))
            for gap in accounting_by_date.get(d,[]):
                qty=account.quantity(gap['instrument_id'])
                if qty:
                    account.data_issues.append(dict(gap,held_shares=qty,
                        reference_gap_exposure=qty*gap.get('reference_gap_per_share',0.),
                        meaning='unexplained_reference_price_exposure_not_profit_correction_or_bias_bound'))
            if hasattr(p['bars'],'for_codes'):
                bar_names=set(account.quantities())|{o.instrument_id for o in account.orders if o.status in ('pending','partial')}
                if d in reviews or strategy=='schedule' and d in target_schedule: bar_names.update(close.columns)
                bars=p['bars'].for_codes(d,bar_names)
            else: bars = p["bars"].get(d, {})
            if d in blocked_sessions:
                bars = {c: dict(b, open_status="halted") for c, b in bars.items()}
            morning = p["records"][d]
            rules = p["rules"][d].copy()
            # The opening uses the preceding close's budget, never today's close.
            account.cash_buffer = budget.cash_buffer
            account.corporate_actions(d, session, actions_by_date.get(d,pd.DataFrame()), dataset.calendar)
            adv = amount.loc[amount.index < d].tail(20).mean().to_dict()
            adv = {c: v * liquidity_mult for c, v in adv.items()}
            execution_risk=None
            prior_dates=close.index[close.index<d]
            risk_codes=sorted(set(account.quantities())|{o.instrument_id for o in account.orders if o.status in ('pending','partial')})
            risk_codes=sorted(set(risk_codes)|{c for intent in delayed if intent[0]<=session for c,w in intent[1].items() if w>0})
            exp=exposure_matrix(dataset,d,morning,risk_codes,as_of=timestamp(d)+pd.Timedelta(hours=9,minutes=25))
            if not enforce_industry: exp = exp.iloc[:0]
            if constrained_strategy and crowding and len(prior_dates) and risk_codes:
                execution_risk=exit_risk(close,p['turnover'],amount,dataset.benchmark.close,prior_dates[-1],risk_codes,
                                         cfg.research.downside_corr,account.nav(),include_penalties=False)
            def can_buy(c, q, price, fees, ledger):
                if c not in morning.index or not _known_bool(morning.loc[c].get("active", False)) or (cfg.universe.a_exclude_st and _st_state(morning.loc[c],bars.get(c),True)): return False
                if risk_policy!='P0' and not last_risk.get('allow_new_stock',True): return False
                nav = ledger.nav() - fees
                vals = {k: ledger.quantity(k) * ledger.marks[k] for k in ledger.quantities()}
                vals[c] = vals.get(c, 0.) + q * price
                cap = cfg.research.max_etf_weight if rules[c].asset_type == "equity_etf" else budget.max_stock_weight
                if vals[c] / nav > cap + 1e-8: return False
                if sum(vals.values()) / nav > min(last_risk["gross_scale"], 1-budget.cash_buffer) + 1e-8: return False
                count_limit = len(dataset.panels) if strategy in ("buy_hold", "monthly_equal", "vol_matched_equal") else cfg.capital.a_share_positions
                if len([k for k in vals if vals[k]>0 and k in rules and rules[k].asset_type == "stock"]) > count_limit: return False
                if strategy not in ("buy_hold", "monthly_equal", "vol_matched_equal") and len(vals) > cfg.capital.target_positions: return False
                if len([k for k in vals if vals[k]>0 and k in rules and rules[k].asset_type == "equity_etf"]) > cfg.research.max_etfs: return False
                if execution_risk is not None:
                    cm=execution_risk.matrix(list(vals))
                    cluster_limits=pd.Series({_g:_group_cap(budget,_g,'cluster') for _g in cm.index})
                    if (cm @ (pd.Series(vals)/nav)>cluster_limits+1e-8).any():return False
                weights = pd.Series(vals).reindex(exp.columns, fill_value=0) / nav
                limits=pd.Series({_g:_group_cap(budget,_g,'sector') for _g in exp.index},dtype=float)
                return bool((exp @ weights <= limits + 1e-8).all())
            # Delayed intents preserve original signal dates, and cannot bypass a later halt.
            for intent in list(delayed):
                if intent[0] <= session:
                    if last_risk["gross_scale"] > 0:
                        account.queue_targets(intent[1], intent[2], intent[3], rules)
                        for order in account.orders:
                            if order.signal_date==str(pd.Timestamp(intent[2]).date()) and order.status in ('pending','partial'):
                                order.risk_policy=risk_policy;order.candidate=candidate
                                order.risk_budget=intent[4] if len(intent)>4 else budget.as_dict()
                                order.execute_not_before=str(d.date())
                                order.cash_at_submission=account.cash
                                order.estimated_notional=order.quantity*account.marks[order.instrument_id]
                                order.estimated_fees=sum(account.fee(d,rules[order.instrument_id],order.side,order.estimated_notional))
                    delayed.remove(intent)
            account.execute(d, session, bars, rules, adv, can_buy=can_buy)
            prices = {c: b.get("close", np.nan) for c, b in bars.items()
                      if pd.notna(b.get("available_at")) and timestamp(b["available_at"]) <= decision_time(d)
                      and b.get("quality_status") == "ok"}
            state = account.close(d, session, prices)
            records = p['close_records'][d]
            exp = exposure_matrix(dataset, d, records, list(records.index) if d in reviews else list(state['weights']))
            observed_exposure = exp @ pd.Series(state['weights'], dtype=float).reindex(exp.columns,fill_value=0)
            if not enforce_industry: exp = exp.iloc[:0]
            old_level = last_risk["level"]
            prior_cash_budget=budget.cash_buffer
            if strategy not in ("buy_hold", "monthly_equal") and (risk_mode == "daily" or d in reviews):
                if risk_mode=='daily':
                    exit_blocked=any(o.side=='sell' and o.status in ('pending','partial') and
                                     (o.reject_reason or o.status=='partial') and pd.Timestamp(o.signal_date)<d for o in account.orders)
                    bench_known=dataset.benchmark.loc[:d,'close']
                    stale=bool(state['stale']) or bench_known.empty or bench_known.index[-1]!=d or pd.isna(bench_known.iloc[-1])
                    budget=controller.evaluate(d,state['nav'],data_stale=stale,exit_blocked=exit_blocked)
                    last_risk=dict(budget.risk)
                else:
                    last_risk = risk.evaluate(state["nav"], regime.state(d)["multiplier"] if regime is not None else 1., date=d)
            exposure_now = exp @ pd.Series(state["weights"], dtype=float).reindex(exp.columns,fill_value=0)
            limits=pd.Series({_g:_group_cap(budget,_g,'sector') for _g in exp.index},dtype=float)
            breaches = exposure_now[exposure_now > limits + 1e-6].to_dict()
            states.append(dict(date=d, **last_risk, industry_policy=industry_policy, actual_sector_exposure=observed_exposure.to_dict(),
                               unresolved_sector_breaches=breaches,risk_budget=budget.as_dict()))
            evaluate_universe=d in reviews or d in observation_dates
            eligible = p["eligible"][d] if evaluate_universe else []
            states[-1]['eligible_count'] = len(eligible) if evaluate_universe else None
            if d in observation_dates:
                from quantlab.strategy.candidates import candidate_decision
                observation=dated_score(d,[c for c in eligible if c in rules and rules[c].asset_type=='stock'])
                observation_scores,_,obs_evidence=candidate_decision(dataset,p,d,observation.scores,candidate)
                retained=list(dict.fromkeys(list(observation_scores.index[:100])+list(account.quantities())))
                observations.append(dict(date=str(d.date()),status=observation.status,observation_only=True,
                    scores=observation_scores.reindex(retained).dropna().to_dict(),
                    expert_weights=observation.trace.get('expert_weights',{}),
                    expert_candidates=observation.trace.get('expert_candidates',{}),candidate_evidence=obs_evidence))
            # Stop decisions are close-time intents, never fabricated fills.
            stops = [c for c in account.quantities() if account.cost_basis(c)>0 and
                     account.marks[c] / account.cost_basis(c) - 1 < cfg.risk.stop_loss_single]
            status_exits = {c: ('inactive_security' if not _known_bool(records.loc[c].active) else 'st_security')
                for c in account.quantities() if c in records.index and
                (not _known_bool(records.loc[c].active) or (cfg.universe.a_exclude_st and _st_state(records.loc[c],bars.get(c))))}
            stops = sorted(set(stops) | set(status_exits))
            if strategy in ("buy_hold", "monthly_equal") or (risk_mode != "daily" and d not in reviews):
                stops = []; status_exits = {}
            states[-1]['eligibility_exit_reasons'] = status_exits
            cash_budget_tightened=risk_policy!='P0' and budget.cash_buffer>prior_cash_budget+1e-12
            # Cash is reserved on every buy. Price appreciation is not a reason
            # to sell one lot of every holding to replenish a few yuan of cash.
            # A deliberate policy tightening does generate a reduction intent.
            gross_cap=min(last_risk['gross_scale'],1-budget.cash_buffer) if cash_budget_tightened else last_risk['gross_scale']
            states[-1].update(cash_budget_tightened=cash_budget_tightened,
                              cash_reserve_shortfall=max(0.,state['nav']*budget.cash_buffer-state['cash']))
            gross_breach = (risk_mode == "daily" or d in reviews) and gross_cap < 1 and sum(state["weights"].values()) > gross_cap + 1e-4
            portfolio_vol=_portfolio_vol(close,d,state['weights'],cfg) if risk_policy!='P0' else None
            effective_vol=budget.target_vol*last_risk['vol_scale']
            vol_breach=portfolio_vol is not None and portfolio_vol>effective_vol+1e-6
            states[-1].update(estimated_portfolio_vol=portfolio_vol,effective_target_vol=effective_vol)
            constraint_target=dict(state['weights'])
            if constrained_strategy and constraint_target:
                for c,w in list(constraint_target.items()):
                    cap=cfg.research.max_etf_weight if c in rules and rules[c].asset_type=='equity_etf' else budget.max_stock_weight
                    soft=cap*.98 if risk_policy=='P0' else budget.soft_stock_weight
                    if w>cap+1e-6:constraint_target[c]=soft
                matrices=[(exp,'sector')]
                if crowding:
                    cr=exit_risk(close,p['turnover'],amount,dataset.benchmark.close,d,list(constraint_target),cfg.research.downside_corr,account.nav(),include_penalties=False)
                    matrices.append((cr.matrix(list(constraint_target)),'cluster'))
                for matrix,kind in matrices:
                    for group,row in matrix.iterrows():
                        cap=_group_cap(budget,group,kind)
                        soft=cap*.98 if risk_policy=='P0' else _group_cap(budget,group,kind,soft=True)
                        exposure=sum(row.get(c,0)*w for c,w in constraint_target.items())
                        if exposure>cap+1e-6:
                            for c in constraint_target:
                                if row.get(c,0)>0:constraint_target[c]*=soft/exposure
            constraint_breach=any(constraint_target[c]<w-1e-8 for c,w in state['weights'].items())
            states[-1]['constraint_exit_required']=constraint_breach
            risk_exit = last_risk["level"] > old_level or bool(stops) or gross_breach or constraint_breach or vol_breach
            if risk_policy!='P0' and not last_risk.get('allow_new_stock',True):
                account.cancel_buys('risk_budget_restricts_new_risk');delayed.clear()
            target = None
            reason = ""
            trace = {}
            if risk_exit:
                weights = account.weights()
                scale = min(1., gross_cap / max(sum(weights.values()), 1e-12))
                if risk_policy=='P0' and last_risk["level"] == 1: scale = min(scale, cfg.risk.vol_reduction_factor)
                if risk_policy!='P0' and portfolio_vol is not None and portfolio_vol>0:
                    scale=min(scale,effective_vol/portfolio_vol)
                target = {c: min(w * scale,constraint_target.get(c,w)) if c not in stops else 0. for c, w in weights.items()}
                trace.update(risk_reduction=dict(gross_cap=gross_cap,estimated_vol=portfolio_vol,
                                                effective_target_vol=effective_vol,scale=scale))
                reason = "risk_exit"
                account.cancel_buys(reason); delayed.clear()
            elif strategy == "schedule" and d in target_schedule:
                target, reason = target_schedule[d], "scheduled_target"
            elif strategy == 'legacy_repaired' and d in reviews:
                held = account.quantities()
                stocks = [c for c in eligible if c in rules and rules[c].asset_type=='stock']
                score = dated_score(d, stocks)
                trace = dict(score.trace, reasons=score.reasons)
                if score.status!='ok':
                    account.cancel_buys('invalid_legacy_score')
                    traces.append(dict(date=str(d.date()),status=score.status,**trace));continue
                from quantlab.strategy.candidates import candidate_decision
                scores,allowed_increases,candidate_trace=candidate_decision(dataset,p,d,score.scores,candidate)
                trace.update(candidate=candidate,candidate_evidence=candidate_trace,scores=scores.to_dict())
                locked = {}
                for c in held:
                    permission = c in rules and trade_permission(bars.get(c),rules[c],'sell',d,allow_open_assumption=data_mode=='free_research')[0]
                    if (not permission or account.available(c,session)<held[c] or
                        account.holding_age(c,session)<cfg.capital.min_holding_period_days):locked[c]=state['weights'][c]
                pending_exits = {o.instrument_id for o in account.orders if o.reason=='risk_exit' and o.status in ('pending','partial')}
                desired = list(sorted(locked))
                for c in scores.index:
                    if c in pending_exits or c in desired:continue
                    if not last_risk.get('allow_new_stock',True) and c not in held:continue
                    if allowed_increases is not None and c not in allowed_increases and c not in held:continue
                    if len(desired)>=cfg.capital.a_share_positions:break
                    desired.append(c)
                codes = sorted(set(desired)|set(held))
                rs = exit_risk(close,p['turnover'],amount,dataset.benchmark.close,d,codes,cfg.research.downside_corr,account.nav(),include_penalties=False) if crowding and codes else None
                exposures = exp.reindex(columns=codes,fill_value=0.)
                returns = close.loc[:d,codes].pct_change(fill_method=None).tail(cfg.portfolio.vol_est_window)
                x,status,capital = allocate_legacy(returns,cfg,selected=desired,previous=state['weights'],locked=locked,
                    exposures=exposures,risk_snapshot=rs,gross=last_risk['gross_scale'],vol_scale=last_risk['vol_scale'],
                    budget=budget,equal_weight=allocation_policy=='equal',allowed_increases=allowed_increases)
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
                # Delay the same signal-day share order; do not recompute its
                # quantity using a later NAV/price while calling that a delay.
                account.queue_targets(target, d, reason, rules)
                invested = True
                new_orders = [o for o in account.orders if o.signal_date==str(d.date())]
                for o in new_orders:
                    o.industry_policy=industry_policy;o.risk_policy=risk_policy;o.risk_budget=budget.as_dict();o.candidate=candidate
                    o.estimated_notional=o.quantity*account.marks[o.instrument_id]
                    o.estimated_fees=sum(account.fee(exec_date or d,rules[o.instrument_id],o.side,o.estimated_notional))
                    o.cash_at_submission=account.cash
                    o.execute_not_before=str(exec_date.date()) if exec_date is not None else None
                    o.execute_not_before_session=session+1+execution_delay
                trace.update(industry_policy=industry_policy,capital=dict(trace.get('capital',{}),
                    target_gross=sum(target.values()),cash_before_orders=account.cash,nav=account.nav(),
                    submitted_notional=sum(o.quantity*account.marks[o.instrument_id] for o in new_orders),
                    order_ids=[o.order_id for o in new_orders]))
                decisions.append({"signal_date": str(d.date()), "exec_date": str(exec_date.date()) if exec_date is not None else None,
                                  "weights": target, "factor_weights": trace.get("factor_weights", {}),
                                  "experts": trace.get("expert_weights", {}), "risk": dict(last_risk),
                                  "regime": regime.state(d) if regime is not None else {}, "reason": reason,
                                  "risk_budget":budget.as_dict(),"candidate":candidate,"allocation_policy":allocation_policy})
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
                  "risk_policy":risk_policy,"risk_policy_version":controller.version,"candidate":candidate,"allocation_policy":allocation_policy,
                  "research_exceptions":[] if enforce_industry else ['industry_limit_omitted_for_diagnostic'],
                  "weight_history": [pd.Series(x["weights"], name=x["date"], dtype=float) for x in account.snapshots],
                  "factor_panels": {}, "ic_matrix": scorer.ic if scorer is not None else pd.DataFrame(), "codes": list(close), "market": self.market,
                  "run_id": run_id, "data_version": p["version"], "model_version": reported_model,
                  "as_of": str(end), "status": status, "reasons": errors, "promotion_approved": False,
                  "mandate_version": mandate.version if mandate else "diagnostic_unrestricted",
                  "config": config_dump, "benchmark": dataset.benchmark.copy(), "benchmark_manifest": dataset.manifest,
                  "risk_history": states, "signal_trace": traces, "allocation_issues": allocation_issues,
                  "weekly_observations":observations,
                  "pending_intents": delayed, "strategy": strategy, "risk_mode": risk_mode,
                  "input_panels": dataset.panels, "input_instruments": dataset.instruments,
                  "input_actions": dataset.actions, "calendar": dataset.calendar, **account.export()}
        result['_runtime']=dict(ledger=account,risk=risk,last_risk=last_risk,delayed=delayed,outside_count=outside_count,
                                risk_policy=risk_policy,controller=controller,budget=budget,candidate=candidate,allocation_policy=allocation_policy,
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
        from quantlab.report.advice import rotation_advice
        result['rotation_advice']=rotation_advice(result)
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
