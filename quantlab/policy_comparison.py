"""A finite policy attribution protocol on one frozen, market-wide dataset.

The main eight configurations separate universe, risk, financial coverage,
quality selection, industry rotation and allocation. Two reference slots link
verified artifacts for the original 18.82% report and its cash/share replay.
The reference has different permissions and is not ranked against primary arms.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import json
import sqlite3
import tarfile

import numpy as np
import pandas as pd

from quantlab.config import CONFIG, ROOT
from quantlab.data.contracts import fingerprint, load_dataset, asof_records, decision_time, InstrumentRules
from quantlab.data.qualification import research_assessment
from quantlab.research import ExperimentRegistry
from quantlab.research_statistics import (paired_returns, simultaneous_block_ci,
                                         family_sharpe, retrospective_segments, fixed_two_year_segments)
from quantlab.report.runs import atomic_json, persist_run, load_run, provenance
from quantlab.report.metrics import performance_table
from quantlab.report.research_diagnostics import account_diagnostics
from quantlab.strategy.legacy import legacy_config


ARMS = {
    'U30P0': dict(minimum_adv=30_000_000., risk_policy='P0', candidate='baseline', allocation_policy='legacy'),
    'U5P0': dict(minimum_adv=5_000_000., risk_policy='P0', candidate='baseline', allocation_policy='legacy'),
    'U5P1': dict(minimum_adv=5_000_000., risk_policy='P1', candidate='baseline', allocation_policy='legacy'),
    'U5P2': dict(minimum_adv=5_000_000., risk_policy='P2', candidate='baseline', allocation_policy='legacy'),
    'qualityP2': dict(minimum_adv=5_000_000., risk_policy='P2', candidate='quality_pb', allocation_policy='legacy'),
    'qualityCoverageP2': dict(minimum_adv=5_000_000., risk_policy='P2', candidate='quality_coverage', allocation_policy='legacy'),
    'industryP2': dict(minimum_adv=5_000_000., risk_policy='P2', candidate='industry_rotation', allocation_policy='legacy'),
    'equalP2': dict(minimum_adv=5_000_000., risk_policy='P2', candidate='baseline', allocation_policy='equal'),
}
CONTRASTS = {
    'universe': ('U5P0', 'U30P0'),
    'risk_P1': ('U5P1', 'U5P0'),
    'risk_P2': ('U5P2', 'U5P1'),
    'quality_total': ('qualityP2', 'U5P2'),
    'quality_signal': ('qualityP2', 'qualityCoverageP2'),
    'quality_coverage': ('qualityCoverageP2', 'U5P2'),
    'industry_rotation': ('industryP2', 'U5P2'),
    'allocation': ('equalP2', 'U5P2'),
}
REFERENCE_ARMS = ('reference1882_original', 'reference1882_cash_ledger')
STRESS_NAMES = ('cost_double', 'liquidity_30pct', 'fixed_lock_5', 'adverse_exit_lock_5', 'full_halt_5',
                'execution_delay_2', 'missing_held_quotes_5', 'opening_gap_10pct',
                'correlated_fall_5')


def _record_once(registry, trial, outcome):
    """Resume a completed append without changing an already recorded outcome."""
    fold = str(outcome.get('fold', 'default'))
    with sqlite3.connect(registry.path) as db:
        row = db.execute('SELECT body FROM outcomes WHERE id=? AND fold=?', (trial, fold)).fetchone()
    if row:
        if fingerprint(json.loads(row[0])) != fingerprint(json.loads(json.dumps(outcome, default=str))):
            raise ValueError('recorded_outcome_differs_on_resume')
    else:
        registry.record(trial, outcome)


def comparison_payload(run):
    """Keep full evidence on disk; retain only downstream statistical inputs."""
    return {key:run[key] for key in ('run_id','daily_return','account','fills','risk_history') if key in run}


def evidence_health(summary):
    reasons = []
    if not summary.get('accounting',{}).get('passed'): reasons.append('accounting_failed_or_missing')
    if not summary.get('experiment_health',{}).get('passed'): reasons.append('experiment_not_exercised_or_unverified')
    health = summary.get('run_health')
    if health is None: reasons.append('run_health_unavailable')
    else:
        for key, reason in (('allocation_failures','allocation_failures'),
                            ('forecast_failures','invalid_or_warmup_decisions'),
                            ('data_issue_count','held_data_gaps')):
            if health.get(key,0): reasons.append(reason)
    return dict(passed=not reasons, reasons=reasons)


def recorded_stress_results(registry, trial, checkpoint=None):
    """SQLite is written first, so it also recovers a crash before JSON replace."""
    with sqlite3.connect(registry.path) as db:
        rows = db.execute('SELECT fold,body FROM outcomes WHERE id=? AND fold LIKE ?',
                          (trial,'stress:%')).fetchall()
    recorded = {}
    for fold, body in rows:
        name = fold.removeprefix('stress:')
        if name not in STRESS_NAMES: raise ValueError('unknown_registered_stress_scenario:'+name)
        outcome = json.loads(body)
        if outcome.pop('fold',None) != fold: raise ValueError('stress_registered_fold_mismatch')
        recorded[name] = outcome
    for name, outcome in (checkpoint or {}).items():
        if name not in recorded: raise ValueError('stress_checkpoint_not_registered:'+name)
        normalized = json.loads(json.dumps(outcome,default=str))
        if fingerprint(normalized)!=fingerprint(recorded[name]):
            raise ValueError('stress_checkpoint_differs_from_registry:'+name)
    return recorded


def validate_stress_artifact(outcome, root):
    """Never skip a completed replay based on an unchecked progress JSON."""
    if outcome.get('status') not in ('completed','not_exercised','failed'):
        raise ValueError('invalid_stress_checkpoint_status')
    run_id = outcome.get('run_id')
    if not run_id:
        if outcome['status']=='completed': raise ValueError('completed_stress_artifact_missing')
        return
    folder = (Path(root)/run_id).resolve()
    if not folder.is_relative_to(Path(root).resolve()): raise ValueError('invalid_stress_artifact_path')
    run = load_run(folder)
    if run.get('run_id')!=run_id: raise ValueError('stress_artifact_identity_mismatch')
    actual = json.loads(json.dumps(summarize_run(run),default=str))
    expected = json.loads(json.dumps(outcome.get('run'),default=str))
    if fingerprint(actual)!=fingerprint(expected): raise ValueError('stress_artifact_summary_mismatch')


def reference_outcome(name, audit):
    key = 'original_replay' if name == 'reference1882_original' else 'cash_ledger_replay'
    section = audit.get(key, {})
    return dict(fold='source_audit', status=section.get('status', 'not_replayed'),
                audit=section, report=audit.get('report'),
                reason=section.get('reason', 'see_verified_reconstruction_artifacts'),
                ranked_against_main_strategy=False)


def frozen_protocol(ds, *, start='2020-01-02', end='2026-06-30', run_stress=True):
    from quantlab.strategy.risk_policy import policy_definition
    scope = ds.manifest.get('research_scope', {})
    if scope.get('sample_size') or any(word in str(scope.get('scope', '')).lower()
                                      for word in ('sample', 'cohort')):
        raise ValueError('fixed_sample_is_not_market_wide_policy_evidence')
    if ds.manifest.get('universe_coverage') != 'historical':
        raise ValueError('historical_dynamic_universe_required')
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError('invalid_protocol_interval')
    return dict(version='policy-convergence-v1', seed=20260924, start=str(start), end=str(end),
        arms=deepcopy(ARMS), reference_arms=list(REFERENCE_ARMS), contrasts=deepcopy(CONTRASTS),
        registered_configurations=len(ARMS) + len(REFERENCE_ARMS), max_configurations=12,
        risk_policy_definitions={name:policy_definition(name) for name in ('P0','P1','P2')},
        primary='252_times_mean_paired_daily_net_return_difference',
        bootstrap=dict(blocks=[20, 60], samples=2000, confidence=.95, correction='max_absolute_t_declared_family'),
        dsr=dict(periods_per_year=252, moments='observed_returns', historical_search_complete=False),
        history_status='all_historical_periods_retrospective_not_blind',
        selection='Report all registered configurations; no test-set tuning or automatic promotion.',
        matching='Same daily cash ledger, costs, execution, position limits and data snapshot; declared policy changes only.',
        industry_mode=ds.manifest.get('industry_mode', 'verified_pit'), universe_scope=scope,
        stress=dict(enabled=bool(run_stress), names=list(STRESS_NAMES),
                    scope='U5P2 plus primary configurations with positive net profit',
                    locks='fixed/adverse locks reject sells only; full_halt_5 rejects both sides; actual dated rejection events verify exit demand',
                    price_shocks='synthetic OHLC and feature prices obey dated limits and tick rounding; report requested and realized shock',
                    adverse_window='worst five-day account return among windows containing a baseline sell fill; ex-post diagnosis'),
        references='Original 100-holding/500000-CNY result is a source audit, not ranked against 20-holding/600000-CNY arms.')


def arm_config(spec, cfg=None):
    result = legacy_config(cfg or CONFIG)
    result.seed = 20260924
    result.research.feature_profile = 'price_volume'
    result.research.universe_mode = 'pit_liquidity'
    result.universe.a_min_avg_amount_wan = spec['minimum_adv'] / 10_000
    return result


def account_checks(run):
    states = run.get('account', [])
    errors = []
    for state in states:
        securities = sum(q * state['marks'][c] for c, q in state['quantities'].items())
        residual = state['nav'] - state['cash'] - securities - state.get('receivables', 0.) + state.get('payables', 0.)
        if not np.isfinite([state['nav'], state['cash'], securities, residual]).all() or abs(residual) > .02:
            errors.append('account_identity')
        if state['cash'] < -.001 or any(q < 0 for q in state['quantities'].values()):
            errors.append('negative_cash_or_shares')
    # Delisting settlement is an accounting event, not a signal-driven order.
    if any(pd.Timestamp(f['date']) <= pd.Timestamp(f['signal_date'])
           for f in run.get('fills', []) if f.get('reason') != 'delisting'):
        errors.append('same_day_or_early_execution')
    return dict(passed=bool(states) and not errors, sessions=len(states), errors=sorted(set(errors)))


def experiment_health(run):
    """A flat cash account is not an observed strategy return experiment."""
    states, traces = run.get('account', []), run.get('signal_trace', [])
    buys = [f for f in run.get('fills', []) if f.get('side') == 'buy' and f.get('quantity', 0) > 0]
    invested = sum(bool(s.get('quantities')) for s in states)
    decisions = [t for t in traces if t.get('scores') and t.get('target_weights')]
    eligible = [x['eligible_count'] for x in run.get('risk_history', []) if x.get('eligible_count') is not None]
    candidate = run.get('candidate', 'baseline')
    candidate_counts = []
    for trace in traces:
        evidence = trace.get('candidate_evidence', {})
        if candidate in ('quality_pb', 'quality_coverage'):
            candidate_counts.append(evidence.get('qualified_count', 0))
        elif candidate == 'industry_rotation':
            candidate_counts.append(len(evidence.get('industry_adjustments', {})))
    reasons = []
    if not buys or not invested: reasons.append('no_funded_positions')
    if not decisions: reasons.append('no_valid_score_to_target_decisions')
    if eligible and max(eligible) == 0: reasons.append('no_historically_eligible_securities')
    if candidate != 'baseline' and (not candidate_counts or max(candidate_counts) == 0):
        reasons.append('candidate_not_exercised')
    cfg = run.get('config', {})
    return dict(passed=not reasons, status='exercised' if not reasons else 'experiment_not_exercised', reasons=reasons,
        expected_bounds=dict(minimum_adv=cfg.get('universe', {}).get('a_min_avg_amount_wan', 0.) * 10_000,
                             max_positions=cfg.get('capital', {}).get('a_share_positions'),
                             risk_policy=run.get('risk_policy', 'P0'), candidate=candidate,
                             allocation_policy=run.get('allocation_policy', 'legacy')),
        coverage=dict(sessions=len(states), invested_sessions=invested, buy_fills=len(buys), valid_decisions=len(decisions),
                      eligible_min=min(eligible) if eligible else None,
                      eligible_median=float(np.median(eligible)) if eligible else None,
                      eligible_max=max(eligible) if eligible else None,
                      candidate_coverage_max=max(candidate_counts) if candidate_counts else None,
                      candidate_covered_decisions=sum(x > 0 for x in candidate_counts)))


def summarize_run(run):
    states = run.get('account', [])
    checks = account_checks(run)
    health = experiment_health(run)
    if not states:
        return dict(run_id=run['run_id'], status=run['status'], reasons=run.get('reasons', []),
                    accounting=checks, experiment_health=health, metrics=None, conditional_metrics=None, net_profit=None)
    metrics = performance_table(run['nav'], run['daily_return'])
    fees = sum(x['fees'] for x in run.get('fills', []))
    slippage = sum(x.get('slippage_cost', 0.) for x in run.get('fills', []))
    contribution = {}
    for f in run.get('fills', []):
        reason = str(f.get('reason', 'unknown'))
        bucket = contribution.setdefault(reason, dict(fills=0, notional=0., fees=0., slippage=0.))
        bucket['fills'] += 1
        bucket['notional'] += f.get('notional', 0.)
        bucket['fees'] += f['fees']
        bucket['slippage'] += f.get('slippage_cost', 0.)
    budgets = _budget_observations(run)
    budget_share = budgets.state.value_counts(normalize=True).to_dict() if not budgets.empty else {}
    return dict(run_id=run['run_id'], status=run['status'], risk_policy=run.get('risk_policy', 'P0'),
        reasons=run.get('reasons', []), metrics=None,
        conditional_metrics=metrics if health['passed'] else None, accounting=checks, experiment_health=health,
        run_health=run.get('run_health', {}),
        net_profit=float(states[-1]['nav'] - run['config']['capital']['init_cash']),
        average_gross=float(np.mean([sum(s['weights'].values()) for s in states])),
        average_holdings=float(np.mean([len(s['quantities']) for s in states])), fees=fees, slippage=slippage,
        two_way_notional=sum(x.get('notional', 0.) for x in run.get('fills', [])),
        realized_annualized_volatility=float(run['daily_return'].std(ddof=1) * np.sqrt(252)),
        mean_target_volatility=float(budgets.target_vol.mean()) if not budgets.empty else None,
        risk_budget_state_share=budget_share,
        order_reasons=contribution, pending_orders=sum(o['status'] in ('pending', 'partial') for o in run.get('orders', [])),
        capital_diagnostics=run.get('capital_diagnostics', {}),
        account_diagnostics=account_diagnostics(run),
        promotion_approved=False)


def _budget_observations(run):
    rows = []
    for row in run.get('risk_history', []):
        budget = row.get('risk_budget')
        if not budget: continue
        rows.append(dict(date=pd.Timestamp(row['date']), state=budget['state'],
                         target_vol=budget['target_vol'] * budget.get('vol_scale', 1.),
                         gross_scale=budget.get('gross_scale', 1.), cash_buffer=budget.get('cash_buffer', 0.)))
    return pd.DataFrame(rows).set_index('date').sort_index() if rows else pd.DataFrame()


def lagged_budget_attribution(candidate, control):
    a, b = candidate['daily_return'], control['daily_return']
    if not a.index.equals(b.index): raise ValueError('budget_attribution_calendar_mismatch')
    observed = _budget_observations(candidate).reindex(a.index)
    if observed.empty or 'state' not in observed:
        return dict(status='budget_history_unavailable')
    lagged = observed.shift(1)
    previous = observed.shift(2)
    change = ((lagged.state != previous.state) | (lagged.target_vol != previous.target_vol)
              | (lagged.gross_scale != previous.gross_scale)) & lagged.state.notna() & previous.state.notna()
    frame = lagged.assign(candidate_return=a, control_return=b, net_excess=a-b, budget_changed=change).dropna(subset=['state'])
    def group_stats(group):
        return dict(sessions=len(group), annualized_mean_net_excess=float(group.net_excess.mean() * 252),
                    candidate_annualized_volatility=float(group.candidate_return.std(ddof=1) * np.sqrt(252)) if len(group)>1 else None,
                    mean_prior_target_vol=float(group.target_vol.mean()), mean_prior_gross_ceiling=float(group.gross_scale.mean()))
    return dict(status='descriptive_not_causal', budget_alignment='preceding_close_budget_for_current_session_return',
                by_state={str(k): group_stats(g) for k,g in frame.groupby('state')},
                by_budget_change={str(bool(k)): group_stats(g) for k,g in frame.groupby('budget_changed')},
                caution='P2 changes permitted risk. Higher return with higher exposure or volatility is not, by itself, timing alpha.')


def stress_windows(run, calendar):
    """A fixed calendar window plus a separately labelled adverse exit window."""
    dates = pd.DatetimeIndex(run['daily_return'].index)
    eligible = dates[dates >= pd.Timestamp(f'{dates[0].year}-07-01')]
    fixed_start = eligible[0] if len(eligible) else dates[len(dates) // 2]
    fixed = list(dates[dates.searchsorted(fixed_start):dates.searchsorted(fixed_start) + 5])
    if len(fixed) != 5: fixed = []
    sell_days = sorted({pd.Timestamp(f['date']).normalize() for f in run.get('fills', []) if f['side'] == 'sell'})
    candidates = []
    for day in sell_days:
        j = dates.searchsorted(day)
        window = dates[j:j + 5]
        if len(window) == 5:
            candidates.append((float((1 + run['daily_return'].loc[window]).prod() - 1), day, list(window)))
    adverse = min(candidates, key=lambda x: (x[0], x[1]))[2] if candidates else []
    return dict(fixed=fixed, adverse=adverse, baseline_sell_dates=sell_days)


def _stress_dataset(ds, run, sessions, kind):
    altered = copy(ds)
    altered.manifest = deepcopy(ds.manifest)
    altered.manifest['stress_variant'] = dict(kind=kind, sessions=[str(x) for x in sessions])
    prior = [s for s in run['account'] if pd.Timestamp(s['date']) < sessions[0]]
    codes = list(prior[-1]['quantities']) if prior else []
    altered.panels = dict(ds.panels)
    evidence = altered.manifest['stress_variant']
    evidence.update(synthetic=True,clipped_price_cells=0,price_impacts={})
    stress_dates=pd.DatetimeIndex(sessions)
    records=asof_records(ds.instruments,decision_time(sessions[0])) if kind!='missing_held_quotes_5' else None
    def rounded(value,tick):
        return float((Decimal(str(value))/Decimal(str(tick))).quantize(Decimal('1'),rounding=ROUND_HALF_UP)*Decimal(str(tick)))
    def bounded(value,lower,upper,tick):
        if not np.isfinite(value):return value
        result=rounded(value,tick)
        clipped=max(lower,result) if np.isfinite(lower) else result
        clipped=min(upper,clipped) if np.isfinite(upper) else clipped
        evidence['clipped_price_cells']+=int(abs(clipped-result)>tick/10)
        return clipped
    from quantlab.backtest.ledger import price_bounds
    from quantlab.data.exchange_rules import limit_prices,limit_ratio
    for code in codes:
        panel = ds.panels[code].copy()
        if kind == 'missing_held_quotes_5':
            panel = panel.loc[~panel.index.isin(sessions)]
        elif kind == 'opening_gap_10pct':
            if sessions[0] in panel.index:
                rule=InstrumentRules.from_record(records.loc[code]);date=sessions[0]
                original=panel.loc[date]
                if not np.isfinite(original.open) or original.open<=0:
                    evidence['price_impacts'][code]=dict(status='not_exercised',reason='original_open_missing')
                    continue
                upper,lower=price_bounds(original,rule)
                price=bounded(float(original.open)*.90,lower,upper,rule.price_tick)
                panel.loc[date,'open']=price
                panel.loc[date,'high']=max(float(original.get('high',original.close)),float(original.close),price)
                panel.loc[date,'low']=min(float(original.get('low',original.close)),float(original.close),price)
                evidence['price_impacts'][code]=dict(requested_open_ratio=.90,actual_open_ratio=price/float(original.open),
                    actual_open_vs_reference=price/float(original.reference_price)-1,
                    clipped_to_exchange_limit=abs(price-rounded(float(original.open)*.90,rule.price_tick))>rule.price_tick/10)
        elif kind == 'correlated_fall_5':
            rule=InstrumentRules.from_record(records.loc[code]);tick=rule.price_tick
            previous_factor=1.;previous_steps=0
            updates={}
            for date,original in ds.panels[code].loc[sessions[0]:].iterrows():
                steps=int(stress_dates.searchsorted(date,side='right'))
                requested_factor=previous_factor*.98**(steps-previous_steps)
                if not np.isfinite(original.reference_price) or original.reference_price<=0:
                    raise ValueError('stress_reference_unavailable:'+code+':'+str(date.date()))
                reference=rounded(float(original.reference_price)*previous_factor,tick)
                ratio=limit_ratio(code,date,original.get('is_st_at_open',np.nan),original.get('listing_session_lower_bound'))
                if ratio is None:
                    # Observed bars can supply their exact dated band even when
                    # the generic rule helper cannot identify a fixture/market.
                    upper0,lower0=price_bounds(original,rule)
                    matching=[r for r in (.05,.10,.20,.30) if np.isfinite(upper0) and np.isfinite(lower0)
                              and all(abs(a-b)<tick/10 for a,b in zip(limit_prices(original.reference_price,r,tick),(upper0,lower0)))]
                    if len(matching)==1:ratio=matching[0]
                    elif rule.limit_ratio is not None:raise ValueError('stress_dated_limits_unverified:'+code+':'+str(date.date()))
                upper,lower=limit_prices(reference,ratio,tick) if ratio is not None else (np.nan,np.nan)
                values={field:bounded(float(original[field])*requested_factor,lower,upper,tick)
                        for field in ('open','high','low','close') if field in panel and pd.notna(original[field])}
                if 'high' in values:values['high']=max(values.values())
                if 'low' in values:values['low']=min(values.values())
                actual_factor=values['close']/float(original.close) if 'close' in values else requested_factor
                values.update(reference_price=reference,limit_up=upper,limit_down=lower)
                if 'feature_close' in panel:values['feature_close']=original.feature_close*actual_factor
                updates[date]=values
                previous_factor,previous_steps=actual_factor,steps
                if date<=sessions[-1]:
                    evidence['price_impacts'][code]=dict(requested_cumulative_ratio=.98**steps,
                        actual_close_ratio=actual_factor,through=str(date.date()),feature_price_shocked='feature_close' in panel)
            if updates:
                updated=pd.DataFrame.from_dict(updates,orient='index')
                panel.loc[updated.index,updated.columns]=updated
        altered.panels[code] = panel
    return altered, codes


def run_stress_suite(ds, cfg, spec, base, replay, *, prepared=None, resume=None,
                     verify_resume=None, on_result=None):
    windows = stress_windows(base, ds.calendar)
    results = {}
    if resume and verify_resume is None: raise ValueError('stress_resume_verifier_required')
    if set(resume or {})-set(STRESS_NAMES): raise ValueError('unknown_resumed_stress_scenario')
    def remember(name, outcome):
        if on_result is not None: on_result(name,outcome)
        results[name] = outcome
    for name in STRESS_NAMES:
        if name in (resume or {}):
            verify_resume(name,resume[name])
            results[name] = resume[name]
            continue
        kwargs = {}
        changed = ds
        active_prepared = prepared
        sessions = windows['adverse'] if name in ('adverse_exit_lock_5', 'missing_held_quotes_5',
                                                 'opening_gap_10pct', 'correlated_fall_5') else windows['fixed']
        if name == 'cost_double': kwargs['cost_mult'] = 2.
        if name == 'liquidity_30pct': kwargs['liquidity_mult'] = .3
        if name in ('fixed_lock_5', 'adverse_exit_lock_5'): kwargs['blocked_sell_sessions'] = sessions
        if name == 'full_halt_5': kwargs['blocked_sessions'] = sessions
        if name == 'execution_delay_2': kwargs['execution_delay'] = 2
        codes = []
        if name in ('missing_held_quotes_5', 'opening_gap_10pct', 'correlated_fall_5'):
            if sessions:
                try:
                    changed, codes = _stress_dataset(ds, base, sessions, name)
                except (ValueError,KeyError,TypeError,ArithmeticError) as exc:
                    remember(name,dict(status='failed',constraints_passed=False,reason=str(exc)))
                    continue
                active_prepared = None
        needs_window = name not in ('cost_double', 'liquidity_30pct', 'execution_delay_2')
        if needs_window and (not sessions or (name in ('missing_held_quotes_5', 'opening_gap_10pct', 'correlated_fall_5') and not codes)):
            remember(name,dict(status='not_exercised', constraints_passed=False, reason='no_held_exit_window'))
            continue
        try:
            stress = replay(changed, cfg, spec, prepared=active_prepared, **kwargs)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            remember(name,dict(status='failed', constraints_passed=False, reason=str(exc)))
            continue
        summary = summarize_run(stress)
        locks=('fixed_lock_5','adverse_exit_lock_5','full_halt_5')
        fills_during_lock = [f for f in stress.get('fills', []) if pd.Timestamp(f['date']).normalize() in sessions
                             and (name=='full_halt_5' or f['side']=='sell')]
        exercised = bool(stress.get('fills'))
        matching_exits = []
        rejections=stress.get('corporate_action_events',[])
        matching_rejections=[]
        if name in locks:
            # Orders can receive different IDs once paths diverge. Dated actual
            # execution attempts establish that a sell was due and was blocked;
            # the final order reason may already be cleared by a later fill.
            orders={o['order_id']:o for o in stress.get('orders',[])}
            reason='unknown_or_halted_open' if name=='full_halt_5' else 'stress_sell_lock'
            for event in rejections:
                if (event.get('type')!='execution_rejected' or event.get('side')!='sell'
                    or event.get('reason')!=reason or event.get('remaining',0)<=0
                    or pd.Timestamp(event['date']).normalize() not in sessions):continue
                order=orders.get(event.get('order_id'))
                if (order and order['side']=='sell' and order['quantity']>0
                    and order['instrument_id']==event.get('instrument_id')
                    and pd.Timestamp(order['signal_date'])<pd.Timestamp(event['date'])
                    and (not order.get('execute_not_before') or pd.Timestamp(order['execute_not_before'])<=pd.Timestamp(event['date']))):
                    matching_rejections.append(event)
            matching_ids={e['order_id'] for e in matching_rejections}
            matching_exits=[orders[i] for i in sorted(matching_ids)]
            exercised = bool(matching_exits)
        elif name in ('missing_held_quotes_5','opening_gap_10pct','correlated_fall_5'):
            held_states=[s for s in stress.get('account',[]) if pd.Timestamp(s['date'])<sessions[0]]
            exercised=bool(held_states and set(held_states[-1]['quantities'])&set(codes))
            if name=='opening_gap_10pct':
                exercised=exercised and any(e.get('actual_open_ratio',1.)<1. for e in
                    changed.manifest.get('stress_variant',{}).get('price_impacts',{}).values())
            if name=='missing_held_quotes_5':
                exercised=exercised and any(str(e.get('reason','')).startswith('stale_') and
                    pd.Timestamp(e['date']).normalize() in sessions for e in stress.get('data_issues',[]))
        malformed=[e for e in rejections if e.get('type')=='execution_rejected' and
            e.get('instrument_id') in codes and e.get('reason') in
            ('derived_limit_mismatch','derived_open_outside_rule_bounds','unverified_derived_limit_rule')
            and pd.Timestamp(e['date'])>=sessions[0]] if sessions else []
        constraints = summary['accounting']['passed'] and not (name in locks and fills_during_lock) and not malformed
        remember(name,dict(status='completed' if exercised else 'not_exercised',
            constraints_passed=bool(constraints and exercised), window=[str(x.date()) for x in sessions],
            ex_post_window=name in ('adverse_exit_lock_5','missing_held_quotes_5','opening_gap_10pct','correlated_fall_5'), held_codes=codes,
            scenario_exercised=exercised,matching_exit_request=bool(matching_exits) if name in locks else None,
            run=summary, run_id=stress['run_id'],
            lock_scope='all_sides' if name=='full_halt_5' else 'sell_only' if name in locks else None,
            rejection_observations=matching_rejections,invalid_stress_rejections=malformed,
            price_stress_evidence=changed.manifest.get('stress_variant') if changed is not ds else None,
            requested_exit_orders=len(matching_exits), drawdown_target=.20 if spec['risk_policy']=='P0' else .25,
            risk_target_met=bool(summary.get('conditional_metrics') and
                summary['conditional_metrics']['max_drawdown'] >= (-.20 if spec['risk_policy']=='P0' else -.25))))
        del stress
    return results


def verdict(summary, comparison, ci20, ci60, stress, *, data_assessment, control_summary=None):
    candidate_health = evidence_health(summary)
    control_health = evidence_health(control_summary or {})
    reasons = candidate_health['reasons'][:]
    reasons.extend('control_'+reason for reason in control_health['reasons'])
    if not data_assessment.get('accounting_verified'): reasons.append('accounting_coverage_unverified')
    if summary.get('net_profit') is None or summary['net_profit'] <= 0: reasons.append('net_profit_not_positive')
    if not comparison or comparison['annualized_net_excess'] <= 0: reasons.append('increment_not_positive')
    segments = [s for s in (comparison or {}).get('segments', []) if s.get('complete', False)]
    if len(segments) < 3 or sum(s['annualized_net_excess'] > 0 for s in segments) / len(segments) < 2 / 3:
        reasons.append('cross_period_consistency_not_met')
    if not ci20 or not ci60 or min(ci20.get('simultaneous_lower', -1), ci60.get('simultaneous_lower', -1)) <= 0:
        reasons.append('family_adjusted_increment_not_proven')
    metrics = summary.get('conditional_metrics') or {}
    drawdown_target = .20 if summary.get('risk_policy', 'P0') == 'P0' else .25
    if metrics.get('max_drawdown', -1) < -drawdown_target: reasons.append('drawdown_target_exceeded')
    if not stress or any(not s.get('constraints_passed') for s in stress.values()) or len(stress) != len(STRESS_NAMES):
        reasons.append('stress_missing_failed_or_not_exercised')
    # Solvent, executable orders do not imply that the account stayed within
    # its drawdown budget. Keep the separately recorded pressure risk verdict.
    if not stress or any(not s.get('risk_target_met') for s in stress.values()):
        reasons.append('stress_drawdown_target_missing_or_exceeded')
    if stress.get('cost_double', {}).get('run', {}).get('net_profit', 0.) <= 0:
        reasons.append('double_cost_profit_not_positive')
    if data_assessment.get('warnings'): reasons.append('research_data_assumptions_require_review')
    return dict(status='retrospective_candidate_for_forward_observation' if not reasons else 'unproven',
                reasons=sorted(set(reasons)), promotion_approved=False,
                comparison_health=dict(candidate=candidate_health,control=control_health),
                retrospective_only=True, historical_search_complete=False, drawdown_target=drawdown_target,
                note='No automatic adoption. Historical selection, concentration, execution calibration and forward evidence remain separate.')


def run_comparison(manifest, output, *, start='2020-01-02', end='2026-06-30',
                   run_stress=True, reference_audit=None, stop_after_primary=False):
    """Freeze, register and replay. Call only after all implementation is frozen."""
    from quantlab.backtest.engine import Engine, prepare
    ds = load_dataset(manifest, profile='free_research')
    assessment = research_assessment(ds, require_industry=False)
    if assessment['status'] != 'research_ready': return assessment
    data_version = ds.version
    protocol = frozen_protocol(ds, start=start, end=end, run_stress=run_stress)
    source = provenance()
    identity = fingerprint(dict(protocol=protocol, data=data_version, config=asdict(CONFIG), source=source))[:24]
    root = Path(output) / identity
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'summary.json').exists(): return json.loads((root / 'summary.json').read_text())
    frozen = dict(protocol=protocol, data_version=data_version, source=source, config=asdict(CONFIG), assessment=assessment)
    if (root / 'protocol.json').exists() and json.loads((root / 'protocol.json').read_text()) != json.loads(json.dumps(frozen, default=str)):
        raise ValueError('protocol_already_frozen_differently')
    atomic_json(root / 'protocol.json', frozen)
    if not (root / 'source.tar.gz').exists():
        with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
            for file in sorted([*(ROOT / 'quantlab').rglob('*.py'), *(ROOT / 'scripts').glob('*.py')]):
                archive.add(file, arcname=str(file.relative_to(ROOT)))
    registry = ExperimentRegistry(root / 'registry')
    registered = {name: registry.register('policy_attribution', spec, data_version, 'fixed', protocol['version'])
                  for name, spec in ARMS.items()}
    for name in REFERENCE_ARMS:
        registered[name] = registry.register('reference1882_source_audit', dict(reference=name), data_version,
                                             'source_audit', protocol['version'])
    # Registration is complete before the first account is evaluated.
    before = provenance()
    if before != source: raise ValueError('source_changed_before_replay')
    cache = {}
    completed = {}
    summaries = {}
    outcomes_path = root / 'outcomes.json'
    outcomes = json.loads(outcomes_path.read_text()) if outcomes_path.exists() else {}

    def prepared_for(cfg):
        key = fingerprint(asdict(cfg))
        if key not in cache:
            cache[key] = prepare(ds, cfg, shared=next(iter(cache.values())) if cache else None)
        return cache[key]

    def replay(data, cfg, spec, *, prepared=None, **kwargs):
        result = Engine(cfg).run_dataset(data, strategy='legacy_repaired', start=start, end=end,
                    data_mode='free_research', industry_policy='enforced', prepared=prepared,
                    risk_policy=spec['risk_policy'], candidate=spec['candidate'],
                    allocation_policy=spec['allocation_policy'], **kwargs)
        if provenance() != source: raise ValueError('source_or_environment_changed_during_replay')
        compact = {k: v for k, v in result.items()
                   if k not in ('input_panels', 'input_instruments', 'input_actions', '_runtime', 'factor_panels')}
        compact['source_dataset'] = dict(manifest=str(Path(manifest).resolve()), data_version=data_version,
                                        stress_variant=data.manifest.get('stress_variant'))
        persist_run(compact, root=root / ('runs' if not kwargs and data is ds else 'stress_runs'))
        result['run_id'], result['code_version'] = compact['run_id'], compact['code_version']
        return result

    for name, spec in ARMS.items():
        print('[policy] replay', name, flush=True)
        if name in outcomes:
            prior = outcomes[name]
            _record_once(registry, registered[name], prior)
            if prior.get('run_id'):
                folder = (root / 'runs' / prior['run_id']).resolve()
                if not folder.is_relative_to((root / 'runs').resolve()):
                    raise ValueError('invalid_primary_artifact_path')
                result = load_run(folder)
                summary = summarize_run(result)
                if (result.get('run_id') != prior['run_id'] or
                    fingerprint(json.loads(json.dumps(summary, default=str))) != fingerprint(prior.get('summary'))):
                    raise ValueError('primary_artifact_summary_mismatch')
                completed[name] = comparison_payload(result)
                summaries[name] = summary
                del result
            continue
        cfg = arm_config(spec)
        try:
            result = replay(ds, cfg, spec, prepared=prepared_for(cfg))
            summary = summarize_run(result)
            outcome = dict(fold='fixed', status=result['status'], run_id=result['run_id'], summary=summary)
            completed[name] = comparison_payload(result)
            summaries[name] = summary
            del result
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            outcome = dict(fold='fixed', status='failed', reason=str(exc))
        _record_once(registry, registered[name], outcome)
        outcomes[name] = outcome
        atomic_json(outcomes_path, outcomes)
        print('[policy] saved', name, outcome['status'], flush=True)
        if provenance() != source: raise ValueError('source_or_environment_changed_during_replay')
    # Release the caller's process between phases without changing the protocol
    # or recording absent reference evidence as an immutable completed outcome.
    primary = dict(run_id=identity,status='primary_completed_pending_validation',
        data_version=data_version,model_version=protocol['version'],as_of=end,
        artifact_directory=str(root),runs=summaries,completed_primary=[k for k in ARMS if k in outcomes],
        failed={k:v for k,v in outcomes.items() if k in ARMS and v['status'] in ('failed','blocked_data')},
        promotion_approved=False,source_unchanged=provenance()==source,
        remaining=['reference_source_audit','paired_statistics','registered_stress','adjudication'])
    atomic_json(root/'primary_summary.json',primary)
    lines=['# 主研究阶段结果','',
        '预登记主配置已回放；参考、统计和压力尚未全部完成，不据此采用策略。条件收益的数据缺口见每组run_health。','',
        '| 配置 | 状态 | 净利润 | 年化 | 最大回撤 | 平均仓位 | 平均持仓数 | 费用及滑点 | 数据问题 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ARMS:
        summary=summaries.get(name,{})
        metrics=summary.get('conditional_metrics')
        if metrics:
            lines.append(f"| {name} | {summary['status']} | {summary['net_profit']:,.2f} | {metrics['annual_return']:.2%} | {metrics['max_drawdown']:.2%} | {summary['average_gross']:.2%} | {summary['average_holdings']:.1f} | {summary['fees']+summary['slippage']:,.2f} | {summary['run_health'].get('data_issue_count',0)} |")
        else:
            lines.append(f"| {name} | {outcomes[name]['status']}: {outcomes[name].get('reason','')} | — | — | — | — | — | — | — |")
    (root/'PRIMARY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    if stop_after_primary:return primary
    audit = json.loads(Path(reference_audit).read_text()) if reference_audit else {}
    for name in REFERENCE_ARMS:
        if name not in outcomes:
            outcome = reference_outcome(name, audit)
            _record_once(registry, registered[name], outcome)
            outcomes[name] = outcome
            atomic_json(outcomes_path, outcomes)
        else:
            _record_once(registry, registered[name], outcomes[name])
    # Keep conditional differences even when an exercised arm has failures.
    # Adoption checks below require both candidate and control to be healthy.
    usable = {n: r for n, r in completed.items() if summaries[n]['accounting']['passed'] and summaries[n]['experiment_health']['passed']}
    contrasts = {k: pair for k, pair in CONTRASTS.items() if all(n in usable for n in pair)}
    family = paired_returns(usable, contrasts) if contrasts else pd.DataFrame()
    ci = {str(block): simultaneous_block_ci(family, block=block, seed=protocol['seed'])
          for block in (20, 60)} if not family.empty else {}
    comparisons = {name: dict(candidate=contrasts[name][0], control=contrasts[name][1],
                             comparison_health={side:evidence_health(summaries[arm])
                                                for side,arm in zip(('candidate','control'),contrasts[name])},
                             annualized_net_excess=float(values.mean() * 252),
                             relative_wealth=float((1 + usable[contrasts[name][0]]['daily_return']).prod() /
                                                   (1 + usable[contrasts[name][1]]['daily_return']).prod() - 1),
                             segments=fixed_two_year_segments(values)['stages'],
                             observation_2026H1=fixed_two_year_segments(values)['observation_2026H1'],
                             halfyears=retrospective_segments(values),
                             lagged_budget_attribution=lagged_budget_attribution(usable[contrasts[name][0]], usable[contrasts[name][1]]))
                   for name, values in family.items()}
    dsr = family_sharpe(usable, registered_trials=len(ARMS), historical_search_complete=False) if usable else {}
    stress_path = root / 'stress.json'
    stresses = json.loads(stress_path.read_text()) if stress_path.exists() else {}
    if run_stress:
        for name in usable:
            if name != 'U5P2' and summaries[name]['net_profit'] <= 0: continue
            print('[policy] stress', name, flush=True)
            spec = ARMS[name]
            cfg = arm_config(spec)
            resumed = recorded_stress_results(registry, registered[name], stresses.get(name))
            stresses[name] = dict(resumed)
            def checkpoint(scenario, outcome):
                _record_once(registry, registered[name], dict(fold='stress:' + scenario, **outcome))
                stresses[name][scenario] = outcome
                atomic_json(stress_path, stresses)
            def verify(scenario, outcome):
                validate_stress_artifact(outcome, root / 'stress_runs')
            remaining = set(STRESS_NAMES) - set(resumed)
            stresses[name] = run_stress_suite(ds, cfg, spec, usable[name], replay,
                prepared=prepared_for(cfg) if remaining else None,
                resume=resumed, verify_resume=verify, on_result=checkpoint)
            atomic_json(stress_path, stresses)
    adjudications = {}
    for contrast, comparison in comparisons.items():
        name = comparison['candidate']
        adjudications[contrast] = verdict(summaries[name], comparison,
            ci.get('20', {}).get('comparisons', {}).get(contrast), ci.get('60', {}).get('comparisons', {}).get(contrast),
            stresses.get(name, {}), data_assessment=assessment, control_summary=summaries[comparison['control']])
    summary = dict(run_id=identity, status='research_unvalidated', data_version=data_version,
        model_version=protocol['version'], as_of=end, metrics=None, promotion_approved=False,
        historical_data_already_examined=True, protocol=protocol, runs=summaries,
        failed={k: v for k, v in outcomes.items() if v['status'] in ('failed', 'blocked_data')},
        comparisons=comparisons, block_family_intervals=ci, dsr=dsr, stress=stresses,
        adjudications=adjudications, references={k: outcomes[k] for k in REFERENCE_ARMS},
        limitations=assessment['warnings'] + ['historical_search_inventory_incomplete', 'not_blind_oos'],
        artifact_directory=str(root), source_unchanged=provenance() == source)
    atomic_json(root / 'summary.json', summary)
    lines = ['# QuantLab 固定政策对照', '', '历史回顾性研究；未批准实盘。全部预登记配置和失败均保留。', '',
             '| 配置 | 状态 | 净利润 | 年化 | 最大回撤 | 平均仓位 | 费用及滑点 |', '|---|---|---:|---:|---:|---:|---:|']
    for name in ARMS:
        s = summaries.get(name, {})
        m = s.get('conditional_metrics')
        if m:
            lines.append(f"| {name} | {s['status']} | {s['net_profit']:,.2f} | {m['annual_return']:.2%} | {m['max_drawdown']:.2%} | {s['average_gross']:.2%} | {s['fees'] + s['slippage']:,.2f} |")
        else:
            lines.append(f"| {name} | {outcomes[name]['status']}: {outcomes[name].get('reason', '')} | — | — | — | — | — |")
    lines += ['', '因果配对、20/60日分块家族区间、DSR实测矩、压力结果及未通过原因见 summary.json。',
              '100只股票的18.82%参考按独立权限核验；复原状态与证据见references，不与主策略直接排名。']
    lines += ['', '## 账户路径与执行诊断', '',
              '| 配置 | 最差完整半年 | 最深回撤恢复 | 期末仍在回撤 | 最大单股/行业 | 日频群组 | 年化双边换手 | 卖单完成中位天数/未完数量 |',
              '|---|---|---|---|---|---|---|---|']
    for name in ARMS:
        diagnostic = summaries.get(name,{}).get('account_diagnostics',{})
        if not diagnostic or diagnostic.get('status')=='no_account': continue
        half = diagnostic['halfyear_returns']['worst_complete']
        half_text = f"{half['period']} {half['net_return']:.2%}" if half else '完整半年不足'
        recovery = diagnostic['drawdown_recovery']
        deepest = recovery['deepest']
        recovery_text = (f"{deepest['peak_to_recovery_sessions']}交易日" if deepest['recovered'] else
                         f"未恢复，已{deepest['elapsed_sessions']}交易日") if deepest else '未发生回撤'
        concentration = diagnostic['realized_concentration']
        def maximum_text(field):
            value = concentration[field]['maximum']
            if value is None: return '不可计算'
            return f'{value:.2%}' + ('（部分日期）' if concentration[field]['status']=='partial' else '')
        group_text = maximum_text('group') if concentration['group']['status']=='complete' else '覆盖不足' if concentration['group']['status']=='partial' else '不可计算'
        turnover = diagnostic['turnover']['annualized_two_way']
        turnover_text = f'{turnover:.2f}倍' if turnover is not None else '不可计算'
        exits = diagnostic['sell_completion']['all_sells']
        duration = f"{exits['completed_median_sessions']:g}" if exits['completed_median_sessions'] is not None else '—'
        lines.append(f"| {name} | {half_text} | {recovery_text} | {'是' if recovery['unrecovered_at_end'] else '否'} | {maximum_text('single_security')} / {maximum_text('industry')} | {group_text} | {turnover_text} | {duration} / {exits['pending_orders']} |")
    lines += ['', '换手=各日实际买卖成交额/前收盘净值之和×252/账户交易日数，含首次建仓；行业集中度按每日真实市值/NAV，UNKNOWN单列归组。',
              '回撤恢复从前高计交易日；未恢复时长截至末次账户估值。卖单完结不等于整只股票清仓，取消/替换与待执行分开统计。',
              '没有保存每日真实群组划分时明确不可计算，不将月度群组前填。完整逐日覆盖、半年度收益与逐单耗时见account_diagnostics。']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return summary
