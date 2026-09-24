"""A finite policy attribution protocol on one frozen, market-wide dataset.

The main eight configurations separate universe, risk, financial coverage,
quality selection, industry rotation and allocation. Two reference slots link
verified artifacts for the original 18.82% report and its cash/share replay.
The reference has different permissions and is not ranked against primary arms.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import sqlite3
import tarfile

import numpy as np
import pandas as pd

from quantlab.config import CONFIG, ROOT
from quantlab.data.contracts import fingerprint, load_dataset
from quantlab.data.qualification import research_assessment
from quantlab.research import ExperimentRegistry
from quantlab.research_statistics import (paired_returns, simultaneous_block_ci,
                                         family_sharpe, retrospective_segments, fixed_two_year_segments)
from quantlab.report.runs import atomic_json, persist_run, load_run, provenance
from quantlab.report.metrics import performance_table
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
STRESS_NAMES = ('cost_double', 'liquidity_30pct', 'fixed_lock_5', 'adverse_exit_lock_5',
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
    if any(pd.Timestamp(f['date']) <= pd.Timestamp(f['signal_date']) for f in run.get('fills', [])):
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
    for code in codes:
        panel = ds.panels[code].copy()
        if kind == 'missing_held_quotes_5':
            panel = panel.loc[~panel.index.isin(sessions)]
        elif kind == 'opening_gap_10pct':
            if sessions[0] in panel.index:
                panel.loc[sessions[0], 'open'] *= .90
        elif kind == 'correlated_fall_5':
            positions = ds.calendar.searchsorted(panel.index) - ds.calendar.searchsorted(sessions[0]) + 1
            scale = np.power(.98, np.clip(positions, 0, 5))
            previous_scale = np.power(.98, np.clip(positions - 1, 0, 5))
            for field in ('open', 'high', 'low', 'close'):
                if field in panel:
                    panel[field] = panel[field] * scale
            for field in ('reference_price', 'limit_up', 'limit_down'):
                if field in panel:
                    panel[field] = panel[field] * previous_scale
        altered.panels[code] = panel
    return altered, codes


def run_stress_suite(ds, cfg, spec, base, replay, *, prepared=None):
    windows = stress_windows(base, ds.calendar)
    results = {}
    for name in STRESS_NAMES:
        kwargs = {}
        changed = ds
        active_prepared = prepared
        sessions = windows['adverse'] if name in ('adverse_exit_lock_5', 'missing_held_quotes_5',
                                                 'opening_gap_10pct', 'correlated_fall_5') else windows['fixed']
        if name == 'cost_double': kwargs['cost_mult'] = 2.
        if name == 'liquidity_30pct': kwargs['liquidity_mult'] = .3
        if name in ('fixed_lock_5', 'adverse_exit_lock_5'): kwargs['blocked_sessions'] = sessions
        if name == 'execution_delay_2': kwargs['execution_delay'] = 2
        codes = []
        if name in ('missing_held_quotes_5', 'opening_gap_10pct', 'correlated_fall_5'):
            if sessions:
                changed, codes = _stress_dataset(ds, base, sessions, name)
                active_prepared = None
        needs_window = name not in ('cost_double', 'liquidity_30pct', 'execution_delay_2')
        if needs_window and (not sessions or (name in ('missing_held_quotes_5', 'opening_gap_10pct', 'correlated_fall_5') and not codes)):
            results[name] = dict(status='not_exercised', constraints_passed=False, reason='no_held_exit_window')
            continue
        try:
            stress = replay(changed, cfg, spec, prepared=active_prepared, **kwargs)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            results[name] = dict(status='failed', constraints_passed=False, reason=str(exc))
            continue
        summary = summarize_run(stress)
        fills_during_lock = [f for f in stress.get('fills', []) if pd.Timestamp(f['date']).normalize() in sessions]
        exercised = bool(stress.get('fills'))
        matching_exits = []
        if name in ('fixed_lock_5', 'adverse_exit_lock_5'):
            # Verify the baseline actually demanded an exit during this window;
            # a lock with no requested sale is not an exit stress test.
            baseline_exit_ids = {f['order_id'] for f in base.get('fills', [])
                                 if f['side'] == 'sell' and pd.Timestamp(f['date']).normalize() in sessions}
            matching_exits = [o for o in stress.get('orders', []) if o['side'] == 'sell'
                              and o['order_id'] in baseline_exit_ids and o['quantity'] > 0
                              and pd.Timestamp(o['signal_date']).normalize() < sessions[-1]]
            exercised = bool(matching_exits)
        constraints = summary['accounting']['passed'] and not (name in ('fixed_lock_5', 'adverse_exit_lock_5') and fills_during_lock)
        results[name] = dict(status='completed' if exercised else 'not_exercised',
            constraints_passed=bool(constraints and exercised), window=[str(x.date()) for x in sessions],
            ex_post_window=name != 'fixed_lock_5' and needs_window, held_codes=codes,
            matching_exit_request=exercised, run=summary, run_id=stress['run_id'],
            requested_exit_orders=len(matching_exits), drawdown_target=.20 if spec['risk_policy']=='P0' else .25,
            risk_target_met=bool(summary.get('conditional_metrics') and
                summary['conditional_metrics']['max_drawdown'] >= (-.20 if spec['risk_policy']=='P0' else -.25)))
    return results


def verdict(summary, comparison, ci20, ci60, stress, *, data_assessment):
    reasons = []
    if not summary.get('accounting', {}).get('passed'): reasons.append('accounting_failed_or_missing')
    if not data_assessment.get('accounting_verified'): reasons.append('accounting_coverage_unverified')
    if summary.get('run_health', {}).get('allocation_failures', 0): reasons.append('allocation_failures')
    if summary.get('run_health', {}).get('forecast_failures', 0): reasons.append('invalid_or_warmup_decisions')
    if summary.get('run_health', {}).get('data_issue_count', 0): reasons.append('held_data_gaps')
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
    if stress.get('cost_double', {}).get('run', {}).get('net_profit', 0.) <= 0:
        reasons.append('double_cost_profit_not_positive')
    if data_assessment.get('warnings'): reasons.append('research_data_assumptions_require_review')
    return dict(status='retrospective_candidate_for_forward_observation' if not reasons else 'unproven',
                reasons=sorted(set(reasons)), promotion_approved=False,
                retrospective_only=True, historical_search_complete=False, drawdown_target=drawdown_target,
                note='No automatic adoption. Historical selection, concentration, execution calibration and forward evidence remain separate.')


def run_comparison(manifest, output, *, start='2020-01-02', end='2026-06-30',
                   run_stress=True, reference_audit=None):
    """Freeze, register and replay. Call only after all implementation is frozen."""
    from quantlab.backtest.engine import Engine, prepare
    ds = load_dataset(manifest, profile='free_research')
    assessment = research_assessment(ds, require_industry=False)
    if assessment['status'] != 'research_ready': return assessment
    protocol = frozen_protocol(ds, start=start, end=end, run_stress=run_stress)
    source = provenance()
    identity = fingerprint(dict(protocol=protocol, data=ds.version, config=asdict(CONFIG), source=source))[:24]
    root = Path(output) / identity
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'summary.json').exists(): return json.loads((root / 'summary.json').read_text())
    frozen = dict(protocol=protocol, data_version=ds.version, source=source, config=asdict(CONFIG), assessment=assessment)
    if (root / 'protocol.json').exists() and json.loads((root / 'protocol.json').read_text()) != json.loads(json.dumps(frozen, default=str)):
        raise ValueError('protocol_already_frozen_differently')
    atomic_json(root / 'protocol.json', frozen)
    with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
        for file in sorted([*(ROOT / 'quantlab').rglob('*.py'), *(ROOT / 'scripts').glob('*.py')]):
            archive.add(file, arcname=str(file.relative_to(ROOT)))
    registry = ExperimentRegistry(root / 'registry')
    registered = {name: registry.register('policy_attribution', spec, ds.version, 'fixed', protocol['version'])
                  for name, spec in ARMS.items()}
    for name in REFERENCE_ARMS:
        registered[name] = registry.register('reference1882_source_audit', dict(reference=name), ds.version,
                                             'source_audit', protocol['version'])
    # Registration is complete before the first account is evaluated.
    before = provenance()
    if before != source: raise ValueError('source_changed_before_replay')
    cache = {}
    completed = {}
    summaries = {}
    outcomes_path = root / 'outcomes.json'
    outcomes = json.loads(outcomes_path.read_text()) if outcomes_path.exists() else {}

    def replay(data, cfg, spec, *, prepared=None, **kwargs):
        result = Engine(cfg).run_dataset(data, strategy='legacy_repaired', start=start, end=end,
                    data_mode='free_research', industry_policy='enforced', prepared=prepared,
                    risk_policy=spec['risk_policy'], candidate=spec['candidate'],
                    allocation_policy=spec['allocation_policy'], **kwargs)
        if provenance() != source: raise ValueError('source_or_environment_changed_during_replay')
        compact = {k: v for k, v in result.items()
                   if k not in ('input_panels', 'input_instruments', 'input_actions', '_runtime', 'factor_panels')}
        compact['source_dataset'] = dict(manifest=str(Path(manifest).resolve()), data_version=ds.version,
                                        stress_variant=data.manifest.get('stress_variant'))
        persist_run(compact, root=root / ('runs' if not kwargs and data is ds else 'stress_runs'))
        result['run_id'], result['code_version'] = compact['run_id'], compact['code_version']
        return result

    for name, spec in ARMS.items():
        print('[policy] replay', name, flush=True)
        if name in outcomes:
            prior = outcomes[name]
            if prior.get('run_id'):
                result = load_run(root / 'runs' / prior['run_id'])
                completed[name] = result
                summaries[name] = summarize_run(result)
            continue
        cfg = arm_config(spec)
        key = fingerprint(asdict(cfg))
        try:
            if key not in cache:
                cache[key] = prepare(ds, cfg, shared=next(iter(cache.values())) if cache else None)
            result = replay(ds, cfg, spec, prepared=cache[key])
            summary = summarize_run(result)
            outcome = dict(fold='fixed', status=result['status'], run_id=result['run_id'], summary=summary)
            completed[name] = {k:v for k,v in result.items()
                               if k not in ('input_panels','input_instruments','input_actions','_runtime','factor_panels')}
            summaries[name] = summary
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            outcome = dict(fold='fixed', status='failed', reason=str(exc))
        _record_once(registry, registered[name], outcome)
        outcomes[name] = outcome
        atomic_json(outcomes_path, outcomes)
        if provenance() != source: raise ValueError('source_or_environment_changed_during_replay')
    audit = json.loads(Path(reference_audit).read_text()) if reference_audit else {}
    for name in REFERENCE_ARMS:
        if name not in outcomes:
            outcome = reference_outcome(name, audit)
            _record_once(registry, registered[name], outcome)
            outcomes[name] = outcome
            atomic_json(outcomes_path, outcomes)
    usable = {n: r for n, r in completed.items() if summaries[n]['accounting']['passed'] and summaries[n]['experiment_health']['passed']}
    contrasts = {k: pair for k, pair in CONTRASTS.items() if all(n in usable for n in pair)}
    family = paired_returns(usable, contrasts) if contrasts else pd.DataFrame()
    ci = {str(block): simultaneous_block_ci(family, block=block, seed=protocol['seed'])
          for block in (20, 60)} if not family.empty else {}
    comparisons = {name: dict(candidate=contrasts[name][0], control=contrasts[name][1],
                             annualized_net_excess=float(values.mean() * 252),
                             relative_wealth=float((1 + usable[contrasts[name][0]]['daily_return']).prod() /
                                                   (1 + usable[contrasts[name][1]]['daily_return']).prod() - 1),
                             segments=fixed_two_year_segments(values)['stages'],
                             observation_2026H1=fixed_two_year_segments(values)['observation_2026H1'],
                             halfyears=retrospective_segments(values),
                             lagged_budget_attribution=lagged_budget_attribution(usable[contrasts[name][0]], usable[contrasts[name][1]]))
                   for name, values in family.items()}
    dsr = family_sharpe(usable, registered_trials=len(ARMS), historical_search_complete=False) if usable else {}
    stresses = {}
    if run_stress:
        for name in usable:
            if name != 'U5P2' and summaries[name]['net_profit'] <= 0: continue
            print('[policy] stress', name, flush=True)
            spec = ARMS[name]
            cfg = arm_config(spec)
            stresses[name] = run_stress_suite(ds, cfg, spec, usable[name], replay,
                                               prepared=cache.get(fingerprint(asdict(cfg))))
            for scenario, outcome in stresses[name].items():
                _record_once(registry, registered[name], dict(fold='stress:' + scenario, **outcome))
            atomic_json(root / 'stress.json', stresses)
    adjudications = {}
    for contrast, comparison in comparisons.items():
        name = comparison['candidate']
        adjudications[contrast] = verdict(summaries[name], comparison,
            ci.get('20', {}).get('comparisons', {}).get(contrast), ci.get('60', {}).get('comparisons', {}).get(contrast),
            stresses.get(name, {}), data_assessment=assessment)
    summary = dict(run_id=identity, status='research_unvalidated', data_version=ds.version,
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
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return summary
