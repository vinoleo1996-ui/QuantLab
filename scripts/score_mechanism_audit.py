#!/usr/bin/env python3
"""Q2-a offline evidence audit. No account execution or market-data loading."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'output/frozen_source/6157c64'
FROZEN_COMMIT = '6157c64c84717dc8645dde3fe6228590b9f8538e'
TASK = ROOT / 'output/score_mechanism_20260927'
ATTRIBUTION = ROOT / 'output/continuity_review_20260927'
SUMMARY_SHA = '15a1af9cf89960db457dbfa2f637daf6fe69e5347e5c8b8d7057958a589a1327'
RESULT_SHA = 'f8becbec36caee759bb4e45b2a2371427060860e92b03b6365ac287619ade7c9'
EXPERTS = ('momentum', 'value', 'lowvol_quality', 'flow', 'composite')
MODELS = (*EXPERTS, 'mixture')
PHASES = ('2020-21', '2022-23', '2024-25', '2026H1')
HORIZONS = (20, 60)

_spec = importlib.util.spec_from_file_location('score_audit_attribution', ROOT / 'scripts/continuity_attribution.py')
_attribution = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attribution)
sha = _attribution.sha
score_buckets = _attribution.score_buckets
phase = _attribution.phase
records = _attribution.records


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_hashes(expected):
    verified = {}
    for path, digest in expected.items():
        path = Path(path)
        if not path.is_file() or sha(path) != digest:
            raise ValueError('source_hash_mismatch:' + str(path))
        verified[str(path)] = digest
    return verified


def start_attempt(output, protocol):
    output = Path(output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError('attempt_exists:' + str(output)) from error
    atomic_json(output / 'protocol.json', protocol)
    atomic_json(output / 'status.json', dict(status='running', started_at_utc=utc_now(),
                                           promotion_approved=False, account_replayed=False))


def validate_reconstruction(actual, expected, label):
    return _attribution.validate_scores(actual, expected, {}, {}, label)


def rank_stat(score, outcome):
    valid = np.isfinite(score) & np.isfinite(outcome)
    x, y = score[valid], outcome[valid]
    if len(x) < 10:
        return None, 'insufficient_observations'
    if x.nunique() < 2:
        return None, 'constant_score'
    if y.nunique() < 2:
        return None, 'constant_outcome'
    return float(x.rank(method='average').corr(y.rank(method='average'))), 'observed'


def finite_mean(values):
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def difference(left, right):
    return left - right if left is not None and right is not None else None


def metric_row(block, model, horizon, common, bucket):
    score = block['score_' + model]
    observed = block[f'coverage_{horizon}'].eq('observed') & np.isfinite(block[f'return_{horizon}'])
    usable = common & observed
    return_col, residual_col = f'return_{horizon}', f'residual_{horizon}'
    rankic, rankic_status = rank_stat(score[usable], block.loc[usable, return_col])
    residual_ic, residual_status = rank_stat(score[usable], block.loc[usable, residual_col])
    row = dict(model=model, horizon=horizon, requested_count=len(block),
               finite_score_count=int(np.isfinite(score).sum()),
               missing_model_score_count=int((~np.isfinite(score)).sum()),
               excluded_other_model_score_count=int((np.isfinite(score) & ~common).sum()),
               common_finite_score_count=int(common.sum()), future_observed_count=int(usable.sum()),
               future_missing_count=int((common & ~observed).sum()),
               missing_reasons=json.dumps(block.loc[common & ~observed, f'coverage_{horizon}']
                                          .fillna('missing_outcome_status').value_counts().to_dict(), sort_keys=True),
               rankic=rankic, rankic_status=rankic_status,
               residual_rankic=residual_ic, residual_rankic_status=residual_status,
               return_mean=finite_mean(block.loc[usable, return_col]),
               residual_mean=finite_mean(block.loc[usable, residual_col]))
    for q in (3, 5):
        qmask = common & bucket.eq(q)
        row[f'q{q}_requested'] = int(qmask.sum())
        row[f'q{q}_observed'] = int((qmask & observed).sum())
        row[f'q{q}_mean'] = finite_mean(block.loc[qmask & observed, return_col])
        row[f'q{q}_residual'] = finite_mean(block.loc[qmask & observed, residual_col])
    row['q5_minus_q3'] = difference(row['q5_mean'], row['q3_mean'])
    row['q5_minus_q3_residual'] = difference(row['q5_residual'], row['q3_residual'])
    row['quintile_status'] = ('insufficient_observations' if usable.sum() < 10 else
                              'constant_score' if score[common].nunique() < 2 else
                              'missing_quintile_observations' if row['q5_minus_q3'] is None else 'observed')
    if row['quintile_status'] != 'observed':
        row['q5_minus_q3'] = row['q5_minus_q3_residual'] = None
    return row


def model_diagnostics(frame, models=MODELS, horizons=HORIZONS):
    """Freeze buckets on the common finite signal universe before outcomes."""
    block = frame.copy()
    common = np.isfinite(block[['score_' + name for name in models]]).all(axis=1)
    block['common_finite_models'] = common
    rows = []
    for model in models:
        bucket = pd.Series(np.nan, index=block.index)
        if common.any():
            bucket.loc[common] = score_buckets(block.loc[common, 'score_' + model])
        block['bucket_' + model] = bucket
        for horizon in horizons:
            row = metric_row(block, model, horizon, common, bucket)
            if len(block) and 'date' in block:
                row['date'] = str(block.date.iloc[0])
                row['phase'] = phase(row['date'])
            rows.append(row)
    return pd.DataFrame(rows), block


def weight_proxy(ic_matrix, decision, factor_order, specs, lookback=12):
    """Original proxy, explicitly cut on matured IC endpoint before last 12 rows."""
    history = ic_matrix.loc[ic_matrix.index <= pd.Timestamp(decision), factor_order].tail(lookback)
    rows, factors = [], []
    for expert, names in specs.items():
        group = [f for f in (names or factor_order) if f in factor_order and f in history]
        if not group:
            raise ValueError('missing_expert_ic_group:' + expert)
        panel = history[group]
        per_row = panel.mean(axis=1)
        signed = float(per_row.mean()) if per_row.notna().any() else 0.
        rows.append(dict(date=str(pd.Timestamp(decision).date()), expert=expert, factor_count=len(group),
                         signed_proxy=signed, clipped_q=max(signed, 0.), ic_rows=len(history),
                         observed_ic_rows=int(per_row.notna().sum()),
                         oldest_ic_endpoint=str(history.index[0].date()) if len(history) else None,
                         latest_ic_endpoint=str(history.index[-1].date()) if len(history) else None))
        for factor in group:
            # Handles differing per-row finite factor counts; contributions sum to signed proxy.
            contribution = panel[factor].div(panel.notna().sum(axis=1)).fillna(0.)
            contribution = contribution[per_row.notna()]
            factors.append(dict(date=str(pd.Timestamp(decision).date()), expert=expert, factor=factor,
                                observed_ic_rows=int(panel[factor].notna().sum()),
                                factor_ic_mean=finite_mean(panel[factor]),
                                contribution_to_group_mean=float(contribution.mean()) if len(contribution) else 0.))
    q = np.array([row['clipped_q'] for row in rows])
    temperature = max(float(np.std(q)), .015)
    weights = np.full(len(rows), 1. / len(rows)) if q.sum() <= 0 else np.exp((q - q.max()) / temperature)
    weights /= weights.sum()
    for row, weight in zip(rows, weights):
        row.update(temperature=temperature, softmax_applied=bool(q.sum() > 0),
                   full_precision_weight=float(weight), rounded_weight=round(float(weight), 4))
    return pd.DataFrame(rows), pd.DataFrame(factors)


def load_frozen_api():
    """Reject already imported live code instead of silently mixing versions."""
    for name, module in list(sys.modules.items()):
        if name == 'quantlab' or name.startswith('quantlab.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(FROZEN.resolve()):
                raise ValueError('non_frozen_module:' + name)
    frozen_path = str(FROZEN.resolve())
    if not sys.path or sys.path[0] != frozen_path:
        sys.path.insert(0, frozen_path)
    from quantlab.strategy import experts, scoring
    for module in (experts, scoring):
        if not Path(module.__file__).resolve().is_relative_to(FROZEN.resolve()):
            raise ValueError('non_frozen_import:' + module.__name__)
    return experts, scoring


def reconstruct_trace(trace, ic_matrix, saved, factor_config, top_k):
    experts, scoring = load_frozen_api()
    if factor_config.expert_weight_policy != 'original':
        raise ValueError('unexpected_expert_weight_policy')
    d = pd.Timestamp(trace['date'])
    full = {name: pd.DataFrame([values], index=[d]) for name, values in trace['standardized_features'].items()}
    history = ic_matrix.loc[ic_matrix.index <= d, list(full)]
    penalty, _ = scoring.factor_crowding_penalty(history, factor_config.crowding_corr_window,
                                               factor_config.crowding_corr_alert)
    if not penalty.empty:
        penalty = penalty.reindex(penalty.index.union([d])).sort_index().ffill().loc[[d]]
    factor_weights = pd.DataFrame([trace['factor_weights']], index=[d])
    final, rounded = experts.experts_score(full, factor_weights, penalty, history,
                                          lookback=factor_config.expert_ic_lookback, cfg=factor_config)
    if final is None or d not in final.index:
        raise ValueError('missing_reconstructed_score:' + str(d))
    error = validate_reconstruction(final.loc[d], pd.Series(trace['scores']), str(d))
    if rounded != trace['expert_weights']:
        raise ValueError('expert_weight_mismatch:' + str(d))
    eligible = list(saved.instrument_id)
    saved_error = validate_reconstruction(final.loc[d], saved.set_index('instrument_id').score, str(d))
    candidates = experts.expert_top_union({f: v.reindex(columns=eligible) for f, v in full.items()},
                                          factor_weights, penalty, None, top_k, cfg=factor_config)
    if candidates != trace['expert_candidates']:
        raise ValueError('candidate_order_mismatch:' + str(d))
    union = set(c for group in candidates.values() for c in group)
    if union != set(trace['scores']):
        raise ValueError('candidate_union_score_membership_mismatch:' + str(d))
    proxy, factors = weight_proxy(ic_matrix, d, list(full), experts.EXPERT_SPECS,
                                  factor_config.expert_ic_lookback)
    if proxy.set_index('expert').rounded_weight.to_dict() != rounded:
        raise ValueError('full_precision_proxy_weight_mismatch:' + str(d))
    proxy['stored_rounded_weight'] = proxy.expert.map(trace['expert_weights'])
    proxy['phase'] = phase(d)
    panel = saved.copy().set_index('instrument_id')
    entries = trace.get('candidate_evidence', {}).get('entries', {})
    panel['financial_covered'] = pd.Series(
        {code: entries.get(code, {}).get('covered') if code in union else None for code in panel.index})
    panel['financial_entry_present'] = [code in union and code in entries for code in panel.index]
    mixture = pd.Series(0., index=final.columns)
    for expert, names in experts.EXPERT_SPECS.items():
        raw = experts._group_score(full, factor_weights, penalty, names).loc[d]
        percentile = raw.rank(pct=True, method='average')
        panel['raw_' + expert] = raw.reindex(panel.index)
        panel['score_' + expert] = percentile.reindex(panel.index)
        weight = proxy.loc[proxy.expert.eq(expert), 'full_precision_weight'].iloc[0]
        mixture += percentile.reindex(mixture.index) * weight
    precision_error = validate_reconstruction(mixture, final.loc[d].dropna(), str(d))
    panel['score_mixture'] = final.loc[d].reindex(panel.index)
    for factor, values in full.items():
        panel['factor_' + factor] = values.loc[d].reindex(panel.index)
    factors['factor_weight'] = factors.factor.map(trace['factor_weights'])
    factors['crowding_penalty'] = factors.factor.map(
        penalty.loc[d].fillna(1.).to_dict() if not penalty.empty else {}).fillna(1.)
    factors['phase'] = phase(d)
    return panel.reset_index(), proxy, factors, dict(
        date=str(d.date()), scored_count=len(saved), union_count=len(union),
        max_score_error=error, max_saved_eligible_error=saved_error,
        full_precision_mixture_error=precision_error, candidate_order_equal=True,
        rounded_weights_equal=True, latest_ic_endpoint=str(history.index[-1].date()),
        full_rank_universe_count=len(final.columns), factor_order=list(full))


def matured_quality(metrics, decisions, lookback=12):
    metrics = metrics.copy()
    # Concatenating a fully censored cohort with observed cohorts can retain
    # object dtype. Missing IC remains NaN, never a zero-quality observation.
    metrics['rankic'] = pd.to_numeric(metrics.rankic, errors='raise')
    rows = []
    for (model, horizon), block in metrics.groupby(['model', 'horizon'], sort=False):
        block = block.sort_values('date', kind='stable')
        for decision in decisions:
            mature = block[block.date.lt(decision) & block.endpoint.notna() & block.endpoint.le(decision)]
            observable = mature[np.isfinite(mature.rankic)]
            use = observable.tail(lookback)
            rows.append(dict(date=decision, phase=phase(decision), model=model, horizon=int(horizon),
                             matured_cohorts=len(mature), observable_cohorts=len(observable),
                             used_cohorts=len(use), actual_rankic_mean=finite_mean(use.rankic),
                             oldest_signal_date=use.date.iloc[0] if len(use) else None,
                             latest_signal_date=use.date.iloc[-1] if len(use) else None,
                             oldest_endpoint=use.endpoint.min() if len(use) else None,
                             latest_endpoint=use.endpoint.max() if len(use) else None,
                             unavailable_rankic_reasons=json.dumps(mature.loc[~np.isfinite(mature.rankic),
                                                   'rankic_status'].value_counts().to_dict(), sort_keys=True),
                             history_status='complete_12_cohort_history' if len(use) == lookback else
                                            'insufficient_12_cohort_history',
                             history_scope='archived_signal_cohorts_from_2020_only'))
    return pd.DataFrame(rows)


def matched_factor_diagnostics(panel, specs, horizons=HORIZONS):
    """Same finite factor/expert population and outcome rows for aggregation contrast."""
    factor_names = [c.removeprefix('factor_') for c in panel if c.startswith('factor_')]
    columns = ['factor_' + f for f in factor_names] + ['score_' + e for e in specs]
    if 'score_mixture' in panel:
        columns.append('score_mixture')
    common = np.isfinite(panel[columns]).all(axis=1)
    factor_rows, expert_rows = [], []
    for horizon in horizons:
        observed = common & panel[f'coverage_{horizon}'].eq('observed') & np.isfinite(panel[f'return_{horizon}'])
        value = panel.loc[observed, f'return_{horizon}']
        residual = panel.loc[observed, f'residual_{horizon}']
        common_fields = dict(date=str(panel.date.iloc[0]), phase=phase(panel.date.iloc[0]), horizon=horizon,
                             requested_count=len(panel), finite_count=int(common.sum()), observed_count=int(observed.sum()))
        values = {}
        for factor in factor_names:
            ic, status = rank_stat(panel.loc[observed, 'factor_' + factor], value)
            residual_ic, residual_status = rank_stat(panel.loc[observed, 'factor_' + factor], residual)
            values[factor] = ic
            factor_rows.append(dict(**common_fields, factor=factor, rankic=ic, rankic_status=status,
                                    residual_rankic=residual_ic, residual_rankic_status=residual_status))
        for expert, group in specs.items():
            group = [f for f in (group or factor_names) if f in factor_names]
            series = pd.Series([values[f] for f in group], dtype=float)
            group_mean = finite_mean(series)
            expert_ic, status = rank_stat(panel.loc[observed, 'score_' + expert], value)
            expert_rows.append(dict(**common_fields, expert=expert, factor_count=len(group),
                                    finite_factor_ics=int(np.isfinite(series).sum()),
                                    group_factor_rankic_mean=group_mean, expert_rankic=expert_ic,
                                    expert_rankic_status=status,
                                    expert_minus_factor_mean=difference(expert_ic, group_mean)))
    return pd.DataFrame(factor_rows), pd.DataFrame(expert_rows)


def complete_membership(scored, requested, horizons=HORIZONS):
    keys = ['date', 'instrument_id']
    if scored.duplicated(keys).any() or requested.duplicated(keys).any():
        raise ValueError('duplicate_membership')
    left, right = scored.set_index(keys), requested.set_index(keys)
    shared = left.index.intersection(right.index)
    for column in ['selected', 'held_weight', 'sector'] + [f'{prefix}_{h}' for h in horizons for prefix in ('return', 'coverage')]:
        a, b = left.loc[shared, column], right.loc[shared, column]
        if pd.api.types.is_numeric_dtype(a) and column != 'selected':
            equal = np.isclose(a.astype(float), b.astype(float), equal_nan=True, rtol=0., atol=1e-12)
        else:
            equal = a.eq(b) | (a.isna() & b.isna())
        if not np.all(equal):
            raise ValueError('requested_member_evidence_conflict:' + column)
    expected = left.index[left.selected | left.held_weight.gt(0)]
    if not expected.isin(right.index).all():
        raise ValueError('requested_membership_missing_scored_selection_or_holding')
    left['in_scored'] = True
    extra = right.loc[~right.index.isin(left.index)].copy()
    extra['in_scored'] = False
    extra['in_expert_union'] = False
    extra['quality_pass'] = None
    extra['financial_covered'] = None
    extra['financial_entry_present'] = False
    out = pd.concat([left, extra], axis=0).reset_index()
    out['financial_status'] = 'unknown'
    union = out.in_expert_union.eq(True)
    covered = out.financial_covered.eq(True)
    out.loc[union & covered & out.quality_pass.eq(True), 'financial_status'] = 'pass'
    out.loc[union & covered & out.quality_pass.eq(False), 'financial_status'] = 'known_fail'
    # Baseline never changes when adding requested names or filtering a stage/model.
    for horizon in horizons:
        observed = scored[f'coverage_{horizon}'].eq('observed') & np.isfinite(scored[f'return_{horizon}'])
        peers = scored.loc[observed].groupby(['date', 'sector'])[f'return_{horizon}'].agg(['mean', 'count'])
        peers = peers.rename(columns={'mean': f'sector_peer_mean_{horizon}', 'count': f'sector_peer_count_{horizon}'})
        out = out.merge(peers, left_on=['date', 'sector'], right_index=True, how='left', validate='many_to_one')
        out[f'residual_{horizon}'] = out[f'return_{horizon}'] - out[f'sector_peer_mean_{horizon}']
    return out


def paired_transition(panel, parent, child, edge, horizons=HORIZONS):
    rows = []
    for day, block in panel.groupby('date', sort=True):
        p, c = parent.reindex(block.index), child.reindex(block.index)
        kept, removed = p & c, p & ~c
        for horizon in horizons:
            observed = block[f'coverage_{horizon}'].eq('observed') & np.isfinite(block[f'return_{horizon}'])
            ko, ro = kept & observed, removed & observed
            paired = bool(ko.any() and ro.any())
            km = finite_mean(block.loc[ko, f'return_{horizon}'])
            rm = finite_mean(block.loc[ro, f'return_{horizon}'])
            kr = finite_mean(block.loc[ko, f'residual_{horizon}'])
            rr = finite_mean(block.loc[ro, f'residual_{horizon}'])
            rows.append(dict(date=day, phase=phase(day), edge=edge, horizon=horizon,
                             parent_count=int(p.sum()), kept_count=int(kept.sum()), removed_count=int(removed.sum()),
                             child_outside_parent_count=int((c & ~p).sum()),
                             retention=float(kept.sum() / p.sum()) if p.any() else None,
                             kept_observed_count=int(ko.sum()), removed_observed_count=int(ro.sum()),
                             kept_missing_reasons=json.dumps(block.loc[kept & ~observed, f'coverage_{horizon}']
                                                            .value_counts().to_dict(), sort_keys=True),
                             removed_missing_reasons=json.dumps(block.loc[removed & ~observed, f'coverage_{horizon}']
                                                               .value_counts().to_dict(), sort_keys=True),
                             kept_return_mean=km, removed_return_mean=rm,
                             kept_residual_mean=kr, removed_residual_mean=rr,
                             paired_observable=paired,
                             kept_minus_removed=difference(km, rm) if paired else None,
                             kept_minus_removed_residual=difference(kr, rr) if paired else None))
    return pd.DataFrame(rows)


def funnel_diagnostics(panel, horizons=HORIZONS):
    stages = dict(eligible=panel.in_scored.eq(True), expert_union=panel.in_expert_union.eq(True),
                  quality_pass=panel.financial_status.eq('pass'), selected=panel.selected.eq(True),
                  selected_quality_pass=panel.selected.eq(True) & panel.financial_status.eq('pass'),
                  selected_known_fail=panel.selected.eq(True) & panel.financial_status.eq('known_fail'),
                  selected_unknown=panel.selected.eq(True) & panel.financial_status.eq('unknown'),
                  held=panel.held_weight.gt(0))
    rows = []
    for day, block in panel.groupby('date', sort=True):
        for stage, members in stages.items():
            sub = block[members.reindex(block.index)]
            metrics, _ = model_diagnostics(sub, ('mixture',), horizons)
            for row in records(metrics):
                horizon = row['horizon']
                observed = sub[f'coverage_{horizon}'].eq('observed') & np.isfinite(sub[f'return_{horizon}'])
                row.update(date=day, phase=phase(day), stage=stage, future_observed_all=int(observed.sum()),
                           outside_scored_count=int((~sub.in_scored.eq(True)).sum()),
                           financial_pass_count=int(sub.financial_status.eq('pass').sum()),
                           financial_fail_count=int(sub.financial_status.eq('known_fail').sum()),
                           financial_unknown_count=int(sub.financial_status.eq('unknown').sum()),
                           all_member_return_mean=finite_mean(sub.loc[observed, f'return_{horizon}']),
                           all_member_residual_mean=finite_mean(sub.loc[observed, f'residual_{horizon}']),
                           all_member_missing_reasons=json.dumps(sub.loc[~observed, f'coverage_{horizon}']
                                                                .value_counts().to_dict(), sort_keys=True))
                rows.append(row)
    transitions = []
    for parent, child in [('eligible', 'expert_union'), ('expert_union', 'quality_pass'), ('quality_pass', 'selected')]:
        transitions.append(paired_transition(panel, stages[parent], stages[child], parent + '_to_' + child, horizons))
    return pd.DataFrame(rows), pd.concat(transitions, ignore_index=True)


def summarize_dates(frame, group_columns, metric_columns):
    rows = []
    for key, grouped in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        identity = dict(zip(group_columns, key))
        for period in ('overall', *PHASES):
            block = grouped if period == 'overall' else grouped[grouped.phase.eq(period)]
            for metric in metric_columns:
                values = pd.to_numeric(block[metric], errors='coerce')
                finite = values[np.isfinite(values)]
                rows.append(dict(**identity, phase=period, metric=metric,
                                 requested_dates=len(block), valid_dates=len(finite),
                                 missing_dates=len(block) - len(finite), mean=finite_mean(finite),
                                 median=float(finite.median()) if len(finite) else None,
                                 positive_frequency=float(finite.gt(0).mean()) if len(finite) else None))
    return pd.DataFrame(rows)


def write_manifest(output):
    files = {path.name: sha(path) for path in sorted(output.iterdir())
             if path.is_file() and path.name != 'manifest.json'}
    atomic_json(output / 'manifest.json', dict(algorithm='sha256', sha256=files))


def execute_attempt(output, protocol, operation):
    output = Path(output)
    start_attempt(output, protocol)
    try:
        summary = operation(output)
        atomic_json(output / 'summary.json', summary)
        atomic_json(output / 'status.json', dict(status='complete', completed_at_utc=utc_now(),
                                               promotion_approved=False, account_replayed=False))
        write_manifest(output)
        return summary
    except Exception as error:
        atomic_json(output / 'status.json', dict(status='failed', failed_at_utc=utc_now(),
                                               failure_type=type(error).__name__, failure_reason=str(error),
                                               promotion_approved=False, account_replayed=False))
        write_manifest(output)
        raise


def registered_protocol():
    return dict(
        created_at_utc=utc_now(), question='Q2-a: Which original expert, mixture, or archived filter loses ranking evidence?',
        hypothesis='Persistent lowvol proxy dominance need not imply actual expert ranking increment; a filter may change the observable opportunity mix.',
        decision_change='Identify at most one component for a separately approved single-variable protocol, or close the original-scoring-plus-allocation route. No winner or policy is adopted here.',
        frozen_source=dict(path=str(FROZEN), commit=FROZEN_COMMIT),
        anchored_sources=dict(attribution_summary_sha256=SUMMARY_SHA, main_result_sha256=RESULT_SHA),
        script_sha256=sha(Path(__file__)), test_sha256=sha(ROOT / 'tests/test_score_mechanism_audit.py'),
        reused_helper_sha256=sha(ROOT / 'scripts/continuity_attribution.py'),
        source_resolution='All original5 and final12 parquet hashes, main archive hashes, data manifest/calendar are verified through the anchored attribution summary before use; identities saved in verified_sources.json.',
        account=dict(capital_rmb=600000, frozen_positions=20, user_position_ceiling=50,
                     long_only=True, leverage=False, risk_policy='P2', candidate='quality_pb',
                     costs='unchanged archived costs; future outcome metrics do not include execution costs'),
        models=list(MODELS), horizons=list(HORIZONS), phases=list(PHASES), decisions=74,
        metrics=['RankIC', 'Q5_minus_Q3', 'sector_peer_residual_RankIC', 'sector_peer_residual_Q5_minus_Q3',
                 'factor_IC_group_mean_vs_actual_expert_IC', 'original_weight_proxy',
                 'asof_last12_observable_expert_IC', 'paired_kept_minus_removed', 'coverage'],
        reconstruction=dict(tolerance=1e-10, require_all_scores=True, require_ordered_candidate_lists=True,
                            ranking_before_eligibility=True, weights='full precision original; compare stored rounded values'),
        sample_rule='Per date, all6 model scores finite before outcomes; freeze tie-preserving average-rank quintiles before future coverage. Same horizon-valid rows for all model comparisons.',
        matched_rule='All11 factors and all6 expert/mixture scores finite on the same eligible rows; same observed horizon subset for each factor and expert.',
        sector_baseline='Mean of horizon-valid saved eligible scored peers per date and sector, fixed for all models and stages; not full sector neutralization.',
        metric_validity='RankIC requires at least10 finite pairs and variation. Q5-Q3 requires at least10 observed stage/model pairs and both quintiles; unavailable remains null.',
        asof_rule='Only earlier signal dates with horizon endpoint <= decision; last12 finite expert RankIC cohorts; retain early insufficient history and 2020 archive truncation.',
        funnel_rule='eligible -> union -> archived passes -> intersection with selected; selected outside passes remains separate. covered=false is unavailable/unknown, not observed economic failure.',
        aggregation='Date-equal means, medians, direction frequencies overall and phases. Paired transitions require both kept and removed observed on the same date.',
        limitations=['Descriptive same-sample conditional research, not independent OOS or a causal intervention.',
                     '74monthly dates and overlapping20/60-session windows are not independent observations.',
                     'Future adjusted-feature-price outcomes are not executable counterfactuals or account returns; no annualization.',
                     'Original IC uses full historical factor cross-sections and monthly endpoints; actual cohorts start2020 and use eligible20/60-session outcomes.',
                     'Data quality and adoption gates remain failed or unverified; no automatic trading or new weight policy.'],
        account_replayed=False, new_prices_loaded=False, promotion_approved=False)


def verified_inputs():
    summary_path = ATTRIBUTION / 'attribution_final/summary.json'
    expected = {summary_path: SUMMARY_SHA}
    verify_hashes(expected)
    summary = json.loads(summary_path.read_text())
    original = ROOT / summary['original_parquets']['path']
    expected.update({original / name: digest for name, digest in summary['original_parquets']['sha256'].items()})
    final = ATTRIBUTION / 'attribution_final'
    expected.update({final / name: digest for name, digest in summary['artifacts'].items()})
    archive = Path(summary['source']['directory'])
    if summary['source']['verified_sha256']['result.pkl'] != RESULT_SHA:
        raise ValueError('main_result_identity_mismatch')
    expected.update({archive / name: digest for name, digest in summary['source']['verified_sha256'].items()})
    manifest_path = _attribution.MANIFEST
    expected[manifest_path] = summary['manifest_sha256']
    expected[ROOT / 'scripts/continuity_attribution.py'] = summary['script_sha256']
    verified = verify_hashes(expected)
    manifest = json.loads(manifest_path.read_text())
    calendar_path = manifest_path.parent / manifest['calendar']['path']
    verified.update(verify_hashes({calendar_path: manifest['calendar']['sha256']}))
    commit = subprocess.check_output(['git', '-C', str(FROZEN), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(FROZEN), 'status', '--porcelain'], text=True).strip()
    if commit != FROZEN_COMMIT or dirty:
        raise ValueError('frozen_source_identity_or_cleanliness_mismatch')
    frozen_files = ['quantlab/config.py', 'quantlab/strategy/scoring.py', 'quantlab/strategy/experts.py',
                    'quantlab/strategy/decision.py', 'quantlab/strategy/candidates.py']
    frozen_hashes = {name: sha(FROZEN / name) for name in frozen_files}
    archived_manifest = json.loads((archive / 'artifacts.json').read_text())
    if archived_manifest['sha256'] != summary['source']['verified_sha256']:
        raise ValueError('archive_manifest_conflict')
    verified[str(archive / 'artifacts.json')] = sha(archive / 'artifacts.json')
    load_frozen_api()
    with (archive / 'result.pkl').open('rb') as stream:
        run = pickle.load(stream)
    scored = pd.read_parquet(original / 'scored_outcomes.parquet')
    requested = pd.read_parquet(final / 'complete_requested_outcomes.parquet')
    calendar = pd.DatetimeIndex(pd.read_parquet(calendar_path).session)
    calendar = calendar[calendar <= pd.Timestamp('2026-06-30')]
    archived_calendar = pd.DatetimeIndex(run['calendar'])
    archived_calendar = archived_calendar[archived_calendar <= pd.Timestamp('2026-06-30')]
    if not calendar.equals(archived_calendar):
        raise ValueError('frozen_calendar_mismatch')
    identity = dict(sha256=verified, frozen_commit=commit, frozen_clean=True, frozen_file_sha256=frozen_hashes,
                    data_version=run['data_version'], archive_status=run['status'],
                    source_count=len(verified), calendar_end=str(calendar[-1].date()))
    return run, scored, requested, calendar, identity


def horizon_endpoint(calendar, day, horizon):
    position = calendar.get_indexer([pd.Timestamp(day)])[0]
    if position < 0:
        raise ValueError('decision_not_in_frozen_calendar:' + day)
    return str(calendar[position + horizon].date()) if position + horizon < len(calendar) else None


def analyze_archives(output):
    run, scored, requested, calendar, identity = verified_inputs()
    atomic_json(output / 'verified_sources.json', identity)
    if scored.duplicated(['date', 'instrument_id']).any() or requested.duplicated(['date', 'instrument_id']).any():
        raise ValueError('duplicate_archived_membership')
    traces = [t for t in run['signal_trace'] if 'standardized_features' in t and 'selected' in t]
    dates = sorted(t['date'] for t in traces)
    if len(traces) != 74 or len(set(dates)) != 74 or set(dates) != set(scored.date.unique()):
        raise ValueError('expected_exactly74_saved_decisions')
    if set(requested.date.unique()) != set(dates):
        raise ValueError('requested_dates_mismatch')
    for horizon in HORIZONS:
        for label, frame in [('scored', scored), ('requested', requested)]:
            if not frame[f'coverage_{horizon}'].eq('observed').equals(np.isfinite(frame[f'return_{horizon}'])):
                raise ValueError('outcome_coverage_inconsistent:' + label)
    cfg = SimpleNamespace(**run['config']['factors'])
    if cfg.expert_ic_lookback != 12:
        raise ValueError('unexpected_expert_lookback')
    top_k = sum([run['config']['capital']['a_share_positions'],
                 run['config']['portfolio']['n_candidates_pad'], run['config']['portfolio']['turnover_rank_thresh']])
    if top_k != 28:
        raise ValueError('unexpected_candidate_count')
    panels, weights, contributions, checks = [], [], [], []
    for index, trace in enumerate(sorted(traces, key=lambda t: t['date'])):
        day = trace['date']
        saved = scored[scored.date.eq(day)]
        union = set(c for values in trace['expert_candidates'].values() for c in values)
        if set(saved.loc[saved.in_expert_union, 'instrument_id']) != union:
            raise ValueError('archived_union_membership_mismatch:' + day)
        entries = trace['candidate_evidence']['entries']
        for member in saved[saved.in_expert_union].itertuples():
            if bool(entries.get(member.instrument_id, {}).get('passes')) != member.quality_pass:
                raise ValueError('archived_quality_membership_mismatch:' + day)
        if set(requested.loc[requested.date.eq(day) & requested.selected, 'instrument_id']) != set(trace['selected']):
            raise ValueError('selected_request_trace_mismatch:' + day)
        panel, proxy, factors, check = reconstruct_trace(trace, run['ic_matrix'], saved, cfg, top_k)
        panels.append(panel); weights.append(proxy); contributions.append(factors); checks.append(check)
        atomic_json(output / 'reconstruction_checks.json', dict(complete=False, expected_decisions=74, checks=checks))
        if (index + 1) % 10 == 0 or index + 1 == 74:
            print(f'reconstructed {index + 1}/74 archived score dates', flush=True)
    atomic_json(output / 'reconstruction_checks.json', dict(complete=True, expected_decisions=74, checks=checks))
    print('all score, full-precision weight, and ordered candidate gates passed; computing offline diagnostics', flush=True)
    recovered = pd.concat(panels, ignore_index=True)
    proxy = pd.concat(weights, ignore_index=True)
    contribution = pd.concat(contributions, ignore_index=True)
    members = complete_membership(recovered, requested)
    model_metrics, scored_panels, factor_metrics, matched_metrics = [], [], [], []
    experts, _ = load_frozen_api()
    for day, block in members[members.in_scored].groupby('date', sort=True):
        metrics, signal = model_diagnostics(block)
        metrics['endpoint'] = [horizon_endpoint(calendar, day, h) for h in metrics.horizon]
        for horizon in HORIZONS:
            if horizon_endpoint(calendar, day, horizon) is None and block[f'coverage_{horizon}'].eq('observed').any():
                raise ValueError('observed_after_calendar_end:' + day)
        factors, matched = matched_factor_diagnostics(block, experts.EXPERT_SPECS)
        model_metrics.append(metrics); scored_panels.append(signal)
        factor_metrics.append(factors); matched_metrics.append(matched)
    metrics = pd.concat(model_metrics, ignore_index=True)
    scores = pd.concat(scored_panels, ignore_index=True)
    actual_history = matured_quality(metrics, dates)
    factor_metrics = pd.concat(factor_metrics, ignore_index=True)
    matched_metrics = pd.concat(matched_metrics, ignore_index=True)
    stages, transitions = funnel_diagnostics(members)
    quality_comparison = proxy.merge(metrics.rename(columns={'model': 'expert'}), on=['date', 'phase', 'expert'],
                                     how='left', validate='one_to_many', suffixes=('', '_actual'))
    quality_comparison = quality_comparison.merge(actual_history.rename(columns={'model': 'expert'}),
                                                  on=['date', 'phase', 'expert', 'horizon'], how='left', validate='one_to_one')
    quality_comparison['proxy_vs_actual_scope'] = 'Proxy: monthly full-factor historical cross-section. Actual: saved eligible20/60-session cohorts from2020; not a deployable expert-quality validation.'
    model_summary = summarize_dates(metrics, ['model', 'horizon'],
                                    ['rankic', 'residual_rankic', 'q5_mean', 'q3_mean', 'q5_minus_q3', 'q5_minus_q3_residual'])
    stage_summary = summarize_dates(stages, ['stage', 'horizon'],
                                    ['all_member_return_mean', 'all_member_residual_mean', 'rankic', 'residual_rankic',
                                     'q5_minus_q3', 'q5_minus_q3_residual', 'requested_count', 'future_observed_all'])
    transition_summary = summarize_dates(transitions, ['edge', 'horizon'],
                                         ['kept_minus_removed', 'kept_minus_removed_residual', 'retention',
                                          'kept_count', 'removed_count', 'child_outside_parent_count'])
    weight_summary = summarize_dates(proxy, ['expert'], ['signed_proxy', 'clipped_q', 'full_precision_weight'])
    factor_summary = summarize_dates(contribution, ['expert', 'factor'], ['factor_ic_mean', 'contribution_to_group_mean'])
    matched_summary = summarize_dates(matched_metrics, ['expert', 'horizon'],
                                      ['group_factor_rankic_mean', 'expert_rankic', 'expert_minus_factor_mean'])
    coverage_columns = ['date', 'phase', 'model', 'horizon', 'requested_count', 'finite_score_count',
                        'missing_model_score_count', 'excluded_other_model_score_count', 'common_finite_score_count',
                        'future_observed_count', 'future_missing_count', 'missing_reasons', 'rankic_status',
                        'residual_rankic_status', 'quintile_status', 'q3_requested', 'q3_observed', 'q5_requested', 'q5_observed']
    selected = members[members.selected]
    exceptions = members[(members.selected | members.held_weight.gt(0)) &
                         (~members.in_scored | ~members.financial_status.eq('pass'))]
    candidate_rows = [dict(date=t['date'], expert=e, position=i + 1, instrument_id=c)
                      for t in traces for e, values in t['expert_candidates'].items() for i, c in enumerate(values)]
    artifacts = dict(score_rank_membership=scores, complete_funnel_membership=members,
                     model_date_metrics=metrics, model_phase_summary=model_summary,
                     weight_proxy=proxy, weight_factor_contributions=contribution, weight_phase_summary=weight_summary,
                     proxy_factor_phase_summary=factor_summary, matured_expert_quality=actual_history,
                     proxy_vs_actual_quality=quality_comparison, matched_factor_date_metrics=factor_metrics,
                     matched_expert_date_metrics=matched_metrics, matched_phase_summary=matched_summary,
                     funnel_stage_metrics=stages, funnel_phase_summary=stage_summary,
                     funnel_transitions=transitions, funnel_transition_summary=transition_summary,
                     model_coverage=metrics[coverage_columns], requested_membership_exceptions=exceptions,
                     archived_expert_candidates=pd.DataFrame(candidate_rows))
    hashes = {}
    for name, frame in artifacts.items():
        path = output / (name + '.parquet')
        frame.to_parquet(path, index=False)
        hashes[path.name] = sha(path)
    leader = proxy.loc[proxy.groupby('date').full_precision_weight.idxmax()]
    summary = dict(created_at_utc=utc_now(), status='complete_conditional_offline_diagnostic',
                   promotion_approved=False, account_replayed=False, new_prices_loaded=False,
                   source_identity=identity, decisions=len(traces), scored_rows=len(scored),
                   full_requested_members=len(requested), complete_funnel_rows=len(members),
                   reconstruction=dict(all_passed=True, decisions=len(checks),
                                       max_score_error=max(c['max_score_error'] for c in checks),
                                       max_saved_eligible_error=max(c['max_saved_eligible_error'] for c in checks),
                                       max_full_precision_mixture_error=max(c['full_precision_mixture_error'] for c in checks),
                                       candidate_order_equal=True, rounded_weights_equal=True),
                   lowvol_dominance=dict(leader_counts=leader.expert.value_counts().to_dict(),
                                        leader_switches=int(leader.sort_values('date').expert.ne(
                                            leader.sort_values('date').expert.shift()).sum() - 1)),
                   membership=dict(selected_count=len(selected), selected_outside_scored=int((~selected.in_scored).sum()),
                                   selected_financial_states=selected.financial_status.value_counts().to_dict(),
                                   selected_outside_union=int((~selected.in_expert_union).sum()),
                                   selected_outside_pass=int((~selected.financial_status.eq('pass')).sum()),
                                   held_count=int(members.held_weight.gt(0).sum()),
                                   held_outside_scored=int((members.held_weight.gt(0) & ~members.in_scored).sum())),
                   model_phase_summary=records(model_summary), weight_phase_summary=records(weight_summary),
                   matched_phase_summary=records(matched_summary), funnel_transition_summary=records(transition_summary),
                   coverage_by_horizon=records(metrics.groupby(['model', 'horizon'], as_index=False)[
                       ['requested_count', 'common_finite_score_count', 'future_observed_count', 'future_missing_count']].sum()),
                   metric_unavailability=records(metrics.groupby(['model', 'horizon', 'rankic_status']).size().rename('dates').reset_index()),
                   scope_limits=registered_protocol()['limitations'], artifacts=hashes)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=TASK / 'run1')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(TASK.resolve()) or args.output.resolve() == TASK.resolve():
        raise ValueError('output_must_be_new_attempt_under_task_directory')
    protocol = registered_protocol()
    summary = execute_attempt(args.output, protocol, analyze_archives)
    print(json.dumps(dict(status=summary['status'], decisions=summary['decisions'],
                          reconstruction=summary['reconstruction'], membership=summary['membership'],
                          promotion_approved=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
