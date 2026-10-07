"""Fixed Q2-b offline candidate comparison; never invokes an account engine."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
import pandas as pd

KEY = ['date', 'instrument_id']
HORIZONS = (20, 60)
PHASES = ('2020-21', '2022-23', '2024-25', '2026H1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_hashes(expected):
    checked = {}
    for path, wanted in expected.items():
        actual = sha(path)
        if actual != wanted:
            raise ValueError('source_hash_mismatch:' + str(path))
        checked[str(path)] = actual
    return checked


def execute_attempt(output, protocol, operation):
    output = Path(output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError('attempt_exists:' + str(output)) from error
    atomic_json(output / 'protocol.json', protocol)
    try:
        atomic_json(output / 'status.json', dict(status='running', started_at_utc=utc_now(), promotion_approved=False))
        verified = verify_hashes(protocol['sources'])
        atomic_json(output / 'source_checks.json', dict(status='hashes_verified', sha256=verified))
        result = operation(output)
        verify_hashes(protocol['sources'])
        atomic_json(output / 'summary.json', result)
        atomic_json(output / 'status.json', dict(status='complete', completed_at_utc=utc_now(), promotion_approved=False))
        return result
    except Exception as error:
        atomic_json(output / 'status.json', dict(status='technical_failure', completed_at_utc=utc_now(),
                                                error_type=type(error).__name__, error=str(error), promotion_approved=False))
        raise
    finally:
        atomic_json(output / 'manifest.json', dict(algorithm='sha256', sha256={
            p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'}))


def phase(day):
    year = int(str(day)[:4])
    if not 2020 <= year <= 2026 or (year == 2026 and str(day) > '2026-06-30'):
        raise ValueError('out_of_phase_date:' + str(day))
    return PHASES[min((year - 2020) // 2, 3)]


def construct_membership(scored, archived, expected_dates=74):
    """Selection reads only date, symbol, original membership and frozen score."""
    if scored[KEY].isna().any().any() or scored.duplicated(KEY).any():
        raise ValueError('duplicate_key_or_missing_key')
    if not np.isfinite(scored.score_mixture).all():
        raise ValueError('nonfinite_score')
    if scored.date.nunique() != expected_dates:
        raise ValueError('date_count')
    actual = set(scored.loc[scored.in_expert_union, KEY].itertuples(index=False, name=None))
    expected = set(archived[KEY].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError('union_mismatch')
    selected, date_rows = [], []
    for day, block in scored.groupby('date', sort=True):
        selection = block[KEY + ['score_mixture', 'in_expert_union']].copy()
        a = set(selection.loc[selection.in_expert_union, 'instrument_id'])
        if not 0 < len(a) <= len(block):
            raise ValueError('invalid_daily_union_size')
        ordered = selection.sort_values(['score_mixture', 'instrument_id'], ascending=[False, True])
        top = ordered.head(len(a))
        b = set(top.instrument_id)
        boundary = float(top.score_mixture.iloc[-1])
        tied = ordered[ordered.score_mixture.eq(boundary)]
        date_rows.append(dict(date=day, eligible_count=len(block), A_count=len(a), B_count=len(b),
                              common_count=len(a & b), boundary_score=boundary,
                              boundary_tie_count=len(tied),
                              boundary_tie_selected=sorted(set(tied.instrument_id) & b)))
        members = block[block.instrument_id.isin(a | b)].copy()
        members['in_A'] = members.instrument_id.isin(a)
        members['in_B'] = members.instrument_id.isin(b)
        members['membership_group'] = np.where(members.in_A & members.in_B, 'common',
                                              np.where(members.in_A, 'A_only', 'B_only'))
        selected.append(members)
    return (pd.concat(selected).sort_values(KEY).reset_index(drop=True), pd.DataFrame(date_rows))


def membership_groups(block):
    return {'A': block[block.in_A], 'B': block[block.in_B],
            **{name: block[block.membership_group.eq(name)]
               for name in ('common', 'A_only', 'B_only')}}


def attach_financial(scored, archived):
    if archived[KEY].isna().any().any() or archived.duplicated(KEY).any():
        raise ValueError('duplicate_key_or_missing_key:financial_archive')
    if not archived.financial_status.isin(['pass', 'known_fail', 'unknown']).all():
        raise ValueError('invalid_financial_status')
    if archived.loc[~archived.in_expert_union, 'financial_status'].ne('unknown').any():
        raise ValueError('financial_outside_union')
    left, right = scored.set_index(KEY), archived.set_index(KEY)
    if not left.index.isin(right.index).all():
        raise ValueError('missing_archived_members')
    right = right.reindex(left.index)
    for column in left.columns.intersection(right.columns):
        equal = left[column].eq(right[column]) | (left[column].isna() & right[column].isna())
        if not equal.fillna(False).all():
            raise ValueError('shared_field_mismatch:' + column)
    result = scored.copy()
    result['financial_status'] = right.financial_status.to_numpy()
    return result


def validate_source_metadata(summary, checks, scored, expected_dates=74, expected_rows=292532):
    if summary.get('status') != 'complete_conditional_offline_diagnostic' or summary.get('promotion_approved') is not False:
        raise ValueError('source_not_approved_for_diagnostic')
    reconstruction = summary['reconstruction']
    if (summary['decisions'] != expected_dates or len(scored) != expected_rows or
            summary['scored_rows'] != expected_rows or scored.date.nunique() != expected_dates or
            reconstruction['all_passed'] is not True or reconstruction['decisions'] != expected_dates or
            checks['complete'] is not True or checks['expected_decisions'] != expected_dates or
            len(checks['checks']) != expected_dates):
        raise ValueError('incomplete_reconstruction')
    if (any(reconstruction[key] != 0 for key in ('max_score_error', 'max_saved_eligible_error', 'max_full_precision_mixture_error')) or
            reconstruction['candidate_order_equal'] is not True or reconstruction['rounded_weights_equal'] is not True):
        raise ValueError('failed_reconstruction')
    dates = set()
    for check in checks['checks']:
        day = check['date']
        block = scored[scored.date.eq(day)]
        if (day in dates or len(block) != check['scored_count'] or int(block.in_expert_union.sum()) != check['union_count'] or
                any(check[key] != 0 for key in ('max_score_error', 'max_saved_eligible_error', 'full_precision_mixture_error')) or
                check['candidate_order_equal'] is not True or check['rounded_weights_equal'] is not True):
            raise ValueError('failed_daily_reconstruction:' + day)
        dates.add(day)
    if dates != set(scored.date):
        raise ValueError('reconstruction_dates_mismatch')


def measure(members, horizons=HORIZONS):
    metric_rows, pair_rows, coverage_rows = [], [], []
    for day, block in members.groupby('date', sort=True):
        for horizon in horizons:
            group_metrics = {}
            for group, subset in membership_groups(block).items():
                raw = subset[f'coverage_{horizon}'].eq('observed') & np.isfinite(subset[f'return_{horizon}'])
                residual = raw & np.isfinite(subset[f'residual_{horizon}'])
                row = dict(date=day, phase=phase(day), horizon=horizon, group=group,
                           requested_count=len(subset), raw_count=int(raw.sum()),
                           residual_count=int(residual.sum()),
                           raw_mean=float(subset.loc[raw, f'return_{horizon}'].mean()) if raw.any() else np.nan,
                           residual_mean=float(subset.loc[residual, f'residual_{horizon}'].mean()) if residual.any() else np.nan)
                group_metrics[group] = row
                metric_rows.append(row)
                for kind, mask in [('raw', raw), ('residual', residual)]:
                    reasons = subset[f'coverage_{horizon}'].fillna('missing_coverage_label').copy()
                    reasons.loc[reasons.eq('observed') & ~raw] = 'nonfinite_raw_outcome'
                    if kind == 'residual':
                        reasons.loc[raw & ~residual] = 'unavailable_sector_residual'
                    reasons.loc[mask] = 'observed'
                    for reason, count in reasons.value_counts().items():
                        coverage_rows.append(dict(date=day, horizon=horizon, group=group,
                                                  measure=kind, reason=reason, count=int(count)))
            a, b = group_metrics['A'], group_metrics['B']
            pair = dict(date=day, phase=phase(day), horizon=horizon,
                        A_requested=a['requested_count'], B_requested=b['requested_count'])
            for kind in ('raw', 'residual'):
                valid = a[f'{kind}_count'] > 0 and b[f'{kind}_count'] > 0
                pair.update({f'A_{kind}_count': a[f'{kind}_count'], f'B_{kind}_count': b[f'{kind}_count'],
                             f'{kind}_paired': valid,
                             f'{kind}_complete': valid and a[f'{kind}_count'] == a['requested_count'] and b[f'{kind}_count'] == b['requested_count'],
                             f'A_{kind}_mean': a[f'{kind}_mean'], f'B_{kind}_mean': b[f'{kind}_mean'],
                             f'{kind}_difference': b[f'{kind}_mean'] - a[f'{kind}_mean'] if valid else np.nan})
            pair_rows.append(pair)
    return pd.DataFrame(metric_rows), pd.DataFrame(pair_rows), pd.DataFrame(coverage_rows)


def summarize(pairs):
    rows = []
    for label in PHASES:
        for horizon in sorted(pairs.horizon.unique()):
            block = pairs[pairs.phase.eq(label) & pairs.horizon.eq(horizon)]
            for kind in ('raw', 'residual'):
                for subset, flag in [('paired', 'paired'), ('complete', 'complete')]:
                    observed = block.loc[block[f'{kind}_{flag}'], f'{kind}_difference'].dropna()
                    rows.append(dict(phase=label, horizon=horizon, measure=kind, subset=subset,
                                     requested_dates=len(block), date_count=len(observed),
                                     mean=float(observed.mean()) if len(observed) else np.nan,
                                     median=float(observed.median()) if len(observed) else np.nan,
                                     positive_frequency=float(observed.gt(0).mean()) if len(observed) else np.nan))
    return pd.DataFrame(rows)


def decide(summary):
    index = pd.MultiIndex.from_product([PHASES[:3], HORIZONS, ('raw', 'residual'), ('paired', 'complete')],
                                      names=['phase', 'horizon', 'measure', 'subset'])
    main = summary.set_index(index.names).reindex(index)
    full = main.xs('paired', level='subset')
    complete = main.xs('complete', level='subset')
    full_missing = ~np.isfinite(full['mean']) | full.date_count.fillna(0).lt(1)
    complete_missing = ~np.isfinite(complete['mean']) | complete.date_count.fillna(0).lt(1)
    full_bad = full['mean'].le(0) & ~full_missing
    complete_bad = complete['mean'].le(0) & ~complete_missing
    if full_bad.any():
        decision = 'not_supported_close_priority'
    elif full_missing.any() or complete_missing.any():
        decision = 'insufficient_evidence'
    elif complete_bad.any():
        decision = 'not_supported_close_priority'
    else:
        decision = 'conditional_signal_only'
    return dict(decision=decision, nonpositive_main_cells=int(full_bad.sum()),
                missing_main_cells=int(full_missing.sum()), missing_complete_cells=int(complete_missing.sum()),
                nonpositive_complete_cells=int(complete_bad.sum()), promotion_approved=False)


def profiles(members):
    rows, finances = [], []
    for day, block in members.groupby('date', sort=True):
        for group, subset in membership_groups(block).items():
            counts = subset.financial_status.value_counts()
            finances.append(dict(date=day, phase=phase(day), group=group, requested_count=len(subset),
                                 **{state: int(counts.get(state, 0)) for state in ('pass', 'known_fail', 'unknown')},
                                 known_fraction=float(subset.financial_status.ne('unknown').mean()) if len(subset) else np.nan))
            dimensions = {'all': pd.Series('all', index=subset.index),
                          'sector': subset.sector.fillna('unknown').replace('', 'unknown'),
                          'bucket_mixture': subset.bucket_mixture.map(lambda x: str(int(x)) if pd.notna(x) else 'unknown')}
            for dimension, labels in dimensions.items():
                categories = labels.unique() if len(labels) else ['empty']
                for category in sorted(categories):
                    part = subset.loc[labels.eq(category)]
                    rows.append(dict(date=day, phase=phase(day), group=group, dimension=dimension,
                                     category=category, group_count=len(subset), member_count=len(part),
                                     member_share=len(part) / len(subset) if len(subset) else np.nan,
                                     score_mean=float(part.score_mixture.mean()) if len(part) else np.nan,
                                     score_min=float(part.score_mixture.min()) if len(part) else np.nan,
                                     score_max=float(part.score_mixture.max()) if len(part) else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(finances)


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def analyze_archives(output, source, expected_dates=74, expected_rows=292532):
    source = Path(source)
    scored = pd.read_parquet(source / 'score_rank_membership.parquet')
    archived = pd.read_parquet(source / 'complete_funnel_membership.parquet')
    expert = pd.read_parquet(source / 'archived_expert_candidates.parquet')
    source_summary = json.loads((source / 'summary.json').read_text())
    reconstruction = json.loads((source / 'reconstruction_checks.json').read_text())
    validate_source_metadata(source_summary, reconstruction, scored, expected_dates, expected_rows)
    scored = attach_financial(scored, archived)
    columns = KEY + ['score_mixture', 'in_expert_union', 'sector', 'bucket_mixture', 'financial_status']
    columns += [f'{label}_{h}' for h in HORIZONS for label in ('return', 'coverage', 'residual')]
    columns += [c for c in ('sector_peer_mean_20', 'sector_peer_count_20', 'sector_peer_mean_60',
                           'sector_peer_count_60', 'eligible_bucket') if c in scored]
    members, dates = construct_membership(scored[columns], expert, expected_dates)
    # This write precedes all outcome aggregation; the selection function has no outcome dependency.
    members.to_parquet(output / 'membership.parquet', index=False)
    dates.to_parquet(output / 'date_membership.parquet', index=False)
    metrics, pairs, coverage = measure(members)
    phases = summarize(pairs)
    distributions, financial = profiles(members)
    for name, frame in [('date_metrics', metrics), ('paired_differences', pairs), ('coverage_reasons', coverage),
                        ('phase_summary', phases), ('sector_score_profiles', distributions), ('financial_coverage', financial)]:
        frame.to_parquet(output / (name + '.parquet'), index=False)
    source_checks = json.loads((output / 'source_checks.json').read_text())
    source_checks.update(status='all_checks_passed', dates=expected_dates, eligible_rows=len(scored),
                         reconstruction_all_passed=True, shared_fields_equal=True, archived_union_equal=True,
                         finite_mixture_scores=True, daily_equal_size=bool(dates.A_count.eq(dates.B_count).all()))
    atomic_json(output / 'source_checks.json', source_checks)
    membership_counts = {name: int(members[flag].sum()) for name, flag in [('A', 'in_A'), ('B', 'in_B')]}
    membership_counts.update({name: int(members.membership_group.eq(name).sum()) for name in ('common', 'A_only', 'B_only')})
    return dict(created_at_utc=utc_now(), status='complete_conditional_offline_diagnostic',
                promotion_approved=False, account_replayed=False, new_prices_loaded=False,
                decisions=expected_dates, eligible_rows=len(scored), membership=membership_counts,
                daily_size_min=int(dates.A_count.min()), daily_size_max=int(dates.A_count.max()),
                boundary_multiple_tie_dates=int(dates.boundary_tie_count.gt(1).sum()),
                decision=decide(phases), phase_summary=records(phases),
                coverage_totals=records(metrics[metrics.group.isin(['A', 'B'])].groupby(['group', 'horizon'])[
                    ['requested_count', 'raw_count', 'residual_count']].sum().reset_index()),
                financial_totals=records(financial.groupby('group')[['requested_count', 'pass', 'known_fail', 'unknown']].sum().reset_index()),
                limitations=[
                    'Financial pre-gate candidate diagnostic; unknown archived eligibility remains unknown.',
                    'A/B price means and saved eligible-sector-peer residuals are not account returns or causal alpha.',
                    '74 dates and overlapping 20/60-session outcomes are not independent observations; no annualization or significance claim.',
                    'Same previously examined data, not independent OOS; 2026H1 is supplementary.',
                    'Missing outcomes are never zero-filled or replaced; complete sensitivity fixes the entire preselected A/B cohorts.',
                    'All-positive directions would be descriptive necessary conditions only; one complete date is not stable evidence.',
                    'Data, executable account returns, doubled costs, execution stress and frozen forward-validation gates remain failed or unverified.'])


def registered_protocol(source, anchors, documents):
    source = Path(source)
    sources = verify_hashes(anchors)
    manifest = json.loads((source / 'manifest.json').read_text())['sha256']
    sources.update(verify_hashes({str(source / name): digest for name, digest in manifest.items()}))
    identities = {name: dict(path=str(path), sha256=sha(path)) for name, path in documents.items()}
    sources.update({item['path']: item['sha256'] for item in identities.values()})
    return dict(created_at_utc=utc_now(), question='Q2-b: original expert union versus equal-size original mixture top N',
                hypothesis='Changing only candidate construction may improve observed price and fixed peer-residual means across all three full phases.',
                decision_change='Retain this construction only as a conditional mechanism clue if all frozen direction checks pass; otherwise close its priority or record insufficient coverage. No follow-on scans.',
                frozen_commit='6157c64c84717dc8645dde3fe6228590b9f8538e',
                working_head='0dab5b29455871b0e8b4d001b77cbf2111bc4d7b',
                data_version='5403b0902a7ca7737c9cc11cfc1c2a67e32934a3e5264265713a713b554eb1a2',
                sources=sources, source_anchors=anchors, documents=identities,
                account=dict(capital_rmb=600000, target_net_annualized=.15, research_drawdown_budget=.25,
                             user_max_holdings=50, unchanged_frozen_max_holdings=20, long_only=True, leverage=False,
                             account_engine_used=False),
                inputs=dict(dates=74, eligible_rows=292532, financial='archived only; unknown outside original union'),
                selection=dict(A='original archived expert union within eligible', B='original mixture descending then symbol ascending head(N)',
                               N='same date A size', future_used=False, missing_outcome_replacement=False),
                horizons=list(HORIZONS), phases=list(PHASES), supplementary_phase='2026H1',
                outcomes=dict(raw='coverage==observed AND finite saved return',
                              residual='raw observable AND finite saved residual; fixed eligible-sector-peer baseline unchanged',
                              paired='both arms have at least one observable measure on same date',
                              complete='all preselected members in both arms have the relevant observable measure',
                              aggregation='date equal mean, median and fraction strictly positive'),
                decision_rules=dict(main_cells=12, main_phases=list(PHASES[:3]), minimum_complete_dates_per_cell=1,
                    precedence=['any nonpositive paired main mean -> not_supported_close_priority',
                                'otherwise any missing paired or complete main cell -> insufficient_evidence',
                                'otherwise any nonpositive complete main mean -> not_supported_close_priority',
                                'otherwise conditional_signal_only'],
                    missing_coverage_reported_even_if_nonpositive_main=True,
                    conditional_signal_is_not_stable_evidence=True),
                account_replayed=False, promotion_approved=False, new_prices_loaded=False,
                forbidden=['new scores or weights', 'new eligible membership', 'today financial backfill', 'N/parameter scans',
                           'account replay', 'annualization', 'independent-observation/significance claims', 'automatic trading'])


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'output/score_mechanism_20260927/run2'
    output = root / 'output/candidate_construction_20260927/run1'
    if output.exists():
        raise ValueError('attempt_exists:' + str(output))
    anchors = {
        str(source / 'summary.json'): 'cd23b08efb30e1c14074869b7f4b49b3778696c5b03ec597ebc886beff517cf2',
        str(source / 'manifest.json'): '27c0f631d034bb08a9983035c0caf64d134e775e279f0fe86f39ac48be92e63a',
    }
    documents = dict(script=Path(__file__), tests=root / 'tests/test_candidate_construction_audit.py',
                     plan=output.parent / 'PLAN.md', brief=output.parent / 'task-1-brief.md',
                     source_protocol=source.parent / 'NEXT_PROTOCOL.md')
    try:
        protocol = registered_protocol(source, anchors, documents)
    except Exception as error:
        # A registration failure is an immutable failed attempt, too.
        def failed_registration(_output):
            raise error
        execute_attempt(output, dict(created_at_utc=utc_now(), sources=anchors,
                                     registration_error=str(error), promotion_approved=False), failed_registration)
        raise
    result = execute_attempt(output, protocol, lambda path: analyze_archives(path, source))
    print(json.dumps(dict(status=result['status'], decision=result['decision'], membership=result['membership']), indent=2))


if __name__ == '__main__':
    main()
