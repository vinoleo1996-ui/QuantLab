"""Offline reference selection analysis using immutable, previously saved outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.router_random_diagnostic import block_indices, clean, freeze, sample_basket, sha

ROOT = Path('/Users/agiuser/Desktop/QuantLab')
SEED = 20260927
DRAWS = 1000


def build_decisions(period, period_end, inputs, targets, old_decisions):
    """Map the archived target to information already known on its signal date."""
    rows = []
    prior = set()
    seen = set()
    for target in targets['targets']:
        signal = target['signal_date']
        known = [day for day in inputs['heads'] if day <= signal]
        if not known or signal in seen or signal > period_end:
            raise ValueError('invalid_signal_information_timing')
        seen.add(signal)
        information = max(known)
        heads = inputs['heads'][information]
        industries = {code: inputs['industries'][information].get(code, 'unclassified')
                      for code in heads}
        selected = sorted(target['weights'])
        matches = [(i, x) for i, x in enumerate(old_decisions)
                   if x['period'] == period and x['signal_date'] == signal]
        if len(matches) != 1:
            raise ValueError('primary_identity_missing_or_duplicate')
        old_index, old = matches[0]
        expected = dict(information_date=information, period_end=period_end,
                        heads=heads, industries=industries)
        for key, value in expected.items():
            if old[key] != value:
                raise ValueError('primary_identity_mismatch:' + key)
        if len(old['selected']) != len(selected):
            raise ValueError('primary_identity_mismatch:n')
        if not selected or len(selected) > 50 or not set(selected) <= set(heads):
            raise ValueError('invalid_reference_members')
        if target['sell_only'] or not np.isclose(target['max_exposure'], .9):
            raise ValueError('invalid_reference_exposure')
        if any(not np.isclose(w, .9 / len(selected)) for w in target['weights'].values()):
            raise ValueError('reference_not_equal_weight')
        if max(Counter(industries[c] for c in selected).values()) > 15:
            raise ValueError('reference_industry_cap')
        rows.append(dict(period=period, signal_date=signal, information_date=information,
                         period_end=period_end, heads=heads, industries=industries,
                         selected=selected, retained_count=len(prior.intersection(selected)),
                         old_index=old_index))
        prior = set(selected)
    return rows


def make_memberships(specs, old_members, draws=DRAWS):
    """Copy certified primary draws; freeze new exact reference strata draws."""
    rng = np.random.default_rng(SEED)
    result = {}
    for j, spec in enumerate(specs):
        codes = sorted(spec['heads'])
        lookup = {code: i for i, code in enumerate(codes)}
        n = len(spec['selected'])
        primary = np.asarray(old_members[f"{spec['old_index']}_primary"])
        if primary.shape != (draws, n) or not np.issubdtype(primary.dtype, np.integer):
            raise ValueError('primary_membership_shape_or_type')
        if np.any(primary < 0) or np.any(primary >= len(codes)):
            raise ValueError('primary_membership_index')
        if np.any(np.diff(np.sort(primary, axis=1), axis=1) == 0):
            raise ValueError('primary_duplicate_member')
        for row in primary:
            if max(Counter(spec['industries'][codes[i]] for i in row).values()) > 15:
                raise ValueError('primary_industry_cap')
        result[f'{j}_primary'] = primary.copy()
        strata = np.asarray([
            [lookup[c] for c in sample_basket(spec['heads'], spec['industries'], n,
                                              rng, spec['selected'])]
            for _ in range(draws)], dtype=np.int32)
        result[f'{j}_strata'] = strata
    return result


def reconstruct_labels(frame, specs, calendar):
    """Rebuild selected from reference targets; verify label keys and cutoff states."""
    keys = ['period', 'signal_date', 'instrument', 'horizon']
    labels = frame[keys + ['value', 'reason']].copy()
    if labels.duplicated(keys).any():
        raise ValueError('duplicate_label_identity')
    indexed = labels.set_index(keys)
    result = []
    expected_count = 0
    for j, spec in enumerate(specs):
        day = pd.Timestamp(spec['signal_date'])
        start = calendar.get_indexer([day])[0]
        if start < 0:
            raise ValueError('signal_not_calendar_session')
        codes = sorted(spec['heads'])
        for horizon in [20, 60]:
            endpoint = None if start + horizon >= len(calendar) else str(calendar[start + horizon].date())
            cutoff_reason = ('immature' if endpoint is None else
                             'period_end' if endpoint > spec['period_end'] else None)
            requested = pd.MultiIndex.from_tuples([
                (spec['period'], spec['signal_date'], c, horizon) for c in codes], names=keys)
            if not requested.isin(indexed.index).all():
                raise ValueError('missing_label_identity')
            sub = indexed.loc[requested].reset_index()
            if cutoff_reason:
                if not sub.reason.eq(cutoff_reason).all() or sub.value.notna().any():
                    raise ValueError('label_endpoint_mismatch')
            elif not sub.reason.isin(['observed', 'known_gap', 'missing_path']).all():
                raise ValueError('label_endpoint_mismatch')
            observed = sub.reason.eq('observed').to_numpy()
            values = sub.value.to_numpy(float)
            if not np.array_equal(observed, np.isfinite(values)) or np.any(values[observed] < -1):
                raise ValueError('invalid_label_value_reason')
            sub['selected'] = sub.instrument.isin(spec['selected'])
            sub['information_date'] = spec['information_date']
            sub['endpoint'] = endpoint
            sub['decision_index'] = j
            result.append(sub)
            expected_count += len(codes)
    if len(labels) != expected_count:
        raise ValueError('extra_label_identity')
    return pd.concat(result, ignore_index=True)


def basket_statistics(values, selected_indices, draws):
    """Fixed original denominators; any missing name makes its whole basket unknown."""
    sy, ay = values[selected_indices], values[draws]
    n = len(selected_indices)
    selected_complete = bool(np.isfinite(sy).all())
    selected_mean = float(sy.mean()) if selected_complete else None
    random_complete = np.isfinite(ay).all(axis=1)
    complete_means = ay[random_complete].mean(axis=1)
    paired = complete_means if selected_complete else np.array([])
    return dict(
        selected_unknown=int(np.sum(~np.isfinite(sy))),
        selected_known_fraction=float(np.isfinite(sy).mean()),
        random_unknown_names_total=int(np.sum(~np.isfinite(ay))),
        random_full_known_fraction=float(random_complete.mean()),
        paired_draws=len(paired),
        selected_mean=selected_mean,
        random_conditional_mean=float(paired.mean()) if len(paired) else None,
        selected_minus_random=float(selected_mean - paired.mean()) if len(paired) else None,
        selected_conditional_percentile=float(100 * (np.sum(paired < selected_mean) +
                                                    .5 * np.sum(paired == selected_mean)) /
                                              len(paired)) if len(paired) else None,
        selected_lower_bound=float((np.nansum(sy) - np.sum(~np.isfinite(sy))) / n),
        selected_upper_bound=selected_mean if selected_complete else 'unbounded',
        random_average_lower_bound=float(np.mean((np.nansum(ay, axis=1) -
                                                   np.sum(~np.isfinite(ay), axis=1)) / n)),
        random_average_upper_bound=float(ay.mean()) if np.isfinite(ay).all() else 'unbounded')


def bootstrap_comparisons(values, calendar, horizon, seed, family_tests=8):
    """Original circular session-block estimator with an explicit comparison family."""
    if horizon not in (20, 60) or family_tests < 1 or len(calendar) == 0:
        raise ValueError('invalid_bootstrap_parameters')
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError('invalid_bootstrap_calendar')
    sparse = pd.Series(np.nan, index=calendar)
    seen = set()
    for day, value in values:
        day = pd.Timestamp(day)
        if day in seen or day not in calendar or not np.isfinite(value):
            raise ValueError('invalid_bootstrap_observation')
        seen.add(day)
        sparse.loc[day] = value
    a = sparse.to_numpy()
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(10000):
        sampled = a[block_indices(len(a), horizon, rng)]
        sampled = sampled[np.isfinite(sampled)]
        if len(sampled):
            means.append(float(sampled.mean()))
    return dict(
        mean_pp=float(np.nanmean(a) * 100) if np.isfinite(a).any() else None,
        ci_family95_pp=(np.quantile(means, [.05 / (2 * family_tests),
                                           1 - .05 / (2 * family_tests)]) * 100).tolist() if means else None,
        bootstrap_draws=10000, valid_bootstrap_draws=len(means),
        calendar_sessions=len(a), block_sessions=horizon,
        observed_dates=int(np.isfinite(a).sum()),
        effective_blocks_upper_bound=min(int(np.isfinite(a).sum()), len(a) / horizon),
        inference='conditional_complete_basket_dates; low_power; not_independent_OOS',
        family_tests=family_tests, seed=seed)


def load_frozen_labels(out, label_path, specs, calendar):
    """No parquet content is read until immutable memberships and source hashes pass."""
    identity_path = out / 'membership_identity.json'
    if not identity_path.exists():
        raise ValueError('memberships_not_frozen')
    identity = json.loads(identity_path.read_text())
    if identity['stage'] != 'memberships_frozen_before_outcomes':
        raise ValueError('memberships_not_frozen')
    for name, expected in identity['hashes'].items():
        if sha(out / name) != expected:
            raise ValueError('membership_frozen_hash_changed:' + name)
    protocol = json.loads((out / 'protocol.json').read_text())
    if sha(label_path) != protocol['source_hashes'][str(label_path)]:
        raise ValueError('frozen_label_source_changed')
    freeze(out / 'label_read_event.json', dict(
        read_started_at=datetime.now(timezone.utc).isoformat(),
        membership_identity_sha256=sha(identity_path),
        source_sha256=sha(label_path),
        columns=['period', 'signal_date', 'instrument', 'horizon', 'value', 'reason'],
        old_selected_column_read=False))
    frame = pd.read_parquet(label_path, columns=['period', 'signal_date', 'instrument',
                                                'horizon', 'value', 'reason'])
    return reconstruct_labels(frame, specs, calendar)


def analyse(specs, members, labels, calendar):
    rows = []
    for j, spec in enumerate(specs):
        codes = sorted(spec['heads'])
        lookup = {code: i for i, code in enumerate(codes)}
        selected = np.array([lookup[c] for c in spec['selected']])
        def strata(names):
            return [dict(tier=t, industry=i, count=n) for (t, i), n in sorted(
                Counter((spec['heads'][c], spec['industries'][c]) for c in names).items())]
        for horizon in [20, 60]:
            sub = labels.loc[labels.decision_index.eq(j) & labels.horizon.eq(horizon)].set_index('instrument').loc[codes]
            values = sub.value.to_numpy(float)
            for control in ['primary', 'strata']:
                row = dict(period=spec['period'], signal_date=spec['signal_date'],
                           information_date=spec['information_date'], horizon=horizon,
                           endpoint=sub.endpoint.iloc[0], control=control, n=len(selected),
                           pool_n=len(codes), retained_count=spec['retained_count'],
                           selected_strata=strata(spec['selected']), pool_strata=strata(codes),
                           selected_missing_reasons=dict(Counter(sub.loc[spec['selected'], 'reason'])),
                           pool_missing_reasons=dict(Counter(sub.reason)),
                           draw_count=len(members[f'{j}_{control}']))
                row.update(basket_statistics(values, selected, members[f'{j}_{control}']))
                rows.append(row)
    summaries = []
    for period in dict.fromkeys(spec['period'] for spec in specs):
        period_specs = [spec for spec in specs if spec['period'] == period]
        start = min(spec['signal_date'] for spec in period_specs)
        end = period_specs[0]['period_end']
        period_calendar = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
        for horizon in [20, 60]:
            for control in ['primary', 'strata']:
                sub = [r for r in rows if r['period'] == period and r['horizon'] == horizon and r['control'] == control]
                observed = [r for r in sub if r['selected_minus_random'] is not None]
                values = [(r['signal_date'], r['selected_minus_random']) for r in observed]
                seed = SEED + horizon + len(summaries)
                stat = bootstrap_comparisons(values, period_calendar, horizon, seed, family_tests=8)
                joint = bootstrap_comparisons(values, period_calendar, horizon, seed, family_tests=12)
                def verdict(interval):
                    if len(observed) < 2 or interval is None:
                        return 'insufficient'
                    return 'positive' if interval[0] > 0 else 'negative' if interval[1] < 0 else 'inconclusive'
                stat.update(
                    period=period, horizon=horizon, control=control,
                    ci_joint12_family95_pp=joint['ci_family95_pp'], joint_family_tests=12,
                    decision=verdict(joint['ci_family95_pp']),
                    decision_original8=verdict(stat['ci_family95_pp']),
                    total_decisions=len(sub), selected_complete_decisions=sum(r['selected_unknown'] == 0 for r in sub),
                    random_full_known_fraction_all_dates=float(np.mean([r['random_full_known_fraction'] for r in sub])),
                    random_full_known_fraction_paired_dates=float(np.mean([r['random_full_known_fraction'] for r in observed])) if observed else None,
                    median_conditional_percentile=float(np.median([r['selected_conditional_percentile'] for r in observed])) if observed else None,
                    paired_draws_total=sum(r['paired_draws'] for r in sub),
                    unconditional_effect='not_identified_due_to_missing_outcomes_and_retention_confound')
                summaries.append(stat)
    return clean(rows), clean(summaries)


def run(out):
    """One immutable analysis: freeze sources, freeze members, then read outcomes."""
    out = Path(out)
    if out.exists() and any(p.name not in ('tests.log', 'execution.log') for p in out.iterdir()):
        raise ValueError('output_not_empty; preserve prior attempt and review before retry')
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    freeze(out / 'started.json', dict(started_at=started, status='preparing_frozen_analysis'))
    old_dir = ROOT / 'output/expert_selection_20260927/random_diagnostic'
    challenge = ROOT / 'output/reference_challenge_20260927/run2'
    dataset = ROOT / 'output/account_2025_20260927/extension/dataset'
    input_paths = {
        '2022_2025': ROOT / 'output/reference_sustainability_20260927/2022_2025/decision_inputs.json',
        '2025_recent': ROOT / 'output/reference_recent_20260927/inputs.json'}
    ends = {'2022_2025': '2025-12-31', '2025_recent': '2026-09-22'}
    target_paths = {period: challenge / period / 'targets_50.json' for period in ends}
    archived_hashes = json.loads((old_dir / 'result_hashes.json').read_text())
    old_names = ['protocol.json', 'frozen_decisions.json', 'memberships.npz',
                 'labels.parquet', 'membership_identity.json']
    for name in old_names:
        if sha(old_dir / name) != archived_hashes[name]:
            raise ValueError('archived_diagnostic_hash_changed:' + name)
    old_protocol = json.loads((old_dir / 'protocol.json').read_text())
    old_decisions = json.loads((old_dir / 'frozen_decisions.json').read_text())
    challenge_protocol = json.loads((challenge / 'protocol.json').read_text())
    manifest = json.loads((dataset / 'manifest.json').read_text())
    calendar_path = dataset / manifest['calendar']['path']
    source_root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__).resolve(), source_root / 'tests/test_reference_selection_diagnostic.py',
                  source_root / 'scripts/router_random_diagnostic.py',
                  source_root / 'tests/test_router_random_diagnostic.py']
    inherited_used = [*input_paths.values(), dataset / 'manifest.json', calendar_path, *code_paths[2:]]
    for path in inherited_used:
        if str(path) not in old_protocol['source_hashes'] or sha(path) != old_protocol['source_hashes'][str(path)]:
            raise ValueError('archived_input_or_sampler_changed:' + str(path))
    if sha(calendar_path) != manifest['calendar']['sha256']:
        raise ValueError('calendar_manifest_mismatch')
    specs = []
    for period, path in input_paths.items():
        target_path = target_paths[period]
        if sha(target_path) != challenge_protocol['targets'][str(target_path)]:
            raise ValueError('reference_target_hash_changed:' + period)
        specs.extend(build_decisions(period, ends[period], json.loads(path.read_text()),
                                     json.loads(target_path.read_text()), old_decisions))
    if len(specs) != 70 or any(len(spec['selected']) != 50 for spec in specs):
        raise ValueError('unexpected_reference_decision_count_or_size')
    paths = [*code_paths, *input_paths.values(), *target_paths.values(), calendar_path,
             dataset / 'manifest.json', challenge / 'protocol.json',
             *[old_dir / name for name in old_names], old_dir / 'result_hashes.json',
             old_dir / 'verification.json', old_dir / 'completion.json',
             ROOT / 'output/reference_selection_20260927/PLAN.md',
             ROOT / 'output/reference_selection_20260927/AMENDMENT.md']
    identities = {str(path): sha(path) for path in paths}
    source_dir = out / 'source'
    source_dir.mkdir()
    for path in code_paths:
        with (source_dir / path.name).open('xb') as stream:
            stream.write(path.read_bytes())
        if sha(source_dir / path.name) != identities[str(path)]:
            raise ValueError('source_snapshot_mismatch')
    freeze(out / 'protocol.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        question='Does frozen reference50 selection add stable 20/60-session price return within its own heads pool?',
        decision='Positive joint12 intervals retain selection evidence only; otherwise end this frozen selection direction without parameter rescue. No strategy adoption.',
        source_hashes=identities, seed=SEED, draws=DRAWS, bootstrap_draws=10000,
        original_family_tests=8, joint_family_tests=12, horizons=[20, 60],
        bootstrap_seed_rule='20260927 + horizon + comparison_index, shared between original8 and joint12',
        observer='Reference run2 targets_50; equal weight, 0.9 target exposure, original code-order and top150 retention; old router labels.selected never read.',
        account_conditions='RMB600000, at most50, long-only, unlevered; no account/model replay; signal-close adjusted price proxy excludes costs/execution/cash/dividends.',
        primary='Reuse old primary only after period, signal, information, heads, industry, N and cutoff match; financial-tier random sequential priority, industry cap15; not uniform over all feasible baskets.',
        strata='Generate new1000 baskets per decision from seed20260927, exact reference selected financial tier x industry counts, without replacement; no random retention.',
        timing='Latest archived input no later than signal; recent warm start signal2024-12-31 uses2024-12-12 snapshot. Members persist before labels.parquet columns read.',
        missing='No member filtering, replacement, zero-fill or renormalization; any missing label makes whole basket unknown. Paired means condition on selected-complete dates and random-complete draws.',
        bounds='Each price return >=-1 with unbounded upper tail. Basket lower bound=(known_sum-missing_count)/original_N. Unconditional effect not identified when missing outcomes exist.',
        temporal='Circular blocks of20/60 exchange sessions over sparse date observations;10000 resamples. Arithmetic equal-date means in pp, never annualized.',
        verdict='Use joint12: fewer than2 observed dates or no CI => insufficient; lower>0 positive; upper<0 negative; otherwise inconclusive. All are conditional historical inferences.',
        inherited_data='Reuse certified archived labels by hash; original panel hashes and verification are referenced, panels are not reread in this task.',
        limits=['Historical windows overlap and have been examined; not independent OOS.',
                'No pure alpha/beta or financial-filter causal identification; retention/other exposures differ.',
                'PIT/revision/industry/full corporate-actions gates remain unverified.',
                'No inference that15% is impossible or18% stable;100-stock reference is outside account mandate, not a mathematical upper bound.']))
    freeze(out / 'frozen_decisions.json', specs)
    with np.load(old_dir / 'memberships.npz', allow_pickle=False) as old_members:
        members = make_memberships(specs, old_members)
    with (out / 'memberships.npz').open('xb') as stream:
        np.savez_compressed(stream, **members)
    freeze(out / 'membership_identity.json', dict(
        stage='memberships_frozen_before_outcomes', frozen_at=datetime.now(timezone.utc).isoformat(),
        hashes={name: sha(out / name) for name in ['protocol.json', 'frozen_decisions.json', 'memberships.npz']},
        decisions=len(specs), baskets=len(specs) * 2 * DRAWS,
        primary_reused=len(specs) * DRAWS, strata_new=len(specs) * DRAWS))
    print('Sources and 140000 memberships frozen; now loading saved labels.', flush=True)
    calendar = pd.DatetimeIndex(pd.read_parquet(calendar_path).session).tz_localize(None).normalize()
    labels = load_frozen_labels(out, old_dir / 'labels.parquet', specs, calendar)
    labels.to_parquet(out / 'labels.parquet', index=False)
    freeze(out / 'labels_reference.json', dict(
        source=str(old_dir / 'labels.parquet'), source_sha256=identities[str(old_dir / 'labels.parquet')],
        rows=len(labels), old_selected_column_read=False,
        value_and_reason='unchanged from archived labels; selected reconstructed exclusively from reference targets',
        reconstructed_sha256=sha(out / 'labels.parquet')))
    rows, summaries = analyse(specs, members, labels, calendar)
    freeze(out / 'per_date.json', rows)
    freeze(out / 'summary.json', dict(
        status='completed_conditional_diagnostic', promotion_approved=False,
        account_replayed=False, model_trained=False, comparisons=summaries,
        decision_rule='joint12_family95; historical conditional evidence only',
        decision_counts=dict(Counter(stat['decision'] for stat in summaries))))
    after = {path: sha(path) for path in identities}
    changed = [path for path, digest in identities.items() if after[path] != digest]
    freeze(out / 'source_verification.json', dict(checked_at=datetime.now(timezone.utc).isoformat(),
        checked=len(identities), changed=changed, before=identities, after=after))
    if changed:
        raise ValueError('sources_changed_during_analysis')
    freeze(out / 'completion.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        started_at=started, status='completed', sources_verified=len(identities), decisions=len(specs),
        labels=len(labels), baskets=140000, comparisons=len(summaries), promotion_approved=False))
    freeze(out / 'result_hashes.json', {str(path.relative_to(out)): sha(path)
        for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ['tests.log', 'execution.log']})
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/reference_selection_20260927/diagnostic')
    args = parser.parse_args()
    try:
        run(args.out)
    except Exception as exc:
        if args.out.exists() and not (args.out / 'failure.json').exists():
            freeze(args.out / 'failure.json', dict(at=datetime.now(timezone.utc).isoformat(),
                status='technical_failure', exception=type(exc).__name__, message=str(exc)))
        raise
