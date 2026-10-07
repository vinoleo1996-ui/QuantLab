#!/usr/bin/env python3
"""Frozen-source, research-only Q1 cadence experiment. No production policy edits."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'output/frozen_source/6157c64'
COMMIT = '6157c64c84717dc8645dde3fe6228590b9f8538e'
DATA_VERSION = '5403b0902a7ca7737c9cc11cfc1c2a67e32934a3e5264265713a713b554eb1a2'
MANIFEST = ROOT / 'data/local_market/1c85fd9aceaa03c3c4863bec/manifest.json'
OUT = ROOT / 'output/continuity_review_20260927/cadence13'
START, END = '2020-01-02', '2026-06-30'
SPEC = dict(minimum_adv=5e6, risk_policy='P2', candidate='quality_pb', allocation_policy='legacy')
MAIN = ROOT / 'output/dynamic_risk/0e4aeb60bbd59c257851f53e/runs/3790f7ddea061e76873d6375-f81393ceb582'
RAW = {name: ROOT / 'output/turnover_ab' / name for name in ('qP2_monthly', 'qP2_monthly_hold90')}


def load_frozen():
    """Never silently mix canonical working-tree modules with the frozen Engine."""
    for name, module in list(sys.modules.items()):
        if name == 'quantlab' or name.startswith('quantlab.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(FROZEN):
                raise ValueError('nonfrozen_quantlab_already_imported:' + name)
    if str(FROZEN) not in sys.path:
        sys.path.insert(0, str(FROZEN))


def effective_config():
    load_frozen()
    from quantlab.policy_comparison import arm_config
    return arm_config(SPEC)


def date_strings(values):
    import pandas as pd
    return sorted({str(pd.Timestamp(x).date()) for x in values})


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError('artifact_checksum_mismatch:' + str(path))
    return actual


def checked_pickle(path, expected):
    """Only trusted local outputs; hash check precedes pickle execution."""
    verify_file(path, expected)
    with Path(path).open('rb') as source:
        return pickle.load(source)


def publish(path, writer, *, immutable=True):
    """Publish complete bytes atomically; never clobber an immutable artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise ValueError('immutable_output_exists:' + str(path)) from exc
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze_json(path, value, *, immutable=True):
    data = json.dumps(value, sort_keys=True, indent=2, default=str, allow_nan=False).encode()
    publish(path, lambda handle: handle.write(data), immutable=immutable)


def source_identity():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FROZEN, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=FROZEN, text=True).strip()
    if head != COMMIT or dirty:
        raise ValueError('frozen_source_identity_mismatch')
    files = sorted(list((FROZEN / 'quantlab').rglob('*.py')) + list((FROZEN / 'scripts').glob('*.py')))
    hashes = {str(p.relative_to(FROZEN)): sha256(p) for p in files}
    return dict(commit=head, dirty=False, code_sha256=hashes, code_digest=digest_json(hashes),
                dependency_lock_sha256=sha256(FROZEN / 'requirements-research.lock'))


def baseline_identity():
    artifacts = json.loads((MAIN / 'artifacts.json').read_text())
    for name, digest in artifacts['sha256'].items():
        path = (MAIN / name).resolve()
        if not path.is_relative_to(MAIN):
            raise ValueError('main_artifact_path_escape')
        verify_file(path, digest)
    metadata = json.loads((MAIN / 'metadata.json').read_text())
    provenance = json.loads((MAIN / 'provenance.json').read_text())
    if provenance['git_commit'] != COMMIT or provenance['git_dirty'] or metadata['data_version'] != DATA_VERSION:
        raise ValueError('main_baseline_identity_mismatch')
    result = dict(main=dict(path=str(MAIN), artifacts_sha256=sha256(MAIN / 'artifacts.json'),
                            files=artifacts['sha256']))
    for name, folder in RAW.items():
        arm = json.loads((folder / 'arm.json').read_text())
        expected_over = dict(rebalance_rule='monthly')
        if name.endswith('hold90'):
            expected_over['min_holding_period_days'] = 90
        if arm['name'] != name or arm['spec'] != SPEC or arm['cfg_over'] != expected_over or arm['status'] == 'failed':
            raise ValueError('raw_baseline_identity_mismatch:' + name)
        result[name] = dict(path=str(folder), files={n: sha256(folder / n) for n in ('result.pkl', 'arm.json')},
                           provenance='raw pickle lacked original artifact manifest/source attestation; checksums captured now; arm.json is a prior claim; numeric parity independently checked during compare')
    return result


def protocol_identity():
    """Cheap preflight: streaming checksums and calendar only, no pickle/data replay."""
    import pandas as pd
    load_frozen()
    from quantlab.strategy.decision import review_dates
    source = source_identity()
    manifest = json.loads(MANIFEST.read_text())
    entry = manifest['calendar']
    calendar_path = (MANIFEST.parent / entry['path']).resolve()
    if not calendar_path.is_relative_to(MANIFEST.parent):
        raise ValueError('calendar_path_escape')
    verify_file(calendar_path, entry['sha256'])
    calendar = pd.DatetimeIndex(pd.read_parquet(calendar_path)['session'])
    requested = date_strings(d for d in review_dates(calendar, 'every_n_weeks', 13) if pd.Timestamp(START) <= d <= pd.Timestamp(END))
    cfg = asdict(effective_config())
    if cfg['capital']['init_cash'] != 600000 or cfg['capital']['a_share_positions'] != 20:
        raise ValueError('frozen_capital_identity_mismatch')
    return dict(version='cadence13-v1', question='Q1: reduce ordinary turnover without worsening net account risk',
        source=source, harness_sha256=sha256(__file__), tests_sha256=sha256(ROOT / 'tests/test_continuity_cadence.py'),
        manifest_path=str(MANIFEST), manifest_sha256=sha256(MANIFEST), expected_data_version=DATA_VERSION,
        effective_config=json.loads(json.dumps(cfg, default=str)), spec=SPEC, start=START, end=END,
        requested_rule='every_n_weeks', requested_n_weeks=13, requested_dates=requested,
        requested_dates_sha256=digest_json(requested), requested_count=len(requested),
        calendar_sha256=entry['sha256'], baseline_artifacts=baseline_identity(),
        preparation='monthly effective legacy config; original monthly FactorDecisionService IC; only account review_override changes',
        calendar_phase='review_dates on the entire frozen calendar, then window filter; not calendar-quarter ends',
        uncertainty=dict(blocks=[20, 60], samples=2000, seed=20260924, family=['qP2_monthly_hold90-minus-monthly', 'cadence13-minus-monthly']),
        promotion_approved=False, evidence_limit='same-sample descriptive research; data gaps block promotion; no independent OOS or stress/cost-double pass')


def preflight():
    identity = protocol_identity()
    path = OUT / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('protocol_identity_mismatch')
    else:
        freeze_json(path, identity)
    return dict(status='preflight_ready', protocol=str(path), protocol_sha256=sha256(path),
                requested_count=identity['requested_count'], replay_started=False)


STAGES = {'2020-21': (START, '2021-12-31'), '2022-23': ('2022-01-01', '2023-12-31'),
          '2024-25': ('2024-01-01', '2025-12-31'), '2026H1': ('2026-01-01', END)}


def normalized_metrics(result, start=START, end=END):
    """Use absolute ledger NAV and initial capital, including first-day losses."""
    import numpy as np
    import pandas as pd
    initial = float(result['config']['capital']['init_cash'])
    accounts = result['account']
    nav = pd.Series([x['nav'] for x in accounts], index=pd.to_datetime([x['date'] for x in accounts]), dtype=float)
    if nav.empty or nav.index.has_duplicates or not nav.index.is_monotonic_increasing or not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError('invalid_account_nav')
    if str(nav.index[0].date()) != start or str(nav.index[-1].date()) != end:
        raise ValueError('account_window_mismatch')
    recorded = pd.Series(result['nav'], dtype=float).copy()
    recorded.index = pd.to_datetime(recorded.index)
    if not recorded.index.equals(nav.index) or not np.allclose(recorded, nav / initial, atol=1e-12, rtol=0):
        raise ValueError('account_recorded_nav_mismatch')
    daily = nav.pct_change(fill_method=None)
    daily.iloc[0] = nav.iloc[0] / initial - 1
    if 'daily_return' in result:
        stored = result['daily_return'].copy()
        stored.index = pd.to_datetime(stored.index)
        if not stored.index.equals(daily.index) or not np.allclose(stored, daily, atol=1e-12, rtol=0):
            raise ValueError('account_daily_return_mismatch')
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    stages = {}
    for name, (a, b) in STAGES.items():
        selected, before = nav.loc[a:b], nav.loc[nav.index < pd.Timestamp(a)]
        if selected.empty:
            stages[name] = dict(net_return=None, status='no_sessions')
            continue
        base = float(before.iloc[-1]) if len(before) else initial
        stages[name] = dict(net_return=float(selected.iloc[-1] / base - 1),
                            initial_boundary_nav=base, terminal_nav=float(selected.iloc[-1]),
                            boundary_date=str(before.index[-1].date()) if len(before) else 'initial_capital',
                            first_session=str(selected.index[0].date()), last_session=str(selected.index[-1].date()))
    costs = {name: dict(fills=0, notional=0., fees=0., slippage=0.) for name in ('ordinary', 'risk_exit', 'other')}
    for fill in result.get('fills', []):
        group = 'ordinary' if fill['reason'] == 'legacy_repaired' else 'risk_exit' if fill['reason'] == 'risk_exit' else 'other'
        costs[group]['fills'] += 1
        for key, field in (('notional', 'notional'), ('fees', 'fees'), ('slippage', 'slippage_cost')):
            costs[group][key] += float(fill[field])
    for row in costs.values():
        for key in ('notional', 'fees', 'slippage'):
            row[key] = round(row[key], 2)
    valid_stages = {k: v['net_return'] for k, v in stages.items() if v['net_return'] is not None}
    holdings = [len([q for q in x['quantities'].values() if q > 0]) for x in accounts]
    gross = [sum(x['weights'].values()) for x in accounts]
    return dict(status=result['status'], initial_capital=initial, terminal_nav=float(nav.iloc[-1]),
        start=start, end=end, observations=len(nav), years=years,
        total_net_return=float(nav.iloc[-1] / initial - 1), net_cagr=float((nav.iloc[-1] / initial) ** (1 / years) - 1),
        max_drawdown=float((nav / nav.cummax().clip(lower=initial) - 1).min()),
        stages=stages, worst_stage=min(valid_stages, key=valid_stages.get),
        mean_gross=float(np.mean(gross)), mean_holdings=float(np.mean(holdings)), max_holdings=max(holdings),
        fees=round(sum(x['fees'] for x in costs.values()), 2),
        slippage=round(sum(x['slippage'] for x in costs.values()), 2), costs=costs,
        filled_notional=round(sum(x['notional'] for x in costs.values()), 2),
        data_issue_count=len(result.get('data_issues', [])), promotion_approved=False,
        cost_note='CNY from actual fills; slippage already embedded in fill prices; no CAGR cost addback or no-cost counterfactual'), daily


def review_audit(result, requested, start=START, end=END):
    """Ordinary decisions may be preempted by risk exits or invalid score/allocation.

    Engine logs decision intents, not fills. It logs failed reviews in signal_trace;
    daily risk exits take precedence over ordinary reviews in the same session.
    """
    import pandas as pd
    expected = date_strings(d for d in requested if pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end))
    actual = date_strings(t['signal_date'] for t in result.get('trades', []) if t['reason'] == 'legacy_repaired')
    risk = set(date_strings(t['signal_date'] for t in result.get('trades', []) if t['reason'] == 'risk_exit'))
    traces = {str(pd.Timestamp(t['date']).date()): t for t in result.get('signal_trace', [])}
    accounts = set(date_strings(s['date'] for s in result.get('account', [])))
    deviations, unexplained = [], []
    for day in sorted(set(expected) - set(actual)):
        trace = traces.get(day, {})
        if day in risk:
            reason = 'risk_exit_preempted_ordinary_review'
        elif trace.get('status') not in (None, 'ok'):
            reason = 'review_failed:' + trace['status']
        else:
            reason = 'missing_account_session' if day not in accounts else 'unexplained_missing_review'
            unexplained.append(day)
        deviations.append(dict(date=day, reason=reason, trace_reasons=trace.get('reasons', [])))
    return dict(requested_dates=expected, actual_ordinary_dates=actual,
                requested_count=len(expected), actual_ordinary_count=len(actual),
                requested_sha256=digest_json(expected), actual_ordinary_sha256=digest_json(actual),
                accounted_count=len(expected) - len(unexplained), deviations=deviations,
                unexplained_dates=unexplained, unexpected_ordinary_dates=sorted(set(actual) - set(expected)),
                logging_semantics='close-time ordinary decision intents; risk_exit has precedence; failed reviews appear in signal_trace; fills execute later')


def replay13(dataset, cfg=None, *, start=START, end=END):
    load_frozen()
    from quantlab.backtest.engine import Engine, prepare
    from quantlab.strategy.legacy import legacy_config
    from quantlab.strategy.decision import review_dates
    cfg = legacy_config(cfg or effective_config())
    prepared = prepare(dataset, cfg)
    requested = review_dates(dataset.calendar, 'every_n_weeks', 13)
    result = Engine(cfg).run_dataset(dataset, strategy='legacy_repaired', start=start, end=end,
        data_mode='free_research', industry_policy='enforced', prepared=prepared,
        risk_policy='P2', candidate='quality_pb', allocation_policy='legacy', review_override=requested)
    audit = review_audit(result, requested, start, end)
    return result, audit


def reserve_attempt(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError('existing_run_attempt:no_automatic_rerun:' + str(path)) from exc


def frozen_protocol():
    path = OUT / 'protocol.json'
    if not path.exists():
        raise ValueError('preflight_protocol_required')
    protocol = json.loads(path.read_text())
    if protocol != protocol_identity():
        raise ValueError('protocol_identity_mismatch')
    return protocol


def validate_result(result, protocol, name):
    import copy
    expected = copy.deepcopy(protocol['effective_config'])
    if name == 'qP2_monthly_hold90':
        expected['capital']['min_holding_period_days'] = 90
    actual = json.loads(json.dumps(result.get('config'), default=str))
    if actual != expected:
        raise ValueError('result_config_mismatch:' + name)
    required = dict(data_version=DATA_VERSION, candidate='quality_pb', risk_policy='P2',
                    allocation_policy='legacy', data_mode='free_research', industry_policy='enforced',
                    strategy='legacy_repaired', promotion_approved=False)
    for key, value in required.items():
        if result.get(key) != value:
            raise ValueError('result_identity_mismatch:' + name + ':' + key)
    if result.get('status') not in ('unverified_data_gaps', 'research_unvalidated'):
        raise ValueError('incomplete_or_unexpected_result:' + name)


def run13():
    """Explicit CLI only. An attempt directory prevents duplicate/recovery reruns."""
    import traceback
    folder = OUT / 'run13'
    if folder.exists():
        raise ValueError('existing_run_attempt:no_automatic_rerun:' + str(folder))
    protocol = frozen_protocol()
    reserve_attempt(folder)
    status = folder / 'status.json'
    try:
        freeze_json(status, dict(status='loading', promotion_approved=False), immutable=False)
        from quantlab.data.contracts import load_dataset
        from quantlab.data.qualification import research_assessment
        ds = load_dataset(MANIFEST, profile='free_research')
        if ds.version != DATA_VERSION:
            raise ValueError('dataset_version_mismatch')
        assessment = research_assessment(ds, require_industry=False)
        if assessment['status'] != 'research_ready':
            raise ValueError('dataset_not_research_ready')
        freeze_json(folder / 'data_assessment.json', assessment)
        freeze_json(status, dict(status='running', promotion_approved=False), immutable=False)
        result, audit = replay13(ds)
        # Persist the completed raw result before downstream checks for recovery.
        publish(folder / 'result.pkl', lambda handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL))
        freeze_json(folder / 'artifacts.json', dict(result_sha256=sha256(folder / 'result.pkl'),
                    protocol_sha256=sha256(OUT / 'protocol.json')))
        validate_result(result, protocol, 'cadence13')
        freeze_json(folder / 'review_audit.json', audit)
        if audit['requested_dates'] != protocol['requested_dates'] or audit['unexpected_ordinary_dates'] or audit['unexplained_dates']:
            raise ValueError('review_calendar_validation_failed')
        summary, daily = normalized_metrics(result)
        if summary['max_holdings'] > 20:
            raise ValueError('observed_holding_limit_exceeded')
        freeze_json(folder / 'summary.json', summary)
        publish(folder / 'daily_net_returns.csv', lambda handle: handle.write(daily.to_csv().encode()))
        completion = dict(status='complete', engine_status=result['status'], promotion_approved=False,
                          result_sha256=sha256(folder / 'result.pkl'), protocol_sha256=sha256(OUT / 'protocol.json'))
        freeze_json(folder / 'complete.json', completion)
        freeze_json(status, completion, immutable=False)
        return completion
    except BaseException as exc:
        error = dict(status='failed', error=str(exc), traceback=traceback.format_exc(),
                     promotion_approved=False, recovery='inspect saved raw artifact; no automatic replay')
        freeze_json(folder / 'error.json', error, immutable=False)
        freeze_json(status, error, immutable=False)
        raise


def compact_result(result, protocol, name):
    load_frozen()
    from quantlab.strategy.decision import review_dates
    validate_result(result, protocol, name)
    summary, daily = normalized_metrics(result)
    if summary['max_holdings'] > 20:
        raise ValueError('observed_holding_limit_exceeded:' + name)
    calendar = result.get('calendar')
    if calendar is None:
        import pandas as pd
        manifest = json.loads(MANIFEST.read_text())
        calendar = pd.DatetimeIndex(pd.read_parquet(MANIFEST.parent / manifest['calendar']['path'])['session'])
    requested = review_dates(calendar, 'every_n_weeks', 13) if name == 'cadence13' else review_dates(calendar, 'monthly')
    audit = review_audit(result, requested)
    if audit['unexplained_dates'] or audit['unexpected_ordinary_dates']:
        raise ValueError('review_calendar_validation_failed:' + name)
    # Compact fingerprints and daily series survive release of the multi-GB raw run.
    signature = {key: digest_json(result.get(key, [])) for key in ('account', 'orders', 'fills', 'trades')}
    return dict(summary=summary, daily=daily, review_audit=audit, signature=signature)


def paired_evidence(runs):
    import pandas as pd
    baseline = runs['qP2_monthly']['daily']
    differences, descriptive = {}, {}
    for name, run in runs.items():
        if name == 'qP2_monthly':
            continue
        values = run['daily']
        if not values.index.equals(baseline.index) or values.isna().any() or baseline.isna().any():
            raise ValueError('paired_return_calendar_mismatch:' + name)
        label = name + '-minus-monthly'
        differences[label] = values - baseline
        groups = {}
        for group, mask in (('baseline_rising_days', baseline > 0), ('other_days', baseline <= 0)):
            a, b = values[mask], baseline[mask]
            groups[group] = dict(days=int(mask.sum()),
                mean_candidate_daily_net=float(a.mean()) if len(a) else None,
                mean_baseline_daily_net=float(b.mean()) if len(b) else None,
                mean_daily_net_difference=float((a - b).mean()) if len(a) else None,
                sum_daily_net_differences=float((a - b).sum()),
                underperformed_days=int((a < b).sum()))
        descriptive[label] = dict(groups=groups, interpretation='ex-post realized net differences conditional on baseline returns; non-causal, not a missed-trade counterfactual')
    frame = pd.DataFrame(differences)
    return frame, descriptive


def compare():
    """Postprocess sequentially. Never call Engine or load the full market dataset."""
    import gc
    import traceback
    protocol = frozen_protocol()
    include13 = (OUT / 'run13/complete.json').exists()
    folder = OUT / ('comparison_with13' if include13 else 'comparison_baselines')
    reserve_attempt(folder)
    status = folder / 'status.json'
    try:
        artifacts = protocol['baseline_artifacts']
        freeze_json(status, dict(status='loading_main'), immutable=False)
        main = checked_pickle(MAIN / 'result.pkl', artifacts['main']['files']['result.pkl'])
        main_compact = compact_result(main, protocol, 'main')
        del main
        gc.collect()
        runs = {}
        for name, root in RAW.items():
            freeze_json(status, dict(status='loading', arm=name), immutable=False)
            result = checked_pickle(root / 'result.pkl', artifacts[name]['files']['result.pkl'])
            runs[name] = compact_result(result, protocol, name)
            del result
            gc.collect()
        import numpy as np
        base = runs['qP2_monthly']
        same_calendar = base['daily'].index.equals(main_compact['daily'].index)
        max_delta = float(np.max(np.abs(base['daily'].to_numpy() - main_compact['daily'].to_numpy()))) if same_calendar else None
        parity = dict(calendar_equal=same_calendar, max_daily_net_difference=max_delta,
                      terminal_nav_difference=base['summary']['terminal_nav'] - main_compact['summary']['terminal_nav'],
                      fees_difference=base['summary']['fees'] - main_compact['summary']['fees'],
                      slippage_difference=base['summary']['slippage'] - main_compact['summary']['slippage'],
                      account_orders_fills_trades_hash_equal=base['signature'] == main_compact['signature'])
        parity['numeric_parity'] = same_calendar and max_delta <= 1e-12 and all(abs(parity[k]) < .01 for k in ('terminal_nav_difference', 'fees_difference', 'slippage_difference'))
        freeze_json(folder / 'monthly_main_parity.json', parity)
        if not parity['numeric_parity']:
            raise ValueError('monthly_main_numeric_parity_failed')
        if include13:
            completion = json.loads((OUT / 'run13/complete.json').read_text())
            if completion['protocol_sha256'] != sha256(OUT / 'protocol.json'):
                raise ValueError('run13_protocol_identity_mismatch')
            result = checked_pickle(OUT / 'run13/result.pkl', completion['result_sha256'])
            runs['cadence13'] = compact_result(result, protocol, 'cadence13')
            del result
            gc.collect()
        frame, descriptive = paired_evidence(runs)
        from quantlab.research_statistics import simultaneous_block_ci
        uncertainty = {str(block): simultaneous_block_ci(frame, block=block, samples=2000, seed=20260924) for block in (20, 60)}
        evidence = dict(status='same_sample_descriptive_only', promotion_approved=False,
            protocol_sha256=sha256(OUT / 'protocol.json'), arms={k: v['summary'] for k, v in runs.items()},
            monthly_main_parity=parity, reviews={k: v['review_audit'] for k, v in runs.items()},
            paired_uncertainty=uncertainty, missed_upside_descriptive=descriptive,
            cadence13_status='completed' if include13 else 'pending_authorized_explicit_run',
            limitations=['raw monthly/hold90 source attestation was not persisted originally; current hashes do not retrospectively certify provenance',
                         'same data and window, not independent OOS; no cost-double or stress replay pass',
                         'partial corporate actions and PIT proxies block promotion; no automatic parameter or risk-recovery changes',
                         'bootstrap estimates annualized arithmetic mean daily differences, not CAGR differences; family correction excludes earlier searches'])
        freeze_json(folder / 'comparison.json', evidence)
        publish(folder / 'paired_daily_net_differences.csv', lambda handle: handle.write(frame.to_csv().encode()))
        completion = dict(status='complete', compared=list(runs), promotion_approved=False,
                          comparison_sha256=sha256(folder / 'comparison.json'))
        freeze_json(folder / 'complete.json', completion)
        freeze_json(status, completion, immutable=False)
        return dict(completion, output=str(folder))
    except BaseException as exc:
        failure = dict(status='failed', error=str(exc), traceback=traceback.format_exc(), promotion_approved=False)
        freeze_json(folder / 'error.json', failure, immutable=False)
        freeze_json(status, failure, immutable=False)
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run13', 'compare'])
    args = parser.parse_args()
    action = {'preflight': preflight, 'run13': run13, 'compare': compare}[args.command]
    try:
        print(json.dumps(action(), indent=2, default=str))
    except Exception as exc:
        print(json.dumps(dict(status='blocked', error=str(exc))), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
