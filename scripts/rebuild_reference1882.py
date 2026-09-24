"""Rebuild the archived broad-basket rules, preserving source and target identity.

This is a reconstruction on the CURRENT local database, not an exact replay of
an unidentified historical database snapshot. The other checkout stays read-only.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

REFERENCE_COMMIT = 'c01bef11dc708f8d50cc26148f3741f9508741f5'
REFERENCE_REPORT = 'reports/validation/micro_broad100_quality_pb3_2022_2025.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, default=str, ensure_ascii=False, indent=2))
    temporary.replace(path)


def freeze_sources(repo, destination):
    report = json.loads((repo / REFERENCE_REPORT).read_text())
    expected = report['code_sha256']
    for relative, sha in expected.items():
        raw = subprocess.check_output(['git', 'show', REFERENCE_COMMIT + ':' + relative], cwd=repo)
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('reference_commit_hash_mismatch:' + relative)
    archive = subprocess.check_output(['git', 'archive', '--format=tar', REFERENCE_COMMIT,
                                       'a_share_quant', 'scripts'], cwd=repo)
    recovered = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.py'): continue
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()): raise ValueError('source_archive_path_escape')
            raw = tar.extractfile(member).read()
            sha = hashlib.sha256(raw).hexdigest()
            if target.exists() and digest(target) != sha:
                raise ValueError('frozen_source_already_differs:' + member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.write_bytes(raw)
            recovered[member.name] = sha
    metadata = dict(commit=REFERENCE_COMMIT, declared_sources_verified=len(expected),
                    declared_sources=expected, full_snapshot_sources=recovered,
                    snapshot_sha256=hashlib.sha256(archive).hexdigest(),
                    original_report_sha256=digest(repo / REFERENCE_REPORT))
    save(destination / 'source_manifest.json', metadata)
    return metadata


def worker(args):
    frozen = args.output / 'frozen_source'
    metadata = json.loads((frozen / 'source_manifest.json').read_text())
    for relative, sha in metadata['full_snapshot_sources'].items():
        if digest(frozen / relative) != sha: raise ValueError('frozen_source_changed:' + relative)
    sys.path.insert(0, str(frozen))
    spec = importlib.util.spec_from_file_location('reference1882_original_runner', frozen / 'scripts/run_growth_core.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    identity = json.loads((args.output / 'rebuild_protocol.json').read_text())
    catalog_stat = args.catalog.stat()
    if catalog_stat.st_size != identity['current_catalog']['size'] or catalog_stat.st_mtime_ns != identity['current_catalog']['mtime_ns']:
        raise ValueError('catalog_changed_after_protocol_freeze')
    start, end = (date(2022, 1, 1), date(2025, 12, 31)) if args.period == '2022_2025' else (date(2018, 1, 1), date(2021, 12, 31))
    namespace = SimpleNamespace(catalog=args.catalog, start=start, end=end, universe=2000,
        universe_offset=3000, minimum_adv=5_000_000., exclude_st=False, cadence=20,
        no_guidance=True, broad_basket=100, broad_quality_filter=0., broad_max_pb=3.,
        broad_illiquidity_tilt=False, signals=['quality_filtered_equal'], sleeves=6,
        no_drawdown_scaler=True, index_trend_filter=False, breadth_cycle_gate=False,
        take_profit=None, stop_loss=None, trail_arm=None, trail_drop=None)
    target_records = []
    expected_targets = {}
    ever_targeted = set()
    original_replay = original.replay_daily_targets
    def recording_replay(*positional, **kwargs):
        policy = kwargs['decision_callback']
        records = []
        def observe(state):
            target = policy(state)
            if target is not None:
                expected = expected_targets.get(str(target.decision_on))
                if expected != dict(target.weights):
                    raise ValueError('observed_target_differs_from_frozen_policy_precomputation')
                records.append(dict(signal_date=str(target.decision_on), weights=dict(target.weights),
                                    sell_only=target.sell_only, max_exposure=target.max_exposure,
                                    account_equity=state.equity, account_cash=state.cash))
            return target
        kwargs['decision_callback'] = observe
        positional = list(positional)
        positional[1] = tuple(asset for asset in positional[1] if asset.symbol in ever_targeted)
        replay = original_replay(*positional, **kwargs)
        target_records.append(records)
        save(args.output / ('targets_' + args.period + f'_cost{len(target_records)}.json'),
             dict(status='original_rules_on_current_data', source_commit=REFERENCE_COMMIT,
                  source_manifest_sha256=digest(frozen / 'source_manifest.json'),
                  data_identity=identity['current_catalog'], initial_cash=500000., holdings=100,
                  targets=records, targets_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                  caveats=['historical_database_snapshot_unidentified', 'historical_reconstruction_not_blind_oos',
                           'target_schedule_from_adjusted_unit_reference_account']))
        save(args.output / ('account_' + args.period + f'_cost{len(target_records)}.json'),
             dict(status='original_rules_current_data_zero_target_assets_omitted', source_commit=REFERENCE_COMMIT,
                  metrics=original.curve_metrics(500000., replay.equity_curve),
                  equity_curve=[[str(p.trading_day),p.equity,p.cash] for p in replay.equity_curve],
                  fill_count=len(replay.fills), target_consistency_passed=True))
        return replay
    original.replay_daily_targets = recording_replay
    rows, industries, bundle, coverage, cross_section_start = original.build_signals_bundle(namespace)
    # Save every decision's actual candidate count before replay, including thin
    # early years where ranks 3001..5000 cannot supply 2000 securities.
    base = bundle.stock_heads['defensive_base']
    save(args.output / ('coverage_' + args.period + '.json'),
         dict(requested_start=str(start), requested_end=str(end), score_history_start=str(cross_section_start),
              first_available_score=str(min(base)) if base else None, last_available_score=str(max(base)) if base else None,
              decision_candidate_counts={str(day): len(cross) for day, cross in base.items()}, coverage=coverage))
    # With all optional exits and scaling disabled, this frozen policy depends
    # only on score snapshots and its prior basket, never on account equity or
    # fills. Precompute it twice under different account states to assert that
    # fact before eliminating securities that always have zero target weight.
    from a_share_quant.stock_only.micro_filters import load_profit_pb_snapshots, quality_pass
    from a_share_quant.stock_only.broad_basket import BroadBasketPolicy
    from a_share_quant.daily_replay import ReplayDecisionState
    passing = quality_pass(load_profit_pb_snapshots(args.catalog, base), minimum_profit=0., maximum_pb=3.)
    heads = {day:{symbol:(1. if passing.get((symbol,day)) else 0.) for symbol in cross}
             for day,cross in base.items() if day>=start}
    sessions = [day for day in original.trading_sessions(args.catalog) if start<=day<=end]
    policies = [BroadBasketPolicy(heads, {d:industries[d] for d in heads}, {sessions[0]:{}},
                                  holdings=100,buffer=50,drawdown_scaler=None) for _ in range(2)]
    for day in sessions:
        left = policies[0](ReplayDecisionState(day,500000.,500000.,()))
        right = policies[1](ReplayDecisionState(day,800000.,400000.,(('000001.SZ',100),)))
        if left != right: raise ValueError('reference_policy_is_account_dependent_cannot_compress')
        if left is not None:
            expected_targets[str(day)] = dict(left.weights)
            ever_targeted.update(symbol for symbol,weight in left.weights if weight>0)
    save(args.output / ('precomputed_targets_' + args.period + '.json'),
         dict(source_commit=REFERENCE_COMMIT,current_catalog=identity['current_catalog'],
              status='original_rules_current_data',account_independence_checked=True,
              targets=[dict(signal_date=d,weights=w,sell_only=False,max_exposure=.9) for d,w in expected_targets.items()],
              ever_positive_target_count=len(ever_targeted),original_candidate_union=len({c for cross in base.values() for c in cross}),
              limitations=['full_signal_cross_sections_preserved','historical_database_snapshot_identity_unknown']))
    original_load_bars = original.load_bars
    def compressed_load_bars(catalog, symbols, first, last):
        if not ever_targeted.issubset(symbols): raise ValueError('precomputed_target_outside_original_pool')
        bars, closes, missing = original_load_bars(catalog, ever_targeted, first, last)
        connection = original.duckdb.connect(str(catalog),read_only=True)
        try:
            all_days = {r[0] for r in connection.execute(
                'SELECT DISTINCT trading_day FROM daily_bars WHERE symbol IN (SELECT unnest(?)) AND trading_day BETWEEN ? AND ?',
                [sorted(symbols),first,last]).fetchall()}
        finally: connection.close()
        if {bar.trading_day for bar in bars} != all_days:
            raise ValueError('zero_target_asset_compression_changed_trading_calendar')
        print(f'zero-target asset compression {len(symbols)} -> {len(ever_targeted)}; complete calendar checked',flush=True)
        return bars,closes,missing
    original.load_bars = compressed_load_bars
    payload = original.run_portfolios(namespace, industries, bundle, coverage, cross_section_start)
    after = args.catalog.stat()
    if after.st_size != catalog_stat.st_size or after.st_mtime_ns != catalog_stat.st_mtime_ns:
        raise ValueError('catalog_changed_during_reference_rebuild')
    for relative, sha in metadata['full_snapshot_sources'].items():
        if digest(frozen / relative) != sha: raise ValueError('frozen_source_changed_during_rebuild')
    archived = json.loads((args.repo / REFERENCE_REPORT).read_text())
    payload['rebuild_identity'] = dict(source_commit=REFERENCE_COMMIT, declared_source_hashes_match=True,
        current_catalog=identity['current_catalog'], exact_historical_reproduction=False,
        original_signal_sha256=archived.get('signal_sha256'),
        signal_sha256_matches_original=payload['signal_sha256'] == archived.get('signal_sha256') if args.period == '2022_2025' else None,
        scope='fixed_original_rules_current_data_retrospective',
        targets_exported=len(target_records), source_unchanged=True, database_file_stat_unchanged=True,
        zero_target_asset_compression=dict(ever_targeted=len(ever_targeted), account_independence_checked=True,
                                           all_actual_targets_checked=True, trading_calendar_checked=True))
    save(args.output / ('rebuild_' + args.period + '.json'), payload)
    print(json.dumps(dict(period=args.period, status='completed_conditional_reference_rebuild',
        signal_match=payload['rebuild_identity']['signal_sha256_matches_original'],
        runs=[{k: r[k] for k in ('cost_multiplier', 'annualized_return', 'max_drawdown', 'fill_count')} for r in payload['runs']]), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/Users/agiuser/Desktop/A股ETF量化策略'))
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--period', choices=['2022_2025', '2018_2021'])
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    args.catalog = (args.catalog or args.repo / 'data/panel/catalog.duckdb').resolve()
    if args.worker:
        if not args.period: parser.error('--worker requires --period')
        worker(args)
        return
    sources = freeze_sources(args.repo, args.output / 'frozen_source')
    driver_sha = digest(Path(__file__))
    frozen_driver = args.output / 'driver_versions' / (driver_sha + '.py')
    frozen_driver.parent.mkdir(parents=True,exist_ok=True)
    if not frozen_driver.exists(): frozen_driver.write_bytes(Path(__file__).read_bytes())
    save(args.output / ('driver_variant_' + driver_sha[:12] + '.json'),
         dict(driver_sha256=driver_sha,source_commit=REFERENCE_COMMIT,
              optimization='omit assets with zero target throughout; full score snapshots retained',
              checks=['account_state_independence','every_callback_target','complete_trading_calendar']))
    stat = args.catalog.stat()
    protocol = dict(version='reference1882-rebuild-v1', source_commit=REFERENCE_COMMIT,
        original_source_hashes_matched=sources['declared_sources_verified'],
        current_catalog=dict(path=str(args.catalog), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(args.catalog)),
        periods=['2022_2025', '2018_2021'], configure_from='archived_18.82_percent_report',
        changes=['database_is_current_snapshot_not_identified_original_snapshot',
                 'record_targets_without_changing_policy_or_execution'],
        historical_status='retrospective_original_rule_reconstruction_not_blind_oos', promotion_approved=False)
    if (args.output / 'rebuild_protocol.json').exists():
        prior = json.loads((args.output / 'rebuild_protocol.json').read_text())
        if prior != protocol: raise ValueError('rebuild_protocol_already_frozen_differently')
    else: save(args.output / 'rebuild_protocol.json', protocol)
    python = args.python or args.repo / '.venv/bin/python'
    for period in ([args.period] if args.period else protocol['periods']):
        if (args.output / ('rebuild_' + period + '.json')).exists():
            print('existing frozen reconstruction:', period, flush=True)
            continue
        command = [str(python), str(frozen_driver), '--worker', '--period', period,
                   '--repo', str(args.repo), '--catalog', str(args.catalog), '--output', str(args.output)]
        result = subprocess.run(command, cwd=args.output)
        if result.returncode:
            save(args.output / ('failure_' + period + '.json'), dict(status='failed', returncode=result.returncode,
                 period=period, source_commit=REFERENCE_COMMIT))
            raise SystemExit(result.returncode)


if __name__ == '__main__': main()
