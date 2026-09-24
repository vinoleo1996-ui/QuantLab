"""Read-only provenance recovery for the archived 18.82% adjusted-unit result.

Inspect current sources, bounded Git blob history and bounded source archives.
No code from the other repository is executed or modified. A report NAV is not
an order schedule, so recovering a curve alone cannot reproduce its trades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.report.runs import atomic_json


DEFAULT_REPO = Path('/Users/agiuser/Desktop/A股ETF量化策略')
DEFAULT_REPORT = 'reports/validation/micro_broad100_quality_pb3_2022_2025.json'
CONFIG_FIELDS = ('universe_budget', 'universe_offset', 'minimum_adv', 'exclude_st', 'start', 'end',
                 'score_history_start', 'initial_capital', 'sleeves', 'broad_basket_holdings', 'cadence',
                 'broad_quality_filter', 'broad_max_pb', 'broad_illiquidity_tilt', 'broad_basket_exits',
                 'holding_sessions', 'drawdown_scaler', 'drawdown_bands', 'index_trend_filter', 'breadth_cycle_gate')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def recover_sources(repo, expected, *, max_git_blobs=1500, max_archives=24):
    """Return exact source locations without extracting or importing them."""
    recovered = {}
    mismatches = {}
    for relative, digest in expected.items():
        file = repo / relative
        if file.is_file():
            actual = _sha(file.read_bytes())
            if actual == digest: recovered[relative] = dict(kind='current_file', path=str(file), sha256=actual)
            else: mismatches[relative] = dict(expected=digest, current=actual)
    missing = set(expected) - set(recovered)
    inspected = dict(git_blobs=0, archives=0, git_search_truncated=False, archive_search_truncated=False)
    git_error = None
    if missing and (repo / '.git').exists():
        try:
            listing = subprocess.check_output(['git', 'rev-list', '--objects', '--all', '--', *sorted(missing)],
                                               cwd=repo, text=True, stderr=subprocess.DEVNULL, timeout=30)
            objects = []
            for line in listing.splitlines():
                fields = line.split(' ', 1)
                if len(fields) == 2 and fields[1] in expected:
                    objects.append((fields[0], fields[1]))
            inspected['git_search_truncated'] = len(objects) > max_git_blobs
            for oid, relative in objects[:max_git_blobs]:
                if relative not in missing: continue
                raw = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=repo,
                                              stderr=subprocess.DEVNULL, timeout=10)
                inspected['git_blobs'] += 1
                if _sha(raw) == expected[relative]:
                    recovered[relative] = dict(kind='git_blob', object_id=oid, sha256=expected[relative])
                    missing.remove(relative)
        except (OSError, subprocess.SubprocessError) as exc:
            git_error = type(exc).__name__ + ':' + str(exc)
    # Source archives are searched only in known artifact directories. Raw data
    # archives and the 3GB database are never scanned as source candidates.
    archives = []
    if missing:
        for folder in ('output', 'reports', 'archives', 'backups'):
            root = repo / folder
            if not root.exists(): continue
            for file in root.rglob('*.tar.gz'):
                if any(word in file.name.lower() for word in ('source', 'src', 'snapshot')) and file.stat().st_size <= 128 * 1024 * 1024:
                    archives.append(file)
                    if len(archives) > max_archives: break
            if len(archives) > max_archives: break
    inspected['archive_search_truncated'] = len(archives) > max_archives
    archive_errors = []
    for file in sorted(archives)[:max_archives]:
        if not missing: break
        inspected['archives'] += 1
        try:
            with tarfile.open(file, 'r:gz') as archive:
                for item in archive:
                    if not item.isfile() or item.size > 5 * 1024 * 1024: continue
                    relative = next((name for name in missing if item.name == name or item.name.endswith('/' + name)), None)
                    if relative is None: continue
                    stream = archive.extractfile(item)
                    if stream is not None and _sha(stream.read()) == expected[relative]:
                        recovered[relative] = dict(kind='source_archive', path=str(file), member=item.name,
                                                   archive_sha256=_sha(file.read_bytes()), sha256=expected[relative])
                        missing.remove(relative)
        except (tarfile.TarError, OSError) as exc:
            archive_errors.append(dict(path=str(file), reason=str(exc)))
    return dict(recovered=recovered, missing=sorted(missing), current_mismatches=mismatches,
                limits=dict(max_git_blobs=max_git_blobs, max_archives=max_archives), inspected=inspected,
                git_error=git_error, archive_errors=archive_errors)


def audit_reference(repo=DEFAULT_REPO, report=DEFAULT_REPORT, *, max_git_blobs=1500, max_archives=24):
    repo = Path(repo).resolve()
    path = Path(report)
    if not path.is_absolute(): path = repo / path
    raw = path.read_bytes()
    artifact = json.loads(raw)
    expected = artifact.get('code_sha256', {})
    recovery = recover_sources(repo, expected, max_git_blobs=max_git_blobs, max_archives=max_archives)
    runs = artifact.get('runs', [])
    original = next((r for r in runs if r.get('signal') == 'quality_filtered_equal' and r.get('cost_multiplier') == 1), None)
    boundary = []
    if not expected: boundary.append('original_source_hash_manifest_missing')
    if recovery['missing']: boundary.append('original_source_hashes_not_recovered')
    if not artifact.get('data_version') and not artifact.get('catalog_sha256'):
        boundary.append('original_database_snapshot_identity_missing')
    if not artifact.get('target_schedule'):
        boundary.append('original_dated_target_schedule_missing')
    if not artifact.get('signal_artifact'):
        boundary.append('signal_hash_without_recoverable_signal_artifact')
    if original is None: boundary.append('expected_reference_result_missing')
    curve = original.get('equity_curve', []) if original else []
    return dict(kind='reference1882_provenance_audit', status='cannot_exactly_reproduce' if boundary else 'source_inputs_recovered_not_yet_replayed',
        report=dict(path=str(path), sha256=_sha(raw), kind=artifact.get('kind'), status=artifact.get('status')),
        configuration={name: artifact.get(name) for name in CONFIG_FIELDS},
        historical_result={key: original.get(key) for key in ('annualized_return', 'max_drawdown', 'final_equity', 'mean_exposure', 'fill_count')} if original else None,
        source_recovery=recovery, signal_sha256=artifact.get('signal_sha256'),
        equity_curve=dict(rows=len(curve), first_day=curve[0][0] if curve else None,
                          last_day=curve[-1][0] if curve else None, meaning='reported adjusted-unit account; not reconstructed actual-share fills'),
        reproduction_blockers=boundary,
        original_replay=dict(status='not_run', reason='exact_sources_and_input_snapshot_required'),
        cash_ledger_replay=dict(status='not_run', reason='dated targets and actual-share corporate action data required; NAV cannot reconstruct orders'),
        next_action='Recover documented source/input identities and dated targets; otherwise label a new mechanism reconstruction as a different experiment.',
        caveats=artifact.get('caveats', []), promotion_approved=False)


def attach_reconstruction(audit, directory):
    """Verify recovered source, actual curves and target artifacts before linking.

    Exact equity parity on the present database does not recover the identity of
    the historical database and does not turn adjusted units into actual shares.
    """
    root = Path(directory).resolve()
    linked = json.loads(json.dumps(audit))
    paths = {}
    def read(relative):
        path = root / relative
        raw = path.read_bytes()
        paths[relative] = dict(path=str(path), sha256=_sha(raw))
        return json.loads(raw)
    source = read('frozen_source/source_manifest.json')
    original_raw = Path(audit['report']['path']).read_bytes()
    if _sha(original_raw) != audit['report']['sha256'] or source['original_report_sha256'] != _sha(original_raw):
        raise ValueError('reference_report_changed')
    original = json.loads(original_raw)
    if source['declared_sources'] != original['code_sha256']:
        raise ValueError('reference_declared_source_identity_mismatch')
    for relative, digest in source['declared_sources'].items():
        path = (root / 'frozen_source' / relative).resolve()
        if not path.is_relative_to(root / 'frozen_source') or _sha(path.read_bytes()) != digest:
            raise ValueError('reference_source_checksum_mismatch:' + relative)
    protocol = read('rebuild_protocol.json')
    if protocol['source_commit'] != source['commit']:
        raise ValueError('reference_rebuild_commit_mismatch')
    periods = {}
    for period in ('2022_2025', '2018_2021'):
        rebuilt = read('rebuild_' + period + '.json')
        identity = rebuilt['rebuild_identity']
        if identity['source_commit'] != source['commit'] or identity['current_catalog'] != protocol['current_catalog']:
            raise ValueError('reference_rebuild_identity_mismatch')
        if rebuilt['code_sha256'] != original['code_sha256']:
            raise ValueError('reference_rebuild_sources_mismatch')
        counts = read('coverage_' + period + '.json')['decision_candidate_counts']
        runs = []
        for cost in (1, 2):
            target = read('targets_' + period + '_cost' + str(cost) + '.json')
            if target['source_commit'] != source['commit'] or target['data_identity'] != protocol['current_catalog']:
                raise ValueError('reference_target_identity_mismatch')
            if target['source_manifest_sha256'] != paths['frozen_source/source_manifest.json']['sha256']:
                raise ValueError('reference_target_source_manifest_mismatch')
            if _sha(json.dumps(target['targets'], sort_keys=True).encode()) != target['targets_sha256']:
                raise ValueError('reference_target_checksum_mismatch')
            run = next(x for x in rebuilt['runs'] if x['cost_multiplier'] == cost and x['signal'] == 'quality_filtered_equal')
            account = read('account_' + period + '_cost' + str(cost) + '.json')
            if account['equity_curve'] != run['equity_curve']:
                raise ValueError('reference_account_curve_mismatch')
            first_invested = next((row[0] for row in run['equity_curve'] if row[1] - row[2] > .01), None)
            item = {k:run.get(k) for k in ('cost_multiplier','annualized_return','max_drawdown','final_equity','fill_count')}
            item.update(sessions=len(run['equity_curve']), first_invested=first_invested,
                        targets=len(target['targets']))
            if period == '2022_2025':
                archived = next(x for x in original['runs'] if x['cost_multiplier'] == cost and x['signal'] == 'quality_filtered_equal')
                pairs = list(zip(run['equity_curve'], archived['equity_curve']))
                same_dates = len(run['equity_curve']) == len(archived['equity_curve']) and all(a[0] == b[0] for a,b in pairs)
                difference = max((abs(a[1]-b[1]) for a,b in pairs), default=None)
                if not same_dates or difference != 0 or rebuilt['signal_sha256'] != original['signal_sha256']:
                    raise ValueError('archived_reference_equity_or_signal_not_reproduced')
                item.update(archived_equity_max_difference=difference, archived_dates_equal=same_dates,
                            archived_fills_equal=run.get('fill_count') == archived.get('fill_count'))
            runs.append(item)
        periods[period] = dict(runs=runs, candidate_min=min(counts.values()) if counts else 0,
                               candidate_max=max(counts.values()) if counts else 0,
                               history_status='retrospective_not_blind_oos')
    linked['status'] = 'reconstructed_on_current_data_exact_archived_equity_parity'
    linked['original_replay'] = dict(status=linked['status'], source_commit=source['commit'],
        periods=periods, evidence=paths, historical_database_snapshot_identified=False,
        current_database=protocol['current_catalog'], accounting='original_adjusted_units',
        caveats=['exact_2022_2025_equity_and_signal_parity_does_not_identify_historical_database',
                 '2018_2021_extension_has_no_investment_in_2018_and_is_not_blind_oos',
                 'actual_cash_share_reconstruction_is_a_separate_experiment'])
    linked['cash_ledger_replay'] = dict(status='not_run', reason='waiting_for_final_raw_price_and_corporate_action_dataset',
        dated_targets_recovered=True)
    linked['next_action'] = 'Replay recovered targets with raw prices, actual share lots, cash and explicit corporate actions.'
    return linked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--report', default=DEFAULT_REPORT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-git-blobs', type=int, default=1500)
    parser.add_argument('--max-archives', type=int, default=24)
    parser.add_argument('--reconstruction', type=Path)
    args = parser.parse_args()
    result = audit_reference(args.repo, args.report, max_git_blobs=args.max_git_blobs, max_archives=args.max_archives)
    if args.reconstruction: result = attach_reconstruction(result, args.reconstruction)
    atomic_json(args.output, result)
    print(json.dumps(dict(status=result['status'], blockers=result['reproduction_blockers'], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__': main()
