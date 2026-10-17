"""Append-only global trial ledger (``docs/trial_ledger.csv``).

Every registered variant is one row. Families are counted separately for the
Deflated Sharpe Ratio; the global count is always reported next to it.
Outcomes keep technical failure, no improvement and insufficient data apart.
"""
from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'docs/trial_ledger.csv'
COLUMNS = ('family', 'trial_id', 'role', 'registered_at', 'prereg', 'prereg_sha256', 'source', 'data',
           'segment', 'confirm_uses', 'outcome', 'note')
ROLES = ('trial', 'diagnostic')
OUTCOMES = ('pending', 'pass', 'no_improvement', 'technical_failure', 'insufficient_data', 'withdrawn')


def sha256_file(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ''


def rows(ledger=LEDGER):
    if not Path(ledger).exists():
        return []
    with open(ledger, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def register(family, trial_id, *, prereg='', source='', data='', segment='', role='trial', confirm_uses=0,
             outcome='pending', note='', ledger=LEDGER):
    """Append one row; an existing (family, trial_id) is refused rather than overwritten."""
    if role not in ROLES or outcome not in OUTCOMES:
        raise ValueError('invalid_role_or_outcome')
    if any(r['family'] == family and r['trial_id'] == trial_id for r in rows(ledger)):
        raise ValueError(f'trial_already_registered:{family}/{trial_id}')
    row = {'family': family, 'trial_id': trial_id, 'role': role,
           'registered_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'prereg': str(prereg), 'prereg_sha256': sha256_file(ROOT / prereg) if prereg else '',
           'source': source, 'data': data, 'segment': segment, 'confirm_uses': int(confirm_uses),
           'outcome': outcome, 'note': note}
    path = Path(ledger)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return row


def record_outcome(family, trial_id, outcome, note='', *, role=None, ledger=LEDGER):
    """Outcomes are appended as a new row with role copied, never edited in place.

    Rows are numbered ``#outcome1, #outcome2, ...`` and carry the sha256 of the pre-registration as it
    is when the outcome is recorded, so a header correction after registration stays visible.
    ``role`` can reclassify a trial for the effective record (e.g. a non-independent baseline)."""
    if outcome not in OUTCOMES:
        raise ValueError('invalid_outcome')
    if role is not None and role not in ROLES:
        raise ValueError('invalid_role')
    existing = [r for r in rows(ledger) if r['family'] == family]
    base = next((r for r in existing if r['trial_id'] == trial_id), None)
    if base is None:
        raise ValueError(f'trial_not_registered:{family}/{trial_id}')
    number = 1 + sum(r['trial_id'].startswith(f'{trial_id}#outcome') for r in existing)
    row = dict(base, trial_id=f'{trial_id}#outcome{number}', outcome=outcome, note=note,
               role=role or base['role'],
               prereg_sha256=sha256_file(ROOT / base['prereg']) if base['prereg'] else '',
               registered_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    with open(ledger, 'a', newline='', encoding='utf-8') as handle:
        csv.DictWriter(handle, fieldnames=COLUMNS).writerow(row)
    return row


def base_trial_id(trial_id):
    return str(trial_id).split('#outcome', 1)[0]


def outcome_number(trial_id):
    match = re.fullmatch(r'.+#outcome(\d+)', str(trial_id))
    return int(match.group(1)) if match else None


def _corrections(family, trial_id, ledger=LEDGER):
    base = base_trial_id(trial_id)
    found = [r for r in rows(ledger) if r['family'] == family and base_trial_id(r['trial_id']) == base
             and r['trial_id'] != base and outcome_number(r['trial_id']) is not None]
    return sorted(found, key=lambda r: outcome_number(r['trial_id']))


def _effective_map(ledger=LEDGER):
    """``(family, base_trial_id) -> latest row`` without rereading the file per trial."""
    result = {}
    for row in rows(ledger):
        key = (row['family'], base_trial_id(row['trial_id']))
        number = outcome_number(row['trial_id'])
        if number is None:
            result.setdefault(key, row)
            continue
        current = result.get(key)
        current_number = outcome_number(current['trial_id']) if current else None
        if current is None or current_number is None or number >= current_number:
            result[key] = row
    return result


def effective(family, trial_id, ledger=LEDGER):
    """Latest outcome row, falling back to the registration row."""
    return _effective_map(ledger).get((family, base_trial_id(trial_id)))


def effective_outcome(family, trial_id, ledger=LEDGER):
    row = effective(family, trial_id, ledger)
    return row['outcome'] if row else None


def effective_confirm_uses(family, trial_id, ledger=LEDGER):
    row = effective(family, trial_id, ledger)
    return int(row['confirm_uses'] or 0) if row else 0


def count(family=None, ledger=LEDGER, include_diagnostics=True):
    """Distinct registered trials (outcome rows are not new trials). ``role`` is effective:
    an append-only correction can reclassify a trial as diagnostic."""
    payload = rows(ledger)
    effective_rows = _effective_map(ledger)
    names = {(r['family'], r['trial_id']) for r in payload if '#outcome' not in r['trial_id']
             and (family is None or r['family'] == family)}
    if not include_diagnostics:
        names = {(family, trial_id) for family, trial_id in names
                 if (effective_rows.get((family, trial_id)) or {}).get('role') == 'trial'}
    return len(names)


def families(ledger=LEDGER, include_diagnostics=False):
    """Distinct families under the same effective-role rule as ``count``."""
    payload = rows(ledger)
    effective_rows = _effective_map(ledger)
    names = {(r['family'], r['trial_id']) for r in payload if '#outcome' not in r['trial_id']}
    if not include_diagnostics:
        names = {(family, trial_id) for family, trial_id in names
                 if (effective_rows.get((family, trial_id)) or {}).get('role') == 'trial'}
    return sorted({family for family, _ in names})


def confirm_uses(family, ledger=LEDGER):
    names = {r['trial_id'] for r in rows(ledger)
             if r['family'] == family and '#outcome' not in r['trial_id']}
    return sum(effective_confirm_uses(family, trial_id, ledger) for trial_id in names)


def _placeholder(value):
    text = str(value).strip() if value is not None else ''
    return not text or '—' in text or text.lower() in ('-', 'n/a', 'pending')


def ledger_report(ledger=LEDGER):
    """Integrity report with errors, warnings and rows explained by an append-only correction."""
    payload = rows(ledger)
    errors, warnings, corrected = [], [], []
    if not payload:
        return dict(ok=True, errors=[], warnings=['ledger_empty'], corrected=[],
                    rows=0, trials=0, families=0)
    missing = [column for column in COLUMNS if column not in payload[0]]
    if missing:
        errors.append(f'missing_columns:{",".join(missing)}')
    seen, base_rows, outcome_numbers = {}, {}, {}
    for line, row in enumerate(payload, start=2):
        family, trial_id = row.get('family', ''), row.get('trial_id', '')
        key = (family, trial_id)
        seen.setdefault(key, []).append((line, row))
        row['_line'] = line
        try:
            datetime.fromisoformat(str(row.get('registered_at', '')).replace('Z', '+00:00'))
        except ValueError:
            errors.append(f'line {line}: invalid registered_at {row.get("registered_at")!r} ({family}/{trial_id})')
        number = outcome_number(trial_id)
        if number is not None:
            base = base_trial_id(trial_id)
            base_key = (family, base, number)
            outcome_numbers.setdefault(base_key, []).append((line, row))
        else:
            base_rows[key] = row
    # Append-only history may contain repeated outcome ids. The latest valid row supersedes
    # the earlier ones; repeated *base* ids remain an error because no ordering is defined.
    for key, items in seen.items():
        if len(items) < 2:
            continue
        family, trial_id = key
        if outcome_number(trial_id) is None:
            errors.append(f'duplicate base trial_id {family}/{trial_id} at lines '
                          f'{", ".join(str(i) for i, _ in items)}')
        else:
            latest_line, latest = max(items, key=lambda item: item[0])
            if latest.get('outcome') in OUTCOMES:
                corrected.append(f'duplicate historical {family}/{trial_id}: '
                                 f'superseded by line {latest_line}')
            else:
                errors.append(f'line {latest_line}: latest duplicate has invalid outcome '
                              f'{latest.get("outcome")!r} ({family}/{trial_id})')
    for key, items in outcome_numbers.items():
        if len(items) < 2:
            continue
        family, base, number = key
        latest_line, latest = max(items, key=lambda item: item[0])
        if latest.get('outcome') not in OUTCOMES:
            errors.append(f'line {latest_line}: duplicate {family}/{base}#outcome{number} '
                          f'has invalid outcome {latest.get("outcome")!r}')
    for (family, trial_id), row in base_rows.items():
        line = row['_line']
        corrections = _corrections(family, trial_id, ledger)
        if row.get('role') not in ROLES:
            errors.append(f'line {line}: invalid role {row.get("role")!r} ({family}/{trial_id})')
        if row.get('outcome') not in OUTCOMES:
            if corrections and corrections[-1].get('outcome') in OUTCOMES:
                corrected.append(f'line {line}: outcome {row.get("outcome")!r} -> '
                                 f'{corrections[-1]["outcome"]} ({family}/{trial_id})')
            else:
                errors.append(f'line {line}: invalid outcome {row.get("outcome")!r} ({family}/{trial_id})')
        prereg = row.get('prereg', '')
        if _placeholder(prereg):
            if corrections and corrections[-1].get('outcome') == 'withdrawn':
                corrected.append(f'line {line}: placeholder prereg explained by withdrawal ({family}/{trial_id})')
            else:
                errors.append(f'line {line}: missing prereg ({family}/{trial_id})')
        elif not (ROOT / prereg).exists():
            errors.append(f'line {line}: prereg does not exist {prereg!r} ({family}/{trial_id})')
        if _placeholder(row.get('prereg_sha256')):
            if corrections and corrections[-1].get('outcome') == 'withdrawn':
                corrected.append(f'line {line}: placeholder sha explained by withdrawal ({family}/{trial_id})')
            else:
                errors.append(f'line {line}: placeholder prereg_sha256 ({family}/{trial_id})')
        elif prereg and (ROOT / prereg).exists() and row['prereg_sha256'] != sha256_file(ROOT / prereg):
            warnings.append(f'line {line}: prereg_sha256 differs from current file (amended?) ({family}/{trial_id})')
    for line, row in enumerate(payload, start=2):
        trial_id = row.get('trial_id', '')
        if outcome_number(trial_id) is not None:
            family = row.get('family', '')
            if (family, base_trial_id(trial_id)) not in base_rows:
                errors.append(f'line {line}: outcome row without base trial ({family}/{trial_id})')
            if row.get('outcome') not in OUTCOMES:
                errors.append(f'line {line}: invalid outcome {row.get("outcome")!r} ({family}/{trial_id})')
    families = {r['family'] for r in payload if '#outcome' not in r['trial_id']}
    return dict(ok=not errors, errors=errors, warnings=warnings, corrected=corrected,
                rows=len(payload), trials=count(ledger=ledger), families=len(families))


def validate_ledger(ledger=LEDGER):
    """Empty list means no unexplained integrity error; historical rows healed by a
    later ``#outcomeN`` correction are reported by ``ledger_report()['corrected']``."""
    return ledger_report(ledger)['errors']


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('validate',))
    parser.add_argument('--ledger', default=str(LEDGER))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = ledger_report(args.ledger)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        for line in report['errors']:
            print('ERROR', line)
        for line in report['corrected']:
            print('CORRECTED', line)
        for line in report['warnings']:
            print('WARN', line)
        print(f"validate_ledger: {len(report['errors'])} error(s), "
              f"{len(report['corrected'])} corrected, {report['trials']} trials")
    raise SystemExit(0 if report['ok'] else 1)
