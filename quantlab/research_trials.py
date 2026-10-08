"""Append-only global trial ledger (``docs/trial_ledger.csv``).

Every registered variant is one row. Families are counted separately for the
Deflated Sharpe Ratio; the global count is always reported next to it.
Outcomes keep technical failure, no improvement and insufficient data apart.
"""
from __future__ import annotations

import csv
import hashlib
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


def record_outcome(family, trial_id, outcome, note='', ledger=LEDGER):
    """Outcomes are appended as a new row with role copied, never edited in place.

    Rows are numbered ``#outcome1, #outcome2, ...`` and carry the sha256 of the pre-registration as it
    is when the outcome is recorded, so a header correction after registration stays visible."""
    if outcome not in OUTCOMES:
        raise ValueError('invalid_outcome')
    existing = [r for r in rows(ledger) if r['family'] == family]
    base = next((r for r in existing if r['trial_id'] == trial_id), None)
    if base is None:
        raise ValueError(f'trial_not_registered:{family}/{trial_id}')
    number = 1 + sum(r['trial_id'].startswith(f'{trial_id}#outcome') for r in existing)
    row = dict(base, trial_id=f'{trial_id}#outcome{number}', outcome=outcome, note=note,
               prereg_sha256=sha256_file(ROOT / base['prereg']) if base['prereg'] else '',
               registered_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    with open(ledger, 'a', newline='', encoding='utf-8') as handle:
        csv.DictWriter(handle, fieldnames=COLUMNS).writerow(row)
    return row


def count(family=None, ledger=LEDGER, include_diagnostics=True):
    """Distinct registered trials (outcome rows are not new trials)."""
    names = {(r['family'], r['trial_id']) for r in rows(ledger) if '#outcome' not in r['trial_id']
             and (family is None or r['family'] == family) and (include_diagnostics or r['role'] == 'trial')}
    return len(names)


def confirm_uses(family, ledger=LEDGER):
    return sum(int(r['confirm_uses'] or 0) for r in rows(ledger)
               if r['family'] == family and '#outcome' not in r['trial_id'])
