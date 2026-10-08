#!/usr/bin/env python3
"""Report-only audit: how much did the "previous-target-only" sell list distort stable_book replays?

  python scripts/audit_adapter_sells.py <targets.json>

Replays the frozen targets as written and with a cumulative sell list (every name ever targeted and
not targeted now). Counts share-locked names (held at a signal close, neither targeted nor listed for
sale). Writes output/audit_adapter_sells_20261008/<label>.json. Does not change any frozen result.
"""
from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import pandas as pd

import stable_book as book
from quantlab.data.contracts import MarketDataset

OUT = ROOT / 'output/audit_adapter_sells_20261008'


def cumulative(rows):
    rows, ever = copy.deepcopy(rows), set()
    for row in rows:
        names = {book.canon(c) for c, w in row['weights'].items() if w > 0}
        row['sell'] = sorted((ever | {book.canon(c) for c in row['sell']}) - names)
        ever |= names
    return rows


def stuck(result, rows):
    states = {str(pd.Timestamp(s['date']).date()): s for s in result['account']}
    counts, weights = [], []
    for row in rows:
        state = states.get(row['signal_date'])
        if state is None:
            continue
        held = {c for c, q in state['quantities'].items() if q > 0}
        names = {book.canon(c) for c, w in row['weights'].items() if w > 0} | {book.canon(c) for c in row['sell']}
        locked = held - names
        counts.append(len(locked))
        weights.append(sum(state['quantities'][c] * state['marks'].get(c, 0.0) for c in locked) / state['nav'])
    return {'mean_locked': float(pd.Series(counts).mean()), 'max_locked': int(max(counts, default=0)),
            'mean_locked_weight': float(pd.Series(weights).mean())}


def main(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text())['targets']
    root = book.MANIFEST.parent
    manifest = json.loads(book.MANIFEST.read_text())
    fixed = cumulative(rows)
    codes = sorted({book.canon(c) for r in fixed for c in list(r['weights']) + r['sell']})
    read = lambda entry: pd.read_parquet(io.BytesIO(book._read_checked(root, entry)))
    calendar = pd.DatetimeIndex(book._dated(read(manifest['calendar'])).session)
    panels = {c: book._dated(read(manifest['panels'][c])) for c in codes if c in manifest['panels']}
    dataset = MarketDataset(panels, read(manifest['instruments']), calendar, read(manifest['actions']),
                            book._dated(read(manifest['benchmark'])), manifest=manifest)
    start, end = rows[0]['signal_date'], book.END
    label = f'{path.parent.name}__{path.stem}'
    report = {'targets': str(path.relative_to(ROOT)), 'start': start, 'end': end, 'promotion_approved': False}
    for name, plan in (('as_frozen', rows), ('cumulative_sell', fixed)):
        result = book.replay_account(dataset, plan, start=start, end=end)
        s = result['summary']
        report[name] = {'cagr': s['cagr'], 'max_drawdown': s['max_drawdown'], 'fills': s['fills'],
                        'checks_passed': s['account_checks_passed'], **stuck(result, plan)}
        print(json.dumps({label: {name: report[name]}}, default=float), flush=True)
    write(label, report)


def write(label, report):
    report['cagr_change_pp'] = (report['cumulative_sell']['cagr'] - report['as_frozen']['cagr']) * 100
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{label}.json').write_text(json.dumps(report, indent=1, default=str))


def recover(log):
    """Rebuild <label>.json from the per-replay lines printed to ``log`` (the replays are not rerun)."""
    reports = {}
    for line in Path(log).read_text().splitlines():
        if line.startswith('{"'):
            (label, part), = json.loads(line).items()
            reports.setdefault(label, {'recovered_from': str(Path(log).resolve().relative_to(ROOT)),
                                       'promotion_approved': False}).update(part)
    for label, report in reports.items():
        write(label, report)
    return sorted(reports)


def reconcile():
    """Frozen CAGR vs the as-frozen replay, re-annualized on the frozen clock (first session 2018-01-02)."""
    rows, end = [], pd.Timestamp(str(book.END)[:10])
    for path in sorted(OUT.glob('*__targets*.json')):
        audit = json.loads(path.read_text())
        folder, stem = path.stem.split('__')
        targets = json.loads((ROOT / 'output' / folder / f'{stem}.json').read_text())['targets']
        phase = stem.removeprefix('targets_')
        summary = ROOT / 'output' / folder / (f'{phase}_cost1' if phase != 'targets' else 'cost1') / 'summary.json'
        if not summary.exists():
            summary = ROOT / 'output' / folder / 'summary.json'
        frozen = json.loads(summary.read_text())
        clock = frozen.get('first') or (frozen.get('window') or [None])[0]
        first = targets[0]['signal_date']
        ratio = (end - pd.Timestamp(first)).days / (end - pd.Timestamp(clock)).days
        frozen_cagr = frozen.get('cagr')
        if frozen_cagr is None and 'ending_nav' in frozen:
            frozen_cagr = (frozen['ending_nav'] / 600000) ** (365.25 / (end - pd.Timestamp(clock)).days) - 1
        rows.append({'line': path.stem, 'first_signal': first, 'frozen_clock': clock, 'frozen_cagr': frozen_cagr,
                     'replay_cagr_first_signal_clock': audit['as_frozen']['cagr'],
                     'replay_cagr_frozen_clock': (1 + audit['as_frozen']['cagr']) ** ratio - 1,
                     'fixed_cagr_frozen_clock': (1 + audit['cumulative_sell']['cagr']) ** ratio - 1,
                     'frozen_fills': frozen.get('fills'), 'replay_fills': audit['as_frozen']['fills'],
                     'frozen_mdd': frozen.get('max_drawdown'), 'replay_mdd': audit['as_frozen']['max_drawdown'],
                     'fixed_mdd': audit['cumulative_sell']['max_drawdown']})
    frame = pd.DataFrame(rows)
    frame['residual_pp'] = (frame.replay_cagr_frozen_clock - frame.frozen_cagr) * 100
    frame['fix_change_pp'] = (frame.fixed_cagr_frozen_clock - frame.replay_cagr_frozen_clock) * 100
    frame.to_csv(OUT / 'level_reconciliation.csv', index=False)
    return frame


if __name__ == '__main__':
    if sys.argv[1] == '--reconcile':
        pd.set_option('display.width', 250)
        print(reconcile().round(4).to_string())
    elif sys.argv[1] == '--recover':
        print(recover(sys.argv[2]))
    else:
        main(sys.argv[1])
