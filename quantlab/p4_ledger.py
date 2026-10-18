"""P4 account replay: the existing ``stable_book`` share ledger at the P4 mandate capital.

``stable_book.replay_account`` reads its capital from a module constant. The
constant is swapped only for the duration of one replay and restored after, so
frozen stable_book users are unaffected.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

from quantlab.p4_mandate import MANDATE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import stable_book as book  # noqa: E402
from quantlab.smallcap.panel import resolve_trusted_panel  # noqa: E402


def to_targets(schedule):
    """``{date: {code: weight}}`` to replay rows. ``sell`` lists every name ever targeted and not targeted
    now, so a name that failed to sell (limit-down, suspension) is never share-locked."""
    rows, ever = [], set()
    for day in sorted(schedule):
        weights = {book.canon(code): float(w) for code, w in schedule[day].items() if w > 0}
        rows.append({'signal_date': str(pd.Timestamp(day).date()), 'weights': weights, 'active': list(weights),
                     'sell': sorted(ever - set(weights)), 'sleeve': 0})
        ever |= set(weights)
    return rows


def load_dataset(codes, manifest_path=None):
    from quantlab.data.contracts import MarketDataset
    manifest_path = Path(manifest_path or book.MANIFEST)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())

    def read(entry):
        return pd.read_parquet(io.BytesIO(book._read_checked(root, entry)))

    calendar = pd.DatetimeIndex(book._dated(read(manifest['calendar'])).session)
    panels = {book.canon(c): book._dated(read(manifest['panels'][book.canon(c)]))
              for c in codes if book.canon(c) in manifest['panels']}
    # The extension manifest root predates the delistings file; the panel builds
    # publish it under data/smallcap_v4/<version>/. Fall back to the newest build.
    delist_path = root / 'delistings.parquet'
    if not delist_path.exists():
        # never take the alphabetically last directory: an ad-hoc merge can be newer and wrong
        try:
            delist_path = resolve_trusted_panel()['path'] / 'delistings.parquet'
        except ValueError:
            delist_path = None
        if delist_path is not None and not delist_path.is_file():
            delist_path = None
    if delist_path is not None:
        delistings = pd.read_parquet(delist_path)
        delistings.attrs['source'] = str(delist_path.relative_to(ROOT))
        delistings.attrs['sha256'] = hashlib.sha256(delist_path.read_bytes()).hexdigest()
    else:
        delistings = pd.DataFrame()
    return MarketDataset(panels, read(manifest['instruments']), calendar, read(manifest['actions']),
                         book._dated(read(manifest['benchmark'])), manifest=manifest,
                         delistings=delistings)


def replay(targets, *, start, end, cost_mult=1.0, dataset=None, manifest_path=None, capital_cny=None, gross=None,
           delisting_haircut=0.0, suspension_evidence=None, cash_buffer=None):
    codes = sorted({c for row in targets for c in list(row['weights']) + row['sell']})
    dataset = dataset or load_dataset(codes, manifest_path)
    return book.replay_account(dataset, targets, start=start, end=end, cost_mult=cost_mult,
                               delisting_haircut=delisting_haircut,
                               suspension_evidence=suspension_evidence,
                               capital=capital_cny or MANDATE.capital_cny, gross=gross,
                               cash_buffer=cash_buffer)
