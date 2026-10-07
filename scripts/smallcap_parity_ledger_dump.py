#!/usr/bin/env python3
"""Diagnostic only: replay the frozen 2018-extended targets through the research Ledger
and dump daily cash / gross so the fast simulator's exposure can be compared."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd

import stable_book as book

TARGETS = ROOT / 'output/continuous_band_2018ext_20260928/targets.json'
OUT = ROOT / 'output/smallcap_rebuild_20261006/parity'


def main():
    root = book.MANIFEST.parent
    manifest = json.loads(book.MANIFEST.read_text())
    codes = sorted({book.canon(code) for row in json.loads(TARGETS.read_text())['targets']
                    for code in list(row['weights']) + row['sell']})
    calendar = pd.DatetimeIndex(book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['calendar'])))).session)
    instruments = pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['instruments'])))
    actions = pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['actions'])))
    benchmark = book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['benchmark']))))
    panels = {code: book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['panels'][code]))))
              for code in codes}
    from quantlab.data.contracts import MarketDataset
    dataset = MarketDataset(panels, instruments, calendar, actions, benchmark, manifest=manifest)
    result = book.replay_account(dataset, TARGETS, start=book.START, end=book.END)
    rows = []
    for state in result['account']:
        held = sum(q * state['marks'].get(c, 0.0) for c, q in state['quantities'].items())
        rows.append({'date': state['date'], 'nav': state['nav'], 'cash': state['cash'], 'held': held,
                     'names': len(state['quantities'])})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'ledger_states.csv', index=False)
    events = pd.DataFrame([e for e in result.get('events', []) if e.get('type') == 'execution_rejected'])
    if len(events):
        events['reason'].value_counts().to_csv(OUT / 'ledger_rejections.csv')
    fills = pd.DataFrame(result['fills'])
    fills.to_csv(OUT / 'ledger_fills.csv', index=False)
    print(json.dumps(result['summary'], default=str)[:600])


if __name__ == '__main__':
    main()
