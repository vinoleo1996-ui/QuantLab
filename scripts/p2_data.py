#!/usr/bin/env python3
"""Phase-2 shared data: month-end top-1500 float-cap pool and trailing dividend yield.

  python scripts/p2_data.py

Writes data/broad/{top1500.parquet, dividend_yield.parquet, summary.json}. Inputs are the v4 panel
and the ledger dataset's distribution table (``available_at`` = notice date, next session).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import pandas as pd

import stable_book as book
from quantlab.broad import dividend_yield, month_ends, top_pool
from quantlab.smallcap.panel import Panel

VERSION = '20261006T152412Z'
OUT = ROOT / 'data/broad'


def main():
    panel = Panel(VERSION)
    days = month_ends(panel.sessions, '2015-01-01', '2026-09-22')
    pools = top_pool(panel, days)
    rows = [{'date': day, 'code': code, 'rank': rank + 1} for day, codes in pools.items() for rank, code in enumerate(codes)]
    pool = pd.DataFrame(rows)
    manifest = json.loads(book.MANIFEST.read_text())
    actions = pd.read_parquet(io.BytesIO(book._read_checked(book.MANIFEST.parent, manifest['actions'])))
    actions['instrument_id'] = actions.instrument_id.astype(str)
    yields = dividend_yield(actions, panel.field('close'), days)
    long = yields.stack().rename('dividend_yield').reset_index().rename(columns={'level_0': 'date', 'level_1': 'code'})
    long = long.merge(pool[['date', 'code']], on=['date', 'code'])
    OUT.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(OUT / 'top1500.parquet', index=False)
    long.to_parquet(OUT / 'dividend_yield.parquet', index=False)
    sizes = pool.groupby('date').size()
    payers = long.groupby('date').dividend_yield.apply(lambda s: float((s > 0).mean()))
    summary = {'panel': VERSION, 'actions_sha256': manifest['actions']['sha256'], 'months': len(days),
               'first': str(days[0].date()), 'last': str(days[-1].date()),
               'pool_size_min': int(sizes.min()), 'pool_size_max': int(sizes.max()),
               'payer_share_median': float(payers.median()),
               'dividend_yield_median': float(long.dividend_yield[long.dividend_yield > 0].median()),
               'rule': 'month-end close; SH/SZ, traded, not ST, >=252 traded sessions; top 1500 float cap'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
