#!/usr/bin/env python3
"""Baostock quarterly financial ratios with announcement dates (first-published values).

  python scripts/collect_baostock_fin.py profit|growth|cash_flow [first_year] [last_year]

One parquet per stock under ``data/baostock/<table>/``; existing files are skipped, so the run
resumes. Baostock keeps one record per period (no correction history). Research code must use a
record from max(pubDate, CNINFO date) + 1 session.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import baostock as bs

from quantlab.smallcap.panel import Panel

QUERIES = {'profit': bs.query_profit_data, 'growth': bs.query_growth_data, 'cash_flow': bs.query_cash_flow_data}


def rows(query, code, year, quarter):
    for attempt in range(5):
        rs = query(code=code, year=year, quarter=quarter)
        if rs.error_code == '0':
            out = []
            while rs.next():
                out.append(rs.get_row_data())
            return rs.fields, out
        time.sleep(2 + 3 * attempt)
        bs.logout()
        bs.login()
    raise RuntimeError(f'query_failed:{code}:{year}Q{quarter}:{rs.error_msg}')


def main(table, first_year=2014, last_year=2026):
    out = ROOT / 'data/baostock' / table
    out.mkdir(parents=True, exist_ok=True)
    codes = [c for c in Panel('20261006T152412Z').codes if not c.startswith('bj.')]
    bs.login()
    started, done = time.time(), 0
    for n, code in enumerate(codes):
        path = out / f'{code}.parquet'
        if path.exists():
            continue
        frames, fields = [], None
        for year in range(first_year, last_year + 1):
            for quarter in (1, 2, 3, 4):
                fields, data = rows(QUERIES[table], code, year, quarter)
                frames += data
        pd.DataFrame(frames, columns=fields).to_parquet(path, index=False)
        done += 1
        if done % 100 == 0:
            print(json.dumps({'table': table, 'index': n + 1, 'total': len(codes),
                              'minutes': round((time.time() - started) / 60, 1)}), flush=True)
    bs.logout()
    print(json.dumps({'table': table, 'complete': True, 'files': len(list(out.glob('*.parquet')))}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], *map(int, sys.argv[2:4]))
