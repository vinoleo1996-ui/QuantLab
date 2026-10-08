#!/usr/bin/env python3
"""Baostock quarterly financial ratios with announcement dates (first-published values).

  python scripts/collect_baostock_fin.py profit|growth|cash_flow [shard] [shards]

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
QUERIES_FIELDS = {
    'profit': ('code', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM',
               'MBRevenue', 'totalShare', 'liqaShare'),
    'growth': ('code', 'pubDate', 'statDate', 'YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI'),
    'cash_flow': ('code', 'pubDate', 'statDate', 'CAToAsset', 'NCAToAsset', 'tangibleAssetToAsset',
                  'ebitToInterest', 'CFOToOR', 'CFOToNP', 'CFOToGr')}


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


def periods(code, basic, first=(2014, 1), last=(2026, 2)):
    """Quarters from one year before listing to one quarter after delisting, within [first, last]."""
    row = basic.get(code, {})
    start, stop = first, last
    if row.get('ipoDate'):
        ipo = pd.Timestamp(row['ipoDate'])
        start = max(first, (ipo.year - 1, (ipo.month - 1) // 3 + 1))
    if row.get('outDate'):
        out = pd.Timestamp(row['outDate']) + pd.offsets.QuarterEnd(1)
        stop = min(last, (out.year, (out.month - 1) // 3 + 1))
    return [(y, q) for y in range(start[0], stop[0] + 1) for q in (1, 2, 3, 4) if start <= (y, q) <= stop]


def main(table, shard=0, shards=1):
    out = ROOT / 'data/baostock' / table
    out.mkdir(parents=True, exist_ok=True)
    codes = [c for c in Panel('20261006T152412Z').codes if not c.startswith('bj.')][shard::shards]
    basic = pd.read_parquet(ROOT / 'data/ipo/baostock_stock_basic.parquet').set_index('code').to_dict('index')
    bs.login()
    started, done = time.time(), 0
    for n, code in enumerate(codes):
        path = out / f'{code}.parquet'
        if path.exists():
            continue
        frames, fields = [], list(QUERIES_FIELDS[table])
        for year, quarter in periods(code, basic):
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
