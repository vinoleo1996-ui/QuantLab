#!/usr/bin/env python3
"""Collect Baostock earnings forecasts and performance express reports for all A shares.

Raw rows are appended per code to ``data/smallcap_events/raw/<kind>.jsonl`` and never
rewritten. A rerun skips codes already present, so the job can resume after a drop.
Publication dates come from the exchange disclosure date Baostock reports; the
intraday release time is unknown, so consumers must trade no earlier than the next
session open.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import baostock as bs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/smallcap_events/raw'
KINDS = {
    'forecast': bs.query_forecast_report,
    'express': bs.query_performance_express_report,
}


def done_codes(path):
    if not path.exists():
        return set()
    codes = set()
    for line in path.read_text().splitlines():
        if line.strip():
            codes.add(json.loads(line)['code'])
    return codes


def fetch(kind, code, start, end):
    result = KINDS[kind](code, start_date=start, end_date=end)
    if result.error_code != '0':
        raise RuntimeError(f'{kind}:{code}:{result.error_code}:{result.error_msg}')
    rows = []
    while result.next():
        rows.append(dict(zip(result.fields, result.get_row_data())))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', required=True, help='text file with one baostock code per line')
    parser.add_argument('--start', default='2014-01-01')
    parser.add_argument('--end', default='2026-09-30')
    args = parser.parse_args()
    codes = [line.strip() for line in Path(args.codes).read_text().splitlines() if line.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    login = bs.login()
    if login.error_code != '0':
        raise SystemExit('baostock_login_failed:' + login.error_msg)
    try:
        for kind in KINDS:
            path = OUT / f'{kind}.jsonl'
            finished = done_codes(path)
            with path.open('a') as handle:
                for index, code in enumerate(codes):
                    if code in finished:
                        continue
                    for attempt in range(3):
                        try:
                            rows = fetch(kind, code, args.start, args.end)
                            break
                        except Exception as error:  # network drops: retry, then record
                            if attempt == 2:
                                rows = None
                                print(json.dumps({'kind': kind, 'code': code, 'error': str(error)}), file=sys.stderr, flush=True)
                            time.sleep(1 + attempt)
                    if rows is None:
                        continue
                    handle.write(json.dumps({'code': code, 'kind': kind, 'rows': rows,
                                             'captured_at': datetime.now(timezone.utc).isoformat()},
                                            ensure_ascii=False) + '\n')
                    handle.flush()
                    if index % 500 == 0:
                        print(json.dumps({'kind': kind, 'index': index, 'total': len(codes)}), flush=True)
    finally:
        bs.logout()


if __name__ == '__main__':
    main()
