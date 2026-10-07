#!/usr/bin/env python3
"""Fetch daily ETF bars (raw and back-adjusted) from Tencent, one call per year.

Raw response bytes are archived under ``data/smallcap_assets/raw`` before parsing.
The output parquet has one row per (code, date) with ``close_raw`` and
``close_hfq``; total-return series must use ``close_hfq`` ratios.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/smallcap_assets'
URL = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,{start},{end},640,{fq}'
ETFS = {
    'sh518880': 'gold_etf',
    'sh511260': 'treasury_10y_etf',
    'sh513100': 'nasdaq100_qdii_etf',
    'sh510300': 'csi300_etf',
    'sh512100': 'csi1000_etf',
    'sh511010': 'treasury_5y_etf',
    'sh512480': 'semiconductor_etf',
    'sh512880': 'securities_etf',
    'sh512660': 'defense_etf',
}


def fetch(code, year, fq):
    url = URL.format(code=code, start=f'{year}-01-01', end=f'{year}-12-31', fq=fq)
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError(f'fetch_failed:{code}:{year}:{fq}')


def parse(payload, code, fq):
    data = json.loads(payload)['data']
    if not isinstance(data, dict) or code not in data:
        return []
    key = {'': 'day', 'hfq': 'hfqday'}[fq]
    rows = data[code].get(key) or data[code].get('day') or []
    return [{'date': row[0], 'open': float(row[1]), 'close': float(row[2]), 'high': float(row[3]),
             'low': float(row[4]), 'volume': float(row[5])} for row in rows]


def main(first_year=2014, last_year=2026):
    raw = OUT / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    frames = []
    for code, name in ETFS.items():
        parts = {}
        for fq in ('', 'hfq'):
            rows = []
            for year in range(first_year, last_year + 1):
                archived = sorted(raw.glob(f'{code}_{fq or "raw"}_{year}_*.json'))
                if archived:
                    payload = archived[-1].read_bytes()
                else:
                    payload = fetch(code, year, fq)
                    digest = hashlib.sha256(payload).hexdigest()[:16]
                    (raw / f'{code}_{fq or "raw"}_{year}_{digest}.json').write_bytes(payload)
                    time.sleep(0.5)
                rows += parse(payload, code, fq)
            parts[fq or 'raw'] = pd.DataFrame(rows).drop_duplicates('date').set_index('date')
        frame = parts['raw'][['open', 'close', 'volume']].rename(columns={'open': 'open_raw', 'close': 'close_raw'})
        frame = frame.join(parts['hfq'][['open', 'close']].rename(columns={'open': 'open_hfq', 'close': 'close_hfq'}), how='outer')
        frame['code'] = code
        frame['asset'] = name
        frames.append(frame.reset_index())
        print(json.dumps({'code': code, 'rows': len(frame), 'first': frame.index.min() if len(frame) else None}), flush=True)
    result = pd.concat(frames, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    result.to_parquet(OUT / 'etf_bars.parquet', index=False)
    print(json.dumps({'rows': len(result), 'path': str(OUT / 'etf_bars.parquet')}))


if __name__ == '__main__':
    main(*map(int, sys.argv[1:3])) if len(sys.argv) > 2 else main()
