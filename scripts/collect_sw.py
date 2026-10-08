#!/usr/bin/env python3
"""Shenwan level-1 industry indices (daily history) and current constituents.

  python scripts/collect_sw.py

Source: swsresearch.com public index API. Raw responses go to ``data/sw/raw``. Constituent lists
are the current (SW 2021) membership only; delisted stocks are absent, so stock-level backtests
carry survivorship bias. Index levels are official end-of-day values.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/sw'
RAW = OUT / 'raw'
API = 'https://www.swsresearch.com/institute-sw/api/index_publish'
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.swsresearch.com/institute_sw/allIndex/releasedIndex'}


def get(url, name):
    """System curl verifies the site's chain (Python's bundle lacks its intermediate); TLS checks stay on."""
    for attempt in range(4):
        try:
            payload = subprocess.run(['curl', '-sf', '-m', '60', '-A', HEADERS['User-Agent'], '-e', HEADERS['Referer'], url],
                                     capture_output=True, check=True).stdout
            RAW.mkdir(parents=True, exist_ok=True)
            (RAW / f'{name}_{hashlib.sha256(payload).hexdigest()[:16]}.json').write_bytes(payload)
            return json.loads(payload)['data']
        except Exception:
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f'fetch_failed:{url}')


def main():
    level1 = get(f'{API}/current/?page=1&page_size=50&indextype=%E4%B8%80%E7%BA%A7%E8%A1%8C%E4%B8%9A', 'level1')['results']
    names = {row['swindexcode']: row['swindexname'] for row in level1}
    bars, members = [], []
    for code, name in names.items():
        rows = get(f'{API}/trend/?swindexcode={code}&period=DAY', f'trend_{code}')
        bars.append(pd.DataFrame(rows).assign(name=name))
        page = 1
        while True:
            data = get(f'{API}/details/component_stocks/?swindexcode={code}&page={page}&page_size=200', f'members_{code}_p{page}')
            members += [dict(r, swindexcode=code, name=name) for r in data['results']]
            if not data['next']:
                break
            page += 1
        time.sleep(0.3)
    bars = pd.concat(bars, ignore_index=True).rename(columns={'bargaindate': 'date'})
    bars['date'] = pd.to_datetime(bars.date)
    bars.to_parquet(OUT / 'level1_bars.parquet', index=False)
    members = pd.DataFrame(members)
    members.to_parquet(OUT / 'level1_members.parquet', index=False)
    summary = {'industries': len(names), 'bars': len(bars), 'first': str(bars.date.min().date()),
               'last': str(bars.date.max().date()), 'members': len(members)}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
