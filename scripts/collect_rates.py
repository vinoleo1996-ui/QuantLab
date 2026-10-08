#!/usr/bin/env python3
"""Treasury yields (Eastmoney) and the ChinaBond AAA corporate vs government curves.

  python scripts/collect_rates.py treasury
  python scripts/collect_rates.py chinabond [first_session] [last_session]

Raw responses are archived under ``data/macro_rates/raw`` before parsing. Both sources publish
end-of-day values, so research code must use a value no earlier than the next session.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data/macro_rates'
RAW = OUT / 'raw'
EAST = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
TREASURY = {'EMM00588704': 'cn_2y', 'EMM00166462': 'cn_5y', 'EMM00166466': 'cn_10y', 'EMM00166469': 'cn_30y'}
CHINABOND = 'https://yield.chinabond.com.cn/cbweb-mn/yc/searchYc'
CURVES = {'2c9081e50a2f9606010a3068cae70001': 'gov', '2c9081e50a2f9606010a309f4af50111': 'corp_aaa'}
TENORS = (1.0, 3.0, 5.0)


def get(url, data=None):
    request = urllib.request.Request(url, data=data, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception:
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f'fetch_failed:{url[:120]}')


def archive(name, payload):
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f'{name}_{hashlib.sha256(payload).hexdigest()[:16]}.json').write_bytes(payload)


def treasury():
    rows, page, pages = [], 1, 1
    while page <= pages:
        query = urllib.parse.urlencode({'reportName': 'RPTA_WEB_TREASURYYIELD', 'columns': 'ALL',
                                        'filter': "(SOLAR_DATE>='2014-01-01')", 'sortColumns': 'SOLAR_DATE',
                                        'sortTypes': 1, 'pageNumber': page, 'pageSize': 500,
                                        'source': 'WEB', 'client': 'WEB'})
        payload = get(f'{EAST}?{query}')
        archive(f'treasury_p{page}', payload)
        result = json.loads(payload)['result']
        pages = result['pages']
        rows += result['data']
        page += 1
        time.sleep(0.5)
    frame = pd.DataFrame(rows).rename(columns=TREASURY)
    frame['date'] = pd.to_datetime(frame.SOLAR_DATE)
    frame = frame[['date', *TREASURY.values()]].dropna(subset=['cn_10y']).drop_duplicates('date').sort_values('date')
    frame.to_parquet(OUT / 'treasury_yields.parquet', index=False)
    print(json.dumps({'rows': len(frame), 'first': str(frame.date.min().date()), 'last': str(frame.date.max().date())}))


def chinabond_day(day):
    cached = sorted(RAW.glob(f'chinabond_{day}_*.json'))
    if cached:
        return cached[-1].read_bytes()
    query = ('xyzSelect=txy&&workTimes={day}&&dxbj=0&&qxll=0,&&yqqxN=N&&yqqxK=K&&ycDefIds={ids}'
             '&&wrjxCBFlag=0&&locale=zh_CN').format(day=day, ids=','.join(CURVES))
    payload = get(f'{CHINABOND}?{query}', data=b'')
    archive(f'chinabond_{day}', payload)
    time.sleep(0.3)
    return payload


def chinabond(first='2014-01-02', last='2026-09-30'):
    from quantlab.smallcap.panel import Panel
    sessions = Panel('20261006T152412Z').sessions
    from concurrent.futures import ThreadPoolExecutor
    days = [str(d.date()) for d in sessions if pd.Timestamp(first) <= d <= pd.Timestamp(last)]
    with ThreadPoolExecutor(4) as pool:
        payloads = list(pool.map(chinabond_day, days))
    rows = []
    for index, (day, payload) in enumerate(zip(days, payloads)):
        for curve in json.loads(payload):
            name = CURVES.get(curve['ycDefId'])
            points = dict(map(tuple, curve.get('seriesData') or []))
            if name:
                rows.append({'date': day, **{f'{name}_{int(t)}y': points.get(t) for t in TENORS}})
        if index % 200 == 0:
            print(json.dumps({'day': day, 'index': index, 'total': len(days)}), flush=True)
    frame = pd.DataFrame(rows).groupby('date').first().reset_index()
    frame['date'] = pd.to_datetime(frame.date)
    frame['credit_spread_3y'] = frame.corp_aaa_3y - frame.gov_3y
    frame.to_parquet(OUT / 'chinabond_curves.parquet', index=False)
    missing = sorted(set(days) - set(frame.dropna(subset=['credit_spread_3y']).date.dt.strftime('%Y-%m-%d')))
    (OUT / 'chinabond_coverage.json').write_text(json.dumps({'sessions': len(days), 'rows': len(frame),
                                                             'missing_spread': missing}, indent=1))
    print(json.dumps({'rows': len(frame), 'missing_spread': len(missing)}))


if __name__ == '__main__':
    {'treasury': treasury, 'chinabond': chinabond}[sys.argv[1]](*sys.argv[2:])
