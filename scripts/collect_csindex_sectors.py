#!/usr/bin/env python3
"""Fetch CSI sector / thematic index daily bars for the theme-leg experiments.

Universe rules are fixed in output/smallcap_theme_20261008/PREREGISTRATION.md:
  U-A  "中证全指" search results published by 2016-12-31, category 行业/主题,
       excluding 000985 and the level-1 sectors 000986-000995
  U-B  domestic CNY equity 主题 indices tracked by a product today, plus U-A
Raw list and per-index responses are archived before parsing.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/smallcap_assets/csindex/sector_raw'
OUT = ROOT / 'data/smallcap_assets/csindex'
LIST_URL = 'https://www.csindex.com.cn/csindex-home/index-list/query-index-item'
PERF_URL = 'https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode={code}&startDate=20130101&endDate=20261007'
LEVEL1 = {f'0009{n}' for n in range(85, 96)}
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json'}


def request(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method='POST' if body is not None else 'GET')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except Exception:
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f'fetch_failed:{url}')


def listing(search):
    body = {'sorter': {'sortField': 'null', 'sortOrder': None}, 'pager': {'pageNum': 1, 'pageSize': 5000},
            'indexFilter': {k: None for k in ('ifCustomized', 'ifTracked', 'ifWeightCapped', 'indexCompliance',
                                              'hotSpot', 'indexClassify', 'currency', 'region', 'indexSeries')},
            'searchInput': search}
    payload = request(LIST_URL, body)
    name = 'universe_list_all.json' if not search else 'universe_list_quanzhi.json'
    (RAW / name).write_bytes(payload)
    return json.loads(payload)['data']


def universes_from_archive():
    quanzhi = json.loads((RAW / 'universe_list_quanzhi.json').read_bytes())['data']
    everything = json.loads((RAW / 'universe_list_all.json').read_bytes())['data']
    return select(quanzhi, everything)


def universes():
    return select(listing('中证全指'), listing(''))


def select(quanzhi, everything):
    ua = [r for r in quanzhi if r['indexClassify'] in ('行业', '主题') and r['indexCode'] not in LEVEL1
          and r['publishDate'] <= '2016-12-31' and r['currency'] == '人民币']
    ub = [r for r in everything if r['assetsClassify'] == '股票' and r['region'] == '境内' and r['currency'] == '人民币'
          and r['indexClassify'] == '主题' and r['ifTracked'] == '是']
    rows = {}
    for r in ua:
        rows[r['indexCode']] = {'code': r['indexCode'], 'name': r['indexName'], 'publish': r['publishDate'],
                                'series': r['indexSeries'], 'classify': r['indexClassify'], 'ua': True, 'ub': True}
    for r in ub:
        rows.setdefault(r['indexCode'], {'code': r['indexCode'], 'name': r['indexName'], 'publish': r['publishDate'],
                                         'series': r['indexSeries'], 'classify': r['indexClassify'], 'ua': False, 'ub': True})
    return pd.DataFrame(list(rows.values()))


EM_URL = ('https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3'
          '&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&beg=20130101&end=20261007')


def eastmoney(code):
    """Fallback when csindex blocks: Eastmoney daily kline; amount converted to 亿元."""
    archived = sorted(RAW.glob(f'em_{code}_*.json'))
    prefixes = ['0'] if code.startswith('399') else (['1', '2'] if code.startswith('000') else ['2', '1'])
    payload = archived[-1].read_bytes() if archived else None
    if payload is None:
        for prefix in prefixes:
            try:
                candidate = request(EM_URL.format(secid=f'{prefix}.{code}'))
            except RuntimeError:
                continue
            finally:
                time.sleep(0.6)
            data = json.loads(candidate).get('data')
            if data and data.get('klines'):
                payload = candidate
                break
        if payload is None:
            return None
        digest = hashlib.sha256(payload).hexdigest()[:16]
        (RAW / f'em_{code}_{digest}.json').write_bytes(payload)
    rows = [line.split(',') for line in json.loads(payload)['data']['klines']]
    frame = pd.DataFrame({'tradeDate': [r[0].replace('-', '') for r in rows], 'open': [float(r[1]) for r in rows],
                          'close': [float(r[2]) for r in rows], 'tradingValue': [float(r[6]) / 1e8 for r in rows],
                          'consNumber': np.nan})
    frame['code'] = code
    return frame


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    meta = universes() if not (RAW / 'universe_list_all.json').exists() else universes_from_archive()
    codes = list(meta.code) + ['000985']
    frames, failed, misses = [], [], 0
    for i, code in enumerate(codes):
        archived = sorted(RAW.glob(f'{code}_*.json'))
        if archived:
            payload = archived[-1].read_bytes()
        else:
            if misses >= 2:
                failed.append(code)
                continue
            try:
                payload = request(PERF_URL.format(code=code))
                misses = 0
            except RuntimeError:
                failed.append(code)
                misses += 1
                continue
            digest = hashlib.sha256(payload).hexdigest()[:16]
            (RAW / f'{code}_{digest}.json').write_bytes(payload)
            time.sleep(0.8)
        data = json.loads(payload).get('data') or []
        if data:
            frame = pd.DataFrame(data)[['tradeDate', 'open', 'close', 'tradingValue', 'consNumber']]
            frame['code'] = code
            frames.append(frame)
        if i % 25 == 0:
            print(json.dumps({'done': i, 'total': len(codes), 'code': code, 'rows': len(data)}), flush=True)
    source = {f['code'].iloc[0]: 'csindex' for f in frames}
    still = []
    for code in failed:
        frame = eastmoney(code)
        if frame is None:
            still.append(code)
            continue
        frames.append(frame)
        source[code] = 'eastmoney'
    meta['source'] = meta.code.map(source)
    failed = still
    if failed:
        print(json.dumps({'failed': failed}), flush=True)
    bars = pd.concat(frames, ignore_index=True)
    bars['date'] = pd.to_datetime(bars.pop('tradeDate'))
    bars = bars.rename(columns={'tradingValue': 'amount_yi', 'consNumber': 'cons'})
    bars.to_parquet(OUT / 'sector_bars.parquet', index=False)
    meta.to_parquet(OUT / 'sector_meta.parquet', index=False)
    print(json.dumps({'indices': len(meta), 'ua': int(meta.ua.sum()), 'ub_only': int((~meta.ua).sum()),
                      'rows': len(bars), 'codes_with_data': int(bars.code.nunique()), 'failed': failed}))


if __name__ == '__main__':
    main()
