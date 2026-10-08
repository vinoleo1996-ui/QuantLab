#!/usr/bin/env python3
"""Convertible bonds, listed and delisted: terms list, daily bars and daily conversion value.

  python scripts/collect_cb.py

Sources: Eastmoney RPT_BOND_CB_LIST (terms, incl. delisted), Sina daily bars (incl. delisted),
Eastmoney RPTA_WEB_KZZ_LS (daily conversion value / premium, incl. delisted), CSI convertible
index 000832 from csindex.com.cn. Raw responses are archived under ``data/cb/raw``. Conversion value
and premium are end-of-day figures: research code may use them from the next session.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import collect_rates as rates

OUT = ROOT / 'data/cb'
rates.RAW = OUT / 'raw'
SINA = ('https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
        '?symbol={}&scale=240&ma=no&datalen=10000')
VALUE = ('https://datacenter.eastmoney.com/api/data/get?type=RPTA_WEB_KZZ_LS&sty=ALL&source=WEB'
         '&p=1&ps=8000&st=date&sr=1&filter=(zcode=%22{}%22)')
INDEX = 'https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode=000832&startDate=20140101&endDate=20261008'
TERMS = ['SECURITY_CODE', 'SECUCODE', 'SECURITY_NAME_ABBR', 'CONVERT_STOCK_CODE', 'LISTING_DATE', 'DELIST_DATE',
         'EXPIRE_DATE', 'VALUE_DATE', 'ACTUAL_ISSUE_SCALE', 'RATING', 'INITIAL_TRANSFER_PRICE',
         'TRANSFER_START_DATE', 'TRANSFER_END_DATE', 'REDEEM_CLAUSE', 'RESALE_CLAUSE', 'INTEREST_RATE_EXPLAIN',
         'ISSUE_TYPE', 'PAR_VALUE']


def terms():
    rows, page, pages = [], 1, 1
    while page <= pages:
        payload = rates.get(f'{rates.EAST}?reportName=RPT_BOND_CB_LIST&columns=ALL&pageSize=500&pageNumber={page}'
                            '&sortColumns=PUBLIC_START_DATE&sortTypes=1&quoteColumns=&source=WEB&client=WEB')
        rates.archive(f'list_p{page}', payload)
        result = json.loads(payload)['result']
        pages, page = result['pages'], page + 1
        rows += result['data']
    frame = pd.DataFrame(rows)[TERMS]
    for column in ('LISTING_DATE', 'DELIST_DATE', 'EXPIRE_DATE', 'VALUE_DATE', 'TRANSFER_START_DATE', 'TRANSFER_END_DATE'):
        frame[column] = pd.to_datetime(frame[column], errors='coerce')
    return frame


def symbol(secucode):
    code, market = secucode.split('.')
    return f'{market.lower()}{code}'


def fetch(row):
    code, sym = row
    bars = rates.get(SINA.format(sym))
    rates.archive(f'bars_{code}', bars)
    value = rates.get(VALUE.format(code))
    rates.archive(f'value_{code}', value)
    text = bars.decode('utf8', 'ignore').strip()
    b = pd.DataFrame(json.loads(text) if text.startswith('[') else [])
    if len(b):
        b = b.rename(columns={'day': 'date'}).assign(code=code)
    result = json.loads(value).get('result') or {}
    v = pd.DataFrame(result.get('data') or [])
    if len(v):
        v = v[['DATE', 'SWAPVALUE', 'SWAPOR', 'PUREBONDVALUE']].rename(
            columns={'DATE': 'date', 'SWAPVALUE': 'conv_value', 'SWAPOR': 'premium_pct',
                     'PUREBONDVALUE': 'bond_floor'}).assign(code=code)
    return b, v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    listing = terms()
    listing = listing[listing.LISTING_DATE.notna()]
    listing.to_parquet(OUT / 'terms.parquet', index=False)
    jobs = [(c, symbol(s)) for c, s in zip(listing.SECURITY_CODE, listing.SECUCODE)]
    bars, values, failed = [], [], []
    with ThreadPoolExecutor(4) as pool:
        for (code, _), future in zip(jobs, [pool.submit(fetch, j) for j in jobs]):
            try:
                b, v = future.result()
                bars.append(b), values.append(v)
            except Exception as error:
                failed.append({'code': code, 'error': str(error)[:200]})
    bars = pd.concat([b for b in bars if len(b)], ignore_index=True)
    bars['date'] = pd.to_datetime(bars.date)
    for column in ('open', 'high', 'low', 'close', 'volume'):
        bars[column] = pd.to_numeric(bars[column], errors='coerce')
    bars.to_parquet(OUT / 'bars.parquet', index=False)
    values = pd.concat([v for v in values if len(v)], ignore_index=True)
    values['date'] = pd.to_datetime(values.date)
    values.to_parquet(OUT / 'value.parquet', index=False)
    index = rates.get(INDEX)
    rates.archive('index_000832_csindex', index)
    index = pd.DataFrame(json.loads(index)['data']).rename(columns={'tradeDate': 'date'})
    index['date'] = pd.to_datetime(index.date)
    index[['date', 'open', 'high', 'low', 'close']].astype(
        {c: float for c in ('open', 'high', 'low', 'close')}).to_parquet(OUT / 'index_000832.parquet', index=False)
    coverage = {'bonds_listed': len(listing), 'with_bars': int(bars.code.nunique()),
                'with_value': int(values.code.nunique()), 'failed': failed,
                'delisted': int(listing.DELIST_DATE.notna().sum()),
                'delisted_with_bars': int(listing[listing.DELIST_DATE.notna()].SECURITY_CODE.isin(bars.code).sum()),
                'bars_first': str(bars.date.min().date()), 'bars_last': str(bars.date.max().date())}
    (OUT / 'coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in coverage.items() if k != 'failed'} | {'failed': len(failed)}))


if __name__ == '__main__':
    main()
