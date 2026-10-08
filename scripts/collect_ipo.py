#!/usr/bin/env python3
"""A-share IPO subscription history (SH/SZ, incl. delisted), for the IPO-lottery estimate.

  python scripts/collect_ipo.py

Eastmoney RPTA_APP_IPOAPPLY lists live stocks only. Delisted stocks (union of the Baostock stock
list and the local catalog security master) are filled from Eastmoney F10 RPT_F10_ORG_ISSUEINFO
(issue price, online allotment rate) and Sina daily bars (first-day close / VWAP, open-board day).
Raw responses are archived under ``data/ipo/raw``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import collect_rates as rates

OUT = ROOT / 'data/ipo'
rates.RAW = OUT / 'raw'
CATALOG = Path('/Users/agiuser/Desktop/A股ETF量化策略/data/panel/catalog.duckdb')
LIST = ('https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPTA_APP_IPOAPPLY&columns=ALL'
        '&pageSize=500&pageNumber={}&sortColumns=APPLY_DATE,SECURITY_CODE&sortTypes=-1,-1&source=WEB&client=WEB')
F10 = ('https://datacenter.eastmoney.com/securities/api/data/v1/get?reportName=RPT_F10_ORG_ISSUEINFO'
       '&columns=ALL&filter=(SECUCODE=%22{}%22)&source=HSF10&client=PC')
SINA = ('https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
        '?symbol={}&scale=240&ma=no&datalen=10000')
KEEP = ['SECUCODE', 'SECURITY_NAME', 'APPLY_DATE', 'LISTING_DATE', 'TRADE_MARKET', 'MARKET', 'ISSUE_PRICE',
        'ONLINE_ISSUE_LWR', 'EACHBALLOT_SHARES', 'ONLINE_APPLY_UPPER', 'TOP_APPLY_MARKETCAP', 'LD_CLOSE_CHANGE',
        'LD_AVERAGE_PRICE', 'OPEN_DATE', 'OPEN_AVERAGE_PRICE', 'CONTINUOUS_1WORD_NUM', 'IS_BEIJING']


def live():
    rows, page, pages = [], 1, 1
    while page <= pages:
        payload = rates.get(LIST.format(page))
        rates.archive(f'ipoapply_p{page}', payload)
        result = json.loads(payload)['result']
        pages, page = result['pages'], page + 1
        rows += result['data']
    return pd.DataFrame(rows).drop_duplicates('SECUCODE')[KEEP].assign(source='eastmoney_ipoapply')


def delisted_codes(live_codes):
    basic = pd.read_parquet(OUT / 'baostock_stock_basic.parquet')
    basic = basic[(basic.type == '1') & (basic.status == '0') & (basic.ipoDate >= '2015-06-01')]
    codes = set(basic.code.str[3:] + '.' + basic.code.str[:2].str.upper())
    with duckdb.connect(str(CATALOG), read_only=True) as con:
        master = con.sql("select symbol from security_master where asset_class='stock' group by 1 "
                         "having max(delisted_on) is not null and max(listed_on) >= '2015-06-01'").df()
    codes |= set(master.symbol)
    return sorted(c for c in codes if not c.endswith('.BJ') and c not in live_codes)


def fill(secucode):
    payload = rates.get(F10.format(secucode))
    rates.archive(f'f10_{secucode}', payload)
    info = ((json.loads(payload).get('result') or {}).get('data') or [{}])[0]
    code, market = secucode.split('.')
    bars = rates.get(SINA.format(f'{market.lower()}{code}'))
    rates.archive(f'bars_{secucode}', bars)
    text = bars.decode('utf8', 'ignore').strip()
    b = pd.DataFrame(json.loads(text) if text.startswith('[') else [])
    row = {'SECUCODE': secucode, 'ISSUE_PRICE': info.get('ISSUE_PRICE'),
           'ONLINE_ISSUE_LWR': info.get('ONLINE_ISSUE_LWR'), 'LISTING_DATE': info.get('LISTING_DATE'),
           'TRADE_MARKET': '上海证券交易所' if market == 'SH' else '深圳证券交易所', 'IS_BEIJING': 0,
           'source': 'eastmoney_f10+sina_bars'}
    if not len(b) or info.get('LISTING_DATE') is None or not info.get('ISSUE_PRICE'):
        return row
    b[['open', 'high', 'low', 'close', 'volume']] = b[['open', 'high', 'low', 'close', 'volume']].astype(float)
    b = b[b.day >= info['LISTING_DATE'][:10]].reset_index(drop=True)
    if not len(b) or b.day[0] != info['LISTING_DATE'][:10]:
        return row
    first = b.iloc[0]
    row['LD_CLOSE_CHANGE'] = (first.close / row['ISSUE_PRICE'] - 1) * 100
    row['LD_AVERAGE_PRICE'] = (first.open + first.high + first.low + first.close) / 4
    one_word = (b.high == b.low) & (b.index > 0)
    if first.high == first.low or (len(b) > 1 and one_word.iloc[1]):
        later = b[(b.index > 0) & (b.high > b.low)]
        if len(later):
            opened = later.iloc[0]
            row['OPEN_DATE'] = opened.day
            row['OPEN_AVERAGE_PRICE'] = (opened.open + opened.high + opened.low + opened.close) / 4
            row['CONTINUOUS_1WORD_NUM'] = str(int(opened.name))
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = live()
    gaps = delisted_codes(set(frame.SECUCODE))
    filled = pd.DataFrame([fill(code) for code in gaps])
    out = pd.concat([frame, filled], ignore_index=True)
    out.to_parquet(OUT / 'ipo.parquet', index=False)
    summary = {'rows': len(out), 'live': len(frame), 'delisted_filled': len(filled),
               'delisted_with_sell_price': int(filled.LD_CLOSE_CHANGE.notna().sum()) if len(filled) else 0,
               'delisted_codes': gaps}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'delisted_codes'}))


if __name__ == '__main__':
    main()
