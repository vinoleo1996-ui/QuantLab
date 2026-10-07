#!/usr/bin/env python3
"""Collect Eastmoney datacenter reports month by month (resumable raw pages + one parquet per report).

Usage: collect_info_em.py REPORT [REPORT ...]   (keys of REPORTS; 'all' for every key but margin)
Raw pages: data/smallcap_info/raw/<key>/<YYYY-MM>_<page>.json; tables: data/smallcap_info/<key>.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/smallcap_info'
URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
HEAD = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/'}
START, END = '2017-01', '2026-10'

REPORTS = {
    'margin': ('RPTA_WEB_RZRQ_GGMX', 'DATE', 'DATE,SCODE,RZYE,RZMRE,RZCHE,RQYE,SZ'),
    'billboard': ('RPT_DAILYBILLBOARD_DETAILSNEW', 'TRADE_DATE',
                  'TRADE_DATE,SECURITY_CODE,EXPLANATION,BILLBOARD_NET_AMT,BILLBOARD_DEAL_AMT,FREE_MARKET_CAP'),
    'billboard_inst': ('RPT_ORGANIZATION_TRADE_DETAILSNEW', 'TRADE_DATE',
                       'TRADE_DATE,SECURITY_CODE,BUY_AMT,SELL_AMT,NET_BUY_AMT,FREECAP'),
    'holder_num': ('RPT_HOLDERNUM_DET', 'HOLD_NOTICE_DATE',
                   'HOLD_NOTICE_DATE,END_DATE,SECURITY_CODE,HOLDER_NUM,PRE_HOLDER_NUM,HOLDER_NUM_RATIO'),
    'forecast': ('RPT_PUBLIC_OP_NEWPREDICT', 'NOTICE_DATE',
                 'NOTICE_DATE,REPORT_DATE,SECURITY_CODE,PREDICT_FINANCE_CODE,PREDICT_TYPE,ADD_AMP_LOWER,ADD_AMP_UPPER'),
    'express': ('RPT_FCI_PERFORMANCEE', 'NOTICE_DATE',
                'NOTICE_DATE,REPORT_DATE,SECURITY_CODE,PARENT_NETPROFIT,PARENT_NETPROFIT_SQ,JLRTBZCL,YSTZ'),
    'block_trade': ('RPT_DATA_BLOCKTRADE', 'TRADE_DATE',
                    'TRADE_DATE,SECURITY_CODE,DEAL_AMT,DEAL_PRICE,CLOSE_PRICE,PREMIUM_RATIO'),
    'unlock': ('RPT_LIFT_STAGE', 'FREE_DATE',
               'FREE_DATE,SECURITY_CODE,FREE_SHARES,CURRENT_FREE_SHARES,FREE_RATIO,FREE_SHARES_TYPE'),
    'holder_change': ('RPT_SHARE_HOLDER_INCREASE', 'NOTICE_DATE',
                      'NOTICE_DATE,START_DATE,END_DATE,SECURITY_CODE,HOLDER_NAME,DIRECTION,CHANGE_NUM,CHANGE_FREE_RATIO'),
    'exec_trade': ('RPT_EXECUTIVE_HOLD_DETAILS', 'CHANGE_DATE',
                   'CHANGE_DATE,SECURITY_CODE,PERSON_NAME,CHANGE_SHARES,AVERAGE_PRICE,CHANGE_AMOUNT'),
    'pledge': ('RPT_CSDC_LIST', 'TRADE_DATE', 'TRADE_DATE,SECURITY_CODE,PLEDGE_RATIO,PLEDGE_DEAL_NUM'),
}
UNLOCK_END = '2027-01'


MAX_PAGES = 100
SPLIT_FROM = {'margin': '2021-11'}


def months(start, end):
    return [p.strftime('%Y-%m') for p in pd.period_range(start, end, freq='M')]


def windows(key):
    """(label, first day, last day); months that exceed the 100-page query cap are split into thirds."""
    out = []
    for month in months(START, UNLOCK_END if key == 'unlock' else END):
        period = pd.Period(month, 'M')
        first, last = period.start_time.strftime('%Y-%m-%d'), period.end_time.strftime('%Y-%m-%d')
        if key in SPLIT_FROM and month >= SPLIT_FROM[key]:
            out += [(f'{month}-a', f'{month}-01', f'{month}-10'), (f'{month}-b', f'{month}-11', f'{month}-20'),
                    (f'{month}-c', f'{month}-21', last)]
        else:
            out.append((month, first, last))
    return out


def valid(payload, page):
    """Data page, or the explicit empty-month answer (code 9201) on page 1; anything else is an error."""
    result = payload.get('result') or {}
    if payload.get('success') and result.get('data'):
        return True
    return page == 1 and payload.get('code') == 9201


def fetch(report, columns, col, label, first, last, page):
    params = {'reportName': report, 'columns': columns, 'pageNumber': page, 'pageSize': 500,
              'sortColumns': f'{col},SECURITY_CODE' if 'SECURITY_CODE' in columns else f'{col},SCODE',
              'sortTypes': '1,1', 'filter': f"({col}>='{first}')({col}<='{last}')",
              'source': 'WEB', 'client': 'WEB'}
    for attempt in range(10):
        try:
            response = requests.get(URL, params=params, headers=HEAD, timeout=30)
            payload = response.json()
            if valid(payload, page):
                return payload
        except Exception as error:  # network drops: back off and retry
            payload = {'error': repr(error)}
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'{report} {label} p{page}: {payload}')


def collect(key):
    report, col, columns = REPORTS[key]
    folder = OUT / 'raw' / key
    folder.mkdir(parents=True, exist_ok=True)
    t0, requests_made = time.time(), 0
    for month, first, last in windows(key):
        page, pages = 1, 1
        while page <= pages:
            path = folder / f'{month}_{page:03d}.json'
            payload = json.loads(path.read_text()) if path.exists() else None
            if payload is None or not valid(payload, page):
                payload = fetch(report, columns, col, month, first, last, page)
                path.write_text(json.dumps(payload, ensure_ascii=False))
                requests_made += 1
                time.sleep(0.25)
            pages = ((payload.get('result') or {}).get('pages')) or 0
            if pages > MAX_PAGES:
                raise RuntimeError(f'{key} {month}: {pages} pages exceeds the query cap; split this window')
            page += 1
        print(json.dumps({'key': key, 'month': month, 'pages': pages, 'requests': requests_made,
                          'seconds': round(time.time() - t0)}), flush=True)
    rows, audit = [], {}
    for month, _, _ in windows(key):
        first = json.loads((folder / f'{month}_001.json').read_text())
        result = first.get('result') or {}
        expected, pages, got = result.get('count') or 0, result.get('pages') or 0, 0
        for page in range(1, pages + 1):
            payload = json.loads((folder / f'{month}_{page:03d}.json').read_text())
            if not valid(payload, page):
                raise RuntimeError(f'{key} {month} p{page}: invalid cached page')
            data = payload['result']['data']
            got += len(data)
            rows.extend(data)
        audit[month] = {'expected': expected, 'rows': got, 'pages': pages}
    short = {m: a for m, a in audit.items() if a['rows'] != a['expected']}
    (OUT / f'{key}_audit.json').write_text(json.dumps({'months': audit, 'mismatch': short}, indent=1))
    if short:
        raise RuntimeError(f'{key}: row count mismatch in {sorted(short)}')
    frame = pd.DataFrame(rows).drop_duplicates()
    frame.to_parquet(OUT / f'{key}.parquet', index=False)
    print(json.dumps({'key': key, 'rows': len(frame), 'done': True}), flush=True)


if __name__ == '__main__':
    keys = sys.argv[1:]
    if keys == ['all']:
        keys = [k for k in REPORTS if k != 'margin']
    for key in keys:
        collect(key)
