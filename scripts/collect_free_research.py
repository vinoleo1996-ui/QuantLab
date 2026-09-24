#!/usr/bin/env python3
"""Collect a declared historical membership catalog; sampling requires --count."""
import argparse
import hashlib
import json
import socket
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def select_member_codes(members, count=None):
    """No initial-year or mainboard restriction, and no implicit sample count.

    The supplied catalog still needs lifecycle/coverage verification. Including a
    code here does not certify provider or exchange-rule support for that code.
    """
    if 'code' not in members: raise ValueError('membership_catalog_requires_code')
    if count is not None and count < 1: raise ValueError('diagnostic_count_must_be_positive')
    def canonical(value):
        value = str(value).lower()
        if re.fullmatch(r'\d{6}\.(sh|sz|bj)',value):
            symbol,exchange=value.split('.');value=exchange+'.'+symbol
        return value
    codes = {canonical(c) for c in members.code.dropna()}
    pattern = r'(sh\.(600|601|603|605|688|689)\d{3}|sz\.(000|001|002|003|300|301)\d{3}|bj\.(4\d{5}|8\d{5}|92\d{4}))'
    codes = sorted(c for c in codes if re.fullmatch(pattern,c))
    if not codes: raise ValueError('membership_catalog_has_no_a_shares')
    if count is not None:
        codes = sorted(codes,key=lambda c:hashlib.sha256(('20260924:'+c).encode()).hexdigest())[:count]
    return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/free_market/raw')
    parser.add_argument('--members', required=True,help='Declared historical catalog including later listings and delistings')
    parser.add_argument('--count', type=int, default=None,help='Explicit diagnostic sample only; omitted collects all supplied A-share codes')
    parser.add_argument('--start', default='2014-01-02')
    parser.add_argument('--end', default='2026-09-23')
    parser.add_argument('--adapter-path', default=None)
    args = parser.parse_args()
    if args.adapter_path: sys.path.append(args.adapter_path)
    import baostock as bs
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    members = pd.read_csv(args.members, dtype=str)
    codes = select_member_codes(members,args.count)
    spec = dict(codes=codes, start=args.start, end=args.end, seed=20260924,
                membership_sha256=hashlib.sha256(Path(args.members).read_bytes()).hexdigest(),
                membership_path=str(Path(args.members).resolve()),sample_size=args.count,
                supplied_catalog_rows=len(members),market_coverage='not_established',
                scope='fixed_historical_diagnostic_sample' if args.count is not None else 'provided_historical_members_no_sampling')
    path = root/'cohort.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise ValueError('cohort_is_frozen')
    path.write_text(json.dumps(spec, indent=2))
    socket.setdefaulttimeout(20)
    login = bs.login()
    if login.error_code != '0': raise RuntimeError('public_login_failed:'+login.error_code)
    def query(name, call):
        p = root/(name+'.csv'); meta = root/(name+'.json')
        if p.exists() and meta.exists():
            m=json.loads(meta.read_text())
            if m.get('error_code')=='0' and hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']:return
        try:
            response=call(); rows=[]
            while response.error_code=='0' and response.next(): rows.append(response.get_row_data())
            pd.DataFrame(rows,columns=response.fields).to_csv(p,index=False)
            m=dict(error_code=response.error_code,error_msg=response.error_msg,rows=len(rows),fields=response.fields,
                   source='baostock',fetched_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        except Exception as exc: m=dict(error_code='exception',error_msg=str(exc))
        meta.write_text(json.dumps(m,ensure_ascii=False,indent=2))
        print(name,m.get('rows'),m['error_code'],flush=True)
    try:
        query('calendar',lambda:bs.query_trade_dates(start_date=args.start,end_date=str(pd.Timestamp(args.end).date()+pd.Timedelta(days=10))))
        fields='date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST'
        query('benchmark',lambda:bs.query_history_k_data_plus('sh.000300','date,code,open,high,low,close,preclose,volume,amount',start_date=args.start,end_date=args.end,frequency='d',adjustflag='3'))
        for code in codes:
            query(code+'_basic',lambda:bs.query_stock_basic(code=code))
            query(code+'_daily',lambda:bs.query_history_k_data_plus(code,fields,start_date=args.start,end_date=args.end,frequency='d',adjustflag='3'))
            for year in range(pd.Timestamp(args.start).year,pd.Timestamp(args.end).year+1):
                query(code+'_dividend_'+str(year),lambda:bs.query_dividend_data(code=code,year=str(year),yearType='operate'))
    finally: bs.logout()


if __name__=='__main__':main()
