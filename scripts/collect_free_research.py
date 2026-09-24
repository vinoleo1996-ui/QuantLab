#!/usr/bin/env python3
"""Resumable public samples from a preregistered historical cohort. No replacements."""
import argparse
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/free_research/raw')
    parser.add_argument('--members', default='output/upgrade_20260924/provider_samples/historical_securities_20140102.csv')
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--start', default='2014-01-02')
    parser.add_argument('--end', default='2026-09-23')
    parser.add_argument('--adapter-path', default=None)
    args = parser.parse_args()
    if args.adapter_path: sys.path.append(args.adapter_path)
    import baostock as bs
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    members = pd.read_csv(args.members, dtype=str)
    # Board and cohort are known at the initial date, not selected by later survival.
    codes = sorted(set(members.code[members.code.str.startswith(('sh.600','sh.601','sz.000','sz.002'))]),
                   key=lambda c: hashlib.sha256(('20260924:'+c).encode()).hexdigest())[:args.count]
    spec = dict(codes=codes, start=args.start, end=args.end, seed=20260924,
                membership_date='2014-01-02', membership_sha256=hashlib.sha256(Path(args.members).read_bytes()).hexdigest(),
                scope='fixed_initial_mainboard_cohort; not all-market or current-survivor selection')
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
