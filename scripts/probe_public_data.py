#!/usr/bin/env python3
"""Capture public samples without inventing availability times or qualification."""
import datetime
import hashlib
import json
from pathlib import Path
import socket
import pandas as pd


def main():
    import baostock as bs
    socket.setdefaulttimeout(15)
    out=Path('output/upgrade_20260924/provider_samples');out.mkdir(parents=True,exist_ok=True)
    results=[]
    try:
        login=bs.login();results.append(dict(request='login',code=login.error_code,message=login.error_msg))
        if login.error_code=='0':
            calls=[('stock_daily_raw',lambda:bs.query_history_k_data_plus('sh.600519','date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST',start_date='2014-01-01',end_date='2026-09-24',frequency='d',adjustflag='3')),
                ('benchmark_sh000300',lambda:bs.query_history_k_data_plus('sh.000300','date,code,open,high,low,close,preclose,volume,amount',start_date='2014-01-01',end_date='2026-09-24',frequency='d',adjustflag='3')),
                ('historical_securities_20140102',lambda:bs.query_all_stock(day='2014-01-02')),
                ('dividend_2024',lambda:bs.query_dividend_data(code='sh.600519',year='2024',yearType='report'))]
            for name,call in calls:
                r=call();data=[]
                while r.error_code=='0' and r.next():data.append(r.get_row_data())
                frame=pd.DataFrame(data,columns=r.fields);path=out/(name+'.csv');frame.to_csv(path,index=False)
                results.append(dict(request=name,code=r.error_code,message=r.error_msg,rows=len(frame),fields=r.fields,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status='public_sample_unverified',
                    observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                print(name,len(frame),r.error_code,flush=True)
        bs.logout()
    except Exception as exc:results.append(dict(error=type(exc).__name__,message=str(exc)))
    (out/'manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
