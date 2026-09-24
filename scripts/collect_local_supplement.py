#!/usr/bin/env python3
"""Bounded, resumable public data supplement for the existing local market lake.

The supplied catalog defines the queue, never today's survivors. Every response
is archived before normalization; failed and empty histories stay visible.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import multiprocessing

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.data.local_adapter import canonical_code, _is_stock, verified_csv

FIELDS='date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST,peTTM,pbMRQ'


def collect(database,output,*,start='2014-01-02',end='2026-06-30',budget_seconds=28800,
            reuse_roots=(),max_codes=None,api=None,minimum_interval=1.,worker_id=0,workers=1,include_basic=False):
    import duckdb
    if api is None:
        import baostock as api
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(database),read_only=True)
    symbols=con.execute("SELECT DISTINCT symbol FROM security_master WHERE asset_class='stock' ORDER BY symbol").fetchall();con.close()
    codes=sorted(canonical_code(x[0]) for x in symbols if _is_stock(canonical_code(x[0])))
    if max_codes is not None:codes=codes[:max_codes]
    roots=[output,*map(Path,reuse_roots)]
    specification=dict(start=start,end=end,codes=codes,fields=FIELDS,source='baostock_public',
        membership='all_observed_catalog_not_current_survivors',max_codes=max_codes)
    contract=output/'collection_contract.json'
    if contract.exists() and json.loads(contract.read_text())!=specification:raise ValueError('supplement_contract_is_frozen')
    if not contract.exists():contract.write_text(json.dumps(specification,ensure_ascii=False,indent=2))
    codes=codes[worker_id::workers]
    began=time.monotonic();deadline=began+budget_seconds;counts=dict(completed=0,reused=0,failed=0,unsupported=0,unattempted=0)
    socket.setdefaulttimeout(30)
    login=api.login()
    if str(login.error_code)!='0':raise RuntimeError('baostock_login_failed:'+str(login.error_msg))
    journal=output/('requests.jsonl' if workers==1 else f'requests.{worker_id}.jsonl')
    status_path=output/('collection_status.json' if workers==1 else f'collection_status.{worker_id}.json')
    last_request=[0.]
    def reusable(name):
        for root in roots:
            path=root/(name+'.csv')
            if not path.exists():continue
            frame,meta=verified_csv(path)
            if frame is None or not len(frame):continue
            if name.endswith('_daily'):
                required=set('date,code,open,close,preclose,volume,amount,turn,tradestatus,isST'.split(','))
                if not required.issubset(frame):continue
                # A complete query can return a shorter history for late IPOs;
                # use requested interval evidence, not row endpoints alone.
                a,b=meta.get('requested_start'),meta.get('requested_end')
                cohort=root/'cohort.json'
                if (not a or not b) and cohort.exists():
                    scope=json.loads(cohort.read_text());a,b=scope.get('start'),scope.get('end')
                if not a or not b or pd.Timestamp(a)>pd.Timestamp(start) or pd.Timestamp(b)<pd.Timestamp(end):continue
            return True
        return False
    def request(name,call):
        if reusable(name):return 'reused'
        error=None
        for attempt in range(2):
            if time.monotonic()>=deadline:return 'unattempted'
            try:
                remaining=minimum_interval-(time.monotonic()-last_request[0])
                if remaining>0:time.sleep(remaining)
                last_request[0]=time.monotonic()
                response=call();rows=[]
                while str(response.error_code)=='0' and response.next():rows.append(response.get_row_data())
                if str(response.error_code)!='0':raise RuntimeError(str(response.error_code)+':'+str(response.error_msg))
                frame=pd.DataFrame(rows,columns=response.fields)
                path=output/(name+'.csv');temporary=path.with_suffix('.tmp')
                frame.to_csv(temporary,index=False);digest=hashlib.sha256(temporary.read_bytes()).hexdigest();temporary.replace(path)
                metadata=dict(error_code='0',rows=len(frame),fields=list(frame),source='baostock',
                    requested_start=start,requested_end=end,fetched_at=datetime.now(timezone.utc).isoformat(),sha256=digest)
                path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
                with journal.open('a') as f:f.write(json.dumps(dict(name=name,status='completed',**metadata),ensure_ascii=False)+'\n')
                return 'completed'
            except Exception as exc:
                error=repr(exc)
                if attempt==0:
                    try:api.logout();api.login()
                    except Exception:pass
        with journal.open('a') as f:f.write(json.dumps(dict(name=name,status='failed',error=error,at=datetime.now(timezone.utc).isoformat()))+'\n')
        return 'failed'
    try:
        for index,code in enumerate(codes):
            if time.monotonic()>=deadline:
                counts['unattempted']+=len(codes)-index;break
            if code.startswith('bj.'):
                counts['unsupported']+=1
                with journal.open('a') as f:f.write(json.dumps(dict(code=code,status='unsupported',reason='baostock_bse_unavailable'))+'\n')
                continue
            status=request(code+'_daily',lambda:api.query_history_k_data_plus(code,FIELDS,start_date=start,end_date=end,frequency='d',adjustflag='3'))
            counts[status]+=1
            if include_basic and status in ('completed','reused'):
                request(code+'_basic',lambda:api.query_stock_basic(code=code))
            summary=dict(**counts,processed=index+1,total=len(codes),elapsed_seconds=round(time.monotonic()-began,1),
                         status='collecting',source_database_modified=False)
            status_path.write_text(json.dumps(summary,indent=2))
            if (index+1)%25==0:print(json.dumps(summary),flush=True)
    finally:api.logout()
    summary=dict(**counts,total=len(codes),elapsed_seconds=round(time.monotonic()-began,1),
                 status='budget_exhausted' if counts['unattempted'] else 'completed_with_gaps' if counts['failed'] or counts['unsupported'] else 'completed',
                 source_database_modified=False)
    status_path.write_text(json.dumps(summary,indent=2))
    return summary


def _worker(arguments,queue):
    try:queue.put(dict(worker=arguments['worker_id'],result=collect(**arguments)))
    except BaseException as exc:queue.put(dict(worker=arguments['worker_id'],error=repr(exc)))


def collect_parallel(database,output,**kwargs):
    workers=kwargs.pop('workers',4)
    if workers<1 or workers>4:raise ValueError('workers_must_be_between_one_and_four')
    if workers==1:return collect(database,output,**kwargs)
    context=multiprocessing.get_context('spawn');queue=context.Queue();processes=[]
    for i in range(workers):
        arguments=dict(database=database,output=output,**kwargs,worker_id=i,workers=workers,minimum_interval=float(workers))
        process=context.Process(target=_worker,args=(arguments,queue));process.start();processes.append(process)
        # Stagger sockets and request starts; aggregate request rate <= 1/sec.
        time.sleep(1.)
    results=[]
    try:
        while any(p.is_alive() for p in processes):
            counts={key:0 for key in ['completed','reused','failed','unsupported','unattempted','processed','total']}
            for i in range(workers):
                file=Path(output)/f'collection_status.{i}.json'
                if not file.exists():continue
                try:status=json.loads(file.read_text())
                except json.JSONDecodeError:continue
                for key in counts:counts[key]+=status.get(key,0)
            (Path(output)/'collection_status.json').write_text(json.dumps(dict(**counts,status='collecting',workers=workers),indent=2))
            time.sleep(10.)
        for process in processes:process.join()
        while not queue.empty():results.append(queue.get())
    except KeyboardInterrupt:
        for process in processes:
            if process.is_alive():process.terminate()
        for process in processes:process.join()
        raise
    summary=dict(status='workers_finished',workers=results,source_database_modified=False)
    (Path(output)/'worker_results.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',required=True);p.add_argument('--output',default='data/local_supplement/raw')
    p.add_argument('--start',default='2014-01-02');p.add_argument('--end',default='2026-06-30')
    p.add_argument('--budget-seconds',type=int,default=28800);p.add_argument('--reuse-raw',action='append',default=[])
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--include-basic',action='store_true',help='Also request lifecycle metadata; daily fields have priority')
    p.add_argument('--max-codes',type=int,help='Explicit collection diagnostic limit; no implicit sample')
    args=p.parse_args()
    print(json.dumps(collect_parallel(args.database,args.output,start=args.start,end=args.end,budget_seconds=args.budget_seconds,
        reuse_roots=args.reuse_raw,max_codes=args.max_codes,workers=args.workers,include_basic=args.include_basic),ensure_ascii=False,indent=2))
