#!/usr/bin/env python3
"""Archive public distribution records and match payment dates without guessing.

Eastmoney supplies report-period batches; CNInfo supplies a security's complete
distribution history in one call. Provider return/performance fields stay in raw
archives and are never normalized into model features.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.data.local_adapter import canonical_code, _is_stock

EAST_URL='https://datacenter-web.eastmoney.com/api/data/v1/get'
CNINFO_URL='https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139'


def archive_json(path,fetch,*,source,params):
    path=Path(path);meta=path.with_suffix('.meta.json')
    if path.exists() and meta.exists():
        raw=path.read_bytes();m=json.loads(meta.read_text())
        if hashlib.sha256(raw).hexdigest()!=m['sha256']:raise ValueError('action_raw_checksum_mismatch:'+str(path))
        return json.loads(raw),m
    payload=fetch();raw=json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(path)
    m=dict(source=source,params=params,fetched_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest())
    meta.write_text(json.dumps(m,ensure_ascii=False,indent=2))
    return payload,m


def collect_eastmoney(root,start,end):
    root=Path(root);periods=pd.period_range(str(pd.Timestamp(start).year-1)+'-01-01',end,freq='Q')
    rows=[];failures=[]
    for period in periods:
        report=str(period.end_time.date());page=1;pages=1
        while page<=pages:
            params=dict(reportName='RPT_SHAREBONUS_DET',columns='ALL',pageSize='500',pageNumber=str(page),
                sortColumns='SECURITY_CODE',sortTypes='1',source='WEB',client='WEB',filter="(REPORT_DATE='"+report+"')")
            path=root/'eastmoney'/(report+'-'+str(page)+'.json')
            def fetch():
                with urllib.request.urlopen(EAST_URL+'?'+urllib.parse.urlencode(params),timeout=30) as response:return json.load(response)
            try:
                payload,meta=archive_json(path,fetch,source=EAST_URL,params=params)
                result=payload.get('result')
                if not payload.get('success') or not result:
                    failures.append(dict(report_period=report,page=page,reason=payload.get('message','empty_result')));break
                pages=int(result['pages'])
                for row in result.get('data',[]):rows.append(dict(raw=row,source_hash=meta['sha256'],fetched_at=meta['fetched_at']))
            except Exception as exc:
                failures.append(dict(report_period=report,page=page,reason=repr(exc)));break
            page+=1;time.sleep(.5)
        print(json.dumps(dict(provider='eastmoney',report_period=report,pages=page-1,records=len(rows))),flush=True)
    report=dict(records=len(rows),failed_pages=failures,start=start,end=end)
    (root/'eastmoney_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return rows


def _existing_east_rows(root):
    rows=[]
    for path in sorted((Path(root)/'eastmoney').glob('*.json')):
        if path.name.endswith('.meta.json'):continue
        meta=json.loads(path.with_suffix('.meta.json').read_text());raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('action_raw_checksum_mismatch')
        for row in (json.loads(raw).get('result') or {}).get('data',[]):
            rows.append(dict(raw=row,source_hash=meta['sha256'],fetched_at=meta['fetched_at']))
    return rows


def _cninfo_worker(arguments):
    root,codes,worker,workers,budget=arguments;root=Path(root)
    # Use the installed provider package for its public client token. No source
    # code is vendored into QuantLab and every raw provider response is retained.
    import requests
    import py_mini_racer
    from akshare.stock.stock_dividend_cninfo import _get_file_content_ths
    js=py_mini_racer.MiniRacer();js.eval(_get_file_content_ths('cninfo.js'))
    session=requests.Session();began=time.monotonic();previous=0.;done=failed=0
    for code in codes[worker::workers]:
        if time.monotonic()-began>budget:break
        params={'scode':code.split('.')[1]};path=root/'cninfo'/(code+'.json')
        def fetch():
            nonlocal previous
            delay=workers-(time.monotonic()-previous)
            if delay>0:time.sleep(delay)
            previous=time.monotonic()
            headers={'Accept-Enckey':js.call('getResCode1'),'Origin':'https://webapi.cninfo.com.cn',
                'Referer':'https://webapi.cninfo.com.cn/','User-Agent':'Mozilla/5.0','X-Requested-With':'XMLHttpRequest'}
            response=session.post(CNINFO_URL,params=params,headers=headers,timeout=30);response.raise_for_status()
            payload=response.json()
            if not isinstance(payload.get('records'),list):raise ValueError('cninfo_records_missing:'+str(payload.get('message')))
            return payload
        try:
            archive_json(path,fetch,source=CNINFO_URL,params=params);done+=1
        except Exception as exc:
            failed+=1
            with (root/f'cninfo_failures.{worker}.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(code=code,error=repr(exc),at=datetime.now(timezone.utc).isoformat()))+'\n')
        if (done+failed)%25==0:
            status=dict(worker=worker,completed=done,failed=failed,total=len(codes[worker::workers]),elapsed_seconds=round(time.monotonic()-began,1))
            (root/f'cninfo_status.{worker}.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    status=dict(worker=worker,completed=done,failed=failed,total=len(codes[worker::workers]),elapsed_seconds=round(time.monotonic()-began,1))
    (root/f'cninfo_status.{worker}.json').write_text(json.dumps(status,indent=2))
    return status


def _date(value):
    if value is None or pd.isna(value) or str(value).strip() in ('','--'):return None
    try:return str(pd.Timestamp(value).date())
    except (ValueError,TypeError):return None


def _number(value):
    try:return float(value) if pd.notna(value) else 0.
    except (ValueError,TypeError):return 0.


def normalize_matches(east_rows,cn_rows,calendar,start,end):
    """Require exact security/ex-date and distribution agreement across sources."""
    by_event={};issues=[];actions=[];east_claims={}
    for item in east_rows:
        row=item['raw'];key=(canonical_code(row.get('SECUCODE','')),_date(row.get('EX_DIVIDEND_DATE')))
        if row.get('ASSIGN_PROGRESS')!='实施分配':continue
        signature=(_date(row.get('REPORT_DATE')),_date(row.get('EQUITY_RECORD_DATE')),
                   _number(row.get('PRETAX_BONUS_RMB')),_number(row.get('BONUS_RATIO')),_number(row.get('IT_RATIO')))
        east_claims.setdefault(key,set()).add(signature)
    for code,payload,meta in cn_rows:
        for r in payload.get('records',[]):
            ex=_date(r.get('F020D'))
            if ex:by_event.setdefault((code,ex),[]).append((r,meta))
    seen=set()
    for item in east_rows:
        r=item['raw'];code=canonical_code(r.get('SECUCODE',''));ex=_date(r.get('EX_DIVIDEND_DATE'))
        if not _is_stock(code) or not ex or not start<=ex<=end or r.get('ASSIGN_PROGRESS')!='实施分配':continue
        event=(code,ex)
        if event in seen:continue
        seen.add(event)
        cash=_number(r.get('PRETAX_BONUS_RMB'))/10.
        shares=(_number(r.get('BONUS_RATIO'))+_number(r.get('IT_RATIO')))/10.
        record=_date(r.get('EQUITY_RECORD_DATE'))
        candidates=by_event.get(event,[]);matched=[]
        for cr,meta in candidates:
            cc=_number(cr.get('F012N'))/10.
            cs=(_number(cr.get('F010N'))+_number(cr.get('F011N')))/10.
            if abs(cc-cash)<=1e-6 and abs(cs-shares)<=1e-6 and _date(cr.get('F018D'))==record:matched.append((cr,meta))
        base=dict(instrument_id=code,date=ex,cash_per_share_reported=cash,share_ratio_reported=shares,eastmoney_hash=item['source_hash'])
        if len(east_claims.get(event,()))>1:
            # Two report periods can pay on one ex-date. Neither dropping one
            # nor summing possibly revised proposals is an audited entitlement.
            issues.append(base|dict(reason='multiple_distribution_records_same_day_not_reconciled'));continue
        if len(matched)!=1:
            issues.append(base|dict(reason='distribution_cross_source_missing_or_conflicting'));continue
        cr,meta=matched[0];pay=_date(cr.get('F023D'));share_date=_date(cr.get('F025D'));announced=_date(cr.get('F006D'))
        if not record or not announced or (cash>0 and not pay) or (shares>0 and not share_date):
            issues.append(base|dict(reason='distribution_payment_or_share_date_missing'));continue
        if pd.Timestamp(record)>=pd.Timestamp(ex) or (pay and pay<ex) or (share_date and share_date<ex):
            issues.append(base|dict(reason='distribution_date_conflict'));continue
        following=calendar[calendar>pd.Timestamp(announced)]
        if not len(following):issues.append(base|dict(reason='announcement_outside_calendar'));continue
        actions.append(dict(action_id=code+':'+ex,instrument_id=code,record_date=record,ex_date=ex,
            pay_date=pay if cash else ex,share_date=share_date if shares else ex,cash_per_share=cash,share_ratio=shares,
            available_at=str(following[0].date())+' 00:00:00',source='eastmoney_distribution+cninfo_payment_dates',
            revision=hashlib.sha256((item['source_hash']+meta['sha256']).encode()).hexdigest(),quality_status='ok',
            tax_policy='gross_cn_individual',action_type='distribution',ingested_at=meta['fetched_at'],
            availability_basis='reported_implementation_notice_date_next_session'))
    return actions,issues


def merge(root,calendar_path,start,end):
    root=Path(root);east=_existing_east_rows(root);cn=[]
    for path in sorted((root/'cninfo').glob('*.json')):
        if path.name.endswith('.meta.json'):continue
        raw=path.read_bytes();meta=json.loads(path.with_suffix('.meta.json').read_text())
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('action_raw_checksum_mismatch')
        cn.append((path.stem,json.loads(raw),meta))
    calendar=pd.DatetimeIndex(pd.to_datetime(Path(calendar_path).read_text().splitlines()))
    actions,issues=normalize_matches(east,cn,calendar,start,end)
    columns=['action_id','instrument_id','record_date','ex_date','pay_date','share_date','cash_per_share','share_ratio',
        'available_at','source','revision','quality_status','tax_policy','action_type','ingested_at','availability_basis']
    path=root/'corporate_actions.csv';pd.DataFrame(actions,columns=columns).to_csv(path,index=False)
    path.with_suffix('.json').write_text(json.dumps(dict(error_code='0',source='eastmoney+cninfo',rows=len(actions),
        requested_start=start,requested_end=end,fetched_at=datetime.now(timezone.utc).isoformat(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
    (root/'action_gaps.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2))
    summary=dict(status='partial_reconciled_actions',actions=len(actions),unresolved_events=len(issues),
        eastmoney_records=len(east),cninfo_securities=len(cn),start=start,end=end,
        rights_offerings_and_delisting_settlement='not_covered',promotion_approved=False)
    (root/'action_status.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/corporate_actions/raw');p.add_argument('--start',default='2014-01-02');p.add_argument('--end',default='2026-06-30')
    p.add_argument('--calendar',required=True);p.add_argument('--phase',choices=['eastmoney','cninfo','merge','all'],default='all')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--budget-seconds',type=int,default=28800)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if args.workers<1 or args.workers>4:raise ValueError('workers_must_be_between_one_and_four')
    if args.phase in ('eastmoney','all'):collect_eastmoney(root,args.start,args.end)
    if args.phase in ('cninfo','all'):
        east=_existing_east_rows(root)
        codes=sorted({canonical_code(x['raw'].get('SECUCODE','')) for x in east
            if x['raw'].get('ASSIGN_PROGRESS')=='实施分配' and (_date(x['raw'].get('EX_DIVIDEND_DATE')) or '')>=args.start
            and (_date(x['raw'].get('EX_DIVIDEND_DATE')) or '')<=args.end})
        codes=[c for c in codes if _is_stock(c)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(_cninfo_worker,(str(root),codes,i,args.workers,args.budget_seconds)) for i in range(args.workers)]
            for f in futures:print(json.dumps(f.result()),flush=True)
    if args.phase in ('merge','all'):print(json.dumps(merge(root,args.calendar,args.start,args.end),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
