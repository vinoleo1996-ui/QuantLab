#!/usr/bin/env python3
"""Normalize archived public files; expose missing events and execution assumptions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.data.exchange_rules import limit_ratio,limit_prices,RULE_VERSION


def build(raw,root):
    raw,root=Path(raw),Path(root);root.mkdir(parents=True,exist_ok=True)
    spec=json.loads((raw/'cohort.json').read_text());gaps=[];execution_gaps=[];sources={}
    def read(name):
        path=raw/(name+'.csv');meta=json.loads((raw/(name+'.json')).read_text())
        if meta['error_code']!='0' or hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('invalid_download:'+name)
        sources[name]=meta
        return pd.read_csv(path,dtype={'code':str})
    def write(name,df):
        p=root/(name+'.parquet');df.to_parquet(p)
        return dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    cal=read('calendar');calendar=pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_trading_day==1,'calendar_date']))
    end=pd.Timestamp(spec['end']);rows=[];actions=[];panels={};event_counts={}
    for code in spec['codes']:
        basic=read(code+'_basic')
        if len(basic)!=1:raise ValueError('security_basic_missing:'+code)
        basic=basic.iloc[0];p=read(code+'_daily')
        if p.empty:raise ValueError('price_history_missing_no_replacement:'+code)
        p.index=pd.DatetimeIndex(pd.to_datetime(p.pop('date')));p=p.sort_index()
        p['turnover']=p['turn'];p['reference_price']=p['preclose']
        p['source']='baostock_raw';p['revision']=sources[code+'_daily']['sha256']
        p['quality_status']='ok';p['available_at']=[str(d.date())+' 18:00:00' for d in p.index]
        p['availability_basis']='exchange_daily_close_plus_buffer_assumed_not_vendor_ingest'
        p['open_status_available_at']=[str(d.date())+' 09:30:00' for d in p.index]
        p['open_status_basis']='daily_open_assumption';p['rule_version']=RULE_VERSION
        p['open_status']='unknown';p['open_status_reason']='';p['limit_up']=np.nan;p['limit_down']=np.nan
        # A conservative age bound, not a fabricated exact listing-session count.
        p['listing_session_lower_bound']=np.arange(len(p))+6
        if pd.Timestamp(basic.ipoDate)>pd.Timestamp(spec['start'])-pd.Timedelta(days=180):raise ValueError('cohort_not_seasoned:'+code)
        p['is_st_at_open']=p.isST
        p['limit_ratio_at_open']=[limit_ratio(code,d,bool(r.isST) if pd.notna(r.isST) else np.nan,int(r.listing_session_lower_bound)) for d,r in p.iterrows()]
        prev=None
        for d,r in p.iterrows():
            ratio=r.limit_ratio_at_open
            if np.isfinite(ratio) and np.isfinite(r.preclose) and r.preclose>0:
                p.loc[d,['limit_up','limit_down']]=limit_prices(r.preclose,ratio)
                if r.tradestatus==1 and pd.notna(r.open) and r.open>0:
                    upper,lower=p.loc[d,['limit_up','limit_down']]
                    if lower-.005 <= r.open <= upper+.005:p.loc[d,'open_status']='derived'
                    else:
                        p.loc[d,'open_status_reason']='open_outside_rule_bounds'
                        execution_gaps.append(dict(instrument_id=code,date=str(d.date()),reason='open_outside_rule_bounds',
                            open=float(r.open),reference_price=float(r.preclose),limit_up=float(upper),limit_down=float(lower)))
                elif r.tradestatus==0:p.loc[d,'open_status']='halted'
            if pd.isna(r.isST):gaps.append(dict(instrument_id=code,date=str(d.date()),reason='unknown_daily_st'));continue
            state=(bool(r.isST),float(ratio))
            if state!=prev:
                rows.append(dict(instrument_id=code,available_at=str(d.date())+' 18:00:00',effective_at=str(d.date()),
                    source='baostock_daily_status',revision=p.revision.iloc[0],quality_status='ok',active=True,is_st=state[0],
                    listing_date=basic.ipoDate,sector='UNKNOWN',asset_type='stock',buy_lot=100,sell_lot=100,sell_delay=1,
                    price_tick=.01,commission=.00025,min_commission=5.,limit_ratio=ratio))
                prev=state
        if pd.notna(basic.outDate):
            last=dict(rows[-1]);last.update(active=False,effective_at=basic.outDate,available_at=basic.outDate+' 00:00:00',source='baostock_delisting_lifecycle')
            rows.append(last);gaps.append(dict(instrument_id=code,date=basic.outDate,reason='delisting_settlement_unverified'))
        acts=pd.concat([read(code+'_dividend_'+str(y)) for y in range(pd.Timestamp(spec['start']).year,end.year+1)],ignore_index=True).drop_duplicates()
        event_counts[code]=len(acts)
        for _,a in acts.iterrows():
            if pd.isna(a.dividOperateDate):continue
            ex=pd.Timestamp(a.dividOperateDate)
            if ex>end:continue
            cash=0. if pd.isna(a.dividCashPsBeforeTax) else float(a.dividCashPsBeforeTax)
            bonus=0. if pd.isna(a.dividStocksPs) else float(a.dividStocksPs)
            transfer=0. if pd.isna(a.dividReserveToStockPs) else float(a.dividReserveToStockPs)
            needed=[a.dividRegistDate,a.dividPlanDate]
            if cash:needed.append(a.dividPayDate)
            if bonus+transfer:needed.append(a.dividStockMarketDate)
            if any(pd.isna(x) for x in needed):
                gaps.append(dict(instrument_id=code,date=str(ex.date()),reason='incomplete_distribution_dates'));continue
            announced=pd.Timestamp(a.dividPlanDate);later=calendar[calendar>announced]
            available=str(later[0].date())+' 00:00:00' if len(later) else str(announced+pd.Timedelta(days=1))
            actions.append(dict(action_id=code+':'+str(ex.date()),instrument_id=code,record_date=a.dividRegistDate,
                ex_date=str(ex.date()),pay_date=a.dividPayDate if cash else str(ex.date()),share_date=a.dividStockMarketDate if bonus+transfer else str(ex.date()),
                cash_per_share=cash,share_ratio=bonus+transfer,available_at=available,source='baostock_dividend',revision='public_snapshot',
                quality_status='ok',tax_policy='gross_cn_individual',action_type='distribution'))
        # Reconcile exchange ex-reference against stated distributions, not adjusted closes.
        by_date={pd.Timestamp(a['ex_date']):a for a in actions if a['instrument_id']==code}
        for j in range(1,len(p)):
            d=p.index[j];a=by_date.get(d,{});previous=float(p.close.iloc[j-1])
            expected=(previous-float(a.get('cash_per_share',0)))/(1+float(a.get('share_ratio',0)))
            if abs(expected-float(p.preclose.iloc[j]))>.021:
                gaps.append(dict(instrument_id=code,date=str(d.date()),reason='unexplained_ex_reference',expected=expected,observed=float(p.preclose.iloc[j])))
        panels[code]=write(code,p)
    benchmark=read('benchmark');benchmark.index=pd.DatetimeIndex(pd.to_datetime(benchmark.pop('date')))
    benchmark['available_at']=[str(d.date())+' 18:00:00' for d in benchmark.index]
    benchmark['source']='baostock_index_sh000300';benchmark['revision']=sources['benchmark']['sha256'];benchmark['quality_status']='ok'
    manifest=dict(source='baostock_free_cohort',revision='1',as_of=spec['end'],price_basis='raw',currency='CNY',amount_unit='CNY',turnover_unit='percent',
        universe_coverage='historical',research_scope=spec,actions_coverage='public_reconciled_with_gaps' if gaps else 'public_reconciled',
        benchmark_type='index',benchmark_return_basis='price_return',panels=panels,instruments=write('instruments',pd.DataFrame(rows)),
        actions=write('actions',pd.DataFrame(actions)),calendar=write('calendar',pd.DataFrame({'session':calendar})),benchmark=write('benchmark',benchmark),
        data_gaps=gaps,execution_gaps=execution_gaps,execution_assumption='daily_open_with_rule_limits_not_observed_auction',source_files=sources)
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report=dict(securities=len(panels),bars=sum(m['rows'] for n,m in sources.items() if n.endswith('_daily')),actions=len(actions),
        distribution_rows_by_security=event_counts,gaps=gaps,execution_gaps=execution_gaps,manifest=str(root/'manifest.json'))
    (root/'import_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('gaps','distribution_rows_by_security')}|{'gaps':len(gaps)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',default='data/free_research/raw');p.add_argument('--output',default='data/free_research/dataset')
    a=p.parse_args();build(a.raw,a.output)
