"""Export the existing local market lake without modifying or certifying it.

The lake is a source of historical observations. Qlib interval endpoints are not
IPO/delisting dates, adjustment factors are not cash distributions, and an
industry effective date is not a publication timestamp.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import tarfile
from functools import lru_cache

import numpy as np
import pandas as pd

from quantlab.data.contracts import fingerprint
from quantlab.data.exchange_rules import RULE_VERSION, limit_prices, limit_ratio
from quantlab.data.industry import INDUSTRY_MODES


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def canonical_code(symbol):
    symbol=str(symbol).lower()
    if re.fullmatch(r'\d{6}\.(sh|sz|bj)',symbol):
        code,market=symbol.split('.');return market+'.'+code
    if re.fullmatch(r'(sh|sz|bj)\d{6}',symbol):return symbol[:2]+'.'+symbol[2:]
    return symbol


def _is_stock(code):
    return bool(re.fullmatch(r'(sh\.(600|601|603|605|688|689)\d{3}|sz\.(000|001|002|003|300|301)\d{3}|bj\.\d{6})',code))


def verified_csv(path):
    """Only read archived responses whose sidecar still matches the bytes."""
    path=Path(path);sidecar=path.with_suffix('.json')
    if not path.exists() or not sidecar.exists():return None,None
    meta=json.loads(sidecar.read_text())
    if meta.get('error_code')!='0' or meta.get('sha256')!=sha256_file(path):
        raise ValueError('raw_cache_integrity_failed:'+str(path))
    try: frame=pd.read_csv(path,dtype={'code':str})
    except pd.errors.EmptyDataError: frame=pd.DataFrame()
    return frame,meta


def _cache_index(roots):
    result={};files={}
    for root in map(Path,roots):
        if not root.exists():continue
        for path in sorted(root.glob('*.csv')):
            if path.name in files:continue
            if path.name in ('calendar.csv','benchmark.csv','corporate_actions.csv') or re.match(r'(sh|sz|bj)\.\d{6}_',path.name):
                files[path.name]=path
    for name,path in files.items():
        # Hashes also bind empty distributions and basic records to this export.
        sidecar=path.with_suffix('.json')
        if sidecar.exists():
            meta=json.loads(sidecar.read_text())
            if meta.get('error_code')=='0':result[name]=dict(path=str(path.resolve()),sha256=meta.get('sha256'))
    return result


def qlib_prices(root,symbol,start,end):
    """Decode existing Qlib units, retaining the adjustment snapshot separately."""
    root=Path(root);folder=root/'features'/canonical_code(symbol).replace('.','')
    if not folder.exists():return pd.DataFrame()
    calendar=_qlib_calendar(str(root))
    fields={}
    for name in ('open','high','low','close','volume','amount','factor','change'):
        path=folder/(name+'.day.bin')
        if not path.exists():return pd.DataFrame()
        values=np.fromfile(path,dtype='<f4')
        if not len(values):return pd.DataFrame()
        offset=int(values[0]);data=values[1:]
        if offset<0 or offset+len(data)>len(calendar):raise ValueError('invalid_qlib_offset:'+str(path))
        fields[name]=pd.Series(data,index=calendar[offset:offset+len(data)])
    frame=pd.DataFrame(fields).loc[pd.Timestamp(start):pd.Timestamp(end)].astype(float)
    valid=frame[['open','high','low','close','factor']].gt(0).all(axis=1)&frame.change.gt(-1)
    frame=frame.loc[valid].copy()
    frame['feature_close']=frame.close
    for name in ('open','high','low','close'):frame[name]=frame[name]/frame.factor
    frame['volume']=(frame.volume*frame.factor*100).round()
    frame['amount']=frame.amount*1000
    frame['reference_price']=frame.close/(1+frame.change)
    return frame


@lru_cache(maxsize=4)
def _qlib_calendar(root):
    return pd.DatetimeIndex(pd.to_datetime((Path(root)/'calendars/day.txt').read_text().splitlines()))


def verify_qlib_tree(root,archive):
    """Bind the decoded files actually read to the checked compressed archive."""
    root=Path(root);names={name+'.day.bin' for name in ('open','high','low','close','volume','amount','factor','change')}
    expected={str(p.relative_to(root)) for p in (root/'features').glob('*/*.bin') if p.name in names}
    expected.add('calendars/day.txt');verified=set()
    with tarfile.open(archive,'r|gz') as bundle:
        for member in bundle:
            parts=Path(member.name).parts
            if not parts or parts[0]!='qlib_bin':continue
            relative=str(Path(*parts[1:]))
            if relative not in expected:continue
            if not member.isfile():raise ValueError('qlib_source_is_not_regular_file:'+relative)
            content=bundle.extractfile(member)
            if content is None:raise ValueError('qlib_archive_member_missing:'+relative)
            digest=hashlib.sha256()
            for block in iter(lambda:content.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=sha256_file(root/relative):raise ValueError('qlib_decoded_checksum_mismatch:'+relative)
            verified.add(relative)
    if verified!=expected:raise ValueError('qlib_decoded_files_not_bound_to_archive')
    return len(verified)


def _distributions(code,cache,calendar,end):
    actions=[];issues=[]
    for name,entry in sorted(cache.items()):
        if not name.startswith(code+'_dividend_'):continue
        frame,meta=verified_csv(entry['path'])
        if frame is None:continue
        for _,row in frame.iterrows():
            ex=row.get('dividOperateDate')
            if pd.isna(ex) or pd.Timestamp(ex)>end:continue
            cash=float(row.dividCashPsBeforeTax) if pd.notna(row.get('dividCashPsBeforeTax')) else 0.
            bonus=sum(float(row.get(k)) if pd.notna(row.get(k)) else 0. for k in ('dividStocksPs','dividReserveToStockPs'))
            needed=['dividRegistDate','dividPlanDate']+(['dividPayDate'] if cash else [])+(['dividStockMarketDate'] if bonus else [])
            if any(pd.isna(row.get(k)) for k in needed):
                issues.append(dict(instrument_id=code,date=str(ex),reason='incomplete_distribution_dates',
                    cash_per_share_reported=cash,share_ratio_reported=bonus,
                    record_date=row.get('dividRegistDate') if pd.notna(row.get('dividRegistDate')) else None,
                    source='baostock_dividend',revision=meta['sha256']));continue
            announced=pd.Timestamp(row.dividPlanDate);following=calendar[calendar>announced]
            if not len(following):continue
            actions.append(dict(action_id=code+':'+str(ex),instrument_id=code,record_date=row.dividRegistDate,
                ex_date=ex,pay_date=row.dividPayDate if cash else ex,share_date=row.dividStockMarketDate if bonus else ex,
                cash_per_share=cash,share_ratio=bonus,available_at=str(following[0].date())+' 00:00:00',
                source='baostock_dividend',revision=meta['sha256'],quality_status='ok',tax_policy='gross_cn_individual',
                action_type='distribution',ingested_at=meta.get('fetched_at'),availability_basis='reported_plan_date_next_session'))
    return actions,issues


def _action_gap_inputs(roots,start,end):
    """Bind known unresolved events even when their price difference is tiny."""
    sources=[];by_code=defaultdict(list);seen=set()
    for root in map(Path,roots):
        path=(root/'action_gaps.json').resolve()
        if path in seen or not path.exists():continue
        seen.add(path);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        sidecar=path.with_suffix('.meta.json')
        if sidecar.exists() and json.loads(sidecar.read_text()).get('sha256')!=digest:
            raise ValueError('action_gap_checksum_mismatch:'+str(path))
        rows=json.loads(raw)
        if not isinstance(rows,list):raise ValueError('invalid_action_gap_records:'+str(path))
        sources.append(dict(path=str(path),sha256=digest))
        for row in rows:
            if not isinstance(row,dict):raise ValueError('invalid_action_gap_record:'+str(path))
            code=canonical_code(row.get('instrument_id',''));date=pd.Timestamp(row.get('date'))
            if not _is_stock(code) or pd.isna(date) or not isinstance(row.get('reason'),str):
                raise ValueError('invalid_action_gap_record:'+str(path))
            for field in ('cash_per_share_reported','share_ratio_reported'):
                value=row.get(field)
                if value is not None and (not np.isfinite(float(value)) or float(value)<0):
                    raise ValueError('invalid_action_gap_amount:'+str(path))
            if start<=date<=end:
                by_code[code].append(dict(row,instrument_id=code,date=str(date.date()),gap_source_sha256=digest))
    return by_code,sources


def _distribution_gap_resolved(gap,action,ambiguous_events=()):
    """Only a matching complete entitlement resolves missing-date evidence.

    A shared ex-date alone cannot identify two different report periods. Keep
    explicit conflicts and multi-record events until they are reconciled.
    """
    if action is None or str(pd.Timestamp(gap['date']).date()) in ambiguous_events:return False
    try:cash,shares=float(action['cash_per_share']),float(action['share_ratio'])
    except (KeyError,TypeError,ValueError):return False
    if not np.isfinite([cash,shares]).all() or min(cash,shares)<0:return False
    required=['record_date','ex_date','available_at']
    if cash>0:required.append('pay_date')
    if shares>0:required.append('share_date')
    if action.get('quality_status')!='ok' or any(pd.isna(action.get(field)) for field in required):return False
    if gap.get('reason') not in ('incomplete_distribution_dates','distribution_payment_or_share_date_missing'):
        return False
    if canonical_code(gap['instrument_id'])!=action['instrument_id'] or pd.Timestamp(gap['date'])!=pd.Timestamp(action['ex_date']):
        return False
    for reported,actual in (('cash_per_share_reported','cash_per_share'),('share_ratio_reported','share_ratio')):
        if gap.get(reported) is None or abs(float(gap[reported])-float(action[actual]))>1e-8:return False
    record=gap.get('record_date')
    if pd.notna(record) and pd.Timestamp(record)!=pd.Timestamp(action['record_date']):return False
    return True


def normalize_panel(code,frame,states,cache,calendar,*,qlib_hash=None):
    """Attach observed fields. Unknown values remain unknown, even in research."""
    frame=frame.copy().sort_index()
    frame=frame[~frame.index.duplicated(keep='last')]
    frame['turnover']=np.nan;frame['is_st_at_open']=np.nan;frame['tradestatus']=np.nan
    frame['state_source']='unknown';frame['turnover_source']='unknown'
    if states is not None and len(states):
        s=states.copy();s.index=pd.DatetimeIndex(pd.to_datetime(s.effective_on))
        s=s[~s.index.duplicated(keep='last')].reindex(frame.index)
        frame['is_st_at_open']=pd.to_numeric(s.is_st,errors='coerce')
        frame['tradestatus']=s.is_suspended.map({True:0.,False:1.})
        frame['state_source']=s.source_name.fillna('unknown')
    basic=None;daily_meta=None
    entry=cache.get(code+'_basic.csv')
    if entry:
        table,_=verified_csv(entry['path'])
        if table is not None and len(table)==1:basic=table.iloc[0]
    entry=cache.get(code+'_daily.csv')
    if entry:
        daily,daily_meta=verified_csv(entry['path'])
        if daily is not None and len(daily):
            daily.index=pd.DatetimeIndex(pd.to_datetime(daily.date));d=daily.reindex(frame.index)
            for target,source in [('turnover','turn'),('is_st_at_open','isST'),('tradestatus','tradestatus')]:
                if source in d:
                    values=pd.to_numeric(d[source],errors='coerce');frame[target]=values.combine_first(frame[target])
            mask=d.get('isST',pd.Series(index=d.index,dtype=float)).notna()
            frame.loc[mask,'state_source']='baostock_verified_raw_cache'
            frame.loc[frame.turnover.notna(),'turnover_source']='baostock_verified_raw_cache'
            if 'preclose' in d:frame['reference_price']=pd.to_numeric(d.preclose,errors='coerce').combine_first(frame.reference_price)
            if 'pbMRQ' in d:
                existing=frame.get('pb_mrq',pd.Series(np.nan,index=frame.index))
                frame['pb_mrq']=pd.to_numeric(d.pbMRQ,errors='coerce').combine_first(existing)
    # Count observed quotes, rather than inventing IPO dates from interval starts.
    frame['listing_session_lower_bound']=np.arange(1,len(frame)+1)
    ipo=basic.get('ipoDate') if basic is not None else None
    if pd.notna(ipo):frame['listing_session_lower_bound']=calendar[calendar>=pd.Timestamp(ipo)].searchsorted(frame.index,side='right')
    ready=frame.is_st_at_open.notna()&frame.listing_session_lower_bound.ge(6)
    ratio=pd.Series(np.nan,index=frame.index)
    if code.startswith('sh.688'):ratio.loc[ready]=.20
    elif code.startswith(('sz.300','sz.301')):
        modern=pd.Series(frame.index>=pd.Timestamp('2020-08-24'),index=frame.index)
        ratio.loc[ready&modern]=.20
        ratio.loc[ready&~modern]=np.where(frame.loc[ready&~modern,'is_st_at_open'].astype(bool),.05,.10)
    elif code.startswith(('sh.600','sh.601','sh.603','sh.605','sz.000','sz.001','sz.002','sz.003')):
        ratio.loc[ready]=np.where(frame.loc[ready,'is_st_at_open'].astype(bool),.05,.10)
    frame['limit_ratio_at_open']=ratio
    frame['limit_up']=np.nan;frame['limit_down']=np.nan
    frame['open_status']='unknown';frame['open_status_reason']='historical_state_or_rule_missing'
    halted=frame.tradestatus.eq(0)
    frame.loc[halted,'open_status']='halted';frame.loc[halted,'open_status_reason']='observed_daily_suspension'
    valid=ratio.notna()&np.isfinite(frame.reference_price)&frame.reference_price.gt(0)
    limits=[limit_prices(float(ref),float(r)) for ref,r in zip(frame.loc[valid,'reference_price'],ratio.loc[valid])]
    if limits:frame.loc[valid,['limit_up','limit_down']]=limits
    traded=valid&frame.tradestatus.eq(1)
    in_bounds=frame.open.ge(frame.limit_down-.005)&frame.open.le(frame.limit_up+.005)
    frame.loc[traded&in_bounds,'open_status']='derived';frame.loc[traded&in_bounds,'open_status_reason']=''
    frame.loc[traded&~in_bounds,'open_status_reason']='open_outside_rule_bounds'
    frame['available_at']=[str(d.date())+' 18:00:00' for d in frame.index]
    frame['availability_basis']='daily_close_plus_buffer_research_assumption'
    frame['open_status_available_at']=[str(d.date())+' 09:30:00' for d in frame.index]
    frame['open_status_basis']='daily_open_assumption';frame['rule_version']=RULE_VERSION
    frame['quality_status']='ok'
    if 'source' not in frame:frame['source']='local_qlib_archive'
    if 'revision' not in frame:frame['revision']=qlib_hash or 'unverified'
    frame.index.name='event_time'
    return frame,basic


def _write_frame(root,name,frame):
    path=root/(name+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
    # Repeated provenance strings are represented as categories in memory too.
    frame=frame.copy()
    for col in frame.columns:
        if not (pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col])):continue
        if len(frame)>1000 and frame[col].nunique(dropna=False)<len(frame)//10:frame[col]=frame[col].astype('category')
    frame.to_parquet(path,index=True)
    return dict(path=str(path.relative_to(root)),sha256=sha256_file(path))


def export_local_dataset(database,output,*,qlib_root=None,raw_cache_roots=(),start='2014-01-02',end='2026-06-30',industry_mode='effective_date_proxy',progress=None):
    """Freeze all historical A-share observations. No sampling or current-survivor filter."""
    import duckdb
    if industry_mode not in INDUSTRY_MODES:raise ValueError('unknown_industry_mode')
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start>end:raise ValueError('invalid_interval')
    database=Path(database).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    cache=_cache_index(raw_cache_roots);source_hash=sha256_file(database)
    action_gaps,gap_sources=_action_gap_inputs(raw_cache_roots,start,end)
    qroot=Path(qlib_root).resolve() if qlib_root else None
    qhash=None;qlib_manifest={}
    if qroot:
        metadata=qroot.parent.parent/'qlib_bin.manifest.json'
        archive=qroot.parent.parent/'qlib_bin.tar.gz'
        qlib_manifest=json.loads(metadata.read_text());qhash=sha256_file(archive)
        if qlib_manifest['archive_sha256'].removeprefix('sha256:')!=qhash:raise ValueError('qlib_archive_checksum_mismatch')
        qlib_manifest['verified_decoded_files']=verify_qlib_tree(qroot,archive)
    work=Path(tempfile.mkdtemp(prefix='.local-export-',dir=output));records=[];actions=[];issues=[];entries={};coverage=[];accounting_gaps=[]
    con=duckdb.connect(str(database),read_only=True)
    try:
        con.execute('BEGIN TRANSACTION')
        master=con.execute("SELECT symbol FROM security_master WHERE asset_class='stock'").fetchdf()
        symbols={s for s in master.symbol if _is_stock(canonical_code(s))}
        # Use the complete observed catalog. Source interval endings are not deletions.
        calendar=con.execute("SELECT DISTINCT calendar_day FROM trading_calendar WHERE exchange='CN' AND is_open AND calendar_day>=? AND calendar_day<=? ORDER BY calendar_day",[start,end+pd.Timedelta(days=10)]).fetchdf()
        cal=pd.DatetimeIndex(pd.to_datetime(calendar.calendar_day))
        if qroot:
            qcal=pd.DatetimeIndex(pd.to_datetime((qroot/'calendars/day.txt').read_text().splitlines()))
            cal=cal.union(qcal[(qcal>=start)&(qcal<=end+pd.Timedelta(days=10))]).sort_values()
        if 'calendar.csv' in cache:
            c,_=verified_csv(cache['calendar.csv']['path'])
            cached=pd.DatetimeIndex(pd.to_datetime(c.loc[c.is_trading_day==1,'calendar_date']))
            cal=cal.union(cached[(cached>=start)&(cached<=end+pd.Timedelta(days=10))]).sort_values()
        state_frame=con.execute("SELECT * FROM security_states WHERE effective_on BETWEEN ? AND ? ORDER BY symbol,effective_on",[start,end]).fetchdf()
        states={canonical_code(c):g for c,g in state_frame.groupby('symbol',sort=False)}
        additional_actions={}
        if 'corporate_actions.csv' in cache:
            table,_=verified_csv(cache['corporate_actions.csv']['path'])
            if table is not None and len(table):
                additional_actions={c:g.to_dict('records') for c,g in table.groupby('instrument_id')}
        industries=con.execute('SELECT * FROM industry_history ORDER BY symbol,effective_on').fetchdf()
        industries=industries.rename(columns={'symbol':'instrument_id','industry':'sector','effective_on':'effective_at','source_name':'source','raw_hash':'revision'})
        industries['instrument_id']=industries.instrument_id.map(canonical_code)
        industries=industries[industries.instrument_id.map(_is_stock)].copy()
        industries['available_at']=None;industries['ingested_at']=None;industries['temporal_basis']='effective_only'
        industries['quality_status']='historical_effective_date_proxy'
        industries['effective_at']=industries.effective_at.dt.strftime('%Y-%m-%d')
        financials=con.execute("SELECT * FROM financial_records WHERE metric NOT IN ('pe_ttm','pb_mrq') AND announced_on<=? ORDER BY symbol,metric,report_period,announced_on",[end]).fetchdf()
        financials=financials.rename(columns={'symbol':'instrument_id','source_name':'source','raw_hash':'revision'})
        financials['instrument_id']=financials.instrument_id.map(canonical_code)
        financials=financials[financials.instrument_id.map(_is_stock)].copy()
        financials['available_at']=None;financials['ingested_at']=None
        financials['quality_status']='reported_date_and_current_value_snapshot_unverified'
        # One ordered scan avoids thousands of full-table queries. pb_mrq is a
        # daily valuation observation; it is never called an announcement.
        sql="""SELECT b.*,f.factor,v.value AS pb_mrq FROM daily_bars b
          LEFT JOIN adjustment_factors f USING(symbol,trading_day)
          LEFT JOIN financial_records v ON v.symbol=b.symbol AND v.metric='pb_mrq'
            AND v.report_period=b.trading_day AND v.announced_on=b.trading_day
          WHERE NOT b.adjusted AND b.trading_day BETWEEN ? AND ?
          ORDER BY b.symbol,b.trading_day"""
        batches=con.execute(sql,[start,end]).to_arrow_reader(100000)
        def process(symbol,parts):
            if symbol not in symbols:return
            code=canonical_code(symbol)
            if parts:
                p=pd.concat(parts,ignore_index=True)
                p.index=pd.DatetimeIndex(pd.to_datetime(p.pop('trading_day')))
                p=p.rename(columns={'previous_close':'reference_price','source_name':'source','raw_hash':'revision'})
                p['feature_close']=pd.to_numeric(p.close,errors='coerce')*pd.to_numeric(p.factor,errors='coerce')
                for field in ('open','high','low','close','reference_price','amount','volume','factor','pb_mrq'):
                    p[field]=pd.to_numeric(p[field],errors='coerce')
            else:
                p=qlib_prices(qroot,symbol,start,end) if qroot else pd.DataFrame()
                if len(p):p['source']='chenditc.investment_data.qlib';p['revision']=qhash
            if qroot and len(p) and p.index.min()>start:
                earlier=qlib_prices(qroot,symbol,start,p.index.min()-pd.Timedelta(days=1))
                if len(earlier):
                    earlier['source']='chenditc.investment_data.qlib';earlier['revision']=qhash
                    p=pd.concat([earlier,p],sort=False)
            if p.empty:return
            p,basic=normalize_panel(code,p,states.get(code),cache,cal,qlib_hash=qhash)
            first=str(p.index[0].date());ipo=basic.get('ipoDate') if basic is not None else None
            row=dict(instrument_id=code,available_at=first+' 18:00:00',effective_at=first,
                source='observed_price_history',revision=source_hash,quality_status='ok',active=True,is_st=None,
                state_basis='daily_observation_required',listing_date=ipo if pd.notna(ipo) else None,observation_start=first,
                lifecycle_basis='baostock_basic' if pd.notna(ipo) else 'observed_history_not_ipo',sector='UNKNOWN',
                asset_type='stock',buy_lot=1 if code.startswith(('sh.688','sh.689','bj.')) else 100,
                sell_lot=1 if code.startswith(('sh.688','sh.689','bj.')) else 100,
                min_buy_quantity=200 if code.startswith(('sh.688','sh.689')) else 100,
                min_sell_quantity=200 if code.startswith(('sh.688','sh.689')) else 100,sell_delay=1,
                price_tick=.01,commission=.00025,min_commission=5.,limit_ratio=None)
            records.append(row)
            out=basic.get('outDate') if basic is not None else None
            if pd.notna(out) and pd.Timestamp(out)<=end:
                inactive=dict(row);inactive.update(active=False,effective_at=str(out),available_at=str(out)+' 00:00:00',source='baostock_delisting_lifecycle')
                records.append(inactive);issues.append(dict(instrument_id=code,date=str(out),reason='delisting_settlement_unverified'))
            acts,gaps=_distributions(code,cache,cal,end)
            combined={a['action_id']:a for a in acts}
            conflicts=set()
            for action in additional_actions.get(code,[]):
                if pd.Timestamp(action['ex_date'])>end:continue
                previous=combined.get(action['action_id'])
                if previous and (abs(previous['cash_per_share']-action['cash_per_share'])>1e-6 or abs(previous['share_ratio']-action['share_ratio'])>1e-6):
                    accounting_gaps.append(dict(instrument_id=code,date=action['ex_date'],reason='corporate_action_source_conflict'))
                    conflicts.add(action['action_id'])
                else:combined[action['action_id']]=action
            acts=[a for key,a in combined.items() if key not in conflicts]
            complete={str(pd.Timestamp(a['ex_date']).date()):a for a in acts}
            supplied_gaps=action_gaps.get(code,[])
            ambiguous={g['date'] for g in supplied_gaps if g['reason'] in (
                'multiple_distribution_records_same_day_not_reconciled','distribution_date_conflict')}
            actions.extend(acts)
            issues.extend(g for g in gaps if not _distribution_gap_resolved(g,complete.get(str(pd.Timestamp(g['date']).date())),ambiguous))
            accounting_gaps.extend(g for g in supplied_gaps if not _distribution_gap_resolved(g,complete.get(g['date']),ambiguous))
            # This is a cash-accounting diagnostic, not a synthetic dividend and
            # not a reason to erase an otherwise observed adjusted price feature.
            cash=pd.Series(0.,index=p.index);shares=cash.copy()
            for action in acts:
                ex=pd.Timestamp(action['ex_date'])
                if ex in p.index:
                    cash.loc[ex]+=float(action['cash_per_share']);shares.loc[ex]+=float(action['share_ratio'])
            expected_reference=(p.close.shift(1)-cash)/(1+shares)
            reference_gap=expected_reference-p.reference_price
            p['unexplained_ex_reference']=reference_gap.where(reference_gap.abs()>.021)
            for date,value in p.unexplained_ex_reference.dropna().items():
                accounting_gaps.append(dict(instrument_id=code,date=str(date.date()),reason='unexplained_ex_reference',
                    previous_raw_close=float(p.close.shift(1).loc[date]),observed_reference=float(p.reference_price.loc[date]),
                    reference_gap_per_share=float(value),meaning='price_reference_difference_not_a_cash_entitlement'))
            coverage.append(dict(instrument_id=code,rows=len(p),first=first,last=str(p.index[-1].date()),
                turnover_rows=int(p.turnover.notna().sum()),state_rows=int(p.is_st_at_open.notna().sum()),
                derived_open_rows=int(p.open_status.eq('derived').sum()),feature_rows=int(p.feature_close.notna().sum()),
                lifecycle_verified=basic is not None,distribution_records=len(acts),unexplained_ex_reference_rows=int(p.unexplained_ex_reference.notna().sum())))
            entries[code]=_write_frame(work,'panels/'+code,p)
            if progress and len(entries)%100==0:progress(dict(securities=len(entries),rows=sum(x['rows'] for x in coverage)))
        current=None;parts=[]
        for batch in batches:
            b=batch.to_pandas()
            for symbol,part in b.groupby('symbol',sort=False):
                if current is not None and symbol!=current:process(current,parts);parts=[]
                current=symbol;parts.append(part)
        if current is not None:process(current,parts)
        # Securities which ceased quoting before the DuckDB import boundary
        # must still appear in an earlier-period research universe.
        if qroot:
            for symbol in sorted(symbols):
                if canonical_code(symbol) not in entries:process(symbol,[])
        if 'benchmark.csv' not in cache:raise ValueError('verified_benchmark_cache_required')
        bench,bmeta=verified_csv(cache['benchmark.csv']['path']);bench.index=pd.DatetimeIndex(pd.to_datetime(bench.pop('date')))
        bench=bench.loc[start:end].copy();bench['source']='baostock_index_sh000300';bench['revision']=bmeta['sha256'];bench['quality_status']='ok'
        bench['available_at']=[str(d.date())+' 18:00:00' for d in bench.index];bench.index.name='event_time'
        if bench.empty:raise ValueError('benchmark_interval_empty')
        action_frame=pd.DataFrame(actions)
        if not action_frame.empty:
            action_frame=action_frame.sort_values('available_at').drop_duplicates('action_id',keep='last')
        for item in gap_sources:
            if sha256_file(item['path'])!=item['sha256']:raise ValueError('action_gap_changed_during_export:'+item['path'])
        manifest=dict(source='local_market_snapshot',revision='local_adapter_v3',adapter_sha256=sha256_file(__file__),as_of=str(end.date()),
            price_basis='raw',currency='CNY',amount_unit='CNY',turnover_unit='percent',universe_coverage='historical',
            quantity_rules=dict(declaration='minimum_separate_from_increment',
                main_buy='100_share_multiples',main_sell='conservative_100_share_grid_or_whole_residual',
                star='minimum_200_increment_1_or_whole_sell_residual',
                bse='minimum_100_increment_1_or_whole_sell_residual',
                limitation='main_board_sell_grid_is_a_conservative_research_execution_assumption'),
            industry_mode=industry_mode,feature_price_basis='historical_adjustment_snapshot',actions_coverage='partial_observed_events',
            benchmark_type='index',benchmark_id='sh.000300',benchmark_return_basis='price_return',
            panels=entries,instruments=_write_frame(work,'instruments',pd.DataFrame(records)),
            actions=_write_frame(work,'actions',action_frame),calendar=_write_frame(work,'calendar',pd.DataFrame({'session':cal})),
            benchmark=_write_frame(work,'benchmark',bench),industry_records=_write_frame(work,'industry_records',industries),
            financials=_write_frame(work,'financials',financials),coverage=_write_frame(work,'coverage',pd.DataFrame(coverage)),
            data_gaps=issues,accounting_gaps=accounting_gaps,execution_assumption='daily_open_with_verified_daily_states_and_derived_rule_limits',
            research_scope=dict(scope='all_observed_historical_a_shares_no_sampling',start=str(start.date()),end=str(end.date()),
                                current_survival_filter=False,listing_dates='verified_where_available_else_observation_lower_bound'),
            source_files=dict(database=dict(path=str(database),sha256=source_hash),qlib=dict(root=str(qroot) if qroot else None,manifest=qlib_manifest),raw_cache=cache,accounting_gap_sources=gap_sources),
            limitations=['Industry effective dates do not certify historical publication.',
                         'Missing ST, turnover and corporate events remain unknown.',
                         'Financial values and adjustment factors are current historical snapshots.',
                         'Missing distribution or delisting events prevent qualified account performance.'])
        identity=fingerprint(manifest)
        manifest['snapshot_id']=identity
        (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
        summary=dict(snapshot_id=identity,securities=len(entries),rows=sum(x['rows'] for x in coverage),
            turnover_rows=sum(x['turnover_rows'] for x in coverage),state_rows=sum(x['state_rows'] for x in coverage),
            derived_open_rows=sum(x['derived_open_rows'] for x in coverage),actions=len(action_frame),industry_records=len(industries),
            financial_records=len(financials),start=str(start.date()),end=str(end.date()),status='conditional_research_snapshot',
            unexplained_ex_reference_rows=sum(x['unexplained_ex_reference_rows'] for x in coverage),
            promotion_approved=False,source_database_unchanged=source_hash==sha256_file(database))
        if not summary['source_database_unchanged']:raise ValueError('source_database_changed_during_export')
        (work/'coverage_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        target=output/identity[:24]
        if target.exists():
            if (target/'manifest.json').read_bytes()!=(work/'manifest.json').read_bytes():raise ValueError('snapshot_collision')
            shutil.rmtree(work)
        else:work.rename(target)
        return summary|dict(manifest=str(target/'manifest.json'))
    except BaseException:
        if work.exists():shutil.rmtree(work)
        raise
    finally:con.close()
