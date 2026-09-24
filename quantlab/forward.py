"""Persistent paper account and reconciliation. No brokerage orders are submitted."""
from pathlib import Path
import sqlite3
import json
import pickle
import hashlib
import math
from copy import deepcopy
from dataclasses import asdict
import pandas as pd
from quantlab.config import CONFIG
from quantlab.report.runs import provenance
from quantlab.mandate import Mandate
from quantlab.data.contracts import fingerprint,frame_hash,timestamp,decision_time
from quantlab.backtest.engine import Engine
from quantlab.data.qualification import qualification


class ForwardBook:
    def __init__(self,path,model_version,cfg=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.mandate=Mandate.from_config(cfg or CONFIG);self.model_version=model_version
        if not model_version:raise ValueError('frozen_model_version_required')
        with self.connect() as db:
            db.executescript('CREATE TABLE IF NOT EXISTS policy (key TEXT PRIMARY KEY,value TEXT);'
                'CREATE TABLE IF NOT EXISTS sessions (date TEXT PRIMARY KEY, payload TEXT, state BLOB, hash TEXT, previous_hash TEXT);'
                'CREATE TABLE IF NOT EXISTS actual_fills (fill_id TEXT PRIMARY KEY, payload TEXT);')
            identity=json.dumps(dict(model_version=model_version,mandate_version=self.mandate.version,code_version=fingerprint(provenance()["code_sha256"])),sort_keys=True)
            old=db.execute("SELECT value FROM policy WHERE key='identity'").fetchone()
            if old and old[0]!=identity:raise ValueError('forward_policy_changed_start_new_book')
            db.execute("INSERT OR IGNORE INTO policy VALUES ('identity',?)",(identity,))

    def connect(self):return sqlite3.connect(self.path,timeout=30)

    def _verify(self,db):
        previous=None
        for date,payload,state,digest,parent in db.execute('SELECT * FROM sessions ORDER BY date'):
            computed=hashlib.sha256(payload.encode()+state+(previous or '').encode()).hexdigest()
            if parent!=previous or digest!=computed:raise ValueError('forward_journal_integrity_failed')
            previous=digest
        return previous

    def verify(self):
        with self.connect() as db:return self._verify(db)

    @staticmethod
    def _history(ds,d):
        cutoff=decision_time(d)
        result={'panels':{c:frame_hash(p.loc[:d]) for c,p in ds.panels.items() if len(p.loc[:d])}}
        for name in ('instruments','actions','exposures','announcements','benchmark'):
            frame=getattr(ds,name)
            if len(frame) and 'available_at' in frame:frame=frame[frame.available_at.map(timestamp)<=cutoff]
            result[name]=frame_hash(frame)
        result['calendar']=fingerprint(list(ds.calendar[ds.calendar<=d]))
        return result

    @staticmethod
    def _request_hash(ds,forecast):
        row=asdict(forecast) if forecast is not None else None
        if row is not None:row['expected_returns']=forecast.expected_returns.to_dict()
        return fingerprint(dict(data=ds.version,forecast=row))

    def step(self,ds,forecast=None,now=None,allow_test_fixture=False,*,session=None):
        now=timestamp(now or pd.Timestamp.now(tz='Asia/Shanghai'))
        d=pd.Timestamp(session).normalize() if session is not None else now.tz_localize(None).normalize()
        if now<decision_time(d):raise ValueError('forward_close_not_yet_available')
        if d not in ds.calendar:raise ValueError('not_exchange_session')
        if pd.Timestamp(ds.manifest['as_of']).normalize()!=d:raise ValueError('forward_data_not_current')
        report=qualification(ds)
        if report['status']!='qualified_for_daily_research' and not (allow_test_fixture and ds.manifest.get('source')=='TEST_FIXTURE'):
            raise ValueError('forward_data_not_qualified:'+','.join(report['reasons']))
        if forecast is not None:
            forecast.validate(d)
            if forecast.model_version!=self.model_version:raise ValueError('forward_model_changed')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._verify(db)
            old=db.execute('SELECT payload FROM sessions WHERE date=?',(str(d.date()),)).fetchone()
            if old:
                payload=json.loads(old[0])
                request=self._request_hash(ds,forecast)
                if payload['request_hash']!=request:raise ValueError('same_session_different_inputs')
                return payload
            last=db.execute('SELECT payload,state,hash FROM sessions ORDER BY date DESC LIMIT 1').fetchone()
            initial=None;previous=None
            if last:
                before=json.loads(last[0]);last_day=pd.Timestamp(before['as_of']).normalize()
                next_dates=ds.calendar[ds.calendar>last_day]
                if not len(next_dates) or next_dates[0]!=d:raise ValueError('missing_forward_session_requires_recorded_recovery')
                hashes=self._history(ds,last_day)
                if hashes!=before['history_hashes']:raise ValueError('historical_data_revision_requires_review')
                initial=pickle.loads(last[1]);previous=last[2]
            cfg=self.mandate.config()
            run=Engine(cfg).run(ds,strategy='opportunity',forecasts={d:forecast} if forecast else {},start=d,end=d,initial_state=initial)
            if run['status']=='blocked_data':raise ValueError('forward_run_blocked')
            state=pickle.dumps(run['_runtime']);orders=[o for o in run['orders'] if o['status'] in ('pending','partial')]
            payload=dict(as_of=str(d.date()),recorded_at=str(now),model_version=self.model_version,data_version=ds.version,
                mandate_version=self.mandate.version,status='fixture_test_only' if allow_test_fixture else run['status'],
                run_health=run['run_health'],
                account=run['account'][-1],pending_orders=orders,run_id=run['run_id'],fills=run['fills'],
                request_hash=self._request_hash(ds,forecast),
                history_hashes=self._history(ds,d),recovered_session=session is not None,
                promotion_approved=False,executable=False,development_log=True)
            body=json.dumps(payload,default=str,sort_keys=True)
            digest=hashlib.sha256(body.encode()+state+(previous or '').encode()).hexdigest()
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)',(str(d.date()),body,state,digest,previous))
        return json.loads(body)

    def recover(self,ds,forecasts,now=None,allow_test_fixture=False):
        """Append missing historical sessions, explicitly marked as reconstructed logs."""
        now=timestamp(now or pd.Timestamp.now(tz='Asia/Shanghai'))
        state=self.status()
        if state['end'] is None:raise ValueError('recovery_requires_existing_account')
        end=pd.Timestamp(ds.manifest['as_of']).normalize();last=pd.Timestamp(state['end'])
        results=[]
        for d in ds.calendar[(ds.calendar>last)&(ds.calendar<=end)]:
            partial=deepcopy(ds);partial.manifest['as_of']=str(d.date())
            results.append(self.step(partial,forecasts.get(d),now=now,session=d,allow_test_fixture=allow_test_fixture))
        return results

    def reconcile(self,fills):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._verify(db)
            last=db.execute('SELECT payload,state FROM sessions ORDER BY date DESC LIMIT 1').fetchone()
            if not last:raise ValueError('no_simulated_session')
            if not fills:return dict(status='no_actual_fills',comparisons=[],promotion_approved=False)
            runtime=pickle.loads(last[1]);orders={o.order_id:o for o in runtime['ledger'].orders}
            for fill in fills:
                if not all(k in fill for k in ('fill_id','order_id','date','instrument_id','side','quantity','price','fees')):raise ValueError('actual_fill_fields_missing')
                order=orders.get(fill['order_id'])
                if order is None or order.instrument_id!=fill['instrument_id'] or order.side!=fill['side']:raise ValueError('actual_fill_unknown_order')
                if pd.Timestamp(fill['date'])<=pd.Timestamp(order.signal_date):raise ValueError('same_day_actual_fill')
                if pd.isna(pd.Timestamp(fill['date'])) or pd.Timestamp(fill['date']).normalize()>pd.Timestamp(json.loads(last[0])['as_of']):raise ValueError('actual_fill_date_unverified')
                if not all(math.isfinite(float(fill[k])) for k in ('quantity','price','fees')) or int(fill['quantity'])!=fill['quantity'] or fill['quantity']<=0 or fill['price']<=0 or fill['fees']<0:raise ValueError('invalid_actual_fill')
                body=json.dumps(fill,sort_keys=True)
                old=db.execute('SELECT payload FROM actual_fills WHERE fill_id=?',(fill['fill_id'],)).fetchone()
                if old and old[0]!=body:raise ValueError('actual_fill_revision_requires_new_correction')
                db.execute('INSERT OR IGNORE INTO actual_fills VALUES (?,?)',(fill['fill_id'],body))
            rows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM actual_fills')]
            aggregate={}
            for f in rows:aggregate[f['order_id']]=aggregate.get(f['order_id'],0)+f['quantity']
            if any(q>orders[o].quantity for o,q in aggregate.items()):raise ValueError('actual_fill_exceeds_order')
            sim=runtime['ledger'].fills
            comparisons=[]
            for o,qty in aggregate.items():
                a=[f for f in rows if f['order_id']==o];s=[f for f in sim if f['order_id']==o]
                ap=sum(f['quantity']*f['price'] for f in a)/qty
                sq=sum(f['quantity'] for f in s);sp=sum(f['quantity']*f['price'] for f in s)/sq if sq else None
                comparisons.append(dict(order_id=o,actual_quantity=qty,simulated_quantity=sq,actual_vwap=ap,simulated_vwap=sp,
                                        price_difference=None if sp is None else ap-sp,actual_fees=sum(f['fees'] for f in a)))
            return dict(status='reconciled_fills_only',comparisons=comparisons,
                note='Broker cash, corporate actions and positions still require account statement reconciliation.',promotion_approved=False)

    def status(self):
        with self.connect() as db:
            db.execute('BEGIN')
            digest=self._verify(db)
            dates=[x[0] for x in db.execute('SELECT date FROM sessions ORDER BY date')]
        return dict(sessions=len(dates),start=dates[0] if dates else None,end=dates[-1] if dates else None,journal_hash=digest,
                    model_version=self.model_version,mandate_version=self.mandate.version,promotion_approved=False)
