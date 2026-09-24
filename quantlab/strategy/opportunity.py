"""One causal price/volume opportunity module, calibrated to open-to-open returns."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from quantlab.data.contracts import asof_records, decision_time, timestamp, fingerprint
from quantlab.data.market import panels_for_research

FEATURES=('residual20','residual60','residual120','breadth60','strength20',
          'breadth_change','strength_change','up_down_balance','drawdown20','participation_stability')
GROUPS=(FEATURES[:3],FEATURES[3:7],FEATURES[7:])


def feature_schema(profile):
    if profile=='price_volume_industry':return FEATURES,GROUPS
    if profile=='price_volume':return FEATURES[:3]+FEATURES[7:],(FEATURES[:3],FEATURES[7:])
    raise ValueError('unknown_feature_profile')


@dataclass
class Forecast:
    as_of: str
    horizon: int
    expected_returns: pd.Series
    status: str
    reasons: list
    training_cutoff: str
    model_version: str
    trace: dict = field(default_factory=dict)

    def validate(self,date):
        if pd.Timestamp(self.as_of).normalize()!=pd.Timestamp(date).normalize(): raise ValueError('forecast_date_mismatch')
        if timestamp(self.training_cutoff)>=decision_time(date): raise ValueError('forecast_training_not_prior')
        if self.horizon<1 or not self.model_version: raise ValueError('invalid_forecast_identity')
        if not np.isfinite(self.expected_returns).all(): raise ValueError('invalid_forecast_values')


def industry_history(ds,dates,codes):
    sector=pd.DataFrame(index=dates,columns=codes,dtype=object)
    for d in dates:
        r=asof_records(ds.instruments,decision_time(d))
        if r.empty: continue
        known=r[(r.quality_status=='ok') & r.active & (pd.to_datetime(r.listing_date)<=d)]
        sector.loc[d]=known.sector.reindex(codes).replace({'NA':np.nan,'UNKNOWN':np.nan,'':np.nan})
    return sector


def opportunity_features(ds, min_industry_members=5, profile='price_volume_industry'):
    fields,_=feature_schema(profile)
    close,amount,_=panels_for_research(ds)
    ret=close.pct_change(fill_method=None)
    market=ds.benchmark.close.reindex(close.index).pct_change(fill_method=None)
    sector=industry_history(ds,close.index,close.columns) if profile=='price_volume_industry' else pd.DataFrame(index=close.index)
    breadth=pd.DataFrame(np.nan,index=close.index,columns=close.columns)
    strength=breadth.copy(); industry=breadth.copy()
    ma=close.rolling(60,min_periods=60).mean()
    rel20=close/close.shift(20)-1-(ds.benchmark.close/ds.benchmark.close.shift(20)-1).reindex(close.index).to_numpy()[:,None]
    for d in close.index if profile=='price_volume_industry' else []:
        groups=sector.loc[d].dropna().groupby(sector.loc[d].dropna()).groups
        for _, members in groups.items():
            members=list(members)
            ready=[c for c in members if pd.notna(ma.loc[d,c]) and pd.notna(ret.loc[d,c])]
            if len(ready)<min_industry_members: continue
            rr=ret.loc[d,ready]
            # Leave-one-out industry return avoids mechanically explaining own return.
            industry.loc[d,ready]=(rr.sum()-rr)/(len(ready)-1)
            breadth.loc[d,ready]=(close.loc[d,ready]>ma.loc[d,ready]).mean()
            strength.loc[d,ready]=(rel20.loc[d,ready]>0).mean()
    residual=close*float('nan')
    for c in close:
        y=ret[c]; x=market; z=industry[c]-market
        def mean(v): return v.rolling(120,min_periods=120).mean().shift(1)
        my,mx,mz=mean(y),mean(x),mean(z)
        vx=mean(x*x)-mx*mx+1e-10; vz=mean(z*z)-mz*mz+1e-10
        cxz=mean(x*z)-mx*mz; cxy=mean(x*y)-mx*my; czy=mean(z*y)-mz*my
        if profile=='price_volume':
            bx=cxy/vx
            residual[c]=y-(my-bx*mx)-bx*x
            continue
        determinant=vx*vz-cxz*cxz
        bx=(cxy*vz-czy*cxz)/determinant
        bz=(czy*vx-cxy*cxz)/determinant
        residual[c]=y-(my-bx*mx-bz*mz)-bx*x-bz*z
    raw={f'residual{n}':residual.shift(5).rolling(n,min_periods=n).sum() for n in (20,60,120)}
    # Historical membership changes cannot rewrite previously stored breadth.
    raw.update(breadth60=breadth,strength20=strength,breadth_change=breadth-breadth.shift(20),
               strength_change=strength-strength.shift(20),
               up_down_balance=np.sqrt(ret.clip(lower=0).pow(2).rolling(60).mean())-np.sqrt(ret.clip(upper=0).pow(2).rolling(60).mean()),
               drawdown20=close/close.rolling(20).max()-1,
               participation_stability=-(amount/amount.rolling(20).mean()).rolling(20).std())
    # Full-window validity; do not fill missing industry or insufficient history with zeros.
    raw_frame=pd.concat({name:frame.stack() for name,frame in raw.items()},axis=1)
    raw_frame.index.names=['date','instrument_id']
    raw_frame=raw_frame.replace([np.inf,-np.inf],np.nan)
    raw_frame=raw_frame.reindex(columns=fields)
    # Unknown equity events cannot become training data or cross-event trends.
    # Only decisions at/after the event are invalidated; earlier signals are intact.
    for gap in ds.manifest.get('data_gaps',[]):
        c=gap.get('instrument_id');d=pd.Timestamp(gap['date'])
        j=close.index.searchsorted(d)
        stop=close.index[min(j+245,len(close.index)-1)] if len(close.index) else d
        mask=(raw_frame.index.get_level_values('instrument_id')==c)&(raw_frame.index.get_level_values('date')>=d)
        if gap.get('reason')!='delisting_settlement_unverified':mask &= raw_frame.index.get_level_values('date')<=stop
        raw_frame.loc[mask,:]=np.nan
    return raw_frame


def normalized_features(raw):
    def scale(day):
        lo,hi=day.quantile(.01),day.quantile(.99)
        clipped=day.clip(lo,hi,axis=1)
        std=clipped.std(ddof=0)
        return (clipped-clipped.mean())/std.where(std>1e-12,1.)
    return raw.groupby(level='date',group_keys=False).transform(lambda x:x) if raw.empty else pd.concat(
        {d:scale(group.droplevel('date')) for d,group in raw.groupby(level='date')},names=['date','instrument_id'])


def forward_labels(ds,horizon=20,allow_open_assumption=False):
    rows=[]; cal=ds.calendar
    decisions=pd.Series([decision_time(d) for d in cal],index=cal)
    for c,p in ds.panels.items():
        acts=ds.actions[ds.actions.instrument_id==c].copy() if not ds.actions.empty else pd.DataFrame()
        if not acts.empty:
            acts['_ex']=pd.to_datetime(acts.ex_date)
            acts['_available']=acts.available_at.map(timestamp)
            acts=acts.sort_values('_ex');exdates=pd.DatetimeIndex(acts._ex)
        p=p.reindex(cal)
        prices=p.open.to_numpy()
        statuses=('trading','derived') if allow_open_assumption else ('trading',)
        valid=(p.quality_status.eq('ok') & p.open_status.isin(statuses) & p.open.gt(0) & np.isfinite(p.open)).to_numpy()
        available=p.available_at.map(lambda x:timestamp(x) if pd.notna(x) else pd.NaT)
        gaps=[pd.Timestamp(g['date']) for g in ds.manifest.get('data_gaps',[]) if g.get('instrument_id')==c]
        for j in range(max(0,len(cal)-horizon-1)):
            d=cal[j];a=j+1;b=j+1+horizon
            entry,exit_=cal[j+1],cal[j+1+horizon]
            if not valid[a] or not valid[b] or pd.isna(available.iloc[b]):continue
            if any(entry<=g<=exit_ for g in gaps):continue
            shares,cash=1.,0.;bad=False
            if not acts.empty:
                lo,hi=exdates.searchsorted(entry,side='right'),exdates.searchsorted(exit_,side='right')
                if hi>lo:
                    event=acts.iloc[lo:hi]
                    event=event[event._available<=decisions.iloc[b]].sort_values('_available').drop_duplicates('action_id',keep='last').sort_values('_ex')
                    for _,action in event.iterrows():
                        if pd.Timestamp(action.record_date)<entry: continue
                        if action.get('action_type','distribution')!='distribution': bad=True;break
                        cash+=shares*float(action.cash_per_share)
                        shares*=1+float(action.share_ratio)
            if bad: continue
            rows.append(dict(date=d,instrument_id=c,target=(prices[b]*shares+cash)/prices[a]-1,label_end=exit_,
                             available_at=max(available.iloc[b],decisions.iloc[b])))
    return pd.DataFrame(rows).set_index(['date','instrument_id']) if rows else pd.DataFrame(columns=['target','label_end','available_at'],index=pd.MultiIndex.from_tuples([],names=['date','instrument_id']))


class OpportunityModel:
    def __init__(self,kind='ridge',alpha=10.,horizon=20,profile='price_volume_industry'):
        if kind not in ('ridge','equal'): raise ValueError('unknown_model')
        self.kind,self.alpha,self.horizon=kind,float(alpha),horizon
        self.profile=profile;self.features,self.groups=feature_schema(profile)
        self.scaler=StandardScaler();self.regressor=Ridge(alpha=alpha)
        self.training_cutoff=None

    def _design(self,x):
        if self.kind=='equal': return np.mean([x[:,[self.features.index(k) for k in group]].mean(axis=1) for group in self.groups],axis=0)[:,None]
        return x

    def fit(self,features,labels,dates,as_of,min_rows=200,min_dates=40):
        joint=features.join(labels)
        joint=joint[joint.index.get_level_values('date').isin(dates)]
        joint=joint[joint.available_at.map(timestamp)<=timestamp(as_of)].dropna(subset=list(self.features)+['target'])
        if len(joint)<min_rows or joint.index.get_level_values('date').nunique()<min_dates: raise ValueError('insufficient_training_sample')
        x=self.scaler.fit_transform(joint[list(self.features)])
        self.regressor.fit(self._design(x),joint.target)
        self.training_cutoff=str(joint.available_at.map(timestamp).max())
        self.model_version=fingerprint(dict(kind=self.kind,alpha=self.alpha,horizon=self.horizon,profile=self.profile,
            cutoff=self.training_cutoff,coef=self.regressor.coef_.tolist(),intercept=float(self.regressor.intercept_),
            mean=self.scaler.mean_.tolist(),scale=self.scaler.scale_.tolist()))
        return self

    def predict(self,features,date,eligible=None):
        if self.training_cutoff is None: raise ValueError('model_not_fitted')
        if timestamp(self.training_cutoff)>=decision_time(date): raise ValueError('model_not_available')
        try: x=features.xs(pd.Timestamp(date),level='date').reindex(columns=self.features).dropna()
        except KeyError: x=pd.DataFrame(columns=self.features)
        if eligible is not None: x=x.reindex(eligible).dropna()
        pred=pd.Series(dtype=float) if x.empty else pd.Series(self.regressor.predict(self._design(self.scaler.transform(x))),index=x.index)
        status='ok' if len(pred)>1 and pred.nunique()>1 else 'invalid'
        return Forecast(str(date),self.horizon,pred,status,[] if status=='ok' else ['insufficient_or_constant_forecast'],
                        self.training_cutoff,self.model_version,dict(feature_profile=self.profile,normalized_features=x.to_dict(),predictions=pred.to_dict()))


def save_model(model,path):
    import json
    from pathlib import Path
    payload=dict(kind=model.kind,alpha=model.alpha,horizon=model.horizon,training_cutoff=model.training_cutoff,
        model_version=model.model_version,profile=model.profile,features=list(model.features),coef=model.regressor.coef_.tolist(),
        intercept=float(model.regressor.intercept_),mean=model.scaler.mean_.tolist(),scale=model.scaler.scale_.tolist())
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(payload,sort_keys=True,indent=2)
    if path.exists() and path.read_text()!=raw:raise ValueError('model_artifact_immutable')
    path.write_text(raw)
    return path


def load_model(path):
    import json
    from pathlib import Path
    p=json.loads(Path(path).read_text())
    fields,_=feature_schema(p['profile'])
    if p['features']!=list(fields):raise ValueError('feature_schema_mismatch')
    identity=fingerprint({k:p[k] for k in ('kind','alpha','horizon','profile','coef','intercept','mean','scale')} | {'cutoff':p['training_cutoff']})
    if identity!=p['model_version']:raise ValueError('model_checksum_mismatch')
    model=OpportunityModel(p['kind'],p['alpha'],p['horizon'],p['profile'])
    model.training_cutoff=p['training_cutoff'];model.model_version=p['model_version']
    model.scaler.mean_=np.array(p['mean']);model.scaler.scale_=np.array(p['scale']);model.scaler.n_features_in_=len(fields)
    model.scaler.feature_names_in_=np.array(fields,dtype=object)
    model.regressor.coef_=np.array(p['coef']);model.regressor.intercept_=p['intercept']
    model.regressor.n_features_in_=1 if p['kind']=='equal' else len(fields)
    return model


def load_forecasts(path,data_version):
    import json
    from pathlib import Path
    payload=json.loads(Path(path).read_text())
    if payload['data_version']!=data_version:raise ValueError('forecast_dataset_version_mismatch')
    result={}
    for row in payload['forecasts']:
        row['expected_returns']=pd.Series(row['expected_returns'],dtype=float)
        forecast=Forecast(**row);d=pd.Timestamp(forecast.as_of).normalize();forecast.validate(d)
        if d in result:raise ValueError('duplicate_forecast_date')
        result[d]=forecast
    return result
