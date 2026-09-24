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


def opportunity_features(ds, min_industry_members=5):
    close,amount,_=panels_for_research(ds)
    ret=close.pct_change(fill_method=None)
    market=ds.benchmark.close.reindex(close.index).pct_change(fill_method=None)
    sector=industry_history(ds,close.index,close.columns)
    breadth=pd.DataFrame(np.nan,index=close.index,columns=close.columns)
    strength=breadth.copy(); industry=breadth.copy()
    ma=close.rolling(60,min_periods=60).mean()
    rel20=close/close.shift(20)-1-(ds.benchmark.close/ds.benchmark.close.shift(20)-1).reindex(close.index).to_numpy()[:,None]
    for d in close.index:
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
    return raw_frame.reindex(columns=FEATURES)


def normalized_features(raw):
    def scale(day):
        lo,hi=day.quantile(.01),day.quantile(.99)
        clipped=day.clip(lo,hi,axis=1)
        std=clipped.std(ddof=0)
        return (clipped-clipped.mean())/std.where(std>1e-12,1.)
    return raw.groupby(level='date',group_keys=False).transform(lambda x:x) if raw.empty else pd.concat(
        {d:scale(group.droplevel('date')) for d,group in raw.groupby(level='date')},names=['date','instrument_id'])


def forward_labels(ds,horizon=20):
    rows=[]; cal=ds.calendar
    for c,p in ds.panels.items():
        acts=ds.actions[ds.actions.instrument_id==c] if not ds.actions.empty else pd.DataFrame()
        for j,d in enumerate(cal):
            if j+1+horizon>=len(cal): continue
            entry,exit_=cal[j+1],cal[j+1+horizon]
            if entry not in p.index or exit_ not in p.index: continue
            a,b=p.loc[entry],p.loc[exit_]
            if a.quality_status!='ok' or b.quality_status!='ok' or not np.isfinite(a.open) or not np.isfinite(b.open): continue
            if a.open<=0 or b.open<=0 or a.open_status!='trading' or b.open_status!='trading': continue
            shares,cash=1.,0.;bad=False
            if not acts.empty:
                event=acts[(pd.to_datetime(acts.ex_date)>entry)&(pd.to_datetime(acts.ex_date)<=exit_)]
                event=event[event.available_at.map(timestamp)<=decision_time(exit_)].sort_values('available_at').drop_duplicates('action_id',keep='last').sort_values('ex_date')
                for _,action in event.iterrows():
                    if pd.Timestamp(action.record_date)<entry: continue
                    if action.get('action_type','distribution')!='distribution': bad=True;break
                    cash+=shares*float(action.cash_per_share)
                    shares*=1+float(action.share_ratio)
            if bad: continue
            rows.append(dict(date=d,instrument_id=c,target=(b.open*shares+cash)/a.open-1,label_end=exit_,
                             available_at=max(timestamp(b.available_at),decision_time(exit_))))
    return pd.DataFrame(rows).set_index(['date','instrument_id']) if rows else pd.DataFrame(columns=['target','label_end','available_at'],index=pd.MultiIndex.from_tuples([],names=['date','instrument_id']))


class OpportunityModel:
    def __init__(self,kind='ridge',alpha=10.,horizon=20):
        if kind not in ('ridge','equal'): raise ValueError('unknown_model')
        self.kind,self.alpha,self.horizon=kind,float(alpha),horizon
        self.scaler=StandardScaler();self.regressor=Ridge(alpha=alpha)
        self.training_cutoff=None

    def _design(self,x):
        if self.kind=='equal': return np.mean([x[:,[FEATURES.index(k) for k in group]].mean(axis=1) for group in GROUPS],axis=0)[:,None]
        return x

    def fit(self,features,labels,dates,as_of,min_rows=200,min_dates=40):
        joint=features.join(labels)
        joint=joint[joint.index.get_level_values('date').isin(dates)]
        joint=joint[joint.available_at.map(timestamp)<=timestamp(as_of)].dropna(subset=list(FEATURES)+['target'])
        if len(joint)<min_rows or joint.index.get_level_values('date').nunique()<min_dates: raise ValueError('insufficient_training_sample')
        x=self.scaler.fit_transform(joint[list(FEATURES)])
        self.regressor.fit(self._design(x),joint.target)
        self.training_cutoff=str(joint.available_at.map(timestamp).max())
        self.model_version=fingerprint(dict(kind=self.kind,alpha=self.alpha,horizon=self.horizon,
            cutoff=self.training_cutoff,coef=self.regressor.coef_.tolist(),intercept=float(self.regressor.intercept_),
            mean=self.scaler.mean_.tolist(),scale=self.scaler.scale_.tolist()))
        return self

    def predict(self,features,date,eligible=None):
        if self.training_cutoff is None: raise ValueError('model_not_fitted')
        if timestamp(self.training_cutoff)>=decision_time(date): raise ValueError('model_not_available')
        try: x=features.xs(pd.Timestamp(date),level='date').reindex(columns=FEATURES).dropna()
        except KeyError: x=pd.DataFrame(columns=FEATURES)
        if eligible is not None: x=x.reindex(eligible).dropna()
        pred=pd.Series(dtype=float) if x.empty else pd.Series(self.regressor.predict(self._design(self.scaler.transform(x))),index=x.index)
        status='ok' if len(pred)>1 and pred.nunique()>1 else 'invalid'
        return Forecast(str(date),self.horizon,pred,status,[] if status=='ok' else ['insufficient_or_constant_forecast'],
                        self.training_cutoff,self.model_version,dict(normalized_features=x.to_dict(),predictions=pred.to_dict()))


def save_model(model,path):
    import json
    from pathlib import Path
    payload=dict(kind=model.kind,alpha=model.alpha,horizon=model.horizon,training_cutoff=model.training_cutoff,
        model_version=model.model_version,features=list(FEATURES),coef=model.regressor.coef_.tolist(),
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
    if p['features']!=list(FEATURES):raise ValueError('feature_schema_mismatch')
    identity=fingerprint({k:p[k] for k in ('kind','alpha','horizon','coef','intercept','mean','scale')} | {'cutoff':p['training_cutoff']})
    if identity!=p['model_version']:raise ValueError('model_checksum_mismatch')
    model=OpportunityModel(p['kind'],p['alpha'],p['horizon'])
    model.training_cutoff=p['training_cutoff'];model.model_version=p['model_version']
    model.scaler.mean_=np.array(p['mean']);model.scaler.scale_=np.array(p['scale']);model.scaler.n_features_in_=len(FEATURES)
    model.scaler.feature_names_in_=np.array(FEATURES,dtype=object)
    model.regressor.coef_=np.array(p['coef']);model.regressor.intercept_=p['intercept']
    model.regressor.n_features_in_=1 if p['kind']=='equal' else len(FEATURES)
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
