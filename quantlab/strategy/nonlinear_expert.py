"""Chronological price/volume ensembles, with a DoubleEnsemble-style ablation.

Mechanism reference: Microsoft Qlib be725493 (MIT), DEnsembleModel and
arXiv:2010.01265. This implementation adds date-balanced training and within-date
permutations. It predicts ranks; it does not set account weights or permissions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

FEATURES = ['factor_'+name for name in (
    'momentum_12m1m','momentum_3m','momentum_1m_reversal','momentum_trend',
    'factor_volatility','value_breadth','value_medium','value_lowtension',
    'flow_turnover_low','flow_volume_trend','factor_price_crowding')]


@dataclass(frozen=True)
class EnsemblePolicy:
    cohorts: int = 36
    num_models: int = 6
    rounds: int = 100
    learning_rate: float = .05
    leaves: int = 15
    min_leaf: int = 100
    l2: float = 1.
    bagging: float = .8
    seed: int = 20260927
    bins_sr: int = 10
    bins_fs: int = 5
    alpha1: float = 1.
    alpha2: float = 1.
    decay: float = 1.
    feature_ratios: tuple = (.8,.7,.6,.5,.4)

    def __post_init__(self):
        if min(self.cohorts,self.num_models,self.rounds,self.leaves,self.min_leaf)<1:
            raise ValueError('invalid_ensemble_size')
        if not 0<self.bagging<=1 or not 0<self.learning_rate<=1 or self.l2<0:
            raise ValueError('invalid_ensemble_regularization')
        if self.bins_fs!=len(self.feature_ratios) or min(self.feature_ratios)<=0 or max(self.feature_ratios)>1:
            raise ValueError('invalid_feature_ratios')


def cutoff(value):
    t=pd.Timestamp(value)
    if pd.isna(t): raise ValueError('missing_prediction_time')
    return t.tz_localize('Asia/Shanghai') if t.tzinfo is None else t


def balance_dates(weights, dates):
    w=pd.Series(np.asarray(weights,dtype=float))
    d=pd.Series(np.asarray(dates))
    if len(w)!=len(d) or not np.isfinite(w).all() or w.le(0).any():
        raise ValueError('invalid_sample_weights')
    w=w / w.groupby(d,sort=False).transform('sum')
    return (w / w.mean()).to_numpy()


def shuffle_within_dates(values, dates, rng):
    out=np.array(values,copy=True)
    d=np.asarray(dates)
    for day in pd.unique(d):
        positions=np.flatnonzero(d==day)
        out[positions]=out[rng.permutation(positions)]
    return out


class MonthlyEnsemble:
    def __init__(self,kind,policy=EnsemblePolicy()):
        if kind not in ('plain','double'): raise ValueError('unknown_nonlinear_expert')
        self.kind,self.policy=kind,policy
        self.models=[];self.sub_features=[];self.evidence={}

    def fit(self,frame,as_of):
        import lightgbm as lgb
        self.as_of=cutoff(as_of)
        required=set(FEATURES+['date','instrument_id','label_available_at','return_20','coverage_20'])
        if not required.issubset(frame): raise ValueError('training_schema_missing')
        if frame.duplicated(['date','instrument_id']).any(): raise ValueError('duplicate_training_rows')
        available=pd.to_datetime(frame.label_available_at,utc=True,errors='coerce')
        signal=pd.to_datetime(frame.date,errors='raise')
        prior=signal < self.as_of.tz_localize(None).normalize()
        eligible=frame.loc[prior & available.le(self.as_of) & frame.coverage_20.eq('observed') & np.isfinite(frame.return_20)].copy()
        counts=eligible.groupby('date').return_20.agg(['size','nunique'])
        valid_dates=sorted(counts.index[(counts['size']>=30)&(counts['nunique']>1)])
        if len(valid_dates)<self.policy.cohorts: raise ValueError('insufficient_matured_training_cohorts')
        dates=valid_dates[-self.policy.cohorts:]
        train=eligible[eligible.date.isin(dates)].sort_values(['date','instrument_id']).reset_index(drop=True)
        x=train[FEATURES].replace([np.inf,-np.inf],np.nan)
        y=train.groupby('date').return_20.rank(method='average',pct=True).to_numpy()
        groups=train.date.to_numpy()
        base=balance_dates(np.ones(len(train)),groups)
        weights=base.copy();features=list(FEATURES)
        rng=np.random.default_rng(self.policy.seed)
        self.models=[];self.sub_features=[];predictions=[];weight_checks=[]
        for k in range(self.policy.num_models):
            params=dict(objective='regression',learning_rate=self.policy.learning_rate,
                num_leaves=self.policy.leaves,min_data_in_leaf=self.policy.min_leaf,
                lambda_l2=self.policy.l2,bagging_fraction=self.policy.bagging,bagging_freq=1,
                feature_fraction=1.,seed=self.policy.seed+k,bagging_seed=self.policy.seed+k,
                feature_fraction_seed=self.policy.seed+k,data_random_seed=self.policy.seed+k,
                deterministic=True,force_col_wise=True,num_threads=4,verbosity=-1)
            model=lgb.train(params,lgb.Dataset(x[features],label=y,weight=weights,free_raw_data=False),
                            num_boost_round=self.policy.rounds)
            self.models.append(model);self.sub_features.append(features.copy())
            predictions.append(model.predict(x[features],num_threads=4))
            sums=pd.Series(weights).groupby(groups).sum()
            weight_checks.append(dict(min_date_weight=float(sums.min()),max_date_weight=float(sums.max())))
            if self.kind=='double' and k+1<self.policy.num_models:
                ensemble=np.mean(predictions,axis=0)
                losses=(y-ensemble)**2
                weights=self._sample_weights(model,x[features],y,losses,groups,k+1)
                features=self._select_features(x,y,losses,groups,base,rng)
        self.evidence=dict(kind=self.kind,as_of=str(self.as_of),cohort_dates=dates,
            latest_label_available_at=str(pd.to_datetime(train.label_available_at,utc=True).max()),
            rows=len(train),training_label='20_session_close_price_return_percentile',
            sub_features=self.sub_features,date_weight_checks=weight_checks,
            policy=asdict(self.policy),lightgbm_version=lgb.__version__,
            unknown_training_labels_excluded=int((prior & available.le(self.as_of) & ~frame.coverage_20.eq('observed')).sum()),
            version=self.version)
        return self

    @property
    def version(self):
        spec=dict(kind=self.kind,policy=asdict(self.policy),features=FEATURES,
                  mechanism='within_date_permutation_date_balanced_v1')
        return 'nonlinear-'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]

    def _sample_weights(self,model,x,y,losses,dates,k):
        ntrees=model.current_iteration();part=max(int(ntrees*.1),1)
        running=np.zeros(len(x));start=[];end=[]
        for t in range(ntrees):
            running+=model.predict(x,start_iteration=t,num_iteration=1,num_threads=4)
            if t<part or t>=ntrees-part:
                ranks=pd.Series((y-running)**2).rank(pct=True).to_numpy()
                if t<part:start.append(ranks)
                if t>=ntrees-part:end.append(ranks)
        full=model.predict(x,num_threads=4)
        if not np.allclose(running,full,rtol=1e-10,atol=1e-10):
            raise ValueError('incremental_tree_loss_curve_mismatch')
        h1=pd.Series(-losses).rank(pct=True)
        h2=pd.Series(np.mean(end,axis=0)/np.mean(start,axis=0)).rank(pct=True)
        h=self.policy.alpha1*h1+self.policy.alpha2*h2
        bins=pd.cut(h,self.policy.bins_sr)
        avg=h.groupby(bins,observed=True).transform('mean')
        w=1/(self.policy.decay**k*avg.to_numpy()+.1)
        return balance_dates(w,dates)

    def _select_features(self,x,y,losses,dates,base,rng):
        importance=[]
        for feature in FEATURES:
            changed=x.copy()
            changed[feature]=shuffle_within_dates(x[feature].to_numpy(),dates,rng)
            pred=np.mean([m.predict(changed[f],num_threads=4) for m,f in zip(self.models,self.sub_features)],axis=0)
            delta=(y-pred)**2-losses
            mean=float(np.average(delta,weights=base))
            std=float(np.sqrt(np.average((delta-mean)**2,weights=base)))
            importance.append(mean/(std+1e-7))
        codes=pd.cut(pd.Series(importance),self.policy.bins_fs,labels=False).fillna(0).astype(int)
        selected=[]
        for bin_id in range(self.policy.bins_fs-1,-1,-1):
            members=np.flatnonzero(codes.to_numpy()==bin_id)
            if not len(members):continue
            ratio=self.policy.feature_ratios[self.policy.bins_fs-1-bin_id]
            selected.extend(rng.choice(members,size=int(np.ceil(ratio*len(members))),replace=False).tolist())
        return [f for i,f in enumerate(FEATURES) if i in set(selected)]

    def predict(self,frame,*,as_of):
        if not self.models: raise ValueError('model_not_fitted')
        if cutoff(as_of)<self.as_of: raise ValueError('prediction_before_training_cutoff')
        x=frame[FEATURES].replace([np.inf,-np.inf],np.nan)
        if x.empty or x.isna().all(axis=1).any(): raise ValueError('missing_prediction_features')
        values=np.mean([m.predict(x[f],num_threads=4) for m,f in zip(self.models,self.sub_features)],axis=0)
        if not np.isfinite(values).all(): raise ValueError('invalid_nonlinear_prediction')
        return pd.Series(values,index=frame.index,name='prediction').rank(pct=True,method='average')
