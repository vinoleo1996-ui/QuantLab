"""Expected-return allocation with cash, real turnover fees and explicit exit constraints."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
from sklearn.covariance import LedoitWolf
from quantlab.data.contracts import InstrumentRules


def allocate(returns, records, exposures, cfg, gross=.98, vol_scale=1., expected=None, previous=None,
             *, risk_snapshot=None, locked=None, nav=None, date=None, cost_mult=1., allowed_new=None, industry_policy="verified"):
    codes=list(returns.columns);empty=pd.Series(0.,index=codes)
    if not codes:return empty,'no_candidates'
    missing=[c for c in codes if records.loc[c].asset_type=='stock' and
             (pd.isna(records.loc[c].get('sector')) or str(records.loc[c].get('sector')) in ('UNKNOWN','NA',''))]
    if missing and industry_policy!='disabled_diagnostic':return empty,'industry_coverage_insufficient'
    prior=pd.Series(dtype=float) if previous is None else previous
    prior=prior.reindex(codes,fill_value=0.).to_numpy()
    r=returns.dropna()
    if len(r)<40:return empty,'insufficient_covariance_history'
    cov=LedoitWolf().fit(r).covariance_ if cfg.portfolio.lw_shrinkage else r.cov().to_numpy()
    caps=np.array([cfg.research.max_etf_weight if records.loc[c].asset_type=='equity_etf' else cfg.research.max_stock_weight for c in codes])
    lower=pd.Series(locked or {},dtype=float).reindex(codes,fill_value=0.).to_numpy()
    if allowed_new is not None:
        caps=np.minimum(caps,np.array([caps[i] if c in allowed_new else prior[i] for i,c in enumerate(codes)]))
    matrix=exposures.reindex(columns=codes,fill_value=0).to_numpy()
    limits=np.full(len(matrix),cfg.portfolio.max_sector_weight)
    if risk_snapshot is not None:
        cluster=risk_snapshot.matrix(codes).to_numpy()
        matrix=np.vstack([matrix,cluster]);limits=np.r_[limits,np.full(len(cluster),cfg.research.max_cluster_weight)]
    max_gross=min(gross,1-cfg.research.cash_buffer,float(caps.sum()))
    if expected is None and len(exposures):
        industry_matrix=exposures.reindex(columns=codes,fill_value=0).to_numpy()
        max_gross=min(max_gross,sum(min(cfg.portfolio.max_sector_weight,float(row@caps)) for row in industry_matrix))
    if np.any(lower>caps+1e-8) or lower.sum()>max_gross+1e-8 or np.any(matrix@lower>limits+1e-8):
        return empty,'locked_holdings_violate_constraints'
    if max_gross<=0:return empty,'cash_only'
    constraints=[{'type':'ineq' if expected is not None else 'eq','fun':lambda w:max_gross-w.sum()}]
    for row,limit in zip(matrix,limits):constraints.append({'type':'ineq','fun':lambda w,a=row,b=limit:b-a@w})
    target=cfg.portfolio.target_vol*vol_scale
    if expected is not None:constraints.append({'type':'ineq','fun':lambda w:target**2/252-float(w@cov@w)})
    nav=float(nav or cfg.capital.init_cash);date=date or cfg.start_date
    from quantlab.backtest.ledger import Ledger
    ledger=Ledger(slippage=cfg.cost.a_slippage,cost_mult=cost_mult)
    rules=[InstrumentRules.from_record(dict(records.loc[c],instrument_id=c)) for c in codes]
    mu=np.zeros(len(codes)) if expected is None else expected.reindex(codes).to_numpy()
    if not np.isfinite(mu).all():return empty,'missing_expected_returns'
    heat=np.zeros(len(codes)) if risk_snapshot is None else risk_snapshot.overheating.reindex(codes,fill_value=1.).to_numpy()
    liquid=np.zeros(len(codes)) if risk_snapshot is None else risk_snapshot.illiquidity.reindex(codes,fill_value=1.).to_numpy()
    def costs(w):
        total=0.
        for i,change in enumerate(w-prior):
            value=abs(change)*nav
            if value<1e-6:continue
            commission,stamp=ledger.fee(date,rules[i],'buy' if change>0 else 'sell',value)
            total+=commission+stamp+value*cfg.cost.a_slippage*cost_mult
        return total/nav
    def objective(w):
        variance=float(w@cov@w)
        if expected is None:return variance*10000
        return (cfg.research.risk_aversion*cfg.research.horizon*variance-mu@w+costs(w)+
                cfg.research.crowding_cost*heat@w+cfg.research.liquidity_cost*liquid@w)
    initial=np.minimum(caps,max_gross/max(len(codes),1)*np.ones(len(codes)))
    if expected is not None:
        positive=np.where(mu>0,caps,lower)
        if positive.sum()>max_gross:positive*=max_gross/positive.sum()
        starts=[lower,np.maximum(lower,np.minimum(prior,caps)),np.maximum(lower,positive)]
    else:
        feasible=linprog(np.zeros(len(codes)),A_ub=matrix if len(matrix) else None,b_ub=limits if len(matrix) else None,
                        A_eq=np.ones((1,len(codes))),b_eq=[max_gross],bounds=list(zip(lower,caps)),method='highs')
        if not feasible.success:return empty,'solver_failed_or_infeasible'
        starts=[feasible.x]
    candidates=[]
    def valid(w):
        return (np.isfinite(w).all() and np.all(w>=lower-1e-7) and np.all(w<=caps+1e-7) and w.sum()<=max_gross+1e-7 and
                np.all(matrix@w<=limits+1e-7) and (expected is None or w@cov@w<=target**2/252+1e-9) and
                (expected is not None or abs(w.sum()-max_gross)<1e-6))
    for x0 in starts:
        opt=minimize(objective,x0,method='SLSQP',bounds=list(zip(lower,caps)),constraints=constraints,options={'ftol':1e-10,'maxiter':200})
        if opt.success and valid(opt.x):candidates.append(opt.x)
    if not candidates:return empty,'solver_failed_or_infeasible'
    w=min(candidates,key=objective)
    if expected is None:
        vol=float(np.sqrt(max(w@cov@w,0)*252))
        if vol>target:w*=target/vol
    return pd.Series(np.maximum(w,0),index=codes),'ok'
