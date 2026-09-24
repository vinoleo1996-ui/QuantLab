"""Original factor strategy on explicit account constraints, without return forecasts."""
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.covariance import LedoitWolf


def legacy_config(cfg):
    """Versioned strategy defaults; never mutate the caller's model configuration."""
    result = deepcopy(cfg)
    result.capital.rebalance_rule = 'monthly'
    return result


def allocate_legacy(returns, cfg, *, selected, previous=None, locked=None,
                    exposures=None, risk_snapshot=None, gross=1., vol_scale=1.,
                    budget=None, equal_weight=False, allowed_increases=None):
    """Max feasible gross, then min variance at that gross. No alpha/penalty fallback.

    Relative position caps act on final gross, not on an impossible full-investment
    portfolio before applying industry constraints. Locked shares are lower bounds.
    """
    codes = list(returns.columns)
    empty = pd.Series(0., index=codes)
    detail = {'policy': 'legacy_max_gross_equal_weight_v1' if equal_weight else
                       'legacy_max_gross_min_variance_v1', 'feasible_gross': None}
    policy = budget.as_dict() if hasattr(budget, 'as_dict') else budget
    relaxed = policy is not None and policy['policy_id'] != 'P0'
    if policy is not None:
        if policy['policy_id'] not in ('P0', 'P1', 'P2'):
            raise ValueError('unknown_risk_policy')
        gross, vol_scale = min(gross, policy['gross_scale']), min(vol_scale, policy['vol_scale'])
        detail.update(risk_policy=policy['policy_id'], risk_policy_version=policy['version'],
                      risk_state=policy['state'])
    if not codes: return empty, 'no_candidates', detail
    r = returns.dropna()
    if len(r) < 40: return empty, 'insufficient_covariance_history', detail
    cov = (LedoitWolf().fit(r).covariance_ if cfg.portfolio.lw_shrinkage else r.cov().to_numpy()) * 252
    if not np.isfinite(cov).all(): return empty, 'invalid_covariance', detail
    n = len(codes); desired = set(selected)
    lower = pd.Series(locked or {}, dtype=float).reindex(codes, fill_value=0.).to_numpy()
    prior = pd.Series({} if previous is None else previous, dtype=float).reindex(codes, fill_value=0.).to_numpy()
    if not np.isfinite(prior).all() or not np.isfinite(lower).all() or np.any(prior < 0) or np.any(lower < 0):
        return empty, 'invalid_previous_or_locked_weights', detail
    if previous is None:
        prior = lower.copy()
    stock_cap = cfg.research.max_stock_weight if policy is None else policy['max_stock_weight']
    soft_stock = stock_cap if not relaxed else policy['soft_stock_weight']
    # A soft entry target never forces a still-permitted existing holding down.
    caps = np.array([min(stock_cap, max(prior[i], soft_stock)) if c in desired else 0.
                     for i, c in enumerate(codes)])
    if allowed_increases is not None:
        allowed = set(allowed_increases)
        caps = np.minimum(caps, [caps[i] if c in allowed else prior[i]
                                 for i, c in enumerate(codes)])
        detail['allowed_increases'] = sorted(allowed)
    cash_buffer = cfg.research.cash_buffer if policy is None else policy['cash_buffer']
    ceiling = min(gross, 1-cash_buffer)
    rows = [np.ones(n)]; limits = [ceiling]; names = ['cash_and_risk_gross']
    def group_limit(name, row, kind):
        legacy_limit = cfg.portfolio.max_sector_weight if kind == 'sector' else cfg.research.max_cluster_weight
        if policy is None or not relaxed:
            return legacy_limit
        unknown = 'UNKNOWN' in str(name).upper()
        hard = policy['unknown_'+kind+'_weight'] if unknown else policy['max_'+kind+'_weight']
        soft = max(0., hard-.02) if unknown else policy['soft_'+kind+'_weight']
        return min(hard, max(soft, float(row @ prior)))
    if exposures is not None:
        for name, row in exposures.reindex(columns=codes, fill_value=0.).iterrows():
            values = row.to_numpy()
            rows.append(values); limits.append(group_limit(name, values, 'sector')); names.append('industry:'+str(name))
    if risk_snapshot is not None:
        for name, row in risk_snapshot.matrix(codes).iterrows():
            values = row.to_numpy()
            rows.append(values); limits.append(group_limit(name, values, 'cluster')); names.append('downside:'+str(name))
    # Archived 1/N + deviation applies to the invested basket, not account NAV.
    relative_cap = None if relaxed else 1/max(len(desired), 1)+cfg.portfolio.weight_max_dev
    if relative_cap is not None:
        for i,c in enumerate(codes):
            if c in desired:
                row = np.full(n, -relative_cap); row[i] += 1
                rows.append(row); limits.append(0.); names.append('relative_weight:'+c)
    a = np.array(rows); b = np.array(limits)
    bounds = list(zip(lower, caps))
    detail.update(candidate_count=len(desired), covariance_rows=len(r), cash_and_risk_ceiling=ceiling,
                  individual_capacity=float(caps.sum()), locked_gross=float(lower.sum()))
    if np.any(lower>caps+1e-8): return empty, 'locked_holdings_violate_constraints', detail
    lp = linprog(-np.ones(n), A_ub=a, b_ub=b, bounds=bounds, method='highs')
    if not lp.success: return empty, 'linear_constraints_infeasible', detail
    target_vol = cfg.portfolio.target_vol if policy is None else policy['target_vol']
    target_var = (target_vol*vol_scale)**2
    scale = max(target_var, 1e-6)
    linear = {'type':'ineq','fun':lambda w:b-a@w,'jac':lambda w:-a}
    vol = {'type':'ineq','fun':lambda w:1-float(w@cov@w)/scale,
           'jac':lambda w:-2*cov@w/scale}
    options = {'ftol':1e-10,'maxiter':500}
    def valid(w):
        return (np.isfinite(w).all() and np.all(w>=lower-1e-7) and np.all(w<=caps+1e-7)
                and np.all(a@w<=b+1e-7) and w@cov@w<=target_var+1e-8)
    if valid(lp.x):
        max_weights = lp.x  # LP is already a certificate of maximal gross.
    else:
        first = minimize(lambda w:-w.sum(), lp.x, jac=lambda w:-np.ones(n), bounds=bounds,
                         constraints=[linear,vol], method='SLSQP', options=options)
        if not first.success or not valid(first.x):
            detail['solver_message'] = first.message
            return empty, 'maximum_gross_solver_failed', detail
        max_weights = first.x
    # Stage 1 may exceed the LP upper bound by roundoff. Do not turn that
    # tolerated error into an infeasible exact equality in stage 2.
    g = min(float(max_weights.sum()), float(lp.x.sum())); detail['feasible_gross'] = g
    if g < 1e-9: return empty, 'ok', detail
    equality = {'type':'eq','fun':lambda w:w.sum()-g,'jac':lambda w:np.ones(n)}
    # Constant-row inequalities are implied by sum(w)=g. Removing them avoids
    # a redundant active equality/inequality at an all-UNKNOWN industry ceiling.
    nonconstant = np.ptp(a,axis=1)>1e-12
    aa,bb=a[nonconstant],b[nonconstant]
    fixed_linear={'type':'ineq','fun':lambda w:bb-aa@w,'jac':lambda w:-aa}
    if equal_weight:
        # An explicit comparator on the same feasible set, never a solver fallback.
        equal = np.array([g/max(len(desired), 1) if c in desired else 0. for c in codes])
        objective, jacobian = lambda w:float((w-equal)@(w-equal)), lambda w:2*(w-equal)
    else:
        objective, jacobian = lambda w:float(w@cov@w)/scale, lambda w:2*cov@w/scale
    second_constraints = [fixed_linear, equality] + ([vol] if equal_weight else [])
    second = minimize(objective, max_weights, jac=jacobian,
                      bounds=bounds, constraints=second_constraints, method='SLSQP', options=options)
    if not second.success or not valid(second.x) or abs(second.x.sum()-g)>1e-7:
        detail['solver_message'] = second.message
        return empty, 'minimum_variance_solver_failed', detail
    w = second.x
    detail.update(target_vol=float(np.sqrt(w@cov@w)),
                  binding_constraints=[name for name,slack in zip(names,b-a@w) if slack<1e-6],
                  target_gross=float(w.sum()), relative_weight_cap=relative_cap,
                  target_weights={c:float(v) for c,v in zip(codes,w)})
    return pd.Series(np.maximum(w,0.), index=codes), 'ok', detail
