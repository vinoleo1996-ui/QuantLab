"""Constrained legacy repair allocation. New alpha and crowding model remain research-gated."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def allocate(returns, records, exposures, cfg, gross=0.98, vol_scale=1.0, expected=None, previous=None):
    codes = list(returns.columns)
    empty = pd.Series(0., index=codes)
    if not codes: return empty, "no_candidates"
    r = returns.dropna()
    if len(r) < 40: return empty, "insufficient_covariance_history"
    cov = LedoitWolf().fit(r).covariance_ if cfg.portfolio.lw_shrinkage else r.cov().to_numpy()
    caps = np.array([cfg.research.max_etf_weight if records.loc[c].asset_type == "equity_etf"
                     else cfg.research.max_stock_weight for c in codes])
    matrix = exposures.reindex(columns=codes, fill_value=0).to_numpy()
    sector_cap = cfg.portfolio.max_sector_weight
    max_gross = min(gross, 1-cfg.research.cash_buffer, caps.sum())
    if len(matrix): max_gross = min(max_gross, sum(min(sector_cap, float(row @ caps)) for row in matrix))
    if max_gross <= 0: return empty, "cash_only"
    x0 = np.minimum(caps, max_gross / len(codes))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - max_gross}]
    for row in matrix:
        constraints.append({"type": "ineq", "fun": lambda w, a=row: sector_cap - a @ w})
    target = cfg.portfolio.target_vol * vol_scale
    def objective(w):
        base = float(w @ cov @ w)
        if expected is None: return base
        # Same 20-session horizon for expected return and covariance.
        mu = expected.reindex(codes).to_numpy()
        prior = previous.reindex(codes, fill_value=0).to_numpy() if previous is not None else np.zeros(len(codes))
        return cfg.research.risk_aversion * cfg.research.horizon * base - mu @ w + cfg.cost.a_slippage * np.abs(w-prior).sum()
    result = minimize(objective, x0, method="SLSQP", bounds=[(0., c) for c in caps],
                      constraints=constraints, options={"ftol": 1e-10, "maxiter": 200})
    if not result.success: return empty, "solver_failed:" + result.message
    w = result.x
    if abs(w.sum()-max_gross) > 1e-6 or (w < -1e-8).any() or (w > caps+1e-7).any() or (matrix @ w > sector_cap+1e-7).any():
        return empty, "solver_constraint_violation"
    vol = float(np.sqrt(max(w @ cov @ w, 0.) * 252))
    if vol > target: w *= target / vol
    return pd.Series(w.clip(0), index=codes), "ok"
