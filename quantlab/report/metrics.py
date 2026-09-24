"""绩效指标与基准对比。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ann_return(nav_or_ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    if len(nav_or_ret) < 2:
        return 0.0
    total = float(nav_or_ret.iloc[-1] / nav_or_ret.iloc[0]) if nav_or_ret.iloc[0] != 0 else 1.0
    yrs = len(nav_or_ret) / periods
    return float(total ** (1 / max(yrs, 1e-6)) - 1) if total > 0 else -1.0


def ann_vol(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    return float(r.std() * np.sqrt(periods)) if r.size > 1 else 0.0


def sharpe(ret: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r=pd.to_numeric(ret,errors='coerce').dropna();v=ann_vol(r,periods)
    return float((r.mean()*periods-rf)/v) if v>0 else 0.0


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min()) if dd.size else 0.0


def calmar(nav: pd.Series) -> float:
    mdd = max_drawdown(nav)
    return float(ann_return(nav) / abs(mdd)) if mdd < 0 else 0.0


def win_rate(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    return float((r > 0).mean()) if r.size else 0.0


def beta_alpha(nav: pd.Series, bench_nav: pd.Series, periods: int = TRADING_DAYS) -> dict:
    a = pd.to_numeric(nav, errors="coerce").pct_change(fill_method=None).dropna()
    b = pd.to_numeric(bench_nav, errors="coerce").pct_change(fill_method=None).dropna()
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if df.shape[0] < 20:
        return {"beta": 0.0, "alpha_ann": 0.0, "excess": 0.0}
    cov = float(np.cov(df["a"], df["b"])[0][1])
    var = float(df['b'].var())
    beta = cov / var if var > 0 else 0.0
    alpha_ann = (df.a.mean()-beta*df.b.mean())*periods
    excess = (df.a.mean()-df.b.mean())*periods
    return {"beta": beta, "alpha_ann": alpha_ann, "excess": excess}


def performance_table(nav: pd.Series, daily_ret: pd.Series,
                      bench_nav: pd.Series | None = None) -> dict:
    if nav.empty or daily_ret.empty:
        return {k: None for k in ("total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown", "calmar", "win_rate")}
    r = daily_ret.dropna()
    total = float((1+r).prod())
    # Include first day's fees/P&L. Calendar CAGR and 252-session annualization are separate.
    days = (pd.Timestamp(r.index[-1])-pd.Timestamp(r.index[0])).days + 1
    years = days / 365.2425
    annual = total ** (1/max(years, 1/365.2425))-1 if total>0 else -1.
    session_annual = total ** (252/len(r))-1 if total>0 else -1.
    vol = ann_vol(r)
    wealth = (1+r).cumprod()
    dd = float((wealth/wealth.cummax().clip(lower=1.)-1).min())
    out = {"total_return": total-1, "annual_return": annual, "annual_return_252": session_annual,
           "annual_vol": vol, "sharpe": float(r.mean()*252/vol) if vol>0 else 0.,
           "max_drawdown": dd, "calmar": annual/abs(dd) if dd<0 else 0., "win_rate": win_rate(r)}
    if bench_nav is not None and len(bench_nav)>1:
        br = bench_nav.pct_change(fill_method=None).reindex(r.index)
        pair = pd.concat({"strategy": r, "benchmark": br}, axis=1).dropna()
        if len(pair)>20:
            beta = pair.strategy.cov(pair.benchmark)/pair.benchmark.var() if pair.benchmark.var()>0 else 0.
            out.update(beta=float(beta), alpha_ann=float((pair.strategy.mean()-beta*pair.benchmark.mean())*252),
                       relative_wealth=float((1+pair.strategy).prod()/(1+pair.benchmark).prod()-1))
    return out


def ic_icir_summary(ic_matrix: pd.DataFrame) -> pd.DataFrame:
    """因子 IC 汇总：平均IC、ICIR、近12期IC。"""
    rows = []
    for col in ic_matrix.columns:
        s = ic_matrix[col].dropna()
        if s.size < 10:
            continue
        rows.append({
            "factor": col,
            "ic_mean": s.mean(), "ic_std": s.std(),
            "icir": s.mean() / (s.std() + 1e-9),
            "recent_ic": s.tail(12).mean(),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("icir", ascending=False)
