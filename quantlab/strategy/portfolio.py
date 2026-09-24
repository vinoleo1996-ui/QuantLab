"""组合构建：候选擦选 → 目标波动率权重(收缩协方差) + 权重/行业约束。

方法：
- Ledoit-Wolf 收缩协方差(arXiv:2507.07107 实证优于朴素MVO)；
- SLSQP 求解 最小方差且有 权重上下界/行业上限 的权重组；
- 组合层降杠杆到目标年化波动率(多余资金按现金持有)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from quantlab.config import CONFIG, PortfolioConfig


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """用 LedoitWolf 收缩估计组合收益协方差。"""
    r = returns.dropna(how="any")
    if r.shape[0] < 30:
        return returns.cov()
    lw = LedoitWolf().fit(r)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


def _solve_mv(cov: np.ndarray, cap: np.ndarray, sector: np.ndarray,
              max_sector_weight: float) -> np.ndarray:
    n = len(cap)
    tgt = np.zeros(n)
    # 目标：企业风险权重对标的等权初始解，约束满足后用 SLSQP 微调最小方差
    x0 = 1.0 / n * np.ones(n)

    def obj(w):
        return w @ cov @ w

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    # 仅对真实存在的行业施加占比上限；sector==-1 表示"NA/未知"，不参与行业约束，
    # 避免：全 NA 时出现 100%(Σw==1) 与 30%(行业上限) 自相矛盾导致 SLSQP 必然失败。
    n_sec = int(max(sector) + 1) if len(sector) else 0
    for s in range(n_sec):
        if s < 0:
            continue
        mask = np.where(sector == s)[0]
        if len(mask) > 0:
            cons.append({"type": "ineq", "fun": lambda w, m=mask, mw=max_sector_weight: mw - w[m].sum()})
    res = minimize(obj, x0, bounds=[(0.0, c) for c in cap], constraints=cons,
                   method="SLSQP", options={"ftol": 1e-8, "maxiter": 300})
    if not res.success:
        raise ValueError(f"portfolio_infeasible:{res.message}")
    w = res.x
    w = np.clip(w, 0.0, None)
    if w.sum() > 0:
        w /= w.sum()
    return w


def target_vol_weights(returns: pd.DataFrame, sector: pd.Series | None = None,
                       cfg: PortfolioConfig | None = None, vol_mult: float = 1.0) -> pd.DataFrame:
    """入参 returns: date x code 收益面板(若干月)。返回两列 {weight, gross}。

    输出权重 w 满足 Σ=1 且每只 <= 1/N + max_dev；组合按目标波动率(可选缩放)降杠杆。
    vol_mult: 目标波动率缩放(风控 L2 降档用)，有效目标 = cfg.target_vol * vol_mult。
    """
    cfg = cfg or CONFIG.portfolio
    n = returns.shape[1]
    if n == 0:
        return pd.DataFrame(columns=["weight", "gross"])
    base = 1.0 / n
    cap = min(base + cfg.weight_max_dev, 1.0)
    cov = ledoit_wolf_cov(returns)
    cov_np = cov.to_numpy()
    if sector is None:
        sec = -1.0 * np.ones(n)
    else:
        codes = list(returns.columns)
        sval = sector.reindex(codes)
        is_na = sval.isna() | (sval.astype(str).str.strip() == "NA")
        groups, _uniq = pd.factorize(sval.astype(str))
        sec = np.where(is_na, -1.0, groups.astype(float))
    w = _solve_mv(cov_np, np.ones(n) * cap, sec, cfg.max_sector_weight)
    # 目标波动率缩放(只降不加，避免高杠杆放大风险)。
    # cov 为日频，需乘 sqrt(252) 折算为年化后再与 target_vol(年化) 比较。
    port_var = w @ cov_np @ w
    port_vol_d = float(np.sqrt(max(port_var, 1e-12)))
    port_vol_y = port_vol_d * np.sqrt(252)
    target = cfg.target_vol * vol_mult
    scale = min(1.0, target / port_vol_y) if port_vol_y > 0 else 1.0
    w_scaled = w * scale
    result = pd.DataFrame({
        "weight": pd.Series(w_scaled, index=returns.columns),
        "gross": pd.Series(np.ones(n) * scale, index=returns.columns),
    })
    return result


def select_top(scored: pd.Series, n: int, extra: int = 0) -> list[str]:
    """取打分最高的 n+extra 只(风控替换的候选)。"""
    s = scored.dropna().sort_values(ascending=False)
    return list(s.index[:n + extra])
