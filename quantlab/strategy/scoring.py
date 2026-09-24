"""因子合成核心：RankIC / ICIR 滚动评估、动态加权、因子拥挤惩罚。

思想来源：
- Qlib: IC/IR 因子评估与合成框架；
- Quant Decoded FCI(2026-04): 因子收益两两相关性是最强的拥挤单信号；
- 东北证券 CNEX / 西南证券动态分域: 拥挤环境因子失效、应降敞口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """逐期截面 RankIC(因子的截面rank 与 未来收益 rank 的 Spearman 相关)。"""
    ic = {}
    dates = factor.index.intersection(fwd_ret.index)
    for t in dates:
        f = factor.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(f.index).dropna()
        f = f.reindex(r.index)
        if len(f) < 10:
            continue
        rho, _ = spearmanr(f, r)
        if np.isfinite(rho):
            ic[t] = rho
    return pd.Series(ic, name="rank_ic").sort_index()


def rolling_icir(ic: pd.Series, window: int = 60) -> pd.DataFrame:
    """滚动 ICIR = rolling_mean(IC)/rolling_std(IC)。"""
    mean = ic.rolling(window, min_periods=24).mean()
    std = ic.rolling(window, min_periods=24).std()
    ir = mean / (std + 1e-9)
    return pd.concat({"ic": mean, "icir": ir, "ic_mean": mean}, axis=1)


def adaptive_weight(icir, floor: float = 0.30, cap: float = 1.60):
    """由 ICIR 决定因子权重：动机正则的线性映射，floor 以下压到 0.1，cap 封顶。
    兼容标量与 Series 输入。"""
    if isinstance(icir, pd.Series):
        w = icir.clip(lower=floor, upper=cap)
        w = (w - floor) / max(cap - floor, 1e-9)
        w[w <= 0.0] = 0.1
        return w.rename("weight")
    s = float(np.clip(float(icir), floor, cap))
    w = (s - floor) / max(cap - floor, 1e-9)
    return float(w) if w > 0.0 else 0.1


def decay_penalty(ic: pd.Series, lookback: int = 24, threshold: float = 0.0) -> float:
    """近 lookback 期 IC 均值的衰减惩罚(近期失效则权重打折)。"""
    if len(ic) < lookback:
        return 1.0
    recent = ic.iloc[-lookback:].mean()
    past = ic.iloc[-lookback * 2:-lookback].mean()
    if not np.isfinite(past):
        return 1.0
    if past <= 0.0:
        # 过去本来就长期无效(负均值)，直接强惩罚，不能豁免
        return 0.4
    if recent <= 0.0:
        return 0.4
    ratio = recent / past
    if ratio >= 1.0:
        return 1.0
    return max(0.4, ratio)


def _rolling_corr_panel(factor_ret: pd.DataFrame, window: int, names: list[str]) -> pd.DataFrame:
    """因子收益两两相关均值(全因子平均对相关)，是 FCI 最强拥挤信号。

    实现：对每对因子做滚动相关系数，再把每个因子与其所有对手的对相关逐期平均。
    """
    out = pd.DataFrame(np.nan, index=factor_ret.index, columns=names)
    cols = list(factor_ret.columns)
    if len(cols) < 2:
        return out
    pairs: list[tuple[str, str, pd.Series]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = factor_ret[cols[i]].rolling(window, min_periods=20).corr(factor_ret[cols[j]])
            pairs.append((cols[i], cols[j], c))
    for k in names:
        rel = [p[2] for p in pairs if k in (p[0], p[1])]
        if not rel:
            continue
        out[k] = pd.concat(rel, axis=1).mean(axis=1)
    return out


def factor_crowding_penalty(factor_ret: pd.DataFrame, corr_window: int = 60,
                            alert: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (惩罚系数面板, 拥挤水平面板)。corr 高于 alert 的因子惩罚系数 < 1。"""
    fac_n = factor_ret.shape[1]
    corr = _rolling_corr_panel(factor_ret, corr_window, list(factor_ret.columns))
    p = 1.0 - (corr - alert).clip(lower=0, upper=1) / (1 - alert) * 0.5
    p = p.clip(lower=0.5, upper=1.0)
    return p, corr


def composite_score(factors: dict[str, pd.DataFrame], weights: pd.DataFrame,
                    crowding_penalty: pd.DataFrame | None) -> pd.DataFrame:
    """Strict score contract: unknown weights/features remain invalid, never zero alpha."""
    if not factors:
        return pd.DataFrame()
    names = list(factors)
    if weights is None or weights.empty or not set(names).issubset(weights.columns):
        raise ValueError("missing_factor_weights")
    if weights[names].isna().any().any() or not np.isfinite(weights[names].to_numpy()).all():
        raise ValueError("missing_factor_weights")
    if (weights[names].abs().sum(axis=1) <= 0).any():
        raise ValueError("zero_factor_weights")
    dates = factors[names[0]].index
    codes = sorted(set.intersection(*(set(f.columns) for f in factors.values())))
    for f in factors.values(): dates = dates.intersection(f.index)
    # Only weight dates are decisions. Earlier factor rows must not inherit future weights.
    dates = dates.intersection(weights.index)
    if dates.empty: raise ValueError("weight_date_mismatch")
    score = pd.DataFrame(0.0, index=dates, columns=codes)
    valid = pd.DataFrame(True, index=dates, columns=codes)
    for name, f in factors.items():
        w = weights[name].reindex(dates)
        penalty = pd.Series(1.0, index=dates)
        if crowding_penalty is not None and name in crowding_penalty:
            penalty = crowding_penalty[name].reindex(dates).fillna(1.0)
        x = f.reindex(index=dates, columns=codes)
        active = w.ne(0)
        valid &= x.notna() | pd.DataFrame(np.repeat((~active).to_numpy()[:, None], len(codes), axis=1), index=dates, columns=codes)
        score += x.fillna(0.0).mul(w * penalty, axis=0)
    score = score.where(valid)
    score.attrs["invalid_dates"] = {str(t): "constant_cross_section" for t in dates
                                    if score.loc[t].dropna().nunique() <= 1}
    for t in dates:
        if str(t) in score.attrs["invalid_dates"]: score.loc[t] = np.nan
    return score


def weight_row(weights: pd.Series, date) -> pd.DataFrame:
    """Construct a dated row explicitly; transposed Series has a name, not a date index."""
    return pd.DataFrame([weights.to_dict()], index=pd.DatetimeIndex([date]))
