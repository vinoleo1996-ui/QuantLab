"""因子中性化：行业横截面去均值 + 市值分位去均值。

背景：裸截面 Z-Score 排名会把"银行天然低 PE / 电子天然高 PE"的行业系统差异
当成 Alpha，导致价值专家永远押注传统板块、动量专家偶尔同仓踩踏。
业界标准做法（Qlib/中信/申万口径）：
  1. 按申万一级行业分组，组内去均值(Demean)；
  2. 再按市值分位桶去均值，剥离"小市值因子"的同源暴露。
此处实现轻量离线版（组数足够时生效，样本<5 跳过）。

仅用于因子打分前的清洗；不改变 IC 计算口径(IC 使用清洗后面板，保持一致)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sector_demean(row: pd.Series, sector_map: dict[str, str]) -> pd.Series:
    """单期截面按行业分组去均值。sector_map: {code: 行业}。
    门槛为组内有效样本 >=3(小池 40~200 只时大多数行业仅 1~3 只，5 会静默跳过 80%)。"""
    codes = list(row.index)
    sec = pd.Series([sector_map.get(str(c), "NA") for c in codes], index=codes)
    out = row.copy()
    for s in set(sec.unique()):
        m = sec == s
        group = row[m]
        if group.notna().sum() >= 3:
            out[m] = group - group.mean(skipna=True)
    return out


def cap_bucket_demean(row: pd.Series, cap_map: dict[str, float], n_buckets: int = 5) -> pd.Series:
    """单期截面按流通市值分位桶去均值。cap_map: {code: float_mktcap}。"""
    codes = list(row.index)
    caps = pd.Series([cap_map.get(str(c), np.nan) for c in codes], index=codes)
    out = row.copy()
    ranks = caps.rank(pct=True, na_option="keep")
    cuts = np.linspace(0.0, 1.0, n_buckets + 1)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        m = (ranks > lo) & (ranks <= hi)
        group = row[m & caps.notna()]
        if group.notna().sum() >= 5:
            out[m & caps.notna()] = group - group.mean(skipna=True)
    return out


def neutralize_panel(panel: pd.DataFrame, sector_map: dict[str, str] | None = None,
                     cap_map: dict[str, float] | None = None) -> pd.DataFrame:
    """面板级中性化：每期截面先行业去均值，再市值分位去均值。

    不修改指数名/列；无行业或市值信息时返回原面板(no-op)。
    """
    if panel is None or panel.empty:
        return panel
    out = panel.copy()
    for d in panel.index:
        row = panel.loc[d]
        if row.notna().sum() < 10:
            continue
        if sector_map:
            row = sector_demean(row, sector_map)
        if cap_map:
            row = cap_bucket_demean(row, cap_map)
        out.loc[d] = row
    return out