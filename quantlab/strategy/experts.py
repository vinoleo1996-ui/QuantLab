"""多专家(Multi-Expert)因子集成：多个"专家"并行打分，按专家级近期 IC 动态加权合并。

思想来源：
- RD-Agent / Qlib 的"多个 agent/模型各司其职、按验证表现融合"；
- 业界"动量/价值/质量/流动 双因素风格切换"归因：不同市场状态由不同专家主导；
- 单一复合打分在结构分化市容易极端集中（如单调同质因子权重过大）。

设计：
  1. 专家 = 一组同类因子的线性合成（沿用 ICIR 动态权重 × 拥挤惩罚）。
  2. 每个专家输出截面 Rank 分位数(0~1)。
  3. 专家级权重 = 该组因子近期 IC 均值的 softmax（正激光权，衰退自动降权）。
  4. 最终 score = Σ(专家权重 × 专家分位)，再交给 top-N 选取与目标波动率配权。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.strategy.scoring import composite_score

EXPERT_SPECS: dict[str, list[str] | None] = {
    "momentum": ["momentum_12m1m", "momentum_3m", "momentum_1m_reversal", "momentum_trend"],
    "value": ["value_breadth", "value_medium", "value_lowtension"],
    "lowvol_quality": ["factor_volatility", "factor_price_crowding"],
    "flow": ["flow_turnover_low", "flow_volume_trend"],
    "composite": None,  # 全因子 = 原单专家
}

EXPERT_CN: dict[str, str] = {
    "momentum": "动量趋势",
    "value": "价值低估",
    "lowvol_quality": "低波质量",
    "flow": "资金流向",
    "composite": "综合信号",
}

FACTOR_CN: dict[str, str] = {
    "momentum_12m1m": "中期动量强(一年行情 剔除近一月)排名靠前",
    "momentum_3m": "近三月动量强",
    "momentum_1m_reversal": "近一月超跌/反转特征",
    "momentum_trend": "价格站在均线之上(趋势向上)",
    "value_breadth": "估值低位(全市场横截面)",
    "value_medium": "中等估值区间具备性价比",
    "value_lowtension": "估值压力小/安全边际厚",
    "factor_volatility": "价格波动低(稳定性)",
    "factor_price_crowding": "交易拥挤度低(未被过度炒作)",
    "flow_turnover_low": "换手率低(筹码稳定)",
    "flow_volume_trend": "成交量温和放大/资金活跃度回升",
}


def _group_score(full: dict[str, pd.DataFrame], w_row: pd.DataFrame,
                 pen: pd.DataFrame | None, names: list[str] | None) -> pd.DataFrame | None:
    fg = {k: v for k, v in full.items() if names is None or k in names}
    if not fg or not any(c in w_row.columns for c in fg):
        return None
    wg = w_row[[c for c in fg.keys() if c in w_row.columns]]
    pg = (pen[[c for c in fg.keys() if c in pen.columns]]
          if pen is not None and not pen.empty else None)
    return composite_score(fg, wg, pg)


def experts_score(full: dict[str, pd.DataFrame], w_row: pd.DataFrame,
                  pen: pd.DataFrame | None, ic_use: pd.DataFrame,
                  specs: dict[str, list[str] | None] | None = None,
                  lookback: int | None = None,
                  ) -> tuple[pd.DataFrame | None, dict[str, float]]:
    """多专家集成打分。

    参数
    ----
    full   : {因子名: 因子面板}（≤ 当期数据）
    w_row  : 当期各因子动态权重(1 行 DataFrame)
    pen    : 因子拥挤惩罚面板(可无)
    ic_use : 因子 IC 历史(DataFrame: index=调仓期, columns=因子名)
    lookback: 专家质量评估用的最近 IC 期数

    返回 (综合得分面板, 专家权重 dict)。
    """
    cfg = CONFIG.factors
    lookback = lookback or getattr(cfg, "expert_ic_lookback", 12)
    specs = specs or EXPERT_SPECS
    rank_panels: dict[str, pd.DataFrame] = {}
    smoke: dict[str, float] = {}
    for name, names in specs.items():
        group = [k for k in (names or list(full.keys())) if k in full]
        ic_grp = [c for c in group if c in ic_use.columns]
        if not ic_grp:
            smoke[name] = 0.0
            continue
        ic_row = ic_use[ic_grp].iloc[-lookback:]
        q = float(ic_row.mean(axis=1).mean(skipna=True)) if len(ic_row) else 0.0
        smoke[name] = q if np.isfinite(q) else 0.0
        s = _group_score(full, w_row, pen, group)
        if s is None or s.empty:
            rank_panels[name] = None
            continue
        rank_panels[name] = s.rank(axis=1, pct=True)

    live = {k: v for k, v in rank_panels.items() if v is not None}
    if not live:
        return None, smoke

    q = np.array([max(smoke[k], 0.0) for k in live])
    if q.sum() <= 0.0:
        w_exp = np.full(len(live), 1.0 / len(live))
    else:
        # 自适应温度：拉开强弱专家权重阶梯，避免 IC 过小时恒等权
        tau = max(float(np.std(q)), 0.015)
        w_exp = np.exp(q / tau) / np.exp(q / tau).sum()

    all_idx = sorted({d for p in live.values() for d in p.index})
    all_cols = sorted({c for p in live.values() for c in p.columns})
    total = pd.DataFrame(0.0, index=all_idx, columns=all_cols)
    for (k, p), w in zip(live.items(), w_exp):
        total += p.reindex(index=all_idx, columns=all_cols) * w
    score = total.sort_index()
    return score, {k: round(float(w), 4) for k, w in zip(live.keys(), w_exp)}


def expert_mention_reasons(full: dict[str, pd.DataFrame], w_row: pd.DataFrame,
                           specs: dict[str, list[str] | None] | None = None,
                           by: str | None = None) -> dict[str, dict[str, str]]:
    """每位专家在当期(最后一行)选中池中，为每只票给出"哪只因子最支持"的中文理由。

    by: 指定某位专家时只返回该专家的 {code: 理由}，否则返回 {专家名: {code: 理由}}。
    """
    specs = specs or EXPERT_SPECS
    out: dict[str, dict[str, str]] = {}
    for name, names in specs.items():
        group = [k for k in (names or list(full.keys())) if k in full]
        s = _group_score(full, w_row, None, group)
        if s is None or s.empty:
            continue
        reason: dict[str, str] = {}
        for code in s.columns:
            left = s.iloc[-1].get(code)
            if left is None or not np.isfinite(left):
                continue
            best, bestv = None, -1.0
            for f in group:
                if f not in full or code not in full[f].columns:
                    continue
                row = full[f].iloc[-1]
                v = row.get(code)
                if v is None or not np.isfinite(v):
                    continue
                rk = float(row.rank(pct=True).get(code, 0.0))
                if rk > bestv:
                    bestv, best = rk, f
            reason[code] = FACTOR_CN.get(best, best or "综合得分")
        out[name] = reason
    if by is not None:
        return out.get(by, {})
    return out


def expert_top_union(full: dict[str, pd.DataFrame], w_row: pd.DataFrame,
                     pen: pd.DataFrame | None, crowd: pd.Series | None,
                     top_k: int, specs: dict[str, list[str] | None] | None = None,
                     ) -> dict[str, list[str]]:
    """每个专家独立选出各自 top_k 候选，返回 {专家名: 候选码} 并集视角。

    与 experts_score 区别：后者把分数跨专家加权成单一排名；这里保留"各专家
    独立性"，供 upper 层做候选并集/投票，避免强因子压垮小众专家的合理信号。
    """
    specs = specs or EXPERT_SPECS
    out: dict[str, list[str]] = {}
    for name, names in specs.items():
        group = [k for k in (names or list(full.keys())) if k in full]
        s = _group_score(full, w_row, pen, group)
        if s is None or s.empty:
            continue
        last = s.iloc[-1].dropna()
        if crowd is not None:
            c_row = crowd.reindex(last.index)
            keep = c_row.rank(pct=True) > (1 - CONFIG.factors.crowding_quantile)
            last = last[keep]
        if last.empty:
            continue
        out[name] = [c for c in last.sort_values(ascending=False, kind="stable").index[:top_k]]
    return out