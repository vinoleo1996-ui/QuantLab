"""第二轮实验:池子偏差 vs 策略负alpha。

- top23 等权基线(与策略持仓数更可比)
- 策略 vs 池基线的超额(月换手/费用对比)
- 分时段(2020-2022/2022-2024/2024-2026)策略与基线的年化
- use_experts=False 变体
- 升级: 池子内今日市值 TOP10/中/小 对总收益的贡献(确认幸存者集中度)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.backtest.engine import Engine, limit_masks
from quantlab.report.metrics import performance_table

from exp1_reproduce import build_panels_local, pool_baseline  # noqa: E402


def pool23_baseline(panels, first_exec: pd.Timestamp, n: int = 23) -> pd.Series:
    """今日流通市值前 n 名等权, 月度调仓(与 pool_baseline 同口径)。"""
    from quantlab.data.sources import DataFeed
    cand = DataFeed(source="auto").fetch_candidates("a")
    cand = cand.sort_values("float_mktcap", ascending=False)
    top = [str(c) for c in cand.index[:n]]
    sub = {c: p for c, p in panels.items() if c in top}
    return pool_baseline(sub, None)


def subperiod_ann(daily: pd.Series, periods) -> dict:
    out = {}
    for name, (a, b) in periods.items():
        seg = daily[(daily.index >= a) & (daily.index < b)]
        if len(seg) < 30:
            out[name] = float("nan")
            continue
        nav = (1 + seg).cumprod()
        out[name] = float(nav.iloc[-1] / nav.iloc[0] - 1), len(seg)
    return out


def main() -> None:
    panels, meta, bench = build_panels_local()
    print(f"面板 {len(panels)} 只\n")

    # --- 策略(复现) ---
    eng = Engine(market="a")
    res = eng.run(panels, meta, bench)
    strat = res["daily_return"]

    # --- 池等权(全池40) ---
    bl = pool_baseline(panels, bench)

    # --- top23 池等权 ---
    bl23 = pool23_baseline(panels, res["nav"].index[0], 23)

    def fmt(name, daily):
        p = performance_table((1 + daily).cumprod(), daily)
        return (f"{name:<16} 年化={p['annual_return']*100:6.2f}%  波动={p['annual_vol']*100:5.2f}%  "
                f"夏普={p['sharpe']:4.2f}  回撤={p['max_drawdown']*100:6.2f}%  总={p['total_return']*100:7.2f}%")

    print(fmt("策略(因子专家)", strat))
    print(fmt("池等权40(基线)", bl))
    print(fmt("今日市值TOP23等权", bl23))
    print()

    # 超额: 策略 - 池
    idx = strat.index.intersection(bl.index)
    ex = (1 + strat.reindex(idx)).cumprod() / (1 + bl.reindex(idx)).cumprod() - 1
    print("策略 vs 池基线 累计超额(扣除池子本身的涨跌):", f"{ex.iloc[-1]*100:.2f}%")
    ex_ann = float(ex.iloc[-1]) ** (1 / (len(idx) / 252)) - 1
    print("策略 vs 池基线 年化超额:", f"{ex_ann*100:.2f}%")
    print()

    # 分时段
    periods = {
        "2020-2022": ("2020-10-01", "2022-01-01"),
        "2022-2024": ("2022-01-01", "2024-01-01"),
        "2024-2026": ("2024-01-01", "2026-10-01"),
    }
    print("=== 分时段粗收益(区间内简单收益%) ===")
    for nm, (a, b) in periods.items():
        s = strat[(strat.index >= a) & (strat.index < b)]
        p = bl[(bl.index >= a) & (bl.index < b)]
        n_s = float((1 + s).prod() - 1) * 100 if len(s) else float("nan")
        n_p = float((1 + p).prod() - 1) * 100 if len(p) else float("nan")
        print(f"{nm:<10} 策略={n_s:7.2f}%  池40={n_p:7.2f}%  (策略-池): {n_s-n_p:7.2f}pp")

    # use_experts=False
    cfg = CONFIG
    old = cfg.factors.use_experts
    cfg.factors.use_experts = False
    try:
        res2 = Engine(market="a").run(panels, meta, bench)
        p2 = performance_table(res2["nav"], res2["daily_return"])
        print(f"\nuse_experts=False: 年化={p2['annual_return']*100:.2f}% 夏普={p2['sharpe']:.2f} "
              f"回撤={p2['max_drawdown']*100:.2f}%")
    finally:
        cfg.factors.use_experts = old

    # 今日前10市值占池收益贡献(用市值权重简单近似)
    from quantlab.data.sources import DataFeed
    cand = DataFeed(source="auto").fetch_candidates("a")
    cand = cand.sort_values("float_mktcap", ascending=False).head(40)
    close = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
    c10 = close[[str(c) for c in cand.index[:10]]].pct_change().mean(axis=1).fillna(0)
    c10_ann = performance_table((1 + c10).cumprod(), c10)
    print(f"\n仅今日市值TOP10(等权)年化: {c10_ann['annual_return']*100:.2f}%  Sharpe={c10_ann['sharpe']:.2f}")
    c40 = close.pct_change().mean(axis=1).fillna(0)
    c40_ann = performance_table((1 + c40).cumprod(), c40)
    print(f"全池40(日度等权, 无遮蔽)年化: {c40_ann['annual_return']*100:.2f}%  Sharpe={c40_ann['sharpe']:.2f}")


if __name__ == "__main__":
    main()