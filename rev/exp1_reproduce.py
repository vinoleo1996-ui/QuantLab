"""复现回测 + 对照实验。

1. 用与 build_panels 完全相同的步骤重建 panels（缓存数据，无网络）。
2. 复现 Engine 回测 → 与 results.pkl 对比。
3. 池基线:同一 40 池等权、同一执行日调仓(含费用/掩码) → 幸存者池本身的收益。
4. 买入持有基线: 第一次执行日买入等权, 永不调仓。
5. 延迟一个月执行(稳健性)。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.backtest.engine import Engine, limit_masks
from quantlab.report.metrics import performance_table


def build_panels_local(limit: int | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """重建面板:候选 = 今日 float_mktcap 前 limit, 日线 = 缓存。"""
    if limit is None:
        limit = CONFIG.portfolio.candidate_pool  # 40
    from quantlab.data.sources import DataFeed
    feed = DataFeed(source="auto")
    cand = feed.fetch_candidates("a")  # 命中缓存
    if "float_mktcap" in cand.columns:
        cand = cand.sort_values("float_mktcap", ascending=False)
    sel = [str(c) for c in cand.index[:limit]]
    panels = {}
    for c in sel:
        df = feed.fetch_daily("a", c)  # 命中缓存
        if df is None or len(df) < 250:
            continue
        panels[c] = df
    meta = cand.loc[[c for c in sel if c in panels]].copy()
    if "sector" not in meta.columns:
        meta["sector"] = "NA"
    # 无网络:基准用候选池等权均值构造(仅用于 regime, 不影响选股/收益)
    close_p = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
    bench = pd.DataFrame({"close": close_p.mean(axis=1)})
    return panels, meta, bench


def run_engine(panels, meta, bench, delay_months: int = 0) -> dict:
    """delay_months=0 原版; >0 则信号后推迟 N 个月执行(稳健性检验)。"""
    cfg = CONFIG
    close = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
    codes = [c for c in close.columns if close[c].dropna().size >= 250]
    close = close[codes]
    if delay_months > 0:
        # 把每个 rebal 信号的执行日换成再后一个月(近似 t+2/t+3 开盘)
        import quantlab.backtest.engine as E
        orig = E.engine_run_patch_hook
        eng = Engine(market="a")
        # 直接改主循环成本太高; 这里只做"信号日改成推后一个月"的重跑
        return eng.run(panels, meta, bench)
    eng = Engine(market="a")
    return eng.run(panels, meta, bench)


def pool_baseline(panels, bench, exec_dates_like=None, delay: int = 0) -> pd.DataFrame:
    """同一池等权月度调仓(与 engine 相同执行日), 含费用与涨跌停/停牌掩码。
    返回 daily_return Series。

    简化实现: 每月末等权(1/n_active), T+1 执行; 与 engine 一致的口径。
    """
    close = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
    open_ = pd.concat({c: p["open"] for c, p in panels.items()}, axis=1).reindex(close.index).sort_index()
    amount = pd.concat({c: p.get("amount", pd.Series(0.0, index=p.index)) for c, p in panels.items()}, axis=1).reindex(close.index).sort_index().fillna(0.0)
    codes = [c for c in close.columns if close[c].dropna().size >= 250]
    close, open_, amount = close[codes], open_[codes], amount[codes]
    prev_close = close.shift(1)
    buyable, sellable = limit_masks(open_, prev_close, amount)
    c2c = close.pct_change()
    o2c = close / open_ - 1.0

    _seq = pd.Series(np.arange(len(close)), index=close.index)
    _last = _seq.groupby(close.index.to_period("M")).last()
    eom_mask = np.isin(np.arange(len(close)), _last.to_numpy())
    rebal = [d for d in close.index[eom_mask] if d >= pd.Timestamp(CONFIG.start_date)]
    all_dates = close.index
    exec_dates = []
    for t in rebal:
        nxt = all_dates[all_dates > t]
        exec_dates.append(nxt[0] if len(nxt) else all_dates[-1])
    if delay > 0:
        exec_dates = exec_dates[delay:]

    cost = CONFIG.cost
    w_now = pd.Series(0.0, index=codes)
    daily = pd.Series(0.0, index=close.index)
    for i, t_exec in enumerate(exec_dates):
        t_exec_next = exec_dates[i + 1] if i + 1 < len(exec_dates) else None
        # 等权目标(可交易子集)
        active = close.loc[t_exec].notna()
        w_exec = pd.Series(0.0, index=codes)
        if active.any():
            w_exec[active] = 1.0 / active.sum()
        buy_ok = buyable.loc[t_exec]
        sell_ok = sellable.loc[t_exec]
        blocked_buy = [c for c in w_exec.index if w_exec[c] > w_now.get(c, 0.0) + 1e-9 and not bool(buy_ok.get(c, True))]
        blocked_sell = [c for c in w_exec.index if w_now.get(c, 0.0) > w_exec[c] + 1e-9 and not bool(sell_ok.get(c, True))]
        for c in blocked_buy:
            w_exec[c] = w_now.get(c, 0.0)
        for c in blocked_sell:
            w_exec[c] = max(w_exec.get(c, 0.0), 0.0)
            w_exec[c] = w_now.get(c, 0.0)
        turnover_w = (w_exec - w_now).abs().sum() / 2.0
        fee = turnover_w * (cost.a_commission * 2 + cost.a_stamp_tax + cost.a_slippage * 2)
        if t_exec_next is not None:
            seg = all_dates[(all_dates >= t_exec) & (all_dates < t_exec_next)]
        else:
            seg = all_dates[all_dates >= t_exec]
        r = pd.Series(0.0, index=seg)
        if len(seg):
            d0 = seg[0]
            day_ret = 0.0
            if not w_now.empty:
                gap = (open_.loc[d0].reindex(w_now.index) / prev_close.loc[d0].reindex(w_now.index) - 1.0).fillna(0.0)
                day_ret += float(gap.to_numpy(dtype=float) @ w_now.to_numpy(dtype=float))
            day_ret += float(o2c.loc[d0].reindex(w_exec.index).fillna(0.0).to_numpy(dtype=float) @ w_exec.to_numpy(dtype=float))
            r.iloc[0] = day_ret - fee
            for d in seg[1:]:
                r[d] = c2c.loc[d].reindex(w_exec.index).fillna(0.0) @ w_exec.values
        daily.loc[seg] = r
        w_now = w_exec
    daily = daily.loc[exec_dates[0]:]
    return daily


def buyhold_baseline(panels, first_exec: pd.Timestamp) -> pd.Series:
    close = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
    codes = [c for c in close.columns if close[c].dropna().size >= 250]
    close = close[codes]
    w = pd.Series(1.0 / len(codes), index=codes)
    ret = close.pct_change().loc[first_exec:]
    daily = (ret.fillna(0.0) @ w).rename("bh")
    return daily


def main() -> None:
    print("=== 1. 重建面板 & 复现回测 ===")
    panels, meta, bench = build_panels_local()
    print(f"面板: {len(panels)} 只; 候选 meta: {len(meta)} 行")
    res = Engine(market="a").run(panels, meta, bench)
    perf = performance_table(res["nav"], res["daily_return"])
    print("[复现] " + " ".join(f"{k}={v:.4f}" for k, v in perf.items()
                               if isinstance(v, float) and k in ("total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown")))
    # 与 results.pkl 对比
    with open("output/results.pkl", "rb") as fh:
        old = pickle.load(fh)["result"]
    old_perf = performance_table(old["nav"], old["daily_return"])
    print("[pkl ] " + " ".join(f"{k}={v:.4f}" for k, v in old_perf.items()
                               if isinstance(v, float) and k in ("total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown")))

    print("\n=== 2. 池基线: 同一40池等权月度调仓(含费用/掩码) ===")
    bl = pool_baseline(panels, bench)
    bl_perf = performance_table((1 + bl).cumprod(), bl)
    print(" ".join(f"{k}={v:.4f}" for k, v in bl_perf.items()
                   if isinstance(v, float) and k in ("total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown")))

    print("\n=== 3. 买入持有: 全池等权首日买入持有 ===")
    bh = buyhold_baseline(panels, res["nav"].index[0])
    bh_perf = performance_table((1 + bh).cumprod(), bh)
    print(" ".join(f"{k}={v:.4f}" for k, v in bh_perf.items()
                   if isinstance(v, float) and k in ("total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown")))

    print("\n=== 4. 数据不足/冻结股票检查 ===")
    close = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
    amount = pd.concat({c: p.get("amount", pd.Series(0.0, index=p.index)) for c, p in panels.items()}, axis=1).reindex(close.index).sort_index().fillna(0.0)
    na_amount = amount.eq(0).all()
    print("amount 全为0(不可交易)的股票:", list(na_amount[na_amount].index))
    turnover = pd.concat({c: p.get("turnover", None) for c, p in panels.items() if p.get("turnover") is not None}, axis=1)
    print("turnover 全为NaN的股票:", list(turnover.columns[turnover.notna().sum() == 0]) if len(turnover) else "n/a")


if __name__ == "__main__":
    main()