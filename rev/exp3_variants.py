"""第三轮:隔离执行日收益处理 / 止损 / 池子宽度的影响。

1. 原版 vs 执行日改按 w_exec 的 c2c(收盘执行, 无跳空/o2c 处理) → 隔离"开盘执行+隔夜跳空"贡献
2. 原版 vs 关掉 L1 单票止损
3. 用全部 53 只缓存股票当池(而非今日市值 TOP40) → 池宽度敏感性, 并重跑池基线
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.backtest.engine import Engine, limit_masks, close_ret_only
from quantlab.report.metrics import performance_table

from exp1_reproduce import build_panels_local, pool_baseline


def engine_variant(panels, meta, bench, mode: str) -> dict:
    """mode: 'orig' | 'close_exec' | 'no_stoploss' | 'close_exec_no_stoploss'. """
    cfg = CONFIG
    fc = cfg.factors
    pc = cfg.portfolio
    close = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
    open_ = pd.concat({c: p["open"] for c, p in panels.items()}, axis=1).reindex(close.index).sort_index()
    amount = pd.concat({c: p.get("amount", pd.Series(0.0, index=p.index)) for c, p in panels.items()}, axis=1).reindex(close.index).sort_index().fillna(0.0)
    turnover = pd.concat({c: p.get("turnover", pd.Series(0.0, index=p.index)) for c, p in panels.items()}, axis=1).reindex(close.index).sort_index().fillna(0.0)
    close = close.dropna(how="all")
    open_ = open_.reindex(close.index)
    amount = amount.reindex(close.index)
    turnover = turnover.reindex(close.index)
    codes = [c for c in close.columns if close[c].dropna().size >= 250]
    close, open_, amount, turnover = close[codes], open_[codes], amount[codes], turnover[codes]
    prev_close = close.shift(1)
    buyable, sellable = limit_masks(open_, prev_close, amount)
    c2c = close.pct_change()
    o2c = close / open_ - 1.0

    from quantlab.factors.engine import FactorEngine
    eng = FactorEngine("a", meta)
    factor_panels = {k: v.reindex(close.index).sort_index() for k, v in eng.compute(close, amount, turnover).items()}
    fnames = list(factor_panels.keys())

    _seq = pd.Series(np.arange(len(close)), index=close.index)
    _last = _seq.groupby(close.index.to_period("M")).last()
    eom_mask = np.isin(np.arange(len(close)), _last.to_numpy())
    rebal = [d for d in close.index[eom_mask] if d >= pd.Timestamp(cfg.start_date)]

    from quantlab.strategy.scoring import adaptive_weight, decay_penalty, factor_crowding_penalty, composite_score
    from quantlab.strategy.experts import expert_top_union
    from quantlab.strategy.risk import Regime, RiskManager

    ic_matrix = pd.DataFrame(np.nan, index=rebal, columns=fnames)
    for i, t in enumerate(rebal):
        if i == 0:
            continue
        t_prev = rebal[i - 1]
        fwd = close.loc[t] / close.loc[t_prev] - 1.0
        for name in fnames:
            fp = factor_panels[name]
            if t_prev not in fp.index:
                continue
            s = fp.loc[t_prev].dropna()
            if s.size < 10:
                continue
            y = fwd.reindex(s.index).dropna()
            s = s.reindex(y.index)
            if s.size < 10:
                continue
            from scipy.stats import spearmanr
            rho, _ = spearmanr(s.to_numpy(), y.to_numpy())
            if np.isfinite(rho):
                ic_matrix.loc[t, name] = rho

    risk = RiskManager(cfg.risk)
    bench_close = bench["close"] if bench is not None and "close" in bench.columns else close.mean(axis=1)
    regime = Regime(bench_close, cfg.risk)
    w_now = pd.Series(0.0, index=codes)
    entry_date: dict[str, pd.Timestamp] = {}
    daily_return = pd.Series(0.0, index=close.index)
    all_dates = close.index
    exec_dates: list[pd.Timestamp] = []
    for t in rebal:
        nxt = all_dates[all_dates > t]
        exec_dates.append(nxt[0] if len(nxt) else all_dates[-1])
    stop_loss = cfg.risk.stop_loss_single
    n_pos = cfg.capital.a_share_positions

    for i, t in enumerate(rebal):
        t_exec = exec_dates[i]
        t_exec_next = exec_dates[i + 1] if i + 1 < len(exec_dates) else None
        ic_t = ic_matrix.loc[:t]
        w_fac = pd.Series(1.0, index=fnames, dtype=float)
        for name in fnames:
            ics = ic_t[name].dropna()
            if ics.size < 24:
                continue
            roll = ics.rolling(60, min_periods=24)
            ir = roll.mean() / (roll.std() + 1e-9)
            if ir.empty or np.isnan(ir.iloc[-1]):
                continue
            w_fac[name] = adaptive_weight(ir.iloc[-1], fc.icir_floor, fc.icir_cap) * decay_penalty(ics, fc.factor_decay_lookback)
        ic_use = ic_t[fnames].fillna(0.0)
        pen, _ = factor_crowding_penalty(ic_use, fc.crowding_corr_window, fc.crowding_corr_alert)
        w_row = w_fac.to_frame("w").T.reindex([t])
        full = {k: v.loc[:t] for k, v in factor_panels.items() if not v.empty}
        if not full:
            continue
        from quantlab.strategy.experts import experts_score
        score, exp_w = experts_score(full, w_row, pen, ic_use)
        if score is None or score.empty:
            continue
        last_score = score.iloc[-1]
        crowd = factor_panels.get("factor_price_crowding")
        crowd_row = crowd.loc[last_score.name] if crowd is not None and last_score.name in crowd.index else None
        to_keep_pad = pc.n_candidates_pad + int(getattr(pc, "turnover_rank_thresh", 0))
        exp_tops = expert_top_union(full, w_row, pen, crowd_row, n_pos + to_keep_pad)
        top = sorted({c for lst in exp_tops.values() for c in lst})
        if not top:
            continue
        if len(top) > n_pos + to_keep_pad:
            top = select_top(last_score.reindex(top).dropna(), n_pos, to_keep_pad)  # noqa: F821
        if len(top) < 3:
            continue
        past = close_ret_only(c2c, top, t, pc.vol_est_window)
        if past is None or past.shape[0] < 40:
            continue
        sector = pd.Series("NA", index=top)
        r_state = regime.state(t)
        r_eval = risk.evaluate(float((1 + daily_return.fillna(0.0)).prod()), r_state["multiplier"])
        scale = r_eval["gross_scale"]
        vol_mult = r_eval["vol_scale"]
        from quantlab.strategy.portfolio import target_vol_weights
        w_t = target_vol_weights(past[sector.index], sector, pc, vol_mult=vol_mult)
        w_target = w_t["weight"].clip(lower=0.0).reindex(top).fillna(0.0)
        if w_target.sum() <= 0:
            continue
        w_exec = w_target * scale
        if t_exec in buyable.index:
            buy_ok = buyable.loc[t_exec]
            sell_ok = sellable.loc[t_exec]
            blocked_buy = [c for c in w_exec.index if w_exec[c] > w_now.get(c, 0.0) + 1e-9 and not bool(buy_ok.get(c, True))]
            blocked_sell = [c for c in w_exec.index if w_now.get(c, 0.0) > w_exec[c] + 1e-9 and not bool(sell_ok.get(c, True))]
            for c in blocked_buy:
                w_exec[c] = w_now.get(c, 0.0)
            for c in blocked_sell:
                w_exec[c] = w_now.get(c, 0.0)
            stop_reasons = []
            if mode in ("orig", "close_exec"):
                for c in list(w_exec.index):
                    if w_exec.get(c, 0.0) <= 1e-9 or c not in entry_date:
                        continue
                    e_d = entry_date[c]
                    if e_d not in close.index:
                        continue
                    px_entry = float(close.loc[e_d, c])
                    px_now = float(close.loc[t, c])
                    if px_now <= 0 or px_entry <= 0:
                        continue
                    if px_now / px_entry - 1.0 < stop_loss:
                        w_exec[c] = 0.0
            min_hold = cfg.capital.min_holding_period_days
            if scale >= 1.0 - 1e-9 and min_hold > 0:
                for c in list(w_exec.index):
                    if w_exec.get(c, 0.0) < w_now.get(c, 0.0) - 1e-9 and c in entry_date:
                        if (t_exec - entry_date[c]).days < min_hold:
                            w_exec[c] = w_now.get(c, 0.0)
            turnover_w = (w_exec - w_now).abs().sum() / 2.0
            fee = turnover_w * (cfg.cost.a_commission * 2 + cfg.cost.a_stamp_tax + cfg.cost.a_slippage * 2)
        else:
            fee = 0.0
        if t_exec_next is not None:
            seg_dates = all_dates[(all_dates >= t_exec) & (all_dates < t_exec_next)]
        else:
            seg_dates = all_dates[all_dates >= t_exec]
        r_seg = pd.Series(0.0, index=seg_dates)
        if len(seg_dates):
            d0 = seg_dates[0]
            if mode == "orig":
                day_ret = 0.0
                if not w_now.empty:
                    gap = (open_.loc[d0].reindex(w_now.index) / prev_close.loc[d0].reindex(w_now.index) - 1.0).fillna(0.0)
                    day_ret += float(gap.to_numpy(dtype=float) @ w_now.to_numpy(dtype=float))
                day_ret += float(o2c.loc[d0].reindex(w_exec.index).fillna(0.0).to_numpy(dtype=float) @ w_exec.to_numpy(dtype=float))
                r_seg.iloc[0] = day_ret - fee
            else:  # close_exec: 执行日在收盘切换
                r_seg.iloc[0] = float(c2c.loc[d0].reindex(w_exec.index).fillna(0.0).to_numpy(dtype=float) @ w_exec.to_numpy(dtype=float)) - fee
            for d in seg_dates[1:]:
                r_seg[d] = c2c.loc[d].reindex(w_exec.index).fillna(0.0) @ w_exec.values
        daily_return.loc[seg_dates] = r_seg
        for c in list(entry_date):
            if w_exec.get(c, 0.0) <= 1e-9:
                entry_date.pop(c, None)
        for c in w_exec.index:
            if w_exec[c] > 1e-9 and w_now.get(c, 0.0) <= 1e-9:
                entry_date[c] = t_exec
        w_now = w_exec

    if exec_dates:
        daily_return = daily_return.loc[exec_dates[0]:]
    nav = (1 + daily_return.fillna(0.0)).cumprod()
    return {"nav": nav, "daily_return": daily_return}


def select_top(scored, n, extra=0):
    s = scored.dropna().sort_values(ascending=False)
    return list(s.index[:n + extra])


def main() -> None:
    panels, meta, bench = build_panels_local()
    print(f"面板 {len(panels)} 只\n")

    for mode, label in [("orig", "原版(开盘执行)"), ("close_exec", "收盘执行(无跳空)")]:
        res = engine_variant(panels, meta, bench, mode)
        p = performance_table(res["nav"], res["daily_return"])
        print(f"{label:<18} 年化={p['annual_return']*100:6.2f}%  夏普={p['sharpe']:4.2f}  回撤={p['max_drawdown']*100:6.2f}%  总收益={p['total_return']*100:7.2f}%")

    print("\n=== 池子宽度: 全部53只 ===")
    from quantlab.data.sources import DataFeed
    cand = DataFeed(source="auto").fetch_candidates("a")
    cand = cand.sort_values("float_mktcap", ascending=False)
    sel = [str(c) for c in cand.index[:200]]
    panels_all = {}
    for c in sel:
        df = panels.get(c)
        if df is not None:
            panels_all[c] = df
    print("扩展池实际股票数:", len(panels_all))
    res = Engine(market="a").run(panels_all, meta, bench)
    p = performance_table(res["nav"], res["daily_return"])
    print(f"策略(53池)      年化={p['annual_return']*100:6.2f}%  夏普={p['sharpe']:4.2f}  回撤={p['max_drawdown']*100:6.2f}%")
    bl = pool_baseline(panels_all, bench)
    pb = performance_table((1 + bl).cumprod(), bl)
    print(f"池53等权基线    年化={pb['annual_return']*100:6.2f}%  夏普={pb['sharpe']:4.2f}  回撤={pb['max_drawdown']*100:6.2f}%")


if __name__ == "__main__":
    main()