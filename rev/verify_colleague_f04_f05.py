#!/usr/bin/env python3
"""独立验证同事 review 的 F04(权重漂移/隐含每日再平衡)与 F05(缺失行情吞收益)。

只读缓存 + 在内存里构建确定性反例,不修改任何生产代码/输出。
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import quantlab.backtest.engine as em
from quantlab.config import Config


def mk_cfg(n_pos=3, fees=True):
    c = Config()
    c.start_date = "2023-01-01"
    c.factors.use_experts = False
    c.factors.neutralize_by_sector = False
    c.capital.a_share_positions = n_pos
    c.capital.min_holding_period_days = 0
    c.portfolio.n_candidates_pad = 0
    c.portfolio.turnover_rank_thresh = 0
    if not fees:
        c.cost.a_commission = c.cost.a_stamp_tax = c.cost.a_slippage = 0.0
    return c


def run_case(mode):
    idx = pd.bdate_range("2022-01-03", "2023-04-14")
    codes = [f"60000{i}" for i in range(6)]
    panels = {c: pd.DataFrame({"open": 10.0, "close": 10.0,
                               "amount": 1e8, "turnover": 1.0}, index=idx)
              for c in codes}
    if mode == "drift":
        # F04: 3 只等权, 1 只 10 -> 11 -> 10, 其余不动, 不调仓
        panels[codes[0]].loc["2023-02-02", ["open", "close"]] = 11.0
    elif mode == "missing":
        # F05: 持仓 10 -> 缺失K线 -> 恢复 8, 三分之一仓位
        panels[codes[0]].loc["2023-02-02", ["open", "close"]] = np.nan
        panels[codes[0]].loc["2023-02-02", "amount"] = 0.0
        panels[codes[0]].loc["2023-02-03":, ["open", "close"]] = 8.0

    idx_ = pd.bdate_range("2022-01-03", "2023-04-14")
    scores = pd.DataFrame(np.tile([6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                                  (len(idx_), 1)), index=idx_, columns=codes)

    def weights(ret, sector, cfg, vol_mult=1.0):
        w = np.ones(len(ret.columns)) / len(ret.columns)
        return pd.DataFrame({"weight": w}, index=ret.columns)

    cfg = mk_cfg(fees=(mode != "drift"))
    with patch.object(em.FactorEngine, "compute", return_value={"fixture": scores}), \
         patch.object(em, "composite_score",
                      side_effect=lambda full, *a: scores.loc[full["fixture"].index]), \
         patch.object(em, "target_vol_weights", side_effect=weights), \
         patch.object(em.Regime, "state", return_value={"trend": 1, "vol_high": False, "multiplier": 1.0}), \
         patch.object(em.RiskManager, "evaluate",
                      return_value={"gross_scale": 1.0, "vol_scale": 1.0, "drawdown": 0.0, "reasons": []}):
        return em.Engine(cfg).run(panels)


print("=== F04: 权重漂移(隐含每日再平衡)===")
r = run_case("drift")
seg = r["daily_return"].loc["2023-02-01":"2023-02-03"]
ret = float((1 + seg).prod() - 1)
print("路径: 3 只等权, 1 只 10->11->10, 其余持平, 期间不调仓")
print(f"引擎记录收益: {ret:+.4%}   理论买入持有: 0.0000%")
print("结论:", "F04 复现(引擎 ≠ 买入持有)" if ret > 2e-4 else "未复现")

print()
print("=== F05: 缺失行情后复牌收益 ===")
r = run_case("missing")
d = r["daily_return"].loc["2023-02-03"]
true = (8 / 10 - 1) / 3
print(f"复牌日(2023-02-03)引擎收益: {float(d):+.4%}   理论(1/3仓位 10->8): {true:+.4%}")
print("结论:", "F05 复现(引擎把缺失视为0)" if abs(float(d)) < 1e-9 else "未复现")