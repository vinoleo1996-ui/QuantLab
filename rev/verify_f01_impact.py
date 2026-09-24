"""独立三方验证:F01 因子索引修复对回测的影响。

用自己的本地管道(缓存重建因子面板, 不复用同事 saved 因子):
- 原版引擎: w_row = w_fac.to_frame("w").T.reindex([t])   (生产代码)
- 修复版引擎: w_row = pd.DataFrame([w_fac.to_dict()], index=[t])
对比年化/回撤。与同事 replay 的 factor_index_only 方向应一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.report.metrics import performance_table
from exp1_reproduce import build_panels_local


def run_variant(label: str, fix_index: bool) -> dict:
    import quantlab.backtest.engine as E

    if fix_index:
        # 提取整个 Engine 类源码, 替换 w_row 行, 重新编译 → 真正生效
        import inspect
        import textwrap

        src = inspect.getsource(E.Engine)
        src = textwrap.dedent(src)
        assert src.count('w_row = w_fac.to_frame("w").T.reindex([t])') == 1
        src = src.replace(
            'w_row = w_fac.to_frame("w").T.reindex([t])',
            'w_row = pd.DataFrame([w_fac.to_dict()], index=[t])',
        )
        ns = dict(E.__dict__)
        exec(compile(src, "<fixed-engine>", "exec"), ns)
        EngFix = ns["Engine"]
    else:
        EngFix = E.Engine

    panels, meta, bench = build_panels_local()
    eng = EngFix(market="a")
    res = eng.run(panels, meta, bench)
    tbl = performance_table(res["nav"], res["daily_return"])
    n_same = sum(1 for t in res["trades"] if t["signal_date"] == t["exec_date"])
    return {
        "label": label,
        "annual": tbl["annual_return"],
        "sharpe": tbl["sharpe"],
        "max_dd": tbl["max_drawdown"],
        "trades": len(res["trades"]),
        "same_day": n_same,
    }


if __name__ == "__main__":
    a = run_variant("原版(生产代码)", fix_index=False)
    b = run_variant("仅修因子索引(F01)", fix_index=True)
    print("=== 独立验证:F01 修复影响 ===")
    for r in (a, b):
        print(f"{r['label']:24s} 年化 {r['annual']:.2%}  Sharpe {r['sharpe']:.2f}  "
              f"MaxDD {r['max_dd']:.2%}  交易 {r['trades']}  末日同日 {r['same_day']}")
    print(f"\n差值(修复-原版): {b['annual'] - a['annual']:+.2%}")