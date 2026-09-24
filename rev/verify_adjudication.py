"""独立复核裁定书(ADJUDICATION_2026-09-24)的三条决定性新主张。

与 scripts/adjudicate_20260924.py 独立：本脚本自带探针与循环，
不复用其输出，只共享数据加载逻辑(引擎/cache 只读)。

验证目标:
  A. F01 全期化：composite_score 每次调用权重是否全 NaN、截面是否退化(唯一值=1)
  B. 恒定候选集：73 期 expert_top_union 名单是否完全相同
  C. 基准序列敏感性: bench=None vs synthetic_rising vs pool_equalweight
     -> 应分别 ~14.62% / ~15.25% / ~15.16%；synthetic_rising 精确复现存档
  D. 池宽敏感性: 39/40/41 -> 16.00% / 14.62% / 13.55%
"""
from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.data.sources import DataFeed
from quantlab.backtest.engine import Engine

OUT = ROOT / "rev" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def load_pool(limit: int = 40, drop: str | None = None):
    feed = DataFeed(source="auto")
    cand = feed.fetch_candidates("a").sort_values("float_mktcap", ascending=False)
    codes_all = [str(c) for c in cand.index]
    sel = [c for c in codes_all[: limit + (1 if drop else 0)] if c != drop][:limit]
    panels = {}
    for c in sel:
        df = feed.fetch_daily("a", c)
        if df is not None and len(df) >= 250:
            panels[c] = df
    return panels, cand.loc[list(panels)]


def ann_of(res):
    nv, dr = res["nav"], res["daily_return"]
    n = len(dr)
    return dict(annual=float(nv.iloc[-1] ** (252 / n) - 1),
                total=float(nv.iloc[-1] - 1),
                vol=float(dr.std() * np.sqrt(252)),
                maxdd=float((nv / nv.cummax() - 1).min()),
                n=n, trades=len(res["trades"]))


def run_probed(panels, meta, bench=None):
    """跑一次回测，记录: 每期 composite_score 权重 NaN/唯一值, 专家候选并集。"""
    import quantlab.strategy.scoring as SC
    import quantlab.strategy.experts as EX
    import quantlab.backtest.engine as E

    cs_stats = []
    orig_cs = SC.composite_score

    def patched_cs(factors, weights, pen):
        out = orig_cs(factors, weights, pen)
        if not out.empty:
            last = out.iloc[-1]
            cs_stats.append(dict(n_unique=int(last.nunique(dropna=True)),
                                 allzero=bool((last.abs() < 1e-12).all()),
                                 w_all_nan=bool(weights.isna().all().all())))
        return out

    basket_logs = []
    orig_bu = E.expert_top_union

    def patched_bu(full, w_row, pen, crowd, top_k, specs=None):
        o = orig_bu(full, w_row, pen, crowd, top_k, specs)
        basket_logs.append(dict(zip(o.keys(), [tuple(v) for v in o.values()])))
        return o

    SC.composite_score = patched_cs
    EX.composite_score = patched_cs
    E.expert_top_union = patched_bu
    try:
        res = Engine(market="a").run(panels, meta, bench)
    finally:
        SC.composite_score = orig_cs
        EX.composite_score = orig_cs
        E.expert_top_union = orig_bu
    return res, cs_stats, basket_logs


def main():
    log = io.StringIO()

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    t0 = time.time()
    panels, meta = load_pool(40)
    say(f"[0] pool={len(panels)} 只 (load {time.time()-t0:.1f}s)")

    # ---------------- A/B: F01 全期化 + 恒定候选集 ----------------
    say("=" * 78)
    t1 = time.time()
    res, cs_stats, bl = run_probed(panels, meta, None)
    say(f"[A/B] 基线回测 bench=None ({time.time()-t1:.1f}s): {ann_of(res)}")
    ev = {
        "score_calls": len(cs_stats),
        "w_all_nan": sum(s["w_all_nan"] for s in cs_stats),
        "unique_score_min": min(s["n_unique"] for s in cs_stats),
        "unique_score_max": max(s["n_unique"] for s in cs_stats),
        "allzero_periods": sum(s["allzero"] for s in cs_stats),
        "expert_calls": len(bl),
        "expert_identical": sum(1 for d in bl if len(set(d.values())) == 1),
        "distinct_baskets": len({tuple(sorted(d["momentum"])) for d in bl}),
        "base_annual": ann_of(res)["annual"],
    }
    say(f"    composite_score 调用 {ev['score_calls']} 次, 权重全NaN={ev['w_all_nan']}, "
        f"唯一值 min/max={ev['unique_score_min']}/{ev['unique_score_max']}, 全零={ev['allzero_periods']}")
    say(f"    expert_top_union 调用 {ev['expert_calls']} 次, 五专家全同={ev['expert_identical']}, "
        f"候选集种类={ev['distinct_baskets']}")

    # ---------------- C: 基准序列敏感性 ----------------
    say("=" * 78)
    close = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
    variants = {}
    for tag, bench in (
        ("bench_none_fallback", None),
        ("synthetic_rising", pd.DataFrame({"close": np.linspace(1, 2.5, len(close))}, index=close.index)),
        ("pool_equalweight_index", pd.DataFrame({"close": (close / close.iloc[0]).mean(axis=1)}, index=close.index)),
    ):
        t = time.time()
        r = Engine(market="a").run(panels, meta, bench)
        a = ann_of(r)
        gs = [x["risk"]["gross_scale"] for x in r["trades"]]
        gross03 = sum(abs(g - 0.3) < 1e-9 for g in gs)
        variants[tag] = dict(**a, gross03=gross03)
        say(f"  {tag:<24} 年化={a['annual']:8.4%} 回撤={a['maxdd']:8.4%} gross0.3={gross03} ({time.time()-t:.1f}s)")
    d = np.load(pickle_path := ROOT / "output" / "results.pkl", allow_pickle=True) if False else None
    import pickle
    saved = pickle.load(open(ROOT / "output" / "results.pkl", "rb"))["result"]
    sa = ann_of({"nav": saved["nav"], "daily_return": saved["daily_return"], "trades": saved["trades"]})
    say(f"  存档 results.pkl                      年化={sa['annual']:8.4%} 回撤={sa['maxdd']:8.4%}")
    ev["variants"] = variants
    ev["saved"] = sa

    # ---------------- D: 池宽敏感性 ----------------
    say("=" * 78)
    frag = {}
    base_basket = ev["distinct_baskets"]
    for tag, kw in (("width_39", dict(limit=39)), ("width_41", dict(limit=41)),
                    ("drop_one_add_41", dict(limit=40, drop=list(panels)[0]))):
        t = time.time()
        p2, m2 = load_pool(**kw)
        r2, cs2, bl2 = run_probed(p2, m2, None)
        b2 = {tuple(sorted(d["momentum"])) for d in bl2}
        frag[tag] = dict(annual=ann_of(r2)["annual"], baskets=len(b2),
                         w_all_nan=sum(s["w_all_nan"] for s in cs2))
        say(f"  {tag:<18} 年化={frag[tag]['annual']:8.4%} 候选集种类={len(b2)} "
            f"权重NaN={frag[tag]['w_all_nan']}/{len(cs2)} ({time.time()-t:.1f}s)")
    ev["fragility"] = frag

    json.dump(ev, open(OUT / "verify_adjudication.json", "w"), ensure_ascii=False, indent=1)
    open(OUT / "verify_adjudication.log", "w").write(log.getvalue())
    say(f"已写入 {OUT/'verify_adjudication.json'}  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()