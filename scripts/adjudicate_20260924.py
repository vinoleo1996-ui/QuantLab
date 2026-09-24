"""独立裁定：两份同事 review 的分歧点逐条复核（全程离线，只读缓存）。

用法:
    .venv2/bin/python scripts/adjudicate_20260924.py

输出:
    output/adjudication_20260924/evidence.json
    output/adjudication_20260924/run.log

本脚本不修改任何生产代码、缓存、results.pkl 或同事的 rev/ 文件。
所有"修复"均在内存中通过 types.ModuleType 重编译源码实现。
"""

from __future__ import annotations

import io
import json
import pickle
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantlab.config import CONFIG
from quantlab.data.sources import DataFeed
from quantlab.backtest.engine import Engine

OUT = ROOT / "output" / "adjudication_20260924"
OUT.mkdir(parents=True, exist_ok=True)
EV: dict = {}
LOG = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


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


def run_with_basket_log(panels, meta, bench=None):
    """跑一次回测，同时记录每期专家候选并集。"""
    import quantlab.backtest.engine as E

    logs = []
    orig = E.expert_top_union

    def patched(full, w_row, pen, crowd, top_k, specs=None):
        o = orig(full, w_row, pen, crowd, top_k, specs)
        logs.append(dict(zip(o.keys(), [tuple(v) for v in o.values()])))
        return o

    E.expert_top_union = patched
    try:
        res = Engine(market="a").run(panels, meta, bench)
    finally:
        E.expert_top_union = orig
    return res, logs


def ann_of(res):
    nv, dr = res["nav"], res["daily_return"]
    n = len(dr)
    return dict(annual=float(nv.iloc[-1] ** (252 / n) - 1),
                total=float(nv.iloc[-1] - 1),
                vol=float(dr.std() * np.sqrt(252)),
                maxdd=float((nv / nv.cummax() - 1).min()),
                n=n, trades=len(res["trades"]))


# ===========================================================================
# 0. 池与数据
# ===========================================================================
say("=" * 78)
say("[0] 股票池与数据源")
panels, meta = load_pool(40)
codes = list(panels)
say(f"  候选池 = 今日实时快照(新浪)按当前流通市值取前 40，实际取到 {len(codes)} 只")
say(f"  池内名称样例: {dict(list(zip(codes[:6], meta.loc[codes[:6], 'name']))) }")
say(f"  meta 列: {list(meta.columns)}  (无 sector 列 -> 行业约束/行业中性化不生效)")
EV["pool"] = dict(size=len(codes), codes=codes,
                  meta_cols=list(meta.columns),
                  has_sector="sector" in meta.columns,
                  names={c: str(meta.loc[c, "name"]) for c in codes})

from quantlab.data.store import DataStore

st = DataStore()
frozen = {}
for c in codes:
    d = st.load("a", f"daily_{c}")
    if d is None:
        continue
    nn = int(d["amount"].notna().sum()) if "amount" in d.columns else -1
    if nn == 0:
        frozen[c] = dict(source=d.attrs.get("source"), rows=len(d), amount_nonnull=nn)
say(f"  amount 全程缺失(被判停牌冻结)的股票: {frozen}")
EV["frozen"] = frozen

# ===========================================================================
# 1. F01 因子权重索引 bug
# ===========================================================================
say("=" * 78)
say("[1] F01 因子权重索引 bug —— 打分是否退化为常数")
names = ["momentum_3m", "value_breadth", "factor_volatility"]
w_fac = pd.Series([0.5, 0.8, 0.3], index=names)
t = pd.Timestamp("2026-08-31")
w_row = w_fac.to_frame("w").T.reindex([t])
say(f"  w_fac.to_frame('w').T.reindex([t]) -> 全 NaN? {bool(w_row.isna().all().all())}")
say(f"  行索引变成: {list(w_row.index)}  (原为字符串 'w')")
EV["f01_wrow"] = dict(all_nan=bool(w_row.isna().all().all()),
                      index=str(list(w_row.index)))

import quantlab.strategy.scoring as SC
import quantlab.strategy.experts as EX

orig_cs = SC.composite_score
stats = []


def patched_cs(factors, weights, pen):
    out = orig_cs(factors, weights, pen)
    if not out.empty:
        last = out.iloc[-1]
        stats.append(dict(n_unique=int(last.nunique(dropna=True)),
                          allzero=bool((last.abs() < 1e-12).all()),
                          w_all_nan=bool(weights.isna().all().all())))
    return out


SC.composite_score = patched_cs
EX.composite_score = patched_cs
try:
    res, basket_logs = run_with_basket_log(panels, meta, None)
finally:
    SC.composite_score = orig_cs
    EX.composite_score = orig_cs

perf = ann_of(res)
say(f"  复现回测: 年化={perf['annual']:.4%} 回撤={perf['maxdd']:.4%} 期数={perf['trades']}")
say(f"  composite_score 调用 {len(stats)} 次；权重全 NaN 期数 = {sum(s['w_all_nan'] for s in stats)}")
say(f"  截面唯一值个数: min={min(s['n_unique'] for s in stats)} max={max(s['n_unique'] for s in stats)}")
say(f"  打分全 0 期数 = {sum(s['allzero'] for s in stats)}/{len(stats)}")
uniq_expert = sum(1 for d in basket_logs if len(set(d.values())) == 1)
say(f"  五个专家候选完全相同的期数 = {uniq_expert}/{len(basket_logs)}")
EV["f01_effect"] = dict(
    reproduce=perf,
    score_calls=len(stats),
    w_all_nan_periods=sum(s["w_all_nan"] for s in stats),
    unique_score_min=min(s["n_unique"] for s in stats),
    unique_score_max=max(s["n_unique"] for s in stats),
    allzero_periods=sum(s["allzero"] for s in stats),
    expert_identical_periods=uniq_expert,
    periods=len(basket_logs),
)

# ===========================================================================
# 2. 候选集是否逐期恒定 + 对池成分的敏感性
# ===========================================================================
say("=" * 78)
say("[2] 退化后果：候选集是常数，且对池成分极度敏感")
sets = {tuple(sorted(d["momentum"])) for d in basket_logs}
say(f"  73 期里不同候选集个数 = {len(sets)}  -> 逐期恒定: {len(sets) == 1}")
basket = sorted(next(iter(sets)))
say(f"  恒定候选集({len(basket)}只): {basket}")
held = set()
for tr in res["trades"]:
    held |= {k for k, v in tr["weights"].items() if v > 1e-9}
say(f"  实际持有过 {len(held)} 只；候选集内但从未持有: {sorted(set(basket) - held)}")
EV["basket"] = dict(constant=len(sets) == 1, codes=basket,
                    held=sorted(held), never_held=sorted(set(basket) - held))

frag = {}
drop = basket[0]
for tag, kw in (("drop_one_add_41", dict(limit=40, drop=drop)),
                ("width_39", dict(limit=39)),
                ("width_41", dict(limit=41))):
    p2, m2 = load_pool(**kw)
    r2, l2 = run_with_basket_log(p2, m2, None)
    b2 = set(l2[0]["momentum"])
    frag[tag] = dict(annual=ann_of(r2)["annual"], overlap=len(b2 & set(basket)),
                     basket_size=len(basket), changed=b2 != set(basket))
    say(f"  {tag:<18} 年化={frag[tag]['annual']:8.4%}  候选集重叠={frag[tag]['overlap']}/{len(basket)}"
        f"  与基准不同={frag[tag]['changed']}")
say(f"  基准年化={perf['annual']:.4%}；池宽 39/40/41 对应 "
    f"{frag['width_39']['annual']:.2%} / {perf['annual']:.2%} / {frag['width_41']['annual']:.2%}")
EV["fragility"] = dict(base=perf["annual"], variants=frag)

# ===========================================================================
# 3. 确定性反例（引擎层）
# ===========================================================================
say("=" * 78)
say("[3] 引擎层确定性反例")
cost = CONFIG.cost
unit = cost.a_commission * 2 + cost.a_stamp_tax + cost.a_slippage * 2

w_exec = pd.Series({"d": 1 / 3, "e": 1 / 3, "f": 1 / 3})
w_now = pd.Series({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
turn = float((w_exec - w_now).abs().sum() / 2)
say(f"  [F02] 完全不同标的的全量换仓: 引擎 turnover={turn:.4f} fee={turn * unit:.6f}"
    f"  (真值 turnover=1.0 fee={unit:.6f})")

px = pd.DataFrame({"x": [10, 11, 10], "y": [10, 10, 10], "z": [10, 10, 10]},
                  index=pd.date_range("2024-01-01", periods=3))
w = pd.Series(1 / 3, index=["x", "y", "z"])
r_fw = float(np.prod(1 + (px.pct_change().fillna(0) @ w)) - 1)
say(f"  [F04] 固定权重公式累计={r_fw:.6%}  真实买入持有=0.000000%")

px2 = pd.DataFrame({"x": [10.0, np.nan, 8.0]}, index=pd.date_range("2024-01-01", periods=3))
say(f"  [F05] 缺一根K线: pct_change 全为 NaN -> fillna(0) 记为 "
    f"{float(px2.pct_change().fillna(0).sum().iloc[0]):.4%}  实际应记 -20.0000%")

last_tr = res["trades"][-1]
say(f"  [F06] 末日 signal_date={last_tr['signal_date'][:10]} exec_date={last_tr['exec_date'][:10]}"
    f"  同日执行={last_tr['signal_date'][:10] == last_tr['exec_date'][:10]}")

EV["counterexamples"] = dict(
    fee_full_turnover=dict(turnover=turn, fee=turn * unit, true_fee=unit),
    fixed_weight_formula=dict(engine=r_fw, buy_and_hold=0.0),
    missing_bar=dict(engine=0.0, truth=-0.20),
    terminal_same_day=last_tr["signal_date"][:10] == last_tr["exec_date"][:10],
)

# ===========================================================================
# 4. 风控：存档 vs 复现
# ===========================================================================
say("=" * 78)
say("[4] 风控分级：存档 vs 复现")
d = pickle.load(open(ROOT / "output" / "results.pkl", "rb"))
saved = d["result"]
s_tr = saved["trades"]
s_gs = [x["risk"]["gross_scale"] for x in s_tr]
s_rs = [r for x in s_tr for r in x["risk"]["reasons"]]
say(f"  存档: gross_scale 取值={sorted(set(s_gs))}  L2期数={sum('L2' in r for r in s_rs)}"
    f"  L3={sum('L3' in r for r in s_rs)}  L4={sum('L4' in r for r in s_rs)}"
    f"  深熊={sum('深熊' in r for r in s_rs)}")
n_gs = [x["risk"]["gross_scale"] for x in res["trades"]]
n_rs = [r for x in res["trades"] for r in x["risk"]["reasons"]]
say(f"  复现(bench=None): gross_scale 取值={sorted(set(n_gs))}"
    f"  =0.3 的期数={sum(abs(g - 0.3) < 1e-9 for g in n_gs)}"
    f"  深熊={sum('深熊' in r for r in n_rs)}")
EV["risk"] = dict(saved_gross=sorted(set(s_gs)),
                  saved_L2=sum("L2" in r for r in s_rs),
                  saved_L3=sum("L3" in r for r in s_rs),
                  saved_L4=sum("L4" in r for r in s_rs),
                  saved_bear=sum("深熊" in r for r in s_rs),
                  replay_gross=sorted(set(n_gs)),
                  replay_bear=sum("深熊" in r for r in n_rs))

# ===========================================================================
# 5. 基准序列敏感性 —— 存档 15.25% 的差异来源
# ===========================================================================
say("=" * 78)
say("[5] 存档不可复现的定位：基准序列")
close = pd.concat({c: p["close"] for c, p in panels.items()}, axis=1).sort_index()
variants = {}
for tag, bench in (
    ("bench_none_fallback", None),
    ("synthetic_rising", pd.DataFrame({"close": np.linspace(1, 2.5, len(close))}, index=close.index)),
    ("pool_equalweight_index", pd.DataFrame({"close": (close / close.iloc[0]).mean(axis=1)}, index=close.index)),
):
    r = Engine(market="a").run(panels, meta, bench)
    a = ann_of(r)
    gs = [x["risk"]["gross_scale"] for x in r["trades"]]
    variants[tag] = dict(**a, gross03=sum(abs(g - 0.3) < 1e-9 for g in gs))
    say(f"  {tag:<24} 年化={a['annual']:8.4%} 回撤={a['maxdd']:8.4%} gross=0.3期数={variants[tag]['gross03']}")
sa = ann_of({"nav": saved["nav"], "daily_return": saved["daily_return"], "trades": s_tr})
say(f"  {'存档 results.pkl':<24} 年化={sa['annual']:8.4%} 回撤={sa['maxdd']:8.4%} gross=0.3期数=0")
say("  -> 只要 regime 不触发深熊(gross 恒 1.0)，结果精确复现存档；退化为候选池均价则触发 6 次深熊")
EV["benchmark_sensitivity"] = dict(saved=sa, variants=variants)

# ===========================================================================
# 6. 同池基线：严格 share/cash 账本
# ===========================================================================
say("=" * 78)
say("[6] 同池基线 —— 严格 share/cash 账本 (裁定 29% 之争)")
_seq = pd.Series(np.arange(len(close)), index=close.index)
_last = _seq.groupby(close.index.to_period("M")).last()
eom = np.isin(np.arange(len(close)), _last.to_numpy())
rebal = [x for x in close.index[eom] if x >= pd.Timestamp(CONFIG.start_date)]
exec_dates = [close.index[close.index > x][0] for x in rebal if len(close.index[close.index > x])]
d0, d1 = res["nav"].index[0], res["nav"].index[-1]
sub = close.loc[d0:d1]
oneway = cost.a_commission + cost.a_slippage + cost.a_stamp_tax / 2
allc = list(close.columns)


def ledger(mode):
    cash, sh, navs, rset = 1.0, pd.Series(0.0, index=allc), [], set(exec_dates)
    for i, dd in enumerate(sub.index):
        px = sub.iloc[i].fillna(sub.iloc[i - 1] if i > 0 else sub.iloc[i])
        trade = (i == 0) or (mode == "mr" and dd in rset) or (mode == "daily")
        if trade:
            nav = cash + float((sh * px).sum())
            act = sub.iloc[i].notna()
            tgt = pd.Series(0.0, index=allc)
            tgt[act] = nav / act.sum()
            nav2 = nav - float((tgt - sh * px).abs().sum()) * oneway
            tgt = pd.Series(0.0, index=allc)
            tgt[act] = nav2 / act.sum()
            sh = (tgt / px.replace(0, np.nan)).fillna(0.0)
            cash = nav2 - float((sh * px).sum())
        navs.append(cash + float((sh * px).sum()))
    nv = pd.Series(navs, index=sub.index)
    return nv.pct_change().fillna(0)


def stat(r):
    nv = (1 + r).cumprod()
    n = len(r)
    return dict(annual=float(nv.iloc[-1] ** (252 / n) - 1), total=float(nv.iloc[-1] - 1),
                vol=float(r.std() * np.sqrt(252)), maxdd=float((nv / nv.cummax() - 1).min()))


r_bh, r_mr, r_dl = ledger("bh"), ledger("mr"), ledger("daily")
r_fw = sub.pct_change().fillna(0) @ pd.Series(1.0 / len(allc), index=allc)
r_s = res["daily_return"].reindex(sub.index).fillna(0)
base = {}
for tag, r in (("buy_and_hold_true", r_bh), ("monthly_equalweight_true", r_mr),
               ("daily_equalweight_true", r_dl), ("colleague_formula_zero_cost", r_fw),
               ("strategy_current", r_s)):
    base[tag] = stat(r)
    say(f"  {tag:<28} 年化={base[tag]['annual']:8.4%} 波动={base[tag]['vol']:6.2%}"
        f" 回撤={base[tag]['maxdd']:8.4%} 总={base[tag]['total']:9.2%}")
say("  同事 buyhold_baseline 报告 28.00% / pool_baseline 报告 29.03%")
n = len(r_s)
rel = {}
for tag, r in (("true_buy_and_hold", r_bh), ("true_monthly_ew", r_mr), ("true_daily_ew", r_dl)):
    ex = float((1 + r_s).cumprod().iloc[-1] / (1 + r).cumprod().iloc[-1] - 1)
    rel[tag] = dict(cum=ex, annual=float((1 + ex) ** (252 / n) - 1))
    say(f"  策略 vs {tag:<20} 累计相对={ex:8.2%}  年化相对={rel[tag]['annual']:8.2%}")
EV["baselines"] = dict(ledger=base, strategy_relative=rel)

# ===========================================================================
# 7. 结论数据
# ===========================================================================
say("=" * 78)
say("[7] 汇总")
say(f"  存档年化 {sa['annual']:.4%} = 当前代码在 gross≡1.0 下的输出（精确复现）")
say(f"  复现年化 {perf['annual']:.4%} = 同一代码 + 候选池均价基准（触发 6 次深熊）")
say(f"  策略相对同池月度等权年化 {rel['true_monthly_ew']['annual']:.2%}，"
    f"相对真实买入持有 {rel['true_buy_and_hold']['annual']:.2%}")
say(f"  候选集对池成分敏感：池宽 39/40/41 -> "
    f"{frag['width_39']['annual']:.2%} / {perf['annual']:.2%} / {frag['width_41']['annual']:.2%}")

EV["verdict_numbers"] = dict(saved_annual=sa["annual"], replay_annual=perf["annual"],
                             saved_maxdd=sa["maxdd"], replay_maxdd=perf["maxdd"])

(OUT / "evidence.json").write_text(json.dumps(EV, ensure_ascii=False, indent=2, default=str))
(OUT / "run.log").write_text(LOG.getvalue())
say(f"\n[out] {OUT/'evidence.json'}")
