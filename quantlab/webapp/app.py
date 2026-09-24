"""Local simulation dashboard reading the latest integrity-checked versioned run.
No archived root results/signals are promoted into current advice.

呈现层：静态资源在 static/（dashboard.css / dashboard.js / echarts.min.js）。
本模块只负责取数与门禁，不做策略判断；未经 run_health 校验的结果不发布绩效。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string

from quantlab.portfolio.current import snapshot, summary, ytd_return
from quantlab.report.metrics import performance_table

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["JSON_AS_ASCII"] = False


def _load():
    from quantlab.portfolio.current import _load_result
    return _load_result()


def _clean_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------- HTML shell
# Kept intentionally small: all styling and behaviour live in static/ so they can
# be iterated on without touching the server module. The shell carries no data.
PAGE = """<!doctype html>
<html lang="zh" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantLab · 量化决策看板</title>
<link rel="stylesheet" href="/static/dashboard.css">
<script src="/static/echarts.min.js"></script>
</head>
<body>

<header class="dash-header">
  <div class="brand">
    <div class="mark">QL</div>
    <div>
      <h1>QuantLab <span class="muted" style="font-weight:400;font-size:11.5px">决策看板</span></h1>
      <div class="sub">A股 · 多因子 · 专家轮动 · 人工执行</div>
    </div>
  </div>
  <div class="header-right">
    <span class="chip hide-sm" id="chipStatus"><span class="dot warn"></span>研究账户 <b>—</b></span>
    <span class="chip hide-sm" id="chipPrice"><span class="dot"></span>行情 <b>—</b></span>
    <span class="chip" id="chipExec"><span class="dot"></span>执行日 <b>—</b></span>
    <button class="btn btn-icon" id="themeToggle" aria-label="切换主题" title="切换主题">☾</button>
  </div>
</header>

<div class="wrap">

  <div class="card">
    <div class="tabs" role="tablist">
      <button class="tab active" data-pane="overview-pane" role="tab" aria-selected="true">总览</button>
      <button class="tab" data-pane="holdings-pane" role="tab" aria-selected="false">实时持仓</button>
      <button class="tab" data-pane="signals-pane" role="tab" aria-selected="false">买卖信号</button>
      <button class="tab" data-pane="experts-pane" role="tab" aria-selected="false">专家决策</button>
      <button class="tab" data-pane="run-pane" role="tab" aria-selected="false">运行指引</button>
    </div>
  </div>

  <!-- overview -->
  <div class="pane active" id="overview-pane">
    <div class="grid-kpi" id="kpis"></div>

    <div class="grid-2 mt" id="overviewGrid">
      <div class="card" id="chart-pane">
        <div class="card-head">
          <h2><span class="dot ok"></span>组合净值与回撤</h2>
          <div class="spacer"></div>
          <div class="seg" id="rangeSeg">
            <button data-range="1y">1年</button>
            <button data-range="3y">3年</button>
            <button data-range="all" class="on">全部</button>
          </div>
        </div>
        <div class="card-body">
          <div class="chart" id="chartNav"></div>
          <div class="chart-legend" style="margin:2px 0 4px">
            <span><i style="background:var(--accent-cyan)"></i>组合净值</span>
            <span><i style="background:var(--text-muted)"></i>基准（若可得）</span>
          </div>
          <div class="chart short" id="chartDD"></div>
        </div>
        <div class="legend-note" id="chartNote">回撤以账户历史净值为基准；面积为负值即为水下。右侧仪表标出当前回撤所在的政策档位。</div>
      </div>

      <div class="card">
        <div class="card-head"><h2><span class="dot warn"></span>风险预算</h2></div>
        <div class="card-body" id="riskBody"><div class="empty">加载中…</div></div>
        <div class="legend-note" id="overviewNote"></div>
      </div>
    </div>
  </div>

  <!-- holdings -->
  <div class="pane" id="holdings-pane">
    <div id="holdNote"></div>
    <div class="card mt">
      <div class="card-head">
        <h2><span class="dot ok"></span>当前持仓 · 实时行情</h2>
        <div class="spacer"></div>
        <button class="btn btn-sm" id="refreshH">立即刷新</button>
      </div>
      <div class="card-body">
        <div class="grid-kpi" id="hkpis"></div>
      </div>
      <div class="card-body tight">
        <div class="tbl-scroll"><table class="data" id="htable"></table></div>
      </div>
      <div class="legend-note">模拟账户按股数计价；整手成本高到可能超配时以 ↑ 标注。点击表头可排序。</div>
    </div>
  </div>

  <!-- signals -->
  <div class="pane" id="signals-pane">
    <div class="card">
      <div class="card-head"><h2><span class="dot ok"></span>最新一期买卖信号 <span class="muted" style="font-weight:400;font-size:11.5px" id="sMeta"></span></h2></div>
      <div class="card-body tight">
        <div class="tbl-scroll"><table class="data" id="stable"></table></div>
      </div>
      <div class="legend-note">信号按上一交易日收盘数据生成，T+1 参考执行价；月度调仓。买卖动作需人工复核成交明细后执行。</div>
    </div>
  </div>

  <!-- experts -->
  <div class="pane" id="experts-pane">
    <div class="row" style="margin-bottom:12px">
      <button class="btn" id="refreshE">重新收集今日信息并分析</button>
      <button class="btn" id="reloadE">读取已有结果</button>
      <span id="loading" class="row" style="display:none"><span class="spin"></span><span class="muted">正在重算专家（10~30 秒）…</span></span>
    </div>
    <div class="callout" id="eMeta"></div>
    <div class="stack mt" id="ecards"></div>
  </div>

  <!-- run -->
  <div class="pane" id="run-pane">
    <div class="grid-kpi" id="runKpis"></div>
    <div class="card mt">
      <div class="card-head"><h2><span class="dot ok"></span>平台能力说明</h2></div>
      <div class="card-body">
        <div class="callout">
          <ol style="margin:6px 0 0 18px;line-height:1.9">
            <li><b>盘中实时行情</b>：持仓/行情接口每次请求直接向新浪（腾讯回退）取实时价，页面每 30 秒自动刷新。</li>
            <li><b>日线数据集</b>：每日收盘后运行 <code>scripts/daily_pulse.py</code>，增量补齐当日 K 线与候选池快照，因子/专家意见即基于最新数据重算。</li>
            <li><b>月度信号</b>：按运行配置评估；没有通过研究关口的信号只用于模拟。</li>
            <li><b>美股</b>：交易链路尚未启用；背景信息也必须满足历史可知时间要求。</li>
          </ol>
        </div>
      </div>
    </div>
    <div class="card mt">
      <div class="card-head"><h2>每日一条命令</h2></div>
      <div class="card-body"><pre class="code">./.venv2/bin/python scripts/daily_pulse.py          # 收盘后：更新日线+候选池 → 更新专家意见
./.venv2/bin/python -m quantlab.cli quote           # 随时：查看当前持仓实时行情
./.venv2/bin/python -m quantlab.cli monitor         # 随时：重算专家今日意见
./.venv2/bin/python -m quantlab.cli web             # 启动本看板</pre></div>
    </div>
  </div>

  <div class="foot">
    当前展示模拟研究账户。历史结果尚未验证；数据不完整时不发布收益或可执行建议，不能视为券商持仓。
  </div>
</div>

<script src="/static/dashboard.js"></script>
</body></html>"""


# ---------------------------------------------------------------- gate helper
def _interpretable(result: dict) -> bool:
    return (result.get("status") == "research_unvalidated"
            and bool((result.get("run_health") or {}).get("research_interpretable")))


@app.get("/")
def index():
    data = _load()
    if not data:
        return "请先运行: python -m quantlab.cli backtest", 404
    return render_template_string(PAGE)


# ---------------------------------------------------------------- overview
@app.get("/api/nav")
def api_nav():
    data = _load()
    if not data:
        return jsonify({"error": "no data"}), 404
    result = data["result"]
    if not _interpretable(result):
        reasons = [str(x) for x in (result.get("reasons") or [])]
        blocked = [str(x) for x in ((result.get("run_health") or {}).get("reasons") or [])]
        return jsonify({
            "status": result.get("status", "legacy_unvalidated"),
            "interpretable": False,
            "blocked_reasons": (reasons + blocked)[:8],
            "last_exec": "", "quote_time": "",
            "metrics": None, "ytd": None, "extra": "",
        })
    nav = result["nav"]
    perf = performance_table(nav, result.get("daily_return"))
    y = ytd_return()
    return jsonify(_clean_json({
        "status": result["status"],
        "interpretable": True,
        "last_exec": str(nav.index[-1].date()),
        "quote_time": _quote_marker(),
        "metrics": perf,
        "target_vol": _target_vol(),
        "ytd": y,
        "extra": _extra_note(result),
    }))


def _target_vol() -> float:
    from quantlab.config import CONFIG
    return float(CONFIG.portfolio.target_vol)


# ---------------------------------------------------------------- charts
@app.get("/api/series")
def api_series():
    """Date-aligned nav / benchmark / drawdown for interactive charts.

    Benchmark is rebased to the nav's first point so the two lines are directly
    comparable; a raw index level would differ by orders of magnitude and make
    the comparison meaningless.
    """
    data = _load()
    if not data:
        return jsonify({"error": "no data"}), 404
    result = data["result"]
    if not _interpretable(result):
        return jsonify({"available": False, "dates": [], "nav": [],
                        "benchmark": None, "drawdown": []})
    nav = result["nav"].sort_index()
    dd = result.get("drawdown")
    dd = dd.reindex(nav.index) if dd is not None else None
    bench = None
    b = result.get("benchmark")
    if b is not None and not b.empty and getattr(b, "close", None) is not None:
        bclose = b.close.reindex(nav.index)
        if bclose.notna().any() and float(bclose.dropna().iloc[0]) != 0:
            bench = (bclose / float(bclose.dropna().iloc[0]) * float(nav.iloc[0])).round(6)
    return jsonify(_clean_json({
        "available": True,
        "dates": [str(d.date()) for d in nav.index],
        "nav": [round(float(x), 6) for x in nav.values],
        "benchmark": [float(x) for x in bench.values] if bench is not None else None,
        "drawdown": [round(float(x), 6) for x in dd.values] if dd is not None else [],
    }))


# ---------------------------------------------------------------- risk
@app.get("/api/risk")
def api_risk():
    """Read-only view of the risk budget that the engine already computed.

    Reuses the recorded risk block of the last executed trade rather than
    recomputing, so the panel can never disagree with the ledger.
    """
    data = _load()
    if not data:
        return jsonify({"available": False, "note": "无数据"})
    result = data["result"]
    if not _interpretable(result):
        return jsonify({"available": False, "note": "结果未通过验证，风险预算不展示"})
    trades = result.get("trades") or []
    last = trades[-1] if trades else {}
    risk = last.get("risk") or {}
    if not risk:
        return jsonify({"available": False, "note": "该次回放未记录风险预算"})
    from quantlab.config import CONFIG
    from quantlab.strategy.risk_policy import policy_definition
    rc = CONFIG.research
    pc = CONFIG.portfolio
    policy_id = risk.get("policy_id", "P0")
    policy_state = risk.get("policy_state", "legacy")
    # The panel must show the budget that is actually in force, not the legacy
    # config: under P2 the caps move with the state, so reading CONFIG alone
    # would let the display disagree with the ledger.
    prof = None
    try:
        definition = policy_definition(policy_id)
        raw = definition.get("profiles")
        if raw == "legacy_config":
            prof = None
        elif isinstance(raw, dict):
            if raw.get("normal") is not None or "max_stock_weight" in raw:
                # P1: a single fixed profile; P2: keyed by state.
                if "max_stock_weight" in raw:
                    prof = raw
                else:
                    prof = raw.get(policy_state if policy_state in raw else "normal")
    except Exception:  # noqa: BLE001
        prof = None
    prof = prof or {}
    return jsonify(_clean_json({
        "available": True,
        "policy_id": policy_id,
        "policy_version": risk.get("policy_version", ""),
        "policy_state": policy_state,
        "drawdown": risk.get("drawdown"),
        "level": risk.get("level", 0),
        "gross_scale": risk.get("gross_scale", 1.0),
        "vol_scale": risk.get("vol_scale", 1.0),
        "target_vol": risk.get("target_vol", pc.target_vol),
        "reasons": risk.get("reasons", []),
        "transition_count": risk.get("transition_count", 0),
        "transition_direction": risk.get("transition_direction", 0),
        "max_stock_weight": prof.get("max_stock_weight", rc.max_stock_weight),
        "max_sector_weight": prof.get("max_sector_weight", pc.max_sector_weight),
        "max_cluster_weight": prof.get("max_cluster_weight", rc.max_cluster_weight),
        "cash_buffer": prof.get("cash_buffer", rc.cash_buffer),
        "unknown_sector_weight": 0.25 if policy_id != "P0" else pc.max_sector_weight,
    }))


# ---------------------------------------------------------------- holdings
@app.get("/api/holdings")
def api_holdings():
    rows = snapshot(with_quotes=True)
    summary_d = summary()
    today = 0.0
    for r in rows:
        if r["pct_chg"] is not None:
            today += r["weight"] * r["pct_chg"]
    quote_time = rows[0].get("quote_time") if rows else None
    return jsonify({
        "status": summary_d.get("status", "blocked_data"),
        "account_type": "simulation",
        "rows": _clean_json(rows),
        "summary": _clean_json(summary_d),
        "today": today if any(r["pct_chg"] is not None for r in rows) else None,
        "ytd": ytd_return(),
        "quote_time": quote_time,
    })


def _quote_marker() -> str:
    try:
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")
    except Exception:  # noqa: BLE001
        return "—"


# ---------------------------------------------------------------- signals
@app.get("/api/signals")
def api_signals():
    from quantlab.cli import signal_payload
    data = _load()
    if not data:
        return jsonify({"status": "blocked_data", "trades": [], "names": {}})
    payload = signal_payload(data["result"])
    payload["names"] = _code_names_json()
    return jsonify(_clean_json(payload))


def _code_names_json() -> dict[str, str]:
    from quantlab.data.store import DataStore
    try:
        cand = DataStore().load("a", "candidates_a")
    except Exception:  # noqa: BLE001
        return {}
    if cand is None or "name" not in cand.columns:
        return {}
    return {str(i): str(r["name"]) for i, r in cand.iterrows()}


# ---------------------------------------------------------------- experts
@app.get("/api/experts")
def api_experts():
    data = _load()
    if not data:
        return jsonify({"status": "blocked_data", "experts": {}})
    result = data["result"]
    trace = result.get("signal_trace", [])
    return jsonify(_clean_json({"status": result.get("status", "legacy_unvalidated"),
                                "experts": {},
                                "note": "专家价值未通过验证；展示同一回放记录，不另行计算或执行。",
                                "date": result.get("as_of"),
                                "n_candidates": len(result.get("codes", [])),
                                "coverage": {"note": "研究模型，未证明专家独立贡献"},
                                "decision": trace[-1] if trace else {}}))


def _extra_note(result: dict) -> str:
    """Short provenance / gate note. Rendered as text, never as executable advice."""
    trades = result.get("trades") or []
    last = trades[-1] if trades else {}
    reasons = (last.get("risk") or {}).get("reasons", [])
    bm = result.get("benchmark_manifest", {})
    src = bm.get("source", "unknown")
    rows = []
    if reasons:
        rows.append("风控触发：" + "；".join(str(r) for r in reasons[:3]))
    rows.append(f"数据源：{src}；股票池覆盖：{bm.get('universe_coverage', 'unknown')}。")
    rows.append("基于每日账本；未完成策略有效性验证，结果仅供研究。")
    from html import escape
    return escape(" ".join(rows))


def main(port: int = 8899) -> None:
    app.run(host="127.0.0.1", port=port, debug=False)
