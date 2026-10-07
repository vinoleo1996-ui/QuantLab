"""QuantLab research decision hub and read-only simulation portfolio view.

The research hub projects a small, allowlisted catalog.  The separate
portfolio view reads the existing simulation-account adapter and public market
adapters, with explicit timestamps and fail-closed missing-data states.  No
endpoint starts research, changes strategy, or trades.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request
from zoneinfo import ZoneInfo

from quantlab.config import CONFIG
from quantlab.portfolio.current import snapshot, summary, ytd_return
from quantlab.portfolio.grok_paper import PAPER_STATUS, load_grok_paper_ledger
from quantlab.data.public_desk import cached_observe, search_a_shares
from quantlab.portfolio.market_observatory import (
    local_panel_fallback,
    parse_ths_revenue_forecast,
    resample_ohlcv,
    validate_period,
)
from quantlab.report.metrics import performance_table
from quantlab.report.research_state import load_research_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_LABEL = "本次请求的 allowlist 证据快照（证据不共享单一截止时间）"
GROK_PAPER_ROOT = Path.home() / "grokbot_isolated" / "paper"
GROK_PAPER_FEED = GROK_PAPER_ROOT / "frontend_feed" / "paper_nav.json"
GROK_PAPER_NAV_CSV = GROK_PAPER_ROOT / "output" / "nav_daily.csv"
LOCAL_MARKET_ROOT = PROJECT_ROOT / "data" / "local_market" / "1c85fd9aceaa03c3c4863bec"
LOCAL_MARKET_CUTOFF = "2026-06-30"

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["JSON_AS_ASCII"] = False


def _load():
    """Read the existing versioned simulation run used by the legacy dashboard."""

    from quantlab.portfolio.current import _load_result

    return _load_result()


def _clean_json(obj):
    """Keep legacy dashboard responses compatible with numpy/pandas values."""

    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="QuantLab 只读研究裁决与证据投影">
  <title>QuantLab · 研究裁决台</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='research_hub.css') }}">
</head>
<body>
  <main id="research-hub" data-research-hub aria-live="polite"></main>
  <noscript>此页面需要启用 JavaScript，才能读取只读研究证据投影。</noscript>
  <script defer src="{{ url_for('static', filename='research_hub.js') }}"></script>
</body>
</html>"""


LEGACY_PAGE = """<!doctype html>
<html lang="zh-CN" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="QuantLab 模拟研究账户决策看板">
<title>QuantLab · 决策看板</title>
<link rel="stylesheet" href="/static/dashboard.css"><script defer src="/static/echarts.min.js"></script>
</head>
<body>
<a class="skip-link" href="#dashboard-main">跳到主要内容</a>
<header class="dash-header">
  <div class="header-inner">
    <a class="brand" href="/dashboard" aria-label="QuantLab 决策看板首页">
      <span class="brand-mark" aria-hidden="true">QL</span>
      <span class="brand-copy"><strong>QuantLab</strong><small>决策看板 · A股研究账户</small></span>
    </a>
    <nav class="dash-nav" aria-label="页面导航">
      <a class="is-current" href="/dashboard" aria-current="page">决策看板</a>
      <a href="/portfolio">账户行情</a>
      <a href="/research">研究裁决</a>
    </nav>
    <div class="header-right">
      <span class="chip hide-sm" id="chipStatus"><span class="dot warn"></span><span>研究账户</span><b>—</b></span>
      <span class="chip hide-sm" id="chipPrice"><span class="dot"></span><span>行情</span><b>—</b></span>
      <span class="chip" id="chipExec"><span class="dot"></span><span>执行日</span><b>—</b></span>
      <button class="btn btn-icon" id="themeToggle" aria-label="切换到亮色主题" title="切换主题" aria-pressed="false"><span class="theme-glyph" aria-hidden="true"></span><span class="sr-only">切换主题</span></button>
    </div>
  </div>
</header>

<main class="dash-main" id="dashboard-main">
  <section class="dashboard-hero" aria-labelledby="dashboard-title">
    <div class="hero-copy">
      <p class="eyebrow">模拟研究账户 · 只读观察</p>
      <h1 id="dashboard-title">先看账户，再看风险。</h1>
      <p class="hero-summary">把组合状态、持仓涨跌和研究门禁放在同一条阅读路径上。这里展示已落盘的模拟结果，不会下单，也不会把缺失数据填成假收益。</p>
    </div>
    <div class="hero-actions" aria-label="快捷入口">
      <a class="btn btn-primary" href="/portfolio">查看账户行情<span class="btn-arrow" aria-hidden="true">→</span></a>
      <a class="btn btn-quiet" href="/research">打开研究裁决<span class="btn-arrow" aria-hidden="true">→</span></a>
    </div>
  </section>

  <section class="status-banner" id="dashboardStatus" role="status" aria-live="polite">
    <span class="status-banner__icon" aria-hidden="true"></span>
    <div><strong id="statusTitle">正在读取账户状态</strong><p id="statusCopy">数据会在通过自身验证后才进入绩效和信号区域。</p></div>
    <span class="status-banner__meta" id="statusMeta">只读</span>
  </section>

  <section class="dashboard-tabs card" aria-label="账户工作区">
    <div class="tabs" role="tablist" aria-label="看板分区">
      <button class="tab active" id="overview-tab" data-pane="overview-pane" role="tab" aria-selected="true" aria-controls="overview-pane" tabindex="0"><span class="tab-index">01</span>总览</button>
      <button class="tab" id="holdings-tab" data-pane="holdings-pane" role="tab" aria-selected="false" aria-controls="holdings-pane" tabindex="-1"><span class="tab-index">02</span>实时持仓</button>
      <button class="tab" id="signals-tab" data-pane="signals-pane" role="tab" aria-selected="false" aria-controls="signals-pane" tabindex="-1"><span class="tab-index">03</span>买卖信号</button>
      <button class="tab" id="experts-tab" data-pane="experts-pane" role="tab" aria-selected="false" aria-controls="experts-pane" tabindex="-1"><span class="tab-index">04</span>专家决策</button>
      <button class="tab" id="run-tab" data-pane="run-pane" role="tab" aria-selected="false" aria-controls="run-pane" tabindex="-1"><span class="tab-index">05</span>运行指引</button>
    </div>
  </section>

  <section class="pane active" id="overview-pane" role="tabpanel" tabindex="0" aria-labelledby="overview-tab">
    <div class="section-heading"><div><p class="eyebrow">账户快照 · 01</p><h2>组合表现</h2></div><span class="section-meta" id="overviewAsOf">等待数据</span></div>
    <div class="grid-kpi" id="kpis" aria-live="polite"></div>
    <div class="overview-grid mt" id="overviewGrid">
      <article class="card chart-card" id="chart-pane">
        <div class="card-head"><div><p class="eyebrow">净值轨迹</p><h2>组合净值与回撤</h2></div><div class="spacer"></div><div class="seg" id="rangeSeg" role="group" aria-label="图表时间范围"><button type="button" data-range="1y">1年</button><button type="button" data-range="3y">3年</button><button type="button" data-range="all" class="on">全部</button></div></div>
        <div class="card-body"><div class="chart" id="chartNav" role="img" aria-label="组合净值曲线"></div><div class="chart-legend"><span><i class="legend-line legend-line--accent"></i>组合净值</span><span><i class="legend-line legend-line--muted"></i>基准（若可得）</span><span class="legend-note-inline">按日账本 · 研究回放</span></div><div class="chart short" id="chartDD" role="img" aria-label="组合回撤曲线"></div></div>
        <div class="legend-note" id="chartNote">回撤以账户历史净值为基准；面积为负值即为水下。</div>
      </article>
      <aside class="card risk-card" aria-labelledby="risk-title"><div class="card-head"><div><p class="eyebrow">政策约束</p><h2 id="risk-title">风险预算</h2></div><span class="card-badge">只读</span></div><div class="card-body" id="riskBody"><div class="empty loading-state"><span class="skeleton-line skeleton-line--wide"></span><span class="skeleton-line"></span><span class="skeleton-line skeleton-line--short"></span></div></div><div class="legend-note" id="overviewNote"></div></aside>
    </div>
  </section>

  <section class="pane" id="holdings-pane" role="tabpanel" tabindex="0" aria-labelledby="holdings-tab">
    <div class="section-heading"><div><p class="eyebrow">账户快照 · 02</p><h2>实时持仓</h2><p class="section-description">看清现价、成本、今日涨跌和组合权重；需要完整日K时进入账户行情。</p></div><div class="section-actions"><a class="btn btn-primary btn-sm" href="/portfolio">打开日K与盈亏</a><button class="btn btn-sm" id="refreshH" type="button">刷新行情</button></div></div>
    <div id="holdNote"></div>
    <article class="card holdings-card mt"><div class="card-head"><div><p class="eyebrow">持仓明细</p><h2>当前持仓 · 公开行情估值</h2></div><span class="card-badge">模拟账户</span></div><div class="card-body"><div class="grid-kpi" id="hkpis"></div></div><div class="card-body tight"><div class="tbl-scroll"><table class="data" id="htable"></table></div></div><div class="legend-note">模拟账户按股数计价；现价缺失时保留账本标记价并明确标注。点击“账户行情”可查看单只证券日K。</div></article>
  </section>

  <section class="pane" id="signals-pane" role="tabpanel" tabindex="0" aria-labelledby="signals-tab">
    <div class="section-heading"><div><p class="eyebrow">执行准备 · 03</p><h2>买卖信号</h2><p class="section-description">只显示上一交易日收盘生成的参考动作，需人工复核后执行。</p></div><span class="section-meta" id="sMeta">信号日等待中</span></div>
    <article class="card"><div class="card-head"><div><p class="eyebrow">最新一期</p><h2>目标权重变化</h2></div><span class="card-badge">T+1 参考</span></div><div class="card-body tight"><div class="tbl-scroll"><table class="data" id="stable"></table></div></div><div class="legend-note">买卖动作不等于已成交委托；页面不会自动下单。</div></article>
  </section>

  <section class="pane" id="experts-pane" role="tabpanel" tabindex="0" aria-labelledby="experts-tab">
    <div class="section-heading"><div><p class="eyebrow">研究模型 · 04</p><h2>专家决策</h2><p class="section-description">只读展示已有回放记录，不在页面里重新训练或改变策略。</p></div><div class="section-actions"><button class="btn btn-sm" id="reloadE" type="button">读取已有结果</button><button class="btn btn-primary btn-sm" id="refreshE" type="button">重新收集信息</button></div></div>
    <div class="loading-inline" id="loading" role="status" aria-live="polite" hidden><span class="spin"></span><span>正在读取专家信息…</span></div><div class="callout" id="eMeta"></div><div class="stack mt" id="ecards"></div>
  </section>

  <section class="pane" id="run-pane" role="tabpanel" tabindex="0" aria-labelledby="run-tab">
    <div class="section-heading"><div><p class="eyebrow">操作说明 · 05</p><h2>运行指引</h2><p class="section-description">把日常查看动作、数据边界和研究权限说清楚。</p></div></div>
    <div class="grid-kpi" id="runKpis"></div>
    <div class="run-grid mt"><article class="card"><div class="card-head"><div><p class="eyebrow">工作流</p><h2>每天看什么</h2></div></div><div class="card-body"><ol class="workflow-list"><li><span>01</span><div><b>收盘后</b><p>运行日线增量与候选池快照，确认数据时间戳。</p></div></li><li><span>02</span><div><b>调仓前</b><p>先看持仓涨跌和风险预算，再人工复核买卖信号。</p></div></li><li><span>03</span><div><b>研究时</b><p>从研究裁决台查看证据门禁，不把条件性结果当成采用许可。</p></div></li></ol></div></article><article class="card"><div class="card-head"><div><p class="eyebrow">能力边界</p><h2>当前可以做什么</h2></div></div><div class="card-body"><div class="callout good">盘中行情、日K和模拟账本是只读展示；交易链路未启用，页面不会创建委托。</div><div class="callout warn mt-sm">数据或验证未通过时，收益、回撤和信号会保持空态，这是门禁状态，不是前端故障。</div></div></article></div>
  </section>

  <footer class="dash-footer"><span>QuantLab · 模拟研究账户只读视图</span><span><a href="/portfolio">账户行情</a><span aria-hidden="true"> · </span><a href="/research">研究裁决</a></span></footer>
</main>
<script src="/static/dashboard.js"></script>
</body></html>"""


def _catalog() -> dict[str, Any]:
    """Load the catalog at request time without retaining mutable research state."""

    return load_research_catalog(PROJECT_ROOT).to_dict()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_relative_path(value: object) -> str | None:
    """Only expose project-relative provenance paths in the browser."""

    if not isinstance(value, str):
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return value


def _reference_index(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    references = catalog.get("evidence_references")
    if not isinstance(references, (list, tuple)):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for reference in references:
        item = _mapping(reference)
        artifact_id = item.get("id")
        if isinstance(artifact_id, str):
            indexed[artifact_id] = item
    return indexed


def _public_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Keep API provenance compact and free of artifact payloads/absolute paths."""

    return {
        "id": reference.get("id"),
        "label": reference.get("label"),
        "path": _safe_relative_path(reference.get("path")),
        "availability": reference.get("availability"),
        "integrity": reference.get("integrity"),
        "evidence_label": reference.get("evidence_label"),
        "content_sha256": reference.get("content_sha256"),
    }


def _provenance(catalog: Mapping[str, Any], artifact_ids: object) -> list[dict[str, Any]]:
    index = _reference_index(catalog)
    ids = artifact_ids if isinstance(artifact_ids, (list, tuple, set)) else (artifact_ids,)
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for artifact_id in ids:
        if not isinstance(artifact_id, str) or artifact_id in seen:
            continue
        seen.add(artifact_id)
        reference = index.get(artifact_id)
        if reference is not None:
            output.append(_public_reference(reference))
    return output


def _envelope(
    *,
    value: object,
    status: str,
    provenance: list[dict[str, Any]],
    caveats: list[str],
    drilldown: Mapping[str, Any],
):
    """Use one explicit response contract for every research-hub panel."""

    return jsonify({
        "value": value,
        "status": status,
        "provenance": provenance,
        "as_of": SNAPSHOT_LABEL,
        "caveats": caveats,
        "drilldown": dict(drilldown),
    })


def _decision_payload(catalog: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adoption = _mapping(catalog.get("adoption"))
    raw_gates = adoption.get("gates")
    gates = [_mapping(gate) for gate in raw_gates] if isinstance(raw_gates, (list, tuple)) else []
    source_ids = ["continuity_ledger"]
    for gate in gates:
        sources = gate.get("source_artifacts")
        if isinstance(sources, (list, tuple, set)):
            source_ids.extend(source for source in sources if isinstance(source, str))

    # The catalog's invariant is expressed in English for code consumers; keep
    # the Chinese product surface direct and readable without paraphrasing any
    # research result.
    decision = "没有任何 allowlist 证据授予策略采用、自动交易或生产默认策略变更。"
    return {
        "title": "当前不可采用",
        "headline": "当前不可采用",
        "adoption_status": "not_adoptable",
        "status": "not_adoptable",
        "summary": decision,
        "safe_action": "维持只读研究接续；不得据此启动账户、下单或改变生产默认策略。",
        "gates": gates,
        # Never trust a caller, stale artifact, or future catalog refactor to
        # convert this presentation layer into an adoption approval.
        "promotion_approved": False,
    }, _provenance(catalog, source_ids)


def _candidate_payload(catalog: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_candidates = catalog.get("candidates")
    candidates = raw_candidates if isinstance(raw_candidates, (list, tuple)) else []
    public: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for candidate in candidates:
        item = _mapping(candidate)
        if not item:
            continue
        item["promotion_approved"] = False
        item["adoptable"] = False
        item["source_path"] = _safe_relative_path(item.get("source_path"))
        source = item.get("source_artifact")
        if isinstance(source, str):
            source_ids.append(source)
        public.append(item)
    return public, _provenance(catalog, source_ids)


def _evidence_payload(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    references = catalog.get("evidence_references")
    if not isinstance(references, (list, tuple)):
        return []
    return [_public_reference(_mapping(reference)) for reference in references if _mapping(reference)]


@app.get("/")
@app.get("/dashboard")
def index():
    """Keep the established decision dashboard as the primary product surface."""

    return render_template_string(LEGACY_PAGE)


@app.get("/research")
def research_hub():
    """Serve the separate evidence-only research decision surface."""

    return render_template_string(PAGE)


@app.get("/api/research/decision")
def api_research_decision():
    catalog = _catalog()
    value, provenance = _decision_payload(catalog)
    return _envelope(
        value=value,
        status="not_adoptable",
        provenance=provenance,
        caveats=[
            "只读取显式 allowlist；不把历史运行指针、原始成交或临时文件视为当前研究状态。",
            "条件性历史结果、测试通过或来源自称的批准均不能构成采用或自动交易授权。",
        ],
        drilldown={
            "source_policy": _mapping(catalog.get("source_policy")),
            "current_execution_state": _mapping(catalog.get("current_execution_state")),
            "integrity_warnings": list(catalog.get("integrity_warnings") or ()),
        },
    )


@app.get("/api/research/candidates")
def api_research_candidates():
    catalog = _catalog()
    value, provenance = _candidate_payload(catalog)
    return _envelope(
        value=value,
        status="not_adoptable",
        provenance=provenance,
        caveats=[
            "候选只用于研究对照；全部强制标记为不可采用。",
            "本页不提供持仓、信号、行情或交易动作。",
        ],
        drilldown={
            "source_policy": _mapping(catalog.get("source_policy")),
            "integrity_warnings": list(catalog.get("integrity_warnings") or ()),
        },
    )


@app.get("/api/research/evidence")
def api_research_evidence():
    catalog = _catalog()
    value = _evidence_payload(catalog)
    return _envelope(
        value=value,
        status="unverified",
        provenance=value,
        caveats=[
            "证据可追溯不等于其结论已经独立复核或可采用。",
            "路径均为项目相对路径；不公开原始临时文件或运行目录。",
        ],
        drilldown={
            "source_policy": _mapping(catalog.get("source_policy")),
            "integrity_warnings": list(catalog.get("integrity_warnings") or ()),
        },
    )


@app.get("/api/research/open-questions")
def api_research_open_questions():
    catalog = _catalog()
    questions = catalog.get("open_questions")
    value = [_mapping(question) for question in questions] if isinstance(questions, (list, tuple)) else []
    source_ids: list[str] = []
    for question in value:
        sources = question.get("source_artifacts")
        if isinstance(sources, (list, tuple, set)):
            source_ids.extend(source for source in sources if isinstance(source, str))
    return _envelope(
        value=value,
        status="pending",
        provenance=_provenance(catalog, source_ids),
        caveats=[
            "开放问题不是已批准实验；任何新实验仍须先预注册并取得用户授权。",
            "不把尚未解决的问题包装成推荐、信号或自动执行任务。",
        ],
        drilldown={
            "current_execution_state": _mapping(catalog.get("current_execution_state")),
            "integrity_warnings": list(catalog.get("integrity_warnings") or ()),
        },
    )


# ---------------------------------------------------------------------------
# Established decision-dashboard APIs

def _interpretable(result: Mapping[str, Any]) -> bool:
    """Only publish performance panels for a run that passed its own gate."""

    return (
        result.get("status") == "research_unvalidated"
        and bool((_mapping(result.get("run_health"))).get("research_interpretable"))
    )


def _target_vol() -> float:
    from quantlab.config import CONFIG

    return float(CONFIG.portfolio.target_vol)


def _quote_marker() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _extra_note(result: Mapping[str, Any]) -> str:
    """Short provenance / gate note rendered as text, never as executable advice."""

    from html import escape

    trades = result.get("trades") or []
    last = trades[-1] if trades else {}
    reasons = (_mapping(last.get("risk"))).get("reasons", [])
    benchmark_manifest = _mapping(result.get("benchmark_manifest"))
    source = benchmark_manifest.get("source", "unknown")
    rows: list[str] = []
    if reasons:
        rows.append("风控触发：" + "；".join(str(reason) for reason in reasons[:3]))
    rows.append(
        f"数据源：{source}；股票池覆盖：{benchmark_manifest.get('universe_coverage', 'unknown')}。"
    )
    rows.append("基于每日账本；未完成策略有效性验证，结果仅供研究。")
    return escape(" ".join(rows))


@app.get("/api/nav")
def api_nav():
    data = _load()
    if not data:
        return jsonify({"error": "no data"}), 404
    result = _mapping(data.get("result"))
    if not _interpretable(result):
        reasons = [str(item) for item in (result.get("reasons") or [])]
        health = _mapping(result.get("run_health"))
        blocked = [str(item) for item in (health.get("reasons") or [])]
        return jsonify({
            "status": result.get("status", "legacy_unvalidated"),
            "interpretable": False,
            "blocked_reasons": (reasons + blocked)[:8],
            "last_exec": "",
            "quote_time": "",
            "metrics": None,
            "ytd": None,
            "extra": "",
        })
    nav = result["nav"]
    perf = performance_table(nav, result.get("daily_return"))
    return jsonify(_clean_json({
        "status": result["status"],
        "interpretable": True,
        "last_exec": str(nav.index[-1].date()),
        "quote_time": _quote_marker(),
        "metrics": perf,
        "target_vol": _target_vol(),
        "ytd": ytd_return(),
        "extra": _extra_note(result),
    }))


@app.get("/api/series")
def api_series():
    """Return date-aligned net value, benchmark and drawdown for the old charts."""

    data = _load()
    if not data:
        return jsonify({"error": "no data"}), 404
    result = _mapping(data.get("result"))
    if not _interpretable(result):
        return jsonify({"available": False, "dates": [], "nav": [], "benchmark": None, "drawdown": []})
    nav = result["nav"].sort_index()
    drawdown = result.get("drawdown")
    drawdown = drawdown.reindex(nav.index) if drawdown is not None else None
    benchmark = None
    raw_benchmark = result.get("benchmark")
    if raw_benchmark is not None and not raw_benchmark.empty and getattr(raw_benchmark, "close", None) is not None:
        close = raw_benchmark.close.reindex(nav.index)
        if close.notna().any() and float(close.dropna().iloc[0]) != 0:
            benchmark = (close / float(close.dropna().iloc[0]) * float(nav.iloc[0])).round(6)
    return jsonify(_clean_json({
        "available": True,
        "dates": [str(date.date()) for date in nav.index],
        "nav": [round(float(value), 6) for value in nav.values],
        "benchmark": [float(value) for value in benchmark.values] if benchmark is not None else None,
        "drawdown": [round(float(value), 6) for value in drawdown.values] if drawdown is not None else [],
    }))


@app.get("/api/risk")
def api_risk():
    """Show the risk budget recorded on the last executed trade."""

    data = _load()
    if not data:
        return jsonify({"available": False, "note": "无数据"})
    result = _mapping(data.get("result"))
    if not _interpretable(result):
        return jsonify({"available": False, "note": "结果未通过验证，风险预算不展示"})
    trades = result.get("trades") or []
    last = trades[-1] if trades else {}
    risk = _mapping(last.get("risk"))
    if not risk:
        return jsonify({"available": False, "note": "该次回放未记录风险预算"})
    from quantlab.config import CONFIG
    from quantlab.strategy.risk_policy import policy_definition

    research_config = CONFIG.research
    portfolio_config = CONFIG.portfolio
    policy_id = risk.get("policy_id", "P0")
    policy_state = risk.get("policy_state", "legacy")
    profile: dict[str, Any] = {}
    try:
        definition = policy_definition(policy_id)
        profiles = definition.get("profiles")
        if profiles == "legacy_config":
            profile = {}
        elif isinstance(profiles, dict):
            if profiles.get("normal") is not None or "max_stock_weight" in profiles:
                if "max_stock_weight" in profiles:
                    profile = profiles
                else:
                    profile = profiles.get(policy_state if policy_state in profiles else "normal") or {}
    except Exception:  # noqa: BLE001 - a missing definition must not break the dashboard.
        profile = {}
    return jsonify(_clean_json({
        "available": True,
        "policy_id": policy_id,
        "policy_version": risk.get("policy_version", ""),
        "policy_state": policy_state,
        "drawdown": risk.get("drawdown"),
        "level": risk.get("level", 0),
        "gross_scale": risk.get("gross_scale", 1.0),
        "vol_scale": risk.get("vol_scale", 1.0),
        "target_vol": risk.get("target_vol", portfolio_config.target_vol),
        "reasons": risk.get("reasons", []),
        "transition_count": risk.get("transition_count", 0),
        "transition_direction": risk.get("transition_direction", 0),
        "max_stock_weight": profile.get("max_stock_weight", research_config.max_stock_weight),
        "max_sector_weight": profile.get("max_sector_weight", portfolio_config.max_sector_weight),
        "max_cluster_weight": profile.get("max_cluster_weight", research_config.max_cluster_weight),
        "cash_buffer": profile.get("cash_buffer", research_config.cash_buffer),
        "unknown_sector_weight": 0.25 if policy_id != "P0" else portfolio_config.max_sector_weight,
    }))


@app.get("/api/holdings")
def api_holdings():
    """Compatibility endpoint for the established holdings tab.

    The explicitly registered simulation ledger remains the only holdings
    source. A blocked run or weight-only replay is never promoted here.
    """

    try:
        rows = snapshot(with_quotes=True)
        summary_data = summary()
    except Exception as exc:  # noqa: BLE001 - expose a visible fail-closed state.
        return jsonify({
            "status": "unavailable",
            "account_type": "simulation",
            "rows": [],
            "summary": {},
            "today": None,
            "ytd": None,
            "quote_time": None,
            "error": f"持仓读取失败：{type(exc).__name__}",
        }), 200
    account_status = summary_data.get("status", "blocked_data")
    today = 0.0
    has_quote = False
    for row in rows:
        pct_change = row.get("pct_chg")
        weight = row.get("weight")
        if pct_change is not None and weight is not None:
            today += float(weight) * float(pct_change)
            has_quote = True
    quote_time = rows[0].get("quote_time") if rows else None
    return jsonify({
        "status": summary_data.get("status", "blocked_data"),
        "account_status": account_status,
        "legacy_snapshot": False,
        "account_type": "simulation",
        "rows": _clean_json(rows),
        "summary": _clean_json(summary_data),
        "today": today if has_quote else None,
        "ytd": ytd_return(),
        "quote_time": quote_time,
        "caveats": ([
            "展示的是固定来源的主模拟账户快照；公开报价只用于估算今日变化。",
            "该账户仍有数据门禁缺口，未获策略采用或自动执行许可。",
        ] if rows else [
            "没有可展示的主模拟账户快照。",
            "不会从历史回放权重、目录扫描或 latest 指针猜测仓位。",
        ]),
    })


@app.get("/api/signals")
def api_signals():
    from quantlab.cli import signal_payload

    data = _load()
    if not data:
        return jsonify({"status": "blocked_data", "trades": [], "names": {}})
    result = _mapping(data.get("result"))
    if result.get("status") != "research_unvalidated":
        return jsonify({"status": result.get("status", "blocked_data"), "trades": [], "names": {}})
    try:
        payload = signal_payload(result)
    except Exception:  # noqa: BLE001 - a stale optional trace must not break the page.
        payload = {"status": result.get("status", "blocked_data"), "trades": []}
    payload["names"] = _code_names_json()
    return jsonify(_clean_json(payload))


def _code_names_json() -> dict[str, str]:
    from quantlab.data.store import DataStore

    try:
        candidates = DataStore().load("a", "candidates_a")
    except Exception:  # noqa: BLE001
        return {}
    if candidates is None or "name" not in candidates.columns:
        return {}
    return {str(index): str(row["name"]) for index, row in candidates.iterrows()}


@app.get("/api/experts")
def api_experts():
    data = _load()
    if not data:
        return jsonify({"status": "blocked_data", "experts": {}})
    result = _mapping(data.get("result"))
    trace = result.get("signal_trace", [])
    return jsonify(_clean_json({
        "status": result.get("status", "legacy_unvalidated"),
        "experts": {},
        "note": "专家价值未通过验证；展示同一回放记录，不另行计算或执行。",
        "date": result.get("as_of"),
        "n_candidates": len(result.get("codes", [])),
        "coverage": {"note": "研究模型，未证明专家独立贡献"},
        "decision": trace[-1] if trace else {},
    }))


# ---------------------------------------------------------------------------
# Simulation portfolio market view

def _finite_number(value: object) -> float | None:
    """Convert scalar quote/account values without leaking NaN into JSON."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quote_timestamp_has_clock(value: object) -> bool:
    """Require a provider-supplied clock time before treating a price as current."""

    return bool(re.search(r"(?:T|\s)\d{1,2}:\d{2}(?::\d{2})?", str(value or "").strip()))


def _public_quote_state(price: float | None, quote_time: object, *, now: datetime | None = None) -> str:
    """Classify a provider quote without allowing an undated value to look live."""

    if price is None or price <= 0:
        return "ledger_mark"
    if not _quote_timestamp_has_clock(quote_time):
        return "timestamp_missing"
    state = _quote_state_from_timestamp(quote_time, now=now)
    return "timestamp_invalid" if state == "unavailable" else state


def _quote_timestamp_is_today(value: object, *, now: datetime) -> bool:
    """Return whether a provider timestamp belongs to the current Shanghai date."""

    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return False
    if pd.isna(parsed):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("Asia/Shanghai")
    else:
        parsed = parsed.tz_convert("Asia/Shanghai")
    return parsed.date() == now.date()


def _portfolio_day_quote_is_eligible(row: Mapping[str, Any], *, now: datetime) -> bool:
    """Admit a quote to portfolio-day P&L only under its matching session rule."""

    if not row.get("price_is_public") or row.get("day_pnl") is None:
        return False
    minute = (now.hour, now.minute)
    # 11:30 is already the lunch break in the quote-state classifier.  Check it
    # before the morning session, whose upper bound also touches 11:30.
    if (11, 30) <= minute < (13, 0):
        # The 11:30 print is the current price until the afternoon open.
        # An earlier morning tick is delayed and must not complete the basket.
        return row.get("quote_state") == "lunch_break" and _quote_timestamp_is_today(row.get("quote_time"), now=now)
    in_session = ((9, 30) <= minute < (11, 30)) or ((13, 0) <= minute <= (15, 0))
    if in_session:
        # A prior close is useful to show per security, but it cannot join a
        # live session's basket return with current ticks.
        return row.get("quote_state") == "trading"
    if minute > (15, 0):
        # After the close, accept only a provider close stamped on this date;
        # a Friday close must not become Monday's "today" return.
        return row.get("quote_state") == "closed" and _quote_timestamp_is_today(row.get("quote_time"), now=now)
    return False


def _json_safe(value: object) -> object:
    """Keep public market responses JSON-safe and explicit about missing data."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
        try:
            return value.isoformat()
        except (ValueError, TypeError):
            pass
    number = _finite_number(value)
    if isinstance(value, (int, float)) and number is not None:
        return number
    return value


def _normalize_portfolio_rows(rows: object, account_summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join dated ledger marks with optional public quotes for display only.

    The ledger's ``price`` is retained as ``mark_price``.  ``current_price`` is
    the public quote when available and otherwise falls back to that dated mark;
    the response exposes the fallback so a missing quote can never look live.
    """

    raw_rows = rows if isinstance(rows, (list, tuple)) else []
    quote_now = _current_shanghai_time()
    if quote_now.tzinfo is None:
        quote_now = quote_now.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    else:
        quote_now = quote_now.astimezone(ZoneInfo("Asia/Shanghai"))
    normalized: list[dict[str, Any]] = []
    for raw in raw_rows:
        item = dict(raw) if isinstance(raw, Mapping) else {}
        quantity_value = _finite_number(item.get("quantity"))
        quantity = quantity_value or 0.0
        quantity_known = quantity_value is not None
        mark_price = _finite_number(item.get("price"))
        received_public_price = _finite_number(item.get("live_quote"))
        quote_state = _public_quote_state(received_public_price, item.get("quote_time"), now=quote_now)
        price_is_public = quote_state in {"trading", "lunch_break", "delayed", "closed", "pre_open"}
        # A source price without a trustworthy timestamp remains visible in
        # diagnostics, but cannot replace the ledger mark in valuation/P&L.
        live_price = received_public_price if price_is_public else None
        current_price = live_price if live_price is not None else mark_price
        cost = _finite_number(item.get("cost"))
        pre_close = _finite_number(item.get("pre_close"))
        market_value = quantity * current_price if quantity_known and current_price is not None else None
        mark_value = quantity * mark_price if quantity_known and mark_price is not None else None
        cost_value = quantity * cost if quantity_known and cost is not None else None
        pnl = market_value - cost_value if market_value is not None and cost_value is not None else None
        day_pnl = quantity * (current_price - pre_close) if quantity_known and current_price is not None and pre_close is not None else None
        normalized.append({
            "code": str(item.get("code") or ""),
            "public_code": str(item.get("public_code") or _public_security_code(item.get("code")) or ""),
            "name": str(item.get("name") or item.get("code") or "—"),
            "industry": item.get("industry") or item.get("build_time_industry"),
            # The fixed Grok ledger carries an industry label and an exchange
            # board, but no independently verified thematic-sector mapping.
            # Do not duplicate the industry as a fake "所属板块".
            "sector": item.get("sector") or None,
            "build_time_industry": item.get("build_time_industry"),
            "build_time_industry_status": item.get("build_time_industry_status"),
            "industry_as_of": item.get("industry_as_of"),
            "board": item.get("board"),
            "board_name": item.get("board"),
            "quantity": quantity_value,
            "quantity_known": quantity_known,
            "weight": _finite_number(item.get("weight")),
            "cost": cost,
            "cost_price": cost,
            "price": mark_price,
            "mark_price": mark_price,
            "live_quote": live_price,
            "quote_received": received_public_price is not None and received_public_price > 0,
            "quote_timestamp_verified": price_is_public,
            "current_price": current_price,
            "price_is_public": price_is_public,
            "pre_close": pre_close,
            "open": _finite_number(item.get("open")),
            "high": _finite_number(item.get("high")),
            "low": _finite_number(item.get("low")),
            "volume": _finite_number(item.get("volume")),
            "amount": _finite_number(item.get("amount")),
            "pct_chg": _finite_number(item.get("pct_chg")),
            "day_pct": _finite_number(item.get("pct_chg")),
            "day_pnl": day_pnl,
            "market_value": market_value,
            "mark_value": mark_value,
            "cost_value": cost_value,
            "unrealized_pnl": pnl,
            "unrealized_return": (current_price / cost - 1.0) if current_price is not None and cost and cost > 0 else None,
            "return_since_entry": (current_price / cost - 1.0) if current_price is not None and cost and cost > 0 else None,
            "return_ytd": _finite_number(item.get("return_ytd")),
            "board_pct": _finite_number(item.get("board_pct")),
            "excess_pct": _finite_number(item.get("excess_pct")),
            "leader_today": item.get("leader_today"),
            "leader_today_pct": _finite_number(item.get("leader_today_pct")),
            "leader_ytd": item.get("leader_ytd"),
            "leader_ytd_pct": _finite_number(item.get("leader_ytd_pct")),
            "revenue_forecast": item.get("revenue_forecast"),
            "industry_top3_1m": item.get("industry_top3_1m"),
            "entry_date": item.get("entry_date"),
            "account_as_of": item.get("as_of"),
            "quote_time": item.get("quote_time"),
            "quote_source": item.get("quote_source"),
            "quote_state": quote_state,
            "quote_state_label": _quote_state_label(quote_state),
        })

    normalized.sort(key=lambda row: row.get("market_value") or 0.0, reverse=True)
    market_value = sum(row["market_value"] for row in normalized if row["market_value"] is not None)
    cost_value = sum(row["cost_value"] for row in normalized if row["cost_value"] is not None)
    unrealized_pnl = sum(row["unrealized_pnl"] for row in normalized if row["unrealized_pnl"] is not None)
    day_rows = [row for row in normalized if _portfolio_day_quote_is_eligible(row, now=quote_now)]
    # A portfolio-level "today" figure is meaningful only when every ledger
    # position has both a public last price and a public previous close.  A
    # partial quote batch may still be useful row by row, but presenting its
    # subtotal as the 50-stock account's return would be misleading.
    day_quote_complete = bool(normalized) and len(day_rows) == len(normalized)
    day_pnl_total = sum(row["day_pnl"] for row in day_rows)
    cash = _finite_number(account_summary.get("cash")) or 0.0
    receivables = _finite_number(account_summary.get("receivables")) or 0.0
    nav_live = market_value + cash + receivables
    previous_nav = nav_live - day_pnl_total if day_quote_complete else None
    for row in normalized:
        row["live_weight"] = row["market_value"] / nav_live if row["market_value"] is not None and nav_live > 0 else None
    quote_rows = [row for row in normalized if row["price_is_public"]]
    received_quote_rows = [row for row in normalized if row["quote_received"]]
    quote_state_counts: dict[str, int] = {}
    for row in received_quote_rows:
        state = str(row.get("quote_state") or "unavailable")
        quote_state_counts[state] = quote_state_counts.get(state, 0) + 1
    quote_time_records: list[tuple[int, str]] = []
    for row in quote_rows:
        raw_time = row.get("quote_time")
        try:
            parsed_time = pd.Timestamp(raw_time)
            if pd.isna(parsed_time):
                continue
            if parsed_time.tzinfo is None:
                parsed_time = parsed_time.tz_localize("Asia/Shanghai")
            else:
                parsed_time = parsed_time.tz_convert("Asia/Shanghai")
            quote_time_records.append((int(parsed_time.value), str(raw_time)))
        except (TypeError, ValueError):
            continue
    quote_time_records.sort(key=lambda item: item[0])
    sources = sorted({str(row["quote_source"]) for row in quote_rows if row.get("quote_source")})
    totals = {
        "market_value": market_value,
        "cost_value": cost_value,
        "cash": cash,
        "receivables": receivables,
        "nav_live": nav_live,
        "nav_mark": _finite_number(account_summary.get("nav_mark")) or _finite_number(account_summary.get("nav")),
        "unrealized_pnl": unrealized_pnl,
        "unrealized_return": unrealized_pnl / cost_value if cost_value else None,
        "day_pnl": day_pnl_total if day_quote_complete else None,
        "day_return": day_pnl_total / previous_nav if previous_nav is not None and previous_nav > 0 else None,
        "day_quote_count": len(day_rows),
        "day_quote_complete": day_quote_complete,
        "holdings_count": len(normalized),
        "quote_count": len(quote_rows),
        "quote_received_count": len(received_quote_rows),
        "quote_coverage": len(quote_rows) / len(normalized) if normalized else 0.0,
        "quote_state_counts": quote_state_counts,
        "real_time_quote_count": quote_state_counts.get("trading", 0),
        "delayed_quote_count": quote_state_counts.get("delayed", 0),
        "closing_quote_count": quote_state_counts.get("closed", 0),
        "quote_time": quote_time_records[-1][1] if quote_time_records else None,
        "quote_time_earliest": quote_time_records[0][1] if quote_time_records else None,
        "quote_time_latest": quote_time_records[-1][1] if quote_time_records else None,
        "quote_sources": sources,
        "account_as_of": account_summary.get("account_as_of") or account_summary.get("exec_date"),
    }
    return normalized, totals


def _overlay_paper_quotes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay public quotes on the fixed paper ledger without changing its marks.

    A quote failure leaves the dated ledger mark visible as an explicitly
    non-live fallback.  It must never make the fixed 50-position ledger look
    empty, and it must never turn the mark into a fake tick.
    """

    quote_map: dict[str, dict[str, Any]] = {}
    public_codes = [str(row.get("public_code") or "") for row in rows]
    public_codes = [code for code in public_codes if re.fullmatch(r"\d{6}", code)]
    if public_codes:
        try:
            from quantlab.data.realtime import batch_quotes

            quotes = batch_quotes(public_codes)
            if isinstance(quotes, pd.DataFrame) and not quotes.empty:
                for _, quote in quotes.iterrows():
                    code = str(quote.get("code") or "").zfill(6)
                    quote_map[code] = dict(quote)
        except Exception:  # noqa: BLE001 - public quotes are optional display data.
            quote_map = {}

    result: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        quote = quote_map.get(str(row.get("public_code") or ""), {})
        live_price = _finite_number(quote.get("price"))
        if live_price is not None and live_price <= 0:
            live_price = None
        row.update({
            "live_quote": live_price,
            "quote_time": quote.get("time") or None,
            "quote_source": quote.get("source") or None,
            "pre_close": _finite_number(quote.get("pre_close")),
            "open": _finite_number(quote.get("open")),
            "high": _finite_number(quote.get("high")),
            "low": _finite_number(quote.get("low")),
            "volume": _finite_number(quote.get("volume")),
            "amount": _finite_number(quote.get("amount")),
            "pct_chg": _finite_number(quote.get("pct_chg")),
            # These need a complete, current industry constituent source.  A
            # selected holding can request the two supplemental endpoints; no
            # 50-stock-only ranking is passed off as a sector statistic.
            "board_pct": None,
            "excess_pct": None,
            "leader_today": None,
            "leader_today_pct": None,
            "leader_ytd": None,
            "leader_ytd_pct": None,
            "revenue_forecast": {
                "status": "unavailable",
                "message": "选择持仓后读取机构预测；不会以已披露营收替代。",
            },
            "industry_top3_1m": {
                "status": "unavailable",
                "message": "需要完整行业成分与近20个交易日行情；不会以50只持仓内部排名替代。",
            },
        })
        result.append(row)
    return result


def _portfolio_snapshot(*, include_public_quotes: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Read the independent Grok paper ledger for the account-observation UI.

    This intentionally bypasses QuantLab's research ``latest.json``.  The
    observation page is a read-only projection of the named external paper
    ledger; it is not a research result or a broker account.
    """

    try:
        ledger = load_grok_paper_ledger()
        if not ledger.get("displayable"):
            reason = "; ".join(str(item) for item in ledger.get("errors", [])[:2])
            return [], dict(ledger.get("summary") or {"status": "unavailable"}), reason or "Grok纸账户账本不可用"
        rows = [dict(row) for row in ledger.get("rows", []) if isinstance(row, Mapping)]
        if include_public_quotes:
            rows = _overlay_paper_quotes(rows)
        summary_data = dict(ledger.get("summary") or {})
        summary_data.update({
            "status": PAPER_STATUS,
            "displayable": True,
            "source_kind": ledger.get("source_kind"),
            "source_label": "Grok Bot 50仓纸账户账本",
            "provenance": ledger.get("provenance"),
        })
        return rows, summary_data, None
    except Exception as exc:  # noqa: BLE001 - UI must degrade to a visible state.
        return [], {"status": "unavailable"}, f"持仓读取失败：{type(exc).__name__}"


def _stdlib_eastmoney_kline(code: str, start: str, end: str, market: str | None = None) -> pd.DataFrame | None:
    """Fetch public daily OHLCV without optional third-party HTTP packages."""

    market_id = "1" if (market or _local_market_exchange(code)) == "sh" else "0"
    params = {
        "secid": f"{market_id}.{code}",
        "klt": "101",
        "fqt": "1",
        "beg": str(start).replace("-", ""),
        "end": str(end).replace("-", ""),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params)
    try:
        request_obj = Request(url, headers={
            "Referer": "https://quote.eastmoney.com/",
            "User-Agent": "Mozilla/5.0",
        })
        with urlopen(request_obj, timeout=8) as response:  # noqa: S310 - fixed public endpoint.
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    entries = ((payload.get("data") or {}).get("klines") or []) if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    for entry in entries:
        parts = str(entry).split(",")
        if len(parts) < 7:
            continue
        try:
            records.append({
                "datetime": pd.Timestamp(parts[0]),
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
                "pct_chg": float(parts[8]) / 100.0 if len(parts) > 8 else math.nan,
            })
        except (TypeError, ValueError):
            continue
    if not records:
        return None
    frame = pd.DataFrame(records).set_index("datetime").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.attrs["source"] = "eastmoney_stdlib_qfq"
    return frame


def _parse_tencent_day_bars(payload: object, symbol: str) -> pd.DataFrame | None:
    """Turn one Tencent daily payload into canonical OHLCV.  Index levels stay unadjusted."""

    data = payload.get("data") if isinstance(payload, dict) else None
    node = data.get(symbol) if isinstance(data, dict) else None
    bars = node.get("day") if isinstance(node, dict) else None
    if not isinstance(bars, list):
        return None
    records: list[dict[str, Any]] = []
    for bar in bars:
        if not isinstance(bar, (list, tuple)) or len(bar) < 9 or isinstance(bar[8], dict):
            continue
        try:
            records.append({
                "datetime": pd.Timestamp(str(bar[0])[:10]),
                "open": float(bar[1]),
                "close": float(bar[2]),
                "high": float(bar[3]),
                "low": float(bar[4]),
                "volume": float(bar[5]),
                # Tencent's daily tape reports amount in 万元. Eastmoney reports 元.
                "amount": float(bar[8]) * 10000.0,
            })
        except (TypeError, ValueError):
            continue
    if not records:
        return None
    frame = pd.DataFrame(records).set_index("datetime").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.attrs["source"] = "tencent_daily"
    return frame


def _tencent_daily_frame(code: str, market: str, start: str, end: str) -> pd.DataFrame | None:
    """Read daily bars for one explicit exchange symbol, including indexes."""

    symbol = _tencent_symbol(code, market)
    collected: list[list[Any]] = []
    end_day = pd.Timestamp(end)
    start_day = pd.Timestamp(start)
    for _page in range(8):
        begin = (end_day - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
        url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newkline/newkline?" + urlencode({
            "param": f"{symbol},day,{begin},{end_day.strftime('%Y-%m-%d')},800,",
        })
        request_obj = Request(url, headers={"Referer": "https://gu.qq.com/", "User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request_obj, timeout=8) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            break
        data = payload.get("data") if isinstance(payload, dict) else None
        node = data.get(symbol) if isinstance(data, dict) else None
        bars = node.get("day") if isinstance(node, dict) and isinstance(node.get("day"), list) else []
        if not bars:
            break
        collected = bars + collected
        first = pd.Timestamp(str(bars[0][0])[:10])
        if first <= start_day:
            break
        end_day = first - pd.Timedelta(days=1)
    if not collected:
        return None
    frame = _parse_tencent_day_bars({"data": {symbol: {"day": collected}}}, symbol)
    if frame is None or frame.empty:
        return None
    return frame.loc[(frame.index >= start_day) & (frame.index <= pd.Timestamp(end))]


def _load_public_kline(code: str, start: str, end: str, market: str | None = None) -> tuple[object, str | None, list[str]]:
    """Read one bounded public daily series with package and stdlib fallbacks."""

    frame = None
    source = None
    errors: list[str] = []
    default_market = _local_market_exchange(code)
    named_exchange = market is not None and market != default_market
    if not named_exchange:
        try:
            from quantlab.data.eastmoney import fetch_kline

            frame = fetch_kline(code, start=start, end=end)
            if frame is not None and len(frame):
                source = "eastmoney"
        except Exception as exc:  # noqa: BLE001 - fallback is intentionally bounded.
            errors.append(f"eastmoney:{type(exc).__name__}")
        if frame is None or len(frame) == 0:
            try:
                from quantlab.data.sina import fetch_kline

                frame = fetch_kline(code, start=start, end=end)
                if frame is not None and len(frame):
                    source = "tencent"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"tencent:{type(exc).__name__}")
    if frame is None or len(frame) == 0:
        stdlib_frame = _stdlib_eastmoney_kline(code, start, end, market)
        if stdlib_frame is not None and len(stdlib_frame):
            frame = stdlib_frame
            source = "eastmoney_stdlib_qfq"
        else:
            errors.append("eastmoney_stdlib:unavailable")
    if frame is None or len(frame) == 0:
        resolved = market or ("bj" if code.startswith(("4", "8", "92")) else "sh" if code.startswith(("5", "6", "9")) else "sz")
        tencent_frame = _tencent_daily_frame(code, resolved, start, end)
        if tencent_frame is not None and len(tencent_frame):
            frame, source = tencent_frame, "tencent_daily"
        else:
            errors.append("tencent_daily:unavailable")
    return frame, source, errors


def _public_security_code(value: object) -> str | None:
    """Accept ledger ids while keeping public quote requests six-digit only."""

    identity = _listed_identity(value)
    return identity[0] if identity else None


def _listed_identity(value: object) -> tuple[str, str, bool] | None:
    """Return code, exchange and whether the caller named the exchange.

    ``000001`` without a prefix is the Shenzhen stock. ``sh000001`` is the
    Shanghai index. Collapsing both to six digits would open the wrong tape.
    """

    text = str(value or "").strip().lower()
    prefixed = re.fullmatch(r"(sh|sz|bj)(\d{6})", text) or re.fullmatch(r"(sh|sz|bj)\.(\d{6})", text)
    if prefixed:
        return prefixed.group(2), prefixed.group(1), True
    dotted_code = re.fullmatch(r"(\d{6})\.(sh|sz|bj)", text)
    if dotted_code:
        return dotted_code.group(1), dotted_code.group(2), True
    plain = re.fullmatch(r"(\d{6})", text)
    if plain is None:
        return None
    code = plain.group(1)
    if code.startswith(("4", "8", "92")):
        return code, "bj", False
    if code.startswith(("5", "6", "9")):
        return code, "sh", False
    return code, "sz", False


def _current_shanghai_time() -> datetime:
    """Return a timezone-aware clock for quote freshness decisions."""

    return datetime.now(ZoneInfo("Asia/Shanghai"))


def _quote_state_label(state: object) -> str:
    """Expose a text label as well as a color/state token to the terminal."""

    return {
        "trading": "实时",
        "lunch_break": "午间休市",
        "delayed": "延迟",
        "closed": "收盘",
        "pre_open": "待开盘",
        "ledger_mark": "账本标记价",
        "timestamp_missing": "行情时间缺失",
        "timestamp_invalid": "行情时间无效",
        "unavailable": "暂无行情",
    }.get(str(state or ""), "行情状态未知")


def _quote_state_from_timestamp(value: object, *, now: datetime | None = None) -> str:
    """Classify public quote freshness without calling a stale quote real-time."""

    current = now or _current_shanghai_time()
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "unavailable"
    if pd.isna(parsed):
        return "unavailable"
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("Asia/Shanghai")
    else:
        parsed = parsed.tz_convert("Asia/Shanghai")
    stamp = parsed.to_pydatetime()
    if stamp.date() != current.date():
        return "closed" if stamp.date() < current.date() else "unavailable"
    minute = (current.hour, current.minute)
    if minute < (9, 30):
        return "pre_open"
    # The exchange is closed from 11:30 through 12:59.  A timestamp from the
    # final morning minute is the freshest valid public quote in that window,
    # not a provider outage.  Earlier timestamps remain genuinely delayed.
    if (11, 30) <= minute < (13, 0):
        return "lunch_break" if (stamp.hour, stamp.minute) >= (11, 29) else "delayed"
    if minute > (15, 0):
        return "closed"
    in_session = ((9, 30) <= minute <= (11, 30)) or ((13, 0) <= minute <= (15, 0))
    if not in_session:
        return "delayed"
    age_seconds = (current - stamp).total_seconds()
    if age_seconds < -60:
        return "unavailable"
    return "trading" if age_seconds <= 5 * 60 else "delayed"


def _portfolio_terminal_html() -> str:
    """Return the recovered terminal shell without rewriting its visual code."""

    path = app.static_folder and Path(app.static_folder) / "portfolio_terminal.html"
    if path is None:
        raise FileNotFoundError("portfolio_terminal.html")
    html = path.read_text(encoding="utf-8")
    adapter = '<script src="/static/portfolio_terminal_adapter.js"></script>'
    terminal = '<script src="/static/portfolio_terminal.js"></script>'
    if adapter not in html:
        html = html.replace(terminal, adapter + "\n" + terminal)
    return html


def _paper_nav_feed() -> tuple[dict[str, Any] | None, str | None]:
    """Load and reconcile the fixed Grok paper NAV feed, never ``latest``."""

    ledger = load_grok_paper_ledger()
    summary_data = dict(ledger.get("summary") or {})
    if not ledger.get("displayable"):
        return None, "Grok纸账户账本不可用"
    expected_date = str(summary_data.get("account_as_of") or "")
    expected_nav = _finite_number(summary_data.get("nav_mark"))

    def valid(feed: Mapping[str, Any]) -> dict[str, Any] | None:
        dates = feed.get("dates")
        nav = feed.get("nav")
        if not isinstance(dates, list) or not isinstance(nav, list) or not dates or len(dates) != len(nav):
            return None
        parsed_nav = [_finite_number(value) for value in nav]
        if any(value is None or value <= 0 for value in parsed_nav):
            return None
        if str(dates[-1]) != expected_date or expected_nav is None:
            return None
        if not math.isclose(float(parsed_nav[-1]), expected_nav, rel_tol=0.0, abs_tol=0.02):
            return None
        out = dict(feed)
        out["dates"] = [str(value)[:10] for value in dates]
        out["nav"] = [float(value) for value in parsed_nav]
        return out

    try:
        decoded = json.loads(GROK_PAPER_FEED.read_text(encoding="utf-8"))
        if isinstance(decoded, Mapping) and (feed := valid(decoded)) is not None:
            return feed, None
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    try:
        frame = pd.read_csv(GROK_PAPER_NAV_CSV)
        if "date" not in frame or "nav" not in frame:
            return None, "Grok纸账户净值CSV缺少日期或净值列"
        feed = {
            "dates": [str(value)[:10] for value in frame["date"].tolist()],
            "nav": frame["nav"].tolist(),
            "benchmarks": {
                str(column): frame[column].tolist()
                for column in frame.columns
                if str(column).endswith(("_price", "_tr"))
            },
        }
        nav_values = np.asarray([float(value) for value in feed["nav"]], dtype=float)
        feed["drawdown"] = (nav_values / np.maximum.accumulate(nav_values) - 1.0).tolist()
        checked = valid(feed)
        return (checked, None) if checked is not None else (None, "Grok纸账户净值与50仓账本不一致")
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None, "Grok纸账户净值数据暂不可用"


def _paper_metrics(feed: Mapping[str, Any]) -> dict[str, Any]:
    """Return display metrics while withholding annualised claims for short feeds."""

    nav = np.asarray(feed.get("nav") or [], dtype=float)
    if not len(nav):
        return {"sessions": 0, "annualisable": False, "total_return": None, "annual_return": None,
                "max_drawdown": None, "sharpe": None, "calmar": None}
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    sessions = int(len(nav))
    total_return = float(nav[-1] / nav[0] - 1.0) if nav[0] else None
    annualisable = sessions >= 60
    annual_return = ((float(nav[-1] / nav[0]) ** (252.0 / max(sessions - 1, 1))) - 1.0) if annualisable and nav[0] else None
    returns = pd.Series(nav).pct_change().dropna()
    sharpe = None
    if annualisable and len(returns) > 1 and float(returns.std(ddof=1)) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if annual_return is not None and max_drawdown < 0 else None
    return {
        "sessions": sessions,
        "annualisable": annualisable,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "calmar": calmar,
    }


@app.get("/portfolio")
@app.get("/holdings")
def portfolio_hub():
    """Serve the recovered, isolated 50-position market terminal."""

    try:
        return Response(_portfolio_terminal_html(), mimetype="text/html")
    except OSError:
        return Response("持仓终端页面暂不可用。", status=503, mimetype="text/plain")


@app.get("/api/portfolio/terminal/nav")
def api_portfolio_terminal_nav():
    feed, error = _paper_nav_feed()
    if feed is None:
        return jsonify({"available": False, "status": PAPER_STATUS, "interpretable": False,
                        "blocked_reasons": [error or "纸账户净值不可用"], "last_exec": None,
                        "quote_time": None, "metrics": None, "ytd": None})
    nav = list(feed["nav"])
    return jsonify(_json_safe({"available": True, "status": PAPER_STATUS, "interpretable": True,
                               "last_exec": feed["dates"][-1], "quote_time": None,
                               "metrics": _paper_metrics(feed), "ytd": None,
                               "extra": "固定 Grok Bot 50仓纸账户账本；不代表券商账户或可采用策略。",
                               "nav_mark": nav[-1], "source_kind": "grok_paper_ledger"}))


@app.get("/api/portfolio/terminal/series")
def api_portfolio_terminal_series():
    feed, error = _paper_nav_feed()
    if feed is None:
        return jsonify({"available": False, "status": PAPER_STATUS, "dates": [], "nav": [], "benchmark": None,
                        "drawdown": [], "error": error})
    nav = np.asarray(feed["nav"], dtype=float)
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    benchmarks = feed.get("benchmarks") if isinstance(feed.get("benchmarks"), Mapping) else {}
    selected = None
    for key in ("CSI2000_tr", "CSI1000_tr", "CSI300_tr"):
        values = benchmarks.get(key) if isinstance(benchmarks, Mapping) else None
        if isinstance(values, list) and len(values) == len(nav):
            parsed = [_finite_number(value) for value in values]
            if parsed and parsed[0] not in (None, 0) and all(value is not None for value in parsed):
                selected = [float(value) / float(parsed[0]) * float(nav[0]) for value in parsed]
                break
    return jsonify(_json_safe({"available": True, "status": PAPER_STATUS, "dates": feed["dates"],
                               "nav": nav.tolist(), "benchmark": selected, "drawdown": drawdown.tolist(),
                               "source_kind": "grok_paper_ledger"}))


@app.get("/api/portfolio/terminal/risk")
def api_portfolio_terminal_risk():
    return jsonify({"available": False, "status": PAPER_STATUS,
                    "note": "该纸账户没有可验证的风险策略状态；页面不会从研究运行或历史权重推断。"})


@app.get("/api/portfolio/terminal/signals")
def api_portfolio_terminal_signals():
    return jsonify({"status": PAPER_STATUS, "trades": [], "names": {}, "signal_date": None, "exec_date": None,
                    "note": "纸账户观察页不生成信号、不重算研究、也不创建委托。"})


@app.get("/api/portfolio/terminal/experts")
def api_portfolio_terminal_experts():
    return jsonify({"status": PAPER_STATUS, "experts": {}, "date": None, "n_candidates": 0, "coverage": {},
                    "note": "纸账户观察页不重算专家或生成交易建议。"})


@app.get("/api/portfolio/holdings")
def api_portfolio_holdings():
    include_public_quotes = request.args.get("live") == "1"
    rows, account_summary, error = _portfolio_snapshot(include_public_quotes=include_public_quotes)
    if error:
        return jsonify({
            "available": False,
            "status": "unavailable",
            "account_type": "simulation",
            "simulation_only": True,
            "public_quotes_enabled": include_public_quotes,
            "rows": [],
            "summary": {},
            "as_of": None,
            "caveats": [error, "页面不会把失败状态降级成虚构持仓或盈亏。"],
        }), 200
    normalized, totals = _normalize_portfolio_rows(rows, account_summary)
    status = str(account_summary.get("status") or "blocked_data")
    displayable = bool(account_summary.get("displayable")) or status == "research_unvalidated"
    available = bool(normalized) and displayable
    if not normalized:
        return jsonify({
            "available": False,
            "status": status,
            "account_status": status,
            "legacy_snapshot": False,
            "account_type": "simulation",
            "simulation_only": True,
            "public_quotes_enabled": include_public_quotes,
            "rows": [],
            "summary": _json_safe(account_summary),
            "as_of": account_summary.get("account_as_of") or account_summary.get("exec_date") or None,
            "caveats": [
                "当前没有可用的已核验模拟账户持仓快照。",
                "这不是券商持仓接口；不会从历史回放权重、目录扫描或 latest 指针猜测当前仓位。",
            ],
        }), 200
    return jsonify({
        "available": available,
        "status": "available" if available else status,
        "account_status": status,
        "legacy_snapshot": False,
        "account_type": "simulation",
        "simulation_only": True,
        "public_quotes_enabled": include_public_quotes,
        "source": _json_safe({
            "kind": account_summary.get("source_kind"),
            "label": account_summary.get("source_label"),
            "provenance": account_summary.get("provenance"),
        }),
        "rows": _json_safe(normalized),
        "summary": _json_safe(totals),
        "as_of": totals.get("quote_time") or totals.get("account_as_of"),
        "quote_time": totals.get("quote_time"),
        "today": totals.get("day_return"),
        "ytd": None,
        "caveats": [
            "持仓来自固定 Grok Bot 50仓纸账户账本；公开行情价仅用于估算展示，不代表券商实时资产。",
            "该纸账户与 QuantLab 研究 latest 指针完全分离；未获策略采用或自动执行许可。",
            ("已按本次明确请求读取公开报价；公开源只收到股票代码，不会收到数量、成本或盈亏。"
             if include_public_quotes else "当前为本地账本观察模式；未向公开行情源发送这 50 个股票代码。"),
        ],
    }), 200


def _local_market_exchange(code: str) -> str:
    if code.startswith(("4", "8", "92")):
        return "bj"
    return "sh" if code.startswith(("5", "6", "9")) else "sz"


def _load_local_market_daily(code: str) -> pd.DataFrame | None:
    """Read exactly one named frozen panel; never scan an arbitrary cache."""

    path = LOCAL_MARKET_ROOT / "panels" / f"{_local_market_exchange(code)}.{code}.parquet"
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, ImportError):
        return None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if "date" in frame.columns:
        frame = frame.set_index("date")
    elif "trade_date" in frame.columns:
        frame = frame.set_index("trade_date")
    aliases = {"vol": "volume", "turnover": "amount", "money": "amount"}
    frame = frame.rename(columns={source: target for source, target in aliases.items()
                                  if source in frame.columns and target not in frame.columns})
    columns = ["open", "high", "low", "close", "volume", "amount"]
    return frame.loc[:, columns].copy() if all(column in frame.columns for column in columns) else None


def _bar_rows(frame: pd.DataFrame, period: str, limit: int) -> list[dict[str, Any]]:
    try:
        source = resample_ohlcv(frame, period)
    except (TypeError, ValueError):
        return []
    if limit > 0:
        source = source.tail(limit)
    rows: list[dict[str, Any]] = []
    for index, row in source.iterrows():
        values = {key: _finite_number(row.get(key)) for key in ("open", "close", "low", "high", "volume")}
        if any(value is None for value in values.values()):
            continue
        date_value = index.date().isoformat() if hasattr(index, "date") else str(index)[:10]
        rows.append({"d": date_value, "o": values["open"], "c": values["close"], "l": values["low"],
                     "h": values["high"], "v": values["volume"], "a": _finite_number(row.get("amount"))})
    return rows


def _public_kline_adjustment(source: object) -> str:
    """Return only a price-adjustment label we can prove from the request path."""

    if source == "eastmoney":
        mode = str(CONFIG.data.a_daily_adjust or "").strip().lower()
        return "raw" if not mode else "hfq" if mode == "hfq" else "qfq"
    if source == "eastmoney_stdlib_qfq":
        return "qfq"
    if source == "tencent_daily":
        return "none"
    # Tencent's wrapper accepts a configurable adjustment argument, but its
    # public response does not preserve which variant was returned.  Withhold
    # the label rather than asserting a price basis we cannot audit here.
    return "source_unverified"


def _terminal_kline(code: str, period: str, limit: int, market: str | None = None) -> dict[str, Any]:
    end_date = datetime.now().date()
    days = 365 * 15 if limit == 0 else max(90, limit * 3)
    frame, source, errors = _load_public_kline(code, (end_date - timedelta(days=days)).isoformat(), end_date.isoformat(), market)
    bars = _bar_rows(frame, period, limit) if isinstance(frame, pd.DataFrame) and not frame.empty else []
    status, source_kind, as_of = "public_quote_unverified", "public_adapter", None
    adjustment = _public_kline_adjustment(source)
    default_market = _local_market_exchange(code)
    if not bars and market not in (None, default_market):
        return {"available": False, "status": "unavailable", "code": code, "frequency": period,
                "bars": [], "candles": [], "source": source, "source_kind": source_kind,
                "adjust": adjustment, "as_of": None, "quote_state": "unavailable",
                "quote_state_label": _quote_state_label("unavailable"),
                "error": "暂时拿不到该指数的公开K线数据。",
                "caveats": ["指数行情不会用同代码的股票面板代替。", *errors[:2]]}
    if not bars:
        fallback, metadata = local_panel_fallback(_load_local_market_daily(code), period=period,
                                                   cutoff=LOCAL_MARKET_CUTOFF,
                                                   source="local_market_1c85fd9aceaa03c3c4863bec")
        if period == "day":
            bars = _bar_rows(fallback, "day", limit)
        else:
            view = fallback.tail(limit) if limit > 0 else fallback
            bars = [{"d": index.date().isoformat() if hasattr(index, "date") else str(index)[:10],
                     "o": _finite_number(row.get("open")), "c": _finite_number(row.get("close")),
                     "l": _finite_number(row.get("low")), "h": _finite_number(row.get("high")),
                     "v": _finite_number(row.get("volume")), "a": _finite_number(row.get("amount"))} for index, row in view.iterrows()]
            bars = [row for row in bars if all(row[key] is not None for key in ("o", "c", "l", "h", "v"))]
        if bars:
            source = metadata.get("source")
            status = str(metadata.get("status") or "local_panel_unverified")
            source_kind = str(metadata.get("source_kind") or "local_panel")
            adjustment, as_of = "none", metadata.get("as_of")
            # ``resample_ohlcv`` deliberately labels a monthly/annual bucket at
            # its calendar end.  That is useful for complete public history,
            # but a bounded local fallback must not make its final incomplete
            # bucket look like data from the future.  Preserve the aggregate
            # values and change only this terminal display label to its known
            # final observation date.
            if period != "day" and as_of:
                terminal_label = pd.to_datetime(bars[-1]["d"], errors="coerce")
                known_as_of = pd.to_datetime(as_of, errors="coerce")
                if not pd.isna(terminal_label) and not pd.isna(known_as_of) and terminal_label > known_as_of:
                    bars[-1]["d"] = known_as_of.date().isoformat()
        else:
            return {"available": False, "status": "unavailable", "code": code, "frequency": period,
                    "bars": [], "candles": [], "source": source, "source_kind": source_kind,
                    "adjust": adjustment, "as_of": None, "quote_state": "unavailable",
                    "quote_state_label": _quote_state_label("unavailable"),
                    "error": "暂时拿不到该证券的公开K线数据。",
                    "caveats": ["公开行情失败时只会使用明确截止日的本地历史面板；不会合成K线。", *errors[:2]]}
    as_of = as_of or bars[-1]["d"]
    pre_close, last = (bars[-2]["c"] if len(bars) > 1 else None), bars[-1]["c"]
    return {"available": True, "status": status, "code": code, "frequency": period, "bars": bars,
            "candles": [{"date": x["d"], "open": x["o"], "high": x["h"], "low": x["l"], "close": x["c"],
                         "volume": x["v"], "amount": None, "pct_chg": None} for x in bars],
            "source": source, "source_kind": source_kind, "adjust": adjustment,
            "adjustment": adjustment, "as_of": as_of,
            "quote_state": "closed", "quote_state_label": _quote_state_label("closed"), "quote_time": None,
            "quote_name": code, "last": last, "pre_close": pre_close,
            "chg": last / pre_close - 1.0 if pre_close not in (None, 0) else None,
            "caveats": ["K线只用于行情观察，不改变纸账户账本、研究状态或交易。"]}


@app.get("/api/portfolio/kline")
def api_portfolio_kline():
    identity = _listed_identity(request.args.get("code"))
    if identity is None:
        return jsonify({"available": False, "status": "invalid_code", "candles": [], "bars": [], "error": "请输入有效的证券代码。"}), 400
    code, market, explicit = identity
    market = market if explicit else None
    requested_period = request.args.get("freq")
    if requested_period is not None:
        try:
            period = validate_period(str(requested_period))
        except ValueError as exc:
            return jsonify({"available": False, "status": "invalid_period", "bars": [], "candles": [], "error": str(exc)}), 400
        try:
            limit = int(str(request.args.get("bars") or "120"))
        except ValueError:
            return jsonify({"available": False, "status": "invalid_bars", "bars": [], "candles": []}), 400
        if limit not in (0, 120, 250, 750):
            return jsonify({"available": False, "status": "invalid_bars", "bars": [], "candles": []}), 400
        return jsonify(_json_safe(_terminal_kline(code, period, limit, market)))

    range_key = str(request.args.get("range") or "6m").lower()
    days_by_range = {"1m": 45, "3m": 140, "6m": 280, "1y": 420, "all": 365 * 6}
    end = datetime.now().date().isoformat()
    start = (datetime.now().date() - timedelta(days=days_by_range.get(range_key, days_by_range["6m"]))).isoformat()
    frame, source, errors = _load_public_kline(code, start, end, market)
    if frame is None or len(frame) == 0:
        return jsonify({"available": False, "status": "unavailable", "code": code, "range": range_key, "candles": [],
                        "source": source, "error": "暂时拿不到该证券的公开日线数据。",
                        "caveats": ["公开行情接口失败或无数据，不用空白K线替代。", *errors[:2]]}), 200
    candles = []
    for index, row in frame.iterrows():
        candles.append({"date": index.date().isoformat() if hasattr(index, "date") else str(index)[:10],
                        "open": _finite_number(row.get("open")), "high": _finite_number(row.get("high")),
                        "low": _finite_number(row.get("low")), "close": _finite_number(row.get("close")),
                        "volume": _finite_number(row.get("volume")), "amount": _finite_number(row.get("amount")),
                        "pct_chg": _finite_number(row.get("pct_chg"))})
    candles = [x for x in candles if all(x[key] is not None for key in ("open", "high", "low", "close"))]
    return jsonify(_json_safe({"available": bool(candles), "status": "public_quote_unverified" if candles else "unavailable",
                               "code": code, "range": range_key, "candles": candles, "source": source,
                               "adjustment": "public_adapter_default", "as_of": candles[-1]["date"] if candles else None,
                               "caveats": ["公开日线仅用于行情观察，不改变研究账本或产生交易信号。"]}))


def _tencent_symbol(code: str, market: str | None = None) -> str:
    if market in {"sh", "sz", "bj"}:
        return f"{market}{code}"
    if code.startswith(("4", "8", "92")):
        return f"bj{code}"
    return f"sh{code}" if code.startswith(("5", "6", "9")) else f"sz{code}"


def _fetch_tencent_intraday(code: str, market: str | None = None) -> dict[str, Any]:
    """Read one Tencent minute tape; no daily-bar or ledger substitution."""

    symbol = _tencent_symbol(code, market)
    request_obj = Request("https://web.ifzq.gtimg.cn/appstock/app/minute/query?" + urlencode({"code": symbol}),
                          headers={"Referer": "https://gu.qq.com/", "User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request_obj, timeout=8) as response:  # noqa: S310
            decoded = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return {"rows": [], "pre_close": None, "last": None, "quote_time": None}
    data = decoded.get("data") if isinstance(decoded, Mapping) else {}
    item = data.get(symbol) if isinstance(data, Mapping) else None
    if not isinstance(item, Mapping):
        return {"rows": [], "pre_close": None, "last": None, "quote_time": None}
    source = item.get("data") if isinstance(item.get("data"), Mapping) else item
    raw_rows = source.get("data") if isinstance(source, Mapping) else None
    date_value = str(source.get("date") or item.get("date") or "") if isinstance(source, Mapping) else ""
    qt = item.get("qt") if isinstance(item.get("qt"), Mapping) else {}
    qt_row = qt.get(symbol) if isinstance(qt, Mapping) else None
    qt_pre_close = qt_row[4] if isinstance(qt_row, (list, tuple)) and len(qt_row) > 4 else None
    # The minute tape itself often omits preClose, while the companion qt row
    # carries the provider's actual previous close.  Without it, the UI would
    # display a real price but hide the day's percent change as unknown.
    pre_close = _finite_number(
        (source.get("preClose") if isinstance(source, Mapping) else None)
        or item.get("preClose")
        or qt_pre_close
    )
    if not isinstance(raw_rows, list):
        quote_name = item.get("name") or (qt_row[1] if isinstance(qt_row, (list, tuple)) and len(qt_row) > 1 else None)
        return {"rows": [], "pre_close": pre_close, "last": None, "quote_time": None, "quote_name": quote_name}
    rows = []
    for raw in raw_rows:
        parts = str(raw).split()
        if len(parts) < 2 or (price := _finite_number(parts[1])) is None:
            continue
        compact = parts[0].replace(":", "")
        timestamp = (f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]} {compact[:2]}:{compact[2:]}"
                     if re.fullmatch(r"\d{4}", compact) and re.fullmatch(r"\d{8}", date_value) else str(parts[0]))
        rows.append({"t": timestamp, "p": price, "v": _finite_number(parts[2]) if len(parts) > 2 else 0.0, "pre": pre_close})
    quote_name = item.get("name") or (qt_row[1] if isinstance(qt_row, (list, tuple)) and len(qt_row) > 1 else None)
    return {"rows": rows, "pre_close": pre_close, "last": rows[-1]["p"] if rows else None,
            "quote_time": rows[-1]["t"] if rows else None, "quote_name": quote_name}


def _fetch_tencent_indices() -> list[dict[str, Any]]:
    try:
        from quantlab.data.realtime import batch_index_quotes
        quotes = batch_index_quotes()
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(quotes, pd.DataFrame) or quotes.empty:
        return []
    labels = {"000001": "上证指数", "399001": "深证成指", "399006": "创业板指"}
    rows = []
    for code, label in labels.items():
        record = quotes.loc[code].to_dict() if code in quotes.index else {}
        price, pre_close = _finite_number(record.get("price")), _finite_number(record.get("pre_close"))
        rows.append({"name": label, "code": code, "price": price, "pre_close": pre_close,
                     "chg": price - pre_close if price is not None and pre_close is not None else None,
                     "pct_chg": price / pre_close - 1.0 if price is not None and pre_close not in (None, 0) else None,
                     "high": _finite_number(record.get("high")), "low": _finite_number(record.get("low")),
                     "amount_yi": (_finite_number(record.get("amount")) or 0.0) / 100_000_000.0 if record.get("amount") is not None else None,
                     "source": record.get("source") or "unavailable",
                     "quote_time": record.get("time") or None,
                     "state": _quote_state_from_timestamp(record.get("time")) if record else "unavailable"})
    return rows


def _fetch_ths_revenue_html(code: str) -> str | None:
    request_obj = Request(f"https://basic.10jqka.com.cn/new/{code}/worth.html",
                          headers={"Referer": f"https://basic.10jqka.com.cn/{code}/", "User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request_obj, timeout=8) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError):
        return None


@app.get("/api/portfolio/intraday")
def api_portfolio_intraday():
    identity = _listed_identity(request.args.get("code"))
    if identity is None:
        return jsonify({"available": False, "status": "invalid_code", "rows": [], "error": "请输入有效的证券代码。"}), 400
    code, market, explicit = identity
    result = _fetch_tencent_intraday(code, market if explicit else None)
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    quote_time = result.get("quote_time")
    state = _quote_state_from_timestamp(quote_time) if quote_time else "unavailable"
    pre_close, last = _finite_number(result.get("pre_close")), _finite_number(result.get("last"))
    return jsonify(_json_safe({"available": bool(rows), "status": "public_quote_unverified" if rows else "unavailable",
                               "code": code, "rows": rows, "pre_close": pre_close, "last": last,
                               "chg": last / pre_close - 1.0 if last is not None and pre_close not in (None, 0) else None,
                               "quote_time": quote_time, "quote_name": result.get("quote_name") or code,
                               "quote_state": state, "quote_state_label": _quote_state_label(state),
                               "source": "tencent_minute",
                               "caveats": ["分时失败时返回空数据；不会用日K、账本标记价或插值冒充分时。"]}))


@app.get("/api/portfolio/terminal/indices")
def api_portfolio_terminal_indices():
    rows = _fetch_tencent_indices()
    return jsonify(_json_safe({"available": bool(rows), "rows": rows, "breadth": {}, "source": "public_quote_unverified",
                               "caveats": ["指数行情仅作公开市场观察，不参与研究或交易。"]}))


@app.get("/api/portfolio/revenue-forecast")
def api_portfolio_revenue_forecast():
    code = _public_security_code(request.args.get("code"))
    if code is None:
        return jsonify({"available": False, "revenue_forecast": {"status": "invalid_code", "forecasts": []}}), 400
    html = _fetch_ths_revenue_html(code)
    forecast = (parse_ths_revenue_forecast(html) if html is not None else {
        "available": False, "status": "request_error", "source": "ths_worth_html", "metric": "营业收入(元)",
        "basis": "analyst_consensus_forecast_average", "forecasts": [],
        "message": "机构预测页面暂时无法读取；不会以已披露营收替代。"})
    return jsonify(_json_safe({"available": bool(forecast.get("available")), "code": code, "revenue_forecast": forecast}))


@app.get("/api/portfolio/search")
def api_portfolio_search():
    """Look up any A-share by code, name or pinyin.  This does not create a position."""

    query = request.args.get("q") or ""
    rows = search_a_shares(query)
    return jsonify(_json_safe({
        "available": bool(rows),
        "query": " ".join(str(query).split()),
        "rows": rows,
        "message": None if rows or not str(query).strip() else "没有匹配的 A 股。",
    }))


@app.get("/api/portfolio/industry-top3")
def api_portfolio_industry_top3():
    code = _public_security_code(request.args.get("code"))
    if code is None:
        return jsonify({"available": False, "industry_top3_1m": {"status": "invalid_code", "top": []}}), 400
    return jsonify(_json_safe(cached_observe(code)))


def main(port: int = 8899) -> None:
    """Keep the CLI entry point stable while serving the read-only hub."""

    app.run(host="127.0.0.1", port=port, debug=False)
