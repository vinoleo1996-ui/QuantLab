/*
 * QuantLab research hub
 *
 * This file deliberately owns only the /research view.  It neither reads the
 * legacy dashboard DOM nor invokes any trading, refresh, or research command.
 * Every visible conclusion is supplied by a read-only research projection.
 */
(() => {
  "use strict";

  const mount = document.querySelector("[data-research-hub]");
  if (!mount) return;

  const ENDPOINTS = Object.freeze({
    decision: "/api/research/decision",
    candidates: "/api/research/candidates",
    evidence: "/api/research/evidence",
    questions: "/api/research/open-questions",
  });
  const THEME_KEY = "quantlab.research-hub.theme";
  const state = {
    controller: null,
    records: {},
    errors: {},
    loading: false,
    theme: resolveTheme(),
  };

  const ICONS = Object.freeze({
    refresh: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8.2 8.2 0 0 0-15.3-3.9L3 9m0 0V5m0 4h4m-3 4a8.2 8.2 0 0 0 15.3 3.9L21 15m0 0v4m0-4h-4"/></svg>',
    sun: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.5"/><path d="M12 2v2.2M12 19.8V22M4.9 4.9l1.55 1.55M17.55 17.55l1.55 1.55M2 12h2.2M19.8 12H22M4.9 19.1l1.55-1.55M17.55 6.45l1.55-1.55"/></svg>',
    moon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.8 15.4A8.8 8.8 0 0 1 8.6 3.2 8.8 8.8 0 1 0 20.8 15.4Z"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>',
    external: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5"/></svg>',
    info: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 10v6M12 7.25v.1"/></svg>',
  });

  const STATUS = Object.freeze({
    adopted: { label: "可采用", tone: "positive" },
    approved: { label: "可采用", tone: "positive" },
    passed: { label: "通过", tone: "positive" },
    pass: { label: "通过", tone: "positive" },
    ready: { label: "可采用", tone: "positive" },
    blocked: { label: "不可采用", tone: "negative" },
    blocked_data: { label: "数据阻断", tone: "negative" },
    rejected: { label: "不可采用", tone: "negative" },
    not_adopted: { label: "不可采用", tone: "negative" },
    not_adoptable: { label: "不可采用", tone: "negative" },
    not_approved: { label: "未获采用", tone: "negative" },
    failed: { label: "未通过", tone: "negative" },
    fail: { label: "未通过", tone: "negative" },
    unverified: { label: "未验证", tone: "caution" },
    research_unvalidated: { label: "研究未验证", tone: "caution" },
    unverified_data_gaps: { label: "数据缺口未验证", tone: "caution" },
    conditional_research: { label: "条件性研究", tone: "caution" },
    same_sample_descriptive_only: { label: "同样本描述", tone: "caution" },
    evidence_unavailable: { label: "证据不可用", tone: "negative" },
    recorded_decision: { label: "已记录裁决", tone: "caution" },
    available: { label: "已纳入", tone: "neutral" },
    missing: { label: "证据缺失", tone: "negative" },
    invalid: { label: "证据无效", tone: "negative" },
    pending: { label: "待验证", tone: "pending" },
    in_progress: { label: "进行中", tone: "pending" },
    unknown: { label: "状态待核", tone: "neutral" },
  });

  const EVIDENCE_LABELS = Object.freeze({
    conditional_or_missing_evidence: "条件性结果或缺少必要证据",
    blocked_or_unverified_data_acceptance: "数据验收尚未通过",
    unverified_execution_evidence: "执行压力证据尚未验证",
    unverified_forward_validation: "冻结前向记录尚未验证",
    promotion_forced_false: "生产授权被明确阻断",
    unverified_open_question: "尚待验证的问题",
    conditional_research_only: "条件性研究，不能采用",
    conditional_research_closed: "已关闭的条件性研究",
    conditional_historical_account_result: "条件性历史账户结果",
    conditional_posthoc_not_account_replay: "条件性事后归因",
    conditional_diagnostic_not_portfolio_alpha: "条件性诊断，不是组合 alpha",
    conditional_research_not_adopted: "条件性研究，未获采用",
    conditional_research_no_promotion: "条件性研究，未获提升",
    conditional_comparison_not_adoption: "条件性比较，不构成采用",
    conditional_account_evidence_strategy_unchanged: "条件性账户证据，策略未变更",
    conditional_closed_research_decision: "已关闭的条件性研究裁定",
    provenance_only_not_live_market_data: "数据身份来源，不是实时行情",
    research_ledger_not_adoption_approval: "研究台账，不构成采用批准",
    unavailable_evidence: "证据不可用",
  });

  const GATE_NAMES = Object.freeze({
    data: "数据与 PIT",
    pit: "数据与 PIT",
    data_pit: "数据与 PIT",
    corporate_actions: "权益事件",
    corporate_action: "权益事件",
    actions: "权益事件",
    execution: "执行压力",
    costs: "成本与执行",
    cost: "成本与执行",
    risk: "风险预算",
    drawdown: "风险预算",
    forward: "冻结前向",
    forward_validation: "冻结前向",
    oos: "冻结前向",
    validation: "独立验证",
    independent_validation: "独立验证",
  });

  function resolveTheme() {
    try {
      const stored = window.localStorage.getItem(THEME_KEY);
      if (stored === "light" || stored === "dark") return stored;
    } catch (_) {
      // Storage is optional; a private browser context may reject it.
    }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function isMeaningful(value) {
    return value !== undefined && value !== null && value !== "";
  }

  function asArray(value) {
    if (!isMeaningful(value)) return [];
    return Array.isArray(value) ? value.filter(isMeaningful) : [value];
  }

  function readAny(object, keys) {
    if (!isObject(object)) return undefined;
    for (const key of keys) {
      if (isMeaningful(object[key])) return object[key];
    }
    return undefined;
  }

  function recordFrom(payload) {
    const outer = isObject(payload) ? payload : { value: payload };
    const nested = isObject(outer.data) && ["value", "status", "provenance", "as_of", "caveats", "drilldown"].some(
      (key) => Object.prototype.hasOwnProperty.call(outer.data, key),
    )
      ? outer.data
      : outer;
    const value = Object.prototype.hasOwnProperty.call(nested, "value") ? nested.value : nested;
    return {
      value,
      status: readAny(nested, ["status", "state"]) ?? readAny(outer, ["status", "state"]) ?? "unknown",
      provenance: readAny(nested, ["provenance", "source", "sources"]) ?? readAny(outer, ["provenance", "source", "sources"]),
      asOf: readAny(nested, ["as_of", "asOf", "updated_at", "timestamp", "date"]) ?? readAny(outer, ["as_of", "asOf", "updated_at", "timestamp", "date"]),
      caveats: readAny(nested, ["caveats", "limitations", "warnings"]) ?? readAny(outer, ["caveats", "limitations", "warnings"]),
      drilldown: readAny(nested, ["drilldown", "details", "detail"]) ?? readAny(outer, ["drilldown", "details", "detail"]),
    };
  }

  function statusInfo(raw, fallback = "unknown") {
    if (raw === true) return { key: "approved", ...STATUS.approved };
    if (raw === false) return { key: "not_approved", ...STATUS.not_approved };
    const normalized = String(raw ?? fallback)
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, "_");
    if (STATUS[normalized]) return { key: normalized, ...STATUS[normalized] };
    if (/(block|reject|fail|did_not_?pass|not_?pass|not_?adopt|denied)/.test(normalized)) return { key: normalized, label: "不可采用", tone: "negative" };
    if (/(pass|approve|adopt|ready)/.test(normalized)) return { key: normalized, label: "通过", tone: "positive" };
    if (/(pending|progress|review|wait)/.test(normalized)) return { key: normalized, label: "待验证", tone: "pending" };
    if (/(unverified|unknown|gap|incomplete)/.test(normalized)) return { key: normalized, label: "未验证", tone: "caution" };
    if (/(conditional|descriptive)/.test(normalized)) return { key: normalized, label: "条件性研究", tone: "caution" };
    return { key: normalized, label: String(raw || STATUS.unknown.label), tone: "neutral" };
  }

  function displayStatus(raw) {
    const info = statusInfo(raw);
    return `<span class="rh-status rh-status--${info.tone}"><span class="rh-status__dot" aria-hidden="true"></span>${escapeHtml(info.label)}</span>`;
  }

  function textValue(value, fallback = "—") {
    if (!isMeaningful(value)) return fallback;
    if (typeof value === "string") return value;
    if (typeof value === "number") return Number.isFinite(value) ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 4 }).format(value) : fallback;
    if (typeof value === "boolean") return value ? "是" : "否";
    if (Array.isArray(value)) return value.map((item) => textValue(item, "")).filter(Boolean).join("；") || fallback;
    if (isObject(value)) {
      const selected = readAny(value, ["label", "title", "name", "summary", "message", "reason", "value"]);
      if (isMeaningful(selected) && selected !== value) return textValue(selected, fallback);
      return Object.entries(value)
        .filter(([, item]) => typeof item !== "object" && isMeaningful(item))
        .slice(0, 4)
        .map(([key, item]) => `${humanize(key)}：${textValue(item, "")}`)
        .join("；") || fallback;
    }
    return String(value);
  }

  function humanize(key) {
    const map = {
      as_of: "截至",
      available_at: "可用时点",
      summary: "摘要",
      status: "状态",
      outcome: "门禁结果",
      source: "来源",
      provenance: "来源",
      caveats: "限制",
      decision: "裁决",
      next_safe_action: "下一安全动作",
      evidence_grade: "证据等级",
      candidate: "候选",
      net_cagr: "扣费年化",
      median_cagr: "中位年化",
      cost2_cagr: "成本翻倍年化",
      max_drawdown: "最大回撤",
      cagr: "扣费年化",
      observations: "观测日",
      start: "起始",
      end: "截至",
      account_runs: "账户回放数",
      source_artifact: "来源产物",
      source_path: "来源路径",
      promotion_approved: "采用授权",
    };
    return map[key] || String(key).replace(/[_-]+/g, " ");
  }

  function evidenceLabel(value) {
    const key = String(value ?? "").trim();
    return EVIDENCE_LABELS[key] || textValue(value, "未提供证据说明。");
  }

  function provenanceLabel(record, artifactId) {
    const match = asArray(record?.provenance).find((entry) => isObject(entry) && entry.id === artifactId);
    return match?.label ?? artifactId;
  }

  function simpleEntries(value, preferredKeys = []) {
    if (!isMeaningful(value)) return [];
    if (Array.isArray(value)) return value.map((item, index) => ({ key: String(index), value: item }));
    if (!isObject(value)) return [{ key: "", value }];
    for (const key of preferredKeys) {
      if (Array.isArray(value[key])) return value[key].map((item, index) => ({ key: String(index), value: item }));
    }
    return Object.entries(value).map(([key, item]) => ({ key, value: item }));
  }

  function listText(value) {
    return asArray(value)
      .map((item) => textValue(item, ""))
      .filter(Boolean);
  }

  function detailBlock(value, summary = "查看依据与限制") {
    if (!isMeaningful(value)) return "";
    const rows = detailRows(value);
    if (!rows) return "";
    return `<details class="rh-details"><summary>${escapeHtml(summary)}${ICONS.arrow}</summary><div class="rh-details__body">${rows}</div></details>`;
  }

  function detailRows(value) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      return `<p>${escapeHtml(textValue(value))}</p>`;
    }
    if (Array.isArray(value)) {
      const items = value.map((item) => `<li>${escapeHtml(textValue(item))}</li>`).join("");
      return items ? `<ul>${items}</ul>` : "";
    }
    if (isObject(value)) {
      const rows = Object.entries(value)
        .filter(([, item]) => isMeaningful(item))
        .map(([key, item]) => `<div class="rh-detail-row"><dt>${escapeHtml(humanize(key))}</dt><dd>${escapeHtml(textValue(item))}</dd></div>`)
        .join("");
      return rows ? `<dl>${rows}</dl>` : "";
    }
    return "";
  }

  function provenanceBlock(record) {
    const sources = listText(record.provenance);
    const asOf = textValue(record.asOf, "未提供");
    const caveats = listText(record.caveats);
    const parts = [];
    parts.push(`<span><b>截至</b>${escapeHtml(asOf)}</span>`);
    if (sources.length) parts.push(`<span><b>来源</b>${escapeHtml(sources.slice(0, 3).join(" · "))}</span>`);
    if (caveats.length) parts.push(`<span><b>限制</b>${escapeHtml(caveats.slice(0, 2).join("；"))}</span>`);
    return `<div class="rh-provenance">${parts.join("")}</div>`;
  }

  function decisionData(record) {
    const source = isObject(record?.value) ? record.value : { summary: record?.value };
    const status = readAny(source, ["adoption_status", "adoption", "decision_status", "status", "state"])
      ?? (typeof source.promotion_approved === "boolean" ? source.promotion_approved : undefined)
      ?? record?.status;
    const title = readAny(source, ["headline", "title", "adoption_status", "status"]) ?? statusInfo(status).label;
    const summary = readAny(source, ["summary", "message", "reason", "rationale", "description", "conclusion", "decision"]) ?? "尚未提供研究裁决摘要。";
    const safeAction = readAny(source, ["next_safe_action", "next_action", "safe_action", "recommended_next_step"]);
    const gates = readAny(source, ["gates", "gate_matrix", "adoption_gates", "checks"]);
    return { source, status, title, summary, safeAction, gates };
  }

  function normalizeGate(entry) {
    const item = isObject(entry.value) ? entry.value : { summary: entry.value };
    const rawKey = entry.key || readAny(item, ["key", "id", "gate", "name", "label"]);
    const label = readAny(item, ["label", "name", "title", "gate"]) || GATE_NAMES[String(rawKey || "").toLowerCase()] || humanize(rawKey || "采用门禁");
    const status = readAny(item, ["outcome", "status", "state", "result"]) ?? (typeof item.passed === "boolean" ? (item.passed ? "passed" : "blocked") : "unknown");
    const reason = readAny(item, ["summary", "message", "reason", "rationale", "detail", "note", "value"])
      ?? evidenceLabel(item.evidence_label);
    return { label, status, reason, drilldown: readAny(item, ["drilldown", "details", "evidence"]) ?? item, raw: item };
  }

  function renderGateMatrix(record) {
    const data = decisionData(record);
    let gates = simpleEntries(data.gates, ["items", "gates", "checks"]).map(normalizeGate);
    if (!gates.length) {
      gates = listText(record.caveats).map((caveat, index) => ({
        label: index === 0 ? "当前限制" : `限制 ${index + 1}`,
        status: record.status,
        reason: caveat,
        drilldown: null,
      }));
    }
    if (!gates.length) {
      return panelEmpty("尚未提供采用门禁", "等待研究投影提供门禁状态、依据和更新时间。", "gate-empty");
    }
    return `<div class="rh-gate-grid">${gates.map((gate) => {
      const info = statusInfo(gate.status);
      return `<article class="rh-gate rh-gate--${info.tone}">
        <div class="rh-gate__head"><h3>${escapeHtml(gate.label)}</h3>${displayStatus(gate.status)}</div>
        <p>${escapeHtml(textValue(gate.reason))}</p>
        ${detailBlock(gate.drilldown, "查看门禁依据")}
      </article>`;
    }).join("")}</div>`;
  }

  function candidateData(record) {
    const value = record?.value;
    if (Array.isArray(value)) return value;
    if (!isObject(value)) return [];
    const direct = readAny(value, ["candidates", "items", "results", "comparisons", "rows"]);
    return Array.isArray(direct) ? direct : simpleEntries(direct).map(({ key, value: item }) => isObject(item) ? { name: key, ...item } : { name: key, summary: item });
  }

  function candidateName(item, index) {
    return textValue(readAny(item, ["name", "label", "title", "candidate", "id", "strategy", "variant"]), `候选 ${index + 1}`);
  }

  function candidateSummary(item) {
    return textValue(readAny(item, ["summary", "headline", "decision", "message", "result", "conclusion"]), "未提供对照摘要。");
  }

  function candidateMetrics(item) {
    const metrics = readAny(item, ["metrics", "comparison", "highlights", "values"]);
    if (!isObject(metrics)) return "";
    const visible = Object.entries(metrics)
      .filter(([, value]) => typeof value !== "object" && isMeaningful(value))
      .slice(0, 4);
    if (!visible.length) return "";
    return `<dl class="rh-metrics">${visible.map(([key, value]) => `<div><dt>${escapeHtml(humanize(key))}</dt><dd>${escapeHtml(metricValue(key, value))}</dd></div>`).join("")}</dl>`;
  }

  function metricValue(key, value) {
    const percentageKeys = new Set(["net_cagr", "median_cagr", "cost2_cagr", "max_drawdown", "cagr"]);
    if (percentageKeys.has(key) && typeof value === "number" && Number.isFinite(value)) {
      return `${new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value * 100)}%`;
    }
    return textValue(value);
  }

  function renderCandidates(record, error) {
    if (!record) return panelError("候选对照暂不可读", error);
    const candidates = candidateData(record);
    if (!candidates.length) return panelEmpty("暂无可比较候选", "这不等于没有候选；当前投影未提供可展示的同口径对照。", "candidate-empty");
    return `<div class="rh-candidate-grid">${candidates.map((candidate, index) => {
      const item = isObject(candidate) ? candidate : { summary: candidate };
      const status = readAny(item, ["recorded_status", "status", "state", "adoption_status", "decision", "recorded_decision"]) ?? record.status;
      const asOf = readAny(item, ["as_of", "asOf", "updated_at", "date"]) ?? record.asOf;
      const candidateSources = [provenanceLabel(record, item.source_artifact), item.source_path].filter(isMeaningful);
      const source = readAny(item, ["provenance", "source", "sources"]) ?? (candidateSources.length ? candidateSources : record.provenance);
      const caveat = readAny(item, ["caveats", "limitations", "warning"]) ?? record.caveats;
      const drilldown = readAny(item, ["drilldown", "details", "evidence"]);
      const candidateRecord = { provenance: source, asOf, caveats: caveat };
      return `<article class="rh-candidate">
        <div class="rh-candidate__top"><span class="rh-index">${String(index + 1).padStart(2, "0")}</span>${displayStatus(status)}</div>
        <h3>${escapeHtml(candidateName(item, index))}</h3>
        <p>${escapeHtml(candidateSummary(item))}</p>
        ${candidateMetrics(item)}
        ${provenanceBlock(candidateRecord)}
        ${detailBlock(drilldown ?? { recorded_decision: item.recorded_decision, evidence_label: item.evidence_label, source_artifact: item.source_artifact, source_path: item.source_path, promotion_approved: item.promotion_approved }, "展开比较依据")}
      </article>`;
    }).join("")}</div>`;
  }

  function evidenceData(record) {
    const value = record?.value;
    if (Array.isArray(value)) return value;
    if (!isObject(value)) return [];
    const direct = readAny(value, ["evidence", "items", "records", "claims", "results"]);
    return Array.isArray(direct) ? direct : simpleEntries(direct).map(({ key, value: item }) => isObject(item) ? { title: key, ...item } : { title: key, summary: item });
  }

  function renderEvidence(record, error) {
    if (!record) return panelError("证据链暂不可读", error);
    const entries = evidenceData(record);
    if (!entries.length) return panelEmpty("暂无可展示证据", "研究结论不能由空白页面替代；请等待证据投影写入。", "evidence-empty");
    return `<ol class="rh-evidence-list">${entries.map((entry, index) => {
      const item = isObject(entry) ? entry : { summary: entry };
      const title = textValue(readAny(item, ["title", "claim", "name", "label", "id"]), `证据 ${index + 1}`);
      const grade = readAny(item, ["grade", "evidence_grade", "evidence_label", "tier", "confidence"]);
      const summaryValue = readAny(item, ["summary", "message", "finding", "result", "description", "caveat"]);
      const summary = isMeaningful(summaryValue) ? textValue(summaryValue) : evidenceLabel(item.evidence_label);
      const status = readAny(item, ["status", "state", "availability"]) ?? record.status;
      const itemRecord = {
        provenance: readAny(item, ["provenance", "source", "sources", "path"]) ?? item.path ?? record.provenance,
        asOf: readAny(item, ["as_of", "asOf", "date", "updated_at"]) ?? record.asOf,
        caveats: readAny(item, ["caveats", "limitations", "warnings"]) ?? record.caveats,
      };
      return `<li class="rh-evidence">
        <div class="rh-evidence__number" aria-hidden="true">${String(index + 1).padStart(2, "0")}</div>
        <div class="rh-evidence__body">
          <div class="rh-evidence__head"><h3>${escapeHtml(title)}</h3>${displayStatus(status)}</div>
          <p>${escapeHtml(summary)}</p>
          ${isMeaningful(grade) ? `<span class="rh-evidence-grade">证据等级：${escapeHtml(evidenceLabel(grade))}</span>` : ""}
          ${provenanceBlock(itemRecord)}
          ${detailBlock(readAny(item, ["drilldown", "details", "data", "links"]) ?? { availability: item.availability, integrity: item.integrity, path: item.path, content_sha256: item.content_sha256 }, "展开证据详情")}
        </div>
      </li>`;
    }).join("")}</ol>`;
  }

  function questionData(record) {
    const value = record?.value;
    if (Array.isArray(value)) return value;
    if (!isObject(value)) return [];
    const direct = readAny(value, ["questions", "items", "open_questions", "backlog"]);
    return Array.isArray(direct) ? direct : simpleEntries(direct).map(({ key, value: item }) => isObject(item) ? { title: key, ...item } : { title: key, summary: item });
  }

  function safeAction(record, decisionRecord) {
    const questionValue = isObject(record?.value) ? record.value : {};
    return readAny(questionValue, ["next_safe_action", "safe_action", "next_action", "recommended_next_step"])
      ?? decisionData(decisionRecord).safeAction;
  }

  function renderQuestions(record, decisionRecord, error) {
    if (!record && !decisionData(decisionRecord).safeAction) return panelError("开放问题暂不可读", error);
    const action = safeAction(record, decisionRecord);
    const questions = questionData(record);
    const actionMarkup = isMeaningful(action)
      ? `<aside class="rh-safe-action"><div class="rh-eyebrow">当前可安全推进</div><p>${escapeHtml(textValue(action))}</p><span>仅描述下一项受控研究或核验；不执行交易、不启动回放。</span></aside>`
      : `<aside class="rh-safe-action rh-safe-action--empty"><div class="rh-eyebrow">当前可安全推进</div><p>尚未提供可安全推进的下一项。</p></aside>`;
    if (!questions.length) return `${actionMarkup}${panelEmpty("暂无开放问题", "这不代表所有问题都已关闭；当前投影没有列出待办。", "question-empty")}`;
    return `${actionMarkup}<ol class="rh-question-list">${questions.map((question, index) => {
      const item = isObject(question) ? question : { summary: question };
      const title = textValue(readAny(item, ["title", "question", "name", "label", "id"]), `开放问题 ${index + 1}`);
      const summaryValue = readAny(item, ["summary", "description", "message", "reason", "decision_change"]);
      const summary = isMeaningful(summaryValue) ? textValue(summaryValue) : evidenceLabel(item.evidence_label);
      const status = readAny(item, ["status", "state", "priority"]) ?? "pending";
      const sourceArtifacts = asArray(item.source_artifacts).map((artifactId) => provenanceLabel(record, artifactId));
      const itemRecord = {
        provenance: readAny(item, ["provenance", "source", "sources"]) ?? (sourceArtifacts.length ? sourceArtifacts : record?.provenance),
        asOf: readAny(item, ["as_of", "asOf", "updated_at", "date"]) ?? record?.asOf,
        caveats: readAny(item, ["caveats", "limitations", "conditions"]) ?? record?.caveats,
      };
      return `<li class="rh-question">
        <div class="rh-question__number">${String(index + 1).padStart(2, "0")}</div>
        <div><div class="rh-question__head"><h3>${escapeHtml(title)}</h3>${displayStatus(status)}</div><p>${escapeHtml(summary)}</p>${provenanceBlock(itemRecord)}${detailBlock(readAny(item, ["drilldown", "details", "protocol"]) ?? { evidence_label: item.evidence_label, source_artifacts: item.source_artifacts }, "查看问题边界")}</div>
      </li>`;
    }).join("")}</ol>`;
  }

  function panelEmpty(title, text, type = "empty") {
    return `<div class="rh-panel-state rh-panel-state--${escapeHtml(type)}"><div class="rh-panel-state__glyph" aria-hidden="true">${ICONS.info}</div><div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(text)}</p></div></div>`;
  }

  function panelError(title, error) {
    const message = error?.message || "接口未返回可用数据。";
    return `<div class="rh-panel-state rh-panel-state--error" role="alert"><div class="rh-panel-state__glyph" aria-hidden="true">${ICONS.info}</div><div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(message)}</p><button class="rh-text-button" type="button" data-rh-action="refresh">重新读取</button></div></div>`;
  }

  function sectionHeading(index, eyebrow, title, description, id) {
    return `<header class="rh-section-heading" id="${escapeHtml(id)}"><span class="rh-section-index" aria-hidden="true">${index}</span><div><div class="rh-eyebrow">${escapeHtml(eyebrow)}</div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(description)}</p></div></header>`;
  }

  function latestAsOf(records) {
    const values = Object.values(records)
      .map((record) => record?.asOf)
      .filter(isMeaningful)
      .map((value) => textValue(value, ""))
      .filter(Boolean);
    return values.length ? values[0] : "未提供";
  }

  function renderLoadingSkeleton() {
    return `<div class="rh-skeletons" aria-label="正在读取研究投影"><div class="rh-skeleton rh-skeleton--hero"></div><div class="rh-skeleton-grid"><div class="rh-skeleton"></div><div class="rh-skeleton"></div><div class="rh-skeleton"></div></div><div class="rh-skeleton rh-skeleton--wide"></div></div>`;
  }

  function page(records, errors, loading) {
    const decision = records.decision;
    const candidate = records.candidates;
    const evidence = records.evidence;
    const questions = records.questions;
    const decisionInfo = decisionData(decision);
    const decisionStatus = decisionInfo.status;
    const currentAsOf = latestAsOf(records);
    const failed = Object.keys(errors).length;
    const showSkeleton = loading && !Object.keys(records).length;
    const errorNote = failed
      ? `<div class="rh-load-note rh-load-note--warning" role="status">${ICONS.info}<span>${failed} 项研究投影未能读取；已展示其余已获取内容。</span></div>`
      : "";

    return `<div class="rh-shell" data-rh-theme="${escapeHtml(state.theme)}">
      <a class="rh-skip-link" href="#rh-decision">跳到研究裁决</a>
      <header class="rh-topbar">
        <a class="rh-brand" href="#rh-decision" aria-label="QuantLab 研究裁决台首页">
          <span class="rh-brand__mark" aria-hidden="true"><svg viewBox="0 0 40 40"><path d="M7 28.5 15.5 20l5.5 5.5L33 13.5"/><path d="M29 13.5H33v4"/></svg></span>
          <span><strong>QuantLab</strong><small>研究裁决台</small></span>
        </a>
          <nav class="rh-nav" aria-label="页面导航">
            <a href="/dashboard">决策看板</a><a href="#rh-decision">裁决</a><a href="#rh-candidates">候选</a><a href="#rh-evidence">证据</a><a href="#rh-questions">开放问题</a><a href="/portfolio">账户行情</a>
        </nav>
        <div class="rh-tools">
          <span class="rh-asof"><b>资料截至</b>${escapeHtml(currentAsOf)}</span>
          <button class="rh-icon-button" type="button" data-rh-action="theme" aria-label="切换浅色或深色主题" aria-pressed="${state.theme === "dark"}">${state.theme === "dark" ? ICONS.sun : ICONS.moon}</button>
          <button class="rh-refresh" type="button" data-rh-action="refresh" aria-label="${loading ? "正在读取研究投影" : "重新读取研究投影"}" ${loading ? "disabled" : ""}>${ICONS.refresh}<span>${loading ? "读取中" : "重新读取"}</span></button>
        </div>
      </header>
      <div class="rh-context" role="note"><span class="rh-context__mark" aria-hidden="true"></span><p>此页面展示研究证据与采用门禁，不展示实时行情，不提供下单或自动执行。</p></div>
      ${errorNote}
      <div class="rh-main" id="rh-main" ${loading ? 'aria-busy="true"' : ""}>
        ${showSkeleton ? renderLoadingSkeleton() : `
          <section class="rh-decision-section" id="rh-decision" aria-labelledby="rh-decision-title">
            <article class="rh-decision-card rh-decision-card--${statusInfo(decisionStatus).tone}">
              <div class="rh-decision-card__main">
                <div class="rh-eyebrow">研究裁决 · 01</div>
                <div class="rh-decision-card__title"><h1 id="rh-decision-title">${escapeHtml(textValue(decisionInfo.title))}</h1>${displayStatus(decisionStatus)}</div>
                <p class="rh-decision-card__summary">${escapeHtml(textValue(decisionInfo.summary))}</p>
                ${decision ? provenanceBlock(decision) : panelError("裁决暂不可读", errors.decision)}
                ${decision ? detailBlock(decision.drilldown, "查看裁决详情") : ""}
              </div>
              <div class="rh-decision-card__side"><span>采用状态</span><strong>${escapeHtml(statusInfo(decisionStatus).label)}</strong><p>状态、来源和限制必须同时成立；单一条件性数值不能构成采用许可。</p></div>
            </article>
          </section>
          <section class="rh-section rh-section--gates" aria-labelledby="rh-gates-title">
            ${sectionHeading("02", "采用门禁", "先回答：为什么现在不能采用", "每一项门禁都需要状态、依据与可追溯的时间边界。", "rh-gates-title")}
            ${decision ? renderGateMatrix(decision) : panelError("采用门禁暂不可读", errors.decision)}
          </section>
          <section class="rh-section" id="rh-candidates" aria-labelledby="rh-candidates-title">
            ${sectionHeading("03", "候选对照", "候选不是推荐", "只在同一研究口径内比较；没有被提升为采用对象的候选会明确保留限制。", "rh-candidates-title")}
            ${renderCandidates(candidate, errors.candidates)}
          </section>
          <section class="rh-section rh-section--evidence" id="rh-evidence" aria-labelledby="rh-evidence-title">
            ${sectionHeading("04", "证据链", "结论必须能回到来源", "显示结论、来源、截至时间与反证边界；没有来源的数字不应被当作绩效。", "rh-evidence-title")}
            ${renderEvidence(evidence, errors.evidence)}
          </section>
          <section class="rh-section rh-section--questions" id="rh-questions" aria-labelledby="rh-questions-title">
            ${sectionHeading("05", "开放问题", "下一步只做可判定的最小验证", "把尚未解决的问题和能改变决策的安全下一步分开，避免重新开题或为了漂亮数字调参。", "rh-questions-title")}
            ${renderQuestions(questions, decision, errors.questions)}
          </section>`}
      </div>
      <footer class="rh-footer"><span>QuantLab · 只读研究投影</span><span>不构成投资建议或自动执行授权</span></footer>
    </div>`;
  }

  function render() {
    mount.innerHTML = page(state.records, state.errors, state.loading);
    mount.dataset.rhTheme = state.theme;
    bindEvents();
  }

  function bindEvents() {
    mount.querySelectorAll("[data-rh-action]").forEach((button) => {
      button.addEventListener("click", () => {
        if (button.dataset.rhAction === "refresh") load();
        if (button.dataset.rhAction === "theme") toggleTheme();
      });
    });
  }

  function toggleTheme() {
    state.theme = state.theme === "dark" ? "light" : "dark";
    try {
      window.localStorage.setItem(THEME_KEY, state.theme);
    } catch (_) {
      // Theme switching remains useful even when persistence is unavailable.
    }
    render();
  }

  async function readJson(url, signal) {
    const response = await window.fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal,
    });
    const body = await response.text();
    let parsed;
    try {
      parsed = body ? JSON.parse(body) : {};
    } catch (_) {
      throw new Error(`服务返回了无法解析的数据（HTTP ${response.status}）。`);
    }
    if (!response.ok) {
      const reason = textValue(readAny(parsed, ["error", "message", "detail"]), `HTTP ${response.status}`);
      throw new Error(reason);
    }
    return parsed;
  }

  async function load() {
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    state.loading = true;
    state.errors = {};
    render();
    const tasks = Object.entries(ENDPOINTS).map(async ([key, url]) => {
      const payload = await readJson(url, controller.signal);
      return { key, record: recordFrom(payload) };
    });
    const settled = await Promise.allSettled(tasks);
    if (state.controller !== controller) return;
    for (const result of settled) {
      if (result.status === "fulfilled") {
        state.records[result.value.key] = result.value.record;
      } else if (result.reason?.name !== "AbortError") {
        const key = Object.keys(ENDPOINTS)[settled.indexOf(result)];
        state.errors[key] = result.reason instanceof Error ? result.reason : new Error("读取研究投影时发生未知错误。");
      }
    }
    state.loading = false;
    state.controller = null;
    render();
  }

  render();
  load();
})();
