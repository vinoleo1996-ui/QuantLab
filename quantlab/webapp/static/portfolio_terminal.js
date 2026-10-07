/* QuantLab 决策看板前端逻辑。
   约定：格式化函数对缺失值一律返回 "—"，绝不返回 0 或 "-0.00%"，
   避免把"没有数据"读成"跌了/持平"。金额与百分比分开格式化，不混用。 */

'use strict';

/* ----------------------------------------------------------------- utils */
const $ = (id) => document.getElementById(id);

function esc(text) {
  return String(text === null || text === undefined ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
const isNum = (v) => v !== null && v !== undefined && v !== '' && !Number.isNaN(Number(v));

function fmtPct(v, d = 2) { return isNum(v) ? (Number(v) * 100).toFixed(d) + '%' : '—'; }
function fmtSignedPct(v, d = 2) { return isNum(v) ? (v >= 0 ? '+' : '') + (Number(v) * 100).toFixed(d) + '%' : '—'; }
function fmtNum(v, d = 2) { return isNum(v) ? Number(v).toFixed(d) : '—'; }
function fmtMoney(v, d = 0) {
  if (!isNum(v)) return '—';
  return Number(v).toLocaleString('zh-CN', { minimumFractionDigits: d, maximumFractionDigits: d });
}
function fmtMoneyCny(v) { return isNum(v) ? '¥' + fmtMoney(v) : '—'; }
function fmtCompact(v) {
  if (!isNum(v)) return '—';
  const n = Math.abs(Number(v));
  const sign = Number(v) < 0 ? '-' : '';
  if (n >= 1e8) return `${sign}${(n / 1e8).toFixed(2)}亿`;
  if (n >= 1e4) return `${sign}${(n / 1e4).toFixed(2)}万`;
  return `${sign}${fmtNum(n, 0)}`;
}
function fmtSignedMoneyCny(v, digits = 0) {
  if (!isNum(v)) return '—';
  const n = Number(v);
  return `${n >= 0 ? '+' : '-'}¥${fmtMoney(Math.abs(n), digits)}`;
}
/* Direction class. A股: 红涨绿跌。 */
function dirClass(v) {
  if (!isNum(v) || Number(v) === 0) return 'flat';
  return Number(v) > 0 ? 'up' : 'down';
}
/* Arrow + value together, so direction is never encoded by colour alone. */
function trend(v, d = 2) {
  if (!isNum(v)) return '<span class="flat">—</span>';
  const n = Number(v);
  const arrow = n > 0 ? '▲' : (n < 0 ? '▼' : '·');
  return `<span class="trend ${dirClass(n)}">${arrow} ${fmtSignedPct(n, d)}</span>`;
}

async function api(path) {
  const r = await fetch(path, { cache: 'no-store' });
  if (!r.ok) {
    let detail = '';
    try { detail = JSON.stringify(await r.json()); } catch (e) { /* ignore */ }
    throw new Error(`${path} → HTTP ${r.status} ${detail}`);
  }
  return r.json();
}

/* Inline sparkline. Recent window only: a 10-year line compressed into 80px
   is noise, and the reader wants "what happened lately" at a glance. */
function sparkline(values, opts = {}) {
  const vals = (values || []).filter(isNum).map(Number);
  if (vals.length < 3) return '';
  const win = opts.window || 90;
  const data = vals.slice(-win);
  const w = opts.w || 96, h = opts.h || 24, pad = 2;
  const min = Math.min(...data), max = Math.max(...data), span = (max - min) || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * (w - 2) + 1;
    const y = h - pad - ((v - min) / span) * (h - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const rising = data[data.length - 1] >= data[0];
  const stroke = opts.color || (rising ? 'var(--up)' : 'var(--down)');
  const fillId = `sg${Math.random().toString(36).slice(2, 8)}`;
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">
    <defs><linearGradient id="${fillId}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${stroke}" stop-opacity=".28"/>
      <stop offset="100%" stop-color="${stroke}" stop-opacity="0"/>
    </linearGradient></defs>
    <polygon points="1,${h} ${pts.join(' ')} ${w - 1},${h}" fill="url(#${fillId})"/>
    <polyline points="${pts.join(' ')}" fill="none" stroke="${stroke}"
      stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"/>
  </svg>`;
}

/* ----------------------------------------------------------------- theme */
const THEME_KEY = 'qlab-theme';
function applyTheme(name) {
  document.documentElement.setAttribute('data-theme', name);
  try { localStorage.setItem(THEME_KEY, name); } catch (e) { /* private mode */ }
  const btn = $('themeToggle');
  if (btn) {
    btn.textContent = name === 'dark' ? '☾' : '☀';
    btn.setAttribute('aria-label', name === 'dark' ? '切换到亮色主题' : '切换到暗色主题');
    btn.title = name === 'dark' ? '切换到亮色' : '切换到暗色';
  }
  // Charts read colours from CSS variables, so a theme change needs a redraw.
  requestAnimationFrame(() => { if (state.seriesLoaded) drawCharts(); });
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (e) { /* ignore */ }
  applyTheme(saved === 'light' ? 'light' : 'dark');
  $('themeToggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
  });
}

/* ----------------------------------------------------------------- state */
const state = {
  charts: {},
  seriesLoaded: false,
  range: 'all',
  cachedSeries: null,
  sortState: {},
};

/* Read a design token so charts follow the active theme automatically. */
function token(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  return (v && v.trim()) || fallback;
}

/* ----------------------------------------------------------------- tabs */
function bindTabs() {
  const buttons = document.querySelectorAll('.tab');
  const show = (name) => {
    buttons.forEach((b) => {
      const on = b.dataset.pane === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    document.querySelectorAll('.pane').forEach((p) => {
      p.classList.toggle('active', p.id === name);
    });
    if ((name === 'chart-pane' || name === 'overview-pane') && state.seriesLoaded) drawCharts();
    if (name === 'holdings-pane' && kChart) requestAnimationFrame(() => kChart.resize());
    const loaders = { 'holdings-pane': loadHoldings, 'signals-pane': loadSignals, 'experts-pane': loadExperts, 'run-pane': loadRun };
    if (loaders[name]) loaders[name]();
  };
  buttons.forEach((b) => b.addEventListener('click', () => show(b.dataset.pane)));
}

/* ----------------------------------------------------------------- charts */
function seriesToArrays(s) {
  const idx = s.dates.map((d, i) => d);
  return {
    dates: idx,
    nav: s.nav,
    benchmark: s.benchmark,
    drawdown: s.drawdown,
  };
}

function sliceRange(a) {
  const n = a.dates.length;
  if (!n) return a;
  if (state.range === 'all') return a;
  const want = state.range === '3y' ? 756 : 252;
  const start = Math.max(0, n - want);
  const sl = (arr) => (arr ? arr.slice(start) : null);
  return { dates: a.dates.slice(start), nav: sl(a.nav), benchmark: sl(a.benchmark), drawdown: sl(a.drawdown) };
}

function drawCharts() {
  if (typeof echarts === 'undefined') return;
  const raw = state.cachedSeries;
  const navEl = $('chartNav'), ddEl = $('chartDD');
  if (!raw || !navEl || !ddEl) return;
  const a = sliceRange(seriesToArrays(raw));
  if (!a.dates.length) return;

  const cUp = token('--up', '#f43f5e');
  const cDown = token('--down', '#10b981');
  const cAccent = token('--accent-cyan', '#06b6d4');
  const cMuted = token('--text-muted', '#64748b');
  const cSurface = token('--bg-surface', '#121826');
  const cBorder = token('--border-subtle', '#232e47');
  const cText = token('--text-secondary', '#94a3b8');

  const base = {
    animationDuration: 420,
    grid: { left: 52, right: 18, top: 18, bottom: 26 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: cSurface,
      borderColor: cBorder,
      textStyle: { color: token('--text-primary', '#f8fafc'), fontSize: 11.5 },
      axisPointer: { type: 'cross', label: { backgroundColor: cBorder } },
    },
    xAxis: {
      type: 'category', data: a.dates, boundaryGap: false,
      axisLine: { lineStyle: { color: cBorder } },
      axisLabel: { color: cMuted, fontSize: 10, hideOverlap: true },
      axisTick: { show: false },
    },
  };

  if (!state.charts.nav) state.charts.nav = echarts.init(navEl);
  const navSeries = [{
    name: '组合净值', type: 'line', data: a.nav, showSymbol: false,
    lineStyle: { width: 1.8, color: cAccent },
    areaStyle: {
      color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: 'rgba(6,182,212,.28)' },
        { offset: 1, color: 'rgba(6,182,212,0)' },
      ]),
    },
  }];
  if (a.benchmark && a.benchmark.some(isNum)) {
    navSeries.push({
      name: '基准', type: 'line', data: a.benchmark, showSymbol: false,
      lineStyle: { width: 1.2, color: cMuted, type: 'dashed' },
    });
  }
  state.charts.nav.setOption({
    ...base,
    legend: {
      top: 0, right: 8, itemWidth: 16, itemHeight: 2,
      textStyle: { color: cText, fontSize: 11 },
      data: navSeries.map((series) => series.name),
    },
    grid: { left: 56, right: 18, top: 28, bottom: 22 },
    tooltip: { ...base.tooltip, valueFormatter: (v) => isNum(v) ? fmtMoneyCny(v) : '—' },
    yAxis: {
      type: 'value', scale: true,
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: cMuted, fontSize: 10 },
      splitLine: { lineStyle: { color: cBorder, opacity: .45 } },
    },
    series: navSeries,
  });

  if (!state.charts.dd) state.charts.dd = echarts.init(ddEl);
  const navs = (a.nav || []).filter(isNum).map(Number);
  if (navs.length && $('acctNav')) {
    const last = navs[navs.length - 1];
    const first = navs[0];
    const chg = first ? last / first - 1 : null;
    $('acctNav').textContent = fmtMoneyCny(last);
    $('acctRet').innerHTML = `<div class="px ${dirClass(chg)}">${fmtSignedPct(chg)}</div><div class="${dirClass(chg)}">所选区间</div>`;
    const end = a.dates[a.dates.length - 1];
    if ($('acctSub') && end) $('acctSub').textContent = `截至 ${String(end).slice(0, 10)}`;
  }

  state.charts.dd.setOption({
    ...base,
    grid: { left: 52, right: 18, top: 14, bottom: 26 },
    tooltip: { ...base.tooltip, valueFormatter: (v) => isNum(v) ? (Number(v) * 100).toFixed(2) + '%' : '—' },
    yAxis: {
      type: 'value', max: 0,
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: cMuted, fontSize: 10, formatter: (v) => (v * 100).toFixed(0) + '%' },
      splitLine: { lineStyle: { color: cBorder, opacity: .45 } },
    },
    series: [{
      name: '回撤', type: 'line', data: a.drawdown, showSymbol: false,
      lineStyle: { width: 1.2, color: cUp },
      areaStyle: { color: 'rgba(244,63,94,.16)' },
    }],
  });
  state.charts.nav.resize();
  state.charts.dd.resize();
}

function bindRange() {
  document.querySelectorAll('#rangeSeg button').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#rangeSeg button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      state.range = b.dataset.range;
      if (state.cachedSeries) drawCharts();
    });
  });
}

/* ----------------------------------------------------------------- tables */
function sortMissing(v) {
  return v === undefined || v === null || v === '';
}

function sortTableRows(rows, key, dir) {
  rows.sort((r1, r2) => {
    const v1 = r1.dataset[key];
    const v2 = r2.dataset[key];
    const n1 = parseFloat(v1);
    const n2 = parseFloat(v2);
    const num = !sortMissing(v1) && !sortMissing(v2) && !Number.isNaN(n1) && !Number.isNaN(n2);
    if (num) return dir === 'desc' ? n2 - n1 : n1 - n2;
    if (sortMissing(v1) && sortMissing(v2)) return 0;
    if (sortMissing(v1)) return 1;
    if (sortMissing(v2)) return -1;
    const c = String(v1).localeCompare(String(v2), 'zh-CN');
    return dir === 'desc' ? -c : c;
  });
  return rows;
}

function paintSort(table, key, dir) {
  table.querySelectorAll('thead th.sortable').forEach((h) => {
    h.classList.remove('sorted');
    const old = h.querySelector('.arrow');
    if (old) old.remove();
    if (h.dataset.key === key) {
      h.classList.add('sorted');
      const arrow = document.createElement('span');
      arrow.className = 'arrow';
      arrow.textContent = dir === 'desc' ? ' ▼' : ' ▲';
      h.appendChild(arrow);
    }
  });
}

/* Sort uses the numeric data-* on each row. A click toggles desc, then asc. */
function makeSortable(table, getRows) {
  if (!table) return;
  table.querySelectorAll('thead th.sortable').forEach((th) => {
    if (!th.getAttribute('title')) th.title = '点击切换降序 / 升序';
    th.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const key = th.dataset.key;
      const cur = state.sortState[table.id] || {};
      const dir = cur.key === key && cur.dir === 'desc' ? 'asc' : 'desc';
      state.sortState[table.id] = { key, dir };
      paintSort(table, key, dir);
      const tbody = table.querySelector('tbody');
      sortTableRows(getRows(), key, dir).forEach((r) => tbody.appendChild(r));
    });
  });
  const saved = state.sortState[table.id];
  if (saved) {
    paintSort(table, saved.key, saved.dir);
    const tbody = table.querySelector('tbody');
    sortTableRows(getRows(), saved.key, saved.dir).forEach((r) => tbody.appendChild(r));
  }
}

/* ----------------------------------------------------------------- overview */
function kpiCard(o) {
  const meta = o.meta !== undefined ? o.meta : (o.trend !== undefined ? trend(o.trend) : (o.note || ''));
  return `<div class="card kpi">
    <div class="label">${esc(o.label)}</div>
    <div class="value ${o.cls || ''}">${o.value}</div>
    <div class="meta">${meta}</div>
    ${o.spark || ''}
  </div>`;
}

async function loadOverview() {
  const host = $('kpis');
  try {
    const [d, s] = await Promise.all([api('/api/nav'), api('/api/series')]);
    state.cachedSeries = s;
    state.seriesLoaded = true;

    $('chipExec').innerHTML = `<span class="dot ${d.interpretable ? 'ok' : 'warn'}"></span>执行日 <b>${esc(d.last_exec || '—')}</b>`;
    const statusName = {
      paper_simulation_unverified: '纸账户',
      research_unvalidated: '研究账户',
    }[d.status] || '模拟账户';
    $('chipStatus').innerHTML = d.interpretable
      ? `<span class="chip ok"><span class="dot ok"></span>${esc(statusName)} <b>可阅读</b></span>`
      : `<span class="chip warn"><span class="dot warn"></span>${esc(statusName)} <b>未验证</b></span>`;

    if (!d.interpretable) {
      const rs = (d.blocked_reasons || []);
      const items = rs.length
        ? `<ul style="margin:8px 0 0 18px;line-height:1.8;text-align:left;display:inline-block">${rs.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`
        : '<div class="muted" style="margin-top:6px">未提供具体阻断原因</div>';
      host.innerHTML = `<div class="card"><div class="empty">
        <span class="big">当前结果不可解释，按设计不发布绩效与买卖建议</span>
        数据或验证未通过；这是门禁的正常结果，不是页面故障。${items}
      </div></div>`;
      // Hide the whole chart + risk region: an empty canvas and an empty gauge
      // read as "broken" rather than "not applicable".
      const grid = document.getElementById('overviewGrid');
      if (grid) grid.style.display = 'none';
      $('overviewNote').innerHTML = '';
      return;
    }
    const grid = document.getElementById('overviewGrid');
    if (grid) grid.style.display = '';
    const body = document.querySelector('#chart-pane .card-body');
    if (body) body.style.display = '';
    const note = $('chartNote');
    if (note) note.textContent = '实线是组合净值，虚线是基准。下方曲线是回撤。';
    $('rangeSeg').style.display = '';

    const ddMag = Math.abs(Number(d.metrics.max_drawdown) || 0);
    const ddCls = ddMag > 0.25 ? 'up' : 'flat';
    // Annualising a handful of sessions is arithmetic, not economics. The backend
    // nulls these out below 60 sessions; never render a number it withheld.
    const thin = d.metrics.annualisable === false;
    const na = '<span class="na" title="样本交易日不足 60 天，年化无统计意义">样本不足</span>';
    host.innerHTML = [
      statChip('累计收益', fmtSignedPct(d.metrics.total_return), dirClass(d.metrics.total_return)),
      statChip('年化', thin ? na : fmtSignedPct(d.metrics.annual_return), thin ? 'flat' : dirClass(d.metrics.annual_return)),
      statChip('最大回撤', fmtPct(d.metrics.max_drawdown), ddCls),
      statChip('夏普', thin ? na : fmtNum(d.metrics.sharpe), 'flat'),
      statChip('交易日', d.metrics.sessions ? `${d.metrics.sessions} 天` : '—', 'flat'),
      statChip('今年', fmtSignedPct(d.ytd), dirClass(d.ytd)),
    ].join('');
    if ($('acctSub')) {
      $('acctSub').textContent = thin
        ? `截至 ${d.last_exec || '—'} · 样本 ${d.metrics.sessions} 个交易日，年化/夏普/卡玛已隐藏`
        : `截至 ${d.last_exec || '—'}`;
    }
    drawCharts();
  } catch (e) {
    host.innerHTML = `<div class="card"><div class="empty">看板数据加载失败：${esc(e.message)}</div></div>`;
  }
}

/* ----------------------------------------------------------------- risk */
/* Drawdown gauge. Scale is fixed at the policy ceiling so the marker means the
   same thing every day; a per-day autoscale would make 2% look like a crash. */
function renderRiskGauge(dd, level) {
  const CEIL = 0.25;
  // dd arrives signed (negative = underwater). The gauge shows magnitude, so
  // abs() is required — without it the marker clamps to zero and lies.
  const mag = Math.abs(Number(dd) || 0);
  const pos = Math.max(0, Math.min(1, mag / CEIL)) * 100;
  const zones = [[0, .12, 'rgba(16,185,129,.30)'], [.12, .18, 'rgba(245,158,11,.32)'], [.18, .22, 'rgba(245,158,11,.48)'], [.22, .25, 'rgba(244,63,94,.42)']]
    .map(([a, b, c]) => `<div class="gauge-zone" style="left:${(a / CEIL) * 100}%;width:${((b - a) / CEIL) * 100}%;background:${c}"></div>`)
    .join('');
  const ticks = [.12, .18, .22, .25].map((t) => `<div class="gauge-tick" style="left:${(t / CEIL) * 100}%">${(t * 100).toFixed(0)}%</div>`).join('');
  return `<div class="gauge">
    <div class="gauge-track">${zones}
      <div class="gauge-marker" style="left:${pos}%" title="当前回撤 ${fmtPct(dd)}"></div>
    </div>    <div class="gauge-ticks">${ticks}</div>
  </div>`;
}

async function loadRisk() {
  const host = $('riskBody');
  try {
    const r = await api('/api/risk');
    if (!r.available) {
      // The whole region is hidden in the blocked state, so leave the panel
      // untouched rather than writing an empty-state box nobody will see.
      const grid = document.getElementById('overviewGrid');
      if (grid && grid.style.display !== 'none') {
        host.innerHTML = '<span class="muted">这笔回放没有单独的风险档位记录</span>';
      }
      return;
    }
    const stateLabel = { defensive: '防守', normal: '常态', active: '积极', legacy: '旧政策' }[r.policy_state] || '—';
    const stateCls = r.policy_state === 'active' ? 'active' : (r.policy_state === 'defensive' ? 'defensive' : 'normal');
    host.innerHTML = `
      <span class="state-badge ${stateCls}"><span class="dot"></span>${esc(stateLabel)}</span>
      <span>当前回撤 <b class="mono">${fmtPct(r.drawdown)}</b></span>
      <span>仓位 <b>${fmtPct(r.gross_scale, 0)}</b></span>
      <span>单股上限 <b>${fmtPct(r.max_stock_weight, 0)}</b></span>
      <span class="muted">${esc(r.policy_id || '')}</span>`;
  } catch (e) {
    host.innerHTML = `<div class="empty">风险预算加载失败：${esc(e.message)}</div>`;
  }
}

/* ----------------------------------------------------------------- holdings */
function weightBar(w) {
  const v = Number(w) || 0;
  return `<div class="wbar"><div class="wbar-track"><div class="wbar-fill" style="width:${Math.min(100, v * 1000)}%"></div></div><span>${fmtPct(v, 1)}</span></div>`;
}

let holdBusy = false;
// The user explicitly authorized public, code-only market requests.  The
// portfolio therefore opens in observable-market mode rather than silently
// showing a dated ledger mark as if it were today's price.
let publicQuotesEnabled = true;

function updatePublicQuoteControls(enabled) {
  const toggle = $('enablePublicQuotes');
  if (toggle) {
    toggle.textContent = enabled ? '关闭公开行情' : '启用公开行情';
    toggle.setAttribute('aria-pressed', enabled ? 'true' : 'false');
    toggle.title = enabled
      ? '停止发送持仓代码到公开行情源，并回到本地账本标记价。'
      : '启用后会向公开行情源发送这 50 个股票代码和 3 个公开指数代码，以读取实时价格、分时和 K 线；不会发送数量、成本或盈亏。';
  }
  document.querySelectorAll('#kFreq button, #kRange button').forEach((button) => {
    button.disabled = !enabled;
    button.setAttribute('aria-disabled', enabled ? 'false' : 'true');
  });
}

function describeQuoteBatch(summary, totalRows) {
  const states = (summary && summary.quote_state_counts) || {};
  const total = Number(totalRows) || 0;
  const verified = Number(summary && summary.quote_count) || 0;
  const received = Number(summary && summary.quote_received_count) || verified;
  const trading = Number(states.trading) || 0;
  const lunchBreak = Number(states.lunch_break) || 0;
  const delayed = Number(states.delayed) || 0;
  const closing = Number(states.closed) || 0;
  const preOpen = Number(states.pre_open) || 0;
  const missingTime = Number(states.timestamp_missing) || 0;
  const invalidTime = Number(states.timestamp_invalid) || 0;
  const earliest = summary && summary.quote_time_earliest;
  const latest = summary && summary.quote_time_latest;
  const time = earliest && latest && earliest !== latest ? `${earliest} 至 ${latest}` : (latest || earliest || '—');
  const sources = Array.isArray(summary && summary.quote_sources) && summary.quote_sources.length
    ? summary.quote_sources.join(' / ')
    : '—';

  if (total && trading === total) return { label: `实时 ${trading}/${total}`, time, sources, tone: 'ok', dot: 'pulse ok' };
  if (total && lunchBreak === total) return { label: `午间休市 ${lunchBreak}/${total}`, time, sources, tone: 'warn', dot: 'warn' };
  if (total && closing === total) return { label: `收盘 ${closing}/${total}`, time, sources, tone: 'warn', dot: 'warn' };
  if (total && preOpen === total) return { label: `待开盘 ${preOpen}/${total}`, time, sources, tone: 'warn', dot: 'warn' };
  if (!total || !verified) {
    const reason = missingTime || invalidTime
      ? `时间未核验 ${received}/${total}`
      : (received ? `报价未计入估值 ${received}/${total}` : `取数失败 0/${total}`);
    return { label: reason, time, sources, tone: 'bad', dot: 'bad' };
  }
  if (delayed) return { label: `行情延迟 ${delayed}/${total}`, time, sources, tone: 'warn', dot: 'warn' };
  return { label: `已核验报价 ${verified}/${total}`, time, sources, tone: 'warn', dot: 'warn' };
}

async function loadHoldings() {
  const host = $('htable');
  if (!host || holdBusy) return;
  holdBusy = true;
  try {
    const d = await api(`/api/holdings${publicQuotesEnabled ? '?live=1' : ''}`);
    publicQuotesEnabled = Boolean(d.public_quotes_enabled);
    updatePublicQuoteControls(publicQuotesEnabled);
    const rg = { '-1': '深度熊市', '0': '震荡', '1': '牛市' }[String((d.summary || {}).regime_trend)] || '—';
    const totW = (d.rows || []).reduce((a, r) => a + (Number(r.weight) || 0), 0);
    const tv = (d.rows || []).reduce((a, r) => {
      const mv = isNum(r.live_quote) && isNum(r.quantity) ? Number(r.live_quote) * Number(r.quantity) : Number(r.market_value) || 0;
      return a + mv;
    }, 0);
    const totalRows = (d.rows || []).length;
    const quoteCount = Number((d.summary || {}).quote_count) || 0;
    const dayQuoteCount = Number((d.summary || {}).day_quote_count) || 0;
    const quoteBatch = publicQuotesEnabled ? describeQuoteBatch(d.summary || {}, totalRows) : null;
    const priceChip = $('chipPrice');
    if (priceChip) {
      priceChip.className = `chip${quoteBatch ? ` ${quoteBatch.tone}` : ''}`;
      priceChip.innerHTML = quoteBatch
        ? `<span class="dot ${quoteBatch.dot}"></span>行情 <b>${esc(quoteBatch.label)} · ${esc(quoteBatch.time)}</b>`
        : '<span class="dot"></span>行情 <b>账本标记价</b>';
    }
    const regime = rg === '—' ? '' : ` · ${rg}`;
    const quoteCoverage = quoteBatch
      ? ` · ${quoteBatch.label} · 来源 ${quoteBatch.sources} · 时间 ${quoteBatch.time}`
      : ' · 本地账本观察';
    const dayCoverage = publicQuotesEnabled && dayQuoteCount < totalRows ? ` · 今日可计价 ${dayQuoteCount}/${totalRows}` : '';
    const holdSummary = `${totalRows} 只 · 仓位 ${fmtPct(totW, 1)} · 市值 ${fmtMoneyCny(tv)}${regime}${quoteCoverage}${dayCoverage}`;
    $('holdNote').textContent = holdSummary;
    $('holdNote').title = holdSummary;

    // 今日盈亏用证券行情软件式大盘块，红涨绿跌，视觉权重必须压过下方列表。
    const dayBox = $('holdDay');
    const todayEl = $('holdToday');
    const pnlEl = $('holdPnl');
    const ytdEl = $('holdYtd');
    if (dayBox && todayEl) {
      const t = d.today;
      const pnl = (d.summary || {}).day_pnl;
      if (isNum(t)) {
        dayBox.classList.remove('up', 'down', 'flat');
        dayBox.classList.add(dirClass(t));
        todayEl.textContent = fmtSignedPct(t);
        if (pnlEl) pnlEl.textContent = isNum(pnl) ? fmtSignedMoneyCny(pnl, 2) : '—';
      } else {
        dayBox.classList.remove('up', 'down');
        dayBox.classList.add('flat');
        todayEl.textContent = publicQuotesEnabled
          ? (quoteBatch ? quoteBatch.label : (totalRows && dayQuoteCount < totalRows ? `覆盖 ${dayQuoteCount}/${totalRows}` : '待开盘'))
          : '账本标记';
        if (pnlEl) pnlEl.textContent = '—';
      }
      const y = (d.rows || []).reduce((a, r) => a + (Number(r.weight) || 0) * (isNum(r.return_ytd) ? Number(r.return_ytd) : 0), 0);
      if (ytdEl) {
        if (Math.abs(y) > 1e-9) {
          ytdEl.className = `pday-y ${dirClass(y)}`;
          ytdEl.textContent = `今年 ${fmtSignedPct(y)}`;
        } else { ytdEl.textContent = ''; }
      }
    }

    if (!(d.rows || []).length) {
      host.innerHTML = '<div class="empty">当前无持仓</div>';
      return;
    }
    const scrollTop = host.scrollTop;
    const body = (d.rows || []).map((r) => {
      const px = isNum(r.live_quote) ? Number(r.live_quote) : r.price;
      const mv = isNum(r.live_quote) && isNum(r.quantity) ? Number(r.live_quote) * Number(r.quantity) : r.market_value;
      const since = isNum(r.cost) && isNum(px) && Number(r.cost) ? Number(px) / Number(r.cost) - 1 : r.return_since_entry;
      const sector = r.sector || '';
      const listingBoard = r.board_name || r.board || '';
      const holdingMeta = [r.code, r.industry || '行业未知', listingBoard, r.quote_state_label || '账本标记价']
        .filter(Boolean).join(' · ');
      const attr = (v) => (isNum(v) ? esc(String(Number(v))) : '');
      const excess = isNum(r.pct_chg) && isNum(r.board_pct) ? Number(r.pct_chg) - Number(r.board_pct) : r.excess_pct;
      return `<button type="button" class="watch-row quote-row${quoteCode === String(r.code) ? ' on' : ''}" data-code="${esc(r.code)}" data-name="${esc(r.name || '')}" data-weight="${attr(r.weight)}" data-entry="${esc(r.entry_date || '')}" data-qty="${attr(r.quantity)}" data-cost="${attr(r.cost)}" data-mark="${attr(r.mark_price || r.price)}" data-px="${attr(px)}" data-chg="${attr(r.pct_chg)}" data-ytd="${attr(r.return_ytd)}" data-since="${attr(since)}" data-pnl="${attr(r.unrealized_pnl)}" data-mv="${attr(mv)}" data-industry="${esc(r.industry || '')}" data-sector="${esc(sector)}" data-listing-board="${esc(listingBoard)}" data-quote-state-label="${esc(r.quote_state_label || '')}" data-quote-source="${esc(r.quote_source || '')}" data-quote-time="${esc(r.quote_time || '')}" data-board="${attr(r.board_pct)}" data-excess="${attr(excess)}" data-lead-today="${esc(r.leader_today || '')}" data-lead-today-pct="${attr(r.leader_today_pct)}" data-lead-ytd="${esc(r.leader_ytd || '')}" data-lead-ytd-pct="${attr(r.leader_ytd_pct)}">
        <span class="wn"><b>${esc(r.name || r.code)}</b><i>${esc(holdingMeta)}</i></span>
        <span class="wp">${fmtNum(px, 2)}</span>
        <span class="wc ${isNum(r.pct_chg) ? dirClass(r.pct_chg) : 'pre'}">${isNum(r.pct_chg) ? fmtSignedPct(r.pct_chg) : '待开盘'}</span>
      </button>`;
    }).join('');
    host.innerHTML = body;
    if (!state.sortState.htable) state.sortState.htable = { key: 'chg', dir: 'desc' };
    applyWatchSort(state.sortState.htable.key, state.sortState.htable.dir, false);
    host.querySelectorAll('.quote-row').forEach((tr) => {
      tr.addEventListener('click', () => {
        externalQuote = false;
        quoteKind = '';
        if (window.QuantLabQuote) window.QuantLabQuote.kind = '';
        if (publicQuotesEnabled) selectQuote(tr.dataset.code, tr.dataset.name);
        else selectLedgerHolding(tr.dataset.code, tr.dataset.name);
      });
    });
    host.scrollTop = scrollTop;
    const first = (d.rows || [])[0];
    if (quoteCode && externalQuote && !holdingRowFor(quoteCode)) {
      selectQuote(quoteCode, quoteName, true);
    } else if (quoteCode) {
      const still = (d.rows || []).some((r) => String(r.code) === String(quoteCode));
      if (!still && first) {
        if (publicQuotesEnabled) selectQuote(first.code, first.name || '');
        else selectLedgerHolding(first.code, first.name || '');
      } else if (!publicQuotesEnabled) {
        selectLedgerHolding(quoteCode, quoteName);
      }
    } else if (first) {
      if (publicQuotesEnabled) selectQuote(first.code, first.name || '');
      else selectLedgerHolding(first.code, first.name || '');
    }
  } catch (e) {
    host.innerHTML = `<div class="empty">持仓加载失败：${esc(e.message)}</div>`;
  } finally {
    holdBusy = false;
  }
}

let quoteCode = '';
let quoteName = '';
let externalQuote = false;
let quoteKind = '';
const boardLabels = new Map();
let quoteBars = 120;
let quoteFreq = 'minute';
let kChart;
let kUiBound = false;
let quoteRequestSeq = 0;

function quoteFrequencyLabel(freq) {
  return ({ minute: '分时', day: '日K', week: '周K', month: '月K', year: '年K' })[freq] || '行情';
}

function clearQuoteChart() {
  if (kChart) kChart.clear();
}

function selectLedgerHolding(code, name) {
  const selectedCode = String(code);
  quoteRequestSeq += 1;
  quoteCode = selectedCode;
  quoteName = name || quoteName;
  document.querySelectorAll('#htable .quote-row').forEach((tr) => {
    tr.classList.toggle('on', tr.dataset.code === selectedCode);
  });
  const row = document.querySelector(`#htable .quote-row[data-code="${CSS.escape(selectedCode)}"]`);
  $('qName').textContent = quoteName || selectedCode;
  $('qCode').textContent = `${selectedCode} · 本地账本观察`;
  paintSector(selectedCode);
  clearQuoteChart();
  const mark = row?.dataset?.mark || row?.dataset?.px;
  $('qPx').innerHTML = `<div class="px muted">${fmtNum(mark, 2)}</div><div class="muted">账本标记价</div><span class="quote-state">未启用公开行情</span>`;
  $('qNote').textContent = '已显示数量、成本、市值和浮盈亏。启用公开行情后可查看分时、日K、周K、月K、年K及机构预测。';
}

function bindKRange() {
  if (kUiBound) return;
  kUiBound = true;
  $('kRange').addEventListener('click', (ev) => {
    const btn = ev.target.closest('button');
    if (!btn) return;
    if (!publicQuotesEnabled) {
      $('qNote').textContent = '启用公开行情后可查看 K 线。';
      return;
    }
    $('kRange').querySelectorAll('button').forEach((b) => b.classList.remove('on'));
    btn.classList.add('on');
    quoteBars = Number(btn.dataset.bars);
    if (quoteCode) selectQuote(quoteCode, quoteName, true);
  });
  $('kFreq').addEventListener('click', (ev) => {
    const btn = ev.target.closest('button');
    if (!btn) return;
    if (!publicQuotesEnabled) {
      $('qNote').textContent = '启用公开行情后可查看分时与 K 线。';
      return;
    }
    $('kFreq').querySelectorAll('button').forEach((b) => b.classList.remove('on'));
    btn.classList.add('on');
    quoteFreq = btn.dataset.freq;
    if (quoteCode) selectQuote(quoteCode, quoteName, true);
  });
}

async function selectQuote(code, name, silent) {
  if (!publicQuotesEnabled) {
    selectLedgerHolding(code, name);
    return;
  }
  bindKRange();
  const selectedCode = String(code);
  const switching = quoteCode !== selectedCode;
  const requestSeq = ++quoteRequestSeq;
  quoteCode = selectedCode;
  quoteName = name || quoteName;
  document.querySelectorAll('#htable .quote-row').forEach((tr) => {
    tr.classList.toggle('on', tr.dataset.code === selectedCode);
  });
  $('qName').textContent = quoteName || selectedCode;
  $('qCode').textContent = selectedCode;
  paintSector(selectedCode);
  if (switching) {
    clearQuoteChart();
    $('qNote').textContent = `${quoteFrequencyLabel(quoteFreq)}读取中…`;
  }
  if (!silent) $('qPx').innerHTML = '<span class="muted">读取行情…</span>';
  try {
    if (quoteFreq === 'minute') {
      const [d, daily] = await Promise.all([
        api(`/api/intraday?code=${encodeURIComponent(selectedCode)}&days=1`),
        api(`/api/kline?code=${encodeURIComponent(selectedCode)}&freq=day&bars=120`).catch(() => null),
      ]);
      if (requestSeq !== quoteRequestSeq || quoteCode !== selectedCode) return;
      if (d.quote_name) $('qName').textContent = d.quote_name;
      renderIntradayHead(d);
      if (!Array.isArray(d.rows) || !d.rows.length) {
        clearQuoteChart();
        $('qNote').textContent = `分时暂无可用数据${d.error ? `：${d.error}` : ''}`;
        return;
      }
      $('qNote').textContent = `分时 · 腾讯分钟 · ${d.rows.length} 个数据点`;
      drawIntraday(d, daily);
      return;
    }
    const d = await api(`/api/kline?code=${encodeURIComponent(selectedCode)}&freq=${quoteFreq}&bars=${quoteBars}`);
    if (requestSeq !== quoteRequestSeq || quoteCode !== selectedCode) return;
    if (d.quote_name && quoteKind !== 'index') $('qName').textContent = d.quote_name;
    renderIntradayHead(d);
    const label = quoteFrequencyLabel(quoteFreq);
    if (!Array.isArray(d.bars) || !d.bars.length) {
      clearQuoteChart();
      $('qNote').textContent = `${label}暂无可用数据${d.error ? `：${d.error}` : ''}`;
      return;
    }
    const adj = ({ raw: '原始成交价', none: '未复权', qfq: '前复权', hfq: '后复权' })[d.adjust] || '复权口径待核';
    candleCaption = `${label} · ${adj} · 最后一根 ${d.as_of || '—'}`;
    $('qNote').textContent = candleCaption;
    drawKline(d);
  } catch (e) {
    if (requestSeq !== quoteRequestSeq || quoteCode !== selectedCode) return;
    clearQuoteChart();
    $('qPx').innerHTML = `<span class="down">${esc(e.message)}</span>`;
    $('qNote').textContent = `${quoteFrequencyLabel(quoteFreq)}暂不可用：${e.message || '请求失败'}`;
  }
}

function renderIntradayHead(d) {
  const chg = d.chg;
  const cls = dirClass(chg);
  const quoteState = String(d.quote_state || '').trim().toLowerCase();
  const live = quoteState === 'trading';
  const stateLabel = d.quote_state_label || {
    trading: '实时', lunch_break: '午间休市', delayed: '延迟', stale: '延迟', closed: '收盘', close: '收盘',
    pre_open: '待开盘', preopen: '待开盘', unavailable: '暂无行情', unknown: '行情状态未知',
  }[quoteState] || (quoteState ? '行情状态待确认' : '行情状态未知');
  const badge = `<span class="quote-state${live ? ' live' : ''}">${esc(stateLabel)}</span>`;
  const hasLast = isNum(d.last);
  const body = hasLast
    ? `<div class="px ${cls}">${fmtNum(d.last, 2)}</div>
       <div class="${cls}">${isNum(chg) ? fmtSignedPct(chg) : '涨跌 —'} <span class="muted">${esc(d.quote_time || d.as_of || '')}</span></div>`
    : `<div class="px muted">—</div><div class="muted">昨收 ${fmtNum(d.pre_close, 2)} · 尚未开盘</div>`;
  $('qPx').innerHTML = body + badge;
}

function volumePace(volumes) {
  const labels = volumes.map(() => null);
  const counts = { heavy: 0, double: 0, light: 0, flat: 0 };
  for (let index = 5; index < volumes.length; index += 1) {
    let sum = 0;
    let usable = true;
    for (let cursor = index - 5; cursor < index; cursor += 1) {
      const value = Number(volumes[cursor]);
      if (!Number.isFinite(value) || value < 0) {
        usable = false;
        break;
      }
      sum += value;
    }
    const current = Number(volumes[index]);
    const baseline = sum / 5;
    if (!usable || !(baseline > 0) || !Number.isFinite(current) || current < 0) continue;
    const ratio = current / baseline;
    let kind = 'flat';
    if (ratio >= 1.5) kind = 'heavy';
    else if (ratio <= 0.7) kind = 'light';
    labels[index] = { kind, ratio };
    counts[kind] += 1;
    if (ratio >= 2) counts.double += 1;
  }
  let streak = 0;
  let streakKind = null;
  for (let index = labels.length - 1; index >= 0; index -= 1) {
    const label = labels[index];
    if (!label) break;
    if (streakKind == null) streakKind = label.kind;
    if (label.kind !== streakKind) break;
    streak += 1;
  }
  return { labels, counts, streak, streakKind };
}

function volumePaceView(volumes, unit) {
  const pace = volumePace(volumes);
  const names = { heavy: '放量', light: '缩量', flat: '平量' };
  const last = pace.labels[pace.labels.length - 1];
  const comparable = pace.counts.heavy + pace.counts.light + pace.counts.flat;
  const stats = [{
    label: '当前',
    value: last ? `${last.ratio.toFixed(2)}倍 ${names[last.kind]}` : '不满5根',
    tone: last ? last.kind : 'flat',
  }];
  if (pace.streak >= 2 && pace.streakKind) {
    stats.push({
      label: '连续',
      value: `${pace.streak}${unit}${names[pace.streakKind]}`,
      tone: pace.streakKind,
    });
  }
  stats.push(
    { label: '放量', value: String(pace.counts.heavy), tone: 'heavy' },
    { label: '倍量', value: String(pace.counts.double), tone: 'heavy' },
    { label: '缩量', value: String(pace.counts.light), tone: 'light' },
    { label: '平量', value: String(pace.counts.flat), tone: 'flat' },
  );
  return {
    pace,
    hint: `每一根和它前面 5 ${unit}的均量比。放量不低于 1.5 倍，倍量不低于 2 倍并计在放量里，缩量不高于 0.7 倍，其余是平量。这窗口可比较 ${comparable} ${unit}。`,
    stats,
  };
}

function paintChartReading(closes, rows) {
  const host = $('kRead');
  if (!host) return;
  const items = [];
  if (overlays.trend) {
    const channel = regressionChannel(closes);
    if (channel) {
      const last = closes[closes.length - 1];
      const upper = channel.upper[channel.upper.length - 1];
      const lower = channel.lower[channel.lower.length - 1];
      const direction = channel.change > 0.01 ? '上升' : channel.change < -0.01 ? '下降' : '走平';
      const windowChange = closes[0] ? last / closes[0] - 1 : 0;
      let place = '最新收盘还在通道里面。';
      if (last > upper) place = '最新收盘到了通道上沿外面，比这段趋势偏高。';
      else if (last < lower) place = '最新收盘到了通道下沿外面，比这段趋势偏低。';
      items.push({
        color: '#f59e0b',
        title: '趋势',
        text: `这段是${direction}的，区间 ${fmtSignedPct(windowChange)}。琥珀实线是回归趋势，同色虚线是上下 1 倍残差。${place}`,
      });
    }
  }
  if (overlays.ma) {
    const latest = (period) => {
      const series = movingAverage(closes, period);
      for (let i = series.length - 1; i >= 0; i -= 1) if (series[i] != null) return series[i];
      return null;
    };
    const m5 = latest(5);
    const m10 = latest(10);
    const m20 = latest(20);
    const m60 = latest(60);
    let text = '均线还没凑够计算天数。';
    if (m5 != null && m10 != null && m20 != null && m60 != null) {
      if (m5 > m10 && m10 > m20 && m20 > m60) text = '黄线在蓝线上，蓝线在紫线上，紫线在青线上。短周期比长周期强。';
      else if (m5 < m10 && m10 < m20 && m20 < m60) text = '短周期均线压在长周期下面。这段偏弱。';
      else text = '四条均线缠在一起，没有排成单边。';
    }
    items.push({ color: '#fbbf24', title: '均线', text: `MA5 黄、MA10 蓝、MA20 紫、MA60 青。${text}` });
  }
  const lastBar = rows[rows.length - 1];
  const prevBar = rows.length > 1 ? rows[rows.length - 2] : null;
  if (lastBar) {
    const open = Number(lastBar.o);
    const close = Number(lastBar.c);
    const high = Number(lastBar.h);
    const low = Number(lastBar.l);
    const span = Math.max(high - low, 0);
    const upper = high - Math.max(open, close);
    const lower = Math.min(open, close) - low;
    const candle = close >= open ? '阳线' : '阴线';
    const change = prevBar && Number(prevBar.c) ? close / Number(prevBar.c) - 1 : (open ? close / open - 1 : 0);
    let shape = '实体和影线差不多。';
    if (span > 0 && upper / span > 0.45) shape = '上影偏长，冲高后回落。';
    else if (span > 0 && lower / span > 0.45) shape = '下影偏长，下探后收回。';
    else if (span > 0 && Math.abs(close - open) / span > 0.7) shape = '实体占了大部分振幅。';
    items.push({
      color: close >= open ? '#f43f5e' : '#10b981',
      title: 'K线',
      text: `最后一根是${candle}，相对前收 ${fmtSignedPct(change)}。${shape}`,
    });
  }
  const band = bollinger(closes, 20, 2);
  const bandMid = band.mid[band.mid.length - 1];
  const bandUpper = band.upper[band.upper.length - 1];
  const bandLower = band.lower[band.lower.length - 1];
  if (bandMid != null) {
    const last = closes[closes.length - 1];
    let place = '收在中轨和下轨之间。';
    if (last > bandUpper) place = '收在上轨外面。';
    else if (last < bandLower) place = '收在下轨外面。';
    else if (last >= bandMid) place = '收在中轨和上轨之间。';
    const earlierUpper = band.upper[band.upper.length - 6];
    const earlierMid = band.mid[band.mid.length - 6];
    const earlierLower = band.lower[band.lower.length - 6];
    let widthText = '';
    if (earlierUpper != null && earlierMid && bandMid) {
      const nowWidth = (bandUpper - bandLower) / bandMid;
      const thenWidth = (earlierUpper - earlierLower) / earlierMid;
      widthText = nowWidth < thenWidth ? '带宽比五根K线前更窄。' : '带宽比五根K线前更宽。';
    }
    items.push({
      color: '#94a3b8',
      title: '布林',
      text: `${overlays.boll ? '灰虚线是 20 日、2 倍标准差。' : '布林线这会儿没画出来。'}${place}${widthText}`,
    });
  }
  const volumes = rows.map((row) => Number(row.v) || 0);
  const paceView = volumePaceView(volumes, '根');
  const lastLabel = paceView.pace.labels[paceView.pace.labels.length - 1];
  if (volumes.some((value) => value > 0)) {
    items.push({
      color: '#fb923c',
      title: '量能',
      hint: paceView.hint,
      stats: paceView.stats,
    });
  }
  const amountValues = rows.map((row) => (isNum(row.a) ? Number(row.a) : null));
  const knownAmounts = amountValues.filter((value) => value != null);
  const amountMa = movingAverage(amountValues.map((value) => value || 0), 5);
  const lastAmount = amountValues[amountValues.length - 1];
  const averageAmount = amountMa[amountMa.length - 1];
  if (knownAmounts.length && lastAmount != null && averageAmount) {
    const ratio = lastAmount / averageAmount;
    let pace = '和近 5 根差不多。';
    if (ratio >= 1.5) pace = '金额明显放大。';
    else if (ratio >= 1.1) pace = '金额高于近 5 根。';
    else if (ratio <= 0.7) pace = '金额明显缩小。';
    items.push({
      color: '#a3e635',
      title: '成交额',
      text: `最后一根 ${fmtCompact(lastAmount)}元，5 根均额 ${fmtCompact(averageAmount)}元，${pace}黄绿线在成交量图右侧。`,
    });
  }
  if (lastBar && lastLabel) {
    const close = Number(lastBar.c);
    const prevClose = prevBar ? Number(prevBar.c) : Number(lastBar.o);
    const rising = close >= prevClose;
    const heavy = lastLabel.kind === 'heavy';
    const light = lastLabel.kind === 'light';
    let verdict = '最后一根的量和前 5 根差不多，没有单独的放量或缩量。';
    if (rising && heavy) verdict = '价格往上，同时放量，上冲有成交配合。';
    else if (rising && light) verdict = '价格往上，同时缩量，上冲缺少成交配合。';
    else if (!rising && heavy) verdict = '价格往下，同时放量，卖压更重。';
    else if (!rising && light) verdict = '价格往下，同时缩量，下跌没有放量确认。';
    items.push({ color: rising ? '#f43f5e' : '#10b981', title: '结论', text: verdict });
  }
  if (overlays.hs) {
    const pattern = findHeadShoulders(rows);
    items.push({
      color: '#e879f9',
      title: '头肩顶',
      text: pattern
        ? `粉线连着 ${rows[pattern.troughLeft].d} 和 ${rows[pattern.troughRight].d} 两个低点。左肩 ${rows[pattern.left].d}，头 ${rows[pattern.head].d}，右肩 ${rows[pattern.right].d}。这是形状，不是卖出指令。`
        : '这根图里没有同时满足肩低、头高、两肩接近的头肩顶，所以不画。',
    });
  }
  if (!items.length) {
    host.hidden = true;
    host.replaceChildren();
    return;
  }
  host.hidden = false;
  host.replaceChildren();
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'k-read-row';
    row.dataset.title = item.title;
    const swatch = document.createElement('i');
    swatch.style.background = item.color;
    const title = document.createElement('b');
    title.textContent = item.title;
    const body = document.createElement('span');
    if (item.stats) {
      body.className = 'vol-stats';
      item.stats.forEach((stat) => {
        const chip = document.createElement('span');
        chip.className = `tone-${stat.tone}`;
        const name = document.createElement('i');
        name.textContent = stat.label;
        const value = document.createElement('b');
        value.textContent = stat.value;
        chip.append(name, value);
        body.append(chip);
      });
      row.title = item.hint || '';
    } else {
      body.textContent = item.text;
    }
    row.append(swatch, title, body);
    host.append(row);
  });
}

function volumePrints(volumes) {
  const values = volumes.map((value) => Number(value) || 0);
  let rises = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[index - 1]) rises += 1;
  }
  const cumulative = values.length >= 8 && rises / (values.length - 1) >= 0.6 && values[values.length - 1] > values[0];
  if (!cumulative) {
    return { prints: values, total: values.reduce((sum, value) => sum + value, 0) };
  }
  const prints = values.map((value, index) => Math.max(0, value - (index === 0 ? 0 : values[index - 1])));
  return { prints, total: values[values.length - 1] };
}

function sessionVolumeRatio(rows, daily) {
  const volumes = rows.map((row) => Number(row.v) || 0);
  const printed = volumePrints(volumes);
  const elapsed = Math.min(Math.max(volumes.length, 1), 240);
  const today = String(rows[rows.length - 1]?.t || '').slice(0, 10);
  const history = ((daily && daily.bars) || [])
    .filter((bar) => bar && String(bar.d || '').slice(0, 10) !== today && Number(bar.v) > 0)
    .map((bar) => Number(bar.v))
    .slice(-5);
  if (history.length < 5 || !(printed.total > 0)) return null;
  const average = history.reduce((sum, value) => sum + value, 0) / history.length;
  const perMinute = average / 240;
  if (!(perMinute > 0)) return null;
  return { ratio: printed.total / (perMinute * elapsed), cumulative: printed.total, average, elapsed, prints: printed.prints };
}

function paintSessionVolume(rows, daily) {
  const host = $('kRead');
  if (!host) return;
  const volumes = (rows || []).map((row) => Number(row.v) || 0);
  if (!volumes.some((value) => value > 0)) {
    host.hidden = true;
    host.replaceChildren();
    return;
  }
  const session = sessionVolumeRatio(rows, daily);
  const minute = volumePace((session && session.prints) || volumePrints(volumes).prints).labels[volumes.length - 1];
  const names = { heavy: '放量', light: '缩量', flat: '平量' };
  const toneOf = (ratio) => (ratio >= 1.5 ? 'heavy' : ratio <= 0.7 ? 'light' : 'flat');
  const stats = [];
  if (session) {
    const tone = toneOf(session.ratio);
    const name = session.ratio >= 2 ? '倍量' : names[tone];
    stats.push({ label: '量比', value: `${session.ratio.toFixed(2)} ${name}`, tone });
    stats.push({ label: '今日累计', value: `${fmtCompact(session.cumulative)}手`, tone: 'flat' });
    stats.push({ label: '5日均量', value: `${fmtCompact(session.average)}手`, tone: 'flat' });
  } else {
    stats.push({ label: '量比', value: '5日均量不足', tone: 'flat' });
  }
  if (minute) {
    stats.push({
      label: '这一分钟',
      value: `${minute.ratio.toFixed(2)}倍 ${names[minute.kind]}`,
      tone: minute.kind,
    });
  }
  host.hidden = false;
  host.replaceChildren();
  const row = document.createElement('div');
  row.className = 'k-read-row';
  row.dataset.title = '量能';
  row.title = '量比 = 今日累计成交量 / (近5个交易日平均每分钟成交量 × 已开市分钟)。这一分钟再和它前面5分钟比。不低于1.5倍为放量，不低于2倍为倍量，不高于0.7倍为缩量。';
  const swatch = document.createElement('i');
  swatch.style.background = '#fb923c';
  const title = document.createElement('b');
  title.textContent = '量能';
  const body = document.createElement('span');
  body.className = 'vol-stats';
  stats.forEach((stat) => {
    const chip = document.createElement('span');
    chip.className = `tone-${stat.tone}`;
    const name = document.createElement('i');
    name.textContent = stat.label;
    const value = document.createElement('b');
    value.textContent = stat.value;
    chip.append(name, value);
    body.append(chip);
  });
  row.append(swatch, title, body);
  host.append(row);
}

function drawIntraday(d, daily) {
  const host = $('chartK');
  if (!host) return;
  if (!kChart) kChart = echarts.init(host, null, { renderer: 'canvas' });
  const rows = (d.rows || []).filter((r) => isNum(r.p));
  if (!rows.length) {
    kChart.clear();
    paintSessionVolume([], null);
    return;
  }
  const pre = isNum(rows[0].pre) ? Number(rows[0].pre) : (isNum(d.pre_close) ? Number(d.pre_close) : null);
  const last = Number(rows[rows.length - 1].p);
  const rising = pre != null && last >= pre;
  const line = rising ? '#f43f5e' : '#10b981';
  const times = rows.map((r) => String(r.t || '').slice(11) || String(r.t || ''));
  const prices = rows.map((r) => Number(r.p));
  const avg = [];
  let cum = 0;
  for (const r of rows) { cum += Number(r.p); avg.push(cum / (avg.length + 1)); }
  const vols = volumePrints(rows.map((r) => Number(r.v) || 0)).prints;
  const minutePace = volumePace(vols);
  kChart.setOption({
    animation: false,
    grid: [
      { left: 58, right: 58, top: 18, height: '58%' },
      { left: 58, right: 58, top: '80%', height: '13%' },
    ],
    axisPointer: { link: [{ xAxisIndex: [0, 1] }] },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' },
      formatter: (ps) => ps.map((p) => {
        const i = p.dataIndex; const base = pre || prices[0];
        const chg = base ? (prices[i] / base - 1) : 0;
        const pace = minutePace.labels[i];
        const paceText = pace
          ? `<br/>量能 ${pace.ratio.toFixed(2)}倍 ${pace.kind === 'heavy' ? '放量' : pace.kind === 'light' ? '缩量' : '平量'}`
          : '';
        return `${times[i]}<br/>价 <b>${fmtNum(prices[i], 2)}</b> (${fmtSignedPct(chg)})<br/>均 ${fmtNum(avg[i], 2)}<br/>量 ${fmtCompact(vols[i])}手${paceText}`;
      }).join('<br/>') },
    xAxis: [
      { type: 'category', data: times, boundaryGap: false,
        axisLabel: { color: '#94a3b8', interval: Math.floor(times.length / 5) || 1 },
        axisLine: { lineStyle: { color: '#334155' } } },
      { type: 'category', gridIndex: 1, data: times, axisLabel: { show: false }, axisTick: { show: false } },
    ],
    yAxis: [
      { scale: true, position: 'right', axisLabel: { color: '#94a3b8', formatter: (v) => v.toFixed(2) },
        splitLine: { lineStyle: { color: '#232e47' } } },
      { scale: true, position: 'right', gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } },
    ],
    series: [
      { type: 'line', data: prices, showSymbol: false, lineStyle: { width: 1.6, color: line },
        areaStyle: { color: rising ? 'rgba(244,63,94,.14)' : 'rgba(16,185,129,.14)' },
        markLine: pre == null ? undefined : {
          silent: true, symbol: 'none',
          lineStyle: { color: '#94a3b8', type: 'dashed', width: 1 },
          label: { formatter: fmtNum(pre, 2), color: '#94a3b8', fontSize: 10, position: 'insideStartTop' },
          data: [{ yAxis: pre }] } },
      { type: 'line', data: avg, showSymbol: false, lineStyle: { width: 1, color: '#fbbf24' } },
      { type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: vols.map((value, index) => {
        const previous = index === 0 ? pre : prices[index - 1];
        const up = previous == null || prices[index] >= previous;
        return { value, itemStyle: { color: up ? 'rgba(244,63,94,.55)' : 'rgba(16,185,129,.55)' } };
      }) },
    ],
  }, true);
  kChart.resize();
  paintSessionVolume(rows, daily);
}

function statChip(label, value, cls) {
  return `<div class="stat"><span>${esc(label)}</span><b class="${cls || ''}">${value}</b></div>`;
}

/* Benchmark tape. Red=up, green=down, matching the A-share convention. */
async function loadIndices() {
  const host = $('tape');
  if (!host) return;
  if (!publicQuotesEnabled) {
    host.innerHTML = '<span class="tape-load">启用公开行情后显示大盘指数</span>';
    return;
  }
  try {
    const d = await api('/api/indices');
    const rows = d.rows || [];
    if (!rows.length) { host.innerHTML = '<span class="tape-load">大盘指数暂不可用</span>'; return; }
    const b = d.breadth || {};
    const bd = (isNum(b.up) && isNum(b.down))
      ? `<div class="idx-bd" title="全A涨跌家数">
           <span class="bd-u">涨 <b>${b.up}</b></span>
           <span class="bd-d">跌 <b>${b.down}</b></span>
           <span class="bd-f">平 <b>${b.flat || 0}</b></span>
           <span class="bd-bar" aria-hidden="true">
             <i class="bd-up" style="flex:${b.up || 0}"></i>
             <i class="bd-fl" style="flex:${b.flat || 0}"></i>
             <i class="bd-dn" style="flex:${b.down || 0}"></i>
           </span>
         </div>` : '';
    host.innerHTML = bd + rows.map((r) => {
      const cls = r.state === 'pre_open' || !isNum(r.pct_chg) ? 'flat'
        : (Number(r.pct_chg) > 0 ? 'up' : Number(r.pct_chg) < 0 ? 'down' : 'flat');
      const chg = isNum(r.pct_chg)
        ? `${r.chg > 0 ? '+' : ''}${fmtNum(r.chg, 2)}  ${fmtSignedPct(r.pct_chg)}`
        : '待开盘';
      // 成交额：腾讯给"亿元"，直接显示
      const amt = isNum(r.amount_yi) ? `${fmtNum(r.amount_yi, 0)}亿` : '';
      // 指数不提供涨跌家数；用高低点区间做市场宽度替代指标
      const rng = (isNum(r.high) && isNum(r.low) && r.price)
        ? `振幅 ${(((r.high - r.low) / r.pre_close) * 100 || 0).toFixed(2)}%` : '';
      const symbol = `${r.code === '000001' ? 'sh' : 'sz'}${r.code}`;
      const on = quoteKind === 'index' && quoteCode === symbol ? ' on' : '';
      return `<button type="button" class="idx ${cls}${on}" data-code="${esc(symbol)}" data-name="${esc(r.name)}">
        <span class="idx-n">${esc(r.name)}</span>
        <span class="idx-p">${isNum(r.price) ? fmtNum(r.price, 2) : '—'}</span>
        <span class="idx-c">${chg}</span>
        <span class="idx-x">${rng}</span>
        <span class="idx-x">${amt}</span>
      </button>`;
    }).join('');
    host.querySelectorAll('.idx').forEach((button) => {
      button.addEventListener('click', () => openIndex(button.dataset.code, button.dataset.name));
    });
  } catch (e) {
    host.innerHTML = `<span class="tape-load">大盘指数载入失败：${esc(e.message)}</span>`;
  }
}

function applyWatchSort(key, dir, toggle) {
  const host = $('htable');
  const seg = $('watchSort');
  if (!host || !seg) return;
  const cur = state.sortState.htable || {};
  const next = toggle && cur.key === key && cur.dir === 'desc' ? 'asc' : (toggle ? 'desc' : dir);
  state.sortState.htable = { key, dir: next };
  const rows = sortTableRows(Array.from(host.querySelectorAll('.quote-row')), key, next);
  rows.forEach((row) => host.appendChild(row));
  seg.querySelectorAll('button').forEach((btn) => {
    const on = btn.dataset.key === key;
    btn.classList.toggle('on', on);
    const base = { chg: '涨跌', since: '收益', pnl: '盈亏', mv: '市值' }[btn.dataset.key] || btn.dataset.key;
    btn.textContent = on ? `${base} ${next === 'desc' ? '↓' : '↑'}` : base;
  });
}

function bareCode(value) {
  const digits = String(value || '').replace(/\D/g, '');
  return digits.length >= 6 ? digits.slice(-6) : '';
}

function listingBoardLabel(code) {
  const bare = bareCode(code);
  if (bare.startsWith('688')) return '科创板';
  if (bare.startsWith('300') || bare.startsWith('301')) return '创业板';
  if (bare.startsWith('4') || bare.startsWith('8') || bare.startsWith('92')) return '北交所';
  if (bare.startsWith('6') || bare.startsWith('9')) return '沪市主板';
  return bare ? '深市主板' : '—';
}

function holdingRowFor(code) {
  const bare = bareCode(code);
  if (!bare) return null;
  return document.querySelector(`#htable .quote-row[data-code$=".${bare}"]`)
    || document.querySelector(`#htable .quote-row[data-code="${CSS.escape(bare)}"]`);
}

function rememberBoard(code, name) {
  const bare = bareCode(code);
  if (!bare || !name) return;
  boardLabels.set(bare, name);
  if (bareCode(quoteCode) === bare) paintSector(quoteCode);
}

function openIndex(symbol, name) {
  if (!symbol) return;
  if (!publicQuotesEnabled) {
    $('qNote').textContent = '启用公开行情后可查看指数行情。';
    return;
  }
  externalQuote = true;
  quoteKind = 'index';
  if (window.QuantLabQuote) window.QuantLabQuote.kind = 'index';
  document.querySelectorAll('#htable .quote-row').forEach((row) => row.classList.remove('on'));
  selectQuote(symbol, name || symbol);
  document.querySelectorAll('.idx').forEach((button) => {
    button.classList.toggle('on', button.dataset.code === symbol);
  });
  const board = document.getElementById('terminal-board');
  const leaders = document.getElementById('terminal-leaders');
  if (board) board.hidden = true;
  if (leaders) leaders.hidden = true;
}

function openSymbol(code, name) {
  const bare = bareCode(code);
  if (!bare) return;
  quoteKind = '';
  if (window.QuantLabQuote) window.QuantLabQuote.kind = '';
  const box = $('symbolSearchResults');
  if (box) box.hidden = true;
  const row = holdingRowFor(bare);
  if (row) {
    externalQuote = false;
    row.click();
    row.scrollIntoView({ block: 'nearest' });
    return;
  }
  if (!publicQuotesEnabled) {
    $('qNote').textContent = '启用公开行情后才能查询持仓以外的股票。';
    return;
  }
  externalQuote = true;
  selectQuote(bare, name || bare);
  document.dispatchEvent(new CustomEvent('quantlab:open-symbol', { detail: { code: bare, name: name || bare } }));
}

function paintSector(code) {
  const host = $('qSector');
  if (!host) return;
  if (quoteKind === 'index') {
    host.innerHTML = [
      statChip('品种', '指数'),
      statChip('代码', esc(code)),
      statChip('持仓', '指数不在账本内'),
    ].join('');
    return;
  }
  const tr = document.querySelector(`#htable .quote-row[data-code="${CSS.escape(code)}"]`) || holdingRowFor(code);
  const sector = boardLabels.get(bareCode(code)) || '';
  if (!tr) {
    host.innerHTML = [
      statChip('查询', '非持仓'),
      statChip('上市板块', esc(listingBoardLabel(code))),
      statChip('所属板块', sector ? esc(sector) : '读取中'),
      statChip('持仓数量', '—'),
      statChip('成本', '—'),
      statChip('浮盈亏', '非持仓，无账本盈亏'),
    ].join('');
    return;
  }
  const shownSector = sector || tr.dataset.sector || '';
  const listingBoard = tr.dataset.listingBoard || '—';
  const since = tr.dataset.since;
  const pnl = tr.dataset.pnl;
  host.innerHTML = [
    statChip('建仓行业', esc(tr.dataset.industry || '—')),
    statChip('上市板块', esc(listingBoard)),
    statChip('所属板块', shownSector ? esc(shownSector) : '读取中'),
    statChip('行情状态', esc(tr.dataset.quoteStateLabel || '账本标记价')),
    statChip('行情来源', esc(tr.dataset.quoteSource || '—')),
    statChip('行情时间', esc(tr.dataset.quoteTime || '—')),
    statChip('持仓数量', isNum(tr.dataset.qty) ? `${fmtNum(tr.dataset.qty, 0)} 股` : '—'),
    statChip('成本', fmtNum(tr.dataset.cost, 2)),
    statChip('当前价格', fmtNum(tr.dataset.px, 2)),
    statChip('账本标记价', fmtNum(tr.dataset.mark, 2)),
    statChip('市值', fmtMoneyCny(tr.dataset.mv)),
    statChip('持仓收益', fmtSignedPct(since), dirClass(since)),
    statChip('浮盈亏', fmtSignedMoneyCny(pnl), dirClass(pnl)),
  ].join('');
}

const overlays = { trend: true, ma: false, boll: false, hs: false };
let lastCandlePayload = null;
let candleCaption = '';

function movingAverage(values, period) {
  const out = values.map(() => null);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}

function bollinger(values, period, width) {
  const mid = movingAverage(values, period);
  const upper = values.map(() => null);
  const lower = values.map(() => null);
  for (let i = period - 1; i < values.length; i += 1) {
    let square = 0;
    for (let j = i - period + 1; j <= i; j += 1) {
      const delta = values[j] - mid[i];
      square += delta * delta;
    }
    const deviation = Math.sqrt(square / period);
    upper[i] = mid[i] + width * deviation;
    lower[i] = mid[i] - width * deviation;
  }
  return { mid, upper, lower };
}

function swingHighs(values, span) {
  const points = [];
  for (let i = span; i < values.length - span; i += 1) {
    let peak = true;
    for (let j = i - span; j <= i + span; j += 1) {
      if (j !== i && values[j] >= values[i]) {
        peak = false;
        break;
      }
    }
    if (peak) points.push(i);
  }
  return points;
}

function swingLows(values, span) {
  const points = [];
  for (let i = span; i < values.length - span; i += 1) {
    let trough = true;
    for (let j = i - span; j <= i + span; j += 1) {
      if (j !== i && values[j] <= values[i]) {
        trough = false;
        break;
      }
    }
    if (trough) points.push(i);
  }
  return points;
}

function latestSwingLine(points, values, rising) {
  for (let later = points.length - 1; later >= 1; later -= 1) {
    const end = points[later];
    for (let earlier = later - 1; earlier >= Math.max(0, later - 3); earlier -= 1) {
      const start = points[earlier];
      const aligned = rising ? values[end] > values[start] : values[end] < values[start];
      if (aligned) return { start, end };
    }
  }
  return null;
}

function regressionChannel(values) {
  const count = values.length;
  if (count < 20) return null;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumXY = 0;
  for (let i = 0; i < count; i += 1) {
    sumX += i;
    sumY += values[i];
    sumXX += i * i;
    sumXY += i * values[i];
  }
  const denom = count * sumXX - sumX * sumX;
  if (!denom) return null;
  const slope = (count * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / count;
  const mid = values.map((_, index) => intercept + slope * index);
  let square = 0;
  for (let i = 0; i < count; i += 1) {
    const error = values[i] - mid[i];
    square += error * error;
  }
  const deviation = Math.sqrt(square / count);
  return {
    mid,
    upper: mid.map((value) => value + deviation),
    lower: mid.map((value) => value - deviation),
    change: mid[0] ? mid[count - 1] / mid[0] - 1 : 0,
  };
}

function findHeadShoulders(rows) {
  if (rows.length < 30) return null;
  const highs = rows.map((row) => Number(row.h));
  const lows = rows.map((row) => Number(row.l));
  const span = rows.length > 180 ? 5 : 3;
  const peaks = swingHighs(highs, span);
  let best = null;
  for (let cursor = 0; cursor < peaks.length - 2; cursor += 1) {
    const left = peaks[cursor];
    const head = peaks[cursor + 1];
    const right = peaks[cursor + 2];
    const leftHigh = highs[left];
    const headHigh = highs[head];
    const rightHigh = highs[right];
    if (!(headHigh > leftHigh && headHigh > rightHigh)) continue;
    if ((headHigh - leftHigh) / headHigh < 0.03 || (headHigh - rightHigh) / headHigh < 0.03) continue;
    if (Math.abs(leftHigh - rightHigh) / headHigh > 0.08) continue;
    let troughLeft = left + 1;
    let troughRight = head + 1;
    for (let i = left + 1; i < head; i += 1) if (lows[i] < lows[troughLeft]) troughLeft = i;
    for (let i = head + 1; i < right; i += 1) if (lows[i] < lows[troughRight]) troughRight = i;
    if (troughLeft >= head || troughRight >= right) continue;
    if (lows[troughLeft] >= Math.min(leftHigh, rightHigh) || lows[troughRight] >= Math.min(leftHigh, rightHigh)) continue;
    best = { left, head, right, troughLeft, troughRight };
  }
  return best;
}

function neckline(length, start, startPrice, end, endPrice, extendTo) {
  const data = Array(length).fill(null);
  if (end === start) return data;
  const slope = (endPrice - startPrice) / (end - start);
  for (let i = start; i <= extendTo && i < length; i += 1) data[i] = startPrice + slope * (i - start);
  return data;
}

function lineSeries(name, data, color, extra) {
  return {
    name,
    type: 'line',
    data,
    showSymbol: false,
    connectNulls: false,
    lineStyle: { width: 1, color },
    emphasis: { disabled: true },
    ...(extra || {}),
  };
}

function drawKline(d) {
  const host = $('chartK');
  if (!host) return;
  lastCandlePayload = d;
  if (!kChart) kChart = echarts.init(host, null, { renderer: 'canvas' });
  const rows = d.bars || [];
  const dates = rows.map((r) => r.d);
  const closes = rows.map((r) => Number(r.c));
  const ohlc = rows.map((r) => [r.o, r.c, r.l, r.h]);
  const vol = rows.map((r) => r.v);
  const colors = rows.map((r) => (r.c >= r.o ? '#f43f5e' : '#10b981'));
  const series = [
    {
      name: 'K线',
      type: 'candlestick',
      data: ohlc,
      itemStyle: { color: '#f43f5e', color0: '#10b981', borderColor: '#f43f5e', borderColor0: '#10b981' },
    },
  ];
  let trendText = '';
  if (overlays.trend) {
    const channel = regressionChannel(closes);
    if (!channel) {
      trendText = '趋势样本不足';
    } else {
      series.push(lineSeries('趋势', channel.mid, '#f59e0b', { lineStyle: { width: 1.6, color: '#f59e0b' } }));
      series.push(lineSeries('趋势上轨', channel.upper, 'rgba(245,158,11,.7)', { lineStyle: { width: 1, type: 'dashed', color: 'rgba(245,158,11,.7)' } }));
      series.push(lineSeries('趋势下轨', channel.lower, 'rgba(245,158,11,.7)', { lineStyle: { width: 1, type: 'dashed', color: 'rgba(245,158,11,.7)' } }));
      const direction = channel.change > 0.01 ? '上升' : channel.change < -0.01 ? '下降' : '走平';
      const windowChange = closes[0] ? closes[closes.length - 1] / closes[0] - 1 : 0;
      const span = rows.length > 180 ? 5 : 3;
      if (channel.change >= 0) {
        const lows = rows.map((row) => Number(row.l));
        const support = latestSwingLine(swingLows(lows, span), lows, true);
        if (support) {
          series.push(lineSeries(
            '上升趋势线',
            neckline(rows.length, support.start, lows[support.start], support.end, lows[support.end], rows.length - 1),
            '#22d3ee',
            { lineStyle: { width: 1.4, color: '#22d3ee' } },
          ));
        }
      } else {
        const highs = rows.map((row) => Number(row.h));
        const resistance = latestSwingLine(swingHighs(highs, span), highs, false);
        if (resistance) {
          series.push(lineSeries(
            '下降趋势线',
            neckline(rows.length, resistance.start, highs[resistance.start], resistance.end, highs[resistance.end], rows.length - 1),
            '#fb7185',
            { lineStyle: { width: 1.2, color: '#fb7185' } },
          ));
        }
      }
      trendText = `趋势${direction}，区间 ${fmtSignedPct(windowChange)}`;
    }
  }
  if (overlays.ma) {
    series.push(lineSeries('MA5', movingAverage(closes, 5), '#fbbf24', { lineStyle: { width: 1.2, color: '#fbbf24' } }));
    series.push(lineSeries('MA10', movingAverage(closes, 10), '#38bdf8', { lineStyle: { width: 1.2, type: 'dashed', color: '#38bdf8' } }));
    series.push(lineSeries('MA20', movingAverage(closes, 20), '#c084fc', { lineStyle: { width: 1.2, type: 'dotted', color: '#c084fc' } }));
    series.push(lineSeries('MA60', movingAverage(closes, 60), '#2dd4bf', { lineStyle: { width: 1.4, color: '#2dd4bf' } }));
  }
  if (overlays.boll) {
    const band = bollinger(closes, 20, 2);
    series.push(lineSeries('布林上轨', band.upper, 'rgba(148,163,184,.7)', { lineStyle: { width: 1, type: 'dashed', color: 'rgba(148,163,184,.7)' } }));
    series.push(lineSeries('布林中轨', band.mid, 'rgba(148,163,184,.45)'));
    series.push(lineSeries('布林下轨', band.lower, 'rgba(148,163,184,.7)', { lineStyle: { width: 1, type: 'dashed', color: 'rgba(148,163,184,.7)' } }));
  }
  let patternText = '';
  if (overlays.hs) {
    const pattern = findHeadShoulders(rows);
    if (pattern) {
      const lows = rows.map((row) => Number(row.l));
      const highs = rows.map((row) => Number(row.h));
      series[0].markPoint = {
        symbol: 'circle',
        symbolSize: 7,
        label: { color: '#e2e8f0', fontSize: 10, position: 'top' },
        itemStyle: { color: '#f59e0b' },
        data: [
          { coord: [dates[pattern.left], highs[pattern.left]], value: '左肩' },
          { coord: [dates[pattern.head], highs[pattern.head]], value: '头' },
          { coord: [dates[pattern.right], highs[pattern.right]], value: '右肩' },
        ],
      };
      series.push(lineSeries(
        '颈线',
        neckline(rows.length, pattern.troughLeft, lows[pattern.troughLeft], pattern.troughRight, lows[pattern.troughRight], pattern.right),
        '#e879f9',
        { lineStyle: { width: 1.2, type: 'dashed', color: '#e879f9' } },
      ));
      patternText = `头肩顶 ${dates[pattern.left]} / ${dates[pattern.head]} / ${dates[pattern.right]}`;
    } else {
      patternText = '头肩顶未成型';
    }
  }
  const volumeAverage = movingAverage(vol, 5);
  const amounts = rows.map((row) => (isNum(row.a) ? Number(row.a) : null));
  const amountKnown = amounts.some((value) => value != null);
  series.push({
    name: '成交量',
    type: 'bar',
    xAxisIndex: 1,
    yAxisIndex: 1,
    data: vol.map((v, i) => ({ value: v, itemStyle: { color: colors[i], opacity: 0.55 } })),
  });
  series.push(lineSeries('量均线', volumeAverage, '#fb923c', {
    xAxisIndex: 1,
    yAxisIndex: 1,
    lineStyle: { width: 1.2, color: '#fb923c' },
  }));
  if (amountKnown) {
    series.push(lineSeries('成交额', amounts, '#a3e635', {
      xAxisIndex: 1,
      yAxisIndex: 2,
      lineStyle: { width: 1.2, color: '#a3e635' },
    }));
  }
  kChart.setOption({
    animation: false,
    axisPointer: { link: [{ xAxisIndex: [0, 1] }] },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: [
      { left: 56, right: 52, top: 16, height: '62%' },
      { left: 56, right: 52, top: '78%', height: '14%' },
    ],
    xAxis: [
      { type: 'category', data: dates, boundaryGap: true, axisLabel: { color: '#94a3b8' }, axisLine: { lineStyle: { color: '#334155' } } },
      { type: 'category', gridIndex: 1, data: dates, axisLabel: { show: false }, axisTick: { show: false } },
    ],
    yAxis: [
      { scale: true, axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: '#232e47' } } },
      { scale: true, gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } },
      {
        scale: true, gridIndex: 1, position: 'right', splitLine: { show: false },
        axisLabel: { color: '#a3e635', fontSize: 10, formatter: (value) => fmtCompact(value) },
      },
    ],
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1] }, { type: 'slider', xAxisIndex: [0, 1], bottom: 4, height: 18 }],
    series,
  }, true);
  kChart.resize();
  const guide = [
    trendText,
    overlays.ma ? '均线 MA5/10/20/60' : '',
    overlays.boll ? '布林 20日、2倍标准差' : '',
    patternText,
    '辅助线只描述这根图，不是买卖信号',
  ].filter(Boolean).join(' · ');
  if ($('qNote') && candleCaption) $('qNote').textContent = candleCaption;
  paintChartReading(closes, rows);
}

function bindOverlays() {
  const host = $('kOverlay');
  if (!host || host.dataset.bound) return;
  host.dataset.bound = '1';
  host.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-overlay]');
    if (!button) return;
    const key = button.dataset.overlay;
    overlays[key] = !overlays[key];
    button.classList.toggle('on', overlays[key]);
    if (quoteFreq !== 'minute' && lastCandlePayload) drawKline(lastCandlePayload);
  });
}

/* ----------------------------------------------------------------- signals */
async function loadSignals() {
  const host = $('stable');
  if (!host) return;
  try {
    const d = await api('/api/signals');
    $('sMeta').textContent = `信号日 ${d.signal_date || '—'} → 建议执行日 ${d.exec_date || '—'}`;
    const trades = d.trades || [];
    if (!trades.length) {
      host.innerHTML = '<tbody><tr><td class="l"><div class="empty">本期无调仓信号</div></td></tr></tbody>';
      return;
    }
    const head = `<thead><tr><th class="l">代码</th><th class="l">名称</th>
      <th class="sortable" data-key="1">目标权重</th><th class="sortable" data-key="2">上期权重</th>
      <th class="l">操作</th></tr></thead>`;
    const body = trades.map((t) => {
      const op = t[3] || '';
      const cls = op.includes('买入') ? 'buy' : (/卖出|清仓|减仓/.test(op) ? 'sell' : 'hold');
      return `<tr><td class="l mono">${esc(t[0])}</td><td class="l">${esc((d.names || {})[t[0]] || '')}</td>
        <td>${fmtPct(t[1], 1)}</td><td>${fmtPct(t[2], 1)}</td>
        <td class="l"><span class="tag ${cls}">${esc(op)}</span></td></tr>`;
    }).join('');
    host.innerHTML = head + `<tbody>${body}</tbody>`;
    makeSortable(host, () => Array.from(host.querySelectorAll('tbody tr')));
  } catch (e) {
    host.innerHTML = `<tbody><tr><td class="l"><div class="empty">信号加载失败：${esc(e.message)}</div></td></tr></tbody>`;
  }
}

/* ----------------------------------------------------------------- experts */
async function loadExperts() {
  const host = $('ecards');
  if (!host) return;
  try {
    const d = await api('/api/experts');
    if (d.status !== 'research_unvalidated') {
      $('eMeta').textContent = '模型有效性尚未验证';
      host.innerHTML = '';
      return;
    }
    $('eMeta').innerHTML = `信息收集截至信号日 <b>${esc(d.date || '—')}</b> · 候选 ${d.n_candidates || 0} 只 · 行情因子 ${(d.coverage || {}).factors || '—'} 个`;
    const cards = Object.entries(d.experts || {}).map(([n, v]) => {
      const conf = Number(v.confidence) || 0;
      return `<div class="card"><div class="card-body">
        <div class="row between"><h3 style="font-size:12.5px">${esc(n)}</h3>
          <span class="mono muted" style="font-size:11px">置信 ${fmtNum(conf)}</span></div>
        <div class="bar-row" style="margin-top:8px">
          <div class="track"><div class="fill" style="width:${Math.max(0, Math.min(1, conf)) * 100}%;background:var(--accent-cyan)"></div></div>
          <div class="val">${(v.top || []).length} 只</div>
        </div>
        <div style="margin-top:6px">${(v.top || []).map((c) => `<span class="tag hold">${esc(c.name || '')} <span class="muted mono">${esc(c.symbol || '')}</span></span>`).join('') || '<span class="muted">—</span>'}</div>
      </div></div>`;
    });
    host.innerHTML = cards.join('') || '<div class="card"><div class="empty">无专家输出</div></div>';
  } catch (e) {
    host.innerHTML = `<div class="card"><div class="empty">专家加载失败：${esc(e.message)}</div></div>`;
  }
}

/* ----------------------------------------------------------------- run */
async function loadRun() {
  const host = $('runKpis');
  if (!host) return;
  host.innerHTML = [
    kpiCard({ label: '公开行情', value: publicQuotesEnabled ? '已启用' : '已暂停', note: '来源与时间以行情页逐笔状态为准' }),
    kpiCard({ label: '日线数据', value: '每日更新', note: '收盘后 daily_pulse' }),
    kpiCard({ label: '研究信号', value: '按配置评估', note: '未获实盘批准' }),
    kpiCard({ label: '美股交易', value: '未启用', note: '背景须校验时序' }),
  ].join('');
}

/* ----------------------------------------------------------------- boot */
function bindActions() {
  const refreshH = $('refreshH');
  if (refreshH) refreshH.addEventListener('click', () => { loadHoldings(); loadOverview(); });
  const quoteToggle = $('enablePublicQuotes');
  if (quoteToggle) quoteToggle.addEventListener('click', () => {
    publicQuotesEnabled = !publicQuotesEnabled;
    loadIndices();
    loadHoldings();
  });
  const watchSort = $('watchSort');
  if (watchSort) watchSort.addEventListener('click', (ev) => {
    const btn = ev.target.closest('button');
    if (!btn) return;
    applyWatchSort(btn.dataset.key, 'desc', true);
  });
  const refreshE = $('refreshE');
  if (refreshE) refreshE.addEventListener('click', async () => {
    const b = refreshE, spinner = $('loading');
    b.disabled = true; if (spinner) spinner.style.display = 'inline-flex';
    try { await api('/api/experts?refresh=1'); await loadExperts(); }
    catch (e) { alert('重算失败: ' + e.message); }
    finally { b.disabled = false; if (spinner) spinner.style.display = 'none'; }
  });
  const reloadE = $('reloadE');
  if (reloadE) reloadE.addEventListener('click', () => loadExperts());
  window.addEventListener('resize', () => {
    clearTimeout(state._rt);
    state._rt = setTimeout(() => { if (state.seriesLoaded) drawCharts(); }, 140);
  });
}

function bindWatchSplit() {
  const split = $('watchSplit');
  const term = document.querySelector('.terminal');
  if (!split || !term || split.dataset.bound) return;
  split.dataset.bound = '1';
  const storageKey = 'quantlab.watchWidth';
  const minWidth = 240;
  const maxWidth = () => Math.min(640, term.getBoundingClientRect().width * 0.55);
  const saved = Number(localStorage.getItem(storageKey));
  const setWidth = (width, persist) => {
    const next = Math.round(Math.max(minWidth, Math.min(maxWidth(), width)));
    term.style.setProperty('--watch-w', `${next}px`);
    split.setAttribute('aria-valuenow', String(next));
    if (persist) localStorage.setItem(storageKey, String(next));
    if (kChart) kChart.resize();
  };
  if (saved >= minWidth) setWidth(saved, false);
  else split.setAttribute('aria-valuenow', '340');
  let dragging = false;
  let resizeFrame = 0;
  let latest = null;
  const paintWidth = (width) => {
    latest = width;
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      if (latest != null) setWidth(latest, false);
    });
  };
  split.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    dragging = true;
    split.classList.add('is-drag');
    document.documentElement.classList.add('is-col-drag');
    try { split.setPointerCapture(event.pointerId); } catch (_) { /* 部分环境没有指针捕获，仍靠本元素的移动事件 */ }
    event.preventDefault();
  });
  split.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    paintWidth(event.clientX - term.getBoundingClientRect().left);
  });
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    split.classList.remove('is-drag');
    document.documentElement.classList.remove('is-col-drag');
    cancelAnimationFrame(resizeFrame);
    if (latest != null) setWidth(latest, true);
    latest = null;
  };
  split.addEventListener('pointerup', stop);
  split.addEventListener('pointercancel', stop);
  split.addEventListener('dblclick', () => {
    term.style.removeProperty('--watch-w');
    localStorage.removeItem(storageKey);
    split.setAttribute('aria-valuenow', '340');
    if (kChart) kChart.resize();
  });
  split.addEventListener('keydown', (event) => {
    const current = parseInt(getComputedStyle(term).getPropertyValue('--watch-w'), 10) || 340;
    if (event.key === 'ArrowLeft') setWidth(current - 24, true);
    else if (event.key === 'ArrowRight') setWidth(current + 24, true);
    else if (event.key === 'Home') {
      term.style.removeProperty('--watch-w');
      localStorage.removeItem(storageKey);
      split.setAttribute('aria-valuenow', '340');
    } else return;
    event.preventDefault();
  });
}

function bindSymbolSearch() {
  const input = $('symbolSearch');
  const box = $('symbolSearchResults');
  if (!input || !box || input.dataset.bound) return;
  input.dataset.bound = '1';
  let timer = 0;
  let rows = [];
  let active = 0;
  const paint = () => {
    if (!rows.length) {
      box.hidden = true;
      box.innerHTML = '';
      return;
    }
    box.hidden = false;
    box.innerHTML = rows.map((row, index) => (
      `<button type="button" class="${index === active ? 'on' : ''}" data-code="${esc(row.code)}" data-name="${esc(row.name)}"><b>${esc(row.name)}</b><i>${esc(row.code)} ${esc(row.market || '')}</i></button>`
    )).join('');
  };
  input.addEventListener('input', () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(async () => {
      const query = input.value.trim();
      if (!query) {
        rows = [];
        paint();
        return;
      }
      try {
        const payload = await api(`/api/portfolio/search?q=${encodeURIComponent(query)}`);
        rows = Array.isArray(payload.rows) ? payload.rows : [];
        active = 0;
        if (!rows.length) {
          box.hidden = false;
          box.innerHTML = `<div class="leader-note" style="padding:8px 10px">${esc(payload.message || '没有匹配的 A 股')}</div>`;
          return;
        }
        paint();
      } catch (error) {
        box.hidden = false;
        box.innerHTML = `<div class="leader-note" style="padding:8px 10px">${esc(error.message || '搜索失败')}</div>`;
      }
    }, 180);
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowDown' && rows.length) {
      event.preventDefault();
      active = Math.min(rows.length - 1, active + 1);
      paint();
    } else if (event.key === 'ArrowUp' && rows.length) {
      event.preventDefault();
      active = Math.max(0, active - 1);
      paint();
    } else if (event.key === 'Enter' && rows[active]) {
      event.preventDefault();
      openSymbol(rows[active].code, rows[active].name);
    } else if (event.key === 'Escape') {
      box.hidden = true;
    }
  });
  box.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-code]');
    if (!button) return;
    openSymbol(button.dataset.code, button.dataset.name);
  });
  document.addEventListener('click', (event) => {
    if (!event.target.closest('.symbol-search')) box.hidden = true;
  });
  document.addEventListener('keydown', (event) => {
    if (event.key !== '/' || event.metaKey || event.ctrlKey || event.altKey) return;
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    event.preventDefault();
    input.focus();
    input.select();
  });
}

window.QuantLabQuote = { open: openSymbol, rememberBoard, kind: '' };

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  bindTabs();
  bindRange();
  bindActions();
  bindSymbolSearch();
  bindOverlays();
  bindWatchSplit();
  updatePublicQuoteControls(publicQuotesEnabled);
  loadOverview();
  loadRisk();
  loadIndices();
  loadHoldings();
  setInterval(loadOverview, 30000);
  setInterval(loadRisk, 30000);
  setInterval(() => {
    if (document.hidden) return;
    loadIndices();
  }, 15000);
  setInterval(() => {
    if (document.hidden) return;
    const pane = $('holdings-pane');
    if (!pane || !pane.classList.contains('active')) return;
    loadHoldings();
    if (quoteCode) selectQuote(quoteCode, quoteName, true);
  }, 5000);
});
