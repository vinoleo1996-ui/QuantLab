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
    btn.setAttribute('aria-label', name === 'dark' ? '切换到亮色主题' : '切换到深色主题');
    btn.setAttribute('aria-pressed', name === 'dark' ? 'true' : 'false');
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
      b.tabIndex = on ? 0 : -1;
    });
    document.querySelectorAll('.pane').forEach((p) => {
      p.classList.toggle('active', p.id === name);
    });
    if (name === 'chart-pane' && state.seriesLoaded) drawCharts();
    const loaders = { 'holdings-pane': loadHoldings, 'signals-pane': loadSignals, 'experts-pane': loadExperts, 'run-pane': loadRun };
    if (loaders[name]) loaders[name]();
  };
  buttons.forEach((b, index) => {
    b.addEventListener('click', () => show(b.dataset.pane));
    b.addEventListener('keydown', (event) => {
      if (!['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      let next = index;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % buttons.length;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + buttons.length) % buttons.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = buttons.length - 1;
      buttons[next].focus();
      show(buttons[next].dataset.pane);
    });
  });
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
    tooltip: { ...base.tooltip, valueFormatter: (v) => isNum(v) ? Number(v).toFixed(4) : '—' },
    yAxis: {
      type: 'value', scale: true,
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: cMuted, fontSize: 10 },
      splitLine: { lineStyle: { color: cBorder, opacity: .45 } },
    },
    series: navSeries,
  });

  if (!state.charts.dd) state.charts.dd = echarts.init(ddEl);
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
      lineStyle: { width: 1, color: cDown },
      areaStyle: { color: 'rgba(16,185,129,.20)' },
    }],
  });
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
/* Generic sortable table: the first data cell decides numeric vs text compare,
   because P&L columns sort as -0.4 < 0.2 and codes must not. */
function makeSortable(table, getRows) {
  if (!table) return;
  const heads = table.querySelectorAll('thead th.sortable');
  heads.forEach((th, idx) => {
    th.setAttribute('aria-sort', 'none');
    th.setAttribute('role', 'columnheader');
    th.setAttribute('tabindex', '0');
    th.addEventListener('click', () => {
      const key = th.dataset.key || idx;
      const cur = state.sortState[table.id] || {};
      const dir = cur.key === key && cur.dir === 'desc' ? 'asc' : 'desc';
      state.sortState[table.id] = { key, dir };
      heads.forEach((h) => { h.classList.remove('sorted'); const a = h.querySelector('.arrow'); if (a) a.remove(); });
      th.classList.add('sorted');
      heads.forEach((h) => h.setAttribute('aria-sort', h === th ? (dir === 'desc' ? 'descending' : 'ascending') : 'none'));
      const arrow = document.createElement('span');
      arrow.className = 'arrow';
      arrow.textContent = dir === 'desc' ? '▼' : '▲';
      th.appendChild(arrow);
      const rows = getRows();
      rows.sort((r1, r2) => {
        const v1 = r1.dataset[key], v2 = r2.dataset[key];
        const n1 = parseFloat(v1), n2 = parseFloat(v2);
        if (v1 !== '' && v2 !== '' && !Number.isNaN(n1) && !Number.isNaN(n2)) {
          return dir === 'desc' ? n2 - n1 : n1 - n2;
        }
        const c = String(v1).localeCompare(String(v2), 'zh-CN');
        return dir === 'desc' ? -c : c;
      });
      const tbody = table.querySelector('tbody');
      rows.forEach((r) => tbody.appendChild(r));
    });
    th.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); th.click(); }
    });
  });
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

function setDashboardStatus(kind, title, copy, meta = '只读') {
  const banner = $('dashboardStatus');
  if (!banner) return;
  banner.classList.remove('is-good', 'is-warn', 'is-error');
  if (kind) banner.classList.add(`is-${kind}`);
  const titleEl = $('statusTitle');
  const copyEl = $('statusCopy');
  const metaEl = $('statusMeta');
  if (titleEl) titleEl.textContent = title;
  if (copyEl) copyEl.textContent = copy;
  if (metaEl) metaEl.textContent = meta;
}

async function loadOverview() {
  const host = $('kpis');
  try {
    const [d, s] = await Promise.all([api('/api/nav'), api('/api/series')]);
    state.cachedSeries = s;
    state.seriesLoaded = true;

    $('chipExec').innerHTML = `<span class="dot ${d.interpretable ? 'ok' : 'warn'}"></span>执行日 <b>${esc(d.last_exec || '—')}</b>`;
    $('chipPrice').innerHTML = `<span class="dot"></span>行情 <b>${esc(d.quote_time || '—')}</b>`;
    const stateChip = d.status === 'research_unvalidated' && d.interpretable
      ? '<span class="chip ok"><span class="dot ok"></span>研究账户 <b>可解释</b></span>'
      : `<span class="chip warn"><span class="dot warn"></span>研究账户 <b>${esc(d.status || '未验证')}</b></span>`;
    $('chipStatus').innerHTML = stateChip;
    $('overviewAsOf').textContent = d.last_exec ? `执行日 ${d.last_exec}` : '等待可解释结果';

    if (!d.interpretable) {
      const rs = (d.blocked_reasons || []);
      const items = rs.length
        ? `<ul class="empty-list">${rs.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`
        : '<div class="muted">未提供具体阻断原因</div>';
      host.innerHTML = `<div class="card"><div class="empty">
        <span class="big">当前结果不可解释，绩效和信号保持空态</span>
        <span>数据或验证未通过；这是研究门禁的正常结果，不是页面故障。</span>${items}
      </div></div>`;
      setDashboardStatus('warn', '当前结果不可解释', '绩效、回撤和买卖信号不会在缺失数据时被填成假数字。', d.status || 'blocked_data');
      // Hide the whole chart + risk region: an empty canvas and an empty gauge
      // read as "broken" rather than "not applicable".
      const grid = document.getElementById('overviewGrid');
      if (grid) grid.style.display = 'none';
      $('overviewNote').innerHTML = '';
      return;
    }
    setDashboardStatus('good', '研究账户可解释', '已通过当前回放自身的解释性门禁，可以查看历史组合曲线和风险预算。', d.last_exec || '研究回放');
    const grid = document.getElementById('overviewGrid');
    if (grid) grid.style.display = '';
    const body = document.querySelector('#chart-pane .card-body');
    if (body) body.style.display = '';
    const note = $('chartNote');
    if (note) note.innerHTML = '回撤以账户历史净值为基准；面积为负值即为水下。右侧仪表标出当前回撤所在的政策档位。';
    $('rangeSeg').style.display = '';

    const metrics = d.metrics || {};
    const sp = s.nav || [];
    // Drawdown is a risk magnitude, not a price move: colouring it by direction
    // would paint a 40% loss in "down" green and read as calm. Severity wins.
    const ddMag = Math.abs(Number(metrics.max_drawdown) || 0);
    const ddCls = ddMag > 0.25 ? 'up' : (ddMag > 0.18 ? 'risk' : 'flat');
    const ddNote = ddMag > 0.25 ? '超出25%研究预算' : (ddMag > 0.18 ? '超18%减仓档' : '预算内');
    const cards = [
      kpiCard({ label: '总收益', value: fmtSignedPct(metrics.total_return), cls: dirClass(metrics.total_return), note: '全期扣费', spark: sparkline(sp) }),
      kpiCard({ label: '年化收益', value: fmtSignedPct(metrics.annual_return), cls: dirClass(metrics.annual_return), note: '研究回放，非预期' }),
      kpiCard({ label: '年化波动', value: fmtPct(metrics.annual_vol), note: `目标 ≤${fmtPct(d.target_vol)}` }),
      kpiCard({ label: '夏普', value: fmtNum(metrics.sharpe), note: '日频年化' }),
      kpiCard({ label: '最大回撤', value: fmtPct(metrics.max_drawdown), cls: ddCls, note: ddNote }),
      kpiCard({ label: '卡玛', value: fmtNum(metrics.calmar), note: '年化/回撤' }),
      kpiCard({ label: '胜率(日)', value: fmtPct(metrics.win_rate), note: '正收益日占比' }),
      kpiCard({ label: '今年以来', value: fmtSignedPct(d.ytd), cls: dirClass(d.ytd), trend: d.ytd }),
    ];
    host.innerHTML = cards.join('');

    $('overviewNote').innerHTML = d.extra || '';
    drawCharts();
  } catch (e) {
    setDashboardStatus('error', '看板数据暂时不可用', '请检查服务状态后重试；页面不会用缓存或猜测填充账户结果。', '读取失败');
    if ($('overviewAsOf')) $('overviewAsOf').textContent = '读取失败';
    host.innerHTML = `<div class="card"><div class="empty"><span class="big">看板数据加载失败</span><span>${esc(e.message)}</span></div></div>`;
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
      if (r.note && grid && grid.style.display !== 'none') {
        host.innerHTML = `<div class="empty">${esc(r.note)}</div>`;
      }
      return;
    }
    const stateLabel = { defensive: '防守', normal: '常态', active: '积极', legacy: '旧政策' }[r.policy_state] || '—';
    const stateCls = r.policy_state === 'active' ? 'active' : (r.policy_state === 'defensive' ? 'defensive' : 'normal');
    const reasons = (r.reasons || []).map((x) => `<li>${esc(x)}</li>`).join('') || '<li class="muted">无触发原因</li>';
    host.innerHTML = `
      <div class="row between" style="margin-bottom:10px">
        <span class="state-badge ${stateCls}"><span class="dot"></span>${esc(stateLabel)}</span>
        <span class="mono sec" style="font-size:11px">${esc(r.policy_id || '')} · ${esc(r.policy_version || '')}</span>
      </div>
      <div class="row between" style="margin-bottom:2px">
        <span class="muted" style="font-size:11px">当前回撤</span>
        <span class="mono ${'down'}" style="font-size:15px;font-weight:600">${fmtPct(r.drawdown)}</span>
      </div>
      ${renderRiskGauge(r.drawdown, r.level)}
      <div class="row" style="margin:10px 0 12px;gap:6px">
        <span class="chip ${r.gross_scale < 1 ? 'warn' : 'ok'}">仓位 <b>${fmtPct(r.gross_scale, 0)}</b></span>
        <span class="chip">波动预算 <b>${fmtPct(r.target_vol, 0)}</b></span>
        ${r.transition_count ? `<span class="chip">档位确认 <b>${r.transition_count}/${r.transition_direction > 0 ? 5 : 2}</b></span>` : ''}
      </div>
      <dl class="kv">
        <dt>单股上限</dt><dd>${fmtPct(r.max_stock_weight, 0)}</dd>
        <dt>行业上限</dt><dd>${fmtPct(r.max_sector_weight, 0)}</dd>
        <dt>同跌群组上限</dt><dd>${fmtPct(r.max_cluster_weight, 0)}</dd>
        <dt>最低现金</dt><dd>${fmtPct(r.cash_buffer, 0)}</dd>
        <dt>未知行业上限</dt><dd>${fmtPct(r.unknown_sector_weight, 0)}</dd>
      </dl>
      <div class="callout ${r.level >= 2 ? 'warn' : ''}" style="margin-top:12px">
        <b>触发原因</b><ul>${reasons}</ul>
      </div>`;
  } catch (e) {
    host.innerHTML = `<div class="empty">风险预算加载失败：${esc(e.message)}</div>`;
  }
}

/* ----------------------------------------------------------------- holdings */
function weightBar(w) {
  const v = Number(w) || 0;
  return `<div class="wbar"><div class="wbar-track"><div class="wbar-fill" style="width:${Math.min(100, v * 1000)}%"></div></div><span>${fmtPct(v, 1)}</span></div>`;
}

async function loadHoldings() {
  const host = $('htable');
  if (!host) return;
  try {
    const d = await api('/api/holdings');
    $('chipPrice').innerHTML = `<span class="dot pulse ok"></span>行情 <b>${esc(d.quote_time || '—')}</b>`;
    const s = d.summary || {};
    const rows = d.rows || [];
    const rg = { '-1': '深度熊市', '0': '震荡', '1': '牛市' }[String(s.regime_trend)] || '待核';
    const displayable = d.status === 'research_unvalidated' || d.status === 'unverified_data_gaps';
    if (!rows.length || !displayable) {
      const copy = d.status === 'blocked_data'
        ? '账本或验证门禁未通过，页面不会从历史回放权重、目录扫描或 latest 指针猜测仓位。'
        : '接口返回空快照，请先确认模拟账户状态。';
      $('holdNote').innerHTML = `<div class="callout warn"><b>当前没有可展示的持仓快照</b>：${esc(copy)} <a href="/portfolio">查看账户行情状态</a></div>`;
    } else {
      $('holdNote').innerHTML = `<div class="callout"><b>仓位口径</b>：市场状态「${esc(rg)}」，组合按 ${fmtPct(s.gross_scale, 0)} 仓位运行。${s.gross_scale < 1 ? '当前处于风险降仓状态；未成交的退出仍受市场限制。' : ''} 这是按股数计价的主模拟账户，不代表实际成交或券商持仓。</div>`;
    }

    const totW = rows.reduce((a, r) => a + (Number(r.weight) || 0), 0);
    const tv = rows.reduce((a, r) => a + (Number(r.market_value) || 0), 0);
    $('hkpis').innerHTML = [
      kpiCard({ label: '持仓数', value: String(rows.length), note: rows.length ? '只' : '当前为空' }),
      kpiCard({ label: '总权重', value: fmtPct(totW, 1), note: '中性档仓位' }),
      kpiCard({ label: '模拟市值', value: fmtMoneyCny(tv), note: '回放截至日' }),
      kpiCard({ label: '今日组合', value: fmtSignedPct(d.today), cls: dirClass(d.today), note: '加权估算' }),
      kpiCard({ label: '今年组合', value: fmtSignedPct(d.ytd), cls: dirClass(d.ytd), note: '今年以来' }),
    ].join('');

    if (!rows.length) {
      $('htable').innerHTML = '<tbody><tr><td class="l"><div class="empty"><span class="big">当前无持仓行</span><span>等待已落盘且通过解释性门禁的模拟账户快照。</span></div></td></tr></tbody>';
      return;
    }
    const head = `<thead><tr>
      <th class="l">代码</th><th class="l">名称</th><th class="sortable" data-key="1">权重</th>
      <th class="sortable" data-key="2">建仓日</th><th class="sortable" data-key="3">成本</th>
      <th class="sortable" data-key="4">现价</th><th class="sortable" data-key="5">今日</th>
      <th class="sortable" data-key="6">今年</th><th class="sortable" data-key="7">建仓至今</th>
      <th class="sortable" data-key="8">市值</th><th class="l">整手</th></tr></thead>`;
    const body = rows.map((r) => `<tr>
      <td class="l mono">${esc(r.code)}</td><td class="l">${esc(r.name || '')}</td>
      <td>${weightBar(r.weight)}</td>
      <td>${esc(r.entry_date || '—')}</td>
      <td>${fmtNum(r.cost, 3)}</td>
      <td>${fmtNum(r.live_quote || r.price, 3)}<span class="data-sub">${r.live_quote ? '公开价' : '账本价'}</span></td>
      <td data-key="5" class="${dirClass(r.pct_chg)}">${fmtSignedPct(r.pct_chg)}</td>
      <td data-key="6" class="${dirClass(r.return_ytd)}">${fmtSignedPct(r.return_ytd)}</td>
      <td data-key="7" class="${dirClass(r.return_since_entry)}">${fmtSignedPct(r.return_since_entry)}</td>
      <td data-key="8">${fmtMoneyCny(r.market_value)}</td>
      <td class="l">${isNum(r.lot_cost) ? fmtMoney(r.lot_cost) + (r.lot_affordable ? '' : ' <span class="muted">可能超配</span>') : '—'}</td>
    </tr>`).join('');
    $('htable').innerHTML = head + `<tbody>${body}</tbody>`;
    makeSortable($('htable'), () => Array.from($('htable').querySelectorAll('tbody tr')));
  } catch (e) {
    host.innerHTML = `<div class="empty">持仓加载失败：${esc(e.message)}</div>`;
  }
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
      $('eMeta').className = 'callout warn';
      $('eMeta').innerHTML = '<b>专家结果保持空态</b>：当前回放未通过研究解释性门禁，页面不会把历史模型输出包装成新的建议。';
      host.innerHTML = '<div class="card"><div class="empty"><span class="big">暂无可展示的专家结果</span><span>请先查看研究裁决台中的证据状态。</span><a class="btn btn-quiet btn-sm" href="/research">打开研究裁决</a></div></div>';
      return;
    }
    $('eMeta').className = 'callout good';
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
    $('eMeta').className = 'callout bad';
    $('eMeta').innerHTML = `<b>专家信息读取失败</b>：${esc(e.message)}`;
    host.innerHTML = '<div class="card"><div class="empty"><span class="big">请稍后重试</span><span>页面没有改变任何策略或账户状态。</span></div></div>';
  }
}

/* ----------------------------------------------------------------- run */
async function loadRun() {
  const host = $('runKpis');
  if (!host) return;
  host.innerHTML = [
    kpiCard({ label: '盘中行情', value: '实时', note: '新浪/腾讯' }),
    kpiCard({ label: '日线数据', value: '每日更新', note: '收盘后 daily_pulse' }),
    kpiCard({ label: '研究信号', value: '按配置评估', note: '未获实盘批准' }),
    kpiCard({ label: '美股交易', value: '未启用', note: '背景须校验时序' }),
  ].join('');
}

/* ----------------------------------------------------------------- boot */
function bindActions() {
  const refreshH = $('refreshH');
  if (refreshH) refreshH.addEventListener('click', () => { loadHoldings(); loadOverview(); });
  const refreshE = $('refreshE');
  if (refreshE) refreshE.addEventListener('click', async () => {
    const b = refreshE, spinner = $('loading');
    b.disabled = true; if (spinner) { spinner.hidden = false; spinner.style.display = 'inline-flex'; }
    try { await api('/api/experts?refresh=1'); await loadExperts(); }
    catch (e) {
      $('eMeta').className = 'callout bad';
      $('eMeta').textContent = `读取失败：${e.message}`;
    }
    finally { b.disabled = false; if (spinner) { spinner.hidden = true; spinner.style.display = 'none'; } }
  });
  const reloadE = $('reloadE');
  if (reloadE) reloadE.addEventListener('click', () => loadExperts());
  window.addEventListener('resize', () => {
    clearTimeout(state._rt);
    state._rt = setTimeout(() => { if (state.seriesLoaded) drawCharts(); }, 140);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  bindTabs();
  bindRange();
  bindActions();
  loadOverview();
  loadRisk();
  setInterval(loadOverview, 30000);
  setInterval(loadRisk, 30000);
});
