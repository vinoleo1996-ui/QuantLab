/*
 * Grok paper-ledger terminal compatibility adapter.
 *
 * Load this file immediately BEFORE portfolio_terminal.js.  The recovered
 * terminal remains the visual and interaction owner; this file only maps its
 * former unscoped API contract to the isolated portfolio-observation APIs and
 * adds the observation affordances that did not exist in the recovered
 * version: institutional revenue forecast status and industry 1-month Top 3
 * status. 年K is now declared in the terminal shell; the adapter retains a
 * fallback insertion for an older recovered shell.
 *
 * This must never point at /output/runs/latest.json or turn an unavailable
 * forecast / ranking into a number.  It is deliberately a read-only adapter.
 */

'use strict';

(() => {
  const INSTALL_KEY = '__quantlabPortfolioTerminalAdapterV1__';
  if (window[INSTALL_KEY]) return;

  const ENDPOINTS = Object.freeze({
    nav: '/api/portfolio/terminal/nav',
    series: '/api/portfolio/terminal/series',
    risk: '/api/portfolio/terminal/risk',
    holdings: '/api/portfolio/holdings',
    kline: '/api/portfolio/kline',
    intraday: '/api/portfolio/intraday',
    indices: '/api/portfolio/terminal/indices',
    signals: '/api/portfolio/terminal/signals',
    experts: '/api/portfolio/terminal/experts',
    // These are deliberately selected-security endpoints.  Fetching a THS
    // forecast for every row on each five-second quote refresh would be both
    // slow and misleadingly aggressive toward a public source.
    revenueForecast: '/api/portfolio/revenue-forecast',
    industryTop3: '/api/portfolio/industry-top3',
  });

  const LEGACY_TO_TERMINAL = Object.freeze({
    '/api/nav': ENDPOINTS.nav,
    '/api/series': ENDPOINTS.series,
    '/api/risk': ENDPOINTS.risk,
    '/api/holdings': ENDPOINTS.holdings,
    '/api/kline': ENDPOINTS.kline,
    '/api/intraday': ENDPOINTS.intraday,
    '/api/indices': ENDPOINTS.indices,
    '/api/signals': ENDPOINTS.signals,
    '/api/experts': ENDPOINTS.experts,
  });

  const PAYLOAD_EVENT = 'quantlab:portfolio-terminal-payload';
  const adapterState = {
    holdingsByCode: new Map(),
    lastKlineByCode: new Map(),
    supplementalByCode: new Map(),
    selectedCode: '',
    publicMarketEnabled: false,
  };

  window[INSTALL_KEY] = Object.freeze({ ENDPOINTS, PAYLOAD_EVENT });
  // A small, explicit public surface is useful for diagnostics without
  // exporting the mutable adapter state.
  window.QuantLabPortfolioTerminal = Object.freeze({ endpoints: ENDPOINTS });

  const isNumeric = (value) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));

  function text(value) {
    return value === null || value === undefined ? '' : String(value);
  }

  function percent(value) {
    if (!isNumeric(value)) return '—';
    const number = Number(value);
    return `${number >= 0 ? '+' : ''}${(number * 100).toFixed(2)}%`;
  }

  function financialNumber(value) {
    if (!isNumeric(value)) return '—';
    return Number(value).toLocaleString('zh-CN', {
      maximumFractionDigits: 2,
      minimumFractionDigits: 0,
    });
  }

  function revenueAmount(value, unit) {
    if (!isNumeric(value)) return '—';
    const amount = Number(value);
    // The market-observatory parser returns the exact number of yuan for the
    // explicitly labelled 营业收入(元) row.  Compress only that known unit for
    // terminal readability; do not infer a unit for an arbitrary provider.
    if (unit === '元' && Math.abs(amount) >= 100000000) return `${(amount / 100000000).toFixed(2)} 亿元`;
    return `${financialNumber(amount)}${unit ? ` ${unit}` : ''}`;
  }

  function firstPresent(object, keys) {
    if (!object || typeof object !== 'object') return undefined;
    for (const key of keys) {
      if (object[key] !== null && object[key] !== undefined && object[key] !== '') return object[key];
    }
    return undefined;
  }

  function normaliseStatus(value) {
    return text(value).trim().toLowerCase();
  }

  function endpointFor(input) {
    if (typeof input !== 'string' && !(input instanceof URL)) return null;
    let url;
    try {
      url = new URL(String(input), window.location.href);
    } catch (_error) {
      return null;
    }
    if (url.origin !== window.location.origin) return null;
    const endpoint = LEGACY_TO_TERMINAL[url.pathname];
    return endpoint ? { endpoint, url } : null;
  }

  /*
   * The recovered script calls fetch only with relative strings.  Keeping the
   * wrapper narrow protects unrelated requests (assets, third-party scripts,
   * and future Request-object calls) from an accidental rewrite.
   */
  const nativeFetch = window.fetch.bind(window);
  window.fetch = function portfolioTerminalFetch(input, init) {
    const route = endpointFor(input);
    if (!route) return nativeFetch(input, init);

    const rewritten = new URL(route.url.href);
    rewritten.pathname = route.endpoint;
    return nativeFetch(rewritten.href, init).then((response) => {
      if (response.ok && typeof response.clone === 'function') {
        response.clone().json().then((payload) => {
          window.dispatchEvent(new CustomEvent(PAYLOAD_EVENT, {
            detail: { endpoint: route.endpoint, url: rewritten.href, payload },
          }));
        }).catch(() => {
          // Some future response may be non-JSON. The recovered script is the
          // source of truth for reporting its own request failure.
        });
      }
      return response;
    });
  };

  function rowCode(row) {
    return text(firstPresent(row, ['code', 'symbol', 'ticker'])).trim();
  }

  function holdingsRows(payload) {
    if (Array.isArray(payload?.rows)) return payload.rows;
    if (Array.isArray(payload?.data?.rows)) return payload.data.rows;
    return [];
  }

  function selectedFromDom() {
    const active = document.querySelector('#htable .quote-row.on');
    return text(active?.dataset?.code).trim();
  }

  function selectedRecord() {
    const code = adapterState.selectedCode || selectedFromDom();
    if (!code) return null;
    const holding = adapterState.holdingsByCode.get(code);
    const supplemental = adapterState.supplementalByCode.get(code);
    if (!holding && !supplemental) return null;
    return { ...(holding || { code }), ...(supplemental || {}) };
  }

  function addStat(panel, label, value, detail, tone) {
    const item = document.createElement('div');
    item.className = 'stat';
    // The existing terminal .stat component supplies the hard-edge visual
    // language; the flex hint merely avoids truncating Top 3 names.
    item.style.flex = '1 1 230px';
    item.style.minWidth = '220px';

    const title = document.createElement('span');
    title.textContent = label;
    const body = document.createElement('b');
    body.textContent = value;
    if (tone) body.className = tone;
    if (detail) {
      body.title = detail;
      item.title = detail;
    }
    item.append(title, body);
    panel.append(item);
  }

  function ensureObservationPanel() {
    let panel = document.getElementById('terminal-observation-status');
    if (panel) return panel;
    const sector = document.getElementById('qSector');
    if (!sector || !sector.parentNode) return null;

    panel = document.createElement('div');
    panel.id = 'terminal-observation-status';
    panel.className = 'stat-row';
    panel.setAttribute('role', 'status');
    panel.setAttribute('aria-live', 'polite');
    panel.setAttribute('aria-label', '机构营收预测和板块近一月 Top 3');
    sector.insertAdjacentElement('afterend', panel);
    return panel;
  }

  function revenueForecastFrom(record) {
    const direct = firstPresent(record, ['revenue_forecast', 'revenue_estimate', 'forecast']);
    if (direct && typeof direct === 'object') return direct;
    if (direct !== undefined) return { status: 'available', display: direct };

    const status = firstPresent(record, ['revenue_forecast_status', 'revenue_estimate_status']);
    const display = firstPresent(record, ['revenue_forecast_display', 'revenue_estimate_display']);
    if (status !== undefined || display !== undefined) {
      return {
        status,
        display,
        source: firstPresent(record, ['revenue_forecast_source', 'revenue_estimate_source']),
        as_of: firstPresent(record, ['revenue_forecast_as_of', 'revenue_estimate_as_of']),
        message: firstPresent(record, ['revenue_forecast_message', 'revenue_estimate_message']),
      };
    }
    return null;
  }

  function forecastView(record) {
    const forecast = revenueForecastFrom(record);
    if (!forecast) {
      return {
        value: '机构预测数据未接入',
        detail: '该持仓响应没有提供机构营收预测字段；不会用已披露营收替代预测。',
        tone: 'flat',
      };
    }

    const status = normaliseStatus(firstPresent(forecast, ['status', 'availability', 'state']));
    if (status === 'loading') {
      return {
        value: '机构预测读取中…',
        detail: '仅在选择这只股票后读取；不会在每次持仓刷新时请求 50 只预测。',
        tone: 'flat',
      };
    }
    if (['error', 'request_error', 'failed'].includes(status)) {
      return {
        value: '机构预测暂不可用',
        detail: text(firstPresent(forecast, ['message', 'reason'])) || '读取预测接口失败；未以已披露营收替代。',
        tone: 'flat',
      };
    }
    const unavailable = ['unavailable', 'missing', 'none', 'not_available', 'not-covered', 'not_covered', 'no_estimate'].includes(status);
    if (unavailable) {
      return {
        value: '暂无机构预测',
        detail: text(firstPresent(forecast, ['message', 'reason'])) || '该标的暂无可展示的机构营收预测；未以已披露营收替代。',
        tone: 'flat',
      };
    }

    const display = firstPresent(forecast, ['display', 'formatted', 'label']);
    const year = firstPresent(forecast, ['fiscal_year', 'year', 'target_year']);
    const value = firstPresent(forecast, ['mean', 'value', 'revenue', 'estimate', 'consensus']);
    const unit = firstPresent(forecast, ['unit', 'currency_unit']);
    const source = firstPresent(forecast, ['source', 'provider']);
    const asOf = firstPresent(forecast, ['as_of', 'updated_at', 'date']);
    const parsedForecasts = firstPresent(forecast, ['forecasts', 'estimates']);

    if (Array.isArray(parsedForecasts) && parsedForecasts.length) {
      const ordered = parsedForecasts
        .filter((item) => item && typeof item === 'object' && isNumeric(firstPresent(item, ['value', 'mean', 'estimate', 'revenue'])))
        .sort((left, right) => Number(firstPresent(left, ['year', 'fiscal_year']) || 0) - Number(firstPresent(right, ['year', 'fiscal_year']) || 0));
      const currentYear = new Date().getFullYear();
      const selected = ordered.find((item) => Number(firstPresent(item, ['year', 'fiscal_year']) || 0) >= currentYear) || ordered[0];
      if (selected) {
        const selectedYear = firstPresent(selected, ['year', 'fiscal_year']);
        const selectedValue = firstPresent(selected, ['value', 'mean', 'estimate', 'revenue']);
        const metric = text(firstPresent(forecast, ['metric']));
        const parsedUnit = metric === '营业收入(元)' ? '元' : firstPresent(forecast, ['unit', 'currency_unit']);
        return {
          value: `${selectedYear ? `${selectedYear}E ` : '预测 '}${revenueAmount(selectedValue, parsedUnit)}`,
          detail: [source && `来源 ${source}`, asOf && `截至 ${asOf}`, '口径：机构一致预期均值'].filter(Boolean).join(' · ') || '机构营收预测',
          tone: 'flat',
        };
      }
    }

    if (display !== undefined) {
      return {
        value: text(display),
        detail: [source && `来源 ${source}`, asOf && `截至 ${asOf}`].filter(Boolean).join(' · ') || '机构营收预测',
        tone: 'flat',
      };
    }
    if (value !== undefined) {
      const prefix = year ? `${year}E ` : '预测 ';
      return {
        value: `${prefix}${revenueAmount(value, unit)}`,
        detail: [source && `来源 ${source}`, asOf && `截至 ${asOf}`].filter(Boolean).join(' · ') || '机构营收预测',
        tone: 'flat',
      };
    }
    return {
      value: '暂无机构预测',
      detail: text(firstPresent(forecast, ['message', 'reason'])) || '预测字段未提供数值；未以已披露营收替代。',
      tone: 'flat',
    };
  }

  function sectorTop3From(record) {
    const direct = firstPresent(record, ['industry_top3_1m', 'sector_top3_1m', 'board_top3_1m']);
    if (direct && typeof direct === 'object') return direct;
    const status = firstPresent(record, ['industry_top3_1m_status', 'sector_top3_1m_status']);
    const rows = firstPresent(record, ['industry_top3_1m_rows', 'sector_top3_1m_rows']);
    if (status !== undefined || rows !== undefined) {
      return {
        status,
        rows,
        source: firstPresent(record, ['industry_top3_1m_source', 'sector_top3_1m_source']),
        as_of: firstPresent(record, ['industry_top3_1m_as_of', 'sector_top3_1m_as_of']),
        message: firstPresent(record, ['industry_top3_1m_message', 'sector_top3_1m_message']),
      };
    }
    return null;
  }

  function sectorTop3View(record) {
    const top3 = sectorTop3From(record);
    if (!top3) {
      return {
        value: '板块近一月 Top 3 数据未接入',
        detail: '需要全行业成分和近 20 个交易日行情；不会以当前 50 只持仓内部排名代替。',
        tone: 'flat',
      };
    }

    const status = normaliseStatus(firstPresent(top3, ['status', 'availability', 'state']));
    if (status === 'loading') {
      return {
        value: '板块 Top 3 读取中…',
        detail: '仅在选择这只股票后读取；要求完整行业成分与近 20 个交易日行情。',
        tone: 'flat',
      };
    }
    if (['error', 'request_error', 'failed'].includes(status)) {
      return {
        value: '板块近一月 Top 3 暂不可用',
        detail: text(firstPresent(top3, ['message', 'reason'])) || '读取行业排名接口失败；未以当前 50 只持仓内部排名代替。',
        tone: 'flat',
      };
    }
    const unavailable = ['unavailable', 'missing', 'none', 'not_available', 'not-covered', 'not_covered'].includes(status);
    if (unavailable) {
      return {
        value: '板块近一月 Top 3 暂不可用',
        detail: text(firstPresent(top3, ['message', 'reason'])) || '缺少可验证的全行业成分或近 20 个交易日行情。',
        tone: 'flat',
      };
    }

    const rows = firstPresent(top3, ['rows', 'top3', 'top', 'items', 'leaders']);
    if (!Array.isArray(rows) || !rows.length) {
      return {
        value: '板块近一月 Top 3 暂不可用',
        detail: text(firstPresent(top3, ['message', 'reason'])) || '未返回可验证的全行业排名。',
        tone: 'flat',
      };
    }
    const labels = rows.slice(0, 3).map((entry) => {
      const name = firstPresent(entry, ['name', 'security_name', 'code', 'symbol']) || '—';
      const oneMonth = firstPresent(entry, ['return_1m', 'one_month_return', 'return_20d', 'twenty_day_return', 'change_1m']);
      return oneMonth === undefined ? text(name) : `${name} ${percent(oneMonth)}`;
    });
    const source = firstPresent(top3, ['source', 'provider']);
    const sourceLabel = source === 'eastmoney_board_members+tencent_daily_close' ? '东财行业成分、腾讯日线' : source;
    const asOf = firstPresent(top3, ['as_of', 'updated_at', 'date']);
    return {
      value: labels.join(' · '),
      detail: [sourceLabel && `来源 ${sourceLabel}`, asOf && `截至 ${asOf}`, '口径：该行业板块全部成分、近 20 个交易日'].filter(Boolean).join(' · '),
      tone: 'flat',
    };
  }

  function renderObservation(record) {
    const panel = ensureObservationPanel();
    if (!panel) return;
    if (window.QuantLabQuote && window.QuantLabQuote.kind === 'index') {
      panel.replaceChildren();
      const board = document.getElementById('terminal-board');
      const leaders = document.getElementById('terminal-leaders');
      if (board) board.hidden = true;
      if (leaders) leaders.hidden = true;
      return;
    }
    panel.replaceChildren();
    if (!record) {
      addStat(panel, '机构营收预测', '选择持仓后显示', '仅展示真实机构预测；不以已披露营收替代。', 'flat');
      addStat(panel, '板块近一月 Top 3', '选择持仓后显示', '仅展示全行业成分的近 20 个交易日排名。', 'flat');
      return;
    }
    if (!adapterState.publicMarketEnabled) {
      addStat(panel, '机构营收预测', '启用公开行情后读取', '只读取已选择股票的真实机构预测；不以已披露营收替代。', 'flat');
      addStat(panel, '板块近一月 Top 3', '启用公开行情后读取', '要求完整行业成分与近 20 个交易日行情；不会以当前 50 只持仓内部排名代替。', 'flat');
      return;
    }
    const forecast = forecastView(record);
    const top3 = sectorTop3View(record);
    const sectorName = text(firstPresent(record.sector, ['name']) || firstPresent(record.industry_top3_1m, ['sector_name']));
    if (sectorName && window.QuantLabQuote && typeof window.QuantLabQuote.rememberBoard === 'function') {
      window.QuantLabQuote.rememberBoard(adapterState.selectedCode, sectorName);
    }
    addStat(panel, '机构营收预测', forecast.value, forecast.detail, forecast.tone);
    renderBoardContext(record);
    renderLeaders(top3, record);
  }

  function renderBoardContext(record) {
    const panel = document.getElementById('terminal-observation-status');
    let host = document.getElementById('terminal-board');
    if (!host && panel) {
      host = document.createElement('div');
      host.id = 'terminal-board';
      host.className = 'board-context';
      panel.insertAdjacentElement('afterend', host);
    }
    if (!host) return;
    host.replaceChildren();
    const context = record && record.context;
    const focus = context && context.focus;
    if (!focus || !focus.name) {
      host.hidden = true;
      return;
    }
    host.hidden = false;
    const name = document.createElement('b');
    name.className = 'board-name';
    name.textContent = text(focus.name);
    host.append(name);
    if (isNumeric(focus.today_pct)) {
      const today = document.createElement('span');
      today.className = Number(focus.today_pct) >= 0 ? 'up' : 'down';
      today.textContent = `板块今日 ${percent(focus.today_pct)}`;
      host.append(today);
    }
    const breadth = context.breadth;
    if (breadth && isNumeric(breadth.up) && isNumeric(breadth.total)) {
      const follow = document.createElement('span');
      follow.textContent = `上涨 ${breadth.up}/${breadth.total}`;
      host.append(follow);
    }
    if (isNumeric(context.selected_rank_20d) && isNumeric(context.selected_count)) {
      const rank = document.createElement('span');
      rank.textContent = `近20日第 ${context.selected_rank_20d}/${context.selected_count}`;
      host.append(rank);
    }
    if (isNumeric(context.versus_board)) {
      const versus = document.createElement('span');
      const gap = Number(context.versus_board);
      versus.className = gap > 0 ? 'up' : gap < 0 ? 'down' : '';
      versus.textContent = gap === 0 ? '与板块持平' : `${gap > 0 ? '强于' : '弱于'}板块 ${percent(gap)}`;
      host.append(versus);
    }
    const leader = focus.today_leader;
    if (leader && leader.code && leader.name) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `当日龙头 ${leader.name}`;
      button.addEventListener('click', () => {
        if (window.QuantLabQuote && typeof window.QuantLabQuote.open === 'function') {
          window.QuantLabQuote.open(leader.code, leader.name);
        }
      });
      host.append(button);
    }
    const chain = Array.isArray(context.chain) ? context.chain : [];
    if (chain.length) {
      const line = document.createElement('div');
      line.className = 'board-chain';
      line.textContent = `行业链 ${chain.map((item) => {
        const label = text(item && item.name);
        return isNumeric(item && item.today_pct) ? `${label} ${percent(item.today_pct)}` : label;
      }).join(' · ')}`;
      host.append(line);
    }
  }

  function renderLeaders(top3, record) {
    let host = document.getElementById('terminal-leaders');
    const panel = document.getElementById('terminal-observation-status');
    if (!host && panel) {
      host = document.createElement('div');
      host.id = 'terminal-leaders';
      host.className = 'leader-board';
      const anchor = document.getElementById('terminal-board') || panel;
      anchor.insertAdjacentElement('afterend', host);
    }
    if (!host) return;
    const ranking = sectorTop3From(record) || {};
    const rows = firstPresent(ranking, ['rows', 'top3', 'top', 'items', 'leaders']);
    host.replaceChildren();
    if (!Array.isArray(rows) || !rows.length) {
      const note = document.createElement('div');
      note.className = 'leader-note';
      note.textContent = top3.value;
      note.title = top3.detail || '';
      host.append(note);
      return;
    }
    rows.slice(0, 3).forEach((entry, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      const code = text(firstPresent(entry, ['code', 'symbol']));
      const name = text(firstPresent(entry, ['name', 'security_name']) || code || '—');
      const change = firstPresent(entry, ['return_1m', 'one_month_return', 'return_20d', 'twenty_day_return', 'change_1m']);
      button.dataset.code = code;
      button.dataset.name = name;
      const rank = document.createElement('span');
      rank.className = 'leader-k';
      rank.textContent = `${text(firstPresent(ranking, ['sector_name']) || '板块')} · 第 ${index + 1}`;
      const title = document.createElement('span');
      title.className = 'leader-n';
      title.textContent = name;
      const value = document.createElement('span');
      value.className = `leader-r ${isNumeric(change) ? (Number(change) >= 0 ? 'up' : 'down') : ''}`;
      value.textContent = isNumeric(change) ? percent(change) : '—';
      button.append(rank, title, value);
      button.addEventListener('click', () => {
        if (window.QuantLabQuote && typeof window.QuantLabQuote.open === 'function') {
          window.QuantLabQuote.open(code, name);
        }
      });
      host.append(button);
    });
    if (top3.detail) {
      const note = document.createElement('div');
      note.className = 'leader-note';
      note.textContent = top3.detail;
      host.append(note);
    }
  }

  function responseObject(payload, names) {
    const direct = firstPresent(payload, names);
    if (direct && typeof direct === 'object') return direct;
    const nested = payload?.data;
    const nestedDirect = firstPresent(nested, names);
    if (nestedDirect && typeof nestedDirect === 'object') return nestedDirect;
    if (nested && typeof nested === 'object') return nested;
    return payload && typeof payload === 'object' ? payload : null;
  }

  async function readSelectedObservation(endpoint, code) {
    const url = new URL(endpoint, window.location.href);
    url.searchParams.set('code', code);
    const response = await nativeFetch(url.href, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  }

  function requestError(message) {
    return { status: 'request_error', message: `读取失败：${message || '未知错误'}` };
  }

  function selectedSupplemental(code) {
    if (!code) return;
    if (!adapterState.publicMarketEnabled) return;
    const previous = adapterState.supplementalByCode.get(code);
    // A selection can trigger both the click listener and a DOM mutation.  A
    // per-code session cache makes those paths idempotent and, crucially,
    // avoids requesting forecasts for all 50 rows on every price refresh.
    if (previous?.loadState === 'loading' || previous?.loadState === 'ready') return;

    adapterState.supplementalByCode.set(code, {
      loadState: 'loading',
      revenue_forecast: { status: 'loading' },
      industry_top3_1m: { status: 'loading' },
    });
    if (adapterState.selectedCode === code) renderObservation(selectedRecord());

    Promise.allSettled([
      readSelectedObservation(ENDPOINTS.revenueForecast, code),
      readSelectedObservation(ENDPOINTS.industryTop3, code),
    ]).then(([forecastResult, top3Result]) => {
      const revenueForecast = forecastResult.status === 'fulfilled'
        ? responseObject(forecastResult.value, ['revenue_forecast', 'forecast', 'result'])
        : requestError(forecastResult.reason?.message);
      const industryPayload = top3Result.status === 'fulfilled' ? top3Result.value : null;
      const industryTop3 = industryPayload
        ? responseObject(industryPayload, ['industry_top3_1m', 'top3', 'result'])
        : requestError(top3Result.reason?.message);
      const sector = industryPayload && typeof industryPayload.sector === 'object' ? industryPayload.sector : null;
      const context = industryPayload && typeof industryPayload.context === 'object' ? industryPayload.context : null;

      adapterState.supplementalByCode.set(code, {
        loadState: 'ready',
        revenue_forecast: revenueForecast || requestError('预测响应格式不正确'),
        industry_top3_1m: industryTop3 || requestError('行业排名响应格式不正确'),
        sector,
        context,
      });
      if (adapterState.selectedCode === code) renderObservation(selectedRecord());
    });
  }

  function setSelectedCode(code) {
    const selected = text(code).trim();
    if (!selected) return;
    adapterState.selectedCode = selected;
    renderObservation(selectedRecord());
    selectedSupplemental(selected);
  }

  function klineNote(payload) {
    const adjustment = ({ raw: '原始成交价', none: '未复权', qfq: '前复权', hfq: '后复权' })[payload?.adjust] || '复权口径待核';
    const asOf = firstPresent(payload, ['as_of', 'quote_time', 'date']) || '—';
    return `年K · ${adjustment} · 最后一根 ${asOf}`;
  }

  function currentFrequency() {
    return document.querySelector('#kFreq button.on')?.dataset?.freq || '';
  }

  function syncYearKNote(payload) {
    if (currentFrequency() !== 'year') return;
    const note = document.getElementById('qNote');
    if (note) note.textContent = klineNote(payload);
  }

  function installYearKControl() {
    const frequencies = document.getElementById('kFreq');
    if (!frequencies || frequencies.querySelector('[data-freq="year"]')) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.freq = 'year';
    button.textContent = '年K';
    button.title = '查看年 K 线';
    button.setAttribute('aria-label', '查看年 K 线');
    frequencies.append(button);

    // The recovered script owns quoteFreq and handles delegated clicks.  This
    // listener only corrects its day/week/month-only explanatory label after
    // the request comes back.
    frequencies.addEventListener('click', (event) => {
      const clicked = event.target.closest('button[data-freq="year"]');
      if (!clicked) return;
      window.setTimeout(() => {
        const code = adapterState.selectedCode || selectedFromDom();
        syncYearKNote(code ? adapterState.lastKlineByCode.get(code) : null);
      }, 0);
    });
  }

  function observeRecoveredKlineNote() {
    const note = document.getElementById('qNote');
    if (!note) return;
    const observer = new MutationObserver(() => {
      if (currentFrequency() !== 'year' || text(note.textContent).startsWith('年K ·')) return;
      const code = adapterState.selectedCode || selectedFromDom();
      syncYearKNote(code ? adapterState.lastKlineByCode.get(code) : null);
    });
    observer.observe(note, { childList: true, characterData: true, subtree: true });
  }

  function observeSelection() {
    const table = document.getElementById('htable');
    if (!table) return;
    table.addEventListener('click', (event) => {
      const row = event.target.closest('.quote-row[data-code]');
      if (!row) return;
      setSelectedCode(row.dataset.code);
    }, true);
    const observer = new MutationObserver(() => {
      const code = selectedFromDom();
      if (!code) return;
      setSelectedCode(code);
    });
    observer.observe(table, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
  }

  function updateFromPayload(detail) {
    const { endpoint, url, payload } = detail || {};
    if (endpoint === ENDPOINTS.holdings) {
      adapterState.publicMarketEnabled = Boolean(payload?.public_quotes_enabled);
      if (!adapterState.publicMarketEnabled) adapterState.supplementalByCode.clear();
      adapterState.holdingsByCode.clear();
      for (const row of holdingsRows(payload)) {
        const code = rowCode(row);
        if (code) adapterState.holdingsByCode.set(code, row);
      }
      // The recovered script paints the list and activates the first row after
      // it consumes this same response.  Resolve selection on the next task so
      // the adapter makes exactly one selected-security request, never 50.
      window.setTimeout(() => {
        if (window.QuantLabQuote && window.QuantLabQuote.kind === 'index') return;
        const selected = selectedFromDom() || adapterState.selectedCode;
        if (selected) setSelectedCode(selected);
        else renderObservation(selectedRecord());
      }, 0);
      return;
    }
    if (endpoint === ENDPOINTS.kline) {
      let code = '';
      try { code = new URL(url, window.location.href).searchParams.get('code') || ''; } catch (_error) { /* ignore */ }
      if (code) adapterState.lastKlineByCode.set(code, payload);
      if (currentFrequency() === 'year' && (!adapterState.selectedCode || adapterState.selectedCode === code)) {
        syncYearKNote(payload);
      }
    }
  }

  function install() {
    installYearKControl();
    ensureObservationPanel();
    observeRecoveredKlineNote();
    observeSelection();
    document.addEventListener('quantlab:open-symbol', (event) => {
      const code = text(event.detail && event.detail.code).trim();
      if (code) setSelectedCode(code);
    });
    renderObservation(selectedRecord());
  }

  window.addEventListener(PAYLOAD_EVENT, (event) => updateFromPayload(event.detail));
  // The recovered script also waits for DOMContentLoaded. Deferring one task
  // lets its original event bindings install first while keeping its layout
  // intact.
  document.addEventListener('DOMContentLoaded', () => window.setTimeout(install, 0), { once: true });
})();
