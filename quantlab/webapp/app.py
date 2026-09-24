"""Local simulation dashboard reading the latest integrity-checked versioned run. No archived root results/signals are promoted into current advice."""

from __future__ import annotations

import base64
import io
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from flask import Flask, jsonify, render_template_string, request

from quantlab.config import OUTPUT_DIR
from quantlab.portfolio.current import snapshot, summary, ytd_return
from quantlab.report.metrics import performance_table

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

_RESULTS: dict | None = None


def _load():
    from quantlab.portfolio.current import _load_result
    return _load_result()


def _clean_json(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantLab · 量化决策看板</title>
<style>
:root{--ink:#16233b;--mute:#6b778a;--line:#e4e9f2;--bg:#eef1f7;--card:#fff;
--blue:#2563eb;--red:#dc2626;--green:#059669;--amber:#d97706}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
margin:0;color:var(--ink);background:var(--bg);font-size:14px}
a{color:var(--blue);text-decoration:none}
.top{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid var(--line)}
.topbar{display:flex;align-items:center;gap:14px;padding:12px 22px}
.logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#2563eb,#6d28d9);
color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:17px}
.brand h1{font-size:16px;margin:0}
.brand .tag{font-size:11px;color:var(--mute)}
.st{display:flex;gap:18px;align-items:center;margin-left:auto;font-size:12px;color:var(--mute)}
.pill{border:1px solid var(--line);background:#f6f8fb;padding:3px 10px;border-radius:12px}
.nav{display:flex;gap:2px;padding:0 22px}
.nav button{border:none;background:none;padding:11px 16px 9px;font-size:14px;color:var(--mute);
border-bottom:2px solid transparent;cursor:pointer}
.nav button.on{color:var(--blue);border-color:var(--blue);font-weight:600}
.banner{background:#fffbeb;color:#945800;font-size:12px;padding:7px 22px;border-bottom:1px solid #fde9c8;line-height:1.55}
main{padding:18px 22px 40px;max-width:1180px;margin:0 auto}
.grid{display:grid;gap:14px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(132px,1fr))}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.kpi .k{font-size:12px;color:var(--mute)}
.kpi .v{font-size:19px;font-weight:700;margin-top:5px}
.kpi .s{font-size:11px;color:var(--mute);margin-top:2px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.card h2{font-size:14px;margin:0 0 12px;display:flex;align-items:center;gap:8px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block}
.dot.gray{background:#cbd5e1}
.chart{width:100%;height:auto;border-radius:10px;border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #eef1f7;padding:9px 10px;text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:600;font-size:12px}
tbody tr:hover{background:#f7f9fc}
.up{color:var(--red)} .down{color:var(--green)} .mute{color:var(--mute)}
.bolder{font-weight:700}
.chip{display:inline-block;background:#eef4ff;border:1px solid #d3e2ff;color:#1d4fd7;
border-radius:11px;padding:2px 9px;margin:2px;font-size:12px}
.chip .r{color:#7c8db5;margin-left:5px}
.tag{display:inline-block;font-size:11px;border-radius:9px;padding:2px 9px;color:#fff}
.tag.buy{background:var(--red)} .tag.hold{background:#94a3b8} .tag.sell{background:var(--green)}
.ex{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fbfcfe}
.ex h3{margin:0 0 2px;font-size:14px}
.ex .sub{font-size:11px;color:var(--mute);margin-bottom:10px}
.flex{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.btn{background:var(--blue);color:#fff;border:none;border-radius:8px;padding:7px 16px;font-size:13px;cursor:pointer}
.btn.ghost{background:#fff;color:var(--blue);border:1px solid var(--blue)}
.btn:disabled{opacity:.55;cursor:progress}
.note{font-size:12px;color:var(--mute);line-height:1.7}
#loading{display:none;font-size:12px;color:var(--blue)}
ol.run li{margin:7px 0}
.navcnt{font-size:11px;color:#7c8db5;font-weight:400}
@media(max-width:760px){.st{display:none}.topbar{padding:10px 14px}.nav{padding:0 10px}}
</style></head><body>
<div class="top">
  <div class="topbar">
    <div class="logo">QL</div>
    <div class="brand"><h1>QuantLab</h1><div class="tag">A股 多因子 · 信号级 · 人工执行</div></div>
    <div class="st">
      <span class="pill">数据源 A股真实行情</span>
      <span class="pill" id="stExec">执行日 —</span>
      <span class="pill" id="stPrice">行情 —</span>
    </div>
  </div>
  <div class="nav">
    <button data-tab="overview" class="on">总览</button>
    <button data-tab="holdings">实时持仓</button>
    <button data-tab="signals">买卖信号</button>
    <button data-tab="experts">专家决策</button>
    <button data-tab="run">运行指引</button>
  </div>
</div>
<div class="banner">当前展示模拟研究账户。历史结果尚未验证；数据不完整时不发布收益或可执行建议，不能视为券商持仓。</div>
<main>
  <div data-panel="overview">
    <div class="grid kpis" id="kpis"></div>
    <div class="card" style="margin-top:14px">
      <h2><span class="dot"></span>组合净值与回撤（A股 真实历史行情）</h2>
      <img class="chart" id="chart"/>
    </div>
    <div class="card" style="margin-top:14px" id="overviewExtra"></div>
  </div>

  <div data-panel="holdings" style="display:none">
    <div class="card" id="holdNote" style="margin-bottom:14px"></div>
    <div class="card">
      <h2><span class="dot"></span>当前持仓 · 实时行情
        <span class="mute" style="font-weight:400;font-size:12px" id="hMeta"></span>
        <button class="btn ghost" style="margin-left:auto" id="refreshH">立即刷新</button>
      </h2>
      <div class="grid kpis" id="hkpis" style="margin-bottom:12px"></div>
      <div style="overflow-x:auto"><table id="htable"></table></div>
    </div>
  </div>

  <div data-panel="signals" style="display:none">
    <div class="card">
      <h2><span class="dot"></span>最新一期买卖信号 <span class="mute" style="font-weight:400;font-size:12px" id="sMeta"></span></h2>
      <div style="overflow-x:auto"><table id="stable"></table></div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>信号含义</h2>
      <div class="note">信号按上一交易日收盘数据生成，T+1 参考执行价；月度调仓。买卖动作需人工复核成交明细后执行。</div>
    </div>
  </div>

  <div data-panel="experts" style="display:none">
    <div class="flex" style="margin-bottom:12px">
      <button class="btn" id="refreshE">重新收集今日信息并分析</button>
      <button class="btn ghost" id="reloadE">读取已有结果</button>
      <span id="loading">正在重算专家（10~30 秒）…</span>
    </div>
    <div class="card"><div class="note" id="eMeta"></div></div>
    <div id="ecards" style="margin-top:14px;display:grid;gap:12px"></div>
  </div>

  <div data-panel="run" style="display:none">
    <div class="grid kpis" id="runKpis"></div>
    <div class="card" style="margin-top:14px">
      <h2><span class="dot"></span>平台实时能力说明</h2>
      <div class="note">
        <ol class="run">
          <li><b>盘中实时行情</b>：持仓/行情接口每次请求都直接向新浪（腾讯回退）取实时价，页面每 30 秒自动刷新 —— 已支持。</li>
          <li><b>日线数据集</b>：每日收盘后运行一次 <code>scripts/daily_pulse.py</code>，增量补齐当日 K 线与候选池快照（新浪全A），因子/专家意见即基于最新数据重算（可加 cron 定时）。</li>
          <li><b>月度信号</b>：按运行配置评估；没有通过研究关口的信号只用于模拟。</li>
          <li><b>美股</b>：交易链路尚未启用；背景信息也必须满足历史可知时间要求。</li>
        </ol>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>每日一条命令</h2>
      <pre style="background:#f6f8fb;border:1px solid var(--line);border-radius:8px;padding:12px;font-size:12px;overflow:auto">
./.venv/bin/python scripts/daily_pulse.py          # 收盘后：更新日线+候选池 → 更新专家意见
./.venv/bin/python -m quantlab.cli quote           # 随时：查看当前持仓实时行情
./.venv/bin/python -m quantlab.cli monitor         # 随时：重算专家今日意见
./.venv/bin/python -m quantlab.cli web             # 启动本看板（本页）</pre>
    </div>
  </div>
</main>
<script>
const pct = v=> (v==null||isNaN(v)) ? '-' : (Number(v)*100).toFixed(2)+'%';
const num = (v,d=2)=> (v==null||isNaN(v)) ? '-' : Number(v).toFixed(d);
const cn = v=> (Number(v)||0)>=0 ? 'up':'down';
const sg = v=> (Number(v)||0)>0?'+':'';
async function gj(u){ const r=await fetch(u); if(!r.ok) throw new Error(r.status+''); return r.json(); }

function kpi(k,v,s,cls){ return `<div class="kpi"><div class="k">${k}</div><div class="v ${cls||''}">${v}</div><div class="s">${s||''}</div></div>`; }

function bindNav(){
  document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>{
    document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    [...document.querySelectorAll('[data-panel]')].forEach(p=>p.style.display = p.dataset.panel===b.dataset.tab?'':'none');
    const t=b.dataset.tab;
    if(t==='holdings') loadHoldings(); if(t==='signals') loadSignals();
    if(t==='experts') loadExperts(); if(t==='run') loadRun();
  });
}

async function loadOverview(){
  const d=await gj('/api/nav');
  document.getElementById('kpis').innerHTML=d.cards;
  document.getElementById('chart').src=d.chart;
  document.getElementById('stExec').textContent='执行日 '+d.last_exec;
  document.getElementById('overviewExtra').innerHTML=d.extra;
}

async function loadHoldings(){
  const r=await fetch('/api/holdings',{cache:'no-store'});
  const d=await r.json();
  const s=d.summary, y=d.ytd;
  document.getElementById('stPrice').textContent='行情 '+d.quote_time;
  const rg = s.regime_trend, sc = s.gross_scale;
  let note=`仓位口径：市场状态=「${rg===-1?'深度熊市':rg===0?'震荡':rg===1?'牛市':'-'}」，组合按中性风险档 ${pct(sc)} 仓位运行`;
  if(sc<1) note+='，处于风险降仓状态；未成交的退出仍受市场限制。';
  note+='　这是按股数计价的模拟账户，不代表实际成交或券商持仓。';
  document.getElementById('holdNote').innerHTML=`<div class="note">${note}</div>`;
  document.getElementById('hMeta').textContent=`组合最新 ${num(s.nav,4)} · 当周净值 ${pct(s.daily)} · 回撤 ${pct(s.drawdown)} · 今年以来 ${pct(y)} · 最新执行日 ${s.exec_date||'-'}`;
  const totW=d.rows.reduce((a,r)=>a+r.weight,0);
  const tv=d.rows.reduce((a,r)=>a+(r.market_value||0),0);
  document.getElementById('hkpis').innerHTML=
    kpi('持仓数',d.rows.length,d.rows.length+'只')+
    kpi('总权重',pct(totW),'中性档仓位')+
    kpi('模拟持仓市值','¥'+num(tv,0),'回放截至日')+
    kpi('今日组合(估)',pct(d.today),cn(d.today),cn(d.today))+
    kpi('今年组合',pct(y),'2026 起',cn(y));
  const th='<tr><th>代码</th><th>名称</th><th>目标权重</th><th>建仓日</th><th>成本</th><th>回放估值价</th><th>今日</th><th>今年</th><th>建仓至今</th><th>持仓市值</th><th>单手成本</th></tr>';
  const tr=d.rows.map(r=>`<tr>
    <td class="bolder">${r.code}</td><td>${r.name}</td>
    <td>${pct(r.weight)}</td><td>${r.entry_date||'-'}</td>
    <td>${num(r.cost)}</td>
    <td class="bolder">${num(r.price)} <span class="mute">${r.pct_chg!=null?pct(r.pct_chg):''}</span></td>
    <td class="${cn(r.return_ytd)}">${r.return_ytd!=null?sg(r.return_ytd)+pct(r.return_ytd):'-'}</td>
    <td class="${cn(r.pct_chg)}">${r.pct_chg!=null?sg(r.pct_chg)+pct(r.pct_chg):'-'}</td>
    <td class="${cn(r.return_since_entry)}">${r.return_since_entry!=null?sg(r.return_since_entry)+pct(r.return_since_entry):'-'}</td>
    <td>¥${num(r.market_value||0,0)}</td>
    <td>${r.lot_cost!=null?num(r.lot_cost,0):'-'}<span class="mute">${r.lot_cost!=null?(r.lot_affordable?'':' ↑整手可能超配'):''}</span></td></tr>`).join('');
  document.getElementById('htable').innerHTML=th+tr;
}

async function loadSignals(){
  const d=await gj('/api/signals');
  document.getElementById('sMeta').textContent=`信号日 ${d.signal_date} → 建议执行日 ${d.exec_date}`;
  const th='<tr><th>代码</th><th>名称</th><th>目标权重</th><th>上期权重</th><th>操作</th></tr>';
  const tr=d.trades.map(t=>`<tr><td class="bolder">${t[0]}</td><td>${(d.names||{})[t[0]]||''}</td>
    <td>${pct(t[1])}</td><td>${pct(t[2])}</td>
    <td><span class="tag ${t[3].includes('买入')?'buy':(/卖出|清仓|减仓/.test(t[3]))?'sell':'hold'}">${t[3]}</span></td></tr>`).join('');
  document.getElementById('stable').innerHTML=th+tr;
}

async function loadExperts(){
  const d=await gj('/api/experts');
  if(d.status!=='research_unvalidated'){document.getElementById('eMeta').textContent='模型有效性尚未验证';document.getElementById('ecards').innerHTML='';return;}
  document.getElementById('eMeta').innerHTML=
    `信息收集截至信号日 <b>${d.date}</b> · 候选 ${d.n_candidates} 只 · 行情因子 ${d.coverage?.factors||'-'} 个 · 缓存可算 ${d.coverage?.cache_daily||'-'} 只<br>`+
    `信息面覆盖：${d.coverage?.note||''}`;
  const cards=Object.entries(d.experts).map(([n,v])=>`
    <div class="ex"><h3>${n} <span class="mute" style="font-weight:400">置信 ${num(v.confidence)}</span></h3>
      <div class="sub">今日信息中，该专家按因子强度选出的 Top${v.top_num}</div>
      <div>${v.top.map(c=>`<span class="chip">${c.name} <span class="r">(${c.symbol})</span>${c.reason?'<span class="r">· '+c.reason+'</span>':''}</span>`).join('')}</div>
    </div>`).join('');
  const newNames=(d.new_names||[]).map(c=>`<span class="chip">${c.name} <span class="r">(${c.symbol})</span><span class="r">· 提及理由: ${c.reason||''}</span></span>`).join('')||'— 无';
  const dropped=(d.held_dropped_out||[]).map(c=>`<span class="chip">${c.name} <span class="r">(${c.symbol})</span></span>`).join('')||'— 全部被跟踪';
  document.getElementById('ecards').innerHTML=cards+`
    <div class="ex"><h3>专家新提及（当前未持仓）</h3><div class="sub">这些股票被至少一位专家在同一天的信息里单独点名放入 Top 池</div>
      <div>${newNames}</div></div>
    <div class="ex"><h3>当前持仓被所有专家遗忘</h3><div class="sub">如月报信号仍选入将维持；如连续多期未再被提及，需关注是否降级</div>
      <div>${dropped}</div></div>`;
}

async function loadRun(){
  const d=await gj('/api/nav');
  const cards=[
    kpi('盘中行情','实时','新浪/腾讯'),
    kpi('日线数据','每日更新','收盘后 daily_pulse'),
    kpi('研究信号','按配置评估','尚未获得实盘批准'),
    kpi('美股交易','未启用','背景信息须校验时序')];
  document.getElementById('runKpis').innerHTML=cards.join('');
}

document.getElementById('refreshH').onclick=()=>{loadHoldings();loadOverview();};
document.getElementById('refreshE').onclick=async()=>{
  const b=document.getElementById('refreshE'); b.disabled=true;
  document.getElementById('loading').style.display='inline';
  try{ await gj('/api/experts?refresh=1'); await loadExperts(); }catch(e){ alert('重算失败: '+e); }
  finally{b.disabled=false; document.getElementById('loading').style.display='none';}
};
document.getElementById('reloadE').onclick=()=>loadExperts();

bindNav(); loadOverview();
setInterval(()=>{ if(document.querySelector('[data-panel="overview"]').style.display!=='none') loadOverview(); },30000);
setInterval(()=>{ if(document.querySelector('[data-panel="holdings"]').style.display!=='none') loadHoldings(); },30000);
</script></body></html>"""


@app.get("/")
def index():
    data = _load()
    if not data:
        return "请先运行: python -m quantlab.cli backtest", 404
    return render_template_string(PAGE)


@app.get("/api/nav")
def api_nav():
    data = _load()
    if not data:
        return jsonify({"error": "no data"}), 404
    result = data["result"]
    if result.get("status") != "research_unvalidated" or not result.get('run_health',{}).get('research_interpretable',False):
        return jsonify({"status": result.get("status", "legacy_unvalidated"), "cards": [], "chart": "",
                        "last_exec": "", "extra": "历史结果或输入数据尚未验证，不发布收益与实盘建议。", "metrics": None})
    nav, dd = result["nav"], result["drawdown"]
    perf = performance_table(nav, result.get("daily_return"))
    y = ytd_return()
    cards = [
        kpi_card("总收益", f"{perf['total_return']:.2%}", "全期"),
        kpi_card("年化收益", f"{perf['annual_return']:.2%}", "研究回放，非实盘预期"),
        kpi_card("年化波动", f"{perf['annual_vol']:.2%}", "目标 ≤14%"),
        kpi_card("夏普比率", f"{perf['sharpe']:.2f}", ""),
        kpi_card("最大回撤", f"{perf['max_drawdown']:.2%}", "越小越好"),
        kpi_card("胜率(按日)", f"{perf['win_rate']:.2%}", "正收益交易日占比"),
        kpi_card("今年以来", f"{y:.2%}" if y is not None else "-", "2026-01-05 起"),
        kpi_card("卡玛比率", f"{perf['calmar']:.2f}", ""),
    ]
    extra = _extra_note(result)
    return jsonify({
        "cards": cards,
        "chart": _nav_chart_v2(nav, dd, result.get('benchmark')),
        "run_id": result.get('run_id'),
        "last_exec": str(nav.index[-1].date()),
        "extra": extra,
    })


def kpi_card(k: str, v: str, s: str = "") -> str:
    return f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


def _extra_note(result: dict) -> str:
    trades = result.get("trades") or []
    last = trades[-1] if trades else {}
    rg = last.get("regime", {}).get("trend")
    scale = float(last.get("risk", {}).get("gross_scale", 1.0)) if trades else 1.0
    reasons = last.get("risk", {}).get("reasons", [])
    state = {-1: "深度熊市(防守)", 0: "震荡(中性)", 1: "上行(进取)"}.get(rg, "—")
    src = result.get("benchmark_manifest", {}).get("source", "unknown")
    rows = [f"市场状态：<b>{state}</b> · 组合风险档位：<b>{scale:.0%} 仓</b>"]
    for r in (reasons or [])[:3]:
        rows.append(f"· {r}")
    from html import escape
    rows.append("数据源："+escape(str(src))+"；股票池覆盖："+escape(str(result.get('benchmark_manifest',{}).get('universe_coverage','unknown')))+"。基于每日账本；未完成策略有效性验证。")
    return f'<h2 style="font-size:14px;margin:0 0 8px"><span class="dot gray"></span>近期风控与口径</h2><div class="note">{"<br>".join(rows)}</div>'


@app.get("/api/holdings")
def api_holdings():
    rows = snapshot(with_quotes=True)
    summary_d = summary()
    today = 0.0
    for r in rows:
        if r["pct_chg"] is not None:
            today += r["weight"] * r["pct_chg"]
    quote_time = None
    if rows:
        q = rows[0].get("price")
        quote_time = rows[0].get('quote_time')
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
        return "-"


@app.get("/api/signals")
def api_signals():
    from quantlab.cli import signal_payload
    data = _load()
    if not data: return jsonify({"status": "blocked_data", "trades": [], "names": {}})
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


@app.get("/api/experts")
def api_experts():
    data = _load()
    if not data: return jsonify({"status": "blocked_data", "experts": {}})
    result = data["result"]
    trace = result.get("signal_trace", [])
    return jsonify(_clean_json({"status": result.get("status", "legacy_unvalidated"), "experts": {},
                    "note": "专家价值未通过验证；展示同一回放记录，不另行计算或执行。",
                    "date": result.get("as_of"), "n_candidates": len(result.get("codes", [])),
                    "coverage": {"note": "研究模型，未证明专家独立贡献"},
                    "decision": trace[-1] if trace else {}}))


def main(port: int = 8899) -> None:
    app.run(host="127.0.0.1", port=port, debug=False)


def _nav_chart_v2(nav: pd.Series, dd: pd.Series, benchmark=None) -> str:
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.4),
                           gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    ax[0].plot(nav.index, nav.values, color="#2563eb", lw=1.4)
    if benchmark is not None and not benchmark.empty:
        b=benchmark.close.reindex(nav.index)
        if b.notna().any():ax[0].plot(b.index,b/b.dropna().iloc[0]*nav.iloc[0],label='Benchmark price return',lw=1.)
        ax[0].legend()
    ax[0].set_title("组合净值", fontsize=12)
    ax[0].grid(alpha=0.25)
    ax[1].fill_between(dd.index, dd.values * 100, 0, color="#dc2626", alpha=0.5)
    ax[1].set_title("回撤(%)", fontsize=10)
    ax[1].grid(alpha=0.25)
    ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
