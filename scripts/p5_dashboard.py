#!/usr/bin/env python3
"""Generate the single-file annual dashboard from output/p5_g0_20261017/annual.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/Users/agiuser/Desktop/QuantLab')
OUT = Path('/Users/agiuser/WorkBuddy/Worktrees/QuantLab/main-a9d77e8e/dashboard_annual.html')
data = json.loads((ROOT / 'output/p5_g0_20261017/annual.json').read_text())

import csv
base = ROOT / 'output/p5_base_orders'
if base.is_dir():
    latest = sorted(d for d in base.iterdir() if d.is_dir())[-1]
    orders = list(csv.DictReader((latest / 'orders.csv').open()))
    summary = json.loads((latest / 'summary.json').read_text())
    for o in orders:
        o['value'] = float(o['value']); o['weight'] = float(o['weight']); o['shares'] = int(o['shares']); o['limit'] = float(o['limit'])
        o['participation'] = float(o['participation']) if o['participation'] else None
    data['holdings'] = {'as_of': latest.name, 'orders': orders, 'summary': summary}

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantLab 年度收益看板</title>
<style>
:root{
  --bg:#0E1116; --card:#151A21; --line:rgba(255,255,255,.08); --line2:rgba(255,255,255,.14);
  --text:#E8EAED; --muted:#9AA0A6; --dim:#6B7280;
  --base:#8A8F98; --g0:#D9A441; --g97:#3BA99C; --red:#E5484D; --green:#2FA36B;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:26px 24px 40px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:18px}
h1{font-size:20px;font-weight:600;margin:0 0 6px}
.meta{font-size:12px;color:var(--muted);line-height:1.6}
.badge{font-size:11px;color:var(--g0);border:1px solid rgba(217,164,65,.35);background:rgba(217,164,65,.08);border-radius:20px;padding:4px 10px;white-space:nowrap}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .lab{font-size:12px;color:var(--muted)}
.kpi .val{font-size:26px;font-weight:600;margin-top:5px;font-variant-numeric:tabular-nums;letter-spacing:-.3px}
.kpi .sub{font-size:11px;color:var(--dim);margin-top:5px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card.full{grid-column:1/-1}
.card h2{font-size:14px;font-weight:600;margin:0 0 4px}
.card .hint{font-size:11px;color:var(--dim);margin:0 0 10px}
svg{display:block;width:100%;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:12px;color:var(--muted);margin-top:10px}
.legend span.it{display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.legend span.it.off{opacity:.35;text-decoration:line-through}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500;font-size:11.5px}
tr:last-child td{border-bottom:none}
td.best{font-weight:600;color:#fff}
.pos{color:var(--red)}
.neg{color:var(--green)}
.note{font-size:12px;color:var(--muted);line-height:1.75}
.note b{color:var(--text);font-weight:500}
.note .row{display:flex;gap:8px;margin-bottom:6px}
.note .k{color:var(--dim);min-width:92px;flex:none}
.foot{font-size:11px;color:var(--dim);margin-top:14px;line-height:1.7}
#tip{position:fixed;display:none;pointer-events:none;z-index:20;background:#1E242D;border:1px solid rgba(255,255,255,.14);border-radius:7px;padding:8px 10px;font-size:12px;color:var(--text);box-shadow:0 6px 20px rgba(0,0,0,.35);line-height:1.6}
#tip .tt{color:var(--muted);font-size:11px;margin-bottom:3px}
@media (max-width:900px){.kpis{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>QuantLab 年度收益看板</h1>
      <div class="meta">2018-01-02 → 2026-09-22 · 80 万 · 50 只 · 只做多 · 季度再平衡 · 打新按 1.5% 封顶单列</div>
    </div>
    <div class="badge">样本内回放 · 不是前向承诺</div>
  </header>

  <section class="kpis">
    <div class="kpi"><div class="lab">基线 ALLOC60（1/vol + 18% 线）</div><div class="val">11.76%</div><div class="sub">10.26% + 打新 1.5 · MDD −19.06%</div></div>
    <div class="kpi"><div class="lab">G0 60/23（恢复 3:3:2 袖）</div><div class="val" style="color:var(--g0)">14.20%</div><div class="sub">12.70% + 1.5 · MDD −19.11%</div></div>
    <div class="kpi"><div class="lab">G97 65/23（当前最佳）</div><div class="val" style="color:var(--g97)">14.69%</div><div class="sub">13.19% + 1.5 · MDD −22.78%</div></div>
    <div class="kpi"><div class="lab">成本 2 倍（G97 65/23）</div><div class="val">13.83%</div><div class="sub">12.33% + 1.5 · 门槛 10%</div></div>
  </section>

  <section class="kpis">
    <div class="kpi"><div class="lab">滚动 12m 胜率</div><div class="val">92%</div><div class="sub">93 个窗口，G97 65/23</div></div>
    <div class="kpi"><div class="lab">最差滚动 12m</div><div class="val" style="color:var(--green)">−17.2%</div><div class="sub">2018–19 那一段</div></div>
    <div class="kpi"><div class="lab">滚动 36m 最低 CAGR</div><div class="val">5.4%</div><div class="sub">没有亏损的 36 个月窗口</div></div>
    <div class="kpi"><div class="lab">2025 占对数收益</div><div class="val" style="color:var(--g0)">31%</div><div class="sub">剔除后年化 9.2%</div></div>
  </section>

  <main class="grid">
    <section class="card full">
      <h2>累计净值</h2>
      <p class="hint">组合层（不含打新），中位路径；2018 与 2026 为不完整年度</p>
      <svg id="navChart" viewBox="0 0 1160 380" role="img" aria-label="累计净值曲线"></svg>
      <div class="legend" id="navLegend"></div>
    </section>

    <section class="card">
      <h2>年度收益</h2>
      <p class="hint">按自然年；2026 截至 09-22</p>
      <svg id="retChart" viewBox="0 0 560 300" role="img" aria-label="年度收益柱状图"></svg>
    </section>

    <section class="card">
      <h2>年内最大回撤</h2>
      <p class="hint">每年净值从当年高点回落的最大幅度</p>
      <svg id="mddChart" viewBox="0 0 560 300" role="img" aria-label="年度最大回撤柱状图"></svg>
    </section>

    <section class="card full">
      <h2>滚动 12 个月收益</h2>
      <p class="hint">任意月末买入并持有 12 个月；G97 65/23 胜率 92%、最差 −17.2%</p>
      <svg id="rollChart" viewBox="0 0 1160 300" role="img" aria-label="滚动12个月收益曲线"></svg>
    </section>

    <section class="card full">
      <h2>逐年明细</h2>
      <p class="hint">收益 / 年内最大回撤；粗体为该年最高收益</p>
      <table id="yearTable"></table>
    </section>

    <section class="card full">
      <h2>当前目标持仓（纸面清单，不是真实持仓）</h2>
      <p class="hint" id="holdHint"></p>
      <div class="note" id="holdSum" style="margin-bottom:10px"></div>
      <div style="max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:8px">
        <table id="holdTable"></table>
      </div>
    </section>

    <section class="card full">
      <h2>研究台账（本轮已关闭的假设）</h2>
      <div class="note">
        <div class="row"><span class="k">P5_G0</span><span><b>通过</b>：恢复 G0 + 接 G97，13.19% / 含打新 14.69%，MDD −22.78%，三门槛全过。</span></div>
        <div class="row"><span class="k">P5_BASE</span><span><b>已登记</b>：G97 65% + G0 为基线；前向从 2026-10-30 首次快照起记，6 个月落后 3pp 停。</span></div>
        <div class="row"><span class="k">P5_E</span><span><b>关闭</b>：慢变量尾滤池级 +1.35pp（t=2.05），50 只账本只剩 +0.61（p10–p90 17.07–19.54），低于 +1pp 门槛。</span></div>
        <div class="row"><span class="k">P5_D</span><span><b>关闭</b>：500 万 → 300 万 ADV 不约束——池子 1597 对 1598 只，只差 1 只；下限远低于相对分位切点。</span></div>
      </div>
    </section>

    <section class="card full">
      <h2>稳健性：14.69% 是怎么来的，哪里会打折</h2>
      <div class="note">
        <div class="row"><span class="k">收益来源</span><span>G0 袖把 ETF 部分从 7.5% 拉到 12.70%（黄金 14.49 / 纳指 20.87 / 十年国债 4.17），60% 权重下 +2.4pp；G97 把股票账从 12.11 拉到 12.86，65% 权重下 +0.5pp。</span></div>
        <div class="row"><span class="k">2025 权重</span><span>2025 年 +37.1%，占 2018–26 全部对数收益的 <b>31%</b>；剔除 2025 后年化 <b>9.2%</b>。这是最大的单年依赖。</span></div>
        <div class="row"><span class="k">路径离散</span><span>8 条相位/种子路径的 CAGR p10–p90 为 <b>12.71–13.46%</b>；13.19 不是单条幸运路径。</span></div>
        <div class="row"><span class="k">袖的前向</span><span>G0 袖 2014–17 样本外（无金牛）只有 6.95%，2018–26 为 12.70%。袖按 7 / 10 / 12.7 计，组合约 <b>12.7 / 13.8 / 14.7</b>。</span></div>
        <div class="row"><span class="k">回撤尾部</span><span>2018 年 −18.1%（年内 −22.8%）、2024 年 −19.7%；黄金 2026-02-02 单日 −10.0%、03-23 −9.7%。</span></div>
        <div class="row"><span class="k">样本边界</span><span>股票腿只有 2018–26（分数从 2018 开始），没有 2014–17 样本外；袖有。前向两本账从 2026-10 起记。</span></div>
      </div>
    </section>
  </main>

  <div class="foot">口径：股票腿 = POOL50 / POOL50_G97 账本（成本 1 倍，除注明）；组合 = 股票袖 × 权重 + G0 袖（黄金 518880 15 / 十年国债 511260 15 / 纳指 513100 10，按 3:3:2 随权重缩放，季末再平衡）；打新 = 1.5% 封顶，仅加在 KPI 与门槛，不进入曲线。证据：output/p5_g0_20261017/annual.json。</div>
</div>
<div id="tip"></div>

<script>
const DATA = __DATA__;
const SERIES = [
  {k:'baseline', name:'基线 ALLOC60', color:'#8A8F98'},
  {k:'g0_60',   name:'G0 60/23',   color:'#D9A441'},
  {k:'g97_65',  name:'G97 65/23',  color:'#3BA99C'}
];
const hidden = new Set();
const tip = document.getElementById('tip');
function pct(v,d){ return (v*100).toFixed(d===undefined?1:d)+'%'; }
function spct(v,d){ const x=v*100; return (x>=0?'+':'')+x.toFixed(d===undefined?1:d)+'%'; }
function showTip(ev, html){ tip.innerHTML=html; tip.style.display='block'; const pad=14; let x=ev.clientX+pad, y=ev.clientY+pad; if(x+tip.offsetWidth>innerWidth-8) x=ev.clientX-tip.offsetWidth-pad; if(y+tip.offsetHeight>innerHeight-8) y=ev.clientY-tip.offsetHeight-pad; tip.style.left=x+'px'; tip.style.top=y+'px'; }
function hideTip(){ tip.style.display='none'; }

function drawNav(){
  const box=document.getElementById('navChart');
  const W=1160,H=380,ml=58,mr=18,mt=16,mb=34;
  const dates=DATA.series.baseline.nav_dates, n=dates.length;
  const vis=SERIES.filter(s=>!hidden.has(s.k));
  const vals=vis.flatMap(s=>DATA.series[s.k].nav);
  let lo=Math.min.apply(null,vals), hi=Math.max.apply(null,vals);
  lo=Math.floor(lo*10)/10; hi=Math.ceil(hi*10)/10;
  const X=i=>ml+(W-ml-mr)*i/(n-1);
  const Y=v=>mt+(hi-v)/(hi-lo)*(H-mt-mb);
  let h='';
  for(let t=0;t<=5;t++){ const v=lo+(hi-lo)*t/5; h+='<line x1="'+ml+'" y1="'+Y(v).toFixed(1)+'" x2="'+(W-mr)+'" y2="'+Y(v).toFixed(1)+'" stroke="rgba(255,255,255,.07)"/><text x="'+(ml-9)+'" y="'+(Y(v)+4).toFixed(1)+'" fill="#9AA0A6" font-size="11" text-anchor="end">'+v.toFixed(1)+'</text>'; }
  dates.forEach((d,i)=>{ if(d.slice(5,7)==='01' && (+d.slice(0,4))%2===0){ h+='<text x="'+X(i).toFixed(1)+'" y="'+(H-12)+'" fill="#9AA0A6" font-size="11" text-anchor="middle">'+d.slice(0,4)+'</text>'; }});
  vis.forEach(s=>{ const a=DATA.series[s.k].nav; let d=''; a.forEach((v,i)=>{ d+=(i?'L':'M')+X(i).toFixed(1)+','+Y(v).toFixed(1); }); h+='<path d="'+d+'" fill="none" stroke="'+s.color+'" stroke-width="2" stroke-linejoin="round"/>'; });
  h+='<line id="navV" x1="0" y1="'+mt+'" x2="0" y2="'+(H-mb)+'" stroke="rgba(255,255,255,.28)" stroke-dasharray="3 3" style="display:none"/>';
  vis.forEach(s=>{ h+='<circle id="navC_'+s.k+'" r="3.6" fill="'+s.color+'" style="display:none"/>'; });
  h+='<rect id="navHit" x="'+ml+'" y="'+mt+'" width="'+(W-ml-mr)+'" height="'+(H-mt-mb)+'" fill="transparent"/>';
  box.innerHTML=h;
  const hit=box.querySelector('#navHit');
  hit.addEventListener('mousemove',ev=>{
    const r=box.getBoundingClientRect(); const sx=(ev.clientX-r.left)/r.width*W;
    let i=Math.round((sx-ml)/((W-ml-mr)/(n-1))); i=Math.max(0,Math.min(n-1,i));
    const vline=box.querySelector('#navV'); vline.style.display=''; vline.setAttribute('x1',X(i)); vline.setAttribute('x2',X(i));
    let rows='';
    SERIES.forEach(s=>{ const c=box.querySelector('#navC_'+s.k); if(hidden.has(s.k)){ if(c)c.style.display='none'; return; } const v=DATA.series[s.k].nav[i]; if(c){ c.style.display=''; c.setAttribute('cx',X(i)); c.setAttribute('cy',Y(v)); } rows+='<div style="display:flex;justify-content:space-between;gap:16px"><span style="color:'+s.color+'">'+s.name+'</span><b>'+v.toFixed(3)+'</b></div>'; });
    showTip(ev,'<div class="tt">'+dates[i]+'</div>'+rows);
  });
  hit.addEventListener('mouseleave',()=>{ hideTip(); box.querySelector('#navV').style.display='none'; SERIES.forEach(s=>{ const c=box.querySelector('#navC_'+s.k); if(c)c.style.display='none'; }); });
}

function drawBars(id, field, ymin, ymax){
  const box=document.getElementById(id);
  const W=560,H=300,ml=46,mr=8,mt=12,mb=28;
  const years=Object.keys(DATA.series.baseline[field]), n=years.length;
  const gw=(W-ml-mr)/n, bw=Math.min(16,gw*0.23);
  const Y=v=>mt+(ymax-v)/(ymax-ymin)*(H-mt-mb);
  let h='';
  for(let t=0;t<=4;t++){ const v=ymin+(ymax-ymin)*t/4; h+='<line x1="'+ml+'" y1="'+Y(v).toFixed(1)+'" x2="'+(W-mr)+'" y2="'+Y(v).toFixed(1)+'" stroke="rgba(255,255,255,.07)"/><text x="'+(ml-7)+'" y="'+(Y(v)+4).toFixed(1)+'" fill="#9AA0A6" font-size="10.5" text-anchor="end">'+Math.round(v)+'%</text>'; }
  h+='<line x1="'+ml+'" y1="'+Y(0)+'" x2="'+(W-mr)+'" y2="'+Y(0)+'" stroke="rgba(255,255,255,.22)"/>';
  years.forEach((yr,i)=>{
    SERIES.forEach((s,j)=>{
      const raw=DATA.series[s.k][field][yr], v=raw*100;
      const cx=ml+gw*i+gw/2+(j-1)*(bw+3.5);
      const top=Math.min(Y(v),Y(0)), hh=Math.max(1,Math.abs(Y(v)-Y(0)));
      h+='<rect x="'+(cx-bw/2).toFixed(1)+'" y="'+top.toFixed(1)+'" width="'+bw+'" height="'+hh.toFixed(1)+'" fill="'+s.color+'" rx="1" data-t="'+yr+'|'+s.name+'|'+spct(raw)+'"/>';
    });
    h+='<text x="'+(ml+gw*i+gw/2).toFixed(1)+'" y="'+(H-8)+'" fill="#9AA0A6" font-size="10.5" text-anchor="middle">'+yr+'</text>';
  });
  box.innerHTML=h;
  box.querySelectorAll('rect[data-t]').forEach(r=>{
    r.addEventListener('mousemove',ev=>{ const p=r.getAttribute('data-t').split('|'); showTip(ev,'<div class="tt">'+p[0]+'</div><div>'+p[1]+' <b>'+p[2]+'</b></div>'); });
    r.addEventListener('mouseleave',hideTip);
  });
}

function drawTable(){
  const years=Object.keys(DATA.series.baseline.annual);
  let h='<thead><tr><th>年份</th><th>基线 收益</th><th>基线 回撤</th><th>G0 收益</th><th>G0 回撤</th><th>G97 收益</th><th>G97 回撤</th></tr></thead><tbody>';
  years.forEach(y=>{
    const a=DATA.series.baseline.annual[y], b=DATA.series.g0_60.annual[y], c=DATA.series.g97_65.annual[y];
    const best=Math.max(a,b,c);
    const cls=v=>v>=0?'pos':'neg';
    const bold=v=>v===best?' best':'';
    h+='<tr><td>'+y+(y==='2018'||y==='2026'?' *':'')+'</td>'
      +'<td class="'+cls(a)+bold(a)+'">'+spct(a)+'</td><td class="neg">'+pct(DATA.series.baseline.annual_mdd[y])+'</td>'
      +'<td class="'+cls(b)+bold(b)+'">'+spct(b)+'</td><td class="neg">'+pct(DATA.series.g0_60.annual_mdd[y])+'</td>'
      +'<td class="'+cls(c)+bold(c)+'">'+spct(c)+'</td><td class="neg">'+pct(DATA.series.g97_65.annual_mdd[y])+'</td></tr>';
  });
  h+='</tbody>';
  document.getElementById('yearTable').innerHTML=h;
}

function drawRoll(){
  const box=document.getElementById('rollChart');
  const W=1160,H=300,ml=58,mr=18,mt=14,mb=28;
  const dates=DATA.series.baseline.nav_dates, n=dates.length;
  const lo=-20, hi=60;
  const X=i=>ml+(W-ml-mr)*(i-12)/(n-13);
  const Y=v=>mt+(hi-v)/(hi-lo)*(H-mt-mb);
  let h='';
  for(let t=0;t<=4;t++){ const v=lo+(hi-lo)*t/4; h+='<line x1="'+ml+'" y1="'+Y(v).toFixed(1)+'" x2="'+(W-mr)+'" y2="'+Y(v).toFixed(1)+'" stroke="rgba(255,255,255,.07)"/><text x="'+(ml-9)+'" y="'+(Y(v)+4).toFixed(1)+'" fill="#9AA0A6" font-size="11" text-anchor="end">'+Math.round(v)+'%</text>'; }
  h+='<line x1="'+ml+'" y1="'+Y(0)+'" x2="'+(W-mr)+'" y2="'+Y(0)+'" stroke="rgba(255,255,255,.22)"/>';
  dates.forEach((d,i)=>{ if(i>=12 && d.slice(5,7)==='01' && (+d.slice(0,4))%2===0){ h+='<text x="'+X(i).toFixed(1)+'" y="'+(H-10)+'" fill="#9AA0A6" font-size="11" text-anchor="middle">'+d.slice(0,4)+'</text>'; }});
  SERIES.forEach(s=>{ if(hidden.has(s.k)) return; const a=DATA.series[s.k].nav; let d=''; for(let i=12;i<n;i++){ const v=(a[i]/a[i-12]-1)*100; d+=(i===12?'M':'L')+X(i).toFixed(1)+','+Y(v).toFixed(1); } h+='<path d="'+d+'" fill="none" stroke="'+s.color+'" stroke-width="2" stroke-linejoin="round"/>'; });
  box.innerHTML=h;
}

function drawLegend(){
  const box=document.getElementById('navLegend');
  box.innerHTML=SERIES.map(s=>'<span class="it'+(hidden.has(s.k)?' off':'')+'" data-k="'+s.k+'"><span class="sw" style="background:'+s.color+'"></span>'+s.name+'</span>').join('');
  box.querySelectorAll('.it').forEach(el=>el.addEventListener('click',()=>{ const k=el.getAttribute('data-k'); if(hidden.has(k))hidden.delete(k); else hidden.add(k); drawLegend(); drawNav(); }));
}

function drawHoldings(){
  const H=DATA.holdings;
  if(!H){ document.getElementById('holdHint').textContent='无订单清单'; return; }
  const s=H.summary;
  document.getElementById('holdHint').textContent='信号日 '+s.signal_date+' · 执行日 '+s.execution_date+' · 80 万 · 股票袖 '+Math.round(s.equity_share*100)+'% + ETF 袖 '+Math.round((1-s.equity_share)*100)+'% · 下次调仓约 2026-10-22（数据需追平至 10-21）';
  const skipped=(s.skipped_detail||[]).map(x=>x.code+'（'+x.reason+'）').join('、')||'无';
  document.getElementById('holdSum').innerHTML=
    '<div class="row"><span class="k">状态</span><span><b>纸面目标</b>：未接券商、无真实持仓、无真实成交；P5_BASE 前向首照 2026-10-30。</span></div>'
   +'<div class="row"><span class="k">股票腿</span><span>'+s.stock_names+' 只目标 → <b>'+s.stock_orders+' 单可下</b>（覆盖率 '+(s.gates.stock_coverage*100).toFixed(0)+'%）· 被拒 '+(s.skipped-(s.etf_orders?1:0))+'：'+skipped+'</span></div>'
   +'<div class="row"><span class="k">ETF 腿</span><span>'+s.etf_orders+' 单'+(s.qdii_paused?' · 纳指溢价 '+(s.nasdaq_premium*100).toFixed(2)+'% 触发 3% 暂停，8.75% 权重 1:1 分流给黄金与十年国债':'')+'</span></div>'
   +'<div class="row"><span class="k">资金</span><span>需款 '+(s.cash_needed/1e4).toFixed(2)+' 万（'+(s.cash_needed/s.capital*100).toFixed(1)+'%）· 收盘余款自动归集 511010'+(s.cash_sweep?'（+'+s.cash_sweep.shares+' 股）':'')+'</span></div>';
  const legName={stock:'股票',etf:'ETF',sweep:'余款归集'};
  let h='<thead><tr><th>代码</th><th>腿</th><th>方向</th><th>股数</th><th>限价</th><th>金额</th><th>权重</th><th>参与率</th></tr></thead><tbody>';
  H.orders.forEach(o=>{
    h+='<tr><td>'+o.code+'</td><td>'+legName[o.leg]+'</td><td class="pos">'+o.side+'</td><td>'+o.shares.toLocaleString()+'</td><td>'+o.limit.toFixed(2)+'</td><td>'+o.value.toFixed(0)+'</td><td>'+(o.weight*100).toFixed(2)+'%</td><td>'+(o.participation===null?'—':(o.participation*100).toFixed(2)+'%')+'</td></tr>';
  });
  document.getElementById('holdTable').innerHTML=h+'</tbody>';
}

drawLegend(); drawNav(); drawBars('retChart','annual',-25,45); drawBars('mddChart','annual_mdd',-25,0); drawRoll(); drawTable(); drawHoldings();
</script>
</body>
</html>
"""

html = HTML.replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
OUT.write_text(html, encoding='utf-8')
print('wrote', OUT, len(html))
