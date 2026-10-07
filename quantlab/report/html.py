"""回测报告生成：单文件、自包含、file:// 可直接打开。

采用独立的报告设计令牌与涨跌配色（红涨绿跌）。图表用内联 canvas 渲染而非外链库：报告要被归档、拷贝、
双击打开，不能假设机器上有 node_modules 或能联网。

数据以 JSON 内嵌（DATA 占位符），渲染逻辑全部内联——这样打开报告不需要
任何服务端，也不需要额外的资源文件。
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

from quantlab.config import OUTPUT_DIR
from quantlab.report.metrics import ic_icir_summary, performance_table

# These report tokens intentionally remain self-contained: an archived report
# must never depend on the local web application's stylesheet.
TOKENS = """
:root{
  --bg-base:#0a0d14;--bg-surface:#121826;--bg-card:#182032;--bg-card-hover:#1e2840;
  --bg-inset:#0f1420;--border-subtle:#232e47;--border-focus:#3b82f6;
  --text-primary:#f8fafc;--text-secondary:#94a3b8;--text-muted:#64748b;
  --accent-cyan:#06b6d4;--accent-blue:#3b82f6;--accent-green:#10b981;
  --accent-rose:#f43f5e;--accent-amber:#f59e0b;
  --up:#f43f5e;--down:#10b981;--flat:#94a3b8;
  --tint-up:rgba(244,63,94,.13);--tint-down:rgba(16,185,129,.13);
  --tint-warn:rgba(245,158,11,.12);--tint-info:rgba(59,130,246,.10);
  --font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  --font-mono:"SF Mono",ui-monospace,Menlo,Consolas,monospace;
  --radius-sm:6px;--radius-md:10px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg-base);color:var(--text-primary);font-family:var(--font-sans);
  font-size:13px;line-height:1.55;-webkit-font-smoothing:antialiased;padding:20px}
.num,.kpi .value,table.data td,table.data th.num,dd{font-variant-numeric:tabular-nums}
.mono{font-family:var(--font-mono)}
h1{font-size:17px;font-weight:650;letter-spacing:.2px}
h2{font-size:12.5px;font-weight:600;letter-spacing:.2px}
.wrap{max-width:1180px;margin:0 auto}
.head{display:flex;align-items:flex-start;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.head .mark{width:30px;height:30px;border-radius:8px;flex:0 0 auto;
  background:linear-gradient(135deg,var(--accent-cyan),var(--accent-blue));
  display:grid;place-items:center;color:#fff;font-weight:700;font-size:13px}
.head .sub{font-size:11.5px;color:var(--text-muted);margin-top:2px}
.spacer{flex:1}
.chip{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:999px;
  background:var(--bg-card);border:1px solid var(--border-subtle);font-size:11px;color:var(--text-secondary)}
.chip b{color:var(--text-primary);font-weight:600;font-family:var(--font-mono)}
.chip.ok{border-color:rgba(16,185,129,.42);background:var(--tint-down)}
.chip.warn{border-color:rgba(245,158,11,.45);background:var(--tint-warn)}
.chip.bad{border-color:rgba(244,63,94,.45);background:var(--tint-up)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--text-muted);flex:0 0 auto}
.dot.ok{background:var(--accent-green)}.dot.warn{background:var(--accent-amber)}.dot.bad{background:var(--accent-rose)}
.card{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-md);
  overflow:hidden;margin-bottom:12px}
.card-head{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--border-subtle)}
.card-body{padding:12px 14px}
.grid-kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:12px}
.kpi{padding:11px 13px;background:var(--bg-card);border:1px solid var(--border-subtle);
  border-radius:var(--radius-md)}
.kpi .label{font-size:10.5px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px}
.kpi .value{font-size:20px;font-weight:650;font-family:var(--font-mono);margin-top:3px;line-height:1.2}
.kpi .meta{font-size:10.5px;color:var(--text-muted);margin-top:3px}
.up{color:var(--up)}.down{color:var(--down)}.flat{color:var(--flat)}.muted{color:var(--text-muted)}
table.data{width:100%;border-collapse:collapse;font-size:12px}
table.data thead th{background:var(--bg-surface);color:var(--text-secondary);font-weight:600;
  font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;text-align:right;
  padding:8px 10px;border-bottom:1px solid var(--border-subtle);white-space:nowrap}
table.data thead th:first-child,table.data th.l,table.data td.l{text-align:left}
table.data tbody td{padding:6px 10px;border-bottom:1px solid var(--border-subtle);text-align:right;
  white-space:nowrap;font-family:var(--font-mono);font-size:11.5px}
table.data tbody td.l{font-family:var(--font-sans)}
table.data tbody tr:last-child td{border-bottom:none}
.tbl-scroll{overflow:auto;max-height:420px}
.callout{border-left:3px solid var(--accent-blue);background:var(--tint-info);padding:10px 13px;
  border-radius:0 var(--radius-sm) var(--radius-sm) 0;font-size:12px;line-height:1.65;color:var(--text-secondary)}
.callout b{color:var(--text-primary)}
.callout.warn{border-left-color:var(--accent-amber);background:var(--tint-warn)}
.callout.bad{border-left-color:var(--accent-rose);background:var(--tint-up)}
.callout pre{margin:8px 0 0;background:var(--bg-inset);padding:9px 11px;border-radius:var(--radius-sm);
  overflow-x:auto;font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);line-height:1.6}
.chart{width:100%;height:300px;display:block}
.legend-note{font-size:10.5px;color:var(--text-muted);padding:8px 14px;border-top:1px solid var(--border-subtle)}
.foot{margin-top:16px;font-size:10.5px;color:var(--text-muted);text-align:center;line-height:1.7}
@media(max-width:700px){body{padding:12px}.head{gap:8px}}
"""


def _fmt(v, kind: str) -> str:
    """Missing values render as —, never as 0 or -0.00%.

    Negative zero is normalised: a beta of -1e-18 is zero, and printing
    "-0.00" makes a reader hunt for a sign convention that does not exist.
    """
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "—"
    if kind == "pct":
        f *= 100
        if abs(f) < 0.005:
            f = 0.0
        return f"{f:.2f}%"
    if kind == "pct0":
        f *= 100
        return f"{f:.0f}%"
    if kind == "num":
        if abs(f) < 0.005:
            f = 0.0
        return f"{f:.2f}"
    return str(v)


def _dir_cls(v) -> str:
    if v is None:
        return "flat"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "flat"
    if f == 0 or f != f:
        return "flat"
    return "up" if f > 0 else "down"


def _blocked_report(result: dict, out: Path) -> Path:
    """Blocked runs get a styled explanation, not a bare <pre> dump."""
    details = {k: result.get(k) for k in
               ("run_id", "data_version", "model_version", "as_of", "status",
                "reasons", "run_health", "industry_policy", "research_exceptions")
               if result.get(k) is not None}
    body = escape(json.dumps(details, ensure_ascii=False, indent=2))
    html = f"""<!doctype html><html lang="zh" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantLab · 数据或结果尚未验证</title><style>{TOKENS}</style></head><body><div class="wrap">
<div class="head"><div class="mark">QL</div>
  <div><h1>QuantLab 研究验证报告</h1>
  <div class="sub">数据或结果尚未验证 — 不发布绩效指标或可执行买卖建议</div></div>
  <div class="spacer"></div><span class="chip bad"><span class="dot bad"></span>
  {escape(str(result.get('status', 'unvalidated')))}</span></div>
<div class="card"><div class="card-head"><h2>阻断详情</h2></div>
<div class="card-body"><div class="callout bad"><b>门禁未通过</b>，因此本报告不展示任何收益、回撤或交易建议。
<pre>{body}</pre></div></div></div>
<div class="foot">当前展示模拟研究账户。历史结果尚未验证；数据不完整时不发布收益或可执行建议，不能视为券商持仓。</div>
</div></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def build_html_report(result: dict, bench: pd.Series | None = None,
                      title: str = "QuantLab 回测报告", out: Path | None = None) -> Path:
    if (result.get("status", "legacy_unvalidated")
            in ("blocked_data", "unverified_data_gaps", "legacy_unvalidated")
            or (result.get("run_health") and not result["run_health"].get("research_interpretable"))):
        return _blocked_report(result, out or OUTPUT_DIR / "report_v2.html")

    nav = result["nav"].sort_index()
    if bench is None and result.get("benchmark") is not None and not result["benchmark"].empty:
        bench = result["benchmark"].close
    dr = result["daily_return"]
    dd = result.get("drawdown")
    dd = dd.reindex(nav.index) if dd is not None else pd.Series(index=nav.index, dtype=float)
    perf = performance_table(nav, dr, bench if bench is not None else None)

    # Chart payload: benchmark rebased onto the nav so the two lines are
    # comparable; a raw index level differs by orders of magnitude.
    bser = None
    if bench is not None and len(bench):
        b = pd.Series(bench).reindex(nav.index).ffill()
        if b.notna().any() and float(b.dropna().iloc[0]) != 0:
            bser = (b / float(b.dropna().iloc[0]) * float(nav.iloc[0])).round(6)
    payload = {
        "dates": [str(d.date()) for d in nav.index],
        "nav": [round(float(x), 6) for x in nav.values],
        "bench": [float(x) for x in bser.values] if bser is not None else None,
        "dd": [round(float(x), 6) for x in dd.values],
    }

    ratio_keys = {"sharpe", "calmar", "beta"}
    stat_rows = []
    for k, v in perf.items():
        if v is None:
            stat_rows.append((k, "—", "flat"))
            continue
        kind = "num" if k in ratio_keys else "pct"
        stat_rows.append((k, _fmt(v, kind), _dir_cls(v) if k in ("total_return", "annual_return") else ""))
    for name, base in (result.get("baselines") or {}).items():
        bp = performance_table(base["nav"], base["daily_return"])
        stat_rows.append((f"对照 {name}", _fmt(bp.get("annual_return"), "pct"),
                          _dir_cls(bp.get("annual_return"))))
    stat_rows.append(("行业约束政策", escape(str(result.get("industry_policy", "enforced"))), ""))
    exc = ", ".join(result.get("research_exceptions", []) or [])
    stat_rows.append(("研究例外", escape(exc) if exc else "无", ""))

    ic_df = ic_icir_summary(result.get("ic_matrix", pd.DataFrame()))
    ic_rows = "".join(
        f"<tr><td class='l'>{escape(str(r.factor))}</td><td>{r.ic_mean:.4f}</td>"
        f"<td>{r.icir:.3f}</td><td>{r.recent_ic:.4f}</td></tr>" for r in ic_df.itertuples())

    trades = result.get("trades") or []
    last = trades[-1] if trades else {}
    wrows = "".join(
        f"<tr><td class='l mono'>{escape(str(c))}</td><td>{float(w) * 100:.2f}%</td></tr>"
        for c, w in sorted((last.get("weights") or {}).items(), key=lambda kv: -kv[1]))

    bm = result.get("benchmark_manifest", {})
    prov = (f"数据源：{escape(str(bm.get('source', 'unknown')))}；"
            f"股票池覆盖：{escape(str(bm.get('universe_coverage', 'unknown')))}")

    # Data is injected as JSON; the renderer below is the only script in the
    # file, so the report works from file:// with no network and no assets.
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    status = escape(str(result.get("status", "")))
    data_version = escape(str(result.get("data_version")))
    nav_last = str(nav.index[-1].date())

    html = f"""<!doctype html><html lang="zh" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{TOKENS}</style></head><body><div class="wrap">

<div class="head">
  <div class="mark">QL</div>
  <div><h1>{escape(title)}</h1>
  <div class="sub">截至 {nav_last} · 数据版本 {data_version} · 模拟研究，未获实盘批准</div></div>
  <div class="spacer"></div>
  <span class="chip ok"><span class="dot ok"></span>状态 <b>{status}</b></span>
</div>

<div class="grid-kpi">
  <div class="kpi"><div class="label">总收益</div>
    <div class="value {_dir_cls(perf.get('total_return'))}">{_fmt(perf.get('total_return'),'pct')}</div>
    <div class="meta">全期扣费</div></div>
  <div class="kpi"><div class="label">年化收益</div>
    <div class="value {_dir_cls(perf.get('annual_return'))}">{_fmt(perf.get('annual_return'),'pct')}</div>
    <div class="meta">研究回放</div></div>
  <div class="kpi"><div class="label">年化波动</div>
    <div class="value">{_fmt(perf.get('annual_vol'),'pct')}</div><div class="meta">日频年化</div></div>
  <div class="kpi"><div class="label">最大回撤</div>
    <div class="value down">{_fmt(perf.get('max_drawdown'),'pct')}</div><div class="meta">越小越好</div></div>
  <div class="kpi"><div class="label">夏普</div>
    <div class="value">{_fmt(perf.get('sharpe'),'num')}</div><div class="meta">日频年化</div></div>
  <div class="kpi"><div class="label">卡玛</div>
    <div class="value">{_fmt(perf.get('calmar'),'num')}</div><div class="meta">年化/回撤</div></div>
</div>

<div class="card">
  <div class="card-head"><h2><span class="dot ok"></span>组合净值与回撤</h2></div>
  <div class="card-body"><canvas class="chart" id="navChart"></canvas>
    <div style="font-size:11px;color:var(--text-secondary);margin-top:6px">
      <span style="display:inline-block;width:9px;height:2px;background:var(--accent-cyan);vertical-align:middle"></span> 组合净值
      <span style="display:inline-block;width:14px;height:0;border-top:1px dashed var(--text-muted);vertical-align:middle;margin:0 4px 0 14px"></span> 基准
      <span style="display:inline-block;width:9px;height:2px;background:var(--down);vertical-align:middle;margin-left:4px"></span> 回撤
    </div>
  </div>
  <div class="legend-note">上下两栏共用时间轴：上为净值（起点归一），下为回撤（水下为负）。悬停查看当日数值。</div>
</div>

<div class="card">
  <div class="card-head"><h2>绩效指标</h2></div>
  <div class="card-body tight"><div class="tbl-scroll"><table class="data">
    <thead><tr><th class="l">指标</th><th>数值</th></tr></thead><tbody>
    {''.join(f"<tr><td class='l'>{escape(str(k))}</td><td class='{c}'>{v}</td></tr>" for k, v, c in stat_rows)}
  </tbody></table></div></div>
</div>

<div class="card">
  <div class="card-head"><h2>因子 IC 统计</h2></div>
  <div class="card-body tight">{(
    '<div class="tbl-scroll"><table class="data"><thead><tr><th class="l">因子</th>'
    '<th>IC均值</th><th>ICIR</th><th>近12期IC</th></tr></thead><tbody>'
    + ic_rows + '</tbody></table></div>') if ic_rows else
    '<div class="muted" style="padding:14px">本次回放没有 IC 矩阵。</div>'}</div>
</div>

<div class="card">
  <div class="card-head"><h2>最新一期目标权重</h2></div>
  <div class="card-body tight">{(
    '<div class="tbl-scroll"><table class="data"><thead><tr><th class="l">代码</th>'
    '<th>目标权重</th></tr></thead><tbody>' + wrows + '</tbody></table></div>') if wrows else
    '<div class="muted" style="padding:14px">无最近一期调仓记录。</div>'}</div>
</div>

<div class="card"><div class="card-body">
  <div class="callout warn"><b>口径与限制</b>：{prov}。基于每日账本；未完成策略有效性验证，
  本报告不构成投资建议，也不能视为券商持仓。</div>
</div></div>

<div class="foot">QuantLab · 研究验证报告 · 模拟账户 · 非实盘业绩</div>
</div>

<script>
const DATA={data_json};
(function(){{
  const c=document.getElementById('navChart');if(!c)return;
  const dpr=window.devicePixelRatio||1;
  const cssW=c.clientWidth, cssH=300;
  c.width=cssW*dpr;c.height=cssH*dpr;
  const g=c.getContext('2d');g.scale(dpr,dpr);
  const css=getComputedStyle(document.documentElement);
  const col=n=>(css.getPropertyValue(n)||'').trim();
  const accent=col('--accent-cyan'), mute=col('--text-muted'),
        down=col('--down'), line=col('--border-subtle'), txt=col('--text-secondary');
  const L=52,R=14,T=12,B=22, gap=26;
  const navH=Math.round((cssH-T-B-gap)*0.68), ddH=cssH-T-B-gap-navH;
  const N=DATA.dates.length;if(!N)return;
  const xs=i=>L+(cssW-L-R)*(N<2?0:i/(N-1));
  // nav panel
  const nv=DATA.nav;let lo=Infinity,hi=-Infinity;
  for(const v of nv){{if(v<lo)lo=v;if(v>hi)hi=v;}}
  if(DATA.bench)for(const v of DATA.bench){{if(v<lo)lo=v;if(v>hi)hi=v;}}
  if(hi===lo){{hi=lo+1;}}
  const pad=(hi-lo)*0.06;lo-=pad;hi+=pad;
  const ny=v=>T+navH-(v-lo)/(hi-lo)*navH;
  g.strokeStyle=line;g.fillStyle=txt;g.font='10px '+col('--font-mono');
  g.lineWidth=1;g.textAlign='right';g.textBaseline='middle';
  // One decimal count for the whole axis: per-tick branching produced a scale
  // like 256/210/164/117/71.21 and read as a rendering fault.
  const span=hi-lo, dec=span>=200?0:(span>=20?1:2);
  for(let i=0;i<=4;i++){{
    const v=lo+span*i/4,y=ny(v);
    g.beginPath();g.moveTo(L,y);g.lineTo(cssW-R,y);g.stroke();
    g.fillText(v.toFixed(dec),L-7,y);
  }}
  // nav area + line
  g.beginPath();g.moveTo(xs(0),T+navH);
  for(let i=0;i<N;i++)g.lineTo(xs(i),ny(nv[i]));
  g.lineTo(xs(N-1),T+navH);g.closePath();
  const gr=g.createLinearGradient(0,T,0,T+navH);
  gr.addColorStop(0,'rgba(6,182,212,.26)');gr.addColorStop(1,'rgba(6,182,212,0)');
  g.fillStyle=gr;g.fill();
  g.beginPath();
  for(let i=0;i<N;i++){{const x=xs(i),y=ny(nv[i]);i?g.lineTo(x,y):g.moveTo(x,y);}}
  g.strokeStyle=accent;g.lineWidth=1.7;g.stroke();
  if(DATA.bench){{
    g.beginPath();g.setLineDash([4,3]);
    for(let i=0;i<N;i++){{const x=xs(i),y=ny(DATA.bench[i]);i?g.lineTo(x,y):g.moveTo(x,y);}}
    g.strokeStyle=mute;g.lineWidth=1.1;g.stroke();g.setLineDash([]);
  }}
  // dd panel
  const dT=T+navH+gap;
  let dmin=0;for(const v of DATA.dd)if(v<dmin)dmin=v;
  dmin*=1.12;
  const dy=v=>dT+(0-v)/(0-dmin)*ddH;
  g.beginPath();g.moveTo(L,dy(0));g.lineTo(cssW-R,dy(0));g.strokeStyle=line;g.stroke();
  g.beginPath();g.moveTo(xs(0),dT+ddH);
  for(let i=0;i<N;i++)g.lineTo(xs(i),dy(DATA.dd[i]));
  g.lineTo(xs(N-1),dT+ddH);g.closePath();
  g.fillStyle='rgba(16,185,129,.18)';g.fill();
  g.beginPath();
  for(let i=0;i<N;i++){{const x=xs(i),y=dy(DATA.dd[i]);i?g.lineTo(x,y):g.moveTo(x,y);}}
  g.strokeStyle=down;g.lineWidth=1;g.stroke();
  g.fillStyle=txt;g.textAlign='right';g.fillText('0%',L-7,dy(0));
  g.fillText('-'+Math.abs(dmin*100).toFixed(Math.abs(dmin*100)<10?1:0)+'%',L-7,dT+ddH-4);
  // x labels
  g.textAlign='center';g.textBaseline='top';
  const step=Math.max(1,Math.floor(N/7));
  for(let i=0;i<N;i+=step){{
    const s=DATA.dates[i];g.fillText(s.slice(0,7),xs(i),cssH-B+5);
  }}
  // hover crosshair
  const hit=document.createElement('div');
  hit.style.cssText='position:fixed;pointer-events:none;background:'+col('--bg-card')+
    ';border:1px solid '+col('--border-subtle')+';border-radius:6px;padding:6px 9px;font:11px '+
    col('--font-mono')+';color:'+col('--text-primary')+';display:none;z-index:9;white-space:nowrap';
  document.body.appendChild(hit);
  c.addEventListener('mousemove',ev=>{{
    const r=c.getBoundingClientRect();
    const px=ev.clientX-r.left;
    let i=Math.round((px-L)/((cssW-L-R)/(N-1||1)));
    i=Math.max(0,Math.min(N-1,i));
    hit.style.display='block';
    hit.style.left=Math.min(window.innerWidth-190,ev.clientX+12)+'px';
    hit.style.top=Math.max(4,ev.clientY-46)+'px';
    const pct=(DATA.nav[i]/DATA.nav[0]-1)*100;
    hit.innerHTML=DATA.dates[i]+'<br>净值 '+DATA.nav[i].toFixed(4)+
      '<br>累计 '+(pct>=0?'+':'')+pct.toFixed(2)+'%<br>回撤 '+(DATA.dd[i]*100).toFixed(2)+'%';
  }});
  c.addEventListener('mouseleave',()=>{{hit.style.display='none';}});
  let t;window.addEventListener('resize',()=>{{clearTimeout(t);t=setTimeout(()=>location.reload(),400);}});
}})();
</script>
</body></html>"""
    out = out or OUTPUT_DIR / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
