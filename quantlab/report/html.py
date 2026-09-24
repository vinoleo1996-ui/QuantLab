"""HTML 可视化报告：净值曲线、指标表、因子IC、调仓明细。"""

from __future__ import annotations

import io
import json
from html import escape
import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pandas as pd


def _setup_cjk():
    for name in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Noto Sans CJK SC"):
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if path and "dejavu" not in path.lower():
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
        except Exception:  # noqa: BLE001
            continue


_setup_cjk()

from quantlab.config import OUTPUT_DIR
from quantlab.report.metrics import ic_icir_summary, performance_table


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _nav_chart(nav: pd.Series, drawdown: pd.Series, bench: pd.Series | None,
               title: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.6, 1]})
    ax1.plot(nav.index, nav.values, label="QuantLab", lw=1.4, color="#c0392b")
    if bench is not None:
        b = bench.reindex(nav.index)
        b = b / b.iloc[0] * nav.iloc[0]
        ax1.plot(b.index, b.values, label="Benchmark", lw=1.0, color="#5b6ee1", alpha=0.8)
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.set_ylabel("NAV")
    ax1.grid(alpha=0.25)
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="#e74c3c", alpha=0.5)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.25)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    return _png_b64(fig)


def _table(rows: list[list], header: list[str]) -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>"


def build_html_report(result: dict, bench: pd.Series | None = None,
                      title: str = "QuantLab 回测报告", out: Path | None = None) -> Path:
    if result.get("status", "legacy_unvalidated") in ("blocked_data", "unverified_data_gaps", "legacy_unvalidated") or (result.get("run_health") and not result["run_health"]["research_interpretable"]):
        out = out or OUTPUT_DIR / "report_v2.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        details = {k: result.get(k) for k in ("run_id", "data_version", "model_version", "as_of", "status", "reasons", "run_health")}
        out.write_text('<!doctype html><meta charset="utf-8"><h1>QuantLab：数据或结果尚未验证</h1>'
                       '<p>不发布绩效指标或可执行买卖建议。</p><pre>' + escape(json.dumps(details, ensure_ascii=False, indent=2)) + '</pre>')
        return out
    nav = result["nav"]
    if bench is None and result.get('benchmark') is not None and not result['benchmark'].empty:
        bench=result['benchmark'].close
    dr = result["daily_return"]
    dd = result["drawdown"]
    perf = performance_table(nav, dr, bench if bench is not None else None)
    ratios={'sharpe','calmar','beta'}
    stats_rows = [[k, (f"{v:.2f}" if k in ratios else f"{v:.2%}") if isinstance(v, float) else v]
                  for k, v in perf.items()]
    for name, base in result.get("baselines", {}).items():
        stats_rows += [[name + ":" + k, str(v)] for k, v in performance_table(base["nav"], base["daily_return"]).items()]
    chart = _nav_chart(nav, dd, bench, title)

    # 因子IC
    ic_df = ic_icir_summary(result.get("ic_matrix", pd.DataFrame()))
    ic_body = ""
    if not ic_df.empty:
        ic_body = _table([[r.factor, f"{r.ic_mean:.4f}", f"{r.icir:.3f}", f"{r.recent_ic:.4f}"]
                          for r in ic_df.itertuples()], ["因子", "IC均值", "ICIR", "近12期IC"])

    # 最近一期调仓
    trades = result.get("trades", [])
    last = trades[-1] if trades else {}
    sign_rows = []
    if last:
        for code, w in sorted(last.get("weights", {}).items(), key=lambda x: -x[1]):
            sign_rows.append([code, f"{w:.1%}"])
    sign_html = _table(sign_rows, ["代码", "目标权重"]) if sign_rows else "<p>无</p>"

    # 最近一期权重的买入/卖出建议
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<style>
body{{font-family:-apple-system,'PingFang SC',sans-serif;margin:24px;color:#222}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#f5f6fa}}
.grid{{display:flex;gap:24px;flex-wrap:wrap}}
.card{{border:1px solid #e2e2e2;border-radius:8px;padding:12px 16px;margin:8px 0}}
img{{max-width:100%}}
</style></head><body>
<h1>{escape(title)}</h1><p>状态：{escape(result.get("status", "legacy_unvalidated"))}；模拟研究，未获实盘批准。</p><p>数据版本：{escape(str(result.get("data_version")))}</p>
<div class="card"><h2>绩效指标</h2>{_table(stats_rows, ["指标", "数值"])}</div>
<img src="data:image/png;base64,{chart}"/>
<h2>因子 IC 统计</h2>{ic_body or "<p>暂无</p>"}
<h2>最新调仓(目标权重)</h2>{sign_html}
</body></html>"""
    out = out or OUTPUT_DIR / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
