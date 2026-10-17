#!/usr/bin/env python3
"""Local board for the high-liquidity retrain. Reads files already on disk."""
from __future__ import annotations

import json
import re
from pathlib import Path

from flask import Flask, jsonify, render_template_string

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/p4_premia_20261016'
LOG = Path('/Users/agiuser/.cursor/projects/Users-agiuser-Desktop-QuantLab/terminals/764577.txt')
LEDGER_LOG = Path('/Users/agiuser/.cursor/projects/Users-agiuser-Desktop-QuantLab/terminals/764582.txt')
YEARS = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>高成交重训</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 16px/1.5 "Songti SC", "Source Han Serif SC", serif;
         background: #14120e; color: #f3ead7; }
  main { max-width: 880px; margin: 0 auto; padding: 40px 24px 80px; }
  h1 { font-weight: 500; font-size: 32px; margin: 0 0 8px; }
  .sub { color: #b7ab96; margin-bottom: 28px; }
  .card { background: #1e1a15; border: 1px solid #3a3228; border-radius: 12px;
          padding: 18px 20px; margin: 14px 0; }
  .k { color: #b7ab96; font-size: 13px; letter-spacing: 0.04em; }
  .v { font-size: 22px; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: right; padding: 8px 6px; border-bottom: 1px solid #3a3228; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #b7ab96; font-weight: 500; }
  .bar { height: 8px; background: #3a3228; border-radius: 99px; overflow: hidden; margin-top: 10px; }
  .bar span { display: block; height: 100%; background: #e2b15a; }
  .wait { color: #e2b15a; }
</style>
</head>
<body>
<main>
  <h1>高成交重训</h1>
  <div class="sub" id="model"></div>
  <div class="card">
    <div class="k">训练进度</div>
    <div class="v" id="progress"></div>
    <div class="bar"><span id="fill"></span></div>
  </div>
  <div class="card">
    <div class="k">样本</div>
    <div class="v" id="sample"></div>
  </div>
  <div class="card">
    <table>
      <thead><tr><th>重训日</th><th>训练行数</th><th>下一年预测行数</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <div class="card">
    <div class="k">种子 1 的平方误差</div>
    <div class="v" id="curve-title">曲线还在补记。</div>
    <svg id="chart" viewBox="0 0 800 220" width="100%" height="220"></svg>
  </div>
  <div class="card">
    <div class="k">账户</div>
    <div class="v wait" id="account">分数还没写完，账本还没开始。</div>
  </div>
</main>
<script>
async function tick() {
  const data = await (await fetch('/status.json')).json();
  document.getElementById('model').textContent = data.model;
  document.getElementById('progress').textContent = data.progress;
  document.getElementById('fill').style.width = data.pct + '%';
  document.getElementById('sample').textContent = data.sample;
  document.getElementById('rows').innerHTML = data.refits.map(row =>
    `<tr><td>${row.refit}</td><td>${row.train_rows}</td><td>${row.test_rows}</td></tr>`).join('');
  document.getElementById('account').textContent = data.account;
  document.getElementById('curve-title').textContent = data.curve_title;
  const svg = document.getElementById('chart');
  const train = data.train_l2 || [];
  const valid = data.valid_l2 || [];
  const n = Math.max(train.length, valid.length);
  if (!n) { svg.innerHTML = ''; return; }
  const all = train.concat(valid);
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = hi - lo || 1;
  const line = (series, color) => {
    const pts = series.map((value, i) => {
      const x = 40 + (720 * i) / Math.max(n - 1, 1);
      const y = 20 + 180 * (1 - (value - lo) / span);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    return `<polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}"/>`;
  };
  svg.innerHTML = line(train, '#8d8272') + line(valid, '#e2b15a')
    + '<text x="40" y="16" fill="#b7ab96" font-size="12">灰线是训练集，黄线是验证集</text>';
}
tick();
setInterval(tick, 5000);
</script>
</body>
</html>
"""


def status():
    sample = {}
    meta = OUT / 'samples_meta.json'
    if meta.exists():
        sample = json.loads(meta.read_text())
    refits = []
    running = False
    if LOG.exists():
        text = LOG.read_text(errors='ignore')
        running = 'exit_code:' not in text.split('---')[-1] and 'status: running' in text
        for line in text.splitlines():
            if '"refit"' not in line:
                continue
            try:
                refits.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    done = len(refits)
    pct = min(100, round(100 * done / len(YEARS)))
    if (OUT / 'scores.parquet').exists():
        progress = f'九个年份都已训完，分数已落盘'
        pct = 100
    elif done == 0:
        progress = '还没有一个年份写完'
    else:
        progress = f'已完成 {done} / {len(YEARS)} 个年份，最近写完 {refits[-1]["refit"]}'
    books = []
    if LEDGER_LOG.exists():
        for line in LEDGER_LOG.read_text(errors='ignore').splitlines():
            if '"cagr"' not in line:
                continue
            try:
                books.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if books:
        latest = books[-1]
        account = f"已完成 {len(books)} / 15 本。最近一本 {latest['path']} 年化 {latest['cagr']:.2%}"
    elif (OUT / 'scores.parquet').exists():
        account = '15 本账本正在回放，第一本年化还没写出来。'
    else:
        account = '分数还没写完，账本还没开始。'
    curve_title = '曲线还在补记。这次正式训练没有保存每棵树的误差。'
    train_l2, valid_l2 = [], []
    curve_path = OUT / 'train_curve.json'
    if curve_path.exists():
        curves = json.loads(curve_path.read_text())
        if curves:
            last = curves[-1]
            train_l2, valid_l2 = last.get('train_l2') or [], last.get('valid_l2') or []
            curve_title = f"{last['refit']} 种子 1，验证集最后是 {valid_l2[-1]:.5f}" if valid_l2 else curve_title
    report = OUT / 'report.json'
    if report.exists():
        body = json.loads(report.read_text())
        account = f"高成交年化 {body.get('liq_high_cagr', 0):.2%}，合并账 {body.get('mix_cagr', 0):.2%}"
    rows = int(sample.get('rows') or 0)
    days = int(sample.get('days') or 0)
    return dict(
        model='LightGBM 梯度提升树，5 个种子取平均。叶子 31，学习率 0.03，叶子最少 500 行，特征抽样 0.7，样本抽样 0.7，L2 为 1，最多 1000 棵，早停耐心 50。目标是 5 日开盘收益。',
        progress=progress,
        pct=pct,
        sample=f'{days} 个交易日，{rows / 10000:.1f} 万行，24 个特征' if rows else '样本还没写完',
        refits=[{'refit': row['refit'],
                 'train_rows': f"{row['train_rows'] / 10000:.1f} 万",
                 'test_rows': f"{row['test_rows'] / 10000:.1f} 万"} for row in refits],
        account=account,
        running=running,
        curve_title=curve_title,
        train_l2=train_l2,
        valid_l2=valid_l2,
    )


@app.get('/')
def index():
    return render_template_string(PAGE)


@app.get('/status.json')
def status_json():
    return jsonify(status())


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8765, debug=False)
