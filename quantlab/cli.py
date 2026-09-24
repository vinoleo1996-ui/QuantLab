"""QuantLab CLI：数据、回测、信号、实时行情、专家监控、Web 看板。

用法：
  python -m quantlab.cli fetch    --source akshare  # 拉取数据
  python -m quantlab.cli backtest --source auto     # 回测 + HTML报告
  python -m quantlab.cli signals  --source auto     # 输出本期买卖建议
  python -m quantlab.cli quote    [--symbols 600519,000001]  # 实时行情(缺省=当前持仓)
  python -m quantlab.cli monitor  --limit 100          # 每日专家监控意见
  python -m quantlab.cli web      --port 8899          # 四合一看板
--source: auto(默认,真实数据优先) | akshare | yfinance
说明: 已移除离线合成数据链路——所有真实后端失败时直接报错，拒绝人造回退。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from quantlab.config import OUTPUT_DIR, CONFIG


def build_panels(source: str, market: str, limit: int | None = None):
    """Snapshot collection only. Research replay requires a versioned manifest."""
    if market != "a": raise ValueError("US execution is disabled; background data only")
    from quantlab.data.sources import DataFeed
    feed = DataFeed(source=source)
    cand = feed.fetch_candidates(market)
    if limit:
        cand = cand.sort_values("float_mktcap", ascending=False).head(limit)
    panels = {}
    for code in cand.index:
        frame = feed.fetch_daily(market, str(code), refresh=True)
        if frame is not None and len(frame): panels[str(code)] = frame
    return panels, cand, feed.fetch_benchmark(market)


def run_backtest(source="auto", market="a", save=True, manifest=None, strategy="repaired_legacy"):
    from quantlab.backtest.engine import Engine, blocked_result
    from quantlab.data.contracts import load_dataset, DataContractError
    from quantlab.report.runs import persist_run, finalize_run_identity
    path = Path(manifest) if manifest else CONFIG.data.cache_dir.parent / "research" / "manifest.json"
    if not path.exists():
        result = blocked_result(["dataset_manifest_missing", str(path), "legacy_caches_are_not_PIT_raw_execution_data"])
    else:
        try: result = Engine(market=market).run(load_dataset(path), strategy=strategy)
        except (DataContractError, KeyError, ValueError) as exc: result = blocked_result([str(exc)])
    finalize_run_identity(result)
    if save:
        folder = persist_run(result)
        result["artifact_directory"] = str(folder)
    return result


def cmd_fetch(args):
    panels, _, bench = build_panels(args.source, args.market, args.limit)
    latest = {c: str(p.index[-1].date()) for c, p in panels.items()}
    print({"status": "public_cache_only", "latest_dates": latest,
           "note": "Fetch does not certify PIT universe, market rules or corporate actions."})


def cmd_backtest(args):
    from quantlab.report.html import build_html_report
    result = run_backtest(args.source, args.market, manifest=getattr(args, "manifest", None),
                          strategy=getattr(args, "strategy", "repaired_legacy"))
    folder = Path(result["artifact_directory"])
    report = build_html_report(result, title="QuantLab 研究验证报告", out=folder / "report.html")
    _print_perf(result)
    print(f"[backtest] {report}")


def _print_perf(result):
    from quantlab.report.metrics import performance_table
    print("[status]", result.get("status", "legacy_unvalidated"), result.get("reasons", []))
    if result.get("status") in ("blocked_data", "unverified_data_gaps"):
        print("[performance] 未验证，绩效指标不发布")
        return
    print("[research metrics; no live return claim]", performance_table(result["nav"], result["daily_return"]))
    for name, baseline in result.get("baselines", {}).items():
        print(name, performance_table(baseline["nav"], baseline["daily_return"]))


def signal_payload(result):
    meta = {k: result.get(k) for k in ("run_id", "data_version", "model_version", "as_of", "status", "reasons")}
    meta.update(promotion_approved=False, account_type="simulation", executable=False)
    if result.get("status") != "research_unvalidated":
        return dict(meta, signal_date=None, exec_date=None, trades=[], orders=[])
    trades = result.get("trades", [])
    latest = trades[-1] if trades else {}
    prior = result.get("account", [{}])[-1].get("weights", {})
    pending = [o for o in result.get("orders", []) if o["status"] in ("pending", "partial")]
    if not pending:
        return dict(meta, signal_date=latest.get("signal_date"), exec_date=None, trades=[], orders=[], note="没有待执行订单")
    target = latest.get("weights", {})
    rows = [(c, target.get(c, 0.), prior.get(c, 0.),
             "清仓" if target.get(c, 0.)==0 else "买入" if target.get(c, 0.)>prior.get(c, 0.) else "减仓" if target.get(c, 0.)<prior.get(c, 0.) else "持有")
            for c in sorted(set(target)|set(prior))]
    return dict(meta, signal_date=latest.get("signal_date"), exec_date=latest.get("exec_date"), trades=rows,
                orders=[o for o in result.get("orders", []) if o["status"] in ("pending", "partial")],
                note="研究模拟账户；实际持仓和成交须另行录入核对。")


def cmd_signals(args):
    import json
    from quantlab.report.runs import persist_run
    result = run_backtest(args.source, args.market, save=False, manifest=getattr(args, "manifest", None))
    folder = persist_run(result)
    payload = signal_payload(result)
    out = folder / "signals.json"
    out.write_text(json.dumps(payload, default=str, ensure_ascii=False, indent=2))
    print(json.dumps(payload, default=str, ensure_ascii=False, indent=2))
    print(out)


def cmd_quote(args) -> None:
    """实时行情查询：可按 --symbols 指定，缺省查当前持仓。"""
    from quantlab.data.realtime import batch_quotes
    from quantlab.portfolio.current import snapshot
    symbols = args.symbols
    if symbols:
        codes = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        holds = snapshot(with_quotes=False)
        codes = [h["code"] for h in holds[:args.limit]] or ["600519", "000001"]
    df = batch_quotes(codes)
    if df.empty:
        print("no quote available")
        return
    cols = ["code", "name", "price", "pre_close", "pct_chg", "amount", "time", "source"]
    print(df[cols].to_string(index=False))
    print(f"\n行情时间: {df['time'].iloc[0]}  数据源: {df['source'].iloc[0]}")


def cmd_monitor(args):
    import json
    from quantlab.strategy.monitor import daily_expert_pulse
    result = daily_expert_pulse(args.limit, manifest=getattr(args, "manifest", None))
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


def cmd_web(args) -> None:
    from quantlab.webapp.app import main as web_main
    web_main(args.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="quantlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "backtest", "signals", "web", "quote", "monitor"):
        sp = sub.add_parser(name)
        sp.add_argument("--source", default="auto",
                        choices=["akshare", "yfinance", "auto"])
        sp.add_argument("--market", default="a", choices=["a", "us"])
        sp.add_argument("--port", type=int, default=8899)
        sp.add_argument("--symbols", default="")
        sp.add_argument("--limit", type=int, default=25)
        sp.add_argument("--manifest", default=None, help="经过哈希校验的研究数据集清单")
        sp.add_argument("--strategy", default="repaired_legacy", choices=["repaired_legacy", "buy_hold", "monthly_equal", "vol_matched_equal"])
        sp.add_argument("--reset-cache", action="store_true", help="强制重拉全部历史(绕过本地缓存)")
    args = p.parse_args(argv)
    if getattr(args, "reset_cache", False):
        CONFIG.data.reset_cache = True
    cmds = {"fetch": cmd_fetch, "backtest": cmd_backtest, "signals": cmd_signals,
            "web": cmd_web, "quote": cmd_quote, "monitor": cmd_monitor}
    cmds[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])