"""QuantLab versioned research, data audit, shared signals, forward paper accounts and public-cache utilities. --source applies to public fetching; research uses --manifest exclusively."""

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


def run_backtest(source="auto", market="a", save=True, manifest=None, strategy="opportunity", forecasts=None, data_mode='strict', feature_profile=None):
    from quantlab.backtest.engine import Engine, blocked_result
    from quantlab.data.contracts import load_dataset, DataContractError
    from quantlab.report.runs import persist_run, finalize_run_identity
    path = Path(manifest) if manifest else CONFIG.data.cache_dir.parent / "research" / "manifest.json"
    if not path.exists():
        result = blocked_result(["dataset_manifest_missing", str(path), "legacy_caches_are_not_PIT_raw_execution_data"])
    else:
        try:
            from quantlab.data.qualification import audit_manifest
            from copy import deepcopy
            cfg=deepcopy(CONFIG)
            if feature_profile:cfg.research.feature_profile=feature_profile
            if data_mode=='free_research':cfg.research.universe_mode='pit_liquidity'
            assessment=audit_manifest(path,profile=data_mode,require_industry=cfg.research.feature_profile=='price_volume_industry')
            if assessment['status'] not in ('qualified_for_daily_research','research_ready'):result=blocked_result(assessment['reasons'])
            else:
                ds=load_dataset(path,profile=data_mode)
                if isinstance(forecasts,(str,Path)):
                    from quantlab.strategy.opportunity import load_forecasts
                    forecasts=load_forecasts(forecasts,ds.version)
                result=Engine(cfg,market=market).run(ds,strategy=strategy,forecasts=forecasts,data_mode=data_mode)
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
                          strategy=getattr(args, "strategy", "opportunity"), forecasts=getattr(args,"forecasts",None),
                          data_mode=getattr(args,'data_mode','strict'),feature_profile=getattr(args,'feature_profile',None))
    folder = Path(result["artifact_directory"])
    report = build_html_report(result, title="QuantLab 研究验证报告", out=folder / "report.html")
    _print_perf(result)
    print(f"[backtest] {report}")


def _print_perf(result):
    from quantlab.report.metrics import performance_table
    print("[run_health]", result.get("run_health", {}))
    print("[status]", result.get("status", "legacy_unvalidated"), result.get("reasons", []))
    if result.get("status") in ("blocked_data", "unverified_data_gaps") or (result.get("run_health") and not result["run_health"]["research_interpretable"]):
        print("[performance] 未验证，绩效指标不发布")
        return
    print("[research metrics; no live return claim]", performance_table(result["nav"], result["daily_return"]))
    for name, baseline in result.get("baselines", {}).items():
        print(name, performance_table(baseline["nav"], baseline["daily_return"]))


def signal_payload(result):
    meta = {k: result.get(k) for k in ("run_id", "data_version", "model_version", "as_of", "status", "reasons", "data_mode", "data_assessment")}
    meta.update(promotion_approved=False, account_type="simulation", executable=False)
    if result.get("status") != "research_unvalidated":
        return dict(meta, signal_date=None, exec_date=None, trades=[], orders=[])
    trades = result.get("trades", [])
    latest = trades[-1] if trades else {}
    state = result.get("account", [{}])[-1]
    prior = state.get("weights", {})
    pending = [o for o in result.get("orders", []) if o["status"] in ("pending", "partial")]
    if not pending:
        return dict(meta, signal_date=latest.get("signal_date"), exec_date=None, trades=[], orders=[], note="没有待执行订单")
    rows=[]
    for o in pending:
        c=o['instrument_id'];qty=state.get('quantities',{}).get(c,0)
        delta=o['remaining']*(1 if o['side']=='buy' else -1)
        target_qty=max(0,qty+delta)
        target_weight=prior.get(c,0)*target_qty/qty if qty else latest.get('weights',{}).get(c,0.)
        rows.append((c,target_weight,prior.get(c,0.),'买入' if delta>0 else '清仓' if target_qty==0 else '减仓'))
    return dict(meta, signal_date=latest.get("signal_date"), exec_date=latest.get("exec_date"), trades=rows,
                orders=pending,
                note="方向与数量来自待执行订单；权重仅为估计。逐单信号日及未成交原因见orders。研究模拟账户，须另行核对实际成交。")


def cmd_signals(args):
    import json
    from quantlab.report.runs import persist_run
    result = run_backtest(args.source, args.market, save=False, manifest=getattr(args, "manifest", None), strategy=args.strategy, forecasts=args.forecasts,
                          data_mode=args.data_mode,feature_profile=args.feature_profile)
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
    result = daily_expert_pulse(args.limit, manifest=getattr(args, "manifest", None), strategy=args.strategy, forecasts=args.forecasts,
                               data_mode=args.data_mode,feature_profile=args.feature_profile)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


def cmd_web(args) -> None:
    from quantlab.webapp.app import main as web_main
    web_main(args.port)


def cmd_research(args):
    import json
    from quantlab.data.qualification import audit_manifest
    from quantlab.data.contracts import load_dataset
    from quantlab.pipeline import run_suite,DEFAULT_PROTOCOL
    report=audit_manifest(args.manifest)
    if report['status']!='qualified_for_daily_research':
        folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
        from quantlab.data.contracts import fingerprint
        (folder/('blocked_'+fingerprint(report)[:12]+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    protocol=json.loads(Path(args.protocol).read_text()) if args.protocol else DEFAULT_PROTOCOL
    print(json.dumps(run_suite(load_dataset(args.manifest),args.output,protocol),ensure_ascii=False,indent=2,default=str))


def cmd_forward(args):
    import json
    from quantlab.forward import ForwardBook
    from quantlab.data.contracts import load_dataset
    from quantlab.strategy.opportunity import Forecast
    book=ForwardBook(args.book,args.model_version)
    if args.cmd=='forward-status':result=book.status()
    elif args.cmd=='forward-reconcile':
        if not args.fills:raise ValueError('--fills required')
        result=book.reconcile(json.loads(Path(args.fills).read_text()))
    else:
        from quantlab.data.qualification import audit_manifest
        assessment=audit_manifest(args.manifest)
        if assessment['status']!='qualified_for_daily_research':print(json.dumps(assessment,ensure_ascii=False));return
        ds=load_dataset(args.manifest)
        if args.cmd=='forward-recover':
            from quantlab.strategy.opportunity import load_forecasts
            result=book.recover(ds,load_forecasts(args.forecasts,ds.version) if args.forecasts else {})
            print(json.dumps(result,ensure_ascii=False,default=str,indent=2));return
        f=None
        if args.model:
            from quantlab.strategy.opportunity import load_model,opportunity_features,normalized_features
            from quantlab.data.market import panels_for_research,universe_at
            model=load_model(args.model);d=pd.Timestamp(ds.manifest['as_of']).normalize()
            _,amount,_=panels_for_research(ds);eligible,_=universe_at(ds,d,amount,CONFIG)
            f=model.predict(normalized_features(opportunity_features(ds)),d,eligible)
        if args.forecast:
            row=json.loads(Path(args.forecast).read_text());row['expected_returns']=pd.Series(row['expected_returns'])
            f=Forecast(**row)
        result=book.step(ds,f)
    print(json.dumps(result,ensure_ascii=False,default=str,indent=2))


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
        sp.add_argument("--strategy", default="opportunity", choices=["opportunity", "buy_hold", "monthly_equal", "vol_matched_equal"])
        sp.add_argument("--forecasts",default=None,help="研究流水线冻结的预测文件")
        sp.add_argument('--data-mode',choices=['strict','free_research'],default='strict')
        sp.add_argument('--feature-profile',choices=['price_volume','price_volume_industry'],default=None)
        sp.add_argument("--reset-cache", action="store_true", help="强制重拉全部历史(绕过本地缓存)")
    audit=sub.add_parser('data-audit');audit.add_argument('--manifest',default='data/research/manifest.json')
    audit.add_argument('--data-mode',choices=['strict','free_research'],default='strict')
    audit.add_argument('--feature-profile',choices=['price_volume','price_volume_industry'],default='price_volume_industry')
    free=sub.add_parser('free-research');free.add_argument('--manifest',default='data/free_research/dataset/manifest.json')
    free.add_argument('--output',default='output/free_research');free.add_argument('--train-start',default='2017-01-01');free.add_argument('--test-start',default='2020-01-01')
    compare=sub.add_parser('compare-strategies');compare.add_argument('--manifest',default='data/free_research_120/dataset/manifest.json')
    compare.add_argument('--output',default='output/return_comparison')
    evidence=sub.add_parser('evidence-audit');evidence.add_argument('--bundle',required=True)
    research=sub.add_parser('research');research.add_argument('--manifest',default='data/research/manifest.json')
    research.add_argument('--protocol');research.add_argument('--output',default='output/research_v3')
    for name in ('forward-step','forward-status','forward-reconcile','forward-recover'):
        fp=sub.add_parser(name);fp.add_argument('--book',required=True);fp.add_argument('--model-version',required=True)
        fp.add_argument('--manifest',default='data/research/manifest.json');fp.add_argument('--forecast');fp.add_argument('--fills');fp.add_argument('--model');fp.add_argument('--forecasts')
    args = p.parse_args(argv)
    if args.cmd=='compare-strategies':
        import json
        from quantlab.comparison import run_comparison
        result=run_comparison(args.manifest,args.output)
        print(json.dumps({k:v for k,v in result.items() if k not in ('selection','data_assessment')},ensure_ascii=False,default=str,indent=2));return
    if args.cmd=='free-research':
        import json
        from quantlab.free_research import run_free_research
        print(json.dumps(run_free_research(args.manifest,args.output,train_start=args.train_start,test_start=args.test_start),ensure_ascii=False,default=str,indent=2));return
    if args.cmd=='evidence-audit':
        import json
        from quantlab.evidence import verify_bundle
        path=Path(args.bundle)
        print(json.dumps(verify_bundle(json.loads(path.read_text()),path.parent),ensure_ascii=False,indent=2,default=str));return
    if args.cmd=='data-audit':
        import json
        from quantlab.data.qualification import audit_manifest
        print(json.dumps(audit_manifest(args.manifest,profile=args.data_mode,require_industry=args.feature_profile=='price_volume_industry'),ensure_ascii=False,indent=2));return
    if args.cmd=='research':
        cmd_research(args);return
    if args.cmd.startswith('forward-'):
        cmd_forward(args);return
    if getattr(args, "reset_cache", False):
        CONFIG.data.reset_cache = True
    cmds = {"fetch": cmd_fetch, "backtest": cmd_backtest, "signals": cmd_signals,
            "web": cmd_web, "quote": cmd_quote, "monitor": cmd_monitor}
    cmds[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
