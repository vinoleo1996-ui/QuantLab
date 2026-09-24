#!/usr/bin/env python3
"""Never turn current stars or an incomplete event stream into a verified annual ranking."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone

RELATED={
'microsoft/qlib','QuantConnect/Lean','freqtrade/freqtrade','microsoft/RD-Agent','OpenBB-finance/OpenBB',
'TauricResearch/TradingAgents','HKUDS/Vibe-Trading','virattt/ai-hedge-fund','ZhuLinsen/daily_stock_analysis','HKUDS/AI-Trader',
'nautechsystems/nautilus_trader','OpenByteInc/QuantDinger','Fincept-Corporation/FinceptTerminal','akfamily/akquant','TraderAlice/OpenAlice',
'hsliuping/TradingAgents-CN','ccxt/ccxt','mementum/backtrader','hummingbot/hummingbot','quantopian/zipline',
'goldmansachs/gs-quant','yutiansut/QUANTAXIS','polakowo/vectorbt','kernc/backtesting.py','jesse-ai/jesse',
'ranaroussi/quantstats','lballabio/QuantLib','charliedream1/ai_quant_trade','waditu/czsc','hugo2046/QuantsPlaybook',
'NoFxAiOS/nofx','tradesdontlie/tradingview-mcp','simonlin1212/TradingAgents-astock','alsk1992/CloddsBot',
'simonlin1212/Vibe-Research','quantskills/quantskills','qusong0627/QuantMind','QuantaAlpha/QuantaAlpha',
'simonlin1212/a-stock-data','shy3130/tick-stock-panel','muxuuu/serenity-skill','HiThink-Tech/Financial-API',
'hello245m/free-stockdb','simonlin1212/global-stock-data','jarrodwatts/jev-trader','mnemox-ai/tradememory-protocol',
'ranaroussi/yfinance','google/tf-quant-finance','JerBouma/FinanceToolkit','JerBouma/FinanceDatabase'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',default='output/upgrade_20260924/github_2026/candidate_metadata.json')
    p.add_argument('--output',default='output/upgrade_20260924/github_2026/ranking_audit.json');args=p.parse_args()
    raw=json.loads(Path(args.metadata).read_text());new=[];missing=[]
    for name in sorted(RELATED):
        r=raw.get(name)
        if r is None:missing.append(dict(repository=name,reason='metadata_missing'));continue
        if r['created_at']>='2026-01-01':new.append(dict(repository=name,created_at=r['created_at'],stars=r['stargazers_count'],
            net_stars_since_creation=r['stargazers_count'],scope='created_in_2026_within_searched_candidates'))
        else:missing.append(dict(repository=name,current_stars=r['stargazers_count'],reason='verified_2026_start_snapshot_missing'))
    new.sort(key=lambda x:(-x['stars'],x['repository']))
    out=dict(status='blocked_historical_ranking',year=2026,collection_date='2026-09-24',
        generated_at=datetime.now(timezone.utc).isoformat(),search_candidate_count=len(raw),related_candidate_count=len(RELATED),
        metric_required='current_total_stars_minus_verified_total_at_2026_start',verified_overall_top10=None,
        new_in_2026_candidate_top10=new[:10],missing_baselines=missing,
        reasons=['OSSInsight_event_history_declares_severe_degradation','ClickHouse_event_table_ends_2026_07_02',
                 'GitHub_timestamped_stargazers_HTTP_401','Wayback_baseline_lookup_timed_out'],
        caveat='New-repository table is not the requested all-repository annual-growth ranking. Search scope is declared, not exhaustive GitHub coverage.')
    Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
