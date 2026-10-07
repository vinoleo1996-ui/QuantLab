#!/usr/bin/env python3
"""A new 2025 cash account using the unchanged frozen quality/PB P2 strategy.

The prior exchange session is a cash-only signal preparation day. It is kept in
the raw ledger, and can be excluded from reported performance only after its
cash, holdings, NAV and lack of fills have been checked. No old runtime is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import sys
import traceback

import continuity_cadence as shared

ROOT = shared.ROOT
OUT = ROOT / 'output/account_2025_20260927'
START, END, PREP = '2025-01-02', '2026-09-22', '2024-12-31'
MANIFEST = OUT / 'extension/dataset/manifest.json'
DATABASE = Path('/Users/agiuser/Desktop/A股ETF量化策略/data/panel/catalog.duckdb')


def now():
    return datetime.now(timezone.utc).isoformat()


def account_window(calendar, start=START, end=END):
    """Select an actual exchange start, with the preceding legal monthly signal."""
    import pandas as pd
    shared.load_frozen()
    from quantlab.strategy.decision import review_dates
    calendar = pd.DatetimeIndex(calendar)
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if sessions.empty or sessions[0] != pd.Timestamp(start) or sessions[-1] != pd.Timestamp(end):
        raise ValueError('requested_window_not_full_exchange_sessions')
    prior = calendar[calendar < sessions[0]]
    if prior.empty or prior[-1] not in review_dates(calendar, 'monthly'):
        raise ValueError('no_prior_month_end_for_first_session_entry')
    return str(prior[-1].date()), sessions


def inventory():
    """Small read-only queries; no dataset/score preparation or market collection."""
    import duckdb
    import pandas as pd
    manifest = json.loads(shared.MANIFEST.read_text())
    tables = [('daily_bars', 'trading_day'), ('adjustment_factors', 'trading_day'),
              ('financial_records', 'announced_on'), ('security_states', 'effective_on'),
              ('price_limits', 'trading_day'), ('market_flow', 'trading_day'),
              ('global_risk', 'china_trading_day'), ('industry_history', 'effective_on'),
              ('trading_calendar', 'calendar_day')]
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        measurements = {}
        for table, field in tables:
            rows, first, last = con.execute(
                f'SELECT COUNT(*), MIN({field}), MAX({field}) FROM {table}').fetchone()
            measurements[table] = dict(rows=rows, first=str(first), last=str(last))
        sources = con.execute('SELECT source_name,COUNT(*),COUNT(DISTINCT symbol),'
            'MIN(trading_day),MAX(trading_day) FROM daily_bars GROUP BY 1 ORDER BY 1').fetchall()
        recent = con.execute("SELECT trading_day,COUNT(*) FROM daily_bars "
            "WHERE trading_day > '2026-06-30' GROUP BY 1 ORDER BY 1 DESC LIMIT 12").fetchall()
        schema = con.execute('DESCRIBE daily_bars').fetchall()
    roots = [ROOT / 'data/local_supplement/raw', ROOT / 'data/free_research_120/raw']
    raw = []
    for root in roots:
        ranges = Counter()
        for path in root.glob('*_daily.json'):
            meta = json.loads(path.read_text())
            ranges[(meta.get('requested_start'), meta.get('requested_end'))] += 1
        raw.append(dict(root=str(root), intervals=[dict(start=a,end=b,files=n)
            for (a,b),n in ranges.items()]))
    cohort_path = roots[1] / 'cohort.json'
    cohort = json.loads(cohort_path.read_text())
    benchmark_path = roots[1] / 'benchmark.csv'
    benchmark = pd.read_csv(benchmark_path, usecols=['date'])
    actions_path = ROOT / 'data/corporate_actions/raw/corporate_actions.json'
    actions_meta = json.loads(actions_path.read_text())
    refs = [cohort_path, benchmark_path, benchmark_path.with_suffix('.json'), actions_path,
        ROOT / 'data/local_supplement/raw/collection_contract.json',
        ROOT / 'quantlab/data/local_adapter.py',
        DATABASE.parents[2] / 'a_share_quant/baostock_source.py']
    report = dict(captured_at=now(), access='read_only', database=str(DATABASE),
        database_sha256=shared.sha256(DATABASE), tables=measurements,
        bar_sources=[dict(source=s,rows=n,securities=c,first=str(a),last=str(b))
                     for s,n,c,a,b in sources],
        recent_bar_counts=[dict(date=str(d),rows=n) for d,n in recent],
        daily_bar_columns=[r[0] for r in schema], raw_daily_coverage=raw,
        small_cohort=dict(path=str(cohort_path),start=cohort['start'],end=cohort['end'],
                          scope=cohort['scope'],securities=len(cohort['codes'])),
        benchmark_raw=dict(path=str(benchmark_path),first=benchmark.date.min(),last=benchmark.date.max()),
        corporate_actions_raw=actions_meta,
        frozen_snapshot=dict(path=str(shared.MANIFEST),as_of=manifest['as_of'],
            securities=len(manifest['panels']),data_version=shared.DATA_VERSION,
            manifest_sha256=shared.sha256(shared.MANIFEST)),
        evidence_sha256={str(p):shared.sha256(p) for p in refs},
        extension_assessment=dict(status='not_same_input_coverage_after_2026_06_30',
            latest_local_price_date=measurements['daily_bars']['last'],
            current_account_cutoff=manifest['as_of'],
            reasons=['5419 verified turnover daily histories all stop at 2026-06-30',
                     'newer turnover raw is only a fixed 120-stock cohort',
                     'catalog daily_bars has no turnover column; BaoStock normalized records do not retain requested turn',
                     'separate cash/share corporate-action collection requested only through 2026-06-30'],
            implication='Direct newer export has missing turnover for most stocks. Flow/crowding scoring and candidate coverage would change; do not label that output the same-input main strategy.',
            required_followup='Versioned market-wide incremental turnover/status raw and 2026-07 onward corporate actions, then new manifest and date coverage checks.',
            benchmark_is_blocker=False), promotion_approved=False)
    shared.freeze_json(OUT / 'data_inventory.json', report)
    return report


def data_version():
    audit=json.loads((OUT/'extension/extension_audit.json').read_text())
    if not audit['passed']:raise ValueError('extension_input_coverage_audit_not_passed')
    if shared.sha256(MANIFEST)!=audit['manifest_sha256']:raise ValueError('extension_manifest_hash_mismatch')
    return audit['data_version']


def protocol_identity():
    import pandas as pd
    shared.load_frozen()
    from quantlab.strategy.decision import review_dates
    manifest = json.loads(MANIFEST.read_text())
    calendar_path = MANIFEST.parent / manifest['calendar']['path']
    shared.verify_file(calendar_path, manifest['calendar']['sha256'])
    calendar = pd.DatetimeIndex(pd.read_parquet(calendar_path).session)
    prep, sessions = account_window(calendar)
    if prep != PREP:
        raise ValueError('preparation_day_mismatch')
    config = asdict(shared.effective_config())
    if config['capital']['init_cash'] != 600000 or config['capital']['a_share_positions'] != 20:
        raise ValueError('main_account_configuration_changed')
    return dict(version='new-2025-quality-pb-p2-v1',
        question='With 600000 cash at the start of 2025, what did the current frozen main strategy earn after costs?',
        hypothesis='A fresh account has different holdings, cash and drawdown history from a slice of the old 2020 account.',
        decision_changed='Report the current strategy new-account experience; no parameter selection, strategy promotion, or new Q1/Q2 experiment.',
        source=shared.source_identity(), harness_sha256=shared.sha256(__file__),
        shared_harness_sha256=shared.sha256(Path(shared.__file__)),
        tests_sha256=shared.sha256(ROOT / 'tests/test_account_2025_review.py'),
        manifest_path=str(MANIFEST),manifest_sha256=shared.sha256(MANIFEST),
        expected_data_version=data_version(),
        inventory_sha256=shared.sha256(OUT / 'data_inventory.json'),
        extension_audit_sha256=shared.sha256(OUT/'extension/extension_audit.json'),
        supplementary_financials='See extension/financial_probe/; exact source paths and rows are in the final manifest.',
        effective_config=json.loads(json.dumps(config, default=str)), spec=shared.SPEC,
        account_start=START, account_end=END, signal_preparation_day=prep,
        sessions=shared.date_strings(sessions), calendar_sha256=manifest['calendar']['sha256'],
        ordinary_signal_dates=shared.date_strings(d for d in review_dates(calendar,'monthly')
            if pd.Timestamp(prep)<=d<=pd.Timestamp(END)),
        initialization=dict(cash=600000,quantities={},receivables=0,payables=0,
            inherit_old_runtime=False,controller_initial_state='normal',
            controller_transition_count=0,account_drawdown=0,
            rule='Engine creates a new Ledger and PolicyController. Prior-session benchmark evaluation follows unchanged Engine behavior. No prior account history.'),
        preparation='Keep full frozen history for rolling features, monthly matured IC and benchmark; prepare dataset once; replay from cash-only 2024-12-31; execute initial monthly order on 2025-01-02.',
        reporting='Raw ledger retains 2024-12-31. Remove that single day only if cash=initial NAV, no positions/claims/fills. Include all first-trading-day costs and returns.',
        execution='Original monthly close decision, following exchange opening; no execution delay override; daily P2 risk unchanged.',
        result_storage='Full account/orders/fills/risk/signals retained. Large input panels and runtime are referenced by immutable manifest instead of duplicated in result pickle.',
        promotion_approved=False, data_status='conditional_free_research',
        limitations=['Account ends on 2026-09-22; no later performance claim', 'Not independent OOS',
            'Existing PIT/industry/corporate-action/execution and forward-validation gates still apply'])


def preflight():
    protocol = protocol_identity()
    path = OUT / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('protocol_identity_mismatch')
    else:
        shared.freeze_json(path, protocol)
    return dict(status='preflight_ready',protocol_sha256=shared.sha256(path),replay_started=False)


def replay_account(dataset, cfg=None, *, start=START, end=END):
    shared.load_frozen()
    from quantlab.backtest.engine import Engine, prepare
    from quantlab.strategy.legacy import legacy_config
    prep, _ = account_window(dataset.calendar, start, end)
    cfg = legacy_config(cfg or shared.effective_config())
    prepared = prepare(dataset, cfg)
    return Engine(cfg).run_dataset(dataset, strategy='legacy_repaired', start=prep, end=end,
        prepared=prepared, data_mode='free_research',industry_policy='enforced',
        risk_policy='P2',candidate='quality_pb',allocation_policy='legacy')


def reporting_account(result, *, start=START, end=END, prep=PREP):
    """Fail closed if excluding the preparation session would hide a real position/loss."""
    import numpy as np
    import pandas as pd
    accounts = result['account']
    dates = pd.DatetimeIndex(pd.to_datetime([s['date'] for s in accounts]))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('invalid_account_calendar')
    initial = float(result['config']['capital']['init_cash'])
    before = [s for s,d in zip(accounts,dates) if d < pd.Timestamp(start)]
    if len(before)!=1 or pd.Timestamp(before[0]['date'])!=pd.Timestamp(prep):
        raise ValueError('account_must_have_exactly_one_cash_preparation_day')
    first = before[0]
    if (first['nav']!=initial or first['cash']!=initial or any(first['quantities'].values()) or
        first.get('receivables',0)!=0 or first.get('payables',0)!=0 or
        any(pd.Timestamp(f['date'])<pd.Timestamp(start) for f in result.get('fills',[]))):
        raise ValueError('preparation_day_has_economic_activity')
    snapshots = [s for s,d in zip(accounts,dates) if pd.Timestamp(start)<=d<=pd.Timestamp(end)]
    if not snapshots or pd.Timestamp(snapshots[0]['date'])!=pd.Timestamp(start) or pd.Timestamp(snapshots[-1]['date'])!=pd.Timestamp(end):
        raise ValueError('reporting_window_mismatch')
    nav = pd.Series([s['nav'] for s in snapshots],index=pd.to_datetime([s['date'] for s in snapshots]),dtype=float)
    if not np.isfinite(nav).all() or (nav<=0).any():
        raise ValueError('invalid_account_nav')
    stored = result['nav'].copy()
    stored.index = pd.to_datetime(stored.index)
    if not np.allclose(stored.reindex(nav.index),nav/initial,atol=1e-12,rtol=0):
        raise ValueError('stored_nav_mismatch')
    daily = nav.pct_change(fill_method=None)
    daily.iloc[0] = nav.iloc[0]/initial-1
    if 'daily_return' in result:
        stored = result['daily_return'].copy()
        stored.index = pd.to_datetime(stored.index)
        if not np.allclose(stored.reindex(nav.index),daily,atol=1e-12,rtol=0):
            raise ValueError('stored_daily_return_mismatch')
    return snapshots,nav,daily


def summarize(result, *, start=START, end=END, prep=PREP):
    import numpy as np
    import pandas as pd
    snapshots,nav,daily = reporting_account(result,start=start,end=end,prep=prep)
    initial = float(result['config']['capital']['init_cash'])
    fills = result.get('fills', [])
    stages = {}
    for year in sorted(set(nav.index.year)):
        dates = nav.index[nav.index.year==year]
        before = nav.loc[nav.index<dates[0]]
        base = float(before.iloc[-1]) if len(before) else initial
        selected = nav.reindex(dates)
        stage_fills = [f for f in fills if pd.Timestamp(f['date']) in dates]
        stage_states = [s for s in snapshots if pd.Timestamp(s['date']) in dates]
        stages[str(year)] = dict(first_session=str(dates[0].date()),last_session=str(dates[-1].date()),
            initial_boundary_nav=base, terminal_nav=float(selected.iloc[-1]),
            net_profit=float(selected.iloc[-1]-base),net_return=float(selected.iloc[-1]/base-1),
            max_drawdown=float((selected/selected.cummax().clip(lower=base)-1).min()),
            mean_gross=float(np.mean([sum(s['weights'].values()) for s in stage_states])),
            fees=round(sum(f['fees'] for f in stage_fills),2),
            slippage=round(sum(f['slippage_cost'] for f in stage_fills),2))
    gross = [sum(s['weights'].values()) for s in snapshots]
    holdings = [sum(q>0 for q in s['quantities'].values()) for s in snapshots]
    first_fills = sorted(fills,key=lambda f:f['date'])
    years = (pd.Timestamp(end)-pd.Timestamp(start)).days/365.25
    costs = {}
    for reason in sorted({f['reason'] for f in fills}):
        rows = [f for f in fills if f['reason']==reason]
        costs[reason] = dict(fills=len(rows),fees=round(sum(f['fees'] for f in rows),2),
            slippage=round(sum(f['slippage_cost'] for f in rows),2),
            notional=round(sum(f['notional'] for f in rows),2))
    new_gaps_path=OUT/'extension/actions/action_gaps.json'
    new_gaps=json.loads(new_gaps_path.read_text()) if new_gaps_path.exists() else []
    gap_keys={(g['instrument_id'],g['date'],g['reason']) for g in new_gaps}
    gap_hits=[g for g in result.get('data_issues',[]) if (g.get('instrument_id'),str(pd.Timestamp(g['date']).date()),g.get('reason')) in gap_keys]
    ordinary=[t for t in result.get('trades',[]) if t['reason']=='legacy_repaired']
    latest_ordinary=max(ordinary,key=lambda t:t['signal_date']) if ordinary else None
    return dict(status=result['status'],start=start,end=end,signal_preparation_day=prep,
        initial_capital=initial,terminal_nav=float(nav.iloc[-1]),net_profit=float(nav.iloc[-1]-initial),
        total_net_return=float(nav.iloc[-1]/initial-1),
        net_cagr=float((nav.iloc[-1]/initial)**(1/years)-1),years=years,
        max_drawdown=float((nav/nav.cummax().clip(lower=initial)-1).min()),
        sessions=len(nav),stages=stages,mean_gross=float(np.mean(gross)),
        min_gross=float(min(gross)),max_gross=float(max(gross)),end_gross=float(gross[-1]),
        mean_holdings=float(np.mean(holdings)),max_holdings=max(holdings),end_holdings=holdings[-1],
        fees=round(sum(f['fees'] for f in fills),2),slippage=round(sum(f['slippage_cost'] for f in fills),2),
        costs=costs,first_fill_date=first_fills[0]['date'] if first_fills else None,
        first_fill_signal_date=first_fills[0]['signal_date'] if first_fills else None,
        initial_preparation_risk=result['risk_history'][0] if result.get('risk_history') else None,
        end_risk=result['risk_history'][-1] if result.get('risk_history') else None,
        incremental_action_gaps=dict(source_events=len(new_gaps),held_account_hits=len(gap_hits),hits=gap_hits),
        latest_ordinary_signal=latest_ordinary,
        data_issue_count=len(result.get('data_issues',[])),
        allocation_issue_count=len(result.get('allocation_issues',[])),
        data_issue_reasons=dict(Counter(i.get('reason','unknown') for i in result.get('data_issues',[]))),
        promotion_approved=False,
        cost_note='Fees and slippage are already in net NAV; do not subtract twice or add to net CAGR.',
        latest_local_prices='2026-09-22',remaining_account_period='2026-09-23 through 2026-09-27 is not evaluated'),daily


def verify_result(result,protocol):
    import pandas as pd
    actual=json.loads(json.dumps(result.get('config'),default=str))
    if actual!=protocol['effective_config']:raise ValueError('result_config_mismatch')
    required=dict(data_version=protocol['expected_data_version'],candidate='quality_pb',risk_policy='P2',allocation_policy='legacy',data_mode='free_research',industry_policy='enforced',strategy='legacy_repaired',promotion_approved=False)
    for key,value in required.items():
        if result.get(key)!=value:raise ValueError('result_identity_mismatch:'+key)
    if result.get('status') not in ('unverified_data_gaps','research_unvalidated'):raise ValueError('incomplete_result')
    snapshots,nav,_ = reporting_account(result)
    if shared.date_strings(nav.index)!=protocol['sessions']:
        raise ValueError('missing_account_sessions')
    errors=[]
    for s in result['account']:
        value=sum(q*s['marks'][c] for c,q in s['quantities'].items())
        residual=s['nav']-s['cash']-value-s.get('receivables',0)+s.get('payables',0)
        if abs(residual)>.02:errors.append('cash_share_nav_mismatch')
        if s['cash'] < -.001 or any(q<0 for q in s['quantities'].values()):errors.append('negative_cash_or_shares')
        if sum(q>0 for q in s['quantities'].values())>20:errors.append('holdings_over_20')
    for f in result.get('fills',[]):
        if pd.Timestamp(f['date'])<=pd.Timestamp(f['signal_date']):errors.append('same_day_or_early_fill')
    if not result.get('fills') or min(pd.Timestamp(f['date']) for f in result['fills'])!=pd.Timestamp(START):
        errors.append('first_session_entry_not_exercised')
    initial_risk=result['risk_history'][0]
    if initial_risk['level']!=0 or initial_risk['drawdown']!=0:errors.append('old_account_risk_inherited')
    review=shared.review_audit(result,protocol['ordinary_signal_dates'],PREP,END)
    if review['unexpected_ordinary_dates'] or review['unexplained_dates']:errors.append('monthly_review_audit_failed')
    if errors:raise ValueError(','.join(sorted(set(errors))))
    return dict(passed=True,reporting_sessions=len(snapshots),raw_sessions=len(result['account']),
        new_account=True,first_day_costs_included=True,cash_shares_nav_reconciled=True,
        first_fill_on_requested_session=True,review_audit=review,promotion_approved=False)


def write_report(summary,audit):
    s=summary
    lines=['# 2025年新账户：当前主策略核算', '',
        f"60万元从2025年第一个交易日重新建仓，截至{s['end']}，期末{s['terminal_nav']:,.2f}元，净利润{s['net_profit']:,.2f}元，累计净收益{s['total_net_return']:.4%}。",
        f"同窗口自然日年化{s['net_cagr']:.4%}，最大回撤幅度{-s['max_drawdown']:.4%}。这是条件性研究结果，未批准采用。", '',
        '| 阶段 | 期初资金 | 期末资金 | 净利润 | 净收益 |',
        '|---|---:|---:|---:|---:|']
    for name,row in s['stages'].items():
        lines.append(f"| {name}（至{row['last_session']}） | {row['initial_boundary_nav']:,.2f} | {row['terminal_nav']:,.2f} | {row['net_profit']:,.2f} | {row['net_return']:.4%} |")
    lines += ['',f"平均仓位{s['mean_gross']:.4%}，期末仓位{s['end_gross']:.4%}；平均持仓{s['mean_holdings']:.2f}只，实际最多{s['max_holdings']}只。",
        f"显式费用{s['fees']:,.2f}元，滑点{s['slippage']:,.2f}元，已进入上述净收益。不能再次扣除，亦不能加到净年化上当毛年化。",'',
        '账户在2024-12-31以60万元现金初始化，按原月末规则生成首次目标，2025-01-02开盘执行。原始账本保留这个无持仓、无费用、净值60万元的信号预备日；报告从1月2日起包含首日成本。滚动因子、专家IC与基准均保留完整冻结历史；未继承2020旧账户的股份、成本或回撤状态。',
        f"首笔信号{s['first_fill_signal_date']}，首笔成交{s['first_fill_date']}；现金/股数/净值、首日成本和月末决策日审计通过。",
        'P2控制器按冻结实现从normal、计数0创建，随后先评估起始日的前一天基准，再评估预备日；账户回撤初始化为0。具体预备日实际状态保留在summary.json。', '',
        '数据边界：今天是2026-09-27。本次补采7—9月换手率、ST/交易状态及现金/股数权益，保留原始历史并追加真实新增证券；共同核算截止2026-09-22，9月23日至今天未核算。data_inventory.json为补采前盘点，extension/extension_audit.json为最终输入核验。财务公告与历史修订仍存在覆盖限制，不把最新价格日期当作完整信息集认证。',
        f"运行状态{s['status']}；持有证券数据问题{s['data_issue_count']}条，配权问题{s['allocation_issue_count']}条。PIT、行业历史、权益完整性、开盘成交容量、成本压力与独立前向验证限制继续保留。", '',
        '策略保持quality_pb、P2、月度换仓、最多20只、单边30bp滑点及原费用/风控。18%参考采用另一个50万100只口径，应另列，不作为本账户成绩。', '',
        '产物：protocol.json锁定源码/参数/数据身份；run/result.pkl保留完整账本、成交、订单、风险及信号；run/summary.json与run/verification.json给出数值和核验。大型输入表引用冻结manifest，未重复复制。']
    shared.publish(OUT/'REVIEW.md',lambda f:f.write(('\n'.join(lines)+'\n').encode()))


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    if protocol!=protocol_identity():raise ValueError('protocol_identity_mismatch')
    folder=OUT/'run'
    shared.reserve_attempt(folder)
    status=folder/'status.json'
    def state(name,**extra):shared.freeze_json(status,dict(status=name,time=now(),promotion_approved=False,**extra),immutable=False)
    try:
        state('loading_frozen_dataset')
        shared.load_frozen()
        from quantlab.data.contracts import load_dataset
        ds=load_dataset(MANIFEST,profile='free_research')
        if ds.version!=data_version():raise ValueError('dataset_version_mismatch')
        state('preparing_and_replaying_single_account')
        result=replay_account(ds)
        # Save completed account before post-processing. Original input hashes bind
        # excluded immutable data; no account evidence is removed from this copy.
        excluded={'input_panels','input_instruments','input_actions','benchmark_manifest','_runtime'}
        stored={k:v for k,v in result.items() if k not in excluded}
        stored['input_reference']=dict(manifest=str(MANIFEST),sha256=shared.sha256(MANIFEST),
            data_version=ds.version,excluded_large_fields=sorted(excluded))
        shared.publish(folder/'result.pkl',lambda f:pickle.dump(stored,f,protocol=pickle.HIGHEST_PROTOCOL))
        shared.freeze_json(folder/'artifacts.json',dict(result_sha256=shared.sha256(folder/'result.pkl'),
            protocol_sha256=shared.sha256(OUT/'protocol.json')))
        state('account_saved_postprocessing')
        audit=verify_result(stored,protocol)
        summary,daily=summarize(stored)
        shared.freeze_json(folder/'verification.json',audit)
        shared.freeze_json(folder/'summary.json',summary)
        shared.publish(folder/'daily_returns.csv',lambda f:f.write(daily.to_csv().encode()))
        write_report(summary,audit)
        state('complete',engine_status=stored['status'],net_profit=summary['net_profit'])
        return summary
    except BaseException as exc:
        error=dict(status='technical_failure',error=str(exc),traceback=traceback.format_exc(),
            time=now(),recovery='Preserve raw result if present; fix postprocessing without account rerun.',promotion_approved=False)
        shared.freeze_json(folder/'error.json',error)
        state('technical_failure',error=str(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['inventory','preflight','run'])
    args=parser.parse_args()
    value={'inventory':inventory,'preflight':preflight,'run':run}[args.command]()
    print(json.dumps(value,ensure_ascii=False,default=str,indent=2))
