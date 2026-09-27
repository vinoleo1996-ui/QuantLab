"""Read archived accounts; reconcile economic P&L, costs and target paths.

No account engine, model, strategy configuration, price loader or downloader is
called. Realized disposal P&L is derived from archived remaining FIFO cost
basis; it is never a sum of daily marked P&L renamed as realized profit.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd

PERIODS = ('2018_2021', '2022_2025', '2025_recent')
SUM_COLUMNS = ['pnl', 'realized_disposal_pnl', 'dividend_accrual', 'unrealized_change',
               'fees', 'slippage_cost', 'buy_notional', 'sell_notional', 'released_cost_basis']
LIMITS = [
    'Conditional archived simulation with unresolved PIT, historical financial revisions, industry and corporate-action gaps; not deployable performance.',
    '100-stock accounts exceed the user 50-stock cap: external return comparators only, never an implementable alternative or mathematical return ceiling.',
    'Industry is the most recent archived decision-input classification carried forward for attribution; it is a source proxy, not a certified historical sector series.',
    'Realized disposal P&L uses changes in archived remaining FIFO basis, including buy fees and bonus adjustments. This audit reconciles snapshots, not a new independent lot engine.',
    'Dividends are income on recorded ex-date, including unpaid receivables. Payment transfers receivable to cash and is not income again. Future dividend tax on unsold shares is not accrued by the source account.',
    'Slippage is embedded in archived traded prices and disclosed separately. Fees are already charged, with buy fees capitalized in basis; neither is subtracted twice or added to CAGR.',
    'Role is the latest strictly prior signal target role. Retained means continued from the previous target, not proof of an executed position or of buffer-protected retention. Role switches transfer existing unrealized P&L explicitly.',
    'Event-time realized P&L may contain price gains earned before its current target role. These are accounting partitions, not causal retained-versus-new selection effects.',
    'A remaining position is marked at the final close and is not assumed sold. End-censored target admissions and unfilled exits remain separate.',
    'Removing top stocks, industries or days is additive concentration description, not a tradable deleted-stock counterfactual.',
    'Periods overlap and all history was viewed. Simulated opening quotes do not calibrate real 30bp slippage; no capacity hit does not pass execution stress.',
]


def day(value):
    return str(pd.Timestamp(value).date())


def canonical(value):
    parts = value.split('.')
    return parts[1].lower()+'.'+parts[0] if len(parts) == 2 and parts[0].isdigit() else value.lower()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite_'+label)
    return result


def attribute_account(run, action_codes):
    """Return daily security accounting and mutually exclusive actual fill types.

    B=remaining total cost, V=marked stock value, P=buy notional+buy fees.
    Released cost=B[t-1]+P-B[t], U=V-B, realized=sales-sell fees-released.
    Daily economic P&L=realized+ex-date income+change(U).
    """
    states = run['account']; dates = [day(s['date']) for s in states]
    if dates != sorted(set(dates)):
        raise ValueError('account_calendar')
    fills_by_day = defaultdict(list); actions_by_day = defaultdict(list)
    for fill in run['fills']:
        if day(fill['date']) not in dates:
            raise ValueError('fill_outside_calendar')
        fills_by_day[day(fill['date'])].append(fill)
    for event in run.get('corporate_action_events', []):
        if event['type'] == 'corporate_action' and (event.get('cash', 0) or event.get('bonus_shares', 0)):
            if event['action_id'] not in action_codes:
                raise ValueError('missing_action_mapping:'+event['action_id'])
            if day(event['date']) not in dates:
                raise ValueError('action_outside_calendar')
            actions_by_day[day(event['date'])].append(event)
    previous_q = {}; previous_v = {}; previous_b = {}
    previous_nav = finite(run['config']['capital']['init_cash'], 'initial_cash')
    ever_owned = set(); rows = []; trade_rows = []
    maximum = Counter(); checks = Counter()
    returns = run.get('daily_return')
    for state in states:
        d = day(state['date']); q = state['quantities']; marks = state['marks']
        if 'cost_basis' not in state or set(q)-set(state['cost_basis']):
            raise ValueError('missing_cost_basis:'+d)
        if set(q)-set(marks):
            raise ValueError('missing_marks:'+d)
        values = {c:finite(n, 'quantity')*finite(marks[c], 'mark') for c,n in q.items()}
        basis = {c:finite(n, 'quantity')*finite(state['cost_basis'][c], 'cost_basis') for c,n in q.items()}
        daily = defaultdict(Counter); rolling_q = dict(previous_q)
        for event in actions_by_day[d]:
            c = action_codes[event['action_id']]
            daily[c]['dividend_accrual'] += finite(event.get('cash',0.), 'dividend')
            rolling_q[c] = rolling_q.get(c,0)+event.get('bonus_shares',0)
        for fill in fills_by_day[d]:
            c = fill['instrument_id']; side = fill['side']; n = int(fill['quantity'])
            if side not in ('buy','sell') or n <= 0 or n != fill['quantity']:
                raise ValueError('invalid_fill')
            if day(fill['signal_date']) >= d:
                raise ValueError('signal_not_before_fill')
            before = rolling_q.get(c,0)
            if side == 'buy':
                category = 'same_name_increase' if before else ('reentry' if c in ever_owned else 'new_entry')
                rolling_q[c] = before+n; ever_owned.add(c)
            else:
                if n>before:raise ValueError('oversold_quantity')
                rolling_q[c] = before-n
                category = 'full_exit' if rolling_q[c] == 0 else 'same_name_reduction'
            notional = finite(fill['notional'], 'notional'); fees = finite(fill['fees'], 'fees')
            daily[c][side+'_notional'] += notional; daily[c][side+'_fees'] += fees
            daily[c]['fees'] += fees; daily[c]['slippage_cost'] += finite(fill.get('slippage_cost',0.), 'slippage')
            daily[c][side+'_quantity'] += n
            trade_rows.append(dict(date=d,instrument_id=c,order_id=fill['order_id'],signal_date=day(fill['signal_date']),
                side=side,quantity=n,notional=notional,fees=fees,slippage_cost=float(fill.get('slippage_cost',0.)),
                commission=float(fill.get('commission',0.)),stamp_tax=float(fill.get('stamp_tax',0.)),
                dividend_tax=float(fill.get('dividend_tax',0.)),category=category,
                before_quantity=before,after_quantity=rolling_q[c]))
        if {c:n for c,n in rolling_q.items() if n} != q:
            raise ValueError('quantity_conservation:'+d)
        checks['quantity_states'] += 1
        pnl_total = 0.
        for c in sorted(set(q)|set(previous_q)|set(daily)):
            x = daily[c]
            released = previous_b.get(c,0.) + x['buy_notional'] + x['buy_fees'] - basis.get(c,0.)
            if not x['sell_quantity'] and abs(released)>1e-6:
                raise ValueError('basis_without_sale:'+d+':'+c)
            if released < -1e-6 or released > previous_b.get(c,0.)+x['buy_notional']+x['buy_fees']+1e-6:
                raise ValueError('invalid_released_cost_basis:'+d+':'+c)
            unrealized_before = previous_v.get(c,0.)-previous_b.get(c,0.)
            unrealized = values.get(c,0.)-basis.get(c,0.)
            realized = x['sell_notional']-x['sell_fees']-released
            pnl = values.get(c,0.)-previous_v.get(c,0.)+x['sell_notional']-x['buy_notional']-x['fees']+x['dividend_accrual']
            residual = realized+x['dividend_accrual']+unrealized-unrealized_before-pnl
            if abs(residual)>1e-6:raise ValueError('component_reconciliation')
            maximum['max_component_error'] = max(maximum['max_component_error'],abs(residual))
            rows.append(dict(date=d,instrument_id=c,pnl=pnl,realized_disposal_pnl=realized,
                dividend_accrual=x['dividend_accrual'],unrealized_change=unrealized-unrealized_before,
                unrealized_pnl=unrealized,unrealized_before=unrealized_before,released_cost_basis=released,
                ending_cost_basis=basis.get(c,0.),market_value=values.get(c,0.),quantity=q.get(c,0),
                fees=x['fees'],slippage_cost=x['slippage_cost'],buy_notional=x['buy_notional'],sell_notional=x['sell_notional']))
            pnl_total += pnl
        nav = finite(state['nav'],'nav'); error = pnl_total-(nav-previous_nav)
        if abs(error)>1e-6:raise ValueError('daily_pnl_reconciliation:'+d+':'+str(error))
        if abs(float(state['cash'])+float(state.get('receivables',0.))+sum(values.values())-nav)>1e-6:
            raise ValueError('nav_components:'+d)
        if returns is not None:
            if pd.Timestamp(d) not in returns.index:raise ValueError('return_calendar')
            err = pnl_total/previous_nav-float(returns.loc[pd.Timestamp(d)])
            if abs(err)>1e-12:raise ValueError('daily_return_reconciliation:'+d)
            maximum['max_daily_return_error'] = max(maximum['max_daily_return_error'],abs(err))
        maximum['max_daily_pnl_error'] = max(maximum['max_daily_pnl_error'],abs(error))
        previous_q,previous_v,previous_b,previous_nav = dict(q),values,basis,nav
    frame = pd.DataFrame(rows, columns=['date','instrument_id','pnl','realized_disposal_pnl','dividend_accrual','unrealized_change','unrealized_pnl','unrealized_before','released_cost_basis','ending_cost_basis','market_value','quantity','fees','slippage_cost','buy_notional','sell_notional'])
    trades = pd.DataFrame(trade_rows)
    check = dict(maximum,**checks,stock_days=len(rows),fills=len(trades),realized_basis='archived_FIFO_remaining_basis_including_buy_fees',
                 all_daily_reconciled=True)
    return frame,trades,check


def account_metrics(states, initial, period):
    """Match common-account annualization, adding full drawdown episodes."""
    nav = pd.Series([float(s['nav']) for s in states],index=pd.to_datetime([s['date'] for s in states]))
    if nav.index.has_duplicates or not nav.index.is_monotonic_increasing or (nav<=0).any():
        raise ValueError('invalid_nav_series')
    returns = nav.pct_change(fill_method=None); returns.iloc[0] = nav.iloc[0]/initial-1
    if period == '2025_recent':
        if day(nav.index[0])!='2024-12-31' or states[0]['quantities'] or nav.iloc[0]!=initial:
            raise ValueError('preparation_day_not_cash')
        returns = returns.loc['2025-01-02':]
    elapsed = (returns.index[-1]-returns.index[0]).days
    annual = {}; previous = initial
    for year,g in nav.loc[returns.index].groupby(returns.index.year):
        annual[str(year)] = dict(net_profit=float(g.iloc[-1]-previous),net_return=float(g.iloc[-1]/previous-1))
        previous = float(g.iloc[-1])
    episodes = []; high = float(initial); peak_day = day(nav.index[0]); peak_i = -1; active = None
    for i,(d,v) in enumerate(nav.items()):
        if v>=high-1e-9:
            if active:
                active.update(recovery_date=day(d),recovered=True,peak_to_recovery_sessions=i-peak_i,
                              trough_to_recovery_sessions=i-active.pop('trough_index'))
                episodes.append(active); active=None
            high=float(v); peak_day=day(d); peak_i=i
        else:
            if active is None:
                active=dict(peak_date=peak_day,first_underwater_date=day(d),trough_date=day(d),
                    trough_index=i,depth=float(v/high-1),underwater_sessions=0,recovery_date=None,recovered=False)
            active['underwater_sessions'] += 1
            if v/high-1<active['depth']:
                active.update(depth=float(v/high-1),trough_date=day(d),trough_index=i)
    if active:
        active.update(end_censored=True,observed_peak_to_end_sessions=len(nav)-1-peak_i)
        active.pop('trough_index');episodes.append(active)
    included = [s for s in states if pd.Timestamp(s['date']) in returns.index]
    gross = [sum(s['weights'].values()) for s in included]
    hhi = [sum((w/sum(s['weights'].values()))**2 for w in s['weights'].values()) if s['weights'] else 0. for s in included]
    return dict(initial_cash=initial,final_equity=float(nav.iloc[-1]),net_profit=float(nav.iloc[-1]-initial),
        cagr=float((nav.iloc[-1]/initial)**(365.25/elapsed)-1) if elapsed>0 else None,
        total_return=float(nav.iloc[-1]/initial-1),max_drawdown=float((nav/nav.cummax().clip(lower=initial)-1).min()),
        start_date=day(returns.index[0]),end_date=day(returns.index[-1]),sessions=len(returns),raw_sessions=len(nav),
        annual=annual,mean_gross=float(np.mean(gross)),min_gross=float(min(gross)),max_gross=float(max(gross)),
        final_gross=float(gross[-1]),all_cash_sessions=sum(x==0 for x in gross),
        mean_holdings=float(np.mean([len(s['quantities']) for s in included])),max_holdings=max(len(s['quantities']) for s in included),
        max_stock_weight=max([max(s['weights'].values(),default=0.) for s in included]),mean_invested_hhi=float(np.mean(hhi)),
        max_invested_hhi=float(max(hhi)),recovery_episodes=episodes,
        max_underwater_sessions=max([e['underwater_sessions'] for e in episodes],default=0),
        unrecovered_at_end=bool(active),final_receivables=float(states[-1].get('receivables',0.)),
        first_invested=next((day(s['date']) for s in included if s['quantities']),None))


def target_paths(targets, inputs):
    info_dates = sorted(inputs['heads']); previous = set(); decisions = []
    for target in targets:
        signal = day(target['signal_date']); idx = bisect_right(info_dates,signal)-1
        if idx<0:raise ValueError('missing_prior_information')
        info = info_dates[idx]; heads = inputs['heads'][info]
        ranked = sorted(heads,key=lambda c:(-heads[c],c))
        ranks = {canonical(c):i+1 for i,c in enumerate(ranked)}
        selected = {canonical(c) for c in target['weights']}
        if selected-set(ranks):raise ValueError('target_outside_original_heads')
        groups = {canonical(c):g for c,g in inputs['industries'][info].items()}
        roles = {c:('retained' if c in previous else 'new_admission') for c in selected}
        roles.update({c:'pending_removed' for c in previous-selected})
        decisions.append(dict(signal_date=signal,information_date=info,selected=sorted(selected),
            previous_target=sorted(previous),roles=roles,rank=ranks,industries=groups,
            protected_retained=sorted(c for c in selected&previous if ranks[c]<=150),
            continued_outside150=sorted(c for c in selected&previous if ranks[c]>150),
            target_count=len(selected),target_exposure=sum(target['weights'].values()),
            max_target_industry_names=max(Counter(groups.get(c,'unclassified') for c in selected).values(),default=0)))
        previous = selected
    admissions = []
    for i,t in enumerate(decisions):
        later = decisions[i+1:]; nxt = later[0] if later else None
        for c in sorted(set(t['selected'])-set(t['previous_target'])):
            first_exit = next((d for d in later if c not in d['selected']),None)
            admissions.append(dict(signal_date=t['signal_date'],information_date=t['information_date'],instrument_id=c,
                industry=t['industries'].get(c,'unclassified'),original_rank=t['rank'][c],outside150=t['rank'][c]>150,
                next_signal_date=nxt['signal_date'] if nxt else None,end_censored=nxt is None,
                exited_next_target=c not in nxt['selected'] if nxt else None,
                next_eligible=c in nxt['rank'] if nxt else None,next_top150=(nxt['rank'].get(c,math.inf)<=150) if nxt else None,
                next_original_rank=nxt['rank'].get(c) if nxt else None,
                first_later_target_exit=first_exit['signal_date'] if first_exit else None,
                later_target_exit_eligible=c in first_exit['rank'] if first_exit else None,
                later_target_exit_top150=first_exit['rank'].get(c,math.inf)<=150 if first_exit else None,
                no_observed_later_exit=first_exit is None))
    return decisions,admissions


def attach_roles(rows, states, decisions):
    """Tag daily contributions; record transfers to reconcile ending role stocks."""
    signal_dates=[d['signal_date'] for d in decisions]; previous_roles={}; previous_u={}; transfers=[]; records=[]
    grouped={d:g.to_dict('records') for d,g in rows.groupby('date',sort=False)}
    latest_industry={}; latest_roles={}; current=None
    for state in states:
        d=day(state['date']); ix=bisect_left(signal_dates,d)-1
        if ix>=0 and ix!=current:
            # Preserve roles of older removed but still held securities.
            for dec in decisions[(current+1 if current is not None else 0):ix+1]:
                latest_roles.update(dec['roles']);latest_industry.update(dec['industries'])
            current=ix
        for row in grouped.get(d,[]):
            c=row['instrument_id'];role=latest_roles.get(c,'other_actual_holding')
            old=previous_roles.get(c,role);u=previous_u.get(c,0.)
            if old!=role and abs(u)>1e-12:
                transfers.extend([dict(date=d,instrument_id=c,role=old,unrealized_transfer=-u),
                                  dict(date=d,instrument_id=c,role=role,unrealized_transfer=u)])
            records.append(dict(row,role=role,industry=latest_industry.get(c,'unclassified'),exchange=c.split('.')[0]))
            previous_roles[c]=role;previous_u[c]=row['unrealized_pnl']
    final=states[-1];ending=[]
    for c,n in final['quantities'].items():
        ending.append(dict(instrument_id=c,role=previous_roles[c],industry=latest_industry.get(c,'unclassified'),
            exchange=c.split('.')[0],quantity=n,market_value=n*final['marks'][c],
            cost_basis=n*final['cost_basis'][c],unrealized_pnl=n*(final['marks'][c]-final['cost_basis'][c])))
    return pd.DataFrame(records),pd.DataFrame(ending),pd.DataFrame(transfers,columns=['date','instrument_id','role','unrealized_transfer'])


def account_specs(root):
    specs=[]
    for p in PERIODS:
        for n in (50,100):
            specs.append(dict(period=p,arm=f'reference{n}',folder=root/f'output/reference_challenge_20260927/run2/{p}/common{n}_cost1',
                target=root/f'output/reference_challenge_20260927/run2/{p}/targets_{n}.json'))
        if p!='2018_2021':
            specs.append(dict(period=p,arm='R',folder=root/f'output/expert_selection_20260927/study/rs/{p}/R',
                target=root/f'output/expert_selection_20260927/study/rs/{p}/R_targets.json'))
    for s in specs:
        s['inputs']=root/('output/reference_recent_20260927/inputs.json' if s['period']=='2025_recent' else f"output/reference_sustainability_20260927/{s['period']}/decision_inputs.json")
    return specs


def verify_hashes(mapping):
    for p,h in mapping.items():
        if sha(p)!=h:raise ValueError('source_identity_changed:'+p)


def prepare(root, output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise ValueError('protocol_already_frozen')
    source=Path(__file__).resolve();test=source.parents[1]/'tests/test_reference_selection_accounts.py'
    manifest=root/'output/account_2025_20260927/extension/dataset/manifest.json'
    m=json.loads(manifest.read_text())
    files={source,test,manifest,root/'output/reference_selection_20260927/PLAN.md',
        root/'output/frozen_source/6157c64/quantlab/backtest/ledger.py',
        root/'output/reference1882/frozen_source/a_share_quant/stock_only/portfolio_primitives.py',
        root/'output/reference_challenge_20260927/run2/protocol.json',
        root/'output/expert_selection_20260927/study/rs/protocol.json',
        root/'output/reference_sustainability_20260927/comparison.json',
        root/'output/reference_sustainability_20260927/protocol.json',
        root/'output/expert_selection_20260927/stress/all/summary.json'}
    for key in ('actions','calendar'):
        p=manifest.parent/m[key]['path']
        if sha(p)!=m[key]['sha256']:raise ValueError('market_table_identity')
        files.add(p)
    for s in account_specs(root):
        files.update((s['target'],s['inputs'],s['folder']/'complete.json'))
        complete=json.loads((s['folder']/'complete.json').read_text())
        for name,h in complete.items():
            p=s['folder']/name
            if sha(p)!=h:raise ValueError('archive_completion_hash:'+str(p))
            files.add(p)
    for period in PERIODS:
        folder=root/f'output/reference_challenge_20260927/run2/{period}/common50_cost2'
        files.add(folder/'complete.json')
        for name,h in json.loads((folder/'complete.json').read_text()).items():
            p=folder/name
            if sha(p)!=h:raise ValueError('cost2_archive_hash')
            files.add(p)
    snapshots=output/'source';snapshots.mkdir(exist_ok=True)
    shutil.copyfile(source,snapshots/source.name);shutil.copyfile(test,snapshots/test.name)
    inputs={str(p):sha(p) for p in sorted(files)}
    write_json(output/'protocol.json',dict(version='reference-selection-accounts-v1',created_at=datetime.now(timezone.utc).isoformat(),
        task='Task2 only archived accounts; no replay or training',root=str(root),sources=inputs,
        frozen_analysis_sources={str(p):sha(p) for p in sorted(snapshots.iterdir())},
        question='Where did archived reference50/reference100/common-updated R profits, costs and target-retention paths arise?',
        decision='Describe concentration and mechanism dependencies without approving a strategy or adding an experiment.',
        account_count=8,source_ledger='6157c64',initial_cash=600000,analysis_only=True,promotion_approved=False,
        source_basis='pre-analysis snapshots and file hashes',limitations=LIMITS,
        reference100_role='external_return_comparator_exceeds_50_stock_cap_not_implementable_not_return_upper_bound'))


def aggregate(frame, keys):
    return frame.groupby(keys,dropna=False)[SUM_COLUMNS].sum().reset_index()


def target_actual_diagnostics(run, decisions, admissions):
    state_by_day={day(s['date']):s for s in run['account']}; dates=sorted(state_by_day)
    output=[]
    trace_by_day={x['date']:x for x in run.get('signal_trace',[])}
    for d in decisions:
        signal=d['signal_date'];state=state_by_day[signal];held=set(state['quantities']);selected=set(d['selected'])
        if trace_by_day:
            trace=trace_by_day[signal]
            if {canonical(c) for c in trace['target_weights']}!=selected:raise ValueError('target_trace_mismatch')
            if set(trace['actual_weights'])!=held:raise ValueError('signal_actual_holdings_mismatch')
        output.append(dict(signal_date=signal,information_date=d['information_date'],target_count=len(selected),
            actual_count=len(held),target_not_held=sorted(selected-held),held_not_target=sorted(held-selected),
            retained_target_count=len(selected&set(d['previous_target'])),new_target_count=len(selected-set(d['previous_target'])),
            buffer_protected_retained_count=len(d['protected_retained']),continued_outside150_count=len(d['continued_outside150']),
            max_target_industry_names=d['max_target_industry_names'],target_exposure=d['target_exposure'],
            timing='signal-close actual quantities before fills at later openings'))
    for a in admissions:
        c=a['instrument_id'];start=a['signal_date'];cut=a['next_signal_date']
        observed=[d for d in dates if d>start and (cut is None or d<=cut)]
        a['first_held_after_signal']=next((d for d in observed if state_by_day[d]['quantities'].get(c,0)>0),None)
        a['held_at_signal']=state_by_day[start]['quantities'].get(c,0)>0
        a['ever_held_before_next_target']=a['first_held_after_signal'] is not None
        exit_day=a['first_later_target_exit']
        if exit_day:
            a['quantity_at_target_exit']=state_by_day[exit_day]['quantities'].get(c,0)
            a['first_flat_after_target_exit']=next((d for d in dates if d>exit_day and state_by_day[d]['quantities'].get(c,0)==0),None)
            a['actual_exit_end_censored']=a['quantity_at_target_exit']>0 and a['first_flat_after_target_exit'] is None
        else:
            a.update(quantity_at_target_exit=None,first_flat_after_target_exit=None,actual_exit_end_censored=None)
    return output


def summarize_paths(admissions):
    def stats(rows):
        observed=[r for r in rows if not r['end_censored']]
        return dict(admissions=len(rows),next_target_observed=len(observed),next_target_end_censored=len(rows)-len(observed),
            exited_next_target=sum(r['exited_next_target'] for r in observed),
            exited_next_while_eligible=sum(r['exited_next_target'] and r['next_eligible'] for r in observed),
            exited_next_while_top150=sum(r['exited_next_target'] and r['next_top150'] for r in observed),
            never_held_before_next_target=sum(not r['ever_held_before_next_target'] and not r['held_at_signal'] for r in rows),
            no_observed_later_exit=sum(r['no_observed_later_exit'] for r in rows),
            actual_exit_end_censored=sum(bool(r['actual_exit_end_censored']) for r in rows))
    return dict(all_admissions=stats(admissions),outside150=stats([r for r in admissions if r['outside150']]),
        denominator='Target admissions, not trades; first target launch counted. Final next-target absence is end-censored, never an observed retention or successful exit.')


def analyze(root, output, run_name):
    protocol=json.loads((output/'protocol.json').read_text())
    verify_hashes(protocol['sources']);verify_hashes(protocol['frozen_analysis_sources'])
    dest=output/run_name;dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'started.json',dict(at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(output/'protocol.json')))
    try:
        manifest=root/'output/account_2025_20260927/extension/dataset/manifest.json';m=json.loads(manifest.read_text())
        actions=pd.read_parquet(manifest.parent/m['actions']['path'])
        if actions.groupby('action_id').instrument_id.nunique().gt(1).any():raise ValueError('ambiguous_action_mapping')
        action_codes=actions.drop_duplicates('action_id').set_index('action_id').instrument_id.to_dict()
        summary=dict(status='conditional_archived_account_attribution',accounts={},comparisons={},source_protocol_sha256=sha(output/'protocol.json'),
            limitations=LIMITS,promotion_approved=False,new_accounts_run=0,new_models=0)
        stocks={}
        for spec in account_specs(root):
            label=spec['period']+'/'+spec['arm'];folder=dest/label;folder.mkdir(parents=True)
            with (spec['folder']/'result.pkl').open('rb') as handle:run=pickle.load(handle)
            payload=json.loads(spec['target'].read_text());inputs=json.loads(spec['inputs'].read_text())
            decisions,admissions=target_paths(payload['targets'],inputs)
            rows,trades,checks=attribute_account(run,action_codes)
            rows,ending,transfers=attach_roles(rows,run['account'],decisions)
            target_actual=target_actual_diagnostics(run,decisions,admissions)
            metrics=account_metrics(run['account'],float(run['config']['capital']['init_cash']),spec['period'])
            stored=json.loads((spec['folder']/'summary.json').read_text())
            for key in ('net_profit','cagr','max_drawdown','mean_gross','max_stock_weight','max_holdings','sessions'):
                if abs(metrics[key]-stored[key])>1e-7:raise ValueError('stored_metric_mismatch:'+label+':'+key)
            if metrics['initial_cash']!=600000:raise ValueError('unexpected_capital')
            if run.get('promotion_approved') or run.get('executable') or run.get('account_type')!='simulation':raise ValueError('unexpected_live_account')
            nav=pd.Series({pd.Timestamp(s['date']):s['nav'] for s in run['account']})
            performance_nav=nav.loc[metrics['start_date']:]
            metrics.update(fees=float(trades.fees.sum()),slippage=float(trades.slippage_cost.sum()),
                commission=float(trades.commission.sum()),stamp_tax=float(trades.stamp_tax.sum()),dividend_tax=float(trades.dividend_tax.sum()),
                total_traded_notional=float(trades.notional.sum()),annualized_two_way_turnover=float(trades.notional.sum()/performance_nav.mean()*252/metrics['sessions']),
                fills=len(trades),orders=len(run['orders']),data_issue_count=len(run.get('data_issues',[])),
                simulation_rejections=dict(Counter(x.get('reason','unknown') for x in run.get('corporate_action_events',[]) if x['type']=='execution_rejected')),
                parent_data_version=run.get('parent_data_version'),data_version=run.get('data_version'),
                role='external_return_comparator_exceeds_50_stock_cap' if spec['arm']=='reference100' else 'conditional_50_stock_research_account',
                implementable_alternative=False,mathematical_return_upper_bound=False,
                source_result=str(spec['folder']/'result.pkl'),source_result_sha256=sha(spec['folder']/'result.pkl'))
            if abs(metrics['fees']-stored['fees'])>1e-6 or abs(metrics['slippage']-stored['slippage'])>1e-6:raise ValueError('stored_cost_mismatch')
            stock=aggregate(rows,['instrument_id']);industry=aggregate(rows,['industry']);exchange=aggregate(rows,['exchange']);daily=aggregate(rows,['date']);role=aggregate(rows,['role'])
            role_end=ending.groupby('role').unrealized_pnl.sum() if len(ending) else pd.Series(dtype=float)
            role_transfer=transfers.groupby('role').unrealized_transfer.sum() if len(transfers) else pd.Series(dtype=float)
            role['ending_unrealized_pnl']=role.role.map(role_end).fillna(0.)
            role['unrealized_transfer']=role.role.map(role_transfer).fillna(0.)
            role['role_reconciliation_error']=role.realized_disposal_pnl+role.dividend_accrual+role.ending_unrealized_pnl-role.unrealized_transfer-role.pnl
            if role.role_reconciliation_error.abs().max()>1e-6:raise ValueError('role_transfer_reconciliation')
            group_errors={name:float(frame.pnl.sum()-metrics['net_profit']) for name,frame in [('stock',stock),('industry',industry),('exchange',exchange),('day',daily),('role',role)]}
            if max(map(abs,group_errors.values()))>1e-6:raise ValueError('aggregate_reconciliation')
            checks.update(aggregate_errors=group_errors,max_role_error=float(role.role_reconciliation_error.abs().max()),
                ending_unrealized_error=float(rows.unrealized_change.sum()-ending.unrealized_pnl.sum()))
            if abs(checks['ending_unrealized_error'])>1e-6:raise ValueError('unrealized_telescope')
            trade_summary=trades.groupby('category')[['notional','fees','slippage_cost']].sum()
            trade_summary['fills']=trades.groupby('category').size()
            trade_summary=trade_summary.reset_index()
            trade_roles={d['signal_date']:d for d in decisions}
            trades['target_role']=trades.apply(lambda r:trade_roles[r.signal_date]['roles'].get(r.instrument_id,'other_actual_holding'),axis=1)
            trades['industry']=trades.apply(lambda r:trade_roles[r.signal_date]['industries'].get(r.instrument_id,'unclassified'),axis=1)
            exposures=[]
            enriched_by_day={d:g for d,g in rows.groupby('date')}
            for s in run['account']:
                d=day(s['date']);r=enriched_by_day.get(d)
                group={} if r is None else (r.groupby('industry').market_value.sum()/s['nav']).to_dict()
                exposures.append(dict(date=d,gross=sum(s['weights'].values()),holdings=len(s['quantities']),
                    max_industry_weight=max(group.values(),default=0.),industry_weights=group))
            metrics['max_actual_industry_weight']=max(x['max_industry_weight'] for x in exposures)
            concentration=dict(best5_stock_profit=float(stock.pnl.nlargest(5).sum()),best5_day_profit=float(daily.pnl.nlargest(5).sum()),
                best_industry_profit=float(industry.pnl.max()),largest_industry=industry.loc[industry.pnl.idxmax(),'industry'],
                top5_stocks=stock.nlargest(5,'pnl')[['instrument_id','pnl']].to_dict('records'),
                top5_days=daily.nlargest(5,'pnl')[['date','pnl']].to_dict('records'))
            concentration.update(best5_stock_share_of_net_profit=concentration['best5_stock_profit']/metrics['net_profit'],
                profit_excluding_best5_stocks_descriptive=metrics['net_profit']-concentration['best5_stock_profit'],
                profit_excluding_best5_days_descriptive=metrics['net_profit']-concentration['best5_day_profit'])
            totals={k:float(rows[k].sum()) for k in SUM_COLUMNS}
            totals['ending_unrealized_pnl']=float(ending.unrealized_pnl.sum())
            result=dict(metrics=metrics,pnl_totals=totals,concentration=concentration,paths=summarize_paths(admissions),checks=checks,
                trade_categories=trade_summary.to_dict('records'),roles=role.to_dict('records'))
            for name,frame in dict(stock_daily=rows,trades=trades,per_stock=stock,per_industry=industry,per_exchange=exchange,
                per_day=daily,per_role=role,ending_positions=ending,role_transfers=transfers,trade_categories=trade_summary,
                admissions=pd.DataFrame(admissions)).items():
                frame.to_parquet(folder/(name+'.parquet'),index=False)
            write_json(folder/'target_actual.json',target_actual);write_json(folder/'exposures.json',exposures)
            write_json(folder/'summary.json',result);summary['accounts'][label]=result;stocks[label]=stock.set_index('instrument_id').pnl
        for period in PERIODS:
            pairs=[('reference50','reference100')]
            if period!='2018_2021':pairs += [('R','reference50'),('R','reference100')]
            for a,b in pairs:
                diff=stocks[period+'/'+a].subtract(stocks[period+'/'+b],fill_value=0.)
                expected=summary['accounts'][period+'/'+a]['metrics']['net_profit']-summary['accounts'][period+'/'+b]['metrics']['net_profit']
                if abs(diff.sum()-expected)>1e-6:raise ValueError('paired_stock_reconciliation')
                summary['comparisons'][period+'/'+a+'_minus_'+b]=dict(net_profit_difference=expected,stock_sum_error=float(diff.sum()-expected),
                    top5_positive_difference=float(diff.nlargest(5).sum()),top5_positive_names=diff.nlargest(5).to_dict(),
                    qualification='Observed account P&L difference partition, not deleted-stock or pure selection counterfactual.')
                diff.rename('net_profit_difference').to_frame().to_parquet(dest/(period+'_'+a+'_minus_'+b+'.parquet'))
        old=json.loads((root/'output/reference_sustainability_20260927/comparison.json').read_text())
        sensitivity=dict(source='output/reference_sustainability_20260927/comparison.json',role='single_fixed_hash_vs_code_order_sensitivity_reuse',
            original_conditions='100 stocks, 500000 CNY, 5bp slippage; different from common 30bp reference50 and R; not a current pair',
            periods={p:{arm:{k:old['periods'][p][arm][k] for k in ('net_profit','cagr','max_drawdown','mean_gross','fees','slippage')}
                        for arm in ('original','neutral')} for p in ('2018_2021','2022_2025')},
            conclusion=old['conclusion'],limitations=old['limitations'])
        cost2={}
        for p in PERIODS:
            x=json.loads((root/f'output/reference_challenge_20260927/run2/{p}/common50_cost2/summary.json').read_text())
            base=summary['accounts'][p+'/reference50']['metrics']
            cost2[p]=dict(cost1={k:base[k] for k in ('net_profit','cagr','max_drawdown','fees','slippage')},
                cost2={k:x[k] for k in ('net_profit','cagr','max_drawdown','fees','slippage')},
                change_net_profit=x['net_profit']-base['net_profit'],source=f'output/reference_challenge_20260927/run2/{p}/common50_cost2/summary.json')
        summary['reused_sensitivity']=sensitivity
        summary['reused_cost2']=dict(reference50=cost2,R='not_available_no_R_cost2_account',reference100='not_available_in_common_challenge',
            A1='Different candidate, not R; existing A1 stress is only context and does not certify R.',
            cost_semantics='Trading commission/stamp/slippage doubled in archived replay; dividend tax follows original tax law. Replayed cash/share path changes, not a cost subtraction shortcut.')
        summary['execution_status']=dict(actual_30bp_calibration='unverified_simulated_quotes_cannot_calibrate',
            reference50_capacity_stress='unverified_no_registered_hit_test',R_capacity_stress='unverified_no_registered_hit_test',
            A1_adv03='Existing different-candidate test did not hit capacity; not a pass for any of these accounts.',
            adoption=False)
        verify_hashes(protocol['sources']);verify_hashes(protocol['frozen_analysis_sources'])
        summary['source_hashes_unchanged']=True
        write_json(dest/'summary.json',summary)
        files={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        write_json(dest/'complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),files=files,
            protocol_sha256=sha(output/'protocol.json'),source_hashes_unchanged=True))
        print(json.dumps(dict(status='completed',accounts=len(summary['accounts']),summary=str(dest/'summary.json'))))
    except Exception as exc:
        write_json(dest/'failed.json',dict(type=type(exc).__name__,error=str(exc),at=datetime.now(timezone.utc).isoformat(),new_accounts_run=0))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,default=Path('/Users/agiuser/Desktop/QuantLab'))
    parser.add_argument('--output',type=Path);parser.add_argument('--run-name',default='run1')
    args=parser.parse_args();root=args.root.resolve();output=args.output or root/'output/reference_selection_20260927/accounts'
    if args.command=='prepare':prepare(root,output)
    else:analyze(root,output,args.run_name)

if __name__=='__main__':main()
