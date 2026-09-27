"""Four independent execution shocks for approved positive-increment candidates.

Targets, signal prices, source panels and the frozen ledger remain untouched.
Only a scoped in-memory ProfileLedger subclass changes execution permissions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import expert_selection_study as study
from quantlab.backtest.ledger import trade_permission, price_bounds

shared, replay, reference = study.shared, study.replay, study.reference
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/expert_selection_20260927'
CASES=('cost2','adv03','lock5','delay1')
VERSION='expert-selection-execution-stress-v1'


def choose_lock(baseline,targets,calendar):
    """First submitted full exit of real shares, tied by code; no planted holding."""
    _,schedule=replay.read_targets(targets)
    calendar=pd.DatetimeIndex(calendar)
    states={pd.Timestamp(a['date']):a for a in baseline['account']}
    choices=[]
    for order in baseline['orders']:
        signal=pd.Timestamp(order['signal_date'])
        code=order['instrument_id']
        held=states.get(signal,{}).get('quantities',{}).get(code,0)
        if order['side']!='sell' or not held or order['quantity']!=held:
            continue
        if schedule.get(signal,{}).get(code,0.)!=0:continue
        position=calendar.get_indexer([signal])[0]
        if position<0 or position+1>=len(calendar):continue
        start=max(position+1,order.get('execute_not_before_session') or 0)
        if start>=len(calendar):continue
        choices.append((start,code,str(order['signal_date']),order,held))
    if not choices:return None
    start,code,signal,order,held=min(choices,key=lambda x:x[:3])
    dates=[str(d.date()) for d in calendar[start:start+5]]
    baseline_fills=[dict(date=f['date'],quantity=int(f['quantity']),price=float(f['price']))
        for f in baseline['fills'] if f['order_id']==order['order_id'] and f['side']=='sell' and f['date'] in dates]
    return dict(instrument_id=code,signal_date=signal,baseline_order_id=order['order_id'],
        submitted_exit_quantity=int(order['quantity']),held_at_submission=int(held),
        start_session=int(start),lock_dates=dates,expected_sessions=5,
        baseline_exit_fills_in_window=baseline_fills,
        selection='first real full-exit order, earliest attempted execution session then instrument code')


def executable_quantity(account,order,date,session,bar,rule,capacity,used,can_buy=None):
    """Remaining shares executable at this state, using frozen quantity rules.

    This is a read-only diagnostic after actual execution. Restoring original
    ADV must permit additional shares after T+1, whole-position odd lots,
    opening permissions, cash and exposure checks; a small capacity remainder
    alone is not evidence that liquidity prevented a trade.
    """
    if order.remaining<=0 or pd.Timestamp(order.signal_date)>=pd.Timestamp(date):return 0
    if order.execute_not_before_session is not None and session<order.execute_not_before_session:return 0
    allowed,_=trade_permission(bar,rule,order.side,date,allow_open_assumption=account.allow_open_assumption)
    if not allowed or not np.isfinite(capacity) or capacity<=0:return 0
    px0=float(bar['open'])
    px=px0*(1+account.slippage*account.cost_mult*(1 if order.side=='buy' else -1))
    px=float((Decimal(str(px))/Decimal(str(rule.price_tick))).quantize(Decimal('1'),rounding=ROUND_HALF_UP))*rule.price_tick
    upper,lower=price_bounds(bar,rule)
    if (order.side=='buy' and np.isfinite(upper) and px>=upper-rule.price_tick/2) or (
        order.side=='sell' and np.isfinite(lower) and px<=lower+rule.price_tick/2):return 0
    if np.isfinite(upper):px=min(px,upper)
    if np.isfinite(lower):px=max(px,lower)
    if px<=0:return 0
    c=order.instrument_id;lot=int(rule.buy_lot if order.side=='buy' else rule.sell_lot)
    qty=min(order.remaining,int(max(0.,capacity-used)/px))
    if order.side=='sell':qty=min(qty,account.available(c,session))
    if order.side=='buy' or qty<account.quantity(c):qty=qty//lot*lot
    if order.side=='buy':
        reserve=account.nav()*account.cash_buffer
        qty=min(qty,max(0,int((account.cash-reserve)/px))//lot*lot)
        while qty>0:
            commission,stamp=account.fee(date,rule,order.side,qty*px)
            if (qty*px+commission+stamp<=account.cash-reserve+.001 and
                (can_buy is None or can_buy(c,qty,px,commission+stamp,account))):break
            qty-=lot
    elif qty>0:
        notional=replay.money(qty*px)
        commission,stamp=account.fee(date,rule,order.side,notional)
        tax=account.dividend_tax(c,qty,session,date)
        if account.cash+notional-replay.money(commission+stamp+tax)<0:return 0
    return int(qty)


@contextmanager
def ledger_override(calendar,case,lock_plan,evidence):
    original=replay.ProfileLedger
    calendar=pd.DatetimeIndex(calendar)
    locked=set(lock_plan['lock_dates']) if lock_plan else set()
    class StressLedger(original):
        def queue_targets(self,weights,signal_date,reason,rules,*args,**kwargs):
            begin=len(self.orders)
            super().queue_targets(weights,signal_date,reason,rules,*args,**kwargs)
            if case=='delay1':
                position=calendar.get_indexer([pd.Timestamp(signal_date)])[0]
                if position<0:raise ValueError('stress_signal_outside_calendar')
                for order in self.orders[begin:]:
                    # The delayed trade retains its original signal date and
                    # original fixed share amount; no later-price resubmission.
                    order.execute_not_before_session=max(order.execute_not_before_session or 0,position+2)
                    order.execute_not_before=(str(calendar[position+2].date()) if position+2<len(calendar) else None)
                    evidence['delay_orders'].append(dict(order_id=order.order_id,signal_date=order.signal_date,
                        quantity=order.quantity,execute_not_before_session=order.execute_not_before_session,
                        execute_not_before=order.execute_not_before))

        def execute(self,date,session,bars,rules,adv,can_buy=None):
            day=str(pd.Timestamp(date).date())
            active=[o for o in self.orders if o.status in ('pending','partial') and pd.Timestamp(o.signal_date)<pd.Timestamp(date)]
            original_bars=bars
            if case=='delay1':
                for order in active:
                    if order.execute_not_before_session is not None and session<order.execute_not_before_session:
                        evidence['delay_withheld'].append(dict(date=day,order_id=order.order_id,
                            remaining=order.remaining,held_shares=self.quantity(order.instrument_id),
                            eligible_session=order.execute_not_before_session))
            locked_orders=[]
            if case=='lock5' and lock_plan and day in locked:
                code=lock_plan['instrument_id']
                bars=dict(bars)
                if code in bars:
                    bars[code]=dict(bars[code],open_status='halted')
                locked_orders=[o for o in active if o.instrument_id==code and o.side=='sell']
                permitted,permission_reason=(trade_permission(original_bars.get(code),rules[code],'sell',date,
                    allow_open_assumption=self.allow_open_assumption) if code in rules else (False,'unknown_rules'))
                evidence['lock_sessions'].append(dict(date=day,instrument_id=code,held_shares=self.quantity(code),
                    active_exit_quantity=sum(o.remaining for o in locked_orders),
                    source_open_status=original_bars.get(code,{}).get('open_status'),
                    source_close=original_bars.get(code,{}).get('close'),
                    source_open=original_bars.get(code,{}).get('open'),
                    source_sell_permission=bool(permitted),source_permission_reason=permission_reason,
                    original_reasons=[o.reason for o in locked_orders]))
            capacity=[]
            changed_adv=adv
            if case=='adv03':
                changed_adv={c:(float(v)*.3 if np.isfinite(v) else v) for c,v in adv.items()}
                for order in active:
                    code=order.instrument_id
                    if code not in rules or code not in bars or not np.isfinite(adv.get(code,np.nan)):continue
                    px=bars[code].get('open',np.nan)
                    if not np.isfinite(px) or px<=0:continue
                    rule=rules[code];lot=rule.buy_lot if order.side=='buy' else rule.sell_lot
                    capacity.append(dict(date=day,order_id=order.order_id,instrument_id=code,side=order.side,
                        remaining_before=order.remaining,source_prior_adv=float(adv[code]),
                        original_notional_capacity=float(adv[code])*self.participation,
                        reduced_notional_capacity=float(changed_adv[code])*self.participation,
                        held_before=self.quantity(code),available_before=self.available(code,session),
                        lot=int(lot),open_price=float(px)))
            begin=len(self.fills)
            super().execute(date,session,bars,rules,changed_adv,can_buy=can_buy)
            newfills=self.fills[begin:]
            for order in locked_orders:
                evidence['lock_attempts'].append(dict(date=day,order_id=order.order_id,
                    reason=order.reject_reason,remaining=order.remaining,
                    held_after=self.quantity(order.instrument_id),cash_after=self.cash))
            for row in capacity:
                order=next(o for o in active if o.order_id==row['order_id'])
                fills=[f for f in newfills if f['order_id']==order.order_id]
                used=sum(f['notional'] for f in self.fills if f['date']==day and f['instrument_id']==order.instrument_id)
                original_qty=executable_quantity(self,order,date,session,bars[order.instrument_id],
                    rules[order.instrument_id],row['original_notional_capacity'],used,can_buy)
                reduced_qty=executable_quantity(self,order,date,session,bars[order.instrument_id],
                    rules[order.instrument_id],row['reduced_notional_capacity'],used,can_buy)
                binding=original_qty>reduced_qty
                row.update(fill_quantity=sum(f['quantity'] for f in fills),fill_notional=sum(f['notional'] for f in fills),
                    daily_used_notional=used,remaining_after=order.remaining,reject_reason=order.reject_reason,
                    held_after=self.quantity(order.instrument_id),available_after=self.available(order.instrument_id,session),
                    extra_executable_original_capacity=original_qty,extra_executable_reduced_capacity=reduced_qty,
                    incremental_executable_shares=original_qty-reduced_qty,
                    capacity_exhausted=bool(binding))
                evidence['liquidity_attempts'].append(row)
    replay.ProfileLedger=StressLedger
    try:yield
    finally:replay.ProfileLedger=original


def perturbation_summary(result,case,lock_plan,raw):
    liquid=raw['liquidity_attempts']
    cap_hits=sum(r['capacity_exhausted'] for r in liquid)
    rejected=sorted({r['date'] for r in raw['lock_attempts'] if r['reason']=='unknown_or_halted_open' and r['remaining']>0})
    lock=dict(plan=lock_plan,rejected_sessions=len(rejected),rejected_dates=rejected,
              completed_exit_date=None,remaining_position=None,incremental_blocked_baseline_quantity=0)
    if lock_plan:
        code=lock_plan['instrument_id'];start=lock_plan['lock_dates'][0]
        after=[a for a in result['account'] if str(pd.Timestamp(a['date']).date())>=start]
        cleared=next((a for a in after if not a['quantities'].get(code,0)),None)
        lock.update(completed_exit_date=str(pd.Timestamp(cleared['date']).date()) if cleared else None,
            remaining_position=int(after[-1]['quantities'].get(code,0)) if after else None,
            incremental_blocked_baseline_quantity=sum(f['quantity'] for f in lock_plan.get('baseline_exit_fills_in_window',[])
                if f['date'] in rejected),
            exit_orders=[o for o in result['orders'] if o['instrument_id']==code and o['side']=='sell' and
                         o['signal_date']==lock_plan['signal_date']])
    exercised=({'neutral':True,'cost2':bool(result['fills']), 'adv03':cap_hits>0,
               'delay1':bool(raw['delay_withheld']),
               'lock5':len(rejected)==5 and lock['incremental_blocked_baseline_quantity']>0 and
                       len({r['date'] for r in raw['lock_sessions'] if r['held_shares']>0 and r['active_exit_quantity']>0})==5}[case])
    return dict(case=case,status='exercised' if exercised else 'not_exercised',exercised=exercised,
        cost_multiplier=2. if case=='cost2' else 1.,
        liquidity=dict(adv_multiplier=.3 if case=='adv03' else 1.,bound_order_sessions=int(cap_hits),
            attempted_order_sessions=len(liquid),evidence='restoring original capacity permits more shares at actual post-execution state, preserving T+1, whole-position odd lots, permissions, cash and exposure'),
        delay=dict(extra_sessions=1 if case=='delay1' else 0,withheld_attempts=len(raw['delay_withheld'])),
        lock=lock,raw=raw,
        rule='unexercised shock is not a passed execution test; no stronger shock substituted')


def simulate(ds,targets,*,start,end,case,lock_plan=None,protocol_hash,arm):
    if case not in CASES+('neutral',):raise ValueError('unregistered_stress_case')
    raw=dict(delay_orders=[],delay_withheld=[],lock_sessions=[],lock_attempts=[],liquidity_attempts=[])
    with ledger_override(ds.calendar,case,lock_plan,raw):
        result=replay.replay_reference(ds,targets,start=start,end=end,
            profile=replay.ReplayProfile.common(50),cost_mult=2. if case=='cost2' else 1.)
    result['stress_evidence']=perturbation_summary(result,case,lock_plan,raw)
    result.update(stress_case=case,stress_arm=arm,stress_protocol_sha256=protocol_hash,
        parent_data_version=study.PARENT_VERSION,model_version=VERSION,promotion_approved=False)
    result['run_id']=shared.digest_json(dict(protocol=protocol_hash,arm=arm,case=case,
        target=result['protocol']['target_schedule_sha256'],start=str(start),end=str(end),data=result['data_version']))[:24]
    if case!='neutral':
        for row in result['orders']+result['fills']:
            row['stress_case']=case;row['stress_arm']=arm;row['stress_protocol_sha256']=protocol_hash
    return result


def destination(stage):return OUT/'stress'/stage


@study.exclusive
def prepare(stage='all'):
    main,prepared=study.verify(stage)
    folder=destination(stage)
    if (folder/'protocol.json').exists():
        verify(stage);study.progress('stress_protocol_already_prepared',study_stage=stage);return
    comparison=study.stage_dir(stage)/'comparison.json'
    value=json.loads(comparison.read_text())
    compared=value['comparison_identity']
    if compared['protocol_sha256']!=prepared['protocol_sha256']:
        raise ValueError('stress_comparison_main_identity_changed')
    for path,digest in compared['completed_accounts'].items():shared.verify_file(path,digest)
    candidates=[a for a in main['arms'] if a!='R' and any(
        p['arms'][a]['net_profit']>p['arms']['R']['net_profit'] for p in value['periods'].values())]
    m=json.loads(reference.MANIFEST.read_text())
    calendar=pd.DatetimeIndex(reference.read_table(reference.MANIFEST.parent,m['calendar']).session)
    sources={str(p):shared.sha256(p) for p in [Path(__file__),ROOT/'tests/test_expert_selection_stress.py',
        Path(study.__file__),Path(replay.__file__),Path(reference.__file__),Path(shared.__file__),
        study.stage_dir(stage)/'protocol.json',study.stage_dir(stage)/'prepared.json',comparison,reference.MANIFEST]}
    plans=[]
    for period,(start,end) in study.PERIODS.items():
        for arm in candidates:
            root=study.account_folder(stage,period,arm)
            complete=study.verify_complete(root)
            for name,digest in complete.items():sources[str(root/name)]=digest
            sources[str(root/'complete.json')]=shared.sha256(root/'complete.json')
            targets=study.target_path(stage,period,arm)
            sources[str(targets)]=shared.sha256(targets)
            baseline=shared.checked_pickle(root/'result.pkl',complete['result.pkl'])
            lock=choose_lock(baseline,targets,calendar)
            for case in CASES:
                plans.append(dict(period=period,arm=arm,case=case,start=start,end=end,
                    target_path=str(targets),target_sha256=shared.sha256(targets),baseline=str(root),
                    baseline_files=complete,lock_plan=lock if case=='lock5' else None))
    protocol=dict(version=VERSION,stage=stage,created_at=datetime.now(timezone.utc).isoformat(),
        question='Does a fixed positive-increment candidate retain profit and account integrity under registered execution perturbations?',
        eligibility='candidate net_profit>R in at least one period; run both periods; no new configuration',
        cases=dict(cost2='existing multiplier2 for commission/tax/slippage',adv03='only execution priorADV multiplied .3',
            lock5='first actual full exit is halted for5sessions with unmodified close marks',
            delay1='earliest execution original_signal_session+2; shares unchanged'),
        candidates=candidates,plans=plans,sources=sources,main_protocol_sha256=prepared['protocol_sha256'],
        ledger=shared.source_identity(),profile=main['profile'],parent_data_version=study.PARENT_VERSION,
        interpretation='not_exercised never passes; no stronger replacement shocks; account integrity and profitability reported separately',
        promotion_approved=False)
    shared.freeze_json(folder/'protocol.json',protocol)
    for path in sources:
        p=Path(path)
        if p.suffix=='.py':shared.publish(folder/'source'/p.relative_to(ROOT),lambda h,p=p:h.write(p.read_bytes()))
    study.progress('stress_prepared',study_stage=stage,candidates=candidates,accounts=len(plans))


def verify(stage):
    main,prepared=study.verify(stage)
    folder=destination(stage)
    protocol=json.loads((folder/'protocol.json').read_text())
    if protocol['main_protocol_sha256']!=prepared['protocol_sha256']:raise ValueError('stress_main_protocol_changed')
    for path,digest in protocol['sources'].items():shared.verify_file(path,digest)
    if protocol['ledger']!=shared.source_identity():raise ValueError('stress_frozen_ledger_changed')
    return protocol,prepared


def checkpoint_identity(protocol_hash,plan):
    return dict(protocol_sha256=protocol_hash,plan_sha256=shared.digest_json(plan))


def load_completed(folder,identity):
    complete=study.verify_complete(folder)
    if json.loads((folder/'identity.json').read_text())!=identity:
        raise ValueError('stress_account_identity_changed')
    return json.loads((folder/'summary.json').read_text())


def save_simulation(folder,result,identity,plan):
    """Freeze result bytes and their independent checkpoint before reporting."""
    shared.verify_file(plan['target_path'],plan['target_sha256'])
    expected=replay.fingerprint(replay.read_targets(plan['target_path'])[1])
    if (result['stress_checkpoint_identity']!=identity or result['stress_case']!=plan['case'] or
        result['stress_arm']!=plan['arm'] or result['stress_protocol_sha256']!=identity['protocol_sha256'] or
        result['parent_data_version']!=study.PARENT_VERSION or
        [str(result['protocol']['start']),str(result['protocol']['end'])]!=[plan['start'],plan['end']] or
        result['protocol']['target_schedule_sha256']!=expected):
        raise ValueError('simulation_checkpoint_identity_mismatch')
    shared.publish(folder/'result.pkl',lambda h:pickle.dump(result,h,pickle.HIGHEST_PROTOCOL))
    shared.freeze_json(folder/'identity.json',identity)
    checkpoint=dict(status='simulation_completed',identity=identity,
        result_sha256=shared.sha256(folder/'result.pkl'),target_file_sha256=plan['target_sha256'],
        target_schedule_sha256=result['protocol']['target_schedule_sha256'],
        period=[plan['start'],plan['end']],case=plan['case'],arm=plan['arm'],
        data_version=result['data_version'],parent_data_version=result['parent_data_version'],
        stress_protocol_sha256=result['stress_protocol_sha256'])
    shared.freeze_json(folder/'simulation_complete.json',checkpoint)


def recover_partial(folder,identity,plan):
    if not folder.exists() or not any(folder.iterdir()):return None
    archive=folder.with_name(folder.name+'.interrupted.'+str(time.time_ns()))
    folder.rename(archive)
    study.progress('stress_partial_preserved',path=str(archive))
    path=archive/'result.pkl'
    checkpoint_path=archive/'simulation_complete.json'
    if not path.exists() or not checkpoint_path.exists():return None
    try:
        checkpoint=json.loads(checkpoint_path.read_text())
        if (checkpoint['status']!='simulation_completed' or checkpoint['identity']!=identity or
            checkpoint['case']!=plan['case'] or checkpoint['arm']!=plan['arm'] or
            checkpoint['period']!=[plan['start'],plan['end']] or
            checkpoint['target_file_sha256']!=plan['target_sha256'] or
            checkpoint['stress_protocol_sha256']!=identity['protocol_sha256'] or
            checkpoint['parent_data_version']!=study.PARENT_VERSION):return None
        shared.verify_file(plan['target_path'],checkpoint['target_file_sha256'])
        result=shared.checked_pickle(path,checkpoint['result_sha256'])
        expected=replay.fingerprint(replay.read_targets(plan['target_path'])[1])
        if (result['stress_checkpoint_identity']==identity and result['stress_case']==plan['case'] and
            result['protocol']['target_schedule_sha256']==expected==checkpoint['target_schedule_sha256'] and
            result['data_version']==checkpoint['data_version'] and result['stress_arm']==plan['arm']):
            return result
    except (ValueError,KeyError,TypeError,EOFError,pickle.UnpicklingError):pass
    return None


@study.exclusive
def run(stage='all'):
    protocol,prepared=verify(stage)
    phash=shared.sha256(destination(stage)/'protocol.json')
    ds=None; summaries={}
    for plan in protocol['plans']:
        period,arm,case=plan['period'],plan['arm'],plan['case']
        key=period+'/'+arm+'/'+case;folder=destination(stage)/period/arm/case
        identity=checkpoint_identity(phash,plan)
        if (folder/'complete.json').exists():
            summaries[key]=load_completed(folder,identity);continue
        result=recover_partial(folder,identity,plan)
        if result is None:
            if ds is None:ds=study.projection(stage,prepared)
            study.progress('stress_running',period=period,arm=arm,case=case)
            result=simulate(ds,plan['target_path'],start=plan['start'],end=plan['end'],case=case,
                lock_plan=plan['lock_plan'],protocol_hash=phash,arm=arm)
        result['stress_checkpoint_identity']=identity
        # Persist completed simulation before report/check calculations. A
        # reporting interruption must not discard a costly completed replay.
        save_simulation(folder,result,identity,plan)
        metrics,returns=reference.metrics(result,period)
        baseline=json.loads((Path(plan['baseline'])/'summary.json').read_text())
        extra=study.diagnostics(result,json.loads(Path(plan['target_path']).read_text())['trace'],returns)
        hits=result['stress_evidence']
        summary=dict(metrics,case=case,arm=arm,profit_change_vs_baseline=metrics['net_profit']-baseline['net_profit'],
            drawdown_change_vs_baseline=metrics['max_drawdown']-baseline['max_drawdown'],
            max_underwater_sessions=extra['max_underwater_sessions'],worst_half_year=extra['worst_half_year'],
            worst_daily_return=float(returns.min()),shock_status=hits['status'],shock_exercised=hits['exercised'],
            profitable=metrics['net_profit']>0,within_drawdown_budget=metrics['max_drawdown']>=-.25,
            exit_resolved=(hits['lock']['completed_exit_date'] is not None) if case=='lock5' else None,
            promotion_approved=False)
        summary['research_stress_pass']=bool(hits['exercised'] and metrics['account_checks']['passed'] and
            metrics['net_profit']>0 and metrics['max_drawdown']>=-.25 and
            (case!='lock5' or summary['exit_resolved']))
        shared.freeze_json(folder/'summary.json',summary)
        shared.freeze_json(folder/'perturbation.json',hits)
        shared.freeze_json(folder/'diagnostics.json',extra)
        shared.publish(folder/'daily_returns.csv',lambda h:h.write(returns.rename('daily_return').to_csv(index_label='date').encode()))
        names=('result.pkl','identity.json','simulation_complete.json','summary.json','perturbation.json','diagnostics.json','daily_returns.csv')
        shared.freeze_json(folder/'complete.json',{n:shared.sha256(folder/n) for n in names})
        summaries[key]=summary
        study.progress('stress_completed',period=period,arm=arm,case=case,net_profit=metrics['net_profit'],
                       exercised=hits['exercised'],research_stress_pass=summary['research_stress_pass'])
    verify(stage)
    final=dict(protocol_sha256=phash,accounts=summaries,eligible_candidates=protocol['candidates'],
        status='completed' if summaries else 'no_positive_increment_candidates',promotion_approved=False,
        limitations=['conditional data','earlier history already seen','stress cannot cover every liquidity failure','six month frozen forward unverified'])
    path=destination(stage)/'summary.json'
    if path.exists():
        if json.loads(path.read_text())!=final:raise ValueError('stress_summary_identity_changed')
    else:shared.freeze_json(path,final)
    study.progress('all_stress_completed',study_stage=stage,accounts=len(summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--stage',choices=['rs','all'],default='all')
    args=parser.parse_args();globals()[args.action](args.stage)
