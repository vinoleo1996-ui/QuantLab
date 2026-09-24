"""Actual-share reconstruction of the frozen 100-name reference targets.

This standalone simulation owns the original reference permissions. It does not
alter QuantLab's main Mandate or insert P0/P1/P2 drawdown/stop-loss policies.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

from quantlab.backtest.ledger import Ledger, money
from quantlab.data.contracts import InstrumentRules, asof_records, decision_time, fingerprint, timestamp
from quantlab.data.local_adapter import canonical_code
from quantlab.report.metrics import performance_table
from quantlab.report.runs import persist_run, atomic_json, provenance


REFERENCE_COST = dict(commission=.0003, minimum_commission=5., one_way_slippage=.0005,
    original_constant_stamp=.001, original_constant_transfer=.00002,
    actual_stamp_change='2023-08-28', actual_transfer_change='2022-04-29',
    min_trade_value=0., history='stamp .001 -> .0005; transfer .00002 -> .00001',
    main_research_cost_difference='Reference retains original 5bp one-way slippage, not main strategy 30bp. Historical tax/transfer changes are corrected.')


class ReferenceLedger(Ledger):
    """Shared quantity/cash accounting, with explicit reference transfer charges."""
    def fee(self, date, rule, side, notional):
        commission, stamp = super().fee(date, rule, side, notional)
        transfer = notional * (.00002 if pd.Timestamp(date) < pd.Timestamp('2022-04-29') else .00001) * self.cost_mult
        return money(commission + transfer), stamp

    def execute(self, date, session, bars, rules, adv, can_buy=None):
        start = len(self.fills)
        super().execute(date, session, bars, rules, adv, can_buy=can_buy)
        # The original reference renews unfilled normal targets until its next
        # explicit target. Keep fixed share quantities; never recalculate them
        # from future NAV while retrying. New target snapshots still supersede.
        for order in self.orders:
            if order.status == 'expired' and order.remaining > 0:
                order.status = 'partial' if order.remaining < order.quantity else 'pending'
        for fill in self.fills[start:]:
            rate = .00002 if pd.Timestamp(date) < pd.Timestamp('2022-04-29') else .00001
            transfer = money(fill['notional'] * rate * self.cost_mult)
            fill['transfer_fee_in_commission'] = transfer
            fill['broker_commission'] = money(fill['commission'] - transfer)


def read_targets(value):
    payload = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    targets = payload.get('targets', [])
    if payload.get('targets_sha256') and hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest() != payload['targets_sha256']:
        raise ValueError('reference_target_schedule_checksum_mismatch')
    dates = [pd.Timestamp(row['signal_date']).normalize() for row in targets]
    if dates != sorted(set(dates)):
        raise ValueError('reference_targets_must_have_unique_sorted_decision_dates')
    schedule = {}
    for day, row in zip(dates, targets):
        if row.get('sell_only', False):
            raise ValueError('unexpected_reference_sell_only_target')
        weights = {canonical_code(c): float(w) for c, w in row['weights'].items()}
        if len(weights) != len(row['weights']): raise ValueError('reference_symbol_collision')
        if len(weights) > 100 or any(not np.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError('invalid_reference_weights')
        if sum(weights.values()) > .90 + 1e-9:
            raise ValueError('reference_target_exceeds_original_exposure')
        schedule[day] = weights
    if not schedule: raise ValueError('reference_schedule_empty')
    return payload, schedule


def _reference_rules(records):
    result = {}
    for code, row in records.iterrows():
        try:
            rule = InstrumentRules.from_record(row)
            if rule.asset_type != 'stock': continue
            # The original reference was unit-based. This experiment explicitly
            # changes it to actual A-share lots and T+1, with original commission.
            result[code] = replace(rule, buy_lot=100, sell_lot=100, sell_delay=1,
                                   commission=REFERENCE_COST['commission'], min_commission=REFERENCE_COST['minimum_commission'])
        except ValueError:
            continue
    return result


def replay_reference(ds, targets, *, start, end, cost_mult=1., original_account=None):
    payload, schedule = read_targets(targets)
    if not np.isfinite(cost_mult) or cost_mult <= 0: raise ValueError('invalid_cost_multiplier')
    calendar = ds.calendar
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if dates.empty: raise ValueError('reference_replay_calendar_empty')
    if any(day not in calendar for day in schedule): raise ValueError('reference_signal_outside_exchange_calendar')
    codes = sorted({c for weights in schedule.values() for c,w in weights.items() if w>0})
    history = ds.instruments[ds.instruments.instrument_id.isin(codes)]
    panels = {c: ds.panels[c] for c in codes if c in ds.panels}
    actions = ds.actions[ds.actions.instrument_id.isin(codes)] if not ds.actions.empty else ds.actions
    missing_codes = sorted(set(codes)-set(panels))
    # Current full-day turnover never funds same-day opening capacity.
    amount = pd.DataFrame({c:p.amount for c,p in panels.items()}).reindex(calendar)
    prior_adv = amount.rolling(20, min_periods=20).mean().shift(1)
    protocol = dict(version='reference-cash-ledger-v1', role='reference_original', account_type='simulation',
        executable=False, initial_cash=500000., max_positions=100, target_gross=.90,
        industry_policy='original_target_selection_max_15_names_per_industry; no new industry-weight constraint',
        risk_policy='original_no_drawdown_scaler_no_individual_stop_no_new_forecasts',
        costs=REFERENCE_COST, cost_multiplier=cost_mult, participation=.01,
        changes=['actual_100_share_lots_and_T_plus_1', 'raw_execution_prices_and_explicit_corporate_actions',
                 'historical_stamp_and_transfer_rates', 'signal_close_fixed_share_intents',
                 'prior_20_session_ADV_1pct_capacity', 'actual_max_100_positions_including_unsold_holdings'],
        start=str(start), end=str(end), target_schedule_sha256=fingerprint(schedule),
        source_commit=payload.get('source_commit'), source_data_identity=payload.get('data_identity', payload.get('current_catalog')))
    ledger = ReferenceLedger(500000., slippage=.0005, cash_buffer=.10, participation=.01,
                             cost_mult=cost_mult, min_trade_value=0., allow_open_assumption=True)
    traces, rejections = [], []
    gaps_by_date, accounting_by_date = {}, {}
    target_codes = set(codes)
    for kind, grouped in (('data_gaps', gaps_by_date), ('accounting_gaps', accounting_by_date)):
        for gap in ds.manifest.get(kind, []):
            if gap['instrument_id'] in target_codes:
                grouped.setdefault(pd.Timestamp(gap['date']).normalize(), []).append(gap)
    actions_by_date = (dict(tuple(actions.groupby(pd.to_datetime(actions.ex_date).dt.normalize())))
                       if not actions.empty else {})
    latest_target = {}
    for day in dates:
        session = int(calendar.get_loc(day))
        relevant = set(ledger.quantities()) | {o.instrument_id for o in ledger.orders if o.status in ('pending','partial')}
        if day in schedule: relevant.update(schedule[day])
        records = asof_records(history, timestamp(day)+pd.Timedelta(hours=9,minutes=25))
        rules = _reference_rules(records)
        bars = {c:panels[c].loc[day].to_dict() for c in relevant if c in panels and day in panels[c].index}
        for gap in gaps_by_date.get(day, []):
            if ledger.quantity(gap['instrument_id']):
                ledger.data_issues.append(dict(gap, held_shares=ledger.quantity(gap['instrument_id'])))
        # Exposure belongs to shares held before this session's actions/fills.
        # It diagnoses an unexplained price-reference change, not a cash credit
        # or a strict upper/lower bound on missing economic profit.
        for gap in accounting_by_date.get(day, []):
            quantity = ledger.quantity(gap['instrument_id'])
            if quantity:
                ledger.data_issues.append(dict(gap, held_shares=quantity,
                    reference_gap_exposure=quantity*gap.get('reference_gap_per_share', 0.),
                    meaning='unexplained_reference_price_exposure_not_profit_correction_or_bias_bound'))
        ledger.corporate_actions(day, session, actions_by_date.get(day), calendar)
        def can_buy(code, quantity, price, fees, account):
            if code not in records.index: return False
            active = records.loc[code].get('active')
            if pd.isna(active) or not bool(active): return False
            held = account.quantities()
            if code not in held and len(held)>=100: return False
            values = {c: q*account.marks[c] for c,q in held.items()}
            values[code] = values.get(code,0.) + quantity*price
            return sum(values.values()) <= .9*(account.nav()-fees)+1e-6
        ledger.execute(day, session, bars, rules, prior_adv.loc[day].to_dict(), can_buy=can_buy)
        for order in ledger.orders:
            if order.status in ('pending','partial') and order.reject_reason:
                rejections.append(dict(date=str(day.date()),order_id=order.order_id,code=order.instrument_id,
                                       side=order.side,remaining=order.remaining,reason=order.reject_reason))
        closing = {c:b.get('close',np.nan) for c,b in bars.items() if b.get('quality_status')=='ok'
                   and pd.notna(b.get('available_at')) and timestamp(b['available_at'])<=decision_time(day)}
        ledger.close(day,session,closing)
        if day in schedule:
            latest_target = schedule[day]
            close_records = asof_records(history, decision_time(day))
            close_rules = _reference_rules(close_records)
            reliable_closes = {c for c, value in closing.items() if np.isfinite(value) and value>0}
            skipped = {}
            for code, weight in latest_target.items():
                if weight <= 0: continue
                if code not in close_rules:
                    skipped[code] = 'missing_reference_instrument_rule'
                elif code not in reliable_closes:
                    skipped[code] = 'missing_reliable_signal_close'
                if code in skipped:
                    # Removing only this submission rule preserves true held
                    # shares. A last mark remains valid for stale valuation,
                    # never for sizing a fresh positive target. Full exits with
                    # a known share quantity can still queue at zero weight.
                    close_rules.pop(code, None)
                    ledger.data_issues.append(dict(date=str(day.date()), instrument_id=code,
                        reason=skipped[code], held_shares=ledger.quantity(code), target_weight=weight,
                        action='skip_positive_target_adjustment_preserve_actual_shares'))
            begin = len(ledger.orders)
            ledger.queue_targets(latest_target, day, 'reference_rebalance', close_rules)
            traces.append(dict(date=str(day.date()), scores=None, target_weights=latest_target,
                nav=ledger.nav(), cash=ledger.cash, actual_weights=ledger.weights(), skipped_targets=skipped,
                orders=[dict(code=o.instrument_id,side=o.side,quantity=o.quantity,signal_date=o.signal_date)
                        for o in ledger.orders[begin:]], reason='original_frozen_target_schedule'))
        for order in ledger.orders:
            order.risk_policy='reference_original'
            order.candidate='reference1882_original'
            order.industry_policy='original_target_industry_name_limit'
    states = ledger.snapshots
    nav = pd.Series({s['date']:s['nav']/500000. for s in states},name='nav')
    returns = nav.pct_change(fill_method=None)
    returns.iloc[0] = nav.iloc[0]-1
    funded = bool(ledger.fills)
    verified = ds.manifest.get('actions_coverage') == 'complete' and not ledger.data_issues and not missing_codes
    accounting_exposures = [issue['reference_gap_exposure'] for issue in ledger.data_issues
                            if 'reference_gap_exposure' in issue]
    status = 'research_unvalidated' if funded and verified else 'conditional_reference_data_gaps' if funded else 'experiment_not_exercised'
    result = dict(run_id=fingerprint(dict(protocol=protocol,data_version=ds.version))[:24],
        data_version=ds.version, model_version=protocol['version'], as_of=str(end), status=status,
        reasons=(['actions_coverage_unverified'] if ds.manifest.get('actions_coverage')!='complete' else [])+
                (['reference_securities_missing'] if missing_codes else [])+
                (['held_or_signal_data_issues'] if ledger.data_issues else []),
        config=dict(capital={'init_cash':500000.}), protocol=protocol, market='a', strategy='reference_original',
        mandate_version='standalone_reference_simulation_only', promotion_approved=False,
        executable=False, account_type='simulation', nav=nav, daily_return=returns,
        drawdown=nav/nav.cummax().clip(lower=1)-1, metrics=None,
        conditional_metrics=performance_table(nav,returns) if funded else None,
        signal_trace=traces, trades=[], allocation_issues=[], risk_history=[], benchmark=ds.benchmark,
        missing_target_securities=missing_codes, rejection_history=rejections,
        accounting_gap_diagnostics=dict(held_events=len(accounting_exposures),
            signed_reference_exposure=sum(accounting_exposures),
            absolute_reference_exposure=sum(abs(x) for x in accounting_exposures),
            meaning='diagnostic_price_reference_exposure_not_profit_correction_or_strict_bias_bound'),
        rejection_counts=dict(Counter(r['reason'] for r in rejections)), **ledger.export())
    if original_account is not None:
        result['original_account_comparison'] = compare_original_account(result,original_account)
    return result


def compare_original_account(result, original_account):
    source = json.loads(Path(original_account).read_text()) if isinstance(original_account,(str,Path)) else original_account
    curve = source['equity_curve']
    original = pd.Series({pd.Timestamp(row[0]):float(row[1])/500000. for row in curve})
    actual = result['nav']
    common = actual.index.intersection(original.index)
    original_ret = original.pct_change(fill_method=None)
    original_ret.iloc[0] = original.iloc[0]-1
    excess = result['daily_return'].loc[common]-original_ret.loc[common]
    return dict(status='conditional_accounting_and_execution_comparison', sessions=len(common),
        dates_match=actual.index.equals(original.index),
        original_only_dates=[str(d.date()) for d in original.index.difference(actual.index)],
        actual_only_dates=[str(d.date()) for d in actual.index.difference(original.index)],
        ending_wealth_difference=float(actual.iloc[-1]-original.iloc[-1])*500000.,
        annualized_mean_daily_difference=float(excess.mean()*252) if len(excess) else None,
        account_differences={str(d.date()):dict(original_equity=float(original[d]*500000.),actual_equity=float(actual[d]*500000.)) for d in common},
        unit_caveat='Original adjusted units are not actual shares. Compare wealth and cash; do not subtract unit quantities from raw share counts.',
        changed_assumptions=result['protocol']['changes'])


def run_reference(manifest, targets, output, *, start, end, cost_mult=1., original_account=None):
    from quantlab.data.contracts import load_dataset
    ds = load_dataset(manifest, profile='free_research')
    before = provenance()
    result = replay_reference(ds,targets,start=start,end=end,cost_mult=cost_mult,original_account=original_account)
    if provenance()!=before: raise ValueError('source_changed_during_reference_cash_replay')
    result['source_dataset'] = dict(manifest=str(Path(manifest).resolve()), data_version=ds.version)
    root = Path(output)
    folder = persist_run(result,root=root/'runs')
    summary = {key:result[key] for key in ('run_id','data_version','model_version','as_of','status','reasons','protocol',
               'conditional_metrics','rejection_counts','missing_target_securities','accounting_gap_diagnostics')}
    summary.update(promotion_approved=False,artifact_directory=str(folder),
                   net_profit=result['account'][-1]['nav']-500000.,
                   original_account_comparison={k:v for k,v in result.get('original_account_comparison',{}).items() if k!='account_differences'})
    atomic_json(root/(result['run_id']+'.json'),summary)
    return summary
