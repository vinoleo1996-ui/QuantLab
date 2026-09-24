"""Descriptive account evidence. This module never changes strategy decisions."""
from collections import defaultdict

import numpy as np
import pandas as pd


def _day(value):
    return pd.Timestamp(value).normalize()


def drawdown_recovery(nav):
    """Count observed sessions from a prior high, including initial capital=1."""
    nav = pd.Series(nav, dtype=float)
    if nav.empty:
        return dict(status='no_account', deepest=None, current=None)
    peak, peak_index, peak_date = 1., -1, None
    episode, episodes = None, []
    for index, (date, value) in enumerate(nav.items()):
        date = _day(date)
        if value >= peak * (1 - 1e-12):
            if episode is not None:
                episode.update(recovered=True, recovery_date=str(date.date()),
                    peak_to_recovery_sessions=index-peak_index,
                    trough_to_recovery_sessions=index-episode.pop('_trough_index'),
                    elapsed_sessions=index-peak_index)
                episodes.append(episode)
                episode = None
            peak, peak_index, peak_date = max(peak, value), index, str(date.date())
            continue
        depth = float(value/peak-1)
        if episode is None:
            episode = dict(peak_date=peak_date, peak_basis='initial_capital' if peak_date is None else 'observed_close',
                peak_nav=float(peak), first_underwater_date=str(date.date()),
                trough_date=str(date.date()), depth=depth, _trough_index=index,
                underwater_sessions=0, recovered=False, recovery_date=None,
                peak_to_recovery_sessions=None, trough_to_recovery_sessions=None)
        episode['underwater_sessions'] += 1
        if depth < episode['depth']:
            episode.update(depth=depth, trough_date=str(date.date()), _trough_index=index)
        episode['elapsed_sessions'] = index-peak_index
    current = None
    if episode is not None:
        episode.pop('_trough_index')
        episode['observed_through'] = str(_day(nav.index[-1]).date())
        episodes.append(episode)
        current = dict(episode)
    deepest = min(episodes, key=lambda row: row['depth']) if episodes else None
    return dict(status='observed' if episodes else 'no_drawdown',
        definition='Peak-to-recovery counts observed trading sessions; a first-day loss is measured from initial NAV=1. Unrecovered durations are censored at the last account date.',
        deepest=deepest, current=current, unrecovered_at_end=current is not None,
        longest_elapsed_sessions=max((row['elapsed_sessions'] for row in episodes), default=0),
        recovered_episodes=sum(row['recovered'] for row in episodes), episodes=episodes)


def halfyear_returns(returns, calendar=None):
    returns = pd.Series(returns, dtype=float)
    exchange_days = pd.DatetimeIndex(calendar if calendar is not None else []).normalize()
    rows = []
    for (year, half), values in returns.groupby([returns.index.year, (returns.index.month>6).astype(int)+1]):
        begin = pd.Timestamp(year=int(year), month=1 if half==1 else 7, day=1)
        end = pd.Timestamp(year=int(year), month=6 if half==1 else 12, day=30 if half==1 else 31)
        expected = exchange_days[(exchange_days>=begin)&(exchange_days<=end)]
        complete = values.index.normalize().equals(expected) if len(expected) else None
        rows.append(dict(period=f'{year}H{half}', sessions=len(values),
            start=str(values.index[0].date()), end=str(values.index[-1].date()), complete=complete,
            net_return=float((1+values).prod()-1)))
    complete = [row for row in rows if row['complete'] is True]
    return dict(definition='Compounded net account return within each calendar half-year; incomplete periods are explicitly labelled.',
        periods=rows, worst_complete=min(complete, key=lambda row:row['net_return']) if complete else None,
        worst_observed=min(rows, key=lambda row:row['net_return']) if rows else None)


def realized_concentration(run):
    states = run.get('account', [])
    history = {_day(row['date']):row for row in run.get('risk_history', [])}
    rows = []
    for state in states:
        date = _day(state['date'])
        weights = {c:float(w) for c,w in state.get('weights', {}).items() if w>0}
        ranked = sorted(weights.values(), reverse=True)
        gross = sum(ranked)
        risk = history.get(date, {})
        sector = risk.get('actual_sector_exposure')
        cluster = risk.get('actual_group_exposure', risk.get('actual_cluster_exposure'))
        def observed(exposures):
            if not weights: return {}
            if exposures is None or not np.isfinite(list(exposures.values())).all(): return None
            if not np.isclose(sum(exposures.values()), gross, rtol=1e-6, atol=1e-8): return None
            return exposures
        sector, cluster = observed(sector), observed(cluster)
        rows.append(dict(date=str(date.date()), holdings=len(weights), gross=gross,
            max_single_security=max(ranked, default=0.), top5=sum(ranked[:5]),
            security_hhi_nav=sum(w*w for w in ranked),
            security_hhi_invested=sum((w/gross)**2 for w in ranked) if gross else None,
            max_industry=max(sector.values(), default=0.) if sector is not None else None,
            unknown_industry_weight=sum(w for name,w in sector.items() if 'UNKNOWN' in str(name).upper() or 'UNCLASSIFIED' in str(name).upper()) if sector is not None else None,
            max_group=max(cluster.values(), default=0.) if cluster is not None else None))
    def summary(field):
        available = [row for row in rows if row[field] is not None]
        peak = max(available,key=lambda row:row[field]) if available else None
        missing = sum(row[field] is None for row in rows)
        funded = [row for row in rows if row['holdings']]
        funded_available = sum(row[field] is not None for row in funded)
        result = dict(status='complete' if rows and missing==0 else 'partial' if available else 'not_computable',
            maximum=peak[field] if peak else None, peak_date=peak['date'] if peak else None,
            mean=float(np.mean([row[field] for row in available])) if available else None,
            observed_sessions=len(available), missing_sessions=missing,
            funded_sessions=len(funded), funded_observed_sessions=funded_available)
        if funded and not funded_available:
            result.update(status='not_computable', maximum=None, peak_date=None, mean=None)
        return result
    groups = summary('max_group')
    if groups['funded_observed_sessions'] < groups['funded_sessions']:
        groups['reason'] = 'daily_actual_group_partition_not_persisted; no interpolation from monthly decision groups'
        if not groups['funded_observed_sessions']:
            groups.update(status='not_computable', maximum=None, peak_date=None, mean=None)
    return dict(definition='Daily actual security market values divided by same-close NAV. Cash excluded; UNKNOWN industry remains one group. Sector/group observations require full gross-exposure coverage on that exact date.',
        single_security=summary('max_single_security'), top5=summary('top5'), industry=summary('max_industry'),
        unknown_industry=summary('unknown_industry_weight'), group=groups, daily=rows)


def turnover_and_exits(run):
    states = run.get('account', [])
    dates = pd.DatetimeIndex([_day(row['date']) for row in states])
    supplied_calendar = run.get('calendar')
    calendar = pd.DatetimeIndex(supplied_calendar if supplied_calendar is not None and len(supplied_calendar) else dates).normalize()
    initial = run.get('config', {}).get('capital', {}).get('init_cash')
    initial = float(initial) if initial is not None else None
    previous = dict(zip(dates, [initial]+[row['nav'] for row in states[:-1]]))
    fills_by_order, notional_by_day = defaultdict(list), defaultdict(float)
    unmatched = set()
    for fill in run.get('fills', []):
        date = _day(fill['date'])
        fills_by_order[fill['order_id']].append(fill)
        notional_by_day[date] += float(fill['notional'])
        if date not in previous: unmatched.add(str(date.date()))
    normalized = [value/previous[day] for day,value in notional_by_day.items()
                  if day in previous and previous[day] is not None and previous[day]>0]
    denominators_known = all(previous[day] is not None and previous[day]>0 for day in notional_by_day if day in previous)
    turnover = dict(definition='252/N times sum of daily actual BUY+SELL filled notional divided by preceding-close NAV; first session uses initial capital. Initial deployment is included. This is two-way turnover, not half of it.',
        annualized_two_way=float(sum(normalized)*252/len(dates)) if len(dates) and not unmatched and denominators_known else None,
        total_filled_notional=sum(notional_by_day.values()), account_sessions=len(dates),
        unmatched_fill_dates=sorted(unmatched), status='observed' if len(dates) and not unmatched and denominators_known else 'incomplete')
    cached_risk = {row['order_id']:row for row in run.get('capital_diagnostics',{}).get('risk_exits',[])}
    last_day = dates[-1] if len(dates) else None
    orders = []
    for order in run.get('orders', []):
        if order.get('side')!='sell': continue
        fills = sorted(fills_by_order[order['order_id']], key=lambda row:pd.Timestamp(row['date']))
        filled_quantity = sum(fill['quantity'] for fill in fills)
        completed = order['remaining']==0 and bool(fills) and filled_quantity>=order['quantity']
        pending = order.get('status') in ('pending','partial') and order['remaining']>0
        end = _day(fills[-1]['date']) if completed else last_day if pending else None
        elapsed = int(((calendar>_day(order['signal_date']))&(calendar<=end)).sum()) if end is not None else None
        cached = cached_risk.get(order['order_id'])
        orders.append(dict(order_id=order['order_id'], instrument_id=order['instrument_id'],
            reason=order.get('reason'), signal_date=order['signal_date'], status=order.get('status'),
            requested_quantity=order['quantity'], filled_quantity=filled_quantity, remaining=order['remaining'],
            completed=completed, pending=pending, elapsed_sessions=elapsed,
            completion_date=str(end.date()) if completed else None,
            observed_through=str(last_day.date()) if pending and last_day is not None else None,
            reject_reason=order.get('reject_reason'),
            capital_diagnostic_elapsed_sessions=cached.get('elapsed_sessions') if cached else None,
            capital_diagnostic_consistent=cached.get('elapsed_sessions')==elapsed if cached and elapsed is not None else None))
    def aggregate(selected):
        completed = [row['elapsed_sessions'] for row in selected if row['completed']]
        pending = [row for row in selected if row['pending']]
        return dict(total_orders=len(selected), completed_orders=len(completed), pending_orders=len(pending),
            cancelled_or_other_incomplete_orders=sum(not row['completed'] and not row['pending'] for row in selected),
            completed_median_sessions=float(np.median(completed)) if completed else None,
            completed_max_sessions=max(completed) if completed else None,
            pending_max_elapsed_sessions=max((row['elapsed_sessions'] for row in pending),default=None),
            pending_remaining_shares=sum(row['remaining'] for row in pending))
    exits = dict(definition='Elapsed exchange sessions from sell signal (excluded) to its final actual fill or last account date. Pending observations are censored; superseded/cancelled orders are separate. A sell-order completion need not liquidate the whole position.',
        calendar_basis='exchange_calendar' if supplied_calendar is not None and len(supplied_calendar) else 'observed_account_sessions_only',
        all_sells=aggregate(orders), risk_exits=aggregate([row for row in orders if row['reason']=='risk_exit']), orders=orders,
        reused_capital_diagnostic_rows=len(cached_risk))
    return turnover, exits


def account_diagnostics(run):
    states = run.get('account', [])
    if not states: return dict(status='no_account')
    turnover, exits = turnover_and_exits(run)
    return dict(status='descriptive_not_strategy_evidence',
        drawdown_recovery=drawdown_recovery(run['nav']),
        halfyear_returns=halfyear_returns(run['daily_return'],run.get('calendar')),
        realized_concentration=realized_concentration(run), turnover=turnover, sell_completion=exits)
