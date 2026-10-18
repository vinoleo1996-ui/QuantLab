#!/usr/bin/env python3
"""P5_BASE forward paper snapshot (P5_FORWARD prereg, amendments 1-3).

Each run replays both sleeves from 2018 through the requested session with the frozen
stock schedule (``p4_ledger``, 52 万 / 97% gross) and the G0 lot sleeve (28 万), blends
them with the frozen quarterly rule, marks the book at the as-of close, and reports the
ledger's still-pending orders for the next open. The state file is an append-only history
of paper marks; it is never used to price or rebuild the book.

No broker connection. A run is ``blocked`` (orders empty, NAV null, state not advanced)
when the requested date is not a complete production session or the data/account gate
fails. Status is ``paper_simulated`` unless holdings and cash come from an explicit
broker file (``--holdings`` + ``--cash`` + ``--broker-confirmed``); broker state is kept
in a separate file. ``--dry-run`` writes under ``output/p5_forward/dryrun/<as-of>/``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import p4_alloc as R
import p5_base_lot_replay as B
import p5_base_orders as O
from quantlab import p4_alloc as A
from quantlab import p4_ledger
from quantlab.data.suspensions import SuspensionEvidence
from quantlab.smallcap.panel import resolve_trusted_panel
from quantlab.smallcap.universe import decision_grid

OUT = ROOT / 'output/p5_forward'
ALPHA = ROOT / 'output/p4_alpha_20261012'
START = '2018-01-02'
DEFAULT_CAPITAL = 800_000.0
EQUITY_SHARE = 0.65
STOCK_GROSS = 0.97
ETF_WEIGHTS = {'gold': 0.375 * 0.35, 'treasury': 0.375 * 0.35, 'nasdaq': 0.25 * 0.35}
ETF_CODES = {'gold': 'sh518880', 'treasury': 'sh511260', 'nasdaq': 'sh513100'}
ETF_NAMES = {code: name for name, code in ETF_CODES.items()}
SLIPPAGE, COMMISSION, MIN_COMMISSION, STAMP = 0.003, 0.0003, 5.0, 0.0005
MIN_TRADE, PARTICIPATION, PREMIUM_PAUSE = 1000.0, 0.01, 0.03
PHASES = (0, 5, 10, 15)
FORWARD_PHASE, FORWARD_SEED = 0, 1
PAPER_STATE = OUT / 'state_paper.json'
BROKER_STATE = OUT / 'state_broker.json'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def next_session(calendar, day):
    later = calendar[calendar > pd.Timestamp(day)]
    return later[0] if len(later) else None


def load_state(path, capital, status):
    if path.exists():
        state = json.loads(path.read_text())
        state['holdings'] = {k: int(v) for k, v in state.get('holdings', {}).items()}
        return state
    return {'status': status, 'as_of': None, 'cash': float(capital), 'holdings': {},
            'last_signal': None, 'last_etf_quarter': None,
            'pending_orders': [], 'pending_execute_on': None, 'history': []}


def broker_state(path, cash):
    frame = pd.read_csv(path)
    missing = {'code', 'shares'} - set(frame.columns)
    if missing:
        raise SystemExit(f'holdings csv missing columns: {sorted(missing)}')
    holdings = {str(row.code): int(row.shares) for row in frame.itertuples() if int(row.shares) > 0}
    return {'status': 'broker_confirmed', 'as_of': None, 'cash': float(cash), 'holdings': holdings,
            'last_signal': None, 'last_etf_quarter': None,
            'pending_orders': [], 'pending_execute_on': None, 'history': []}


def grid_report(schedule, calendar, requested, last_complete, phase):
    """Signal dates for the report, taken from the schedule that actually produced the orders.

    ``decision_grid`` on the production calendar is only a cross-check: when the two disagree the
    disagreement is published as ``calendar_mismatch`` instead of silently relabelling a day.
    """
    requested = pd.Timestamp(requested).normalize()
    signal_days = sorted(pd.Timestamp(day).normalize() for day in schedule)
    last_grid = max((day for day in signal_days if day <= requested), default=None)
    next_grid = next((day for day in signal_days if day > requested), None)
    audit = decision_grid(pd.DatetimeIndex(calendar), START,
                          pd.Timestamp(last_complete).normalize(), phase, 20)
    audit_last = max((day for day in audit if day <= requested), default=None)
    mismatch = None if audit_last == last_grid else {
        'frozen_schedule_last_signal': None if last_grid is None else str(last_grid.date()),
        'decision_grid_last_signal': None if audit_last is None else str(audit_last.date())}
    return {'last_grid': None if last_grid is None else str(last_grid.date()),
            'next_grid': None if next_grid is None else str(next_grid.date()),
            'signal_today': bool(last_grid is not None and last_grid == requested),
            'calendar_mismatch': mismatch}


def alloc60_series(etf_nav, calendar):
    """Rebuild the preregistered ALLOC60 benchmark (P5_FORWARD amendment 2).

    The equity leg is the frozen 90% gross ``POOL50`` ledger, not the replayed ``POOL50_G97`` book:
    the stop-loss check compares ``P5_BASE`` against a benchmark that must not be built from the
    strategy under test. Month ends come from the exchange calendar, so a window that stops mid-month
    does not invent a rebalance.
    """
    frozen_path = ALPHA / 'ledger/nav_POOL50_p0_s1_cost1.csv'
    frozen = pd.read_csv(frozen_path, index_col=0, parse_dates=True)['nav']
    sessions = pd.DatetimeIndex(calendar)
    sessions = sessions[(sessions >= frozen.index[0]) &
                        (sessions <= min(sessions.max(), frozen.index[-1]))]
    equity_nav = frozen.reindex(sessions)
    if equity_nav.isna().any():
        raise ValueError('alloc60_frozen_equity_has_gaps')
    etf_returns = pd.DataFrame({'g0': etf_nav.reindex(sessions)
                                .pct_change(fill_method=None).fillna(0.0)})
    nav, trace = A.allocate(equity_nav, etf_returns, 0.60, cost_mult=1, rebalance='month',
                            basis='calendar', calendar=pd.DatetimeIndex(calendar))
    config = {'equity': 'frozen POOL50 90% ledger nav_POOL50_p0_s1_cost1.csv',
              'frozen_equity_sha256': sha256_file(frozen_path),
              'etf': 'G0 lot sleeve replay', 'equity_share': 0.60, 'rebalance': 'month',
              'basis': 'calendar', 'cost_mult': 1,
              'sessions': int(len(sessions)),
              'first': str(sessions[0].date()), 'last': str(sessions[-1].date())}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return nav, digest, config


def six_month_check(history, alloc_nav, day, nav):
    if not history:
        return {'status': 'insufficient_history'}
    cutoff = pd.Timestamp(day) - pd.DateOffset(months=6)
    prior = [row for row in history
             if pd.Timestamp(row['date']) < pd.Timestamp(day)
             and ((pd.Timestamp(day).year - pd.Timestamp(row['date']).year) * 12
                  + pd.Timestamp(day).month - pd.Timestamp(row['date']).month) >= 6]
    if not prior:
        return {'status': 'insufficient_history', 'cutoff': str(cutoff.date())}
    base = max(prior, key=lambda row: pd.Timestamp(row['date']))
    base_day = pd.Timestamp(base['date'])
    combo_return = nav / float(base['nav']) - 1
    alloc_return = float(alloc_nav.asof(pd.Timestamp(day)) / alloc_nav.asof(base_day) - 1)
    return {'status': 'ok', 'base_date': str(base_day.date()),
            'combo_return': combo_return, 'alloc60_return': alloc_return,
            'relative_return': combo_return - alloc_return,
            'stop_check_triggered': bool(combo_return - alloc_return < -0.03)}


def write_history(path, row):
    frame = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old.date != row['date']]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.sort_values('date').to_csv(path, index=False)


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp{os.getpid()}')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float))
    os.replace(temporary, path)


@contextlib.contextmanager
def state_lock(path):
    path = Path(path)
    lock = path.with_name(path.name + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'state_locked:{lock}')
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def frozen_schedule(phase=FORWARD_PHASE, seed=FORWARD_SEED):
    """The exact POOL50_G97 target schedule used by the frozen P5_BASE runs."""
    from quantlab.alpha.portfolio import random_order, schedule
    scores = pd.read_parquet(ALPHA / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(START)]
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    rebalance = days[phase // 5::4]
    industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
    order = lambda day: random_order(list(by_day[day].code), seed)
    return schedule(rebalance, order, industries, gross=STOCK_GROSS)


def pending_orders(stock_result):
    rows = []
    for order in stock_result['orders']:
        if order.get('status') not in ('pending', 'partial') or int(order.get('remaining', 0)) <= 0:
            continue
        rows.append({'code': order['instrument_id'], 'leg': 'stock', 'side': order['side'],
                     'shares': int(order['remaining']), 'signal_date': order['signal_date'],
                     'reason': order['reason'], 'status': order['status']})
    return rows


def replay_book(as_of, capital=DEFAULT_CAPITAL, cost=1.0, schedule=None):
    """Replay both sleeves from 2018 to ``as_of`` and blend them with the frozen rule."""
    requested = pd.Timestamp(as_of).normalize()
    schedule = frozen_schedule() if schedule is None else schedule
    targets = p4_ledger.to_targets(schedule)
    codes = sorted({code for row in targets for code in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    last_complete = dataset.calendar[-1]
    identity = {'requested_as_of': str(requested.date()),
                'last_complete_session': str(last_complete.date()),
                'manifest': str(p4_ledger.book.MANIFEST.relative_to(ROOT)),
                'manifest_sha256': sha256_file(p4_ledger.book.MANIFEST),
                'scores_sha256': sha256_file(ALPHA / 'scores.parquet'),
                'schedule_last_signal': str(max(pd.Timestamp(day) for day in schedule).date()),
                'stock_capital': float(capital * EQUITY_SHARE),
                'etf_capital': float(capital * (1 - EQUITY_SHARE)),
                'stock_gross': STOCK_GROSS, 'cost_mult': float(cost)}
    if requested > last_complete:
        return dict(status='blocked', reason='as_of_beyond_complete_session', identity=identity,
                    orders=[], nav=None)
    if requested not in dataset.calendar:
        return dict(status='blocked', reason='as_of_not_a_session', identity=identity,
                    orders=[], nav=None)
    stock = p4_ledger.replay(targets, start=START, end=requested, cost_mult=cost, dataset=dataset,
                             capital_cny=capital * EQUITY_SHARE, gross=STOCK_GROSS,
                             suspension_evidence=SuspensionEvidence())
    sessions = dataset.calendar[(dataset.calendar >= pd.Timestamp(START)) &
                                (dataset.calendar <= requested)]
    etf_nav, etf_meta = B.run_g0_lot_sleeve(sessions, capital * (1 - EQUITY_SHARE), cost,
                                            'calendar', dataset.calendar)
    nav_norm, trace, blend_meta = B.run_one(stock['nav'], stock['exposure'], etf_nav,
                                            etf_meta['cash_ratio'], cost, 'calendar',
                                            dataset.calendar)
    alloc = trace['weights'].iloc[-1].to_dict()
    combined_nav = float(capital * nav_norm.iloc[-1])
    stock_sleeve_nav = float(stock['nav'].iloc[-1])
    etf_sleeve_nav = float(etf_nav.iloc[-1])
    stock_scale = alloc['equity'] * combined_nav / stock_sleeve_nav
    etf_scale = alloc['g0'] * combined_nav / etf_sleeve_nav
    account = stock['account'][-1]
    stock_holdings = {code: int(quantity) for code, quantity in account['quantities'].items()
                      if int(quantity) > 0}
    stock_marks = {code: float(account['marks'].get(code, np.nan)) for code in stock_holdings}
    prices = B.etf_prices(sessions)
    etf_mark = prices['mark'].iloc[-1]
    etf_holdings = {B.ETF_CODES[name]: int(quantity)
                    for name, quantity in etf_meta['final_shares'].items() if int(quantity) > 0}
    stock_value = sum(quantity * stock_marks[code] for code, quantity in stock_holdings.items()
                      if np.isfinite(stock_marks[code])) * stock_scale
    etf_value = sum(quantity * float(etf_mark[ETF_NAMES[code]])
                    for code, quantity in etf_holdings.items()) * etf_scale
    sleeve_cash = float(account['cash'])
    cash_value = (sleeve_cash * stock_scale + float(etf_meta['final_cash']) * etf_scale)
    orders = pending_orders(stock)
    blockers = []
    if not stock['summary']['data_quality_ok']:
        blockers.append('stock_data_quality')
    if stock['summary']['missing_panel_count']:
        blockers.append('missing_panel')
    if etf_meta['blocked']:
        blockers.append('etf_missing_on_rebalance')
    diagnostics = B.segment(nav_norm, START, str(requested.date()))
    grid = grid_report(schedule, dataset.calendar, requested, last_complete, FORWARD_PHASE)
    calendar_mismatch = grid['calendar_mismatch']
    if calendar_mismatch:
        blockers.append('calendar_mismatch')
    quarter_key = f'{requested.year}Q{(requested.month - 1) // 3 + 1}'
    return dict(status='blocked' if blockers else 'paper_simulated',
                reason=';'.join(blockers) if blockers else 'ok', identity=identity,
                orders=[] if blockers else orders,
                nav=None if blockers else combined_nav, nav_norm=nav_norm, stock=stock,
                etf_nav=etf_nav,
                etf_meta=etf_meta, combined_weights=alloc, trace=trace,
                stock_sleeve_nav=stock_sleeve_nav, etf_sleeve_nav=etf_sleeve_nav,
                stock_scale=stock_scale, etf_scale=etf_scale,
                stock_holdings=stock_holdings, stock_marks=stock_marks,
                etf_holdings=etf_holdings, etf_marks=etf_mark, sleeve_cash=sleeve_cash,
                stock_value=stock_value, etf_value=etf_value, cash_value=cash_value,
                stock_exposure=stock_value / combined_nav if combined_nav else 0.0,
                etf_exposure=etf_value / combined_nav if combined_nav else 0.0,
                cash_weight=cash_value / combined_nav if combined_nav else 0.0,
                cagr=diagnostics['cagr'], mdd=diagnostics['mdd'],
                last_signal=str(max(pd.Timestamp(day) for day in schedule).date()),
                calendar=dataset.calendar,
                last_grid=grid['last_grid'], next_grid=grid['next_grid'],
                signal_today=grid['signal_today'],
                calendar_mismatch=calendar_mismatch, quarter_basis='calendar',
                quarter_key=quarter_key,
                etf_trades_today=[trade for trade in etf_meta['trade_log']
                                  if trade['date'] == str(requested.date())],
                blend_meta=blend_meta)


def prestart_calendar(as_of, horizon_days=45):
    """Exchange calendar for the prestart preview, extended past ``as_of`` by a provisional window.

    The field panel ends before the requested snapshot date, so future decision dates are
    business-day placeholders. They only populate ``next_grid``; every prestart artifact is
    labelled ``calendar_provisional`` and none of these dates can produce orders.
    """
    trusted = resolve_trusted_panel(need_fields=('close',))
    panel = trusted['path'] / 'close.parquet'
    calendar = pd.DatetimeIndex(pd.read_parquet(panel, columns=[]).index)
    source = str(panel.relative_to(ROOT)) if panel.is_relative_to(ROOT) else str(panel)
    requested = pd.Timestamp(as_of).normalize()
    horizon = max(calendar[-1], requested + pd.Timedelta(days=horizon_days))
    provisional = horizon > calendar[-1]
    if provisional:
        calendar = calendar.append(pd.bdate_range(calendar[-1] + pd.Timedelta(days=1), horizon))
    return calendar, source, provisional


def run_prestart(as_of):
    requested = pd.Timestamp(as_of).normalize()
    calendar, source, provisional = prestart_calendar(requested)
    production_last = p4_ledger.load_dataset([]).calendar[-1]
    phases = {}
    for phase in PHASES:
        grid = decision_grid(calendar, START, calendar[-1], phase, 20)
        past = grid[grid <= requested]
        future = grid[grid > requested]
        phases[str(phase)] = {'last_grid': str(past[-1].date()) if len(past) else None,
                              'next_grid': str(future[0].date()) if len(future) else None,
                              'is_grid_day': bool(len(past) and past[-1] == requested)}
    grid_today = any(row['is_grid_day'] for row in phases.values())
    reason = ('signal_day_without_complete_data' if grid_today else 'no_new_signal')
    snapshot = {'strategy': 'P5_BASE', 'mode': 'prestart', 'requested_as_of': str(requested.date()),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'calendar_source': source, 'calendar_provisional': provisional,
                'production_last_complete_session': str(production_last.date()),
                'data_status': 'available' if requested <= production_last else 'not_available',
                'phases': phases, 'orders': [], 'reason': reason,
                'status': 'prestart', 'promotion_approved': False}
    target = OUT / 'prestart' / str(requested.date())
    atomic_write(target / 'snapshot.json', snapshot)
    print(json.dumps({'snapshot': str(target / 'snapshot.json'), 'reason': reason,
                      'calendar_provisional': provisional, 'phases': phases,
                      'promotion_approved': False}, ensure_ascii=False, default=float))
    return snapshot


def run_parity(capital=DEFAULT_CAPITAL, cost=1.0):
    """Compare the forward engine with the pinned p0_s1 books under both quarter-end bases.

    ``_r1`` is the calendar-basis re-run (B1); ``20261019`` is the legacy window-basis book kept as a
    historical reference. Both are the same ledger, so this is an accounting check, not independence.
    """
    as_of = '2026-09-22'
    book = replay_book(as_of, capital, cost)
    frozen = pd.read_csv(ROOT / 'output/p5_base_lot_20261018/runs' /
                         f'cap{int(capital)}_cost{int(cost)}.csv')
    row = frozen[frozen.path == 'POOL50_G97_p0_s1'].iloc[0]
    calendar_path = (ROOT / 'output/p5_base_lot_20261018_r1/nav' /
                     f'cap{int(capital)}_cost{int(cost)}_POOL50_G97_p0_s1.csv')
    legacy_path = (ROOT / 'output/p5_base_lot_20261019/nav' /
                   f'cap{int(capital)}_cost{int(cost)}_POOL50_G97_p0_s1.csv')
    reference = pd.read_csv(calendar_path, index_col=0, parse_dates=True)
    legacy = pd.read_csv(legacy_path, index_col=0, parse_dates=True)
    legacy_runs = pd.read_csv(ROOT / 'output/p5_base_lot_20261019/runs' /
                              f'cap{int(capital)}_cost{int(cost)}.csv')
    legacy_row = legacy_runs[legacy_runs.path == 'POOL50_G97_p0_s1'].iloc[0]
    reference_nav = float(reference.loc[pd.Timestamp(as_of), 'blended'])
    legacy_nav = float(legacy.loc[pd.Timestamp(as_of), 'blended'])
    delta_cagr = float(book['cagr'] - row.blended_cagr)
    delta_mdd = float(book['mdd'] - row.blended_mdd)
    delta_nav = float(book['nav'] / (capital * reference_nav) - 1)
    within = bool(abs(delta_cagr) <= 0.002 and abs(delta_nav) <= 0.002
                  and abs(delta_mdd) <= 0.002)
    report = {'as_of': as_of, 'capital': capital, 'cost': cost,
              'quarter_basis': book['quarter_basis'],
              'replayed': {'cagr': book['cagr'], 'mdd': book['mdd'], 'nav': book['nav'],
                           'n_sessions': int(len(book['nav_norm']))},
              'frozen': {'cagr': float(row.blended_cagr), 'mdd': float(row.blended_mdd),
                         'nav_reference': capital * reference_nav,
                         'reference_path': str(calendar_path.relative_to(ROOT))},
              'legacy_window_reference': {
                  'cagr': float(legacy_row.blended_cagr), 'mdd': float(legacy_row.blended_mdd),
                  'nav': capital * legacy_nav,
                  'quarter_basis': 'window',
                  'reference_path': str(legacy_path.relative_to(ROOT))},
              'delta': {'cagr_pp': delta_cagr * 100, 'mdd_pp': delta_mdd * 100,
                        'nav_relative': delta_nav},
              'tolerance': {'cagr_pp': 0.20, 'mdd_pp': 0.20, 'nav_relative': 0.002},
              'within_tolerance': within, 'not_independent_validation': True,
              'manifest_sha256': book['identity']['manifest_sha256'],
              'scores_sha256': book['identity']['scores_sha256'],
              'promotion_approved': False}
    target = OUT / 'parity' / as_of
    atomic_write(target / 'snapshot.json', report)
    print(json.dumps(report, ensure_ascii=False, default=float))
    return report


def run(args):
    capital = float(args.capital)
    if args.parity:
        run_parity(capital, float(args.cost))
        return
    if args.prestart:
        run_prestart(args.prestart)
        return
    if not args.as_of:
        raise SystemExit('--as-of is required unless --prestart or --parity is used')
    status = 'broker_confirmed' if args.broker_confirmed else 'paper_simulated'
    if args.broker_confirmed and (not args.holdings or args.cash is None):
        raise SystemExit('broker_confirmed requires --holdings and --cash')
    state_path = Path(args.state) if args.state else (BROKER_STATE if args.broker_confirmed else PAPER_STATE)
    requested = pd.Timestamp(args.as_of).normalize()
    book = replay_book(requested, capital, float(args.cost))
    last_complete = pd.Timestamp(book['identity']['last_complete_session'])
    if book['status'] == 'blocked' and book['reason'] in ('as_of_beyond_complete_session',
                                                         'as_of_not_a_session'):
        snapshot = {'strategy': 'P5_BASE', 'status': 'blocked', 'reason': book['reason'],
                    'requested_as_of': str(requested.date()), 'as_of': None,
                    'last_complete_session': str(last_complete.date()),
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'orders': [], 'nav': None, 'promotion_approved': False,
                    'identity': book['identity']}
        target = OUT / (('dryrun/' if args.dry_run else '') + str(requested.date()))
        atomic_write(target / 'snapshot.json', snapshot)
        print(json.dumps({'snapshot': str(target / 'snapshot.json'), 'status': 'blocked',
                          'reason': book['reason'], 'orders': 0,
                          'promotion_approved': False}, ensure_ascii=False, default=float))
        return
    day = requested
    state = (broker_state(Path(args.holdings), args.cash) if args.broker_confirmed
             else load_state(state_path, capital, status))
    # a blocked run may not carry orders even if the replay handed some back
    orders = [] if book['status'] == 'blocked' else book['orders']
    if args.no_orders:
        orders = []
    execution_day = next_session(book['calendar'], day)
    if execution_day is None:
        execution_day = pd.bdate_range(day + pd.Timedelta(days=1), periods=1)[0]
        snapshot_calendar = 'provisional_next_business_day'
    else:
        snapshot_calendar = 'production_calendar'
    alloc_nav, alloc_hash, alloc_config = alloc60_series(book['etf_nav'], book['calendar'])
    if book['nav'] is None:
        check = {'status': 'blocked_with_the_book', 'stop_check_triggered': False}
    else:
        check = six_month_check(state.get('history', []), alloc_nav, day, book['nav'])
    fills_today = [fill for fill in book['stock']['fills'] if fill.get('date') == str(day.date())]
    weights = {}
    for code, quantity in book['stock_holdings'].items():
        value = quantity * book['stock_marks'][code] * book['stock_scale']
        weights[code] = value / book['nav'] if book['nav'] else 0.0
    for code, quantity in book['etf_holdings'].items():
        value = quantity * float(book['etf_marks'][ETF_NAMES[code]]) * book['etf_scale']
        weights[code] = value / book['nav'] if book['nav'] else 0.0
    top10 = [{'code': code, 'weight': round(weight, 6)}
             for code, weight in sorted(weights.items(), key=lambda item: -item[1])[:10]]
    snapshot = {
        'strategy': 'P5_BASE', 'status': book['status'] if book['status'] == 'blocked' else status,
        'reason': book['reason'],
        'as_of': str(day.date()), 'requested_as_of': str(requested.date()),
        'last_complete_session': str(last_complete.date()),
        'signal_date': book['last_signal'], 'last_grid': book['last_grid'],
        'next_grid': book['next_grid'], 'signal_today': book['signal_today'],
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'capital': capital, 'nav': book['nav'],
        'sleeves': {'stock': {'nav': book['stock_sleeve_nav'], 'cash': book['sleeve_cash'],
                              'scale': book['stock_scale'], 'holdings': book['stock_holdings']},
                    'etf': {'nav': book['etf_sleeve_nav'], 'cash': book['etf_meta']['final_cash'],
                            'scale': book['etf_scale'], 'holdings': book['etf_holdings']}},
        'allocation': book['combined_weights'],
        'stock_value': book['stock_value'], 'etf_value': book['etf_value'],
        'stock_exposure': book['stock_exposure'], 'etf_exposure': book['etf_exposure'],
        'cash_weight': book['cash_weight'], 'stock_names': len(book['stock_holdings']),
        'weights': weights, 'top10': top10,
        'filled_now': fills_today, 'etf_trades_now': book['etf_trades_today'],
        'orders': orders, 'pending_execute_on': str(execution_day.date()) if orders else None,
        'execution_calendar': snapshot_calendar,
        'broker': ({'holdings': state['holdings'], 'cash': state['cash']}
                   if args.broker_confirmed else None),
        'rebalance': {'stocks': bool(book['signal_today'] and orders),
                      'etf': bool(book['etf_trades_today'])},
        'data_quality': {'stock_ok': book['stock']['summary']['data_quality_ok'],
                         'missing_panel_count': book['stock']['summary']['missing_panel_count'],
                         'suspension_coverage_complete':
                             book['stock']['summary']['suspension_coverage_complete'],
                         'unexplained_stale_held_max_days':
                             book['stock']['summary']['unexplained_stale_held_max_days'],
                         'unsettled_delisting':
                             book['stock']['summary']['unsettled_delisting'],
                         'stale_held_max_days': book['stock']['summary']['stale_held_max_days'],
                         'suspended_held_max_days':
                             book['stock']['summary']['suspended_held_max_days'],
                         'raw_uncovered_stale_codes':
                             book['stock']['summary']['raw_uncovered_stale_codes'],
                         'etf_blocked': book['etf_meta']['blocked'],
                         'etf_missing_on_rebalance': book['etf_meta']['missing_on_rebalance'],
                         'etf_price_missing_days': book['etf_meta']['price_missing_days'],
                         'mark_basis': 'production_ledger_hfq',
                         'etf_distributions_credited': False},
        'diagnostics': {'cagr': book['cagr'], 'mdd': book['mdd'],
                        'stock_cagr': book['stock']['summary']['cagr'],
                        'stock_mdd': book['stock']['summary']['max_drawdown'],
                        'etf_cagr': B.segment(book['etf_nav'], START, str(day.date()))['cagr'],
                        'etf_mdd': B.segment(book['etf_nav'], START, str(day.date()))['mdd'],
                        'blend': book['blend_meta']},
        'identity': {**book['identity'], 'book_config_hash': hashlib.sha256(
            json.dumps(book['identity'], sort_keys=True).encode()).hexdigest()},
        'alloc60': {'config_hash': alloc_hash, 'config': alloc_config, 'six_month': check},
        'state_path': str(state_path),
        'promotion_approved': False,
    }
    target = OUT / (('dryrun/' if args.dry_run else '') + str(day.date()))
    atomic_write(target / 'snapshot.json', snapshot)
    if not args.dry_run and book['status'] != 'blocked':
        with state_lock(state_path):
            history = [row for row in state.get('history', []) if row['date'] != str(day.date())]
            history.append({'date': str(day.date()), 'nav': book['nav'],
                            'stock_exposure': book['stock_exposure'],
                            'etf_exposure': book['etf_exposure'],
                            'cash_weight': book['cash_weight'], 'status': status})
            book_reference = {
                'nav': book['nav'], 'allocation': book['combined_weights'],
                'stock': {'nav': book['stock_sleeve_nav'], 'scale': book['stock_scale'],
                          'cash': book['sleeve_cash'], 'holdings': book['stock_holdings']},
                'etf': {'nav': book['etf_sleeve_nav'], 'scale': book['etf_scale'],
                        'cash': book['etf_meta']['final_cash'], 'holdings': book['etf_holdings']},
                'stock_value': book['stock_value'], 'etf_value': book['etf_value'],
                'cash_value': book['cash_value']}
            new_state = dict(state, as_of=str(day.date()), status=status,
                             last_signal=book['last_signal'], last_etf_quarter=book['quarter_key'],
                             book=book_reference, pending_orders=[], pending_execute_on=None,
                             book_config_hash=snapshot['identity']['book_config_hash'],
                             history=history)
            if not args.broker_confirmed:
                # Paper state has no imported broker account; the replay book is authoritative.
                new_state['cash'], new_state['holdings'] = None, {}
            atomic_write(state_path, new_state)
            write_history(OUT / 'history.csv', history[-1])
    print(json.dumps({'snapshot': str(target / 'snapshot.json'), 'nav': book['nav'],
                      'orders': len(orders), 'fills': len(fills_today),
                      'status': book['status'], 'reason': book['reason'],
                      'promotion_approved': False}, ensure_ascii=False, default=float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--as-of')
    parser.add_argument('--capital', type=float, default=DEFAULT_CAPITAL)
    parser.add_argument('--cost', type=float, default=1.0)
    parser.add_argument('--state', default=None)
    parser.add_argument('--holdings')
    parser.add_argument('--cash', type=float)
    parser.add_argument('--broker-confirmed', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-orders', action='store_true')
    parser.add_argument('--prestart', help='calendar-only prestart check, writes no state')
    parser.add_argument('--parity', action='store_true', help='replay 2026-09-22 against the frozen p0_s1')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
