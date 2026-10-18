#!/usr/bin/env python3
"""P5_BASE lot-aware diagnostic: stock and G0 sleeves are actually funded at 65/35.

The stock sleeve runs the frozen G97 targets through ``stable_book`` at 65% of the
account capital (board lots, T+1, limits, suspensions, fees, dividends). The ETF
sleeve is a 100-share-lot G0 ledger at 35%. The two sleeve NAVs are then combined
with the same quarterly allocation rule. Cross-sleeve transfer remains fractional
and is reported as an upper-bound assumption; every trade inside a sleeve is lot
constrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import p4_alloc as R
import p4_gross as G
from quantlab import p4_alloc as A
from quantlab import p4_ledger
from quantlab.data.suspensions import SuspensionEvidence

START, END = '2018-01-02', '2026-09-22'
EQUITY_SHARE = 0.65
STOCK_GROSS = 0.97
ETF_COST = 0.0005
ETF_CODES = {'gold': 'sh518880', 'treasury': 'sh511260', 'nasdaq': 'sh513100'}
OUT = ROOT / 'output/p5_base_lot_20261019'
SUSPENSION_PREREG = ROOT / 'docs/prereg/P5_DATA_QUALITY_SUSPENSION.md'


def g0_target(etf_returns_upto, share):
    sleeve = 1.0 - share
    return np.array([share, 0.375 * sleeve, 0.375 * sleeve, 0.25 * sleeve])


def etf_prices(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide[list(ETF_CODES.values())].rename(columns={v: k for k, v in ETF_CODES.items()})
    raw = wide.reindex(sessions)
    mark = raw.ffill()
    missing = raw.isna()
    missing_dates = [dict(code=ETF_CODES[name], name=name, date=str(pd.Timestamp(day).date()))
                     for day in sessions for name in raw.columns if bool(missing.at[day, name])]
    return dict(raw=raw, mark=mark, missing=missing,
                missing_days=int(missing.any(axis=1).sum()), missing_dates=missing_dates)


def run_g0_lot_sleeve(sessions, capital, cost_mult=1.0, quarter_basis='window', calendar=None):
    """Quarterly G0 sleeve with 100-share lots, cash residual, and 1:1 QDII overflow.

    Missing ETF prices never fill an execution. A rebalance day with a missing
    bar is blocked and recorded; other missing days mark at the last close.
    ``quarter_basis='calendar'`` pins the quarter ends to ``calendar`` so a window that stops
    mid-quarter does not trade on its own final day.
    """
    prices = etf_prices(sessions)
    raw, mark = prices['raw'], prices['mark']
    premium = R.qdii_premium_frame(sessions, basis='adjusted')
    ends = set(A.quarter_ends(sessions, quarter_basis, calendar))
    names = ['gold', 'treasury', 'nasdaq']
    cash = float(capital)
    shares = {name: 0 for name in names}
    navs, cash_ratios, trades = [], [], []
    blocked_dates = []
    for day in sessions:
        px = mark.loc[day]
        nav = cash + sum(shares[name] * px[name] for name in names if np.isfinite(px[name]))
        if day in ends and nav > 0:
            if not np.isfinite(raw.loc[day]).all():
                blocked_dates.append(str(pd.Timestamp(day).date()))
            else:
                px_exec = raw.loc[day]
                current = np.array([shares[name] * px_exec[name] / nav for name in names])
                target = np.array([0.375, 0.375, 0.25])
                if day in premium.index:
                    prem = premium.loc[day].to_dict()
                    value = prem.get('nasdaq')
                    if value is not None and np.isfinite(value) and value > 0.03 and target[2] > current[2]:
                        freed = target[2] - current[2]
                        target[2] = current[2]
                        target[0] += freed / 2
                        target[1] += freed / 2
                target_value = target * nav
                desired = {name: int(np.floor(target_value[i] / (px_exec[name] * 100))) * 100
                           for i, name in enumerate(names)}
                # Sells first, then buys; every order is a whole board lot.
                for name in names:
                    delta = desired[name] - shares[name]
                    if delta < 0:
                        quantity = -delta
                        value = quantity * px_exec[name]
                        fee = value * ETF_COST * cost_mult
                        cash += value - fee
                        shares[name] -= quantity
                        trades.append(dict(date=str(day.date()), code=ETF_CODES[name], side='sell',
                                           shares=quantity, value=round(value, 2), fee=round(fee, 2)))
                for name in names:
                    delta = desired[name] - shares[name]
                    if delta > 0:
                        price = px_exec[name]
                        lot_cost = price * 100
                        fee_rate = ETF_COST * cost_mult
                        affordable = int(np.floor(max(0.0, cash) / (lot_cost * (1 + fee_rate)))) * 100
                        quantity = min(delta, affordable)
                        if quantity > 0:
                            value = quantity * price
                            fee = value * fee_rate
                            cash -= value + fee
                            shares[name] += quantity
                            trades.append(dict(date=str(day.date()), code=ETF_CODES[name], side='buy',
                                               shares=quantity, value=round(value, 2), fee=round(fee, 2)))
                nav = cash + sum(shares[name] * px[name] for name in names)
        navs.append(nav)
        cash_ratios.append(cash / nav if nav else 0.0)
    missing_days = [pd.Timestamp(day) for day in sessions if bool(prices['missing'].loc[day].any())]
    stale_mark_days = [str(day.date()) for day in missing_days if day not in ends]
    nav_series = pd.Series(navs, index=sessions)
    cash_series = pd.Series(cash_ratios, index=sessions)
    return nav_series, dict(cash_ratio_mean=float(cash_series.mean()), cash_ratio=cash_series,
                            trades=len(trades), final_cash=float(cash),
                            final_shares={k: int(v) for k, v in shares.items()}, trade_log=trades,
                            price_missing_days=prices['missing_days'],
                            price_missing_dates=prices['missing_dates'],
                            stale_mark_days=stale_mark_days,
                            blocked=bool(blocked_dates), blocked_dates=blocked_dates,
                            missing_on_rebalance=bool(blocked_dates),
                            premium_basis=premium.attrs.get('basis'),
                            premium_masked=premium.attrs.get('masked', {}),
                            premium_fallback_days=premium.attrs.get('fallback_days', {}))


def run_g0_fractional_sleeve(sessions, capital, cost_mult=1.0, basis='legacy',
                             quarter_basis='window', calendar=None):
    """Same G0 rule without lot rounding: the nominal reference for the ETF sleeve."""
    prices = etf_prices(sessions)
    raw, mark = prices['raw'], prices['mark']
    premium = R.qdii_premium_frame(sessions, basis=basis)
    ends = set(A.quarter_ends(sessions, quarter_basis, calendar))
    values = np.array([0.375, 0.375, 0.25]) * float(capital)
    previous = None
    navs = []
    for day in sessions:
        px = mark.loc[day].to_numpy(float)
        if previous is not None:
            values = values * px / previous
        previous = px
        nav = float(values.sum())
        if day in ends and nav > 0 and np.isfinite(raw.loc[day]).all():
            current = values / nav
            target = np.array([0.375, 0.375, 0.25])
            if day in premium.index:
                value = premium.loc[day].to_dict().get('nasdaq')
                if value is not None and np.isfinite(value) and value > 0.03 and target[2] > current[2]:
                    freed = target[2] - current[2]
                    target[2] = current[2]
                    target[0] += freed / 2
                    target[1] += freed / 2
            new = target * nav
            turnover = float(np.abs(new - values).sum())
            values = new - turnover * ETF_COST * cost_mult
            nav = float(values.sum())
        navs.append(nav)
    return pd.Series(navs, index=sessions)


def segment(nav, start, end):
    seg = nav.loc[start:end]
    if len(seg) < 2:
        return dict(cagr=float('nan'), mdd=float('nan'), n=len(seg))
    years = (seg.index[-1] - seg.index[0]).days / 365.25
    return dict(cagr=float((seg.iloc[-1] / seg.iloc[0]) ** (1 / years) - 1),
                mdd=float((seg / seg.cummax() - 1).min()), n=len(seg))


def g0_sleeve_target(etf_returns_upto, share):
    return np.array([share, 1.0 - share])


def run_one(stock_nav, stock_exposure, etf_nav, etf_cash_ratio, cost,
            quarter_basis='window', calendar=None):
    """Blend two sleeve NAVs; the G0 sleeve is one asset because its own lot ledger already
    applies the 3:3:2 split, quarterly trades, fees and the QDII overflow."""
    sessions = stock_nav.index
    etf_returns = pd.DataFrame({'g0': etf_nav.pct_change(fill_method=None).fillna(0.0)})
    saved = (A.DISASTER_DD, A.RECOVER_DD, A.DISASTER_EQUITY)
    A.DISASTER_DD, A.RECOVER_DD, A.DISASTER_EQUITY = 0.23, 0.08, 0.50
    try:
        nav, trace = A.allocate(
            stock_nav, etf_returns, EQUITY_SHARE, cost_mult=cost, rebalance='quarter',
            target_fn=g0_sleeve_target, basis=quarter_basis, calendar=calendar)
    finally:
        A.DISASTER_DD, A.RECOVER_DD, A.DISASTER_EQUITY = saved
    weights = trace['weights']
    held_stock = weights['equity'] * stock_exposure.reindex(sessions).fillna(0.0)
    sleeve_cash = etf_cash_ratio.reindex(sessions).fillna(1.0)
    held_etf = weights['g0'] * (1 - sleeve_cash)
    return nav, trace, dict(stock_exposure_mean=float(held_stock.mean()),
                            etf_exposure_mean=float(held_etf.mean()),
                            cash_mean=float((1 - held_stock - held_etf).mean()))


def run_config(capital, cost, plans, targets, dataset, sessions, out, quarter_basis='window'):
    rows = []
    run_path = out / 'runs' / f'cap{int(capital)}_cost{int(cost)}.csv'
    nav_path = out / 'nav'
    run_path.parent.mkdir(parents=True, exist_ok=True)
    nav_path.mkdir(parents=True, exist_ok=True)
    stock_capital, etf_capital = capital * EQUITY_SHARE, capital * (1 - EQUITY_SHARE)
    etf_nav, etf_meta = run_g0_lot_sleeve(sessions, etf_capital, cost, quarter_basis,
                                          dataset.calendar)
    evidence = SuspensionEvidence()
    for name in sorted(targets):
        stock = p4_ledger.replay(targets[name], start=START, end=END, cost_mult=cost,
                                 dataset=dataset, capital_cny=stock_capital, gross=STOCK_GROSS,
                                 suspension_evidence=evidence)
        stock_nav = stock['nav']
        nav, trace, blend_meta = run_one(stock_nav, stock['exposure'], etf_nav,
                                         etf_meta['cash_ratio'], cost, quarter_basis,
                                         dataset.calendar)
        pd.DataFrame({'blended': nav, 'stock': stock_nav.reindex(sessions),
                      'etf': etf_nav.reindex(sessions)}).to_csv(
            nav_path / f'cap{int(capital)}_cost{int(cost)}_{name}.csv')
        summary = stock['summary']
        row = dict(path=name, capital=capital, cost=cost, quarter_basis=quarter_basis,
                   stock_cagr=summary['cagr'], stock_mdd=summary['max_drawdown'],
                   stock_mean_exposure=summary['mean_exposure'],
                   stock_min_exposure=summary['min_exposure'],
                   stock_max_exposure=summary['max_exposure'],
                   stock_mean_cash=summary['mean_cash_weight'],
                   stock_max_holdings=summary['max_holdings'],
                   stock_fills=summary['fills'],
                   stock_data_issues=summary['data_issues'],
                   stock_data_quality_ok=summary['data_quality_ok'],
                   stock_delisting_liquidations=summary['delisting_liquidations'],
                   stock_missing_panel_count=summary['missing_panel_count'],
                   stock_stale_held_max_days=summary['stale_held_max_days'],
                   stock_suspended_held_max_days=summary['suspended_held_max_days'],
                   stock_suspended_held_codes=json.dumps(summary['suspended_held_codes'], sort_keys=True),
                   stock_unexplained_stale_held_max_days=summary['unexplained_stale_held_max_days'],
                   stock_unexplained_stale_held_codes=json.dumps(summary['unexplained_stale_held_codes'],
                                                                 sort_keys=True),
                   stock_unexplained_runs=json.dumps(summary['unexplained_runs'], sort_keys=True),
                   stock_suspension_coverage_complete=summary['suspension_coverage_complete'],
                   stock_stale_sessions_total=summary['stale_sessions_total'],
                   stock_stale_sessions_covered=summary['stale_sessions_covered'],
                   stock_raw_uncovered_stale_codes=json.dumps(summary['raw_uncovered_stale_codes']),
                   stock_data_issues_by_reason=json.dumps(summary['data_issues_by_reason'],
                                                         ensure_ascii=False, sort_keys=True),
                   etf_cagr=segment(etf_nav, START, END)['cagr'],
                   etf_mdd=segment(etf_nav, START, END)['mdd'],
                   etf_cash_mean=etf_meta['cash_ratio_mean'],
                   etf_trades=etf_meta['trades'],
                   etf_price_missing_days=etf_meta['price_missing_days'],
                   etf_price_missing_dates=json.dumps(etf_meta['price_missing_dates'], sort_keys=True),
                   etf_stale_mark_days=json.dumps(etf_meta['stale_mark_days']),
                   etf_missing_on_rebalance=etf_meta['missing_on_rebalance'],
                   etf_blocked=etf_meta['blocked'],
                   etf_blocked_dates=json.dumps(etf_meta['blocked_dates']),
                   premium_basis=etf_meta['premium_basis'],
                   premium_masked=json.dumps(etf_meta['premium_masked'], sort_keys=True),
                   blended_cagr=segment(nav, START, END)['cagr'],
                   blended_mdd=segment(nav, START, END)['mdd'],
                   ex2025_cagr=segment(nav, START, '2024-12-31')['cagr'],
                   ex2025_mdd=segment(nav, START, '2024-12-31')['mdd'],
                   y2025_cagr=segment(nav, '2025-01-01', END)['cagr'],
                   disaster_triggers=trace['disaster_triggers'],
                   **blend_meta)
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_path, index=False)
        print(json.dumps(row, ensure_ascii=False, default=float), flush=True)
    return rows


def frozen_nominal():
    path = ROOT / 'output/p5_g0_20261017/report.json'
    cell = json.loads(path.read_text())['results']['G97_65_23_q']
    return dict(source=str(path.relative_to(ROOT)), cagr=cell['cagr'], mdd=cell['mdd'],
                cagr_with_ipo=cell['cagr_with_ipo'], cost2_cagr=cell['cost2']['cagr'],
                cost2_mdd=cell['cost2']['mdd'])


def frozen_stock(cost):
    path = ROOT / 'output/p4_gross_20261013/ledger' / f'summary_cost{cost:g}.csv'
    frame = pd.read_csv(path)
    frame = frame[frame.path.str.startswith('POOL50_G97_')]
    return dict(source=str(path.relative_to(ROOT)), cagr=float(frame.cagr.median()),
                mdd=float(frame.mdd.median()))


def sleeve_references():
    """Fractional G0 sleeve with legacy and adjusted QDII premium, same sessions and costs."""
    dataset = p4_ledger.load_dataset([])
    sessions = dataset.calendar[(dataset.calendar >= pd.Timestamp(START)) &
                                (dataset.calendar <= pd.Timestamp(END))]
    out = {}
    for cost in (1.0, 2.0):
        row = {}
        for basis in ('legacy', 'adjusted'):
            nav = run_g0_fractional_sleeve(sessions, 1.0, cost, basis)
            row[basis] = segment(nav, START, END)
        out[str(int(cost))] = row
    return out


def assemble(out):
    files = sorted((out / 'runs').glob('cap*_cost*.csv'))
    if len(files) != 4:
        raise SystemExit(f'need 4 run files, found {[str(f.name) for f in files]}')
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame = frame.sort_values(['capital', 'cost', 'path']).reset_index(drop=True)
    frame.to_csv(out / 'summary.csv', index=False)
    metrics = ('blended_cagr', 'blended_mdd', 'ex2025_cagr', 'ex2025_mdd', 'y2025_cagr',
               'stock_cagr', 'stock_mdd', 'stock_mean_exposure', 'stock_min_exposure',
               'stock_max_exposure', 'stock_mean_cash', 'stock_max_holdings', 'stock_fills',
               'etf_cagr', 'etf_mdd', 'etf_cash_mean', 'etf_trades',
               'stock_exposure_mean', 'etf_exposure_mean', 'cash_mean')

    def merge_code_days(values):
        merged = {}
        for value in values:
            for code, days in json.loads(value).items():
                merged[code] = max(merged.get(code, 0), int(days))
        return merged

    def merge_lists(values):
        return sorted({item for value in values for item in json.loads(value)})

    def merge_records(values):
        merged = {(item['code'], item['date']): item
                  for value in values for item in json.loads(value)}
        return [merged[key] for key in sorted(merged)]

    cells, quality = {}, {}
    for capital in sorted(frame.capital.unique()):
        capital_frame = frame[frame.capital == capital]
        cells[str(int(capital))] = {}
        for cost in sorted(frame.cost.unique()):
            sub = capital_frame[capital_frame.cost == cost]
            cells[str(int(capital))][str(int(cost))] = {
                **{key: float(sub[key].median()) for key in metrics},
                'paths': int(len(sub)),
                'blended_mdd_worst_path': float(sub.blended_mdd.min()),
                'worst_path_ok': bool(sub.blended_mdd.min() >= -0.25),
                'paths_mdd_below_-25': int((sub.blended_mdd < -0.25).sum()),
                'stock_data_quality_ok': bool(sub.stock_data_quality_ok.all()),
                'stock_delisting_liquidations': int(sub.stock_delisting_liquidations.sum()),
                'stock_missing_panel_count': int(sub.stock_missing_panel_count.max()),
                'stock_stale_held_max_days': int(sub.stock_stale_held_max_days.max()),
                'stock_suspended_held_max_days': int(sub.stock_suspended_held_max_days.max()),
                'stock_suspended_held_codes': merge_code_days(sub.stock_suspended_held_codes),
                'stock_unexplained_stale_held_max_days':
                    int(sub.stock_unexplained_stale_held_max_days.max()),
                'stock_unexplained_stale_held_codes':
                    merge_code_days(sub.stock_unexplained_stale_held_codes),
                'stock_suspension_coverage_complete':
                    bool(sub.stock_suspension_coverage_complete.all()),
                'stock_stale_sessions_total': int(sub.stock_stale_sessions_total.sum()),
                'stock_stale_sessions_covered': int(sub.stock_stale_sessions_covered.sum()),
                'stock_raw_uncovered_stale_codes':
                    merge_lists(sub.stock_raw_uncovered_stale_codes),
                'stock_unexplained_runs': [run for value in sub.stock_unexplained_runs
                                           for run in json.loads(value)],
                'stock_data_issues': int(sub.stock_data_issues.sum()),
                'etf_price_missing_days': int(sub.etf_price_missing_days.max()),
                'etf_price_missing_dates': merge_records(sub.etf_price_missing_dates),
                'etf_stale_mark_days': merge_lists(sub.etf_stale_mark_days),
                'etf_missing_on_rebalance': bool(sub.etf_missing_on_rebalance.any()),
                'etf_blocked': bool(sub.etf_blocked.any()),
                'etf_blocked_dates': merge_lists(sub.etf_blocked_dates),
            }
        quality[str(int(capital))] = {
            'data_quality_ok': bool(capital_frame.stock_data_quality_ok.all()),
            'suspension_coverage_complete':
                bool(capital_frame.stock_suspension_coverage_complete.all()),
            'suspended_held_max_days': int(capital_frame.stock_suspended_held_max_days.max()),
            'unexplained_stale_held_max_days':
                int(capital_frame.stock_unexplained_stale_held_max_days.max()),
            'raw_uncovered_stale_codes':
                merge_lists(capital_frame.stock_raw_uncovered_stale_codes),
            'etf_blocked': bool(capital_frame.etf_blocked.any()),
            'etf_missing_on_rebalance': bool(capital_frame.etf_missing_on_rebalance.any()),
            'etf_price_missing_days': int(capital_frame.etf_price_missing_days.max()),
            'etf_price_missing_dates': merge_records(capital_frame.etf_price_missing_dates),
            'etf_stale_mark_days': merge_lists(capital_frame.etf_stale_mark_days),
            'premium_basis': sorted(capital_frame.premium_basis.unique().tolist()),
            'stock_data_issues_by_reason': {
                reason: sum(json.loads(value).get(reason, 0)
                            for value in capital_frame.stock_data_issues_by_reason)
                for reason in sorted({key for value in capital_frame.stock_data_issues_by_reason
                                      for key in json.loads(value)})},
        }
    nominal = frozen_nominal()
    refs = sleeve_references()
    frozen = {1.0: frozen_stock(1.0), 2.0: frozen_stock(2.0)}
    verdicts = {}
    for capital, by_cost in cells.items():
        c1, c2 = by_cost['1'], by_cost['2']
        gates_prereg = {'cost1_cagr_plus_ipo_ge_13': c1['blended_cagr'] + 0.015 >= 0.13,
                        'mdd_ge_-25': c1['blended_mdd'] >= -0.25,
                        'cost2_cagr_plus_ipo_ge_11': c2['blended_cagr'] + 0.015 >= 0.11}
        gates_strict = {'cost1_cagr_ge_13_no_ipo': c1['blended_cagr'] >= 0.13,
                        'mdd_ge_-25': c1['blended_mdd'] >= -0.25,
                        'cost2_cagr_ge_11_no_ipo': c2['blended_cagr'] >= 0.11}
        worst_path_ok = bool(c1['blended_mdd_worst_path'] >= -0.25
                             and c2['blended_mdd_worst_path'] >= -0.25)
        data_ok = bool(quality[capital]['data_quality_ok'] and not quality[capital]['etf_blocked'])
        arithmetic_pass = all(gates_strict.values())
        gates_all_pass = bool(data_ok and arithmetic_pass and worst_path_ok)
        reasons = []
        if not quality[capital]['data_quality_ok']:
            reasons.append('stock_data_quality_gate_failed')
            if quality[capital]['unexplained_stale_held_max_days'] > 5:
                reasons.append('unexplained_stale_gt_5')
            if not quality[capital]['suspension_coverage_complete']:
                reasons.append('suspension_coverage_incomplete')
            if quality[capital]['raw_uncovered_stale_codes']:
                reasons.append('raw_uncovered_stale_codes')
        if quality[capital]['etf_blocked']:
            reasons.append('etf_missing_on_rebalance')
        status = 'no_improvement' if data_ok else 'insufficient_data'
        if not data_ok:
            detail = 'data gate failed: ' + ';'.join(reasons)
        elif not arithmetic_pass:
            detail = 'arithmetic gates failed; worst_path_ok=' + str(worst_path_ok)
        elif not worst_path_ok:
            detail = 'arithmetic gates passed but at least one path breached -25%'
        else:
            detail = ('all arithmetic/data/worst-path gates passed, but P5_BASE is a '
                      'diagnostic and cannot be promoted')
        verdicts[capital] = {
            'gates_prereg_with_ipo': gates_prereg,
            'gates_strict_no_ipo': gates_strict,
            'worst_path_ok': worst_path_ok,
            'worst_path_mdd': {'cost1': c1['blended_mdd_worst_path'],
                               'cost2': c2['blended_mdd_worst_path']},
            'gates_all_pass': gates_all_pass,
            'prereg_gates_arithmetic_pass': bool(all(gates_prereg.values())),
            'strict_gates_arithmetic_pass': bool(arithmetic_pass),
            'prereg_all_passed': False,
            'data_ok': data_ok,
            'data_quality_reason': 'ok' if data_ok else ';'.join(reasons),
            'candidate_for_p1': False,
            'status': status,
            'status_detail': detail,
            'delta_vs_nominal': {'cost1_cagr': c1['blended_cagr'] - nominal['cagr'],
                                 'cost1_mdd': c1['blended_mdd'] - nominal['mdd'],
                                 'cost2_cagr': c2['blended_cagr'] - nominal['cost2_cagr'],
                                 'cost2_mdd': c2['blended_mdd'] - nominal['cost2_mdd']},
            'bridge': {},
        }
        for cost, cell, nominal_key in ((1, c1, 'cagr'), (2, c2, 'cost2_cagr')):
            ref = refs[str(cost)]
            stock_change = EQUITY_SHARE * (cell['stock_cagr'] - frozen[float(cost)]['cagr'])
            etf_lot_drag = (1 - EQUITY_SHARE) * (cell['etf_cagr'] - ref['adjusted']['cagr'])
            premium_basis = (1 - EQUITY_SHARE) * (ref['adjusted']['cagr'] - ref['legacy']['cagr'])
            nominal_cagr = nominal['cagr'] if cost == 1 else nominal[nominal_key]
            residual = cell['blended_cagr'] - (nominal_cagr + stock_change + etf_lot_drag + premium_basis)
            verdicts[capital]['bridge'][f'cost{cost}'] = {
                'nominal_cagr': nominal_cagr,
                'stock_frozen_cagr': frozen[float(cost)]['cagr'],
                'stock_fixed_cagr': cell['stock_cagr'],
                'stock_change_pp': stock_change * 100,
                'etf_fractional_legacy_cagr': ref['legacy']['cagr'],
                'etf_fractional_adjusted_cagr': ref['adjusted']['cagr'],
                'etf_lot_cagr': cell['etf_cagr'],
                'etf_lot_drag_pp': etf_lot_drag * 100,
                'premium_basis_pp': premium_basis * 100,
                'residual_rebalance_and_median_pp': residual * 100,
                'realized_cagr': cell['blended_cagr'],
            }
    report = {'strategy': 'P5_BASE', 'status': 'diagnostic', 'window': [START, END],
              'equity_share': EQUITY_SHARE, 'stock_gross': STOCK_GROSS,
              'rebalance': 'quarter',
              'quarter_basis': (sorted(frame['quarter_basis'].astype(str).unique().tolist())
                                if 'quarter_basis' in frame.columns else ['window']),
              'cross_sleeve_rebalance': 'fractional_upper_bound',
              'stock_sleeve_execution': 'stable_book board lots, T+1, limits, suspensions, fees',
              'etf_sleeve_execution': '100-share lots, 0.05% fee, quarterly, 1:1 QDII overflow; '
                                      'no ffill execution on a missing bar',
              'premium_basis': 'adjusted', 'nominal_basis': 'legacy_qdii_premium',
              'suspension_prereg': str(SUSPENSION_PREREG.relative_to(ROOT)),
              'suspension_prereg_sha256': hashlib.sha256(SUSPENSION_PREREG.read_bytes()).hexdigest(),
              'premium_basis_effect_pp': (1 - EQUITY_SHARE) *
                                         (refs['1']['adjusted']['cagr'] - refs['1']['legacy']['cagr']) * 100,
              'delta_caveat': ('delta_vs_nominal is decomposed under verdicts[*].bridge: '
                               'stock-sleeve execution/lot change, ETF lot drag, premium basis, and '
                               'a small rebalance/median residual; the standalone G0 sleeve return is '
                               'identical under legacy and adjusted premium, so the delta is not a '
                               'premium-basis artifact'),
              'data_quality_caveat': ('stale runs are classified under '
                                      'docs/prereg/P5_DATA_QUALITY_SUSPENSION.md: a run counts as '
                                      'suspended only when every session has a verified raw-cache row '
                                      'with tradestatus=0, volume=0, amount=0 and close==preclose; any '
                                      'gap or contradiction stays unexplained. ETF missing days are '
                                      'stale_mark only when non-rebalance, trade-free, absent from both '
                                      'Tencent and Eastmoney, and isolated; otherwise blocked. The '
                                      'report keeps the raw, unexplained, coverage and ETF fields per path.'),
              'nominal_frozen': nominal,
              'cells': cells, 'quality': quality, 'verdicts': verdicts,
              'decision_rule': ('strict arithmetic without IPO: cost1 median >=13.0%, median mdd >=-25%, '
                                'cost2 median >=11.0%; plus all 8 paths >=-25% for each cost. '
                                'P5_BASE is diagnostic: status is no_improvement or insufficient_data, '
                                'never passed.'),
              'promotion_approved': False}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({'report': str(out / 'report.json'), 'rows': len(frame),
                      'verdicts': verdicts}, ensure_ascii=False, indent=1, default=float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', choices=('800000', '600000', 'both'), default='both')
    parser.add_argument('--cost', choices=('1', '2', 'both'), default='both')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--quarter-basis', choices=('window', 'calendar'), default='window',
                        help='window (legacy): a window ending mid-quarter rebalances on its own '
                             'final day; calendar: quarter ends are pinned to the full calendar')
    parser.add_argument('--assemble', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    if args.assemble:
        assemble(out)
        return
    capitals = [800_000.0, 600_000.0] if args.capital == 'both' else [float(args.capital)]
    costs = [1.0, 2.0] if args.cost == 'both' else [float(args.cost)]
    plans = G.plans()
    targets = {name: p4_ledger.to_targets(plan) for name, plan in plans.items()}
    codes = sorted({code for rows in targets.values() for row in rows
                    for code in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    sessions = dataset.calendar[(dataset.calendar >= pd.Timestamp(START)) &
                                (dataset.calendar <= pd.Timestamp(END))]
    for capital in capitals:
        for cost in costs:
            run_config(capital, cost, plans, targets, dataset, sessions, out,
                       args.quarter_basis)
    if len(capitals) == 2 and len(costs) == 2:
        assemble(out)


if __name__ == '__main__':
    main()
