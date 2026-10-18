"""Pre-flight feasibility for a drawdown-triggered defensive overlay on the P5_H daily sleeve.

Answers one question before any trial is registered: can a trigger/unlock rule on the P5_H
stock sleeve keep the worst path inside the -25% drawdown budget without giving back the
return that made the sleeve worth protecting?

Defensive asset is the frozen G0 sleeve (gold/treasury/Nasdaq) exactly as the production
ledger marks it. Signals act on the sleeve's own drawdown from its running peak; the switch
is executed one session later and charged a round-trip cost.

This is a feasibility computation on 2018-2026 data, not a trial. Any parameter that looks
good here is *fixed* for a later preregistration, never refitted on the same window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P5H = ROOT / 'output/p5_h_20261018'
R1 = ROOT / 'output/p5_base_lot_20261018_r1/nav/cap800000_cost1_POOL50_G97_p0_s1.csv'
OUT = ROOT / 'output/p7_defense_preflight_20261018'

PATHS = ['p0_s1', 'p0_s2', 'p5_s1', 'p5_s2', 'p10_s1', 'p10_s2', 'p15_s1', 'p15_s2']
TRIGGERS = [0.12, 0.15, 0.18]
UNLOCKS = [('newhigh', 0.0), ('hyst', 0.05), ('hyst', 0.09), ('time', 60), ('time', 120)]
SWITCH_COST = 0.003  # round trip on the switched portion, small-cap illiquid band


def sleeve_returns(cost):
    frames = {}
    for path in PATHS:
        nav = pd.read_csv(P5H / f'nav_EVENT_H_{path}_cost{cost}.csv', index_col=0,
                          parse_dates=True)['nav'].sort_index()
        frames[path] = nav
    return frames


def overlay(stock_nav, defense_nav, trigger, unlock, lag=1):
    """Return (overlay nav series, days in defense, switches, max lockout, premium paid)."""
    calendar = stock_nav.index.intersection(defense_nav.index)
    r_stock = stock_nav.reindex(calendar).pct_change(fill_method=None).fillna(0.0)
    r_def = defense_nav.reindex(calendar).pct_change(fill_method=None).fillna(0.0)
    peak = stock_nav.reindex(calendar)
    run_max = peak.cummax()
    depth = peak / run_max - 1.0

    kind, param = unlock
    in_stock = True
    lock_started = None
    nav, defense_days, switches, max_lock = 1.0, 0, 0, 0
    path = [1.0]
    premium = 0.0
    for i, day in enumerate(calendar):
        prev = in_stock
        if i >= lag:
            signal_depth = depth.iloc[i - lag]
            signal_level = run_max.iloc[i - lag]
            if prev and signal_depth <= -trigger:
                in_stock = False
                lock_started = i
            elif not prev:
                if kind == 'newhigh':
                    in_stock = peak.iloc[i - lag] >= signal_level - 1e-9
                elif kind == 'hyst':
                    in_stock = signal_depth >= -(trigger - param)
                elif kind == 'time' and lock_started is not None:
                    in_stock = (i - lock_started) >= param
        switches += int(prev != in_stock)
        defense_days += int(not in_stock)
        if not in_stock:
            premium += float(r_stock.iloc[i] - r_def.iloc[i])
        step = r_stock.iloc[i] if in_stock else r_def.iloc[i]
        if prev != in_stock:
            step -= SWITCH_COST
        nav *= 1.0 + step
        path.append(nav)
        if not in_stock and lock_started is not None:
            max_lock = max(max_lock, i - lock_started)
    return pd.Series(path[1:], index=calendar), defense_days, switches, max_lock, premium


def summarise(series, calendar):
    cagr = float(series.iloc[-1] ** (252.0 / len(series)) - 1.0)
    mdd = float((series / series.cummax() - 1.0).min())
    return cagr, mdd


def ex_year(series, year):
    keep = series[[d.year != year for d in series.index]]
    months = len(keep) / 252.0
    total = float(keep.iloc[-1] / keep.iloc[0])
    return total ** (1.0 / months) - 1.0


def alternative_defense_assets():
    """Individual ETFs and cash, so the verdict cannot be an artifact of the G0 blend.

    Uses ``close_hfq``: ``close_raw`` carries the 2022-01-14 513100 share conversion as an
    -80.45% one-day step, and the production ETF sleeve already reads ``close_hfq``.
    """
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    close = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    close.index = pd.to_datetime(close.index)
    out = {'cash': None}
    for name, code in {'treasury_511260': 'sh511260', 'gold_518880': 'sh518880',
                       'nasdaq_513100': 'sh513100', 'etf_159920': 'sz159920',
                       'dividend_510880': 'sh510880'}.items():
        if code in close.columns:
            out[name] = close[code].dropna()
    return out


def defense_states(stock_nav, defense_index, trigger, unlock, lag=1):
    """Replay the state machine and return the per-day in-stock flag."""
    calendar = stock_nav.index.intersection(defense_index)
    peak = stock_nav.reindex(calendar)
    run_max = peak.cummax()
    depth = peak / run_max - 1.0
    kind, param = unlock
    in_stock, lock_started, states = True, None, []
    for i in range(len(calendar)):
        if i >= lag:
            signal_depth, signal_level = depth.iloc[i - lag], run_max.iloc[i - lag]
            if in_stock and signal_depth <= -trigger:
                in_stock, lock_started = False, i
            elif not in_stock:
                if kind == 'newhigh':
                    in_stock = peak.iloc[i - lag] >= signal_level - 1e-9
                elif kind == 'hyst':
                    in_stock = signal_depth >= -(trigger - param)
                elif kind == 'time' and lock_started is not None:
                    in_stock = (i - lock_started) >= param
        states.append(in_stock)
    return pd.Series(states, index=calendar)


def rebound_capture(stock_nav, defense_nav, trigger, unlock, lag=1):
    """How much of the sleeve's gain arrives on days the overlay is defensive.

    The illiquidity premium is separable from the drawdown only if the up-sessions sit outside
    the defensive windows. This counts them.
    """
    calendar = stock_nav.index.intersection(defense_nav.index)
    r_stock = stock_nav.reindex(calendar).pct_change(fill_method=None).fillna(0.0)
    states = defense_states(stock_nav, calendar, trigger, unlock, lag)
    gains = r_stock[r_stock > 0]
    defensive_gain = float(gains[~states.reindex(gains.index).astype(bool)].sum())
    top10 = r_stock.nlargest(10)
    return {'up_gain_defensive_share': defensive_gain / float(gains.sum()),
            'top10_defensive_count': int((~states.reindex(top10.index).astype(bool)).sum()),
            'top10_cumulative': float(top10.sum())}


def run():
    defense = pd.read_csv(R1, index_col=0, parse_dates=True)['etf'].sort_index()
    report = {'strategy': 'P7_DEFENSE_PREFLIGHT', 'stage': 'preflight',
              'switch_cost_round_trip': SWITCH_COST, 'defensive_asset': 'G0 (frozen 3:3:2)',
              'sleeve': 'P5_H daily, 8 paths x cost 1/2', 'configs': {},
              'not_a_trial': True, 'parameters_locked_if_used': True,
              'promotion_approved': False}
    bare = {}
    for cost in (1, 2):
        per_path = {}
        for path, nav in sleeve_returns(cost).items():
            cagr, mdd = summarise(nav / nav.iloc[0], nav.index)
            per_path[path] = {'cagr': cagr, 'mdd': mdd, 'ex2025_cagr': ex_year(nav / nav.iloc[0], 2025)}
        bare[f'cost{cost}'] = {
            'cagr_median': float(np.median([v['cagr'] for v in per_path.values()])),
            'mdd_median': float(np.median([v['mdd'] for v in per_path.values()])),
            'mdd_worst': float(min(v['mdd'] for v in per_path.values())),
            'ex2025_cagr_median': float(np.median([v['ex2025_cagr'] for v in per_path.values()])),
            'paths': per_path}
    report['sleeve_bare'] = bare

    for cost in (1, 2):
        for trigger in TRIGGERS:
            for unlock in UNLOCKS:
                key = f'cost{cost}_T{trigger:.2f}_{unlock[0]}{unlock[1]:g}'
                rows = []
                for path, nav in sleeve_returns(cost).items():
                    series, days, switches, max_lock, premium = overlay(nav, defense, trigger, unlock)
                    cagr, mdd = summarise(series, series.index)
                    rows.append({'path': path, 'cagr': cagr, 'mdd': mdd,
                                 'defense_share': days / len(series), 'switches': switches,
                                 'max_lockout_sessions': max_lock, 'premium_paid': premium})
                report['configs'][key] = {
                    'cagr_median': float(np.median([r['cagr'] for r in rows])),
                    'mdd_median': float(np.median([r['mdd'] for r in rows])),
                    'mdd_worst': float(min(r['mdd'] for r in rows)),
                    'defense_share_median': float(np.median([r['defense_share'] for r in rows])),
                    'switches_median': float(np.median([r['switches'] for r in rows])),
                    'max_lockout_median': float(np.median([r['max_lockout_sessions'] for r in rows])),
                    'premium_paid_median': float(np.median([r['premium_paid'] for r in rows])),
                    'paths_inside_minus_25': int(sum(1 for r in rows if r['mdd'] >= -0.25)),
                    'per_path': rows}

    best_unlocks = [(0.12, ('time', 120)), (0.12, ('time', 60)), (0.15, ('time', 120)),
                    (0.12, ('hyst', 0.09)), (0.12, ('newhigh', 0.0))]

    report['defense_asset_sweep'] = {}
    for name, alt in alternative_defense_assets().items():
        per = {}
        for trigger, unlock in best_unlocks:
            cagrs, mdds = [], []
            for path, nav in sleeve_returns(1).items():
                defense_asset = (pd.Series(1.0, index=nav.index) if alt is None else alt)
                out, _, _, _, _ = overlay(nav, defense_asset, trigger, unlock)
                cagr, mdd = summarise(out, out.index)
                cagrs.append(cagr)
                mdds.append(mdd)
            per[f'T{trigger:.2f}_{unlock[0]}{unlock[1]:g}'] = {
                'cagr_median': float(np.median(cagrs)), 'mdd_median': float(np.median(mdds))}
        report['defense_asset_sweep'][name] = per

    report['rebound_capture'] = {}
    for trigger, unlock in best_unlocks:
        stats = [rebound_capture(nav, defense, trigger, unlock)
                 for nav in sleeve_returns(1).values()]
        report['rebound_capture'][f'T{trigger:.2f}_{unlock[0]}{unlock[1]:g}'] = {
            'up_gain_defensive_share_median': float(np.median(
                [s['up_gain_defensive_share'] for s in stats])),
            'top10_defensive_count_median': float(np.median(
                [s['top10_defensive_count'] for s in stats])),
            'top10_cumulative_median': float(np.median([s['top10_cumulative'] for s in stats]))}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({'sleeve_bare': {k: {kk: vv for kk, vv in v.items() if kk != 'paths'}
                                      for k, v in report['sleeve_bare'].items()},
                      'configs': {k: {kk: vv for kk, vv in v.items() if kk != 'per_path'}
                                  for k, v in report['configs'].items()}},
                     ensure_ascii=False, indent=1, default=float))
    return report


if __name__ == '__main__':
    run()
