#!/usr/bin/env python3
"""P1-A small-cap beta through an ETF proxy with fixed position rules (docs/prereg/P1_A_SMALLCAP_BETA_ETF.md).

  python scripts/p1_a_position.py [--window is|confirm]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import smallcap_audit as AU
from quantlab import research_statistics as RS
from quantlab import research_trials as T
from quantlab.smallcap import universe
from quantlab.smallcap.panel import Panel

VERSION = '20261006T152412Z'
OUT = ROOT / 'output/p1_a_20261008'
WINDOWS = {'is': ('2018-01-02', '2026-09-22'), 'confirm': ('2015-07-01', '2017-12-29')}
FEE, COST = 0.006, 0.001
LOOKBACK, MIN_LOOKBACK = 750, 250
STRESS = ('2024-01-02', '2024-02-05')
RULES = ('trend200', 'voltarget20', 'liquidity_share', 'relative_pb')
FAMILY = 'A_position'


def crowd_map(p):
    return 1 - np.clip((p - 0.7) / 0.3, 0, 1)


def trailing_pct(series):
    values = series.to_numpy()
    out = np.full(len(values), np.nan)
    for t in range(len(values)):
        if not np.isfinite(values[t]):
            continue
        window = values[max(0, t - LOOKBACK + 1):t + 1]
        window = window[np.isfinite(window)]
        if len(window) >= MIN_LOOKBACK:
            out[t] = (window <= values[t]).mean()
    return pd.Series(out, index=series.index)


def market_series():
    """Daily R-pool amount share and relative PB from the v4 panel (decided at each close)."""
    panel = Panel(VERSION)
    features = universe.rolling_features(panel, with_st=True)
    adv, history, st = features['adv'], features['history'], features['st']
    eligible = (history >= 252) & adv.notna() & ~st.eq(1)
    ranked = adv.where(eligible).rank(axis=1, ascending=False, method='first')
    count = eligible.sum(axis=1)
    pool = ranked.gt(np.ceil(0.6 * count), axis=0) & (adv >= universe.MIN_ADV)
    amount = panel.field('amount').astype(float)
    share = amount.where(pool).sum(axis=1) / amount.sum(axis=1)
    pb = panel.field('pb_mrq').astype(float).where(lambda x: x > 0)
    cap = features['cap']
    top300 = cap.rank(axis=1, ascending=False, method='first') <= 300
    relative = pb.where(pool).median(axis=1) / pb.where(top300).median(axis=1)
    return pd.DataFrame({'share20': share.rolling(20, min_periods=20).mean(), 'relative_pb': relative})


def positions(level, market):
    r = level.pct_change(fill_method=None)
    sigma = r.rolling(20, min_periods=20).std() * math.sqrt(252)
    market = market.reindex(level.index)
    raw = pd.DataFrame({
        'trend200': (level > level.rolling(200, min_periods=200).mean()).astype(float).where(
            level.rolling(200, min_periods=200).mean().notna()),
        'voltarget20': np.minimum(1.0, 0.20 / sigma),
        'liquidity_share': crowd_map(trailing_pct(market.share20)),
        'relative_pb': crowd_map(trailing_pct(market.relative_pb)),
    })
    raw['composite'] = raw[list(RULES)].mean(axis=1, skipna=False)
    return raw


def backtest(level, signal, start, end, cash=None, cost=COST):
    """Weekly signal at the last session of each week, executed at the next session's close."""
    dates = level.loc[start:end].index
    r = level.pct_change(fill_method=None).reindex(dates).fillna(0.0) - FEE / 252
    cash_r = (cash.pct_change(fill_method=None).reindex(dates).fillna(0.0) if cash is not None
              else pd.Series(0.0, index=dates))
    week = pd.Series(dates.isocalendar().week.to_numpy() + 100 * dates.year.to_numpy(), index=dates)
    signal_days = dates[(week != week.shift(-1)).to_numpy()]
    target = pd.Series(np.nan, index=dates)
    for day in signal_days:
        position = dates.get_loc(day)
        if position + 1 < len(dates) and np.isfinite(signal.get(day, np.nan)):
            target.iloc[position + 1] = signal[day]
    held = target.ffill().fillna(0.0)
    exposure = held.shift(1).fillna(0.0)
    trade = held.diff().abs().fillna(held.iloc[0])
    daily = exposure * r + (1 - exposure) * cash_r - trade * cost
    nav = (1 + daily).cumprod()
    return nav, held, daily


def stats(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {'cagr': float(nav.iloc[-1] ** (1 / years) - 1),
            'max_drawdown': float((nav / nav.cummax() - 1).min()),
            'vol': float(nav.pct_change().std() * math.sqrt(252))}


def run(window):
    start, end = WINDOWS[window]
    level = AU.bench_series('932000CNY010')
    csi1000 = AU.bench_series('H00852')
    etf = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    bond = etf[etf.code == 'sh511010'].set_index('date').close_hfq.sort_index()
    bond.index = pd.to_datetime(bond.index)
    market = market_series()
    pos = positions(level, market)
    buy_hold = pd.Series(1.0, index=level.index)
    out = {'window': window, 'family_trials': T.count(FAMILY), 'global_trials': T.count()}
    bh_nav, _, bh_daily = backtest(level, buy_hold, start, end)
    out['buy_hold'] = stats(bh_nav)
    results = {}
    for name in (*RULES, 'composite'):
        nav, held, daily = backtest(level, pos[name], start, end)
        info = stats(nav)
        info['mean_position'] = float(held.mean())
        info['win_rates'] = RS.win_rates(daily, benchmark=bh_daily)
        info['cost2x'] = stats(backtest(level, pos[name], start, end, cost=2 * COST)[0])
        info['cash_in_511010'] = stats(backtest(level, pos[name], start, end, cash=bond.reindex(level.index).ffill())[0])
        stress = held.loc[STRESS[0]:STRESS[1]]
        info['stress_min_position'] = float(stress.min()) if len(stress) else None
        results[name] = info
        if name == 'composite':
            (OUT / window).mkdir(parents=True, exist_ok=True)
            nav.to_frame('nav').assign(position=held).to_csv(OUT / window / 'composite_nav.csv')
    out['rules'] = results
    out['buy_hold']['win_rates'] = RS.win_rates(bh_daily)
    c, b = results['composite'], out['buy_hold']
    gate = {'drawdown_cut>=30%': abs(c['max_drawdown']) <= 0.7 * abs(b['max_drawdown']),
            'cagr>=2/3_buy_hold': c['cagr'] >= (2 / 3 * b['cagr'] if b['cagr'] > 0 else b['cagr'])}
    if window == 'is':
        gate['stress_position<=40%'] = c['stress_min_position'] is not None and c['stress_min_position'] <= 0.40
    gate['pass'] = all(gate.values())
    out['gate'] = gate
    nav, held, daily = backtest(level, pos['composite'], start, end)
    out['composite_dsr'] = RS.deflated_sharpe(daily, trials=max(T.count(FAMILY), 2)).get('value')
    c1000 = backtest(csi1000, pos['composite'], start, end)[0]
    out['csi1000_vehicle'] = {'composite': stats(c1000), 'buy_hold': stats(backtest(csi1000, buy_hold, start, end)[0])}
    out['positions_quarterly'] = {str(k.date()): round(float(v), 3) for k, v in held.resample('QE').last().items()}
    out['prereg_sha256'] = T.sha256_file(ROOT / 'docs/prereg/P1_A_SMALLCAP_BETA_ETF.md')
    out['notes'] = ['percentile lookback uses all available history up to 750 sessions, minimum 250',
                    'cash earns 0 in the gate; 511010 variant is report-only']
    out['promotion_approved'] = False
    (OUT / window).mkdir(parents=True, exist_ok=True)
    (OUT / window / 'report.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({'buy_hold': out['buy_hold'], 'composite': {k: results['composite'][k] for k in
                      ('cagr', 'max_drawdown', 'mean_position', 'stress_min_position')}, 'gate': gate}, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='is', choices=tuple(WINDOWS))
    run(parser.parse_args().window)
