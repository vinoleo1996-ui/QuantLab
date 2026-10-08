#!/usr/bin/env python3
"""P2 multi-asset ETF trend/momentum rotation (docs/prereg/P2_ETF_ROTATION.md).

  python scripts/p2_etf_rotation.py [--window is|confirm]

Writes output/p2_etf_rotation_20261008/<window>/{nav.csv, holdings.csv, report.json}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from quantlab import research_trials as T
from quantlab.broad import month_ends
from quantlab.etf_account import replay, stats

PREREG = 'docs/prereg/P2_ETF_ROTATION.md'
OUT = ROOT / 'output/p2_etf_rotation_20261008'
WINDOWS = {'is': ('2018-01-02', '2026-09-22'), 'confirm': ('2015-01-05', '2017-12-29')}
CANDIDATES = ['sh510300', 'sh510500', 'sh512100', 'sz159920', 'sh513100', 'sh513500',
              'sh511010', 'sh511260', 'sh518880', 'sz159985', 'sz159980']
LOOKBACKS, TREND, HISTORY, TOP, WEIGHT = (21, 63, 126, 252), 200, 252, 3, 0.33


def load():
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    return bars[bars.code.isin(CANDIDATES)].copy()


def signals(bars, start, end):
    observed = bars.pivot_table(index='date', columns='code', values='close_hfq').sort_index()
    close = observed.ffill()
    signal_days = month_ends(close.index, pd.Timestamp(start) - pd.Timedelta(days=40), end)
    signal_days = signal_days[signal_days < pd.Timestamp(start)][-1:].append(
        signal_days[signal_days >= pd.Timestamp(start)])
    rotation, equal = {}, {}
    for day in signal_days:
        history = close.loc[:day]
        counts = observed.loc[:day].notna().sum()
        available = [c for c in CANDIDATES if counts.get(c, 0) >= HISTORY and pd.notna(history[c].iloc[-1])]
        equal[day] = {c: 0.99 / len(available) for c in available} if available else {}
        scores = {}
        for code in available:
            series = history[code].dropna()
            if series.iloc[-1] <= series.iloc[-TREND:].mean():
                continue
            scores[code] = sum(series.iloc[-1] / series.iloc[-1 - k] - 1 for k in LOOKBACKS) / len(LOOKBACKS)
        chosen = sorted(scores, key=scores.get, reverse=True)[:TOP]
        rotation[day] = {c: WEIGHT for c in chosen}
    return rotation, equal


def year_returns(nav):
    ends = nav.groupby(nav.index.year).last()
    starts = pd.Series(nav.iloc[0], index=[ends.index[0]])
    previous = pd.concat([starts, ends.iloc[:-1]]).to_numpy()
    return {int(y): float(e / p - 1) for y, e, p in zip(ends.index, ends.to_numpy(), previous)}


def main(window):
    start, end = WINDOWS[window]
    bars = load()
    rotation, equal = signals(bars, start, end)
    result = {name: replay(bars, plan, start, end, cost_mult=1.0) for name, plan in (('rotation', rotation), ('equal', equal))}
    cost2 = replay(bars, rotation, start, end, cost_mult=2.0)
    no_gold = {d: {c: w for c, w in v.items() if c != 'sh518880'} for d, v in rotation.items()}
    nav = result['rotation']['nav']
    years = year_returns(nav)
    close = bars.pivot_table(index='date', columns='code', values='close_hfq').sort_index().loc[start:end]
    csi300 = year_returns(close['sh510300'].dropna())
    csi1000 = year_returns(close['sh512100'].dropna()) if close['sh512100'].notna().any() else {}
    bear = [y for y, r in csi300.items() if r < 0]
    small_bear = [y for y, r in csi1000.items() if r < 0]
    held = sum(1 for y in bear if years.get(y, -1) >= 0)
    s, b, s2 = stats(nav), stats(result['equal']['nav']), stats(cost2['nav'])
    gates = {'cagr>equal_weight': s['cagr'] > b['cagr'], 'max_drawdown<=15%': s['max_drawdown'] >= -0.15,
             'equity_bear_years_nonnegative>=2/3': held >= math.ceil(2 * len(bear) / 3) if bear else True,
             'cost2_cagr>0': s2['cagr'] > 0}
    gates['pass'] = all(gates.values())
    holdings = pd.DataFrame([{'date': d, **{c: 1 for c in v}} for d, v in rotation.items()]).set_index('date').fillna(0)
    report = {'window': window, 'rotation': s, 'equal_weight': b, 'cost2': s2,
              'no_gold': stats(replay(bars, no_gold, start, end)['nav']),
              'sharpe': float(nav.pct_change().mean() / nav.pct_change().std() * math.sqrt(252)),
              'years': years, 'csi300_years': csi300, 'equity_bear_years': bear,
              'small_cap_bear_years': small_bear, 'small_cap_bear_returns': {y: years.get(y) for y in small_bear},
              'trades': result['rotation']['trades'], 'fees': result['rotation']['fees'],
              'holding_share': holdings.mean().round(3).to_dict(), 'gates': gates,
              'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG), 'promotion_approved': False}
    m2 = ROOT / 'output/p1_m2_20261008/is/ledger_nav_0_1_cost1.csv'
    if window == 'is' and m2.exists():
        m2nav = pd.read_csv(m2, index_col=0, parse_dates=True)['nav']
        monthly = pd.concat([nav.resample('ME').last().pct_change(), m2nav.resample('ME').last().pct_change()], axis=1)
        report['corr_with_m2_monthly'] = float(monthly.dropna().corr().iloc[0, 1])
    folder = OUT / window
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'rotation': nav, 'equal': result['equal']['nav'], 'cost2': cost2['nav']}).to_csv(folder / 'nav.csv')
    holdings.to_csv(folder / 'holdings.csv')
    (folder / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ('rotation', 'equal_weight', 'cost2', 'gates', 'equity_bear_years')}, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='is', choices=tuple(WINDOWS))
    main(parser.parse_args().window)
