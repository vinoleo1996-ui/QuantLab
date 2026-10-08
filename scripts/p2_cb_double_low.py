#!/usr/bin/env python3
"""P2 convertible-bond double-low leg (docs/prereg/P2_CB_DOUBLE_LOW.md).

  python scripts/p2_cb_double_low.py

Writes output/p2_cb_double_low_20261008/{targets.json, nav_*.csv, report.json}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantlab import research_trials as T
from quantlab.broad import month_ends
from quantlab.etf_account import replay, stats

PREREG = 'docs/prereg/P2_CB_DOUBLE_LOW.md'
DATA = ROOT / 'data/cb'
OUT = ROOT / 'output/p2_cb_double_low_20261008'
START, END, DATA_END = '2018-01-02', '2026-09-30', pd.Timestamp('2026-09-30')
TOP, WEIGHT, MIN_AGE, MIN_AMOUNT, EXIT_DAYS, GAP_DAYS, FLAT = 20, 0.049, 20, 5e6, 30, 30, 30


def load():
    terms = pd.read_parquet(DATA / 'terms.parquet')
    terms = terms[~terms.SECURITY_CODE.str.startswith('404')].set_index('SECURITY_CODE')
    bars = pd.read_parquet(DATA / 'bars.parquet')
    bars = bars[bars.code.isin(terms.index)].sort_values(['code', 'date'])
    value = pd.read_parquet(DATA / 'value.parquet')
    index = pd.read_parquet(DATA / 'index_000832.parquet').set_index('date').close
    last = bars.groupby('code').date.max()
    delist = terms.DELIST_DATE.reindex(last.index)
    gap = (delist - last).dt.days
    default = (gap > GAP_DAYS) | (delist.isna() & ((DATA_END - last).dt.days > GAP_DAYS))
    kind = pd.Series(np.where(default, 'default', np.where(delist.notna(), 'redeemed', 'live')), index=last.index)
    return terms, bars, value, index, last, kind


def account_bars(bars, sessions, last, kind, recovery):
    """Raw bars plus post-exit rows: redeemed -> flat cash-equivalent bars; default -> marked, untradable."""
    extra = []
    closes = bars.groupby('code').close.last()
    for code, day in last.items():
        after = sessions[sessions > day]
        if kind[code] == 'redeemed' and len(after):
            px = closes[code]
            extra.append(pd.DataFrame({'date': after[:FLAT], 'code': code, 'open': px, 'close': px}))
        elif kind[code] == 'default' and len(after):
            extra.append(pd.DataFrame({'date': after[:1], 'code': code, 'open': np.nan,
                                       'close': closes[code] * recovery}))
    frame = pd.concat([bars[['date', 'code', 'open', 'close']], *extra], ignore_index=True)
    return frame.assign(open_raw=frame.open, open_hfq=frame.open, close_hfq=frame.close)


def targets(bars, value, terms, last, kind, sessions, exclude_exit=True):
    close = bars.pivot(index='date', columns='code', values='close').reindex(sessions)
    amount = (bars.close * bars.volume).groupby([bars.date, bars.code]).sum().unstack().reindex(sessions)
    adv = amount.rolling(20, min_periods=20).mean()
    age = close.notna().cumsum()
    premium = value.pivot_table(index='date', columns='code', values='premium_pct', aggfunc='last').reindex(sessions)
    out, eligible_counts = {}, {}
    for day in month_ends(sessions, START, END):
        ok = close.loc[day].notna() & (age.loc[day] > MIN_AGE) & (adv.loc[day] >= MIN_AMOUNT)
        ok &= premium.loc[day].reindex(ok.index).notna()
        if exclude_exit:
            delist = terms.DELIST_DATE.reindex(ok.index)
            near = (kind.reindex(ok.index) == 'redeemed') & ((delist - day).dt.days <= EXIT_DAYS)
            ok &= ~near
        score = (close.loc[day] + premium.loc[day].reindex(close.columns))[ok]
        eligible_counts[str(day.date())] = int(ok.sum())
        out[day] = {c: WEIGHT for c in score.nsmallest(TOP).index}
    return out, eligible_counts


def yearly(nav):
    return {str(y): float(v) for y, v in nav.groupby(nav.index.year).last().div(
        nav.groupby(nav.index.year).last().shift(1).fillna(nav.iloc[0])).sub(1).items()}


def main():
    terms, bars, value, index, last, kind = load()
    sessions = pd.DatetimeIndex(sorted(bars.date.unique()))
    sessions = sessions[sessions <= DATA_END]
    OUT.mkdir(parents=True, exist_ok=True)
    tg, counts = targets(bars, value, terms, last, kind, sessions)
    (OUT / 'targets.json').write_text(json.dumps({str(k.date()): v for k, v in tg.items()}, indent=0))
    runs = {}
    for name, cost, recovery, sched in [('main', 1.0, 0.3, tg), ('cost2', 2.0, 0.3, tg),
                                        ('recovery0', 1.0, 0.0, tg), ('recovery100', 1.0, 1.0, tg),
                                        ('no_exit_rule', 1.0, 0.3, None)]:
        if sched is None:
            sched, _ = targets(bars, value, terms, last, kind, sessions, exclude_exit=False)
        result = replay(account_bars(bars, sessions, last, kind, recovery), sched, START, END, cost_mult=cost, lot=10)
        result['nav'].to_csv(OUT / f'nav_{name}.csv')
        runs[name] = result
    nav = runs['main']['nav']
    bench = index.loc[START:END]
    b = stats(bench / bench.iloc[0])
    m = stats(nav)
    held = {c for v in tg.values() for c in v}
    m2 = pd.read_csv(ROOT / 'output/p1_m2_20261008/is/ledger_nav_0_1_cost1.csv', index_col=0, parse_dates=True)['nav']
    corr = nav.resample('ME').last().pct_change().corr(m2.resample('ME').last().pct_change())
    opens = bars.pivot(index='date', columns='code', values='open').reindex(sessions)
    gap = opens / bars.pivot(index='date', columns='code', values='close').reindex(sessions).ffill().shift(1) - 1
    limit_trades, previous = 0, set()
    for day, weights in tg.items():
        nxt = sessions[sessions.searchsorted(day, side='right')] if day < sessions[-1] else None
        if nxt is not None:
            traded = set(weights) ^ previous
            limit_trades += int((gap.loc[nxt].reindex(list(traded)).abs() > 0.195).sum())
        previous = set(weights)
    report = {
        'open_beyond_19_5pct_trades': limit_trades,
        'main': m, 'benchmark_000832': b, 'cost2': stats(runs['cost2']['nav']),
        'gate': {'beats_index': m['cagr'] > b['cagr'], 'cost2_positive': stats(runs['cost2']['nav'])['cagr'] > 0},
        'yearly': yearly(nav), 'yearly_index': yearly(bench),
        'sensitivity': {k: stats(runs[k]['nav']) for k in ('recovery0', 'recovery100', 'no_exit_rule')},
        'trades': runs['main']['trades'], 'fees': runs['main']['fees'],
        'corr_m2_monthly_path0_1': float(corr),
        'eligible_min_median': [min(counts.values()), float(np.median(list(counts.values())))],
        'defaults_held': sorted(f'{c} {terms.SECURITY_NAME_ABBR.get(c, "")}' for c in held if kind.get(c) == 'default'),
        'default_count_universe': int((kind == 'default').sum()),
        'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG), 'promotion_approved': False,
    }
    report['gate']['pass'] = all(report['gate'].values())
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ('main', 'benchmark_000832', 'cost2', 'gate')}, default=float))


if __name__ == '__main__':
    main()
