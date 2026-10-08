#!/usr/bin/env python3
"""Forward record of the fixed risk-parity portfolio RP3 (docs/prereg/P2_FORWARD_RP3.md).

  python scripts/p2_forward.py snapshot --asof 2026-09-30
  python scripts/p2_forward.py settle --to 2026-10-30

``snapshot`` freezes leg weights and holdings at a month end; ``settle`` books the period from the
latest snapshot to ``--to`` at closing prices. M2 is booked only when the stock panel covers the
whole period; otherwise the period stays ``pending_data``. Not an account ledger.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p2_cb_double_low as CB
import smallcap_audit as AU
from quantlab.smallcap.panel import Panel, latest_version

PREREG = 'docs/prereg/P2_FORWARD_RP3.md'
OUT = ROOT / 'output/p2_forward_rp3'
PATHS = [f'{p}_{s}' for p in (0, 4, 8, 12, 16) for s in (1, 2)]
VOL = 120
LEG_COST = {'m2': 0.004, 'cb': 0.001, 'c': 0.0005}
TRADE_COST = {'m2': 0.002, 'cb': 0.00075}
RECOVERY = 0.3


def nav_csv(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)['nav']


def c_returns():
    cb = pd.read_parquet(ROOT / 'data/cb/index_000832.parquet').set_index('date').close
    h = AU.bench_series('H20269')
    days = cb.index.intersection(h.index)
    return (0.5 * cb.reindex(days).pct_change() + 0.5 * h.reindex(days).pct_change()).dropna()


def leg_history(out):
    """Daily leg returns: backtest NAVs, then booked forward returns."""
    m2 = pd.concat([nav_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{p}_cost1.csv').pct_change() for p in PATHS],
                   axis=1).mean(axis=1)
    cb = nav_csv(ROOT / 'output/p2_cb_double_low_20261008/nav_main.csv').pct_change()
    frame = pd.DataFrame({'m2': m2, 'cb': cb, 'c': c_returns()}).loc['2018-01-03':]
    booked = out / 'daily.csv'
    if booked.exists():
        forward = pd.read_csv(booked, index_col=0, parse_dates=True)[['m2', 'cb', 'c']]
        frame = pd.concat([frame[frame.index < forward.index.min()], forward])
    return frame


def m2_holdings(asof):
    import p1_m2 as M
    import smallcap_layers as L
    M.use('is')
    L.load_state()
    out = {}
    for path in PATHS:
        schedule = M.schedule_for(*map(int, path.split('_')))
        day = max(d for d in schedule if d <= asof)
        out[path] = {'signal_date': str(day.date()), 'weights': {c: float(w) for c, w in schedule[day].items()}}
    return out


def cb_holdings(asof):
    terms, bars, value, index, last, kind = CB.load()
    sessions = pd.DatetimeIndex(sorted(bars.date.unique()))
    sessions = sessions[sessions <= asof]
    CB.END = str(asof.date())
    targets, _ = CB.targets(bars, value, terms, last, kind, sessions)
    day = max(targets)
    return {'signal_date': str(day.date()), 'weights': targets[day]}


def snapshot(asof, out):
    asof = pd.Timestamp(asof)
    history = leg_history(out).loc[:asof].dropna()
    vol = history.tail(VOL).std()
    weights = (1 / vol) / (1 / vol).sum()
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    snap = {'asof': str(asof.date()), 'leg_weights': weights.to_dict(), 'vol_window_end': str(history.index[-1].date()),
            'm2': m2_holdings(asof), 'cb': cb_holdings(asof), 'panel_version': latest_version(),
            'cb_bars_last': str(pd.read_parquet(ROOT / 'data/cb/bars.parquet').date.max().date()),
            'source_commit': commit, 'prereg': PREREG, 'promotion_approved': False, 'forward_decision': False}
    out.mkdir(parents=True, exist_ok=True)
    (out / f'snapshot_{snap["asof"]}.json').write_text(json.dumps(snap, indent=1))
    print(json.dumps({k: snap[k] for k in ('asof', 'leg_weights', 'vol_window_end', 'panel_version', 'cb_bars_last')}))


def basket(prices, weights, start, days):
    """Daily value of a buy-and-hold basket (cash at zero return), relative to ``start``."""
    codes = [c for c in weights if c in prices.columns]
    px = prices[codes].ffill()
    base = px.loc[start]
    w = pd.Series({c: weights[c] for c in codes})
    value = (px.loc[days] / base * w).sum(axis=1) + (1 - w.sum())
    return value, sorted(set(weights) - set(codes))


def turnover(new, old):
    codes = set(new) | set(old)
    return sum(abs(new.get(c, 0.0) - old.get(c, 0.0)) for c in codes)


def settle(to, out):
    to = pd.Timestamp(to)
    snaps = sorted(out.glob('snapshot_*.json'))
    snap = json.loads(snaps[-1].read_text())
    prev = json.loads(snaps[-2].read_text()) if len(snaps) > 1 else None
    start = pd.Timestamp(snap['asof'])
    panel = Panel(latest_version())
    adj = panel.field('adj_close')
    adj.columns = [c.lower() for c in adj.columns]
    c_ret = c_returns()
    days = c_ret.loc[(c_ret.index > start) & (c_ret.index <= to)].index
    row = {'period_start': str(start.date()), 'period_end': str(to.date()), 'snapshot': snaps[-1].name}
    if panel.sessions[-1] < to:
        row.update(status='pending_data', reason=f'stock panel ends {panel.sessions[-1].date()}')
        print(json.dumps(row))
        return row
    m2_values, missing = [], set()
    anchor = adj.index[adj.index <= start][-1]
    for path in PATHS:
        weights = {c.lower(): w for c, w in snap['m2'][path]['weights'].items()}
        value, miss = basket(adj, weights, anchor, days)
        m2_values.append(value)
        missing |= set(miss)
    m2 = pd.concat(m2_values, axis=1).mean(axis=1)
    terms, bars, value_frame, index, last, kind = CB.load()
    close = bars.pivot(index='date', columns='code', values='close').sort_index()
    for code in close.columns:
        if kind.get(code) == 'default' and last[code] < close.index[-1]:
            close.loc[close.index > last[code], code] = close.loc[last[code], code] * RECOVERY
    cb, cb_missing = basket(close.reindex(close.index.union(days)), snap['cb']['weights'], start, days)
    c = (1 + c_ret.loc[days]).cumprod()
    legs = pd.DataFrame({'m2': m2, 'cb': cb, 'c': c})
    w = pd.Series(snap['leg_weights'])
    cost = 0.0
    if prev:
        old_m2 = {c: np.mean([prev['m2'][p]['weights'].get(c, 0.0) for p in PATHS])
                  for p in PATHS for c in prev['m2'][p]['weights']}
        new_m2 = {c: np.mean([snap['m2'][p]['weights'].get(c, 0.0) for p in PATHS])
                  for p in PATHS for c in snap['m2'][p]['weights']}
        cost += w['m2'] * turnover(new_m2, old_m2) * TRADE_COST['m2']
        cost += w['cb'] * turnover(snap['cb']['weights'], prev['cb']['weights']) * TRADE_COST['cb']
        cost += sum(abs(w[k] - prev['leg_weights'][k]) * LEG_COST[k] for k in w.index)
    else:
        cost += w['m2'] * 0.9 * TRADE_COST['m2'] + w['cb'] * 0.98 * TRADE_COST['cb'] + w['c'] * LEG_COST['c']
    portfolio = (legs * w).sum(axis=1) * (1 - cost)
    daily = pd.DataFrame({'m2': legs.m2.pct_change().fillna(legs.m2.iloc[0] - 1),
                          'cb': legs.cb.pct_change().fillna(legs.cb.iloc[0] - 1),
                          'c': legs.c.pct_change().fillna(legs.c.iloc[0] - 1),
                          'portfolio': portfolio.pct_change().fillna(portfolio.iloc[0] - 1)})
    path = out / 'daily.csv'
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        daily = pd.concat([old[old.index <= start], daily])
    daily.to_csv(path)
    row.update(status='booked', m2=float(m2.iloc[-1] - 1), cb=float(cb.iloc[-1] - 1), c=float(c.iloc[-1] - 1),
               cost=cost, portfolio=float(portfolio.iloc[-1] - 1), m2_missing=sorted(missing),
               cb_missing=cb_missing, panel_version=latest_version())
    ledger = out / 'ledger.jsonl'
    with ledger.open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    print(json.dumps(row))
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('snapshot', 'settle'))
    parser.add_argument('--asof')
    parser.add_argument('--to')
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    if args.stage == 'snapshot':
        snapshot(args.asof, Path(args.out))
    else:
        settle(args.to, Path(args.out))
