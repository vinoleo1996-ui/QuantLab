#!/usr/bin/env python3
"""Combine sleeves at NAV level with quarter-end rebalancing (theme-leg experiments).

Usage: smallcap_theme_combo.py <name> <small_navs> <theme_navs|none> [--cost 1|2]
  small_navs  parquet of small-cap sub-account NAVs (one column per run)
  theme_navs  parquet of theme-leg NAVs; the pointwise median across phases is used
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import smallcap_e7 as E7
import smallcap_layers as L
from quantlab.smallcap.sim import metrics, segment_cagr

OUT = ROOT / 'output/smallcap_theme_20261008'
STOCK_COST, ETF_COST = 0.004, 0.0005


def combine(sleeves, weights, stock_sleeves, cost_mult=1.0, initial=600_000.0):
    """``sleeves``: dict name -> price-like series on a shared index (any scale)."""
    frame = pd.DataFrame(sleeves).ffill()
    growth = (frame / frame.shift(1)).fillna(1.0)
    sessions = frame.index
    ends = set(pd.Series(sessions, index=sessions).groupby([sessions.year, sessions.quarter]).max())
    value = {k: initial * w for k, w in weights.items()}
    out = np.empty(len(sessions))
    for t, day in enumerate(sessions):
        for k in value:
            value[k] *= growth.at[day, k]
        total = sum(value.values())
        if day in ends:
            fee = sum(abs(total * w - value[k]) * (STOCK_COST if k in stock_sleeves else ETF_COST) * cost_mult
                      for k, w in weights.items())
            total -= fee
            value = {k: total * w for k, w in weights.items()}
        out[t] = sum(value.values())
    return pd.Series(out, index=sessions)


def summarize(navs):
    rows = []
    for nav in navs:
        stats = metrics(nav)
        stats.update({k: segment_cagr(nav, a, b) for k, (a, b) in L.SEGMENTS.items()})
        stats['ytd_2026'] = float(nav.iloc[-1] / nav.loc[:'2025-12-31'].iloc[-1] - 1)
        stats.pop('yearly')
        rows.append(stats)
    frame = pd.DataFrame(rows)
    cols = ['cagr', 'max_drawdown', *L.SEGMENTS, 'ytd_2026']
    return {'runs': len(frame), 'median': frame[cols].median().round(5).to_dict(),
            'p10': frame[['cagr', 'max_drawdown']].quantile(0.1).round(5).to_dict(),
            'share_cagr15_dd25': float(((frame.cagr >= .15) & (frame.max_drawdown >= -.25)).mean())}


def run(name, small_path, theme_path, small_w, theme_w, cost_mult):
    small = pd.read_parquet(small_path)
    index = small.index
    ret, price = E7.etf_prices(index)
    price = price.reindex(index)
    theme = None
    if theme_path != 'none':
        navs = pd.read_parquet(theme_path)
        theme = (navs / navs.iloc[0]).median(axis=1).reindex(index).ffill()
    weights = {'small': small_w, 'gold': 0.15, 'bond': 0.15, 'ndx': 0.10}
    if theme is not None:
        weights['theme'] = theme_w
    stock_sleeves = {'small', 'theme'}
    results = []
    for column in small.columns:
        sleeves = {'small': small[column], 'gold': price['gold'], 'bond': price['bond'], 'ndx': price['ndx']}
        if theme is not None:
            sleeves['theme'] = theme
        results.append(combine(sleeves, weights, stock_sleeves, cost_mult))
    summary = summarize(results)
    summary.update(name=name, weights=weights, cost_mult=cost_mult, small=str(small_path), theme=str(theme_path))
    (OUT / f'combo_{name}_cost{cost_mult:g}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('small')
    parser.add_argument('theme')
    parser.add_argument('--small-w', type=float, default=0.35)
    parser.add_argument('--theme-w', type=float, default=0.25)
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    run(args.name, args.small, args.theme, args.small_w, args.theme_w, args.cost)
