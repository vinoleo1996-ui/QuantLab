#!/usr/bin/env python3
"""E4: equity leg 60 percent + gold 15 + 10y treasury 15 + Nasdaq QDII 10, quarter-end rebalance.

Each saved equity run (seeds 1-20 x 5 phases) is combined with the same ETF sleeves.
Rebalance transfers cost 0.4 percent on the equity side and 0.05 percent on ETF sides.
A diagnostic with the gold sleeve held as cash isolates gold's contribution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantlab.smallcap.sim import metrics, segment_cagr

OUT = ROOT / 'output/smallcap_rebuild_20261006'
WEIGHTS = {'equity': 0.60, 'sh518880': 0.15, 'sh511260': 0.15, 'sh513100': 0.10}
COST = {'equity': 0.004, 'etf': 0.0005}
SEGMENTS = {'2018_2021': ('2018-01-02', '2021-12-31'), '2022_2025': ('2022-01-01', '2025-12-31'),
            '2025_now': ('2025-01-02', '2026-09-22')}


def etf_returns(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide = wide.reindex(wide.index.union(sessions)).ffill().reindex(sessions)
    return wide.pct_change(fill_method=None).fillna(0.0)


def quarter_ends(sessions):
    frame = pd.Series(sessions, index=sessions)
    return set(frame.groupby([sessions.year, sessions.quarter]).max())


def combine(equity_nav, etf, weights=WEIGHTS, initial=600_000.0):
    sessions = equity_nav.index
    eq_ret = equity_nav.pct_change(fill_method=None).fillna(equity_nav.iloc[0] / initial - 1)
    sleeves = {name: initial * w for name, w in weights.items()}
    ends = quarter_ends(sessions)
    out = np.empty(len(sessions))
    costs = 0.0
    for t, day in enumerate(sessions):
        sleeves['equity'] *= 1 + eq_ret.iloc[t]
        for name in weights:
            if name != 'equity':
                sleeves[name] *= 1 + (etf.at[day, name] if name in etf.columns else 0.0)
        total = sum(sleeves.values())
        if day in ends and len(weights) > 1:
            fee = sum(abs(total * w - sleeves[name]) * (COST['equity'] if name == 'equity' else COST['etf'])
                      for name, w in weights.items())
            costs += fee
            total -= fee
            sleeves = {name: total * w for name, w in weights.items()}
        out[t] = sum(sleeves.values())
    return pd.Series(out, index=sessions), costs


def evaluate(navs, etf, weights):
    rows = []
    for name in navs.columns:
        nav, costs = combine(navs[name], etf, weights)
        stats = metrics(nav)
        stats.update({k: segment_cagr(nav, a, b) for k, (a, b) in SEGMENTS.items()})
        stats['rebalance_costs'] = costs
        stats.pop('yearly')
        rows.append(stats)
    frame = pd.DataFrame(rows)
    columns = ['cagr', 'max_drawdown', *SEGMENTS]
    return {'runs': len(frame), 'median': {c: float(frame[c].median()) for c in columns},
            'p10': {c: float(frame[c].quantile(0.1)) for c in ('cagr', 'max_drawdown')},
            'p90': {c: float(frame[c].quantile(0.9)) for c in ('cagr', 'max_drawdown')},
            'share_cagr15_dd25': float(((frame.cagr >= 0.15) & (frame.max_drawdown >= -0.25)).mean()),
            'trough_years': frame.trough.str[:4].value_counts().to_dict()}


def main(config='A_adv_cost1'):
    navs = pd.read_parquet(OUT / f'equity/{config}/navs.parquet')
    etf = etf_returns(navs.index)
    result = {'config': config, 'weights': WEIGHTS,
              'equity_only': evaluate(navs, etf, {'equity': 1.0}),
              'multi_asset': evaluate(navs, etf, WEIGHTS),
              'gold_as_cash_diagnostic': evaluate(navs, etf.assign(sh518880=0.0), WEIGHTS)}
    singles = {}
    for code in ('sh518880', 'sh511260', 'sh513100'):
        series = (1 + etf[code]).cumprod() * 600_000
        stats = metrics(series)
        stats.update({k: segment_cagr(series, a, b) for k, (a, b) in SEGMENTS.items()})
        stats.pop('yearly')
        singles[code] = stats
    result['etf_alone'] = singles
    result['correlation_daily'] = pd.concat({'equity': navs.pct_change(fill_method=None).mean(axis=1), **{c: etf[c] for c in singles}},
                                            axis=1).dropna().corr().round(3).to_dict()
    result['promotion_approved'] = False
    path = OUT / f'E4_{config}.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ('equity_only', 'multi_asset', 'gold_as_cash_diagnostic')}, indent=1))
    print(json.dumps(result['etf_alone'], indent=1))
    print(json.dumps(result['correlation_daily'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
