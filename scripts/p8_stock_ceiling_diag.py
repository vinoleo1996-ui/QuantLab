#!/usr/bin/env python3
"""Where does the stock sleeve stop? Concentration and a 2-D ADV x turnover pool.

Zero-cost diagnostic, not a trial: random equal-weight baskets rebalanced every 20 sessions.
The question is the trade-off curve (CAGR vs MDD vs Calmar) at 800k-compatible rules, so the
user can see whether "stocks higher" means more return for proportionally more risk.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

MANIFEST = '20261013T023232Z'
OUT = ROOT / 'output/p8_stock_ceiling_20261018'
START, END = '2018-01-02', '2026-09-22'
REBALANCE = 20
COST_PP = 0.012          # flat annual cost so every cell is compared on the same basis
BANDS = {'bottom40': 0.60, 'bottom30': 0.70, 'bottom20': 0.80}
SIZES = (20, 30, 40, 50)
SEEDS = range(1, 9)


def metrics(daily):
    nav = (1 + pd.Series(daily)).cumprod()
    years = len(nav) / 252
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) - COST_PP if years > 0 else float('nan')
    mdd = float((nav / nav.cummax() - 1).min())
    return cagr, mdd, (cagr / abs(mdd) if mdd < 0 else float('inf'))


def basket_returns(returns, days, bands, eligible, turnover, size, seed):
    rng = np.random.default_rng(seed)
    out = []
    for day in days:
        members = np.flatnonzero(bands.loc[day].to_numpy() & eligible.loc[day].to_numpy())
        if len(members) == 0:
            out.extend([0.0] * REBALANCE)
            continue
        chosen = rng.choice(members, size=min(size, len(members)), replace=False)
        codes = returns.columns[chosen]
        window = returns.loc[day:].iloc[1:REBALANCE + 1, chosen].mean(axis=1)
        out.extend(window.fillna(0.0).tolist())
    return out


def main():
    panel = Panel(MANIFEST)
    sessions = panel.sessions[(panel.sessions >= pd.Timestamp(START)) &
                              (panel.sessions <= pd.Timestamp(END))]
    fields = {name: panel.field(name).reindex(sessions) for name in ('adj_close', 'amount', 'is_st')}
    features = U.rolling_features(panel, with_st=True)
    adv = features['adv'].reindex(sessions)
    codes = [c for c in adv.columns if not str(c).startswith('bj.')]
    adv = adv.reindex(columns=codes)
    eligible = (features['history'].reindex(sessions).ge(253) & adv.notna() & adv.ge(5e6)
                & ~features['st'].reindex(sessions).eq(1)).reindex(columns=codes)
    returns = fields['adj_close'].reindex(columns=codes).pct_change(fill_method=None)
    turnover = panel.field('turnover').reindex(sessions).reindex(columns=codes)
    turnover20 = turnover.rolling(20, min_periods=10).mean()
    rank = adv.where(eligible).rank(axis=1, ascending=False, method='first')
    n = eligible.sum(axis=1)
    days = list(sessions[::REBALANCE])

    concentration = {}
    band40 = rank.gt(0.6 * n, axis=0) & eligible
    for size in SIZES:
        rows = []
        for seed in SEEDS:
            daily = basket_returns(returns, days, band40, eligible, turnover20, size, seed)
            rows.append(metrics(daily))
        frame = pd.DataFrame(rows, columns=['cagr', 'mdd', 'calmar'])
        concentration[str(size)] = {'cagr_median': float(frame.cagr.median()),
                                    'mdd_median': float(frame.mdd.median()),
                                    'calmar_median': float(frame.calmar.median()),
                                    'cagr_min': float(frame.cagr.min()),
                                    'mdd_worst': float(frame.mdd.min())}

    cells = {}
    for band, share in BANDS.items():
        members = rank.gt(share * n, axis=0) & eligible
        for tercile in ('low', 'mid', 'high'):
            cell = members.copy()
            for i, day in enumerate(sessions):
                in_band = members.loc[day].to_numpy()
                if not in_band.any():
                    continue
                values = turnover20.loc[day].to_numpy()
                valid = in_band & np.isfinite(values)
                if valid.sum() < 30:
                    cell.iloc[i] = False
                    continue
                cut_low, cut_high = np.quantile(values[valid], [1 / 3, 2 / 3])
                keep = (values <= cut_low) if tercile == 'low' else (
                    (values > cut_low) & (values <= cut_high) if tercile == 'mid' else values > cut_high)
                cell.iloc[i] = in_band & keep
            rows = []
            for seed in SEEDS:
                daily = basket_returns(returns, days, cell, eligible, turnover20, 50, seed)
                rows.append(metrics(daily))
            frame = pd.DataFrame(rows, columns=['cagr', 'mdd', 'calmar'])
            cells[f'{band}_{tercile}'] = {'cagr_median': float(frame.cagr.median()),
                                          'mdd_median': float(frame.mdd.median()),
                                          'calmar_median': float(frame.calmar.median()),
                                          'cagr_min': float(frame.cagr.min()),
                                          'mdd_worst': float(frame.mdd.min()),
                                          'members_median': float(members.sum(axis=1).median())}

    OUT.mkdir(parents=True, exist_ok=True)
    report = {'strategy': 'P8_STOCK_CEILING', 'stage': 'diagnostic', 'zero_cost': True,
              'cost_pp_applied': COST_PP, 'manifest': MANIFEST,
              'window': [START, END], 'rebalance_sessions': REBALANCE,
              'concentration': concentration, 'two_dimensional_pool': cells,
              'note': 'random baskets, equal weight, 8 seeds; not a trial and not a candidate',
              'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
