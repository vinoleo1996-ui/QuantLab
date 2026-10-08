#!/usr/bin/env python3
"""P2 M2 weight switch on external indicators (docs/prereg/P2_M2_TILT.md).

  python scripts/p2_m2_tilt.py

Writes output/p2_m2_tilt_20261008/{indicators.csv, results.csv, report.json}.
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
from quantlab.smallcap.panel import Panel

PREREG = 'docs/prereg/P2_M2_TILT.md'
OUT = ROOT / 'output/p2_m2_tilt_20261008'
START, END = '2018-01-02', '2026-09-22'
PATHS = [f'{p}_{s}' for p in (0, 4, 8, 12, 16) for s in (1, 2)]
VOL, PCT_WINDOW, PCT_MIN = 120, 750, 250
COST = {'m2': 0.004, 'bond': 0.0005}


def trailing_pct(series):
    values = series.to_numpy()
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[max(0, i - PCT_WINDOW):i]
        window = window[np.isfinite(window)]
        if len(window) >= PCT_MIN and np.isfinite(values[i]):
            out[i] = (window < values[i]).mean()
    return pd.Series(out, index=series.index)


def indicators(sessions):
    panel = Panel('20261006T152412Z')
    amount = panel.field('amount').astype('float64').sum(axis=1).rolling(20, min_periods=15).mean()
    t1 = trailing_pct(amount)
    yields = pd.read_parquet(ROOT / 'data/macro_rates/treasury_yields.parquet').set_index('date').cn_10y
    yields = yields.reindex(yields.index.union(sessions)).ffill().reindex(sessions).shift(1)
    t2 = (yields - yields.shift(60)) * 100
    curves = pd.read_parquet(ROOT / 'data/macro_rates/chinabond_curves.parquet').set_index('date')
    spread = curves.credit_spread_3y.reindex(curves.index.union(sessions)).ffill().reindex(sessions).shift(1)
    t3 = trailing_pct(spread)
    since = curves.loc['2015-01-01':].credit_spread_3y
    sessions15 = sessions[sessions >= '2015-01-01']
    missing = 1 - since.dropna().index.isin(sessions15).sum() / len(sessions15)
    frame = pd.DataFrame({'amount_pct': t1.reindex(sessions), 'yield_chg_bp': t2, 'spread_pct': t3})
    return frame, float(missing)


def states(frame):
    s = pd.DataFrame(index=frame.index)
    s['T1'] = np.select([frame.amount_pct >= 0.7, frame.amount_pct <= 0.3], [1, -1], 0)
    s['T2'] = np.select([frame.yield_chg_bp <= -20, frame.yield_chg_bp >= 20], [1, -1], 0)
    s['T3'] = np.select([frame.spread_pct <= 0.3, frame.spread_pct >= 0.7], [1, -1], 0)
    s['T4'] = s[['T1', 'T2', 'T3']].mean(axis=1)
    return 1 + 0.5 * s


def combine(m2, bond, k, signal_days, cost_mult=1.0):
    """Monthly risk parity of M2 vs bond, M2 weight scaled by k; drift within the month."""
    r = pd.DataFrame({'m2': m2.pct_change(), 'bond': bond.pct_change()}).dropna()
    vol = r.rolling(VOL, min_periods=VOL).std()
    weights = None
    sleeves = np.array([0.5, 0.5])
    nav, values, mweights = 1.0, [], []
    pending = None
    for day, row in r.iterrows():
        if pending is not None:
            current = sleeves / sleeves.sum()
            cost = abs(pending[0] - current[0]) * COST['m2'] + abs(pending[1] - current[1]) * COST['bond']
            nav *= 1 - cost * cost_mult
            sleeves = pending.copy()
            pending = None
        growth = sleeves * (1 + row[['m2', 'bond']].to_numpy())
        nav *= growth.sum() / sleeves.sum()
        sleeves = growth / growth.sum()
        values.append(nav)
        mweights.append(sleeves[0])
        if day in signal_days:
            v = vol.loc[day]
            base = 0.5 if not np.isfinite(v).all() else (1 / v['m2']) / (1 / v['m2'] + 1 / v['bond'])
            w = min(base * k.get(day, 1.0), 1.0)
            pending = np.array([w, 1 - w])
    out = pd.Series(values, index=r.index)
    return out, float(np.mean(mweights))


def stats(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1), float((nav / nav.cummax() - 1).min())


def main():
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    bond = bars[bars.code == 'sh511260'].set_index('date').close_hfq.sort_index()
    navs = {p: pd.read_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{p}_cost1.csv', index_col=0,
                           parse_dates=True)['nav'] for p in PATHS}
    navs2 = {p: pd.read_csv(ROOT / f'output/p1_m2_20261008/is/ledger_nav_{p}_cost2.csv', index_col=0,
                            parse_dates=True)['nav'] for p in PATHS}
    sessions = navs[PATHS[0]].index
    bond = bond.reindex(bond.index.union(sessions)).ffill().reindex(sessions)
    signal_days = set(month_ends(sessions, START, END))
    frame, missing = indicators(Panel('20261006T152412Z').sessions)
    frame = frame.reindex(sessions)
    ks = states(frame)
    run = ['T1', 'T2'] + (['T3', 'T4'] if missing <= 0.05 else [])
    rows = []
    for path in PATHS:
        for cost_mult, source in ((1.0, navs), (2.0, navs2)):
            base, base_w = combine(source[path], bond, {}, signal_days, cost_mult)
            bc, bd = stats(base)
            rows.append({'path': path, 'cost_mult': cost_mult, 'rule': 'fixed', 'cagr': bc, 'mdd': bd, 'm2_weight': base_w})
            for name in run:
                nav, w = combine(source[path], bond, ks[name].to_dict(), signal_days, cost_mult)
                c, d = stats(nav)
                rows.append({'path': path, 'cost_mult': cost_mult, 'rule': name, 'cagr': c, 'mdd': d, 'm2_weight': w})
    results = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.assign(**{f'k_{n}': ks[n] for n in ks}).to_csv(OUT / 'indicators.csv')
    results.to_csv(OUT / 'results.csv', index=False)
    median = results.groupby(['cost_mult', 'rule'])[['cagr', 'mdd', 'm2_weight']].median()
    fixed = median.loc[(1.0, 'fixed')]
    report = {'credit_spread_missing_share_since_2015': missing, 'run': run,
              'not_run_insufficient_data': [n for n in ('T3', 'T4') if n not in run],
              'fixed': fixed.to_dict(), 'rules': {}}
    for name in run:
        m = median.loc[(1.0, name)]
        d_cagr, d_mdd = (m.cagr - fixed.cagr) * 100, (m.mdd - fixed.mdd) * 100
        gate = {'cagr+0.5_and_mdd_not_worse': bool(d_cagr >= 0.5 and d_mdd >= 0),
                'mdd+3_and_cagr_loss<=0.5': bool(d_mdd >= 3 and d_cagr >= -0.5)}
        gate['pass'] = any(gate.values())
        report['rules'][name] = {'cagr': m.cagr, 'mdd': m.mdd, 'm2_weight': m.m2_weight,
                                 'delta_cagr_pp': d_cagr, 'delta_mdd_pp': d_mdd,
                                 'cost2_cagr': float(median.loc[(2.0, name)].cagr),
                                 'm2_sleeve_scaled_delta_pp': d_cagr / max(m.m2_weight, 1e-9),
                                 'k_share': ks[name].loc[START:END].value_counts(normalize=True).round(3).to_dict(),
                                 'gate': gate}
    report.update(prereg=PREREG, prereg_sha256=T.sha256_file(ROOT / PREREG), promotion_approved=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, default=float))


if __name__ == '__main__':
    main()
