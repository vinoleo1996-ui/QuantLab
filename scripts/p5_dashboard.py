#!/usr/bin/env python3
"""P5 dashboard data: annual returns, annual drawdowns, cumulative NAV, provenance checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import os
ROOT = Path(os.environ.get('QUANTLAB_ROOT', '/Users/agiuser/Desktop/QuantLab'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alloc as R
from quantlab import p4_alloc as A

START, END = '2018-01-02', '2026-09-22'
ALPHA = ROOT / 'output/p4_alpha_20261012/ledger'
GROSS = ROOT / 'output/p4_gross_20261013/ledger'
OUT = ROOT / 'output/p5_g0_20261017'

ORIG_TARGET, ORIG_MONTH = A.target_weights, A.month_ends


def g0_target(etf_returns_upto, share):
    s = 1.0 - share
    return np.array([share, 0.375 * s, 0.375 * s, 0.25 * s])


def quarter_ends(sessions):
    s = pd.Series(sessions, index=sessions)
    return set(s.groupby([sessions.year, (sessions.month - 1) // 3]).max())


def etf_g0(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide[['sh518880', 'sh511260', 'sh513100']].rename(
        columns={'sh518880': 'gold', 'sh511260': 'treasury', 'sh513100': 'nasdaq'})
    wide = wide.reindex(wide.index.union(sessions)).ffill().reindex(sessions)
    return wide.pct_change(fill_method=None).fillna(0.0)


def annual(nav):
    year = nav.index.year
    out_r, out_d = {}, {}
    for y in sorted(set(year)):
        seg = nav[year == y]
        out_r[str(y)] = float(seg.iloc[-1] / seg.iloc[0] - 1)
        out_d[str(y)] = float((seg / seg.cummax() - 1).min())
    return out_r, out_d


def run(folder, prefix, share, line, sleeve, cost=1):
    A.target_weights = g0_target if sleeve == 'g0' else ORIG_TARGET
    A.month_ends = quarter_ends if sleeve == 'g0' else ORIG_MONTH
    A.DISASTER_DD, A.RECOVER_DD, A.DISASTER_EQUITY = line, 0.08, 0.5
    rows, navs = [], []
    for p in sorted(folder.glob(f'nav_{prefix}p*_cost{cost}.csv')):
        eq = pd.read_csv(p, index_col=0, parse_dates=True)['nav'].loc[START:END]
        etf = etf_g0(eq.index) if sleeve == 'g0' else R.etf_returns(eq.index)
        prem = R.qdii_premium_frame(eq.index)
        nav, tr = A.allocate(eq, etf, share, cost_mult=cost, premium=prem)
        r, d = annual(nav)
        rows.append(dict(path=p.stem, cagr=A.stats(nav)['cagr'], mdd=A.stats(nav)['mdd'],
                         annual=r, annual_mdd=d, triggers=tr['disaster_triggers']))
        navs.append(nav)
    frame = pd.DataFrame(rows)
    years = sorted(rows[0]['annual'])
    med_r = {y: float(np.median([r['annual'][y] for r in rows])) for y in years}
    med_d = {y: float(np.median([r['annual_mdd'][y] for r in rows])) for y in years}
    cagrs = sorted(r['cagr'] for r in rows)
    # representative path: median CAGR
    idx = int(np.argsort([r['cagr'] for r in rows])[len(rows) // 2])
    nav = navs[idx]
    monthly = nav.resample('ME').last()
    return dict(cagr=float(np.median(cagrs)), mdd=float(np.median([r['mdd'] for r in rows])),
                cagr_p10=float(np.percentile(cagrs, 10)), cagr_p90=float(np.percentile(cagrs, 90)),
                annual=med_r, annual_mdd=med_d,
                nav=[round(float(v), 6) for v in monthly.to_numpy()],
                nav_dates=[d.strftime('%Y-%m-%d') for d in monthly.index],
                triggers=[r['triggers'] for r in rows])


def provenance():
    """Daily-return extremes of the stock books and the sleeve, plus limit-band sanity."""
    out = {}
    for name, folder, prefix in (('POOL50', ALPHA, 'POOL50_'), ('G97', GROSS, 'POOL50_G97_')):
        files = sorted(folder.glob(f'nav_{prefix}p0_s1_cost1.csv'))
        eq = pd.read_csv(files[0], index_col=0, parse_dates=True)['nav'].loc[START:END]
        r = eq.pct_change().dropna()
        out[name] = dict(n=len(r), max=float(r.max()), min=float(r.min()),
                         gt5=int((r.abs() > 0.05).sum()), gt3=int((r.abs() > 0.03).sum()),
                         top=[(str(d.date()), round(float(v), 4)) for d, v in r.nlargest(3).items()],
                         bottom=[(str(d.date()), round(float(v), 4)) for d, v in r.nsmallest(3).items()])
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    for code in ('sh518880', 'sh511260', 'sh513100'):
        s = wide[code].loc[START:END].pct_change().dropna()
        out[code] = dict(n=len(s), max=float(s.max()), min=float(s.min()),
                         gt5=int((s.abs() > 0.05).sum()))
    return out


def main():
    specs = [
        ('baseline', ALPHA, 'POOL50_', 0.60, 0.18, 'g1'),
        ('g0_60', ALPHA, 'POOL50_', 0.60, 0.23, 'g0'),
        ('g97_65', GROSS, 'POOL50_G97_', 0.65, 0.23, 'g0'),
        ('g97_70', GROSS, 'POOL50_G97_', 0.70, 1.0, 'g0'),
    ]
    data = {'window': [START, END], 'series': {}}
    for name, folder, prefix, share, line, sleeve in specs:
        data['series'][name] = run(folder, prefix, share, line, sleeve)
        print(name, round(data['series'][name]['cagr'] * 100, 2),
              round(data['series'][name]['mdd'] * 100, 2))
    data['provenance'] = provenance()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'annual.json').write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(json.dumps(data['provenance'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
