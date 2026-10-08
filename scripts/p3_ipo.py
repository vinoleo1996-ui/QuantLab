#!/usr/bin/env python3
"""IPO-lottery increment estimate (docs/prereg/P3_IPO.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/p3_ipo_20261008'
ACCOUNT = 600_000
START, END, RECENT = '2016-01-01', '2026-09-30', '2023-10-01'
SEGMENTS = {'2016-2018': ('2016-01-01', '2018-12-31'), '2019-2022': ('2019-01-01', '2022-12-31'),
            'recent_3y': (RECENT, END), 'full': (START, END)}


def load():
    d = pd.read_parquet(ROOT / 'data/ipo/ipo.parquet')
    d = d[(d.IS_BEIJING == 0) & d.ISSUE_PRICE.notna() & d.ONLINE_ISSUE_LWR.notna()].copy()
    d['code'] = d.SECUCODE.str[:6]
    listing = pd.to_datetime(d.LISTING_DATE)
    d['date'] = pd.to_datetime(d.APPLY_DATE).fillna(listing - pd.Timedelta(days=10))
    d = d[(d.date >= START) & (d.date <= END)].copy()
    d['board'] = np.where(d.code.str.startswith('688'), 'star',
                          np.where(d.SECUCODE.str.endswith('.SH'), 'sh_main',
                                   np.where(d.code.str[:3].isin(['300', '301']), 'chinext', 'sz_main')))
    d['unit'] = pd.to_numeric(d.EACHBALLOT_SHARES, errors='coerce').where(lambda u: u > 0)
    d['unit'] = d.unit.fillna(d.board.map({'sh_main': 1000, 'star': 500, 'chinext': 500, 'sz_main': 500}))
    d['upper'] = d.ONLINE_APPLY_UPPER
    med = d.groupby([d.board, d.date.dt.year]).upper.transform('median')
    d['upper'] = d.upper.fillna(med).fillna(d.groupby('board').upper.transform('median'))
    words = pd.to_numeric(d.CONTINUOUS_1WORD_NUM, errors='coerce').fillna(0)
    d['sell'] = np.where((words > 0) & d.OPEN_AVERAGE_PRICE.notna(), d.OPEN_AVERAGE_PRICE, d.LD_AVERAGE_PRICE)
    d['sell_close'] = d.ISSUE_PRICE * (1 + d.LD_CLOSE_CHANGE / 100)
    return d[d.sell.notna()].reset_index(drop=True)


def applied(d, holding, main_only=False):
    half = holding / 2
    per = {'sh_main': np.floor(half / 10_000) * 1000, 'star': np.floor(half / 5000) * 500,
           'sz_main': np.floor(half / 5000) * 500, 'chinext': np.floor(half / 5000) * 500}
    shares = np.minimum(d.board.map(per).astype(float), d.upper)
    if main_only:
        shares = shares.where(d.board.isin(['sh_main', 'sz_main']), 0.0)
    return shares


def expected(d, shares, sell):
    won = shares * d.ONLINE_ISSUE_LWR / 100
    lots = won / d.unit
    return won * (sell - d.ISSUE_PRICE) - won * sell * 0.0015 - lots * 5


def yearly(d, profit):
    return profit.groupby(d.date.dt.year).sum()


def segment(d, profit):
    out = {}
    for name, (a, b) in SEGMENTS.items():
        mask = (d.date >= a) & (d.date <= b)
        years = (min(pd.Timestamp(b), d.date.max()) - pd.Timestamp(a)).days / 365.25
        out[name] = round(float(profit[mask].sum() / years / ACCOUNT * 100), 2)
    return out


def monte_carlo(d, shares, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    numbers = np.floor(shares / d.unit).astype(int).values
    p = (d.ONLINE_ISSUE_LWR / 100).clip(0, 1).values
    gain = (d.unit * (d.sell * (1 - 0.0015) - d.ISSUE_PRICE) - 5).values
    years = d.date.dt.year.values
    hits = rng.binomial(numbers[None, :].repeat(n, 0), p[None, :])
    frame = pd.DataFrame(hits * gain[None, :]).T.groupby(years).sum().T / ACCOUNT * 100
    return {int(y): [round(float(v), 2) for v in frame[y].quantile([0.1, 0.5, 0.9])] for y in frame.columns}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    main_shares = applied(d, ACCOUNT)
    main = expected(d, main_shares, d.sell)
    report = {'rows': len(d), 'by_board': d.board.value_counts().to_dict(),
              'main_yearly_yuan': {int(k): round(float(v)) for k, v in yearly(d, main).items()},
              'main_yearly_pct': {int(k): round(float(v / ACCOUNT * 100), 2) for k, v in yearly(d, main).items()},
              'main_segments_pct': segment(d, main),
              'holding_scan_segments_pct': {h: segment(d, expected(d, applied(d, h), d.sell))
                                           for h in (600_000, 300_000, 100_000)},
              'main_board_only_segments_pct': segment(d, expected(d, applied(d, ACCOUNT, True), d.sell)),
              'first_close_sensitivity_segments_pct': segment(d, expected(d, main_shares, d.sell_close.fillna(d.sell))),
              'monte_carlo_yearly_pct_p10_p50_p90': monte_carlo(d, main_shares),
              'recent_by_board_yuan_per_year': {
                  b: round(float(main[(d.board == b) & (d.date >= RECENT)].sum() / 3)) for b in d.board.unique()}}
    rec = report['main_segments_pct']['recent_3y']
    report['creditable_increment_pct'] = rec
    report['decision'] = 'creditable' if rec >= 0.5 else 'not_a_fill_item'
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
