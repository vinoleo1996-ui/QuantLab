#!/usr/bin/env python3
"""P4 allocation + IPO overlay (docs/prereg/P4_ALLOC.md). Needs output/p4_alpha_20261012/report.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p3_ipo as IPO
from quantlab import p4_alloc as A
from quantlab import research_trials as T
from quantlab.p4_mandate import MANDATE

PREREG = 'docs/prereg/P4_ALLOC.md'
ALPHA = ROOT / 'output/p4_alpha_20261012'
OUT = ROOT / 'output/p4_alloc_20261012'
START, END = '2018-01-02', '2026-09-22'
SHARES = {'ALLOC60': 0.60, 'ALLOC50': 0.50, 'ALLOC70': 0.70}


def etf_returns(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide[list(A.ETF_CODES.values())].rename(columns={v: k for k, v in A.ETF_CODES.items()})
    wide = wide.reindex(wide.index.union(sessions)).ffill().reindex(sessions)
    return wide.pct_change(fill_method=None).fillna(0.0)


def ipo_increment(equity_value_by_day):
    """Expected IPO gain per year over the recent window, as a share of the 800k account."""
    d = IPO.load()
    dates = pd.DatetimeIndex(d.date.unique())
    by_date = equity_value_by_day.reindex(equity_value_by_day.index.union(dates)).ffill().reindex(dates)
    half = d.date.map(by_date).fillna(0.0).to_numpy() / 2
    per = np.select([d.board.eq('sh_main')], [np.floor(half / 10_000) * 1000], np.floor(half / 5000) * 500)
    shares = pd.Series(np.minimum(per, d.upper.to_numpy()), index=d.index)
    profit = IPO.expected(d, shares, d.sell)
    mask = d.date >= pd.Timestamp(IPO.RECENT)
    years = (d.date.max() - pd.Timestamp(IPO.RECENT)).days / 365.25
    return float(profit[mask].sum() / years / MANDATE.capital_cny)


def main():
    report_path = ALPHA / 'report.json'
    if not report_path.exists():
        raise SystemExit(json.dumps(dict(status='blocked', reason='P4_alpha not judged yet')))
    book = json.loads(report_path.read_text())['equity_book']
    prefix = f'{book}_'
    rows, ipo_values = [], []
    for cost in (1, 2):
        for path in sorted((ALPHA / 'ledger').glob(f'nav_{prefix}*_cost{cost}.csv')):
            equity = pd.read_csv(path, index_col=0, parse_dates=True)['nav'].loc[START:END]
            etf = etf_returns(equity.index)
            for name, share in SHARES.items():
                nav, trace = A.allocate(equity, etf, share, cost_mult=cost)
                rows.append(dict(path=path.stem, cost=cost, variant=name, **A.stats(nav), **trace['mean_weights'],
                                 disaster_triggers=trace['disaster_triggers']))
                if cost == 1 and name == 'ALLOC60':
                    ipo_values.append(ipo_increment(nav * MANDATE.capital_cny * share))
    results = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / 'results.csv', index=False)
    med = results.groupby(['cost', 'variant'])[['cagr', 'mdd']].median()
    ipo = float(np.median(ipo_values))
    credit = min(ipo, MANDATE.ipo_credit_cap)
    main_row, cost2 = med.loc[(1, 'ALLOC60')], med.loc[(2, 'ALLOC60')]
    gates = {'cost1_cagr_plus_ipo_ge_12': main_row.cagr + credit >= MANDATE.min_net_cagr,
             'mdd_ge_-25': main_row.mdd >= -MANDATE.max_drawdown,
             'cost2_cagr_plus_ipo_ge_10': cost2.cagr + credit >= MANDATE.min_net_cagr_cost2}
    report = dict(equity_book=book, ipo_expected_recent=ipo, ipo_credit=credit,
                  main=dict(cagr=main_row.cagr, mdd=main_row.mdd, cagr_with_ipo=main_row.cagr + credit),
                  cost2=dict(cagr=cost2.cagr, mdd=cost2.mdd), gates=gates, passed=all(gates.values()),
                  target_15_reached=bool(main_row.cagr + credit >= MANDATE.target_net_cagr),
                  by_variant={f'{v}@cost{c}': dict(cagr=r.cagr, mdd=r.mdd) for (c, v), r in med.iterrows()},
                  mean_weights=results[(results.cost == 1) & (results.variant == 'ALLOC60')]
                  [['equity', 'gold', 'nasdaq', 'sp500', 'treasury']].median().to_dict(),
                  mandate=MANDATE.version, prereg=PREREG, prereg_sha256=T.sha256_file(ROOT / PREREG),
                  promotion_approved=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
