#!/usr/bin/env python3
"""E0: is the broad basket just the CSI 2000? Same window, index net of a 0.5 percent ETF fee,
and an OLS of the basket's daily return on CSI 300 total return and an in-house SMB."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantlab.smallcap.panel import Panel
from quantlab.smallcap.sim import metrics, segment_cagr

OUT = ROOT / 'output/smallcap_rebuild_20261006'
CSI = ROOT / 'data/smallcap_assets/csindex/bench_raw'
START, END = '2018-01-02', '2026-09-22'
ETF_FEE = 0.005
SEGMENTS = {'2018_2021': ('2018-01-02', '2021-12-31'), '2022_2025': ('2022-01-01', '2025-12-31'),
            '2025_now': ('2025-01-02', '2026-09-22')}


def index_series(name):
    data = json.loads((CSI / f'csindex_{name}.json').read_text())['data']
    series = pd.Series({pd.Timestamp(row['tradeDate']): float(row['close']) for row in data}).sort_index()
    return series


def smb(panel, sessions):
    returns = panel.field('adj_close').pct_change(fill_method=None).reindex(sessions)
    cap = panel.field('float_mktcap').rolling(20, min_periods=10).mean().reindex(sessions)
    out = pd.Series(np.nan, index=sessions)
    small = big = None
    for t, day in enumerate(sessions):
        if t % 20 == 0:
            row = cap.loc[day].dropna()
            row = row[row > 0]
            low, high = row.quantile(0.3), row.quantile(0.7)
            small, big = row[row <= low].index, row[row >= high].index
        if t == 0 or small is None:
            continue
        r = returns.loc[day]
        out.loc[day] = r.reindex(small).mean() - r.reindex(big).mean()
    return out


def ols(y, x):
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    sigma2 = resid @ resid / (len(y) - x.shape[1])
    cov = sigma2 * np.linalg.inv(x.T @ x)
    t = beta / np.sqrt(np.diag(cov))
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return beta, t, r2


def main():
    panel = Panel('20261006T152412Z')
    navs = pd.read_parquet(OUT / 'equity/A_adv_cost1/navs.parquet')
    sessions = navs.index
    basket = navs.pct_change(fill_method=None).mean(axis=1).iloc[1:]
    csi2000 = index_series('932000CNY010').reindex(sessions).ffill()
    csi300 = index_series('H00300').reindex(sessions).ffill()
    fee_daily = (1 - ETF_FEE) ** (1 / 252)
    index_nav = 600_000 * csi2000 / csi2000.iloc[0]
    index_nav = index_nav * pd.Series(fee_daily ** np.arange(len(index_nav)), index=index_nav.index)
    index_stats = metrics(index_nav)
    index_stats.update({k: segment_cagr(index_nav, a, b) for k, (a, b) in SEGMENTS.items()})
    runs = pd.read_csv(OUT / 'equity/A_adv_cost1/runs.csv')
    factor = smb(panel, sessions).iloc[1:]
    mkt = csi300.pct_change(fill_method=None).iloc[1:]
    c2k = csi2000.pct_change(fill_method=None).iloc[1:]
    frame = pd.concat({'y': basket, 'mkt': mkt, 'smb': factor, 'c2k': c2k}, axis=1).dropna()
    frame = frame[frame.y.abs() > 0]
    b1, t1, r1 = ols(frame.y.to_numpy(), frame[['mkt', 'smb']].to_numpy())
    b2, t2, r2 = ols(frame.y.to_numpy(), frame[['c2k']].to_numpy())
    explained = (b1[1] * frame.smb.mean()) / frame.y.mean()
    result = {
        'basket_A_adv_median': {'cagr': float(runs.cagr.median()), 'max_drawdown': float(runs.max_drawdown.median()),
                                **{k: float(runs[k].median()) for k in SEGMENTS}},
        'csi2000_tr_net_0p5pct': index_stats,
        'gate_passed': bool(runs.cagr.median() > index_stats['cagr']),
        'regression_mkt_smb': {'alpha_annual': float(b1[0] * 252), 'alpha_t': float(t1[0]),
                               'beta_mkt': float(b1[1 - 1 + 1]), 'beta_smb': float(b1[2]),
                               't_mkt': float(t1[1]), 't_smb': float(t1[2]), 'r2': float(r1),
                               'smb_share_of_mean_return': float(b1[2] * frame.smb.mean() / frame.y.mean())},
        'regression_csi2000': {'alpha_annual': float(b2[0] * 252), 'alpha_t': float(t2[0]),
                               'beta': float(b2[1]), 'r2': float(r2)},
        'days': int(len(frame)),
        'note': 'basket return = mean daily return of 100 saved runs (seeds 1-20 x 5 phases); index has no trading costs',
        'promotion_approved': False,
    }
    result['regression_mkt_smb'].pop('smb_share_of_mean_return')
    result['regression_mkt_smb']['smb_contribution_share'] = float(b1[2] * frame.smb.mean() / frame.y.mean())
    (OUT / 'E0.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
