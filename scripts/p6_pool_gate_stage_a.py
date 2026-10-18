#!/usr/bin/env python3
"""P6 Stage A: does a point-in-time quality gate on the pool boundary add anything?

The gate keeps names whose latest known net income is non-negative and whose PB is in
(0, 3]; it is applied before the same bottom-40% ADV cut, so the book rules never change.
Zero-cost: both pools are equal-weight, daily-rebalanced adjusted-close portfolios, and the
paired excess is tested with a Newey-West t. Stage A only decides whether a 16-book Stage B
is worth registering.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab.alpha.features import Fundamentals
from quantlab.research_statistics import newey_west
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

MANIFEST = '20261013T023232Z'
FIN = ROOT / 'data/cloud/financials_pit/bundle_20261012/part-all-all.parquet'
OUT = ROOT / 'output/p6_pool_gate_stage_a_20261018'
START, END = '2018-01-02', '2026-09-22'
PREREG = ROOT / 'docs/prereg/P6_MULTI_CRITERIA_POOL.md'
PB_MAX = 3.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def income_asof(sessions, codes):
    columns = ['ts_code', 'ann_date', 'end_date', 'report_type', 'update_flag',
               'available_at', *Fundamentals.COLUMNS]
    table = pd.read_parquet(FIN, columns=columns)
    known = Fundamentals(table).table
    known = known[known.n_income_attr_p.notna()].copy()
    known['code'] = known['code'].astype(str)
    known = known[known.code.isin(codes)]
    position = np.searchsorted(sessions.values,
                               known.available.dt.tz_localize(None).values, side='left')
    known = known[position < len(sessions)].copy()
    position = position[position < len(sessions)]
    known['slot'] = sessions[position]
    latest = (known.sort_values(['slot', 'available'])
              .drop_duplicates(['slot', 'code'], keep='last')
              .pivot(index='slot', columns='code', values='n_income_attr_p'))
    latest = latest.reindex(sessions).ffill()
    return latest.reindex(columns=codes)


def bottom40(adv, eligible):
    ranked = adv.where(eligible).rank(axis=1, ascending=False, method='first')
    cut = np.ceil(0.6 * eligible.sum(axis=1))
    return ranked.gt(cut, axis=0) & eligible


def metrics(returns):
    values = returns.dropna()
    nav = (1 + values).cumprod()
    years = len(values) / 252
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else float('nan')
    mdd = float((nav / nav.cummax() - 1).min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float('inf')
    yearly = (1 + values).groupby(values.index.year).prod() - 1
    return {'cagr': cagr, 'mdd': mdd, 'calmar': calmar,
            'years': {int(k): float(v) for k, v in yearly.items()}}


def main():
    panel = Panel(MANIFEST)
    sessions = panel.sessions
    sessions = sessions[(sessions >= pd.Timestamp(START)) & (sessions <= pd.Timestamp(END))]
    features = U.rolling_features(panel, with_st=True)
    codes = [code for code in panel.field('close').columns if not str(code).startswith('bj.')]
    adv = features['adv'].reindex(sessions).reindex(columns=codes)
    history = features['history'].reindex(sessions)
    st = features['st'].reindex(sessions).reindex(columns=codes)
    pb = panel.field('pb_mrq').reindex(sessions).reindex(columns=codes)
    returns = (panel.field('adj_close').reindex(sessions).reindex(columns=codes)
               .pct_change(fill_method=None))

    income = income_asof(sessions, codes)
    gate = income.ge(0) & pb.gt(0) & pb.le(PB_MAX)
    gate = gate.reindex(columns=adv.columns)

    base = history.reindex(columns=codes).ge(253) & adv.notna() & adv.ge(5e6) & ~st.eq(1)
    gated = base & gate
    pool_base = bottom40(adv, base)
    pool_gated = bottom40(adv, gated)

    # spot-check the vectorised band against the canonical pool_on on 20 sessions
    rng = np.random.default_rng(20261018)
    check_days = sorted(rng.choice(np.arange(20, len(sessions)), 20, replace=False))
    overlaps = []
    for i in check_days:
        day = sessions[i]
        canonical = set(U.pool_on(day, features, 'adv', band='relative', exclude_st=True))
        vector = set(pool_base.columns[pool_base.iloc[i].to_numpy()])
        overlaps.append(len(canonical & vector) / max(1, len(canonical)))
    coverage = pd.Series(gated.sum(axis=1) / base.sum(axis=1), index=sessions)

    def daily(pool):
        values = [np.nan]
        for i in range(1, len(sessions)):
            members = pool.iloc[i - 1].to_numpy()
            if not members.any():
                values.append(np.nan)
                continue
            row = returns.iloc[i].to_numpy()[np.flatnonzero(members)]
            values.append(float(np.nanmean(row)))
        return pd.Series(values, index=sessions)

    ret_base, ret_gated = daily(pool_base), daily(pool_gated)
    paired = (ret_gated - ret_base).dropna()
    t = float(newey_west(paired.to_numpy(), lags=20)[1][0])
    p = float(2 * (1 - norm.cdf(abs(t))))
    ex2025 = paired[paired.index < '2025-01-01']
    base_metrics, gated_metrics = metrics(ret_base), metrics(ret_gated)
    turnover = float((pool_gated.fillna(False).astype(int).diff().abs().sum(axis=1)
                      / pool_gated.sum(axis=1).replace(0, np.nan)).mean())

    gates = {'nw_t_ge_2': t >= 2.0,
             'ex_2025_same_sign': bool(len(ex2025) and np.sign(ex2025.mean()) == np.sign(paired.mean())),
             'coverage_median_ge_70pct': float(coverage.median()) >= 0.70,
             'calmar_not_worse': gated_metrics['calmar'] >= base_metrics['calmar']}
    gates['passed'] = all(gates.values())

    OUT.mkdir(parents=True, exist_ok=True)
    report = {'strategy': 'P6_MULTI_CRITERIA_POOL', 'stage': 'A', 'zero_cost': True,
              'prereg': str(PREREG.relative_to(ROOT)), 'prereg_sha256': sha256(PREREG),
              'manifest': MANIFEST, 'panel_close_sha256': sha256(
                  ROOT / f'data/smallcap_v4/{MANIFEST}/close.parquet'),
              'financials_sha256': sha256(FIN),
              'window': [START, END],
              'band_check': {'days': len(check_days), 'min_overlap': float(min(overlaps)),
                             'mean_overlap': float(np.mean(overlaps))},
              'gate_coverage': {'median': float(coverage.median()),
                                'p10': float(coverage.quantile(0.1))},
              'ungated_pool': base_metrics, 'gated_pool': gated_metrics,
              'paired_excess': {'mean_daily': float(paired.mean()),
                                'annualised': float(paired.mean() * 252),
                                'nw_t': t, 'p_value': p,
                                'ex_2025_mean': float(ex2025.mean()) if len(ex2025) else None,
                                'daily_member_turnover_gated': turnover,
                                'days': int(len(paired))},
              'gates': gates, 'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    pd.DataFrame({'base': ret_base, 'gated': ret_gated, 'coverage': coverage}).to_csv(
        OUT / 'daily.csv')
    print(json.dumps(report, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
