#!/usr/bin/env python3
"""P8 zero-cost diagnostics: unlock veto, industry-conditioned reversal, weight/concentration.

Diagnostics only: no ledger, no confirmatory trial, no candidate. They answer whether any of
the three is worth a preregistered hypothesis before more compute is spent.
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

from quantlab.research_statistics import newey_west
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

MANIFEST = '20261013T023232Z'
SCORES = ROOT / 'output/p4_alpha_20261012/scores.parquet'
UNLOCK = ROOT / 'data/cloud/share_float_pit/bundle_20261012/part-all-all.parquet'
OUT = ROOT / 'output/p8_zero_cost_diag_20261018'
PREREG = ROOT / 'docs/prereg/P8_DIAGNOSTICS.md'
START, END = '2018-01-02', '2026-09-22'
STEP = 20
UNLOCK_RATIO = 5.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nw(series):
    values = pd.Series(series).dropna().to_numpy()
    if len(values) < 30:
        return {'n': int(len(values)), 'mean': float('nan'), 't': float('nan'), 'p': float('nan')}
    t = float(newey_west(values, lags=20)[1][0])
    return {'n': int(len(values)), 'mean': float(np.mean(values)), 't': t,
            'p': float(2 * (1 - norm.cdf(abs(t))))}


def main():
    panel = Panel(MANIFEST)
    sessions = panel.sessions[(panel.sessions >= pd.Timestamp(START)) &
                              (panel.sessions <= pd.Timestamp(END))]
    fields = {name: panel.field(name).reindex(sessions) for name in
              ('close', 'adj_close', 'amount', 'is_st')}
    features = U.rolling_features(panel, with_st=True)
    adv = features['adv'].reindex(sessions)
    eligible = (features['history'].reindex(sessions).ge(253) & adv.notna() & adv.ge(5e6)
                & ~features['st'].reindex(sessions).eq(1))
    eligible = eligible.reindex(columns=[c for c in adv.columns if not str(c).startswith('bj.')])
    adv = adv.reindex(columns=eligible.columns)
    rank = adv.where(eligible).rank(axis=1, ascending=False, method='first')
    pool = rank.gt(np.ceil(0.6 * eligible.sum(axis=1)), axis=0) & eligible
    ret20 = fields['adj_close'].reindex(columns=pool.columns).pct_change(20, fill_method=None)
    forward = (fields['adj_close'].reindex(columns=pool.columns).shift(-20)
               / fields['adj_close'].reindex(columns=pool.columns) - 1)
    days = sessions[::STEP]

    # ---- unlock veto -----------------------------------------------------------------
    unlock = pd.read_parquet(UNLOCK, columns=['ts_code', 'float_date', 'float_ratio', 'available_at'])
    parts = unlock.ts_code.str.split('.')
    unlock['code'] = parts.str[1].str.lower() + '.' + parts.str[0]
    unlock = unlock[(unlock.float_ratio >= UNLOCK_RATIO) & unlock.float_date.notna()]
    unlock['float_date'] = pd.to_datetime(unlock.float_date.astype(str), errors='coerce')
    unlock['available'] = pd.to_datetime(unlock.available_at, errors='coerce').dt.tz_localize(None)
    exposure, coverage = [], []
    for day in days:
        members = pool.columns[pool.loc[day].to_numpy()]
        events = unlock[(unlock.available <= day) &
                        (unlock.float_date > day) &
                        (unlock.float_date <= day + pd.Timedelta(days=90))]
        exposed = set(events.code) & set(members)
        coverage.append(len(exposed) / max(1, len(members)))
        if not exposed or len(exposed) == len(members):
            continue
        values = forward.loc[day]
        exposed_mean = values.reindex(sorted(exposed)).mean()
        rest = [c for c in members if c not in exposed]
        rest_mean = values.reindex(rest).mean()
        exposure.append(exposed_mean - rest_mean)
    unlock_result = {'coverage_median': float(np.median(coverage)) if coverage else 0.0,
                     'coverage_p90': float(np.quantile(coverage, 0.9)) if coverage else 0.0,
                     'n_days': len(exposure), **nw(exposure)}
    unlock_result['verdict'] = ('no_action' if unlock_result['coverage_median'] < 0.08
                                else ('worth_veto_layer'
                                      if unlock_result['t'] is not None
                                      and unlock_result['t'] <= -2.0 else 'not_significant'))

    # ---- industry-conditioned reversal -----------------------------------------------
    scores = pd.read_parquet(SCORES, columns=['date', 'code', 'industry', 'GBDT'])
    industry = scores.sort_values('date').groupby('code').industry.last().to_dict()
    industry_diff = []
    for day in days:
        members = pool.columns[pool.loc[day].to_numpy()]
        past = ret20.loc[day].reindex(members).dropna()
        if len(past) < 50:
            continue
        frame = pd.DataFrame({'past': past, 'forward': forward.loc[day].reindex(past.index)}).dropna()
        frame['industry'] = [industry.get(code, 'unknown') for code in frame.index]
        group = frame.groupby('industry').agg(past=('past', 'mean'), forward=('forward', 'mean'),
                                              n=('forward', 'size'))
        group = group[group.n >= 5]
        if len(group) < 6:
            continue
        cut_high = group.past.quantile(2 / 3)
        cut_low = group.past.quantile(1 / 3)
        industry_diff.append(group[group.past >= cut_high].forward.mean()
                             - group[group.past <= cut_low].forward.mean())
    industry_result = {**nw(industry_diff),
                       'verdict': 'worth_hypothesis' if len(industry_diff) >= 30
                       and nw(industry_diff)['t'] is not None
                       and abs(nw(industry_diff)['t']) >= 2.0 else 'not_significant'}

    # ---- weight / concentration ------------------------------------------------------
    score_dates = np.sort(np.unique(scores.date.to_numpy()))
    by_date = {d: g.set_index('code').GBDT for d, g in scores.groupby('date')}
    weight_diff, concentration = [], {name: [] for name in ('top30', 'top40', 'top50')}
    for day in days:
        k = int(np.searchsorted(score_dates, np.datetime64(day), side='right')) - 1
        if k < 0:
            continue
        sc = by_date[score_dates[k]]
        members = pool.columns[pool.loc[day].to_numpy()]
        ranked = sc.reindex(members).dropna().sort_values(ascending=False)
        if len(ranked) < 50:
            continue
        top50 = ranked.index[:50]
        values = forward.loc[day]
        pool_mean = values.reindex(members).mean()
        equal = values.reindex(top50).mean()
        weights = ranked.iloc[:50].clip(lower=0)
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1 / 50, index=top50)
        weighted = float((values.reindex(top50) * weights).sum())
        weight_diff.append(weighted - equal)
        for name, size in (('top30', 30), ('top40', 40), ('top50', 50)):
            concentration[name].append(values.reindex(ranked.index[:size]).mean() - pool_mean)
    weight_result = {'score_weighted_minus_equal': nw(weight_diff),
                     'concentration_vs_pool': {name: nw(values)
                                               for name, values in concentration.items()},
                     'note': 'diagnostic only; the GBDT ranking itself was judged noise in P5_MATCHED'}

    OUT.mkdir(parents=True, exist_ok=True)
    report = {'strategy': 'P8_DIAGNOSTICS', 'stage': 'diagnostic', 'zero_cost': True,
              'prereg': str(PREREG.relative_to(ROOT)), 'prereg_sha256': sha256(PREREG),
              'manifest': MANIFEST, 'window': [START, END], 'sample_days': int(len(days)),
              'unlock_veto': unlock_result, 'industry_reversal': industry_result,
              'weight_concentration': weight_result, 'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
