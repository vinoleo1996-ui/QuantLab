#!/usr/bin/env python3
"""P5_MATCHED: is GBDT_HYST's +2.27pp alpha, or just the hysteresis rule?

P4_LOWTURN compared GBDT_HYST (entry top-10%, exit top-50%) with POOL50 (calendar
rebalance), so the +2.27pp mixes the model with the retention rule. This control runs
the *same* selector (select_hysteresis, same fractions, same 50 names, same industry
cap, same costs, same 800k ledger) with random ordering instead of GBDT ordering.

If GBDT_HYST does not beat its matched random-lag twin with NW t >= 2, the model line
closes and the gain belongs to the retention rule.

Output: output/p5_matched_20261017/report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alpha as P
from quantlab import p4_ledger
from quantlab.alpha.portfolio import random_order, schedule, select_hysteresis

ALPHA = ROOT / 'output/p4_alpha_20261012'
HYST = ROOT / 'output/p4_lowturn_20261013/ledger'
OUT = ROOT / 'output/p5_matched_20261017'
SEEDS = (1, 2)


def random_plans():
    scores = pd.read_parquet(ALPHA / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(P.ANCHOR)]
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    out = {}
    for phase in P.PHASES:
        rebalance = days[phase // P.STEP::P.REBALANCE // P.STEP]
        industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
        for seed in SEEDS:
            order = lambda day, s=seed: random_order(list(by_day[day].code), s)
            out[f'RAND_HYST_p{phase}_s{seed}'] = schedule(rebalance, order, industries,
                                                          selector=select_hysteresis)
    return out


def nw_t(series, lag=3):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float('nan')
    mean = x.mean()
    gamma0 = np.var(x, ddof=0)
    var = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        cov = np.cov(x[k:], x[:-k])[0, 1] if n > k + 1 else 0.0
        var += 2 * w * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mean / se)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plans = random_plans()
    targets = {name: p4_ledger.to_targets(s) for name, s in plans.items()}
    codes = sorted({c for rows in targets.values() for r in rows for c in list(r['weights']) + r['sell']})
    dataset = p4_ledger.load_dataset(codes)
    report = {'prereg': 'docs/prereg/P5_MATCHED.md', 'window': [P.ANCHOR, P.END],
              'seeds': list(SEEDS), 'cells': {}, 'promotion_approved': False}
    for cost in (1, 2):
        rows = []
        for name, target in sorted(targets.items()):
            result = p4_ledger.replay(target, start=P.ANCHOR, end=P.END, cost_mult=cost,
                                      dataset=dataset)
            nav = pd.Series([s['nav'] for s in result['account']],
                            index=pd.DatetimeIndex([pd.Timestamp(str(s['date'])[:10])
                                                    for s in result['account']]))
            nav = nav / nav.iloc[0]
            months = nav.resample('ME').last().pct_change().dropna()
            cell = dict(path=name, cagr=float(result['summary']['cagr']),
                        mdd=float(result['summary']['mdd']), orders=len(result['orders']),
                        fills=len(result['fills']))
            report['cells'][f'{name}_c{cost}'] = cell
            rows.append((name, cell, months))
            print(json.dumps(cell, ensure_ascii=False), flush=True)
            del result
        # paired comparison against the GBDT_HYST books of the same phase
        diffs, detail = [], {}
        for name, cell, months in rows:
            phase = name.split('_p')[1].split('_')[0]
            hyst_path = HYST / f'nav_GBDT_HYST_p{phase}_cost{cost}.csv'
            if not hyst_path.exists():
                continue
            h = pd.read_csv(hyst_path, index_col=0, parse_dates=True)['nav']
            h = h / h.iloc[0]
            hm = h.resample('ME').last().pct_change().dropna()
            joined = pd.concat([hm.rename('hyst'), months.rename('rand')], axis=1).dropna()
            d = joined.hyst - joined.rand
            diffs.extend(d.tolist())
            detail[name] = dict(hyst_cagr=float((h.iloc[-1]) ** (365.25 / (h.index[-1] - h.index[0]).days) - 1),
                                rand_cagr=cell['cagr'], mean_monthly_diff=float(d.mean()),
                                n=len(d))
        report['cells'][f'compare_c{cost}'] = dict(
            mean_monthly_diff=float(np.mean(diffs)) if diffs else None,
            annualised_diff=float(np.mean(diffs) * 12) if diffs else None,
            nw_t=nw_t(diffs), n_months=len(diffs), detail=detail)
        print('compare c%d: ann diff %s, nw_t %s' % (
            cost, report['cells'][f'compare_c{cost}']['annualised_diff'],
            report['cells'][f'compare_c{cost}']['nw_t']), flush=True)
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print('wrote', OUT / 'report.json')


if __name__ == '__main__':
    main()
