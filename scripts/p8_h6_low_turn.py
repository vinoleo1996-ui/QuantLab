#!/usr/bin/env python3
"""P8_H6_LOW_TURN: daily triggers with the exit band fixed by a prior turnover table.

P5_H failed its own gates (turnover 5.40x, MDD -41.4%) and its four phases x two seeds
collapsed to one path. This is a separate hypothesis, not a rescue: entry stays at the top
10%, the exit band is widened from 0.50 to 0.75 by the preregistered prior mapping
(1 - exit) proportional to turnover, and no value is searched. A same-turnover random-trigger
control and a block-bootstrap MDD distribution stand in for the meaningless phase paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('QUANTLAB_ROOT', '/Users/agiuser/Desktop/QuantLab'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import p4_alpha as P
from quantlab import p4_ledger
from quantlab import research_trials as T
from quantlab.alpha.portfolio import select_hysteresis
from quantlab.data.suspensions import SuspensionEvidence
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

OUT = ROOT / 'output/p8_h6_low_turn_20261018'
PREREG = 'docs/prereg/P8_H6_LOW_TURN.md'
SCORES = ROOT / 'output/p4_alpha_20261012/scores.parquet'
BASELINE = ROOT / 'output/p5_base_lot_20261018_r1/report.json'
START, END = '2018-01-02', '2026-09-22'
ENTRY_FRAC, EXIT_FRAC = 0.10, 0.75
GROSS = 0.97
CONTROL_SEED, CONTROL_REPS = 20261018, 5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_context():
    panel = Panel(P.PANEL)
    features = U.rolling_features(panel, with_st=True)
    sessions = pd.DatetimeIndex(panel.sessions)
    adv, hist, st = features['adv'], features['history'], features['st']
    bj = np.array([str(c).startswith('bj.') for c in adv.columns])
    keep = (hist >= U.MIN_HISTORY) & adv.notna() & (adv >= U.MIN_ADV) & ~st.eq(1)
    mask = keep.to_numpy().copy()
    mask[:, bj] = False
    bands = {}
    for i, day in enumerate(sessions):
        row = adv.iloc[i][mask[i]]
        if row.empty or len(row) < 3:
            bands[i] = frozenset()
            continue
        desc = row.sort_values(ascending=False, kind='mergesort')
        bands[i] = frozenset(desc.index[int(np.ceil(0.6 * len(desc))):])
    scores = pd.read_parquet(SCORES, columns=['date', 'code', 'industry', 'GBDT'])
    score_dates = np.sort(np.unique(scores['date'].to_numpy()))
    by_date = {d: g.set_index('code')['GBDT'] for d, g in scores.groupby(scores['date'])}
    ind_map = scores.sort_values('date').groupby('code').industry.last().to_dict()
    return panel, sessions, bands, score_dates, by_date, ind_map


def basket_at(ctx, day, current):
    panel, sessions, bands, score_dates, by_date, ind_map = ctx
    k = int(np.searchsorted(score_dates, np.datetime64(day), side='right')) - 1
    if k < 0:
        return list(current)
    band = bands[int(sessions.get_loc(day))]
    sc = by_date[score_dates[k]]
    if not band:
        return []
    scored = sorted((c for c in band if c in sc.index), key=lambda c: (-float(sc[c]), str(c)))
    return select_hysteresis(scored, ind_map, current, holdings=U.HOLDINGS,
                             entry_frac=ENTRY_FRAC, exit_frac=EXIT_FRAC, cap=12)


def daily_baskets(ctx, trigger_days=None):
    """``trigger_days=None`` checks every session (the hypothesis); otherwise only those dates."""
    _, sessions, *_ = ctx
    window = sessions[(sessions >= pd.Timestamp(START)) & (sessions <= pd.Timestamp(END))]
    triggers = set(window) if trigger_days is None else set(trigger_days)
    current, events, replaced = [], [], []
    for day in window:
        if day not in triggers:
            continue
        basket = basket_at(ctx, day, current)
        if basket != current:
            if current:
                replaced.append(len(set(basket) - set(current)) / max(len(basket), 1))
            current = basket
            events.append((day, list(basket)))
    years = (window[-1] - window[0]).days / 365.25
    turnover = float(np.mean(replaced)) * len(replaced) / years if replaced else 0.0
    return events, turnover


def to_rows(baskets):
    schedule = {day: {c: GROSS / len(b) for c in b} if b else {} for day, b in baskets}
    return p4_ledger.to_targets(schedule)


def metrics(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
    mdd = float((nav / nav.cummax() - 1).min())
    return cagr, mdd, (cagr / abs(mdd) if mdd < 0 else float('inf'))


def bootstrap_mdd(nav, block=20, samples=2000, seed=20261018):
    returns = nav.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    draws = []
    for _ in range(samples):
        starts = rng.integers(0, len(returns) - block + 1, size=len(returns) // block + 1)
        index = np.concatenate([start + offsets for start in starts])[:len(returns)]
        path = np.cumprod(1 + returns[index])
        peak = np.maximum.accumulate(path)
        draws.append(float((path / peak - 1).min()))
    return {'mdd_5pct': float(np.quantile(draws, 0.05)),
            'mdd_median': float(np.quantile(draws, 0.50)), 'samples': samples, 'block': block}


def baseline():
    cells = json.loads(BASELINE.read_text())['cells']['800000']
    return {cost: {'cagr': cells[cost]['blended_cagr'], 'mdd': cells[cost]['blended_mdd'],
                   'calmar': cells[cost]['blended_cagr'] / abs(cells[cost]['blended_mdd'])}
            for cost in ('1', '2')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    events, turnover = daily_baskets(ctx)
    (OUT / 'events.json').write_text(json.dumps(
        [[str(day.date()), basket] for day, basket in events], ensure_ascii=False))
    targets = to_rows(events)
    rows, navs = [], {}
    for cost in (1.0, 2.0):
        result = p4_ledger.replay(targets, start=START, end=END, cost_mult=cost, gross=GROSS,
                                  suspension_evidence=SuspensionEvidence())
        nav = result['nav']
        cagr, mdd, calmar = metrics(nav)
        nav.rename('nav').to_csv(OUT / f'nav_lowturn_cost{int(cost)}.csv')
        navs[cost] = nav
        rows.append({'cost': int(cost), 'cagr': cagr, 'mdd': mdd, 'calmar': calmar,
                     'turnover': turnover, 'events': len(events), 'fills': len(result['fills']),
                     'checks': result['summary']['account_checks_passed']})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    boot = {str(int(cost)): bootstrap_mdd(nav) for cost, nav in navs.items()}

    sessions = pd.DatetimeIndex(navs[1.0].index)
    window = sessions[(sessions >= pd.Timestamp(START)) & (sessions <= pd.Timestamp(END))]
    rng = np.random.default_rng(CONTROL_SEED)
    control = []
    for rep in range(CONTROL_REPS):
        days = pd.DatetimeIndex(sorted(rng.choice(window, size=len(events), replace=False)))
        control_events, control_turnover = daily_baskets(ctx, days)
        result = p4_ledger.replay(to_rows(control_events), start=START, end=END, cost_mult=1.0,
                                  gross=GROSS, suspension_evidence=SuspensionEvidence())
        cagr, mdd, calmar = metrics(result['nav'])
        control.append({'rep': rep, 'cagr': cagr, 'mdd': mdd, 'calmar': calmar,
                        'turnover': control_turnover, 'events': len(control_events)})
        print(json.dumps(control[-1], ensure_ascii=False), flush=True)

    base = baseline()
    real = {row['cost']: row for row in rows}
    gates = {
        'cost1_cagr_ge_baseline': real[1]['cagr'] >= base['1']['cagr'],
        'cost2_cagr_ge_baseline': real[2]['cagr'] >= base['2']['cagr'],
        'turnover_le_2_95': turnover <= 2.95,
        'calmar_ge_baseline': real[1]['calmar'] >= base['1']['calmar'],
        'bootstrap_mdd_5pct_ge_minus_36_6': boot['1']['mdd_5pct'] >= -0.366,
        'random_trigger_weaker': float(np.median([c['calmar'] for c in control])) < real[1]['calmar'],
    }
    gates['passed'] = all(gates.values())
    report = {'strategy': 'P8_H6_LOW_TURN', 'prereg': PREREG, 'prereg_sha256': sha256(ROOT / PREREG),
              'entry_frac': ENTRY_FRAC, 'exit_frac': EXIT_FRAC, 'gross': GROSS,
              'window': [START, END], 'baseline': base, 'paths': rows,
              'bootstrap': boot, 'random_trigger_control': control, 'gates': gates,
              'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != 'paths'}, ensure_ascii=False,
                     indent=1, default=float))


if __name__ == '__main__':
    main()
