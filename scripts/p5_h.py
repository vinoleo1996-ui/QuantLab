#!/usr/bin/env python3
"""P5_H: event-driven daily triggers (docs/prereg/P5_H_EVENT.md, registered 2026-10-17).

The only change vs the calendar book is the prereg's section 17: every session after
the close, sell names whose GBDT score percentile fell out of the pool's top 50% and
refill vacancies only from the top 10% (entry_frac 0.10 / exit_frac 0.50, the
P4_LOWTURN hysteresis already implemented in quantlab/alpha/portfolio.py). Industry
cap 12 names, 50 holdings, 97% gross. Rows are emitted only on trigger days, so
between triggers the book drifts by market value -- prereg rule 3 verbatim.

Documented deviations (recorded before the run):
  * Event vetoes (rule 1 second clause: ST/退市整理/业绩预减/减持) are not wired --
    no point-in-time event feed (same limit as the 2026-10-14 governance entry).
  * Scores refresh on the pinned 5-session sample grid of
    output/p4_alpha_20261012/scores.parquet; between draws the latest is reused.
  * Pool: P5_G is closed (no_improvement), so the prereg's fallback branch applies:
    the current ADV bottom-40% band, eligibility as the ledger books (history>=253,
    ADV>=5m, minus ST, minus Beijing), i.e. quantlab/smallcap/universe.py pool_on.
  * Tie-break inside equal GBDT scores is (code, seed), fixed per path; the ledger
    books have no scored daily basket to inherit.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import os
ROOT = Path(os.environ.get('QUANTLAB_ROOT', '/Users/agiuser/Desktop/QuantLab'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alpha as P
from quantlab import p4_ledger
from quantlab import research_trials as T
from quantlab.alpha.portfolio import select_hysteresis
from quantlab.data.suspensions import SuspensionEvidence
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

OUT = ROOT / 'output/p5_h_20261018'
PREREG = 'docs/prereg/P5_H_EVENT.md'
SCORES = ROOT / 'output/p4_alpha_20261012/scores.parquet'
START, END = '2018-01-02', '2026-09-22'
PHASES, SEEDS = (0, 5, 10, 15), (1, 2)
GROSS = 0.97
GATES = {'cost1_cagr_ge': 0.1386, 'cost2_cagr_ge': 0.1259, 'turnover_le': 2.95, 'mdd_ge': -0.366}


def build_context():
    panel = Panel(P.PANEL)
    f = U.rolling_features(panel, with_st=True)
    sessions = pd.DatetimeIndex(panel.sessions)
    adv, hist, st = f['adv'], f['history'], f['st']
    bj = np.array([str(c).startswith('bj.') for c in adv.columns])
    keep = (hist >= U.MIN_HISTORY) & adv.notna() & (adv >= U.MIN_ADV) & ~st.eq(1)
    mask = keep.to_numpy().copy()
    mask[:, bj] = False
    # bottom 40% band per session, membership only (score order comes later)
    bands = {}
    for i, day in enumerate(sessions):
        row = adv.iloc[i][mask[i]]
        if row.empty or len(row) < 3:
            bands[i] = frozenset()
            continue
        desc = row.sort_values(ascending=False, kind='mergesort')
        cut = int(np.ceil(0.6 * len(desc)))
        bands[i] = frozenset(desc.index[cut:])
    scores = pd.read_parquet(SCORES, columns=['date', 'code', 'industry', 'GBDT'])
    score_dates = np.sort(np.unique(scores['date'].to_numpy()))
    by_date = {d: g.set_index('code')['GBDT'] for d, g in scores.groupby(scores['date'])}
    ind_map = scores.sort_values('date').groupby('code').industry.last().to_dict()
    return panel, sessions, bands, score_dates, by_date, ind_map


def daily_baskets(ctx, start, end, phase, seed):
    """One path: trigger events (day, basket). Scores lag at most 5 sessions (latest draw is
    reused), so the same draw is re-checked daily -- that is precisely the prereg's change."""
    panel, sessions, bands, score_dates, by_date, ind_map = ctx
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    current, out, replaced = [], [], []
    for day in window:
        k = int(np.searchsorted(score_dates, np.datetime64(day), side='right')) - 1
        if k < 0:
            continue
        band = bands[int(sessions.get_loc(day))]
        sc = by_date[score_dates[k]]
        if not band:
            scored = []
        else:
            scored = sorted((c for c in band if c in sc.index),
                            key=lambda c: (-float(sc[c]), str(c), seed))
        basket = select_hysteresis(scored, ind_map, current, holdings=U.HOLDINGS,
                                   entry_frac=0.10, exit_frac=0.50, cap=12) if scored else []
        if basket != current:
            if current:
                replaced.append(len(set(basket) - set(current)) / max(len(basket), 1))
            current = basket
            out.append((day, list(basket)))
    years = (window[-1] - window[0]).days / 365.25
    turnover = float(np.mean(replaced)) * len(replaced) / years if replaced else 0.0
    return out, turnover


def to_rows(baskets):
    schedule = {day: {c: GROSS / len(b) for c in b} if b else {} for day, b in baskets}
    return p4_ledger.to_targets(schedule)


def metrics(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
    mdd = float((nav / nav.cummax() - 1).min())
    return cagr, mdd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()
    src_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7]
    T.register('P5_h', 'EVENT_H', prereg=PREREG, source=f'scripts/p5_h.py:{src_sha}',
               data=f"panel {P.PANEL} + scores.parquet", segment='2018-01..2026-09-22',
               outcome='pending', note=f'prereg sha256 {sha[:12]}; daily-trigger 4 phases x 2 seeds x 2 costs')
    ctx = build_context()
    rows_out = []
    for phase in PHASES:
        for seed in SEEDS:
            baskets, turnover = daily_baskets(ctx, START, END, phase, seed)
            targets = to_rows(baskets)
            Path(OUT / f'events_p{phase}_s{seed}.json').write_text(json.dumps(
                [[str(d.date()), b] for d, b in baskets], ensure_ascii=False))
            print(f'p{phase} s{seed}: {len(baskets)} trigger events, est one-way turnover {turnover:.2f}x/yr',
                  flush=True)
            for cost in (1, 2):
                result = p4_ledger.replay(targets, start=START, end=END, cost_mult=cost, gross=GROSS,
                                          suspension_evidence=SuspensionEvidence())
                nav = result['nav']
                cagr, mdd = metrics(nav)
                name = f'EVENT_H_p{phase}_s{seed}_cost{cost}'
                nav.rename('nav').to_csv(OUT / f'nav_{name}.csv')
                rows_out.append(dict(path=name, cost=cost, cagr=cagr, mdd=mdd, turnover=turnover,
                                     events=len(baskets), fills=len(result['fills']),
                                     checks=result['summary']['account_checks_passed']))
                print(json.dumps(rows_out[-1], ensure_ascii=False), flush=True)
    df = pd.DataFrame(rows_out)
    med = df.groupby('cost')[['cagr', 'mdd']].median()
    summary = dict(median=med.round(4).to_dict(), turnover_one_way_median=float(df['turnover'].median()),
                   n=len(df),
                   gates=dict(cost1_ge_13_86=bool(med.loc[1, 'cagr'] >= GATES['cost1_cagr_ge']),
                              cost2_ge_12_59=bool(med.loc[2, 'cagr'] >= GATES['cost2_cagr_ge']),
                              turnover_le_2_95=bool(df['turnover'].median() <= GATES['turnover_le']),
                              mdd_ge_minus_36_6=bool(med.loc[1, 'mdd'] >= GATES['mdd_ge'])))
    (OUT / 'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
