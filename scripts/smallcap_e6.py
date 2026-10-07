#!/usr/bin/env python3
"""E6 prep: pead.pkl (live positive-event flags on window sessions) and an event study by year.

Event study (diagnostic only): for confirmed positive events whose stock is in the ADV pool on
the reaction day, the 60-session return from the next open minus the pool's equal-weight return
over the same span.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import smallcap_layers as L
from quantlab.smallcap import layers
from quantlab.smallcap.panel import Panel


def pool_members(snaps, sessions):
    days = sorted(snaps)
    out, current, k = {}, [], 0
    for day in sessions:
        while k < len(days) and days[k] <= day:
            current = list(snaps[days[k]]['adv']['codes'])
            k += 1
        out[day] = current
    return out


def main():
    panel = Panel(L.VERSION)
    with open(L.OUT / 'snapshots.pkl', 'rb') as handle:
        payload = pickle.load(handle)
    window = payload['window']
    members = pool_members(payload['snaps'], window)
    close = panel.field('adj_close').reindex(window)
    opens = (panel.field('open') * panel.field('adj_close') / panel.field('close')).reindex(window)
    daily = close.pct_change(fill_method=None)
    pool_median = pd.Series([daily.loc[d].reindex(m).median() if m else 0.0 for d, m in members.items()],
                            index=window).fillna(0.0)
    events = layers.earnings_events(panel.table('financials'))
    events = events[(events.date >= window[0] - pd.Timedelta(days=120)) & (events.date <= window[-1])]
    flags = layers.pead_flags(panel, events, window, pool_median)
    with open(L.OUT / 'pead.pkl', 'wb') as handle:
        pickle.dump({'pead': flags}, handle)

    pool_ret = pd.Series([daily.loc[d].reindex(m).mean() if m else 0.0 for d, m in members.items()],
                         index=window).fillna(0.0)
    pool_index = (1 + pool_ret).cumprod()
    starts = flags & ~flags.shift(1, fill_value=False)
    rows = []
    values_open, values_close = opens.to_numpy(), close.to_numpy()
    col = {c: j for j, c in enumerate(flags.columns)}
    for t, j in zip(*np.nonzero(starts.to_numpy())):
        day, code = window[t], flags.columns[j]
        if code not in set(members[day]) or t + 61 >= len(window):
            continue
        entry, exit_ = values_open[t + 1, j], values_close[t + 60, j]
        if not (np.isfinite(entry) and np.isfinite(exit_) and entry > 0):
            continue
        bench = pool_index.iloc[t + 60] / pool_index.iloc[t] - 1
        rows.append({'year': day.year, 'excess60': exit_ / entry - 1 - bench})
    study = pd.DataFrame(rows)
    by_year = study.groupby('year').excess60.agg(['count', 'mean', 'median']).round(4)
    sources = events.source.value_counts().to_dict()
    summary = {'events': int(len(events)), 'events_by_source': sources,
               'positive_share': float(events.positive.mean()),
               'flagged_name_days_in_pool_median': float(np.median(
                   [flags.loc[d].reindex(m).sum() for d, m in list(members.items())[::20] if m])),
               'event_study_in_pool': {int(y): r for y, r in by_year.to_dict(orient='index').items()},
               'event_study_all_mean': float(study.excess60.mean()) if len(study) else None,
               'event_study_all_t': float(study.excess60.mean() / study.excess60.std() * np.sqrt(len(study)))
               if len(study) > 1 else None}
    (L.OUT / 'E6_events.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


SHUFFLE_DRAWS = 20
SHUFFLE_SEEDS = (1, 2, 3, 4)
_SHUFFLED = {}


def _shuffle_one(args):
    draw, phase, seed = args
    from quantlab.smallcap import universe
    from quantlab.smallcap.sim import Costs, metrics, simulate
    if draw not in _SHUFFLED:
        real = L._STATE['pead']
        order = np.random.default_rng(1000 + draw).permutation(real.shape[1])
        _SHUFFLED.clear()
        _SHUFFLED[draw] = pd.DataFrame(real.to_numpy()[:, order], index=real.index, columns=real.columns)
    snapshot = L.snapshots_for(L._STATE['snaps'], phase, 'adv', pead=_SHUFFLED[draw])
    schedule = universe.build_schedule(snapshot, seed, key_names=('tier', 'pead'), retain_key_names=('tier',))
    result = simulate(L._STATE['market'], schedule, costs=Costs())
    stats = metrics(result['nav'])
    return {'draw': draw, 'phase': phase, 'seed': seed, 'cagr': stats['cagr'],
            'max_drawdown': stats['max_drawdown'], 'turnover': result['turnover']}


def shuffle_placebo(processes=9):
    import multiprocessing as mp
    L.load_state()
    tasks = [(d, p, s) for d in range(SHUFFLE_DRAWS) for p in L.PHASES for s in SHUFFLE_SEEDS]
    with mp.get_context('fork').Pool(processes) as pool:
        rows = list(pool.imap(_shuffle_one, tasks, chunksize=len(L.PHASES) * len(SHUFFLE_SEEDS)))
    frame = pd.DataFrame(rows)
    frame.to_csv(L.OUT / 'E6_shuffle_runs.csv', index=False)
    subset = lambda f: f[f.seed.isin(SHUFFLE_SEEDS)]
    base = subset(pd.read_csv(L.OUT / 'equity/A_adv_cost1/runs.csv'))
    real = subset(pd.read_csv(L.OUT / 'equity/D_pead_adv_cost1/runs.csv'))
    per_draw = frame.groupby('draw')[['cagr', 'max_drawdown', 'turnover']].median()
    result = {'note': 'post-hoc attribution diagnostic; does not change the pre-registered E6 decision',
              'draws': SHUFFLE_DRAWS, 'seeds_per_draw': list(SHUFFLE_SEEDS),
              'base_subset': base[['cagr', 'max_drawdown', 'turnover']].median().to_dict(),
              'real_subset': real[['cagr', 'max_drawdown', 'turnover']].median().to_dict(),
              'shuffle_cagr': {q: float(per_draw.cagr.quantile(q)) for q in (0.1, 0.5, 0.9)},
              'shuffle_max_drawdown': {q: float(per_draw.max_drawdown.quantile(q)) for q in (0.1, 0.5, 0.9)},
              'shuffle_turnover_median': float(per_draw.turnover.median()),
              'real_cagr_rank_pct': float((per_draw.cagr < real.cagr.median()).mean())}
    (L.OUT / 'E6_shuffle_placebo.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    shuffle_placebo() if sys.argv[1:] == ['shuffle'] else main()
