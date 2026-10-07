#!/usr/bin/env python3
"""E5: crowding de-risk on the ADV equity leg, plus a circular-shift placebo.

Stages:
  prep     crowd.pkl: point-in-time ADV pool membership every 4 sessions from 2015, score, states
  placebo  200 random circular shifts of the real state series (same de-risked day count and
           episode lengths, random start); each draw runs seeds 1-4 x 5 phases
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import smallcap_layers as L
from quantlab.smallcap import layers, universe
from quantlab.smallcap.panel import Panel
from quantlab.smallcap.sim import Costs, metrics, simulate

HISTORY_START = '2015-01-05'
PLACEBO_DRAWS = 200
PLACEBO_SEEDS = (1, 2, 3, 4)


def prep():
    panel = Panel(L.VERSION)
    sessions = panel.sessions
    sessions = sessions[(sessions >= pd.Timestamp(HISTORY_START)) & (sessions <= pd.Timestamp(L.END))]
    features = universe.rolling_features(panel)
    members = {day: universe.pool_on(day, features, 'adv') for day in sessions[::4]}
    score, parts = layers.crowding_score(panel, sessions, members)
    states = layers.crowding_states(score)
    window = states.loc[L.START:L.END]
    payload = {'crowd_score': score, 'crowd_parts': parts, 'crowd_states': window}
    with open(L.OUT / 'crowd.pkl', 'wb') as handle:
        pickle.dump(payload, handle)
    derisk = window[window < 1.0]
    episodes = (window.diff().fillna(0) < 0).sum()
    summary = {'sessions': len(window), 'derisked_sessions': int(len(derisk)), 'episodes': int(episodes),
               'first_valid_score': str(score.dropna().index.min().date()),
               'derisked_by_year': derisk.groupby(derisk.index.year).size().to_dict(),
               'switches': [str(d.date()) for d in window[window.diff().fillna(0) != 0].index]}
    (L.OUT / 'E5_crowd_states.json').write_text(json.dumps(summary, indent=1, default=int))
    print(json.dumps(summary, indent=1, default=int))


def _placebo_one(args):
    draw, offset, phase, seed = args
    real = L._STATE['crowd_states']
    shifted = pd.Series(np.roll(real.to_numpy(), offset), index=real.index)
    snapshot = L.snapshots_for(L._STATE['snaps'], phase, 'adv')
    schedule = universe.build_schedule(snapshot, seed)
    schedule = L.apply_crowding(schedule, shifted, L._STATE['window'])
    stats = metrics(simulate(L._STATE['market'], schedule, costs=Costs())['nav'])
    return {'draw': draw, 'offset': offset, 'phase': phase, 'seed': seed,
            'cagr': stats['cagr'], 'max_drawdown': stats['max_drawdown']}


def placebo(processes=9):
    L.load_state()
    real = L._STATE['crowd_states']
    rng = np.random.default_rng(20261006)
    offsets = rng.integers(60, len(real) - 60, size=PLACEBO_DRAWS)
    tasks = [(d, int(o), p, s) for d, o in enumerate(offsets) for p in L.PHASES for s in PLACEBO_SEEDS]
    rows, t0 = [], time.time()
    with mp.get_context('fork').Pool(processes) as pool:
        for index, row in enumerate(pool.imap_unordered(_placebo_one, tasks, chunksize=4)):
            rows.append(row)
            if index % 200 == 0:
                print(json.dumps({'done': index, 'total': len(tasks), 'seconds': round(time.time() - t0)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(L.OUT / 'E5_placebo_runs.csv', index=False)
    base = pd.read_csv(L.OUT / 'equity/A_adv_cost1/runs.csv')
    crowd = pd.read_csv(L.OUT / 'equity/A_adv_crowd_cost1/runs.csv')
    subset = lambda f: f[f.seed.isin(PLACEBO_SEEDS)]
    base_dd = subset(base).max_drawdown.median()
    base_cagr = subset(base).cagr.median()
    per_draw = frame.groupby('draw').agg(cagr=('cagr', 'median'), max_drawdown=('max_drawdown', 'median'))
    per_draw['dd_improvement'] = per_draw.max_drawdown - base_dd
    real_dd_improvement = subset(crowd).max_drawdown.median() - base_dd
    result = {'draws': PLACEBO_DRAWS, 'seeds_per_draw': list(PLACEBO_SEEDS), 'phases': list(L.PHASES),
              'base_subset': {'cagr': base_cagr, 'max_drawdown': base_dd},
              'real_subset': {'cagr': subset(crowd).cagr.median(), 'max_drawdown': subset(crowd).max_drawdown.median(),
                              'dd_improvement': real_dd_improvement},
              'placebo_dd_improvement': {q: float(per_draw.dd_improvement.quantile(q)) for q in (0.1, 0.5, 0.9)},
              'placebo_cagr': {q: float(per_draw.cagr.quantile(q)) for q in (0.1, 0.5, 0.9)},
              'real_rank_pct': float((per_draw.dd_improvement < real_dd_improvement).mean()),
              'beats_placebo_p90': bool(real_dd_improvement > per_draw.dd_improvement.quantile(0.9)),
              'promotion_approved': False}
    (L.OUT / 'E5_placebo.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prep', 'placebo'))
    args = parser.parse_args()
    prep() if args.stage == 'prep' else placebo()
