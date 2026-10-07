#!/usr/bin/env python3
"""Pre-registered small-cap layer experiments (see PREREGISTRATION.md in OUT).

Stages:
  prep    decision snapshots for both pools on all five phases, screens, industries
  equity  seeds x phases for an equity configuration (A_adv, B_cap, C_screen, D_pead, *_crowd)
  report  medians, segments and the pre-registered layer decisions
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

import numpy as np
import pandas as pd

from quantlab.smallcap import layers, universe
from quantlab.smallcap.panel import Panel
from quantlab.smallcap.sim import Costs, Market, metrics, segment_cagr, simulate

OUT = ROOT / 'output/smallcap_rebuild_20261006'
VERSION = '20261006T152412Z'
START, END = '2018-01-02', '2026-09-22'
PHASES = (0, 4, 8, 12, 16)
SEGMENTS = {'2018_2021': ('2018-01-02', '2021-12-31'), '2022_2025': ('2022-01-01', '2025-12-31'),
            '2025_now': ('2025-01-02', '2026-09-22')}

_STATE = {}


def prep():
    panel = Panel(VERSION)
    sessions = panel.sessions
    window = sessions[(sessions >= pd.Timestamp(START)) & (sessions <= pd.Timestamp(END))]
    t0 = time.time()
    features = universe.rolling_features(panel)
    print(json.dumps({'stage': 'features', 'seconds': round(time.time() - t0)}), flush=True)
    financials = panel.table('financials')
    net_profit = layers.asof_metric(financials, 'net_profit', window, panel.codes)
    pb = panel.field('pb_mrq').reindex(window)
    flags, parts = layers.screen_flags(panel, financials, window)
    industries = universe.Industries(panel.table('industry'))
    days = sorted({d for p in PHASES for d in universe.decision_grid(window, START, END, phase=p)})
    snaps = {}
    for index, day in enumerate(days):
        ind = industries.on(day).to_dict()
        entry = {}
        for kind in ('adv', 'cap'):
            codes = universe.pool_on(day, features, kind)
            tier = universe.financial_tier(codes, net_profit.loc[day], pb.loc[day]).to_numpy()
            flagged = flags.loc[day].reindex(codes).fillna(False).to_numpy()
            entry[kind] = {'codes': codes, 'tier': tier, 'flag': flagged}
        entry['industries'] = {c: ind.get(c, 'unclassified') for k in ('adv', 'cap') for c in entry[k]['codes']}
        snaps[day] = entry
        if index % 50 == 0:
            print(json.dumps({'stage': 'snapshots', 'index': index, 'total': len(days)}), flush=True)
    screen_counts = {name: {str(day.date()): int(frame.loc[day].reindex(snaps[day]['cap']['codes']).fillna(False).sum())
                            for day in days[::20]} for name, frame in parts.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'snapshots.pkl', 'wb') as handle:
        pickle.dump({'snaps': snaps, 'window': window}, handle)
    (OUT / 'screen_counts.json').write_text(json.dumps(screen_counts, indent=1))
    print(json.dumps({'stage': 'prepared', 'days': len(days)}), flush=True)


def snapshots_for(snaps, phase, pool, screen=False, pead=None):
    grid = universe.decision_grid(_STATE['window'], START, END, phase=phase)
    out = []
    for day in grid:
        entry = snaps[day][pool]
        codes, tier, flag = list(entry['codes']), entry['tier'], entry['flag']
        if screen:
            keep = ~flag
            codes = [c for c, k in zip(codes, keep) if k]
            tier = tier[keep]
        keys = {'tier': tier}
        if pead is not None:
            row = pead.loc[day]
            keys['pead'] = row.reindex(codes).fillna(False).to_numpy(dtype=float)
        out.append((day, codes, keys, snaps[day]['industries']))
    return out


def apply_crowding(schedule, states, window):
    """Scale targets by the crowding state; add a rebalance on each state switch."""
    scaled = {}
    current = {}
    switches = states[states.diff().fillna(0) != 0].index
    days = sorted(set(schedule) | set(switches))
    for day in days:
        if day in schedule:
            current = schedule[day]
        scale = states.loc[:day].iloc[-1] if len(states.loc[:day]) else 1.0
        scaled[day] = {c: w * scale for c, w in current.items()}
    return scaled


def run_one(args):
    config, phase, seed, cost_mult = args
    snaps = _STATE['snaps']
    spec = CONFIGS[config]
    snapshot = snapshots_for(snaps, phase, spec['pool'], spec.get('screen', False),
                             _STATE.get('pead') if spec.get('pead') else None)
    keys = ('tier', 'pead') if spec.get('pead') else ('tier',)
    schedule = universe.build_schedule(snapshot, seed, key_names=keys,
                                       retain_key_names=('tier',) if spec.get('pead') else None)
    if spec.get('crowd'):
        schedule = apply_crowding(schedule, _STATE['crowd_states'], _STATE['window'])
    result = simulate(_STATE['market'], schedule, costs=Costs(multiplier=cost_mult))
    nav = result['nav']
    stats = metrics(nav)
    stats.update({name: segment_cagr(nav, a, b) for name, (a, b) in SEGMENTS.items()})
    stats.update(config=config, phase=phase, seed=seed, cost_mult=cost_mult, fees=result['fees'],
                 slippage=result['slippage'], turnover=result['turnover'],
                 delist_events=len(result['events']),
                 median_gross=float(result['gross'][result['gross'] > 0].median()))
    stats.pop('yearly')
    return stats, (nav.to_numpy(dtype=np.float32) if seed <= 20 else None)


CONFIGS = {
    'A_adv': {'pool': 'adv'},
    'B_cap': {'pool': 'cap'},
    'C_screen_adv': {'pool': 'adv', 'screen': True},
    'C_screen_cap': {'pool': 'cap', 'screen': True},
    'A_adv_crowd': {'pool': 'adv', 'crowd': True},
    'D_pead_adv': {'pool': 'adv', 'pead': True},
}


def load_state(need_pead=False, crowd_pool=None):
    panel = Panel(VERSION)
    with open(OUT / 'snapshots.pkl', 'rb') as handle:
        payload = pickle.load(handle)
    _STATE.update(payload)
    delist = panel.table('delistings')
    _STATE['market'] = Market.from_panel(panel, START, END, delist=dict(zip(delist.code, delist.last_bar)))
    for name in ('pead.pkl', 'crowd.pkl'):
        path = OUT / name
        if path.exists():
            with open(path, 'rb') as handle:
                _STATE.update(pickle.load(handle))
    return panel


def equity(config, seeds, cost_mult=1.0, processes=9):
    load_state()
    tasks = [(config, phase, seed, cost_mult) for phase in PHASES for seed in seeds]
    t0 = time.time()
    context = mp.get_context('fork')
    rows, navs = [], {}
    with context.Pool(processes) as pool:
        for index, (stats, nav) in enumerate(pool.imap_unordered(run_one, tasks, chunksize=4)):
            rows.append(stats)
            if nav is not None:
                navs[f"{stats['phase']}_{stats['seed']}"] = nav
            if index % 100 == 0:
                print(json.dumps({'stage': config, 'done': index, 'total': len(tasks),
                                  'seconds': round(time.time() - t0)}), flush=True)
    folder = OUT / 'equity' / f'{config}_cost{cost_mult:g}'
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / 'runs.csv', index=False)
    pd.DataFrame(navs, index=_STATE['market'].dates).to_parquet(folder / 'navs.parquet')
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'stage': 'done', 'config': config, **summary['median']}), flush=True)


def summarize(frame):
    columns = ['cagr', 'max_drawdown', *SEGMENTS, 'median_gross', 'fees', 'slippage', 'turnover', 'delist_events']
    return {'runs': int(len(frame)), 'median': {c: float(frame[c].median()) for c in columns},
            'p10': {c: float(frame[c].quantile(0.1)) for c in ('cagr', 'max_drawdown')},
            'p90': {c: float(frame[c].quantile(0.9)) for c in ('cagr', 'max_drawdown')},
            'share_cagr15_dd25': float(((frame.cagr >= 0.15) & (frame.max_drawdown >= -0.25)).mean()),
            'phase_median_cagr': {int(k): float(v) for k, v in frame.groupby('phase').cagr.median().items()},
            'trough_years': frame.trough.str[:4].value_counts().to_dict(),
            'promotion_approved': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prep', 'equity'))
    parser.add_argument('--config')
    parser.add_argument('--seeds', type=int, default=200)
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    if args.stage == 'prep':
        prep()
    else:
        equity(args.config, range(1, args.seeds + 1), args.cost)
