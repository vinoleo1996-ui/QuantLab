#!/usr/bin/env python3
"""Theme leg, stock version (C series). Rules: output/smallcap_theme_20261008/PREREGISTRATION.md.

Stages:
  clusters   point-in-time correlation clusters every 20 sessions (top 3000 ADV)
  run        C1 (clusters) and C0 (CSRC industry) on phases 0-4, 150k sub-account
  random     200 turnover-matched random group draws on phase 0
  small30    small-cap leg with 30 holdings (D rule) seeds 1-20 x phases, 210k sub-account
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import smallcap_layers as L
from quantlab.smallcap import universe
from quantlab.smallcap.panel import Panel
from quantlab.smallcap.sim import Costs, Market, metrics, segment_cagr, simulate

OUT = ROOT / 'output/smallcap_theme_20261008'
TOP, LOOKBACK, N_CLUSTERS, MIN_MEMBERS = 3000, 120, 40, 5
HOLD, BUFFER_RANK, LEADERS = 4, 8, 5
THEME_CASH, SMALL_CASH = 150_000.0, 210_000.0
GROSS = 0.90
_S = {}
CLUSTER_FILES = {'ward': 'clusters_ward.pkl', 'average': 'clusters.pkl'}


def base_state():
    panel = Panel(L.VERSION)
    features = universe.rolling_features(panel)
    close = panel.field('adj_close')
    amount = panel.field('amount')
    return panel, features, close, amount


def clusters(method='ward'):
    panel, features, close, _ = base_state()
    sessions = panel.sessions
    window = sessions[(sessions >= pd.Timestamp(L.START)) & (sessions <= pd.Timestamp(L.END))]
    returns = close.pct_change(fill_method=None)
    out, t0 = {}, time.time()
    for day in window[::20]:
        adv = features['adv'].loc[day][features['history'].loc[day] >= 253].dropna()
        members = adv.sort_values(ascending=False).index[:TOP]
        end = sessions.get_loc(day)
        block = returns.iloc[end - LOOKBACK + 1:end + 1][members]
        block = block.loc[:, block.notna().sum() >= 100]
        resid = block.sub(block.mean(axis=1), axis=0).fillna(0.0)
        if method == 'ward':
            values = resid.to_numpy()
            z = (values - values.mean(axis=0)) / values.std(axis=0).clip(min=1e-12)
            tree = linkage(z.T, 'ward')
        else:
            corr = np.nan_to_num(np.corrcoef(resid.to_numpy().T), nan=0.0)
            dist = np.clip(1 - corr, 0, 2)
            np.fill_diagonal(dist, 0)
            tree = linkage(squareform(dist, checks=False), 'average')
        labels = fcluster(tree, N_CLUSTERS, 'maxclust')
        out[day] = pd.Series(labels, index=resid.columns)
    with open(OUT / CLUSTER_FILES[method], 'wb') as handle:
        pickle.dump(out, handle)
    sizes = [s.value_counts() for s in out.values()]
    print(json.dumps({'dates': len(out), 'seconds': round(time.time() - t0),
                      'median_clusters_ge5': float(np.median([(v >= MIN_MEMBERS).sum() for v in sizes])),
                      'median_largest_share': float(np.median([v.max() / v.sum() for v in sizes]))}))


def industry_groups(panel, features, window):
    industries = universe.Industries(panel.table('industry'))
    out = {}
    for day in window[::20]:
        adv = features['adv'].loc[day][features['history'].loc[day] >= 253].dropna()
        members = adv.sort_values(ascending=False).index[:TOP]
        ind = industries.on(day).reindex(members).fillna('unclassified')
        out[day] = ind.astype(str).str.split(' ').str[0]
    return out


def group_scores(day, labels, ctx):
    """Score each group (>= MIN_MEMBERS) on flow, diffusion and relative strength."""
    t = ctx['pos'][day]
    members = labels.index
    amount, close, ma60 = ctx['amount'], ctx['close'], ctx['ma60']
    total10 = ctx['total_amount'].iloc[t - 9:t + 1].sum()
    total120 = ctx['total_amount'].iloc[t - 119:t + 1].sum()
    a10 = amount.iloc[t - 9:t + 1][members].sum()
    a120 = amount.iloc[t - 119:t + 1][members].sum()
    above_now = (close.iloc[t][members] > ma60.iloc[t][members]).astype(float)
    above_then = (close.iloc[t - 10][members] > ma60.iloc[t - 10][members]).astype(float)
    r20 = close.iloc[t][members] / close.iloc[t - 20][members] - 1
    r60 = close.iloc[t][members] / close.iloc[t - 60][members] - 1
    frame = pd.DataFrame({'g': labels, 'a10': a10, 'a120': a120, 'up': above_now, 'up0': above_then,
                          'r20': r20, 'r60': r60})
    grouped = frame.groupby('g')
    size = grouped.size()
    stats = pd.DataFrame({
        'flow': (grouped.a10.sum() / total10) / (grouped.a120.sum() / total120).replace(0, np.nan),
        'breadth': grouped.up.mean(), 'breadth_change': grouped.up.mean() - grouped.up0.mean(),
        'rs': (grouped.r20.mean() - frame.r20.mean() + grouped.r60.mean() - frame.r60.mean()) / 2,
    })[size >= MIN_MEMBERS].dropna()
    pct = stats.rank(pct=True)
    diffusion = (pct.breadth + pct.breadth_change) / 2
    score = (pct.flow + diffusion.rank(pct=True) + pct.rs) / 3
    return score.sort_values(ascending=False)


def leaders(day, labels, group, ctx):
    t = ctx['pos'][day]
    members = labels.index[labels == group]
    adv20 = ctx['amount'].iloc[t - 19:t + 1][members].mean()
    tradable = ctx['close'].iloc[t][members].notna()
    return list(adv20[tradable].sort_values(ascending=False).index[:LEADERS])


def context(groups_by_date):
    panel, features, close, amount = base_state()
    sessions = panel.sessions
    return {'pos': {d: i for i, d in enumerate(sessions)}, 'amount': amount, 'close': close,
            'ma60': close.rolling(60, min_periods=60).mean(), 'total_amount': amount.sum(axis=1),
            'groups': groups_by_date, 'sessions': sessions}


def labels_on(day, ctx):
    dates = [d for d in ctx['group_dates'] if d <= day]
    return ctx['groups'][dates[-1]] if dates else None


def theme_schedule(ctx, phase, rng=None, replacements=None):
    """Returns (schedule, replacement counts, group picks). ``rng`` switches to random groups."""
    window = ctx['window']
    days = window[phase::5]
    held, stocks, schedule, counts, picks = [], {}, {}, [], {}
    prev_labels = None
    for k, day in enumerate(days):
        labels = labels_on(day, ctx)
        if labels is None:
            continue
        if prev_labels is not None and labels is not prev_labels and held:
            remapped = []
            for g in held:
                old = [c for c in stocks.get(g, []) if c in labels.index]
                if old:
                    new = labels.reindex(old).mode()
                    if len(new) and new.iloc[0] not in remapped:
                        remapped.append(new.iloc[0])
            held = remapped
        prev_labels = labels
        ranking = group_scores(day, labels, ctx)
        if rng is None:
            keep = [g for g in held if g in list(ranking.index[:BUFFER_RANK])]
            fill = [g for g in ranking.index if g not in keep][:HOLD - len(keep)]
            counts.append(HOLD - len(keep) if held else HOLD)
            held = keep + fill
        else:
            n = replacements[min(k, len(replacements) - 1)]
            eligible = list(ranking.index)
            keep = [g for g in held if g in eligible]
            if len(keep) > HOLD - n:
                keep = list(rng.choice(keep, size=HOLD - n, replace=False))
            pool = [g for g in eligible if g not in keep]
            fill = list(rng.choice(pool, size=min(HOLD - len(keep), len(pool)), replace=False))
            held = keep + fill
        stocks = {g: leaders(day, labels, g, ctx) for g in held}
        names = [c for g in held for c in stocks[g]]
        schedule[day] = {c: GROSS / len(names) for c in names} if names else {}
        picks[day] = list(held)
    return schedule, counts, picks


def prepare(kind):
    with open(OUT / ('clusters.pkl' if kind == 'C1a' else 'clusters_ward.pkl'), 'rb') as handle:
        clusters_by_date = pickle.load(handle)
    if kind == 'C0':
        panel, features, _, _ = base_state()
        sessions = panel.sessions
        window = sessions[(sessions >= pd.Timestamp(L.START)) & (sessions <= pd.Timestamp(L.END))]
        groups = industry_groups(panel, features, window)
    else:
        groups = clusters_by_date
    ctx = context(groups)
    sessions = ctx['sessions']
    ctx['window'] = sessions[(sessions >= pd.Timestamp(L.START)) & (sessions <= pd.Timestamp(L.END))]
    ctx['group_dates'] = sorted(groups)
    panel = Panel(L.VERSION)
    delist = panel.table('delistings')
    ctx['market'] = Market.from_panel(panel, L.START, L.END, delist=dict(zip(delist.code, delist.last_bar)))
    return ctx


def evaluate(nav, initial):
    stats = metrics(nav, initial_cash=initial)
    stats.update({k: segment_cagr(nav, a, b, initial_cash=initial) for k, (a, b) in L.SEGMENTS.items()})
    stats['ytd_2026'] = float(nav.iloc[-1] / nav.loc[:'2025-12-31'].iloc[-1] - 1)
    stats.pop('yearly')
    return stats


def _run_phase(args):
    kind, phase = args
    ctx = _S['ctx']
    schedule, counts, picks = theme_schedule(ctx, phase)
    result = simulate(ctx['market'], schedule, initial_cash=THEME_CASH, costs=Costs())
    stats = evaluate(result['nav'], THEME_CASH)
    stats.update(kind=kind, phase=phase, turnover=result['turnover'], fees=result['fees'], slippage=result['slippage'])
    return stats, result['nav'], counts, picks


def run(kind):
    _S['ctx'] = prepare(kind)
    with mp.get_context('fork').Pool(5) as pool:
        results = pool.map(_run_phase, [(kind, p) for p in range(5)])
    folder = OUT / kind
    folder.mkdir(parents=True, exist_ok=True)
    rows = [r[0] for r in results]
    pd.DataFrame(rows).to_csv(folder / 'runs.csv', index=False)
    pd.DataFrame({f'p{r[0]["phase"]}': r[1] for r in results}).to_parquet(folder / 'navs.parquet')
    with open(folder / 'picks.pkl', 'wb') as handle:
        pickle.dump({r[0]['phase']: {'counts': r[2], 'picks': r[3]} for r in results}, handle)
    frame = pd.DataFrame(rows)
    cols = ['cagr', 'max_drawdown', *L.SEGMENTS, 'ytd_2026']
    summary = {'kind': kind, 'median': frame[cols].median().to_dict(), 'phases': frame[['phase', *cols]].to_dict(orient='records')}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['median']))


def _random_one(draw):
    ctx = _S['ctx']
    rng = np.random.default_rng(7000 + draw)
    schedule, _, _ = theme_schedule(ctx, 0, rng=rng, replacements=_S['counts'])
    nav = simulate(ctx['market'], schedule, initial_cash=THEME_CASH, costs=Costs())['nav']
    stats = evaluate(nav, THEME_CASH)
    stats['draw'] = draw
    return stats


def random_draws(kind='C1', draws=200):
    _S['ctx'] = prepare(kind)
    with open(OUT / kind / 'picks.pkl', 'rb') as handle:
        _S['counts'] = pickle.load(handle)[0]['counts']
    with mp.get_context('fork').Pool(9) as pool:
        frame = pd.DataFrame(pool.map(_random_one, range(draws)))
    frame.to_csv(OUT / kind / 'random_runs.csv', index=False)
    real = pd.read_csv(OUT / kind / 'runs.csv').query('phase == 0').iloc[0]
    result = {'real_phase0': {k: float(real[k]) for k in ('cagr', 'max_drawdown')},
              'random_cagr': {q: float(frame.cagr.quantile(q)) for q in (0.1, 0.5, 0.9)},
              'random_max_drawdown': {q: float(frame.max_drawdown.quantile(q)) for q in (0.1, 0.5, 0.9)},
              'real_rank_pct': float((frame.cagr < real.cagr).mean()),
              'beats_random_p90': bool(real.cagr > frame.cagr.quantile(0.9))}
    (OUT / kind / 'random.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def _small_one(args):
    phase, seed = args
    snapshot = L.snapshots_for(L._STATE['snaps'], phase, 'adv', pead=L._STATE['pead'])
    schedule = universe.build_schedule(snapshot, seed, key_names=('tier', 'pead'), retain_key_names=('tier',),
                                       holdings=30)
    nav = simulate(L._STATE['market'], schedule, initial_cash=SMALL_CASH, costs=Costs())['nav']
    stats = evaluate(nav, SMALL_CASH)
    stats.update(phase=phase, seed=seed)
    return stats, nav


def small30():
    L.load_state()
    tasks = [(p, s) for p in L.PHASES for s in range(1, 21)]
    with mp.get_context('fork').Pool(9) as pool:
        results = pool.map(_small_one, tasks)
    folder = OUT / 'small30'
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r[0] for r in results])
    frame.to_csv(folder / 'runs.csv', index=False)
    pd.DataFrame({f"{r[0]['phase']}_{r[0]['seed']}": r[1] for r in results}).to_parquet(folder / 'navs.parquet')
    cols = ['cagr', 'max_drawdown', *L.SEGMENTS, 'ytd_2026']
    print(json.dumps(frame[cols].median().to_dict()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('clusters', 'run', 'random', 'small30'))
    parser.add_argument('--kind', default='C1')
    parser.add_argument('--method', default='ward')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'clusters': lambda: clusters(args.method), 'run': lambda: run(args.kind), 'random': lambda: random_draws(args.kind),
     'small30': small30}[args.stage]()
