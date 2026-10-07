#!/usr/bin/env python3
"""Second exploration round (see output/smallcap_explore_20261007/PREREGISTRATION.md).

Stages:
  keys      cap / momentum key frames on the decision window
  equity    --config F1|F2|F3|M1 [--cost k]   seeds 1-200 x 5 phases
  season    descriptive May-June stats + twelve two-month halving windows (seeds 1-40)
  assets    G0/G1/G2 on the D equity leg, plus ETF-only out-of-sample 2014-2017
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

import smallcap_e4 as E4
import smallcap_layers as L
from quantlab.smallcap import universe
from quantlab.smallcap.panel import Panel
from quantlab.smallcap.sim import Costs, metrics, segment_cagr, simulate

OUT = ROOT / 'output/smallcap_explore_20261007'
FLOOR = -1e18
CONFIGS = {
    'F1': ('tier', 'small'),
    'F2': ('small',),
    'F3': ('tier', 'pead', 'small'),
    'M1': ('tier', 'mom'),
}
SEASON_SEEDS = range(1, 41)
YTD_START = '2025-12-31'


def keys():
    panel = Panel(L.VERSION)
    features = universe.rolling_features(panel)
    close = panel.field('adj_close')
    with open(L.OUT / 'snapshots.pkl', 'rb') as handle:
        window = pickle.load(handle)['window']
    small = (-features['cap']).reindex(window)
    mom = (close.shift(20) / close.shift(120) - 1).reindex(window)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'keys.pkl', 'wb') as handle:
        pickle.dump({'small': small, 'mom': mom}, handle)
    print(json.dumps({'sessions': len(window), 'small_valid': float(small.notna().mean().mean()),
                      'mom_valid': float(mom.notna().mean().mean())}))


def load():
    L.load_state()
    with open(OUT / 'keys.pkl', 'rb') as handle:
        L._STATE.update(pickle.load(handle))


def snapshots(phase, names):
    grid = universe.decision_grid(L._STATE['window'], L.START, L.END, phase=phase)
    out = []
    for day in grid:
        entry = L._STATE['snaps'][day]['adv']
        codes = list(entry['codes'])
        keyset = {'tier': entry['tier']}
        for name in names:
            if name != 'tier':
                keyset[name] = L._STATE[name].loc[day].reindex(codes).astype(float).fillna(FLOOR).to_numpy()
        out.append((day, codes, keyset, L._STATE['snaps'][day]['industries']))
    return out


def ytd(nav):
    return float(nav.iloc[-1] / nav.loc[:YTD_START].iloc[-1] - 1)


def run_equity(args):
    config, phase, seed, cost_mult = args
    names = CONFIGS[config]
    schedule = universe.build_schedule(snapshots(phase, names), seed, key_names=names)
    result = simulate(L._STATE['market'], schedule, costs=Costs(multiplier=cost_mult))
    nav = result['nav']
    stats = metrics(nav)
    stats.update({k: segment_cagr(nav, a, b) for k, (a, b) in L.SEGMENTS.items()})
    stats.update(config=config, phase=phase, seed=seed, cost_mult=cost_mult, fees=result['fees'],
                 slippage=result['slippage'], turnover=result['turnover'], delist_events=len(result['events']),
                 median_gross=float(result['gross'][result['gross'] > 0].median()), ytd_2026=ytd(nav))
    stats.pop('yearly')
    return stats, (nav.to_numpy(dtype=np.float32) if seed <= 20 else None)


def equity(config, cost_mult):
    load()
    tasks = [(config, p, s, cost_mult) for p in L.PHASES for s in range(1, 201)]
    rows, navs = [], {}
    with mp.get_context('fork').Pool(9) as pool:
        for stats, nav in pool.imap_unordered(run_equity, tasks, chunksize=4):
            rows.append(stats)
            if nav is not None:
                navs[f"{stats['phase']}_{stats['seed']}"] = nav
    folder = OUT / 'equity' / f'{config}_cost{cost_mult:g}'
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'runs.csv', index=False)
    pd.DataFrame(navs, index=L._STATE['market'].dates).to_parquet(folder / 'navs.parquet')
    summary = L.summarize(frame)
    summary['median']['ytd_2026'] = float(frame.ytd_2026.median())
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'config': config, 'cost': cost_mult, **{k: round(v, 4) for k, v in summary['median'].items()}}))


def window_states(sessions, start_month):
    """0.5 from the last session before ``start_month`` to the last session of the following month."""
    state = pd.Series(1.0, index=sessions)
    month_key = sessions.year * 12 + (sessions.month - 1)
    last_of = pd.Series(sessions, index=sessions).groupby(month_key).max()
    for year in range(sessions[0].year - 1, sessions[-1].year + 1):
        first = year * 12 + start_month - 1
        before, end = first - 1, first + 1
        if before not in last_of.index or end not in last_of.index:
            continue
        state.loc[last_of[before]:last_of[end]] = 0.5
        state.loc[last_of[end]] = 1.0
    return state


def run_season(args):
    month, phase, seed = args
    snapshot = L.snapshots_for(L._STATE['snaps'], phase, 'adv')
    schedule = universe.build_schedule(snapshot, seed)
    if month:
        schedule = L.apply_crowding(schedule, window_states(L._STATE['window'], month), L._STATE['window'])
    nav = simulate(L._STATE['market'], schedule, costs=Costs())['nav']
    stats = metrics(nav)
    out = {'month': month, 'phase': phase, 'seed': seed, 'cagr': stats['cagr'], 'max_drawdown': stats['max_drawdown']}
    out.update({k: segment_cagr(nav, a, b) for k, (a, b) in L.SEGMENTS.items()})
    return out


def season():
    load()
    navs = pd.read_parquet(L.OUT / 'equity/A_adv_cost1/navs.parquet')
    monthly = navs.resample('ME').last().pct_change(fill_method=None)
    rows = {}
    for year in range(2018, 2026):
        part = monthly[monthly.index.year == year]
        may_jun = (1 + part[part.index.month.isin([5, 6])]).prod() - 1
        others = part[~part.index.month.isin([5, 6])]
        rows[year] = float(((may_jun - 2 * others.mean())).median())
    negative_years = sum(v < 0 for v in rows.values())
    all_months = {m: float(monthly[monthly.index.month == m].median(axis=1).mean()) for m in range(1, 13)}
    tasks = [(m, p, s) for m in range(0, 13) for p in L.PHASES for s in SEASON_SEEDS]
    with mp.get_context('fork').Pool(9) as pool:
        frame = pd.DataFrame(pool.map(run_season, tasks, chunksize=8))
    frame.to_csv(OUT / 'season_runs.csv', index=False)
    med = frame.groupby('month')[['cagr', 'max_drawdown', *L.SEGMENTS]].median()
    delta = med.sub(med.loc[0])
    ranking = delta.drop(index=0).cagr.rank(ascending=False)
    result = {'may_june_excess_by_year': rows, 'negative_years_of_8': negative_years,
              'mean_monthly_return_by_month_2018_2025': all_months,
              'base': med.loc[0].to_dict(),
              'windows_delta': {int(m): delta.loc[m].round(5).to_dict() for m in delta.index if m},
              'may_window_cagr_rank_of_12': int(ranking.loc[5])}
    (OUT / 'season.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


# ---------- assets ----------

def etf_prices(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.DatetimeIndex(wide.index).as_unit('ns')
    sessions = pd.DatetimeIndex(sessions).as_unit('ns')
    ret = wide.ffill().pct_change(fill_method=None)
    bond = ret['sh511260'].where(wide.index >= pd.Timestamp('2017-08-24'), ret['sh511010'])
    ret = pd.DataFrame({'gold': ret['sh518880'], 'bond': bond, 'ndx': ret['sh513100']}).fillna(0.0)
    price = (1 + ret).cumprod()
    price = price.reindex(price.index.union(sessions)).ffill()
    return price.pct_change(fill_method=None).fillna(0.0), price


BASE = {'gold': 0.15, 'bond': 0.15, 'ndx': 0.10}


def rule_weights(rule, day, price, ret, etf_share):
    base = {k: v / sum(BASE.values()) * etf_share for k, v in BASE.items()}
    if rule == 'G0':
        return base
    hist = price.loc[:day]
    if rule == 'G1':
        recent = ret.loc[:day].tail(60)
        if len(recent) < 20:
            return base
        vol = recent.std()
        inv = 1 / vol.replace(0, np.nan)
        inv = inv / inv.sum()
        return {k: float(inv[k]) * etf_share for k in BASE}
    if rule == 'G2':
        above = {k: hist[k].iloc[-1] > hist[k].tail(200).mean() for k in BASE}
        out = {k: (base[k] if above[k] else 0.0) for k in BASE}
        moved = sum(base[k] for k in BASE if not above[k] and k != 'bond')
        if above['bond']:
            out['bond'] += moved
        out['cash'] = etf_share - sum(out.values())
        return out
    raise ValueError(rule)


def combine_rule(equity_nav, sessions, rule, ret, price, equity_share=0.60, initial=600_000.0):
    etf_share = 1 - equity_share
    if equity_nav is not None:
        eq_ret = equity_nav.pct_change(fill_method=None).fillna(equity_nav.iloc[0] / initial - 1)
    frame = pd.Series(sessions, index=sessions)
    if rule == 'G2':
        ends = set(frame.groupby([sessions.year, sessions.month]).max())
    else:
        ends = set(frame.groupby([sessions.year, sessions.quarter]).max())
    weights = rule_weights(rule, sessions[0], price, ret, etf_share)
    if equity_share:
        weights['equity'] = equity_share
    sleeves = {k: initial * w for k, w in weights.items()}
    out, costs = np.empty(len(sessions)), 0.0
    prev = None
    for t, day in enumerate(sessions):
        for k in sleeves:
            if k == 'equity':
                sleeves[k] *= 1 + eq_ret.iloc[t]
            elif k != 'cash' and prev is not None:
                sleeves[k] *= price.at[day, k] / price.at[prev, k]
        prev = day
        total = sum(sleeves.values())
        if day in ends:
            target = rule_weights(rule, day, price, ret, etf_share)
            if equity_share:
                target['equity'] = equity_share
            names = set(target) | set(sleeves)
            fee = sum(abs(total * target.get(k, 0.0) - sleeves.get(k, 0.0)) *
                      (0.004 if k == 'equity' else 0.0 if k == 'cash' else 0.0005) for k in names)
            costs += fee
            total -= fee
            sleeves = {k: total * w for k, w in target.items()}
        out[t] = sum(sleeves.values())
    return pd.Series(out, index=sessions), costs


def sharpe(nav):
    r = nav.pct_change(fill_method=None).dropna()
    return float(r.mean() / r.std() * np.sqrt(252))


def _asset_one(args):
    rule, column = args
    navs = _ASSET['navs']
    nav, _ = combine_rule(navs[column], navs.index, rule, _ASSET['ret'], _ASSET['price'])
    stats = metrics(nav)
    stats.update({k: segment_cagr(nav, a, b) for k, (a, b) in L.SEGMENTS.items()})
    stats.update(rule=rule, run=column, ytd_2026=ytd(nav), sharpe=sharpe(nav))
    stats.pop('yearly')
    return stats


_ASSET = {}


def assets(equity_config='equity/D_pead_adv_cost1'):
    navs = pd.read_parquet(L.OUT / f'{equity_config}/navs.parquet')
    ret, price = etf_prices(navs.index)
    _ASSET.update(navs=navs, ret=ret, price=price)
    tasks = [(r, c) for r in ('G0', 'G1', 'G2') for c in navs.columns]
    with mp.get_context('fork').Pool(9) as pool:
        frame = pd.DataFrame(pool.map(_asset_one, tasks, chunksize=10))
    frame.to_csv(OUT / 'assets_runs.csv', index=False)
    cols = ['cagr', 'max_drawdown', *L.SEGMENTS, 'ytd_2026', 'sharpe']
    med = frame.groupby('rule')[cols].median()
    share = frame.groupby('rule').apply(lambda f: float(((f.cagr >= .15) & (f.max_drawdown >= -.25)).mean()))
    oos = {}
    for label, (a, b) in {'2014_2017': ('2014-01-02', '2017-12-29'), '2018_2026': ('2018-01-02', '2026-09-22')}.items():
        sessions = ret.loc[a:b].index
        sessions = sessions[sessions.dayofweek < 5]
        r_, p_ = ret.loc[:b], price.loc[:b]
        oos[label] = {}
        for rule in ('G0', 'G1', 'G2'):
            nav, _ = combine_rule(None, sessions, rule, r_, p_, equity_share=0.0)
            stats = metrics(nav)
            oos[label][rule] = {'cagr': stats['cagr'], 'max_drawdown': stats['max_drawdown'], 'sharpe': sharpe(nav)}
    result = {'equity_leg': equity_config, 'median': med.round(5).to_dict(orient='index'),
              'share_cagr15_dd25': share.to_dict(), 'etf_only': oos, 'promotion_approved': False}
    (OUT / f"assets_{equity_config.split('/')[-1]}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('keys', 'equity', 'season', 'assets'))
    parser.add_argument('--config')
    parser.add_argument('--cost', type=float, default=1.0)
    parser.add_argument('--leg', default='equity/D_pead_adv_cost1')
    args = parser.parse_args()
    if args.stage == 'keys':
        keys()
    elif args.stage == 'equity':
        equity(args.config, args.cost)
    elif args.stage == 'season':
        season()
    else:
        assets(args.leg)
