#!/usr/bin/env python3
"""Mainline freeze and overfitting audit (see output/smallcap_audit_20261008/PREREGISTRATION.md).

Stages:
  bitcheck        the new simulator option at its default reproduces the frozen D navs
  sens            D leg with a changed delisting haircut or the whole-day seal rule
  oos_prep        snapshots and PEAD flags for 2015-07-01..2017-12-29 under the frozen rules
  oos_equity      A_adv / D_pead_adv on the out-of-sample window
  stats           Deflated Sharpe, CSCV PBO, size-beta regression, combos for every equity result
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import smallcap_layers as L
from quantlab import research_statistics as RS
from quantlab.smallcap import universe
from quantlab.smallcap.sim import Costs, metrics, segment_cagr, simulate

OUT = ROOT / 'output/smallcap_audit_20261008'
REBUILD = ROOT / 'output/smallcap_rebuild_20261006'
OOS_DIR = OUT / 'oos'
OOS_START, OOS_END = '2015-07-01', '2017-12-29'
OOS_SEGMENTS = {'2015H2_2016': ('2015-07-01', '2016-12-31'), '2017': ('2017-01-01', '2017-12-31')}
BENCH = ROOT / 'data/smallcap_assets/csindex/bench_raw'
WEIGHTS = {'small': 0.60, 'gold': 0.15, 'bond': 0.15, 'ndx': 0.10}

TRIALS = {
    'A_adv': REBUILD / 'equity/A_adv_cost1', 'B_cap': REBUILD / 'equity/B_cap_cost1',
    'C_screen_adv': REBUILD / 'equity/C_screen_adv_cost1', 'C_screen_cap': REBUILD / 'equity/C_screen_cap_cost1',
    'A_adv_crowd': REBUILD / 'equity/A_adv_crowd_cost1', 'D_pead_adv': REBUILD / 'equity/D_pead_adv_cost1',
    'F1': ROOT / 'output/smallcap_explore_20261007/equity/F1_cost1',
    'F2': ROOT / 'output/smallcap_explore_20261007/equity/F2_cost1',
    'F3': ROOT / 'output/smallcap_explore_20261007/equity/F3_cost1',
    'M1': ROOT / 'output/smallcap_explore_20261007/equity/M1_cost1',
    **{k: ROOT / f'output/smallcap_theme_20261008/{k}' for k in ('small30', 'C0', 'C1', 'C1a', 'T1', 'T2', 'T3')},
}
# Registered variants of the small-cap line, counted from the RESULTS files (lower bound):
# 17 equity legs + E4 (A/D x equity-only, cross-asset, gold-as-cash = 6) + exploration season windows (12)
# + asset sets G0/G1/G2 (3) + theme combos (T1, T2, T3, C1, mainline = 5)
N_SMALLCAP = 17 + 6 + 12 + 3 + 5

_STATE = {}


def run_task(args):
    config, phase, seed, overrides = args
    spec = L.CONFIGS[config]
    snapshot = L.snapshots_for(L._STATE['snaps'], phase, spec['pool'], spec.get('screen', False),
                               L._STATE.get('pead') if spec.get('pead') else None)
    keys = ('tier', 'pead') if spec.get('pead') else ('tier',)
    schedule = universe.build_schedule(snapshot, seed, key_names=keys,
                                       retain_key_names=('tier',) if spec.get('pead') else None)
    result = simulate(L._STATE['market'], schedule, costs=Costs(**overrides))
    nav = result['nav']
    stats = metrics(nav)
    stats.update({name: segment_cagr(nav, a, b) for name, (a, b) in L.SEGMENTS.items()})
    stats.update(config=config, phase=phase, seed=seed, fees=result['fees'], turnover=result['turnover'],
                 delist_events=len(result['events']), median_gross=float(result['gross'][result['gross'] > 0].median()))
    stats.pop('yearly')
    return stats, (nav.to_numpy(dtype=np.float32) if seed <= 20 else None)


def run_family(config, seeds, overrides, folder):
    tasks = [(config, phase, seed, overrides) for phase in L.PHASES for seed in seeds]
    rows, navs, t0 = [], {}, time.time()
    with mp.get_context('fork').Pool(9) as pool:
        for index, (stats, nav) in enumerate(pool.imap_unordered(run_task, tasks, chunksize=4)):
            rows.append(stats)
            if nav is not None:
                navs[f"{stats['phase']}_{stats['seed']}"] = nav
            if index % 200 == 0:
                print(json.dumps({'stage': folder.name, 'done': index, 'total': len(tasks),
                                  'seconds': round(time.time() - t0)}), flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'runs.csv', index=False)
    pd.DataFrame(navs, index=L._STATE['market'].dates).to_parquet(folder / 'navs.parquet')
    columns = ['cagr', 'max_drawdown', *L.SEGMENTS, 'median_gross', 'fees', 'turnover', 'delist_events']
    summary = {'runs': len(frame), 'overrides': overrides,
               'median': {c: float(frame[c].median()) for c in columns},
               'p10': {c: float(frame[c].quantile(0.1)) for c in ('cagr', 'max_drawdown')},
               'promotion_approved': False}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'stage': 'done', 'folder': folder.name, **summary['median']}), flush=True)


def bitcheck():
    L.load_state()
    frozen = pd.read_parquet(REBUILD / 'equity/D_pead_adv_cost1/navs.parquet')
    out = {}
    for phase, seed in ((0, 1), (8, 7), (16, 20)):
        _, nav = run_task(('D_pead_adv', phase, seed, {}))
        diff = float(np.max(np.abs(nav - frozen[f'{phase}_{seed}'].to_numpy(dtype=np.float32))))
        out[f'{phase}_{seed}'] = diff
    out['identical'] = all(v == 0.0 for v in out.values())
    (OUT / 'bitcheck.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    if not out['identical']:
        raise SystemExit('default simulator no longer reproduces the frozen navs')


def sens(name, seeds):
    overrides = {'haircut0': {'delist_haircut': 0.0}, 'haircut100': {'delist_haircut': 1.0},
                 'seal_day': {'seal': 'day'}}[name]
    L.load_state()
    run_family('D_pead_adv', seeds, overrides, OUT / 'sens' / name)


REL_DIRS = {'is': OUT / 'rel_is', 'oos': OUT / 'rel_oos'}
FROZEN_POOL_ON = universe.pool_on


def relative_pool_on(day, features, kind):
    """Check R: the bottom 40 percent of eligible names by 60-session amount (rank > ceil(0.6 N))."""
    if kind != 'adv':
        return FROZEN_POOL_ON(day, features, kind)
    adv = features['adv'].loc[day]
    history = features['history'].loc[day]
    eligible = adv[(history >= universe.MIN_HISTORY) & adv.notna()]
    ordered = eligible.sort_values(ascending=False, kind='mergesort')
    band = ordered.iloc[math.ceil(0.6 * len(ordered)):]
    return [code for code in band.index if adv[code] >= universe.MIN_ADV]


def use_window(which):
    if which.startswith('rel'):
        universe.pool_on = relative_pool_on
    folder = {'oos': OOS_DIR, 'rel_is': REL_DIRS['is'], 'rel_oos': REL_DIRS['oos']}[which]
    if which in ('oos', 'rel_oos'):
        L.START, L.END, L.SEGMENTS = OOS_START, OOS_END, OOS_SEGMENTS
    L.OUT = folder
    return folder


def use_oos_window():
    use_window('oos')


def oos_prep(which='oos'):
    import smallcap_e6 as E6
    folder = use_window(which)
    folder.mkdir(parents=True, exist_ok=True)
    L.prep()
    E6.L.OUT = folder
    try:
        E6.main()
    except KeyError as error:  # empty event study (no pool) only breaks the diagnostic table
        print(json.dumps({'event_study_failed': repr(error)}))


def oos_equity(config, seeds, which='oos'):
    folder = use_window(which)
    L.load_state()
    run_family(config, seeds, {}, folder / 'equity' / config)


def bench_series(code):
    rows = []
    for path in sorted(BENCH.glob(f'csindex_{code}*.json')):
        rows.extend(json.loads(path.read_text()).get('data') or [])
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates('tradeDate').assign(date=pd.to_datetime(frame.tradeDate)).set_index('date')
    return frame.close.astype(float).sort_index()


def mean_path(navs, index):
    navs = navs.reindex(index).ffill()
    return navs.pct_change(fill_method=None).iloc[1:].mean(axis=1)


def combo_path(navs, gold_as_cash=False):
    import smallcap_e7 as E7
    from smallcap_theme_combo import combine
    _, price = E7.etf_prices(navs.index)
    price = price.reindex(navs.index).ffill()
    if gold_as_cash:
        price['gold'] = 1.0
    paths = pd.DataFrame({c: combine({'small': navs[c], 'gold': price['gold'], 'bond': price['bond'],
                                      'ndx': price['ndx']}, WEIGHTS, {'small'}) for c in navs.columns})
    return paths


def newey_west(y, x, lags=20):
    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = xe[lag:].T @ xe[:-lag]
        S += w * (g + g.T)
    inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(inv @ S @ inv))
    return beta, beta / se


def pbo_cscv(matrix, blocks=16):
    values = matrix.to_numpy()
    edges = np.linspace(0, len(values), blocks + 1).astype(int)
    parts = [values[edges[i]:edges[i + 1]] for i in range(blocks)]
    logits, chosen = [], []
    for combo in itertools.combinations(range(blocks), blocks // 2):
        ins = np.vstack([parts[i] for i in combo])
        oos = np.vstack([parts[i] for i in range(blocks) if i not in combo])
        sr_in = ins.mean(0) / ins.std(0, ddof=1)
        sr_out = oos.mean(0) / oos.std(0, ddof=1)
        best = int(np.argmax(sr_in))
        rank = (sr_out < sr_out[best]).sum() + 1
        w = rank / (values.shape[1] + 1)
        logits.append(math.log(w / (1 - w)))
        chosen.append(matrix.columns[best])
    logits = np.array(logits)
    return {'pbo': float((logits <= 0).mean()), 'combinations': len(logits), 'blocks': blocks,
            'best_in_sample_counts': pd.Series(chosen).value_counts().to_dict()}


def dsr_block(returns, family_std, label):
    out = {}
    for trials in (N_SMALLCAP, 100, 500):
        result = RS.deflated_sharpe(returns, trials=trials, sharpe_std=family_std)
        out[str(trials)] = {k: result.get(k) for k in ('value', 'annualized_sharpe', 'daily_selection_threshold',
                                                       'observed_skew', 'observed_raw_kurtosis')}
    out['label'] = label
    return out


def stats():
    base = pd.read_parquet(TRIALS['D_pead_adv'] / 'navs.parquet')
    index = base.index
    family = pd.DataFrame({k: mean_path(pd.read_parquet(v / 'navs.parquet'), index) for k, v in TRIALS.items()})
    family = family.fillna(0.0)
    sharpes = family.mean() / family.std(ddof=1)
    family_std = float(sharpes.std(ddof=1))
    report = {'n_smallcap_lower_bound': N_SMALLCAP, 'family_daily_sharpe_std': family_std,
              'family_annual_sharpe': (sharpes * math.sqrt(252)).round(3).to_dict()}

    combo = combo_path(base)
    combo_ret = combo.pct_change(fill_method=None).iloc[1:].mean(axis=1)
    cash = combo_path(base, gold_as_cash=True)
    cash_ret = cash.pct_change(fill_method=None).iloc[1:].mean(axis=1)
    report['dsr'] = {'equity_D': dsr_block(family['D_pead_adv'], family_std, 'D leg, mean of 100 paths'),
                     'mainline_combo': dsr_block(combo_ret, family_std, 'D 60 + G0, mean of 100 paths'),
                     'combo_gold_as_cash': dsr_block(cash_ret, family_std, 'gold sleeve replaced by cash')}
    report['dsr_gate_mainline'] = bool(report['dsr']['mainline_combo'][str(N_SMALLCAP)]['value'] >= 0.95)
    report['pbo'] = pbo_cscv(family)
    report['pbo_gate'] = bool(report['pbo']['pbo'] <= 0.2)

    csi2000 = bench_series('932000CNY010').reindex(index).ffill().pct_change(fill_method=None)
    csi300 = bench_series('H00300').reindex(index).ffill().pct_change(fill_method=None)
    frame = pd.DataFrame({'d': family['D_pead_adv'], 'm': csi2000, 'big': csi300}).dropna()
    frame['size'] = frame.m - frame.big
    beta = {}
    for name, (a, b) in {'all': ('2018-01-01', '2026-12-31'), **L.SEGMENTS}.items():
        part = frame.loc[a:b]
        b1, t1 = newey_west(part.d.to_numpy(), part.m.to_numpy())
        b2, t2 = newey_west(part.d.to_numpy(), part[['big', 'size']].to_numpy())
        beta[name] = {'one_factor': {'alpha_annual': float(b1[0] * 252), 't_alpha': float(t1[0]), 'beta': float(b1[1])},
                      'two_factor': {'alpha_annual': float(b2[0] * 252), 't_alpha': float(t2[0]),
                                     'beta_mkt': float(b2[1]), 'beta_size': float(b2[2])},
                      'sessions': int(len(part))}
    report['size_beta'] = beta
    report['promotion_approved'] = False
    (OUT / 'stats.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ('family_annual_sharpe',)}, ensure_ascii=False,
                     indent=1))


def combos(folder, gold_as_cash=False):
    """G0 combination summary for any equity folder (used for sensitivity and out-of-sample rows)."""
    from smallcap_theme_combo import summarize
    if OOS_DIR in folder.parents or REL_DIRS['oos'] in folder.parents:
        L.SEGMENTS = OOS_SEGMENTS
    navs = pd.read_parquet(folder / 'navs.parquet')
    paths = combo_path(navs, gold_as_cash)
    summary = summarize([paths[c] for c in paths.columns])
    (folder / ('combo_cash.json' if gold_as_cash else 'combo.json')).write_text(json.dumps(summary, indent=1))
    print(json.dumps({'folder': str(folder.relative_to(OUT)), **summary['median']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('bitcheck', 'sens', 'oos_prep', 'oos_equity', 'stats', 'combo'))
    parser.add_argument('--name')
    parser.add_argument('--seeds', type=int, default=200)
    parser.add_argument('--cash', action='store_true')
    parser.add_argument('--window', default='oos', choices=('oos', 'rel_is', 'rel_oos'))
    args = parser.parse_args()
    seeds = range(1, args.seeds + 1)
    if args.stage == 'bitcheck':
        bitcheck()
    elif args.stage == 'sens':
        sens(args.name, seeds)
    elif args.stage == 'oos_prep':
        oos_prep(args.window)
    elif args.stage == 'oos_equity':
        oos_equity(args.name, seeds, args.window)
    elif args.stage == 'stats':
        stats()
    else:
        combos(OUT / args.name, args.cash)
