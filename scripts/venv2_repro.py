#!/usr/bin/env python3
"""Reproduce the frozen small-cap audit numbers under .venv2 without touching the frozen outputs.

Stages (each writes only under output/venv2_repro_20261008):
  bitcheck   default simulator vs the frozen D navs
  prep       rel_is / rel_oos snapshots rebuilt, compared with the frozen pickles
  equity     A_adv for seeds 1..20 on rel_is / rel_oos, navs compared with the frozen navs
  stats      DSR / PBO / size-beta recomputed, compared with the frozen stats.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import smallcap_audit as AU

FROZEN = AU.OUT
OUT = ROOT / 'output/venv2_repro_20261008'


def redirect():
    OUT.mkdir(parents=True, exist_ok=True)
    AU.OUT = OUT
    AU.REL_DIRS = {'is': OUT / 'rel_is', 'oos': OUT / 'rel_oos'}
    AU.OOS_DIR = OUT / 'oos'


def write(name, payload):
    payload['python'] = platform.python_version()
    payload['numpy'] = np.__version__
    payload['pandas'] = pd.__version__
    (OUT / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(payload, ensure_ascii=False, default=str))


def equal(l, r):
    if type(l) is not type(r):
        return False
    if isinstance(l, dict):
        return l.keys() == r.keys() and all(equal(l[k], r[k]) for k in l)
    if isinstance(l, (list, tuple)):
        return len(l) == len(r) and all(equal(a, b) for a, b in zip(l, r))
    if isinstance(l, np.ndarray):
        return l.shape == r.shape and bool(np.array_equal(l, r, equal_nan=l.dtype.kind == 'f'))
    if isinstance(l, (pd.Index, pd.Series, pd.DataFrame)):
        return l.equals(r)
    return l == r


def same_pickle(a, b):
    with open(a, 'rb') as x, open(b, 'rb') as y:
        return equal(pickle.load(x), pickle.load(y))


def prep(window):
    folder = AU.use_window(window)
    folder.mkdir(parents=True, exist_ok=True)
    AU.L.prep()
    frozen = FROZEN / window
    write(f'prep_{window}.json', {
        'window': window,
        'snapshots_equal': same_pickle(folder / 'snapshots.pkl', frozen / 'snapshots.pkl'),
        'screen_counts_equal': json.loads((folder / 'screen_counts.json').read_text())
        == json.loads((frozen / 'screen_counts.json').read_text())})


def equity(window):
    folder = AU.use_window(window)
    AU.L.load_state()
    AU.run_family('A_adv', range(1, 21), {}, folder / 'equity' / 'A_adv')
    new = pd.read_parquet(folder / 'equity/A_adv/navs.parquet')
    old = pd.read_parquet(FROZEN / window / 'equity/A_adv/navs.parquet')[new.columns]
    diff = float(np.nanmax(np.abs(new.to_numpy() - old.to_numpy())))
    runs_new = pd.read_csv(folder / 'equity/A_adv/runs.csv')
    runs_old = pd.read_csv(FROZEN / window / 'equity/A_adv/runs.csv')
    key = ['phase', 'seed']
    merged = runs_new.merge(runs_old, on=key, suffixes=('', '_frozen'))
    cagr_pp = float((merged.cagr - merged.cagr_frozen).abs().max() * 100)
    write(f'equity_{window}.json', {'window': window, 'paths': int(new.shape[1]), 'max_abs_nav_diff': diff,
                                    'max_abs_cagr_diff_pp': cagr_pp, 'runs_compared': len(merged),
                                    'within_0p1pp': cagr_pp <= 0.1})


def stats():
    AU.stats()
    new, old = json.loads((OUT / 'stats.json').read_text()), json.loads((FROZEN / 'stats.json').read_text())
    n = str(AU.N_SMALLCAP)
    pick = lambda d: {'dsr_mainline': d['dsr']['mainline_combo'][n]['value'],
                      'dsr_equity_D': d['dsr']['equity_D'][n]['value'], 'pbo': d['pbo']['pbo'],
                      'alpha_all': d['size_beta']['all']['one_factor']['alpha_annual'],
                      't_alpha_all': d['size_beta']['all']['one_factor']['t_alpha']}
    a, b = pick(new), pick(old)
    write('stats_compare.json', {'venv2': a, 'frozen': b,
                                 'max_abs_diff': max(abs(a[k] - b[k]) for k in a)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('bitcheck', 'prep', 'equity', 'stats'))
    parser.add_argument('--window', default='rel_is', choices=('rel_is', 'rel_oos'))
    args = parser.parse_args()
    redirect()
    if args.stage == 'bitcheck':
        AU.bitcheck()
    elif args.stage == 'prep':
        prep(args.window)
    elif args.stage == 'equity':
        equity(args.window)
    else:
        stats()
