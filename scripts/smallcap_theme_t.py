#!/usr/bin/env python3
"""Theme leg, index version (T series). Rules: output/smallcap_theme_20261008/PREREGISTRATION.md.

Stages:
  run      T1 (U-A), T3 (U-B), T2 (market filter on the better of T1/T3); phases 0-4
  random   200 turnover-matched random draws for a universe (phase 0)
  etf      tracking check of index vs ETF after 2020
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import smallcap_e7 as E7
from quantlab.smallcap.sim import metrics, segment_cagr

OUT = ROOT / 'output/smallcap_theme_20261008'
DATA = ROOT / 'data/smallcap_assets/csindex'
HOLD, BUFFER_RANK, STEP = 4, 8, 5
COST, FEE = 0.0015, 0.005 / 252
BREAK = 0.20
FULL = ('2014-01-02', '2026-09-22')
MAIN = ('2018-01-02', '2026-09-22')
OOS = ('2014-01-02', '2017-12-29')
SEGMENTS = {'2018_2021': ('2018-01-02', '2021-12-31'), '2022_2025': ('2022-01-01', '2025-12-31'),
            '2025_now': ('2025-01-02', '2026-09-22')}
ETF_PAIRS = {'H30184': 'sh512480', '399975': 'sh512880', '399967': 'sh512660', '931152': 'sz159992', 'H30217': 'sz159883'}


def load(universe):
    bars = pd.read_parquet(DATA / 'sector_bars.parquet')
    meta = pd.read_parquet(DATA / 'sector_meta.parquet')
    close = bars.pivot_table(index='date', columns='code', values='close', aggfunc='last').sort_index()
    amount = bars.pivot_table(index='date', columns='code', values='amount_yi', aggfunc='last').sort_index()
    calendar = close['000985'].dropna().index
    close, amount = close.reindex(calendar), amount.reindex(calendar)
    returns = close.pct_change(fill_method=None)
    returns = returns.mask(returns.abs() > BREAK, 0.0)
    close = (1 + returns.fillna(0.0)).cumprod().where(close.notna())
    codes = list(meta.code[meta.ua]) if universe == 'UA' else list(meta.code[meta.ub])
    codes = [c for c in codes if c in close.columns]
    publish = pd.to_datetime(meta.set_index('code').publish).reindex(codes)
    return close, amount, codes, publish, meta


def signals(close, amount, codes, publish):
    c, a = close[codes], amount[codes]
    market_close, market_amount = close['000985'], amount['000985']
    history = c.notna().cumsum()
    live = (history >= 120) & (c.index.values[:, None] > publish.to_numpy()[None, :])
    flow = (a.rolling(10).sum().div(market_amount.rolling(10).sum(), axis=0)
            / a.rolling(120).sum().div(market_amount.rolling(120).sum(), axis=0))
    above = (c > c.rolling(60).mean()).astype(float)
    high = (c >= c.rolling(20).max()).astype(float)
    diffusion = (above + high) / 2
    rs = ((c / c.shift(20) - 1).sub(market_close / market_close.shift(20) - 1, axis=0)
          + (c / c.shift(60) - 1).sub(market_close / market_close.shift(60) - 1, axis=0)) / 2
    parts = [x.where(live) for x in (flow, diffusion, rs)]
    ranks = [x.rank(axis=1, pct=True) for x in parts]
    score = sum(ranks) / 3
    return score.where(live & parts[0].notna() & parts[2].notna())


def holdings_path(score, phase, start, end, rng=None, replacements=None):
    days = score.loc[start:end].index[phase::STEP]
    held, path, counts = [], {}, []
    for k, day in enumerate(days):
        row = score.loc[day].dropna().sort_values(ascending=False)
        if row.empty:
            continue
        if rng is None:
            keep = [g for g in held if g in list(row.index[:BUFFER_RANK])]
            fill = [g for g in row.index if g not in keep][:HOLD - len(keep)]
            counts.append(HOLD - len(keep) if held else HOLD)
        else:
            n = replacements[min(k, len(replacements) - 1)]
            keep = [g for g in held if g in row.index]
            if len(keep) > HOLD - n:
                keep = list(rng.choice(keep, size=HOLD - n, replace=False))
            pool = [g for g in row.index if g not in keep]
            fill = list(rng.choice(pool, size=min(HOLD - len(keep), len(pool)), replace=False))
        held = keep + fill
        path[day] = list(held)
    return path, counts


def simulate(close, path, start, end, market_filter=None, bond=None):
    """Signal at close t; new weights take effect at close t+1 (cost charged there)."""
    sessions = close.loc[start:end].index
    returns = close.pct_change(fill_method=None).reindex(sessions).fillna(0.0)
    bond_ret = bond.reindex(sessions).pct_change(fill_method=None).fillna(0.0) if bond is not None else None
    weights, pending, nav, value = {}, None, [], 1.0
    signal_days = set(path)
    for t, day in enumerate(sessions):
        gross = sum(w * returns.at[day, g] if g != 'bond' else w * bond_ret.at[day] for g, w in weights.items())
        value *= 1 + gross - FEE * sum(w for g, w in weights.items() if g != 'bond')
        if pending is not None:
            drift = {g: w * (1 + (returns.at[day, g] if g != 'bond' else bond_ret.at[day])) for g, w in weights.items()}
            total = sum(drift.values()) + (1 - sum(weights.values()))
            drift = {g: w / total for g, w in drift.items()}
            turnover = sum(abs(pending.get(g, 0) - drift.get(g, 0)) for g in set(pending) | set(drift))
            value *= 1 - COST * turnover
            weights, pending = pending, None
        if day in signal_days:
            names = path[day]
            scale = 1.0
            if market_filter is not None and market_filter.at[day]:
                scale = 0.5
            pending = {g: scale / len(names) for g in names}
            if scale < 1.0:
                pending['bond'] = 1 - scale
        nav.append(value)
    return pd.Series(nav, index=sessions)


def window_stats(nav, start, end):
    part = nav.loc[start:end]
    part = part / part.iloc[0] * 600_000
    stats = metrics(part)
    stats.pop('yearly')
    return stats


def report(nav):
    main = nav.loc[MAIN[0]:MAIN[1]]
    main = main / main.iloc[0] * 600_000
    stats = metrics(main)
    stats.pop('yearly')
    stats.update({k: segment_cagr(main, a, b) for k, (a, b) in SEGMENTS.items()})
    stats['ytd_2026'] = float(main.iloc[-1] / main.loc[:'2025-12-31'].iloc[-1] - 1)
    stats['oos_2014_2017'] = window_stats(nav, *OOS)['cagr'] if nav.index[0] <= pd.Timestamp('2014-06-30') else None
    return stats


def equal_weight(close, codes, publish, score):
    path = {day: list(score.loc[day].dropna().index) for day in score.loc[FULL[0]:FULL[1]].index[::STEP]
            if score.loc[day].notna().any()}
    return simulate(close, path, *FULL)


def run():
    results, navs_out, picks_out = {}, {}, {}
    _, price = E7.etf_prices(pd.DatetimeIndex([]))
    bond = price['bond']
    for name, universe in (('T1', 'UA'), ('T3', 'UB')):
        close, amount, codes, publish, meta = load(universe)
        score = signals(close, amount, codes, publish)
        rows, navs = [], {}
        for phase in range(5):
            path, counts = holdings_path(score, phase, *FULL)
            nav = simulate(close, path, *FULL)
            stats = report(nav)
            stats['phase'] = phase
            rows.append(stats)
            navs[f'p{phase}'] = nav
            if phase == 0:
                picks_out[name] = {'path': path, 'counts': counts}
        ew = report(equal_weight(close, codes, publish, score))
        frame = pd.DataFrame(rows)
        cols = ['cagr', 'max_drawdown', *SEGMENTS, 'ytd_2026', 'oos_2014_2017']
        results[name] = {'universe': universe, 'n_codes': len(codes), 'median': frame[cols].median().to_dict(),
                         'phases': frame[['phase', *cols]].to_dict(orient='records'), 'equal_weight': ew}
        navs_out[name] = pd.DataFrame(navs)
        print(json.dumps({name: results[name]['median'], 'ew': {k: ew[k] for k in cols}}), flush=True)
    best = max(('T1', 'T3'), key=lambda n: results[n]['median']['cagr'])
    close, amount, codes, publish, _ = load('UA' if best == 'T1' else 'UB')
    score = signals(close, amount, codes, publish)
    market = close['000985']
    market_filter = market < market.rolling(120).mean()
    rows, navs = [], {}
    for phase in range(5):
        path, _ = holdings_path(score, phase, *FULL)
        nav = simulate(close, path, *FULL, market_filter=market_filter, bond=bond)
        stats = report(nav)
        stats['phase'] = phase
        rows.append(stats)
        navs[f'p{phase}'] = nav
    frame = pd.DataFrame(rows)
    cols = ['cagr', 'max_drawdown', *SEGMENTS, 'ytd_2026', 'oos_2014_2017']
    results['T2'] = {'base': best, 'median': frame[cols].median().to_dict(), 'phases': frame[['phase', *cols]].to_dict(orient='records')}
    navs_out['T2'] = pd.DataFrame(navs)
    print(json.dumps({'T2': results['T2']['median']}), flush=True)
    for name, frame in navs_out.items():
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        frame.loc[MAIN[0]:MAIN[1]].to_parquet(folder / 'navs.parquet')
    import pickle
    with open(OUT / 'T_picks.pkl', 'wb') as handle:
        pickle.dump(picks_out, handle)
    (OUT / 'T_summary.json').write_text(json.dumps(results, indent=1, default=float))
    names = meta.set_index('code').name.to_dict()
    for name in ('T1', 'T3'):
        path = picks_out[name]['path']
        held = pd.Series([g for day in path for g in path[day]]).value_counts()
        recent = {str(d.date()): [names.get(g, g) for g in v] for d, v in path.items() if d >= pd.Timestamp('2026-01-01')}
        (OUT / f'{name}_picks_2026.json').write_text(json.dumps(
            {'most_held': {names.get(k, k): int(v) for k, v in held.head(15).items()},
             'decisions_2026': recent}, ensure_ascii=False, indent=1))


def random_draws(name='T1', draws=200):
    import pickle
    close, amount, codes, publish, _ = load('UA' if name == 'T1' else 'UB')
    score = signals(close, amount, codes, publish)
    with open(OUT / 'T_picks.pkl', 'rb') as handle:
        counts = pickle.load(handle)[name]['counts']
    rows = []
    for draw in range(draws):
        rng = np.random.default_rng(9000 + draw)
        path, _ = holdings_path(score, 0, *FULL, rng=rng, replacements=counts)
        stats = report(simulate(close, path, *FULL))
        stats['draw'] = draw
        rows.append(stats)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f'{name}_random_runs.csv', index=False)
    summary = json.loads((OUT / 'T_summary.json').read_text())
    real = [p for p in summary[name]['phases'] if p['phase'] == 0][0]
    result = {'real_phase0': {k: real[k] for k in ('cagr', 'max_drawdown', 'oos_2014_2017')},
              'random_cagr': {q: float(frame.cagr.quantile(q)) for q in (0.1, 0.5, 0.9)},
              'random_oos': {q: float(frame.oos_2014_2017.quantile(q)) for q in (0.1, 0.5, 0.9)},
              'real_rank_pct': float((frame.cagr < real['cagr']).mean()),
              'beats_random_p90': bool(real['cagr'] > frame.cagr.quantile(0.9))}
    (OUT / f'{name}_random.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def eastmoney_hfq(code):
    """Back-adjusted daily close from Eastmoney for ETFs Tencent blocked; raw archived."""
    import hashlib
    import urllib.request
    raw = ROOT / 'data/smallcap_assets/raw'
    archived = sorted(raw.glob(f'em_{code}_hfq_*.json'))
    if archived:
        payload = archived[-1].read_bytes()
    else:
        secid = ('0.' if code.startswith('sz') else '1.') + code[2:]
        url = (f'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1&fields2=f51,f53'
               '&klt=101&fqt=2&beg=20130101&end=20261007')
        payload = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30).read()
        (raw / f'em_{code}_hfq_{hashlib.sha256(payload).hexdigest()[:16]}.json').write_bytes(payload)
    rows = [line.split(',') for line in json.loads(payload)['data']['klines']]
    return pd.Series([float(r[1]) for r in rows], index=pd.to_datetime([r[0] for r in rows]))


def etf_check():
    bars = pd.read_parquet(DATA / 'sector_bars.parquet')
    close = bars.pivot_table(index='date', columns='code', values='close', aggfunc='last').sort_index()
    etf = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    etf = etf.pivot(index='date', columns='code', values='close_hfq').sort_index()
    for code in ETF_PAIRS.values():
        if code not in etf:
            try:
                series = eastmoney_hfq(code)
            except Exception:
                continue
            etf = etf.reindex(etf.index.union(series.index))
            etf[code] = series
    out = {}
    for index_code, etf_code in ETF_PAIRS.items():
        if index_code not in close or etf_code not in etf:
            out[index_code] = 'missing'
            continue
        pair = pd.concat({'i': close[index_code], 'e': etf[etf_code]}, axis=1).dropna().loc['2020-01-02':'2026-09-22']
        if len(pair) < 250:
            out[index_code] = 'too_short'
            continue
        r = pair.pct_change().dropna()
        years = (pair.index[-1] - pair.index[0]).days / 365.25
        ci = (pair.i.iloc[-1] / pair.i.iloc[0]) ** (1 / years) - 1
        ce = (pair.e.iloc[-1] / pair.e.iloc[0]) ** (1 / years) - 1
        out[index_code] = {'etf': etf_code, 'start': str(pair.index[0].date()), 'index_cagr': ci, 'etf_cagr': ce,
                           'gap': ce - ci, 'tracking_error': float((r.e - r.i).std() * np.sqrt(252))}
    (OUT / 'T_etf_check.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('run', 'random', 'etf'))
    parser.add_argument('--name', default='T1')
    args = parser.parse_args()
    {'run': run, 'random': lambda: random_draws(args.name), 'etf': etf_check}[args.stage]()
