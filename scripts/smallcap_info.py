#!/usr/bin/env python3
"""Full-information screen for the small-cap leg (see output/smallcap_info_20261008/PREREGISTRATION.md).

Stages:
  signals  build the 18 pre-registered signals on every snapshot day (PIT rules in the preregistration)
  screen   pool-level IC / event spreads on the non-overlapping phase-0 grid, segments, circular-shift placebo
"""
from __future__ import annotations

import argparse
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
from quantlab.smallcap.panel import Panel

INFO = ROOT / 'data/smallcap_info'
OUT = ROOT / 'output/smallcap_info_20261008'
HORIZON = 20
PRIOR = {'M1': -1, 'M2': -1, 'M3': -1, 'B1': -1, 'B2': 1, 'K1': 1, 'K2': -1, 'P1': -1, 'P2': -1, 'P3': -1,
         'P4': -1, 'H1': -1, 'E1': 1, 'E2': 1, 'S1': 1, 'S2': 1, 'G1': -1}
EVENT = {'B1', 'B2', 'K2', 'E1', 'S1', 'S2'}
SIGNED = {'B2', 'E1', 'S1', 'S2'}
FORECAST_SIGN = {'预增': 1, '扭亏': 1, '略增': 1, '续盈': 1, '预减': -1, '首亏': -1, '续亏': -1, '略减': -1}
SEGMENTS = {'2018_2021': ('2018-01-01', '2021-12-31'), '2022_2025': ('2022-01-01', '2025-12-31'),
            '2025_now': ('2025-01-01', '2026-12-31')}
MIN_OFFSET = 12


def to_code(series):
    s = series.astype(str).str.zfill(6)
    prefix = np.where(s.str[0] == '6', 'sh.', np.where(s.str[0].isin(['0', '3']), 'sz.', 'bj.'))
    return pd.Series(prefix, index=s.index) + s


def avail_index(sessions, dates, lag=1):
    """Session position at which an item dated ``dates`` is first usable: first session strictly after
    the date for lag=1; ``lag`` sessions after the date's own session otherwise."""
    dates = pd.to_datetime(dates).dt.normalize().to_numpy()
    if lag == 0:
        return np.searchsorted(sessions.to_numpy(), dates, side='left')
    after = np.searchsorted(sessions.to_numpy(), dates, side='right')
    if lag == 1:
        return after
    on_or_before = after - 1
    return on_or_before + lag


def event_matrix(frame, value, sessions, codes, lag=1, how='sum'):
    """Aggregate events onto (available session x code); missing cells NaN."""
    idx = avail_index(sessions, frame['date'], lag)
    frame = frame.assign(t=idx, code=to_code(frame['SECURITY_CODE']))
    frame = frame[(frame.t < len(sessions)) & frame.code.isin(codes)]
    grouped = frame.groupby(['t', 'code'])[value].agg(how).unstack()
    grouped.index = sessions[grouped.index]
    return grouped.reindex(index=sessions, columns=codes)


def rolling_sum(matrix, window):
    return matrix.fillna(0.0).rolling(window, min_periods=1).sum()


def build_signals():
    panel = Panel(L.VERSION)
    sessions, codes = panel.sessions, panel.codes
    with open(L.OUT / 'snapshots.pkl', 'rb') as handle:
        days = sorted(pickle.load(handle)['snaps'])
    close = panel.field('close')
    adj = panel.field('adj_close')
    amount = panel.field('amount')
    floatcap = panel.field('float_mktcap').ffill(limit=20)
    float_shares = floatcap / close
    turnover = panel.field('turnover')
    sig = {}

    margin = pd.read_parquet(INFO / 'margin.parquet').rename(columns={'SCODE': 'SECURITY_CODE', 'DATE': 'date'})
    for col in ('RZYE', 'RZMRE', 'RQYE'):
        margin[col] = pd.to_numeric(margin[col], errors='coerce')
    margin = margin.drop_duplicates(['date', 'SECURITY_CODE'])
    rzye = event_matrix(margin, 'RZYE', sessions, codes, how='last')
    rzmre = event_matrix(margin, 'RZMRE', sessions, codes, how='last')
    rqye = event_matrix(margin, 'RQYE', sessions, codes, how='last')
    amount_lag = amount.shift(1)
    sig['M1'] = (rzye - rzye.shift(HORIZON)) / floatcap
    sig['M2'] = rzmre.rolling(HORIZON, min_periods=10).sum() / amount_lag.where(rzmre.notna()).rolling(
        HORIZON, min_periods=10).sum()
    sig['M3'] = rqye / floatcap

    board = pd.read_parquet(INFO / 'billboard.parquet').rename(columns={'TRADE_DATE': 'date'})
    board = board.drop_duplicates(['date', 'SECURITY_CODE']).assign(one=1.0)
    sig['B1'] = rolling_sum(event_matrix(board, 'one', sessions, codes), 60)
    inst = pd.read_parquet(INFO / 'billboard_inst.parquet').rename(columns={'TRADE_DATE': 'date'})
    inst['NET_BUY_AMT'] = pd.to_numeric(inst['NET_BUY_AMT'], errors='coerce')
    sig['B2'] = rolling_sum(event_matrix(inst.drop_duplicates(['date', 'SECURITY_CODE']), 'NET_BUY_AMT',
                                         sessions, codes), 60) / floatcap

    block = pd.read_parquet(INFO / 'block_trade.parquet').rename(columns={'TRADE_DATE': 'date'})
    for col in ('DEAL_AMT', 'PREMIUM_RATIO'):
        block[col] = pd.to_numeric(block[col], errors='coerce')
    block['wprem'] = block.DEAL_AMT * block.PREMIUM_RATIO
    amt = rolling_sum(event_matrix(block, 'DEAL_AMT', sessions, codes), 60)
    wprem = rolling_sum(event_matrix(block, 'wprem', sessions, codes), 60)
    sig['K1'] = (wprem / amt).where(amt > 0)
    sig['K2'] = amt / floatcap

    ret = adj.pct_change(fill_method=None)
    sig['P1'] = adj / adj.shift(HORIZON) - 1
    sig['P2'] = turnover.rolling(5, min_periods=3).mean() / turnover.rolling(120, min_periods=60).mean()
    sig['P3'] = ret.rolling(HORIZON, min_periods=10).max()
    hit = (close >= panel.field('limit_up') - 0.005) & (panel.field('tradestatus') > 0)
    sig['P4'] = hit.astype(float).rolling(HORIZON, min_periods=10).sum()

    holders = pd.read_parquet(INFO / 'holder_num.parquet').rename(columns={'HOLD_NOTICE_DATE': 'date'})
    holders['HOLDER_NUM_RATIO'] = pd.to_numeric(holders['HOLDER_NUM_RATIO'], errors='coerce')
    holders = holders.sort_values(['date', 'END_DATE'])
    sig['H1'] = event_matrix(holders, 'HOLDER_NUM_RATIO', sessions, codes, how='last').ffill(limit=250)

    fc = pd.read_parquet(INFO / 'forecast.parquet').rename(columns={'NOTICE_DATE': 'date'})
    fc = fc[fc.PREDICT_FINANCE_CODE.isin(['004', '002'])]
    fc['sign'] = fc.PREDICT_TYPE.map(FORECAST_SIGN).fillna(0.0)
    fc = (fc.sort_values(['date', 'REPORT_DATE', 'PREDICT_FINANCE_CODE'], ascending=[True, True, False])
          .drop_duplicates(['date', 'SECURITY_CODE', 'REPORT_DATE']))
    sig['E1'] = event_matrix(fc, 'sign', sessions, codes, how='last').ffill(limit=120).fillna(0.0)
    ex = pd.read_parquet(INFO / 'express.parquet').rename(columns={'NOTICE_DATE': 'date'})
    ex['yoy'] = pd.to_numeric(ex['JLRTBZCL'], errors='coerce').clip(-200, 200)
    sig['E2'] = event_matrix(ex.sort_values('date'), 'yoy', sessions, codes, how='last').ffill(limit=120)

    hc = pd.read_parquet(INFO / 'holder_change.parquet').rename(columns={'NOTICE_DATE': 'date'})
    direction = np.where(hc.DIRECTION.astype(str).str.contains('减'), -1.0, 1.0)
    hc['signed'] = direction * pd.to_numeric(hc['CHANGE_FREE_RATIO'], errors='coerce').abs().fillna(0.0)
    hc = hc.drop_duplicates(['date', 'SECURITY_CODE', 'HOLDER_NAME', 'CHANGE_NUM', 'START_DATE'])
    sig['S1'] = rolling_sum(event_matrix(hc, 'signed', sessions, codes), 120)
    exe = pd.read_parquet(INFO / 'exec_trade.parquet').rename(columns={'CHANGE_DATE': 'date'})
    exe['shares'] = pd.to_numeric(exe['CHANGE_SHARES'], errors='coerce')
    exe = exe.drop_duplicates(['date', 'SECURITY_CODE', 'PERSON_NAME', 'CHANGE_SHARES'])
    sig['S2'] = rolling_sum(event_matrix(exe, 'shares', sessions, codes, lag=3), 120) / float_shares

    pledge = pd.read_parquet(INFO / 'pledge.parquet').rename(columns={'TRADE_DATE': 'date'})
    pledge['ratio'] = pd.to_numeric(pledge['PLEDGE_RATIO'], errors='coerce')
    sig['G1'] = event_matrix(pledge.drop_duplicates(['date', 'SECURITY_CODE']), 'ratio', sessions, codes,
                             lag=5, how='last').ffill(limit=30)

    out = {name: frame.reindex(days).astype('float32') for name, frame in sig.items()}
    raw_open = panel.field('open')
    opens = raw_open * adj / close
    blocked = (raw_open >= panel.field('limit_up') - 0.005) | ~(panel.field('tradestatus') > 0)
    pos = {d: i for i, d in enumerate(sessions)}
    fwd = {}
    for day in days:
        t = pos[day]
        if t + HORIZON < len(sessions):
            ret = adj.iloc[t + HORIZON] / opens.iloc[t + 1] - 1
            fwd[day] = ret.where(~blocked.iloc[t + 1]).astype('float32')
    out['_fwd'] = pd.DataFrame(fwd).T
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'signals.pkl', 'wb') as handle:
        pickle.dump(out, handle)
    print(json.dumps({'stage': 'signals', 'days': len(days), 'signals': sorted(sig)}), flush=True)


def daily_stats(signal, fwd, members, grid, name):
    """Per decision day: Spearman IC (prior-signed) over names with a value, and the event spread."""
    prior = PRIOR[name]
    rows = []
    for day in grid:
        if day not in fwd.index:
            continue
        names = members[day]
        x = signal.loc[day].reindex(names)
        y = fwd.loc[day].reindex(names)
        y = y - y.mean()
        ok = x.notna() & y.notna()
        nonzero = (x[ok] != 0).mean() if ok.any() else 0.0
        ic = (x[ok].rank().corr(y[ok].rank()) * prior) if ok.sum() >= 30 and x[ok].nunique() > 1 else np.nan
        if name in SIGNED:
            good, bad = y[ok & (x * prior > 0)], y[ok & (x * prior < 0)]
        elif name in EVENT:
            good, bad = y[ok & (x == 0)], y[ok & (x != 0)]
        else:
            q = (x[ok] * prior).rank(pct=True)
            good, bad = y[ok][q >= 0.9], y[ok][q <= 0.1]
        spread = good.mean() - bad.mean() if len(good) >= 5 and len(bad) >= 5 else np.nan
        rows.append({'day': day, 'ic': ic, 'spread': spread, 'coverage': float(ok.mean()),
                     'nonzero': float(nonzero), 'n_bad': int(len(bad))})
    return pd.DataFrame(rows).set_index('day')


def tstat(series):
    s = series.dropna()
    return float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan


def screen():
    with open(OUT / 'signals.pkl', 'rb') as handle:
        sig = pickle.load(handle)
    with open(L.OUT / 'snapshots.pkl', 'rb') as handle:
        payload = pickle.load(handle)
    members = {d: list(e['adv']['codes']) for d, e in payload['snaps'].items()}
    fwd = sig['_fwd']
    grid = [d for d in payload['window'][0::HORIZON] if d in fwd.index]
    results = {}
    for name in PRIOR:
        stats = daily_stats(sig[name], fwd, members, grid, name)
        event_mode = name in EVENT and stats.nonzero.median() < 0.30
        metric = 'spread' if event_mode else 'ic'
        series = stats[metric]
        segments = {k: float(series.loc[a:b].mean()) for k, (a, b) in SEGMENTS.items()}
        sign_ok = sum(v > 0 for v in segments.values() if np.isfinite(v)) >= 2
        n = len(grid)
        placebo = []
        for k in range(MIN_OFFSET, n - MIN_OFFSET + 1):
            shifted_days = {grid[i]: grid[(i + k) % n] for i in range(n)}
            vals = []
            for day in grid:
                ret_day = shifted_days[day]
                here = set(members[day])
                names = [c for c in members[ret_day] if c in here]
                x = sig[name].loc[day].reindex(names)
                y = fwd.loc[ret_day].reindex(names)
                y = y - y.mean()
                ok = x.notna() & y.notna()
                if metric == 'ic':
                    vals.append(x[ok].rank().corr(y[ok].rank()) * PRIOR[name]
                                if ok.sum() >= 30 and x[ok].nunique() > 1 else np.nan)
                else:
                    if name in SIGNED:
                        g, b = y[ok & (x * PRIOR[name] > 0)], y[ok & (x * PRIOR[name] < 0)]
                    else:
                        g, b = y[ok & (x == 0)], y[ok & (x != 0)]
                    vals.append(g.mean() - b.mean() if len(g) >= 5 and len(b) >= 5 else np.nan)
            placebo.append(np.nanmean(vals))
        placebo = np.array(placebo)
        actual = float(series.mean())
        p = float((np.sum(placebo >= actual) + 1) / (len(placebo) + 1))
        t = tstat(series)
        coverage = float(stats.coverage.median())
        passed = bool(t >= 3.0 and sign_ok and p < 0.05 and (coverage >= 0.30 or event_mode))
        results[name] = {'metric': metric, 'mean': actual, 't': t, 'segments': segments,
                         'segments_same_sign': sign_ok, 'placebo_p': p, 'placebo_offsets': int(len(placebo)),
                         'coverage_median': coverage, 'nonzero_median': float(stats.nonzero.median()),
                         'ic_mean': float(stats.ic.mean()), 'ic_t': tstat(stats.ic),
                         'spread_mean': float(stats.spread.mean()), 'spread_t': tstat(stats.spread),
                         'decision_days': int(series.notna().sum()), 'passed': passed}
        print(json.dumps({name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[name].items()
                                 if k != 'segments'}}, ensure_ascii=False), flush=True)
    (OUT / 'screen.json').write_text(json.dumps(results, indent=1, ensure_ascii=False))


def info_scores(sig, names, days, shuffle_seed=None):
    """Pool percentile of each passed signal in its prior direction, missing = 0.5, equal-weight mean."""
    out = {}
    for day, codes in days.items():
        parts = []
        for name in names:
            x = sig[name].loc[day].reindex(codes) * PRIOR[name]
            parts.append(x.rank(pct=True).fillna(0.5).to_numpy())
        score = np.mean(parts, axis=0)
        if shuffle_seed is not None:
            score = np.random.default_rng([shuffle_seed, day.value % 2**32]).permutation(score)
        out[day] = score
    return out


def run_variant(args):
    variant, phase, seed, cost_mult = args
    from quantlab.smallcap import universe
    from quantlab.smallcap.sim import Costs, metrics, segment_cagr, simulate
    state = L._STATE
    base = L.snapshots_for(state['snaps'], phase, 'adv', False, state['pead'])
    shuffle = seed if variant.endswith('R') else None
    scores = state['scores_shuffled'][seed] if shuffle else state['scores']
    snapshot = []
    for day, codes, keys, industries in base:
        score = scores[day]
        if variant.startswith('I2'):
            keep = score > np.quantile(score, 0.2) if len(score) else score.astype(bool)
            codes = [c for c, k in zip(codes, keep) if k]
            keys = {k: v[keep] for k, v in keys.items()}
            key_names = ('tier', 'pead')
        else:
            tercile = np.ceil(pd.Series(score).rank(pct=True).to_numpy() * 3)
            keys = dict(keys, info=tercile)
            key_names = ('tier', 'info', 'pead')
        snapshot.append((day, codes, keys, industries))
    schedule = universe.build_schedule(snapshot, seed, key_names=key_names, retain_key_names=('tier',))
    result = simulate(state['market'], schedule, costs=Costs(multiplier=cost_mult))
    nav = result['nav']
    stats = metrics(nav)
    stats.update({name: segment_cagr(nav, a, b) for name, (a, b) in L.SEGMENTS.items()})
    stats.update(config=variant, phase=phase, seed=seed, cost_mult=cost_mult, fees=result['fees'],
                 slippage=result['slippage'], turnover=result['turnover'], delist_events=len(result['events']),
                 median_gross=float(result['gross'][result['gross'] > 0].median()))
    stats.pop('yearly')
    return stats, (nav.to_numpy(dtype=np.float32) if seed <= 20 else None)


def dleg(variant, seeds, cost_mult):
    import multiprocessing as mp
    import time
    L.load_state()
    with open(OUT / 'signals.pkl', 'rb') as handle:
        sig = pickle.load(handle)
    passed = [k for k, v in json.loads((OUT / 'screen.json').read_text()).items() if v['passed']]
    if not passed:
        raise SystemExit('no signal passed the screen; stage 2 is not run')
    days = {d: list(e['adv']['codes']) for d, e in L._STATE['snaps'].items()}
    L._STATE['scores'] = info_scores(sig, passed, days)
    if variant.endswith('R'):
        L._STATE['scores_shuffled'] = {s: info_scores(sig, passed, days, shuffle_seed=s) for s in seeds}
    tasks = [(variant, phase, seed, cost_mult) for phase in L.PHASES for seed in seeds]
    rows, navs, t0 = [], {}, time.time()
    with mp.get_context('fork').Pool(9) as pool:
        for index, (stats, nav) in enumerate(pool.imap_unordered(run_variant, tasks, chunksize=4)):
            rows.append(stats)
            if nav is not None:
                navs[f"{stats['phase']}_{stats['seed']}"] = nav
            if index % 100 == 0:
                print(json.dumps({'stage': variant, 'done': index, 'total': len(tasks),
                                  'seconds': round(time.time() - t0)}), flush=True)
    folder = OUT / 'equity' / f'{variant}_cost{cost_mult:g}'
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'runs.csv', index=False)
    pd.DataFrame(navs, index=L._STATE['market'].dates).to_parquet(folder / 'navs.parquet')
    summary = L.summarize(frame)
    summary['signals'] = passed
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'stage': 'done', 'variant': variant, **summary['median']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('signals', 'screen', 'dleg'))
    parser.add_argument('--variant', choices=('I1', 'I2', 'I1R', 'I2R'))
    parser.add_argument('--seeds', type=int, default=200)
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    if args.stage == 'dleg':
        dleg(args.variant, range(1, args.seeds + 1), args.cost)
    else:
        {'signals': build_signals, 'screen': screen}[args.stage]()
