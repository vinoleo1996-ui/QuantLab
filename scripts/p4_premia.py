#!/usr/bin/env python3
"""P4 premia books (docs/prereg/P4_PREMIA.md).

High-liquidity names are retrained on a 5-day raw open-to-open return.
The low-liquidity sleeve reuses the published hysteresis net-asset value.
Four ETFs are equal weight. The mix uses a frozen 60-session trailing return.
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

import p4_domain as D
from quantlab import p4_ledger
from quantlab import research_trials as T
from quantlab.alpha import features as F
from quantlab.alpha.labels import forward_open_returns
from quantlab.alpha.models import GBDTEnsemble, percentile_frame
from quantlab.alpha.portfolio import GROSS, random_order, schedule
from quantlab.research_statistics import newey_west
from quantlab.smallcap import universe

PREREG = 'docs/prereg/P4_PREMIA.md'
OUT = ROOT / 'output/p4_premia_20261016'
HORIZON = 5
STEP = 5
PHASES = (0, 1, 2, 3, 4)
SEEDS = (1, 2)
WINDOW = 60
STOCK_COST, ASSET_COST = 0.004, 0.0005
POOL_CAGR, PAPER_CAGR = 0.1211, 0.1176
LOW_NAV = ROOT / 'output/p4_lowturn_20261013/ledger/nav_GBDT_HYST_p0_cost1.csv'
POOL_NAV = ROOT / 'output/p4_alpha_20261012/ledger/nav_POOL50_p0_s1_cost1.csv'
ETFS = ('sh518880', 'sh511010', 'sh513100', 'sh513500')


def liquid_codes(day, features, amount):
    groups = universe.domains_on(day, features, exclude_st=True, min_history=252)
    traded = amount.loc[day]
    return [code for code in groups['liquid'] if traded.get(code, 0) > 0]


def stage_build(out):
    panel, fields, financials, industries = D.load_inputs(False)
    features = universe.rolling_features(panel, with_st=True)
    price = F.PriceVolume(fields)
    fundamentals = F.Fundamentals(financials) if financials is not None else None
    sectors = F.Industries(industries) if industries is not None else None
    label = forward_open_returns(fields, horizon=HORIZON)
    sessions = panel.sessions
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    days = [day for day in sessions if pd.Timestamp('2015-01-05') <= day <= pd.Timestamp(D.END)]
    frames = []
    for n, day in enumerate(days, start=1):
        if n == 1 or n % 50 == 0 or n == len(days):
            print(json.dumps(dict(build_day=n, build_days=len(days), date=str(pd.Timestamp(day).date()))), flush=True)
        pool = liquid_codes(day, features, fields['amount'])
        if not pool:
            continue
        frame = F.cross_section(day, pool, price, fundamentals, sectors, fields['close'])
        frame['label_raw'] = label.loc[day].reindex(pool).to_numpy()
        frame['label_known_pos'] = position[day] + 1 + HORIZON
        frame['industry'] = sectors.on(day).reindex(pool).fillna('unclassified').to_numpy() if sectors else 'unclassified'
        frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    present = [f for f in F.ALL_FEATURES if f in samples]
    samples = percentile_frame(samples, present)
    samples['label'] = samples['label_raw']
    out.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out / 'samples.parquet', index=False)
    meta = dict(features=present, rows=len(samples), days=int(samples.date.nunique()), horizon=HORIZON)
    (out / 'samples_meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def stage_train(out, smoke=False):
    samples = pd.read_parquet(out / 'samples.parquet')
    meta = json.loads((out / 'samples_meta.json').read_text())
    present = meta['features']
    sessions = D.Panel(D.PANEL).sessions
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    refits = D.refit_days(samples.date.unique(), D.ANCHOR)
    scores, diagnostics = [], {}
    for i, refit in enumerate(refits):
        stop = refits[i + 1] if i + 1 < len(refits) else samples.date.max() + pd.Timedelta(days=1)
        train = samples[(samples.label_known_pos <= position[refit]) & samples.label.notna()]
        test = samples[(samples.date >= refit) & (samples.date < stop)].copy()
        if train.date.nunique() < 20 or test.empty:
            print(json.dumps(dict(refit=str(pd.Timestamp(refit).date()), skipped=True)), flush=True)
            continue
        gbdt = GBDTEnsemble().fit(train, present, seeds=(1, 2) if smoke else (1, 2, 3, 4, 5))
        test['GBDT'] = gbdt.predict(test)
        scores.append(test[['date', 'code', 'industry', 'GBDT']])
        diagnostics[str(pd.Timestamp(refit).date())] = dict(
            train_rows=len(train), train_days=int(train.date.nunique()),
            rounds=[b.best_iteration for b in gbdt.boosters])
        print(json.dumps(dict(refit=str(pd.Timestamp(refit).date()), train_rows=len(train), test_rows=len(test))), flush=True)
    frame = pd.concat(scores, ignore_index=True)
    frame.to_parquet(out / 'scores.parquet', index=False)
    (out / 'train_diagnostics.json').write_text(json.dumps(diagnostics, indent=1))


def stage_curve(out):
    """One seed of each finished refit, same data and parameters, so the loss curve can be shown."""
    samples = pd.read_parquet(out / 'samples.parquet')
    meta = json.loads((out / 'samples_meta.json').read_text())
    present = meta['features']
    sessions = D.Panel(D.PANEL).sessions
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    refits = D.refit_days(samples.date.unique(), D.ANCHOR)
    curves = []
    path = out / 'train_curve.json'
    for i, refit in enumerate(refits):
        stop = refits[i + 1] if i + 1 < len(refits) else samples.date.max() + pd.Timedelta(days=1)
        train = samples[(samples.label_known_pos <= position[refit]) & samples.label.notna()]
        if train.date.nunique() < 20:
            continue
        history = []
        GBDTEnsemble().fit(train, present, seeds=(1,), eval_history=history)
        item = dict(refit=str(pd.Timestamp(refit).date()), seed=1,
                    train_l2=history[0].get('train', {}).get('l2', []),
                    valid_l2=history[0].get('valid', {}).get('l2', []))
        curves.append(item)
        path.write_text(json.dumps(curves))
        print(json.dumps(dict(curve=item['refit'], rounds=len(item['valid_l2']),
                              valid_l2_end=item['valid_l2'][-1] if item['valid_l2'] else None)), flush=True)


def stage_plans(out):
    scores = pd.read_parquet(out / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(D.ANCHOR)]
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    result = {}
    for phase in PHASES:
        rebalance = days[phase::STEP]
        industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
        order = lambda day: list(by_day[day].sort_values('GBDT', ascending=False, kind='mergesort').code)
        result[f'LIQ_HIGH_p{phase}'] = schedule(rebalance, order, industries)
        for seed in SEEDS:
            rand = lambda day, s=seed: random_order(list(by_day[day].code), s)
            result[f'LIQ_HIGH_RAND_p{phase}_s{seed}'] = schedule(rebalance, rand, industries)
    payload = {name: {str(pd.Timestamp(day).date()): weights for day, weights in book.items()} for name, book in result.items()}
    (out / 'plans.json').write_text(json.dumps(payload))
    print(json.dumps(dict(paths=len(payload), days=len(days))), flush=True)


def stage_ledger(out, only=None, cost=1.0):
    folder = out / 'ledger'
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.loads((out / 'plans.json').read_text())
    if only:
        payload = {name: payload[name] for name in only}
    built = {name: {pd.Timestamp(day): weights for day, weights in book.items()} for name, book in payload.items()}
    targets = {name: p4_ledger.to_targets(book) for name, book in built.items()}
    codes = sorted({c for rows in targets.values() for row in rows for c in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    rows = []
    for name, target in targets.items():
        result = p4_ledger.replay(target, start=D.ANCHOR, end=D.END, cost_mult=cost, dataset=dataset)
        summary = result['summary']
        result['nav'].to_frame('nav').to_csv(folder / f'nav_{name}_cost{cost:g}.csv')
        row = dict(path=name, cost_mult=cost, cagr=summary['cagr'], mdd=summary['max_drawdown'],
                   fees=summary['fees'], slippage=summary['slippage'], fills=summary['fills'],
                   checks_passed=summary['account_checks_passed'],
                   missing_panels=len(summary.get('missing_panels', [])))
        (folder / f'part_{name}_cost{cost:g}.json').write_text(json.dumps(row))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def asset_nav(index):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    prices = wide[list(ETFS)].reindex(wide.index.union(index)).ffill().reindex(index)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    months = set(pd.Series(index, index=index).groupby([index.year, index.month]).max())
    weights = np.full(len(ETFS), 1 / len(ETFS))
    nav, values = 1.0, []
    for i, day in enumerate(index):
        if i > 0:
            growth = weights * (1 + returns.iloc[i].to_numpy())
            nav *= growth.sum()
            weights = growth / growth.sum()
        if day in months:
            trade = np.abs(np.full(len(ETFS), 1 / len(ETFS)) - weights).sum()
            nav *= 1 - ASSET_COST * trade
            weights = np.full(len(ETFS), 1 / len(ETFS))
        values.append(nav)
    return pd.Series(values, index=index)


def mix_weights(cum):
    """Positive trailing returns share the weight. All non-positive means cash."""
    positive = {name: value for name, value in cum.items() if value > 0}
    if not positive:
        return {name: 0.0 for name in cum}
    total = sum(positive.values())
    return {name: positive.get(name, 0.0) / total for name in cum}


def stage_mix(out, use_high):
    low = pd.read_csv(LOW_NAV, index_col=0, parse_dates=True)['nav']
    asset_index = low.loc[D.ANCHOR:D.END].index
    books = {'low': low.reindex(asset_index).ffill()}
    books['asset'] = asset_nav(asset_index)
    if use_high:
        high = pd.read_csv(out / 'ledger/nav_LIQ_HIGH_p0_cost1.csv', index_col=0, parse_dates=True)['nav']
        books['high'] = high.reindex(asset_index).ffill()
    else:
        books['high'] = pd.Series(1.0, index=asset_index)
    aligned = pd.DataFrame(books).dropna()
    nav, weights = 1.0, {name: 0.0 for name in aligned.columns}
    values, zero_low = [], []
    for i, day in enumerate(aligned.index):
        if i == 0:
            values.append(nav)
            continue
        if i - 1 >= WINDOW:
            cum = {name: aligned[name].iloc[i - 1] / aligned[name].iloc[i - 1 - WINDOW] - 1
                   for name in aligned.columns}
            if not use_high:
                cum['high'] = -1.0
            pending = mix_weights(cum)
            if pending['low'] == 0.0 and weights['low'] > 0:
                zero_low.append(dict(date=str(pd.Timestamp(day).date()), low_cum=cum['low']))
        else:
            pending = {name: 0.0 for name in aligned.columns}
        change = (STOCK_COST * (abs(pending['low'] - weights['low']) + abs(pending['high'] - weights['high']))
                  + ASSET_COST * abs(pending['asset'] - weights['asset']))
        growth = sum(pending[name] * (aligned[name].iloc[i] / aligned[name].iloc[i - 1] - 1)
                     for name in aligned.columns)
        nav *= (1 + growth) * (1 - change)
        weights = pending
        values.append(nav)
    series = pd.Series(values, index=aligned.index)
    series.to_frame('nav').to_csv(out / 'nav_PREMIA_MIX.csv')
    (out / 'low_weight_zero.json').write_text(json.dumps(zero_low, indent=1))
    return series


def _cagr(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def stage_report(out):
    folder = out / 'ledger'
    parts = [json.loads(path.read_text()) for path in sorted(folder.glob('part_*_cost1.json'))]
    frame = pd.DataFrame(parts)
    model = frame[frame.path.str.startswith('LIQ_HIGH_p')]
    rand = frame[frame.path.str.startswith('LIQ_HIGH_RAND_')]
    high_cagr = float(model.cagr.median())
    rand_cagr = float(rand.cagr.median())
    model_nav = pd.read_csv(folder / 'nav_LIQ_HIGH_p0_cost1.csv', index_col=0, parse_dates=True)['nav']
    rand_nav = pd.read_csv(folder / 'nav_LIQ_HIGH_RAND_p0_s1_cost1.csv', index_col=0, parse_dates=True)['nav']
    excess = (model_nav.pct_change() - rand_nav.pct_change()).dropna()
    _, t_stat = newey_west(excess.to_numpy(), lags=HORIZON)
    high_pass = high_cagr - rand_cagr >= 0.02 and float(t_stat[0]) >= 2
    mix = stage_mix(out, use_high=high_pass)
    pool = pd.read_csv(POOL_NAV, index_col=0, parse_dates=True)['nav']
    common = mix.index.intersection(pool.index)
    mix_excess = (mix.reindex(common).pct_change() - pool.reindex(common).pct_change()).dropna()
    _, mix_t = newey_west(mix_excess.to_numpy(), lags=20)
    mix_cagr = _cagr(mix.loc[D.ANCHOR:D.END].dropna())
    gates = {'above_pool50': mix_cagr > POOL_CAGR, 'above_paper': mix_cagr > PAPER_CAGR,
             'nw_t_ge_2': float(mix_t[0]) >= 2}
    passed = all(gates.values())
    report = dict(liq_high_cagr=high_cagr, liq_high_rand_cagr=rand_cagr, liq_high_t=float(t_stat[0]),
                  liq_high_pass=high_pass, mix_cagr=mix_cagr, gates=gates, passed=passed,
                  equity_book='PREMIA_MIX' if passed else 'POOL50',
                  paper_book='PREMIA_MIX' if passed else 'ALLOC60', promotion_approved=False,
                  prereg=PREREG)
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    if not any(row['trial_id'] == 'PREMIA_MIX#outcome1' for row in T.rows()):
        T.record_outcome('P4_premia', 'LIQ_HIGH', 'pass' if high_pass else 'no_improvement',
                         f"cost1 {high_cagr:.2%} vs random {rand_cagr:.2%}; t {float(t_stat[0]):.2f}")
        T.record_outcome('P4_premia', 'LIQ_HIGH_RAND', 'pass' if high_pass else 'no_improvement', 'control for LIQ_HIGH')
        T.record_outcome('P4_premia', 'ASSET_EW', 'no_improvement', 'equal-weight sleeve, no separate gate')
        T.record_outcome('P4_premia', 'PREMIA_MIX', 'pass' if passed else 'no_improvement',
                         f"mix {mix_cagr:.2%}; gates {json.dumps(gates)}")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('build', 'train', 'curve', 'plans', 'ledger', 'report'))
    parser.add_argument('--only', default='')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'build':
        stage_build(OUT)
    elif args.stage == 'curve':
        stage_curve(OUT)
    elif args.stage == 'train':
        stage_train(OUT, smoke=args.smoke)
    elif args.stage == 'plans':
        stage_plans(OUT)
    elif args.stage == 'ledger':
        only = [item for item in args.only.split(',') if item] or None
        stage_ledger(OUT, only)
    else:
        stage_report(OUT)
