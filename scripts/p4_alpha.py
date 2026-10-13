#!/usr/bin/env python3
"""P4 in-pool alpha (docs/prereg/P4_ALPHA.md).

Stages:
  build   sample frame: every 5 sessions anchored at 2018-01-02, M2 pool minus Beijing, 24 features as
          per-day percentiles, H=20 open-to-open percentile label, label-known position
  train   yearly walk-forward (first refit 2018-01-02), LIN and GBDT scores for every sample day >= 2018
  ledger  LIN/GBDT x phases {0,5,10,15} and POOL50 x phases x seeds {1,2} on the 800k share ledger, --cost
  report  prereg gates, written to output/p4_alpha_20261012/report.json
--smoke runs the stages on a short window into output/p4_alpha_smoke/ (mechanics only, not a trial).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

from quantlab import p4_ledger
from quantlab import research_trials as T
from quantlab.alpha import features as F
from quantlab.alpha.labels import HORIZON, forward_open_returns
from quantlab.alpha.models import GBDTEnsemble, LinearICIR, percentile_frame
from quantlab.alpha.portfolio import random_order, schedule
from quantlab.cloud import CloudError, load_snapshot
from quantlab.research_statistics import deflated_sharpe, newey_west
from quantlab.smallcap import universe
from quantlab.smallcap.panel import Panel

PREREG = 'docs/prereg/P4_ALPHA.md'
PANEL = '20261006T152412Z'
SNAPSHOT = 'bundle_20261012'
OUT = ROOT / 'output/p4_alpha_20261012'
SAMPLE_START, ANCHOR, END = '2015-01-05', '2018-01-02', '2026-09-22'
SUB_START = '2022-01-04'
STEP, REBALANCE, PHASES, SEEDS = 5, 20, (0, 5, 10, 15), (1, 2)
FIELDS = ('open', 'high', 'low', 'close', 'adj_close', 'amount', 'turnover', 'float_mktcap', 'tradestatus',
          'limit_up', 'is_st')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_days(sessions, start, anchor, end):
    sessions = pd.DatetimeIndex(sessions)
    position = sessions.get_loc(pd.Timestamp(anchor))
    keep = [(i - position) % STEP == 0 for i in range(len(sessions))]
    days = sessions[np.array(keep)]
    return days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]


def load_inputs(smoke):
    panel = Panel(PANEL)
    fields = {name: panel.field(name) for name in FIELDS}
    try:
        financials = load_snapshot('financials_pit', SNAPSHOT)
        industries = load_snapshot('sw_l1_members', SNAPSHOT)
    except CloudError as exc:
        if not smoke:
            raise SystemExit(json.dumps(dict(status='insufficient_data', reason=str(exc))))
        financials, industries = None, None
    return panel, fields, financials, industries


def stage_build(out, start, end, smoke=False):
    panel, fields, financials, industries = load_inputs(smoke)
    features = universe.rolling_features(panel, with_st=True)
    price = F.PriceVolume(fields)
    fundamentals = F.Fundamentals(financials) if financials is not None else None
    sectors = F.Industries(industries) if industries is not None else None
    label = forward_open_returns(fields)
    sessions = panel.sessions
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    frames = []
    for day in sample_days(sessions, start, ANCHOR, end):
        pool = [c for c in universe.pool_on(day, features, 'adv', band='relative', exclude_st=True, min_history=252)
                if not c.startswith('bj.')]
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
    samples['label'] = samples.groupby('date')['label_raw'].rank(pct=True)
    out.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out / 'samples.parquet', index=False)
    meta = dict(panel=PANEL, snapshot=SNAPSHOT, features=present, rows=len(samples), days=int(samples.date.nunique()),
                label_coverage=float(samples.label.notna().mean()),
                fundamental_coverage={f: float(samples[f].ne(0.5).mean()) for f in F.FUNDAMENTAL_FEATURES if f in samples},
                financials_rows=None if financials is None else len(financials), smoke=smoke)
    (out / 'samples_meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def refit_days(days, start):
    days = pd.DatetimeIndex(sorted(days))
    days = days[days >= pd.Timestamp(start)]
    return [group.iloc[0] for _, group in pd.Series(days, index=days).groupby(days.year)]


def stage_train(out, start, smoke=False):
    samples = pd.read_parquet(out / 'samples.parquet')
    meta = json.loads((out / 'samples_meta.json').read_text())
    present = meta['features']
    sessions = Panel(PANEL).sessions
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    refits = refit_days(samples.date.unique(), start)
    scores, diagnostics = [], {}
    for i, refit in enumerate(refits):
        stop = refits[i + 1] if i + 1 < len(refits) else samples.date.max() + pd.Timedelta(days=1)
        train = samples[(samples.label_known_pos <= position[refit] - HORIZON - 1) & samples.label.notna()]
        test = samples[(samples.date >= refit) & (samples.date < stop)].copy()
        if train.date.nunique() < 20 or test.empty:
            continue
        lin = LinearICIR().fit(train, present)
        gbdt = GBDTEnsemble().fit(train, present, seeds=(1, 2) if smoke else (1, 2, 3, 4, 5))
        test['LIN'], test['GBDT'] = lin.predict(test), gbdt.predict(test)
        scores.append(test[['date', 'code', 'industry', 'label', 'LIN', 'GBDT']])
        diagnostics[str(refit.date())] = dict(train_rows=len(train), train_days=int(train.date.nunique()),
                                              lin_weights=lin.weights.round(4).to_dict(),
                                              gbdt_importance=gbdt.importance().round(1).head(10).to_dict(),
                                              gbdt_rounds=[b.best_iteration for b in gbdt.boosters])
        print(json.dumps(dict(refit=str(refit.date()), train_rows=len(train), test_rows=len(test))), flush=True)
    frame = pd.concat(scores, ignore_index=True)
    frame.to_parquet(out / 'scores.parquet', index=False)
    oos_ic = {m: float(frame.groupby('date').apply(lambda g: g[m].corr(g['label'], method='spearman')).mean())
              for m in ('LIN', 'GBDT')}
    (out / 'train_diagnostics.json').write_text(json.dumps(dict(refits=diagnostics, oos_mean_rank_ic=oos_ic), indent=1))
    print(json.dumps(dict(oos_mean_rank_ic=oos_ic)), flush=True)


def plans(out, start):
    scores = pd.read_parquet(out / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(start)]
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    result = {}
    for phase in PHASES:
        rebalance = days[phase // STEP::REBALANCE // STEP]
        industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
        for model in ('LIN', 'GBDT'):
            order = lambda day, m=model: list(by_day[day].sort_values(m, ascending=False, kind='mergesort').code)
            result[f'{model}_p{phase}'] = schedule(rebalance, order, industries)
        for seed in SEEDS:
            order = lambda day, s=seed: random_order(list(by_day[day].code), s)
            result[f'POOL50_p{phase}_s{seed}'] = schedule(rebalance, order, industries)
    return result


def stage_ledger(out, start, end, cost_mult):
    folder = out / 'ledger'
    folder.mkdir(parents=True, exist_ok=True)
    targets = {name: p4_ledger.to_targets(s) for name, s in plans(out, start).items()}
    codes = sorted({c for rows in targets.values() for r in rows for c in list(r['weights']) + r['sell']})
    dataset = p4_ledger.load_dataset(codes)
    rows = []
    for name, target in targets.items():
        result = p4_ledger.replay(target, start=start, end=end, cost_mult=cost_mult, dataset=dataset)
        s = result['summary']
        result['nav'].to_frame('nav').to_csv(folder / f'nav_{name}_cost{cost_mult:g}.csv')
        rows.append(dict(path=name, cost_mult=cost_mult, cagr=s['cagr'], mdd=s['max_drawdown'], fees=s['fees'],
                         slippage=s['slippage'], fills=s['fills'], checks_passed=s['account_checks_passed']))
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(folder / f'summary_cost{cost_mult:g}.csv', index=False)


def _cagr(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def stage_report(out, sub_start=SUB_START):
    folder = out / 'ledger'
    nav = lambda name, cost: pd.read_csv(folder / f'nav_{name}_cost{cost}.csv', index_col=0, parse_dates=True)['nav']
    s1, s2 = pd.read_csv(folder / 'summary_cost1.csv'), pd.read_csv(folder / 'summary_cost2.csv')
    med = lambda frame, prefix: float(frame[frame.path.str.startswith(prefix + '_')].cagr.median())
    sub = lambda prefix, names: float(np.median([_cagr(nav(n, 1).loc[sub_start:]) for n in names
                                                 if n.startswith(prefix + '_')]))
    names = list(s1.path)
    pool1, pool2, pool_sub = med(s1, 'POOL50'), med(s2, 'POOL50'), sub('POOL50', names)
    control = nav('POOL50_p0_s1', 1).pct_change().dropna()
    verdicts = {}
    for model in ('LIN', 'GBDT'):
        excess = (nav(f'{model}_p0', 1).pct_change().dropna() - control).dropna()
        _, t_stat = newey_west(excess.to_numpy(), lags=20)
        nw_t = float(t_stat[0])
        gates = {'cagr_minus_pool_ge_2pp': med(s1, model) - pool1 >= 0.02,
                 'nw_t_ge_2': nw_t >= 2.0,
                 'cost2_above_pool': med(s2, model) > pool2,
                 'sub_2022_above_pool': sub(model, names) > pool_sub}
        verdicts[model] = dict(cagr_cost1=med(s1, model), cagr_cost2=med(s2, model), cagr_2022=sub(model, names),
                               mdd_cost1=float(s1[s1.path.str.startswith(model + '_')].mdd.median()),
                               nw_t=nw_t, gates=gates, passed=all(gates.values()),
                               dsr_family=deflated_sharpe(excess, trials=2).get('value'),
                               dsr_global=deflated_sharpe(excess, trials=T.count(include_diagnostics=False)).get('value'))
    passed = [m for m in ('LIN', 'GBDT') if verdicts[m]['passed']]
    chosen = None
    if passed == ['LIN', 'GBDT']:
        chosen = 'GBDT' if verdicts['GBDT']['cagr_cost1'] - verdicts['LIN']['cagr_cost1'] >= 0.01 else 'LIN'
    elif passed:
        chosen = passed[0]
    report = dict(pool50=dict(cagr_cost1=pool1, cagr_cost2=pool2, cagr_2022=pool_sub,
                              mdd_cost1=float(s1[s1.path.str.startswith('POOL50_')].mdd.median())),
                  models=verdicts, chosen=chosen, equity_book=chosen or 'POOL50',
                  prereg=PREREG, prereg_sha256=sha256(ROOT / PREREG), promotion_approved=False)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('build', 'train', 'ledger', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'output/p4_alpha_smoke' if args.smoke else OUT
    start, end = ('2017-06-01', '2018-06-29') if args.smoke else (SAMPLE_START, END)
    trade_start, trade_end = ('2018-03-01', '2018-06-29') if args.smoke else (ANCHOR, END)
    if args.stage == 'build':
        stage_build(out, start, end, args.smoke)
    elif args.stage == 'train':
        stage_train(out, '2018-01-02' if not args.smoke else '2018-03-01', args.smoke)
    elif args.stage == 'ledger':
        stage_ledger(out, trade_start, trade_end, args.cost)
    else:
        stage_report(out, '2018-05-02' if args.smoke else SUB_START)
