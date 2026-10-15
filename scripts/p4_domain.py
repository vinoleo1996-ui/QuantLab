#!/usr/bin/env python3
"""P4 domain book (docs/prereg/P4_DOMAIN.md).

Three ADV domains, a separate GBDT in each, fixed slots. The random control uses the same
domains and slots. --smoke checks the mechanics and is not a trial.
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
from quantlab.alpha.models import GBDTEnsemble, percentile_frame
from quantlab.alpha.portfolio import GROSS, random_order, select_domain_book
from quantlab.cloud import CloudError, load_snapshot
from quantlab.research_statistics import deflated_sharpe, newey_west
from quantlab.smallcap import universe
from quantlab.smallcap.panel import Panel

PREREG = 'docs/prereg/P4_DOMAIN.md'
PANEL = '20261006T152412Z'
SNAPSHOT = 'bundle_20261012'
OUT = ROOT / 'output/p4_domain_20261014'
SAMPLE_START, ANCHOR, END = '2015-01-05', '2018-01-02', '2026-09-22'
SUB_START = '2022-01-04'
STEP, REBALANCE, PHASES, SEEDS = 5, 20, (0, 5, 10, 15), (1, 2)
SLOTS = (('liquid', 17), ('mid', 17), ('micro', 16))
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


def ensure_registered():
    source = 'quantlab/smallcap/universe.py domains_on; quantlab/alpha/portfolio.py select_domain_book; scripts/p4_domain.py'
    data = 'v4 panel 20261006T152412Z; bundle_20261012 financials_pit+sw_l1_members; p4_ledger 800k'
    segment = '2018-01-02..2026-09-22; no confirm segment'
    specs = (
        ('DOM_GBDT', 'trial', 'one GBDT per ADV domain, slots 17/17/16'),
        ('DOM_RAND', 'diagnostic', 'same domains and slots, random order inside each domain'),
    )
    for trial_id, role, note in specs:
        try:
            T.register('P4_domain', trial_id, prereg=PREREG, source=source, data=data, segment=segment,
                       role=role, note=note)
        except ValueError as exc:
            if 'trial_already_registered' not in str(exc):
                raise


def stage_build(out, start, end, smoke=False):
    if not smoke:
        ensure_registered()
    panel, fields, financials, industries = load_inputs(smoke)
    features = universe.rolling_features(panel, with_st=True)
    price = F.PriceVolume(fields)
    fundamentals = F.Fundamentals(financials) if financials is not None else None
    sectors = F.Industries(industries) if industries is not None else None
    label = forward_open_returns(fields)
    sessions = panel.sessions
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    frames, counts = [], []
    days = list(sample_days(sessions, start, ANCHOR, end))
    for n, day in enumerate(days, start=1):
        if n == 1 or n % 25 == 0 or n == len(days):
            print(json.dumps(dict(build_day=n, build_days=len(days), date=str(pd.Timestamp(day).date()))), flush=True)
        groups = universe.domains_on(day, features, exclude_st=True, min_history=252)
        counts.append({'date': str(pd.Timestamp(day).date()), **{name: len(codes) for name, codes in groups.items()}})
        for domain, pool in groups.items():
            if not pool:
                continue
            frame = F.cross_section(day, pool, price, fundamentals, sectors, fields['close'])
            frame['domain'] = domain
            frame['label_raw'] = label.loc[day].reindex(pool).to_numpy()
            frame['label_known_pos'] = position[day] + 1 + HORIZON
            frame['industry'] = sectors.on(day).reindex(pool).fillna('unclassified').to_numpy() if sectors else 'unclassified'
            frames.append(frame)
    if not frames:
        raise SystemExit(json.dumps(dict(status='insufficient_data', reason='no_domain_rows')))
    samples = pd.concat(frames, ignore_index=True)
    present = [f for f in F.ALL_FEATURES if f in samples]
    samples = percentile_frame(samples, present, by=['date', 'domain'])
    samples['label'] = samples.groupby(['date', 'domain'])['label_raw'].rank(pct=True)
    out.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out / 'samples.parquet', index=False)
    pd.DataFrame(counts).to_csv(out / 'domain_counts.csv', index=False)
    meta = dict(panel=PANEL, snapshot=SNAPSHOT, features=present, slots=dict(SLOTS),
                rows=len(samples), days=int(samples.date.nunique()),
                rows_by_domain={name: int((samples.domain == name).sum()) for name in universe.DOMAINS},
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
    scores, diagnostics = [], {}
    for domain in universe.DOMAINS:
        block = samples[samples.domain == domain]
        refits = refit_days(block.date.unique(), start)
        for i, refit in enumerate(refits):
            stop = refits[i + 1] if i + 1 < len(refits) else block.date.max() + pd.Timedelta(days=1)
            train = block[(block.label_known_pos <= position[refit] - HORIZON - 1) & block.label.notna()]
            test = block[(block.date >= refit) & (block.date < stop)].copy()
            if train.date.nunique() < 20 or test.empty:
                print(json.dumps(dict(domain=domain, refit=str(pd.Timestamp(refit).date()), skipped=True,
                                       train_days=int(train.date.nunique()))), flush=True)
                continue
            gbdt = GBDTEnsemble().fit(train, present, seeds=(1, 2) if smoke else (1, 2, 3, 4, 5))
            test['GBDT'] = gbdt.predict(test)
            scores.append(test[['date', 'code', 'domain', 'industry', 'label', 'GBDT']])
            diagnostics[f'{pd.Timestamp(refit).date()}/{domain}'] = dict(
                train_rows=len(train), train_days=int(train.date.nunique()),
                gbdt_importance=gbdt.importance().round(1).head(8).to_dict(),
                gbdt_rounds=[b.best_iteration for b in gbdt.boosters])
            print(json.dumps(dict(domain=domain, refit=str(pd.Timestamp(refit).date()),
                                   train_rows=len(train), test_rows=len(test))), flush=True)
    if not scores:
        raise SystemExit(json.dumps(dict(status='insufficient_data', reason='no_scores')))
    frame = pd.concat(scores, ignore_index=True)
    frame.to_parquet(out / 'scores.parquet', index=False)
    by_domain = {}
    for domain in universe.DOMAINS:
        block = frame[frame.domain == domain]
        ic = block.groupby('date').apply(lambda g: g['GBDT'].corr(g['label'], method='spearman') if len(g) > 5 else np.nan)
        by_domain[domain] = float(ic.mean())
    oos_mean = float(np.nanmean(list(by_domain.values())))
    (out / 'train_diagnostics.json').write_text(json.dumps(
        dict(refits=diagnostics, oos_mean_rank_ic=oos_mean, oos_rank_ic_by_domain=by_domain), indent=1))
    print(json.dumps(dict(oos_mean_rank_ic=oos_mean, by_domain=by_domain)), flush=True)


def _book(rebalance, by_day, order_of):
    current, out = [], {}
    for day in rebalance:
        group = by_day[day]
        industries = dict(zip(group.code, group.industry))
        orders = {domain: order_of(group, domain) for domain, _ in SLOTS}
        current = select_domain_book(orders, industries, current, SLOTS)
        out[day] = {code: GROSS / len(current) for code in current} if current else {}
    return out


def plans(out, start):
    scores = pd.read_parquet(out / 'scores.parquet')
    scores = scores[scores.date >= pd.Timestamp(start)]
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    result = {}
    for phase in PHASES:
        rebalance = days[phase // STEP::REBALANCE // STEP]

        def gbdt_order(group, domain):
            block = group[group.domain == domain]
            return list(block.sort_values('GBDT', ascending=False, kind='mergesort').code)

        result[f'DOM_GBDT_p{phase}'] = _book(rebalance, by_day, gbdt_order)
        for seed in SEEDS:
            def rand_order(group, domain, s=seed):
                return random_order(list(group.loc[group.domain == domain, 'code']), s)
            result[f'DOM_RAND_p{phase}_s{seed}'] = _book(rebalance, by_day, rand_order)
    return result


def stage_ledger(out, start, end, cost_mult):
    folder = out / 'ledger'
    folder.mkdir(parents=True, exist_ok=True)
    built = plans(out, start)
    mix = {}
    scores = pd.read_parquet(out / 'scores.parquet')
    domain_of = scores.groupby(['date', 'code'])['domain'].first()
    for name, schedule in built.items():
        held = {domain: 0 for domain in universe.DOMAINS}
        days = 0
        for day, weights in schedule.items():
            days += 1
            for code in weights:
                domain = domain_of.get((pd.Timestamp(day), code))
                if domain in held:
                    held[domain] += 1
        mix[name] = {domain: held[domain] / days if days else 0.0 for domain in held}
    (out / 'holdings_mix.json').write_text(json.dumps(mix, indent=1))
    targets = {name: p4_ledger.to_targets(schedule) for name, schedule in built.items()}
    codes = sorted({c for rows in targets.values() for row in rows for c in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    rows = []
    for name, target in targets.items():
        result = p4_ledger.replay(target, start=start, end=end, cost_mult=cost_mult, dataset=dataset)
        summary = result['summary']
        result['nav'].to_frame('nav').to_csv(folder / f'nav_{name}_cost{cost_mult:g}.csv')
        rows.append(dict(path=name, cost_mult=cost_mult, cagr=summary['cagr'], mdd=summary['max_drawdown'],
                         fees=summary['fees'], slippage=summary['slippage'], fills=summary['fills'],
                         checks_passed=summary['account_checks_passed']))
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(folder / f'summary_cost{cost_mult:g}.csv', index=False)


def _cagr(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def stage_report(out, sub_start=SUB_START, smoke=False):
    folder = out / 'ledger'
    nav = lambda name, cost: pd.read_csv(folder / f'nav_{name}_cost{cost}.csv', index_col=0, parse_dates=True)['nav']
    s1, s2 = pd.read_csv(folder / 'summary_cost1.csv'), pd.read_csv(folder / 'summary_cost2.csv')
    med = lambda frame, prefix: float(frame[frame.path.str.startswith(prefix + '_')].cagr.median())
    names = list(s1.path)

    def sub(prefix):
        series = [_cagr(nav(name, 1).loc[sub_start:]) for name in names if name.startswith(prefix + '_')]
        return float(np.median(series))

    rand1, rand2, rand_sub = med(s1, 'DOM_RAND'), med(s2, 'DOM_RAND'), sub('DOM_RAND')
    excess = (nav('DOM_GBDT_p0', 1).pct_change().dropna() - nav('DOM_RAND_p0_s1', 1).pct_change().dropna()).dropna()
    _, t_stat = newey_west(excess.to_numpy(), lags=20)
    nw_t = float(t_stat[0])
    model1, model2, model_sub = med(s1, 'DOM_GBDT'), med(s2, 'DOM_GBDT'), sub('DOM_GBDT')
    gates = {'cagr_minus_random_ge_2pp': model1 - rand1 >= 0.02,
             'nw_t_ge_2': nw_t >= 2.0,
             'cost2_above_random': model2 > rand2,
             'sub_2022_above_random': model_sub > rand_sub}
    passed = all(gates.values())
    report = dict(
        dom_gbdt=dict(cagr_cost1=model1, cagr_cost2=model2, cagr_2022=model_sub,
                      mdd_cost1=float(s1[s1.path.str.startswith('DOM_GBDT_')].mdd.median()), nw_t=nw_t),
        dom_rand=dict(cagr_cost1=rand1, cagr_cost2=rand2, cagr_2022=rand_sub,
                      mdd_cost1=float(s1[s1.path.str.startswith('DOM_RAND_')].mdd.median())),
        published_pool50_reference=dict(cagr_cost1=0.1211, mdd_cost1=-0.330, source='output/p4_alpha_20261012/report.json'),
        gates=gates, passed=passed, equity_book='DOM_GBDT' if passed else 'POOL50',
        dsr_family=deflated_sharpe(excess, trials=1).get('value'),
        dsr_global=deflated_sharpe(excess, trials=T.count(include_diagnostics=False)).get('value'),
        prereg=PREREG, prereg_sha256=sha256(ROOT / PREREG), promotion_approved=False, smoke=smoke)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    if not smoke:
        note = (f"cost1 {model1:.2%} vs DOM_RAND {rand1:.2%}; cost2 {model2:.2%} vs {rand2:.2%}; "
                f"2022+ {model_sub:.2%} vs {rand_sub:.2%}; NW t {nw_t:.2f}; "
                f"mdd {report['dom_gbdt']['mdd_cost1']:.1%}")
        if not any(r['trial_id'] == 'DOM_GBDT#outcome1' for r in T.rows()):
            T.record_outcome('P4_domain', 'DOM_GBDT', 'pass' if passed else 'no_improvement', note)
            T.record_outcome('P4_domain', 'DOM_RAND', 'pass' if passed else 'no_improvement', 'control for DOM_GBDT')
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('build', 'train', 'ledger', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'output/p4_domain_smoke' if args.smoke else OUT
    start, end = ('2017-06-01', '2018-06-29') if args.smoke else (SAMPLE_START, END)
    trade_start, trade_end = ('2018-03-01', '2018-06-29') if args.smoke else (ANCHOR, END)
    if args.stage == 'build':
        stage_build(out, start, end, args.smoke)
    elif args.stage == 'train':
        stage_train(out, '2018-03-01' if args.smoke else ANCHOR, args.smoke)
    elif args.stage == 'ledger':
        stage_ledger(out, trade_start, trade_end, args.cost)
    else:
        stage_report(out, '2018-05-02' if args.smoke else SUB_START, args.smoke)
