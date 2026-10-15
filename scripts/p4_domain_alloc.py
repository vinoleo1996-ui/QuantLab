#!/usr/bin/env python3
"""P4 cross-domain book (docs/prereg/P4_DOMAIN_ALLOC.md).

Three already-trained domain scores are mapped to lagged decile returns and ranked on one list.
A domain may contribute zero names. Entry and exit depths follow the low-turnover rule, measured
on that day's micro-domain headcount. --only replays named paths so several processes can share
the work; it does not change the book.
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
from quantlab.alpha.expected import assign_decile, daily_decile_totals, lagged_decile_means
from quantlab.alpha.labels import HORIZON
from quantlab.alpha.portfolio import GROSS, random_order, select_hysteresis
from quantlab.research_statistics import deflated_sharpe, newey_west
from quantlab.smallcap import universe

PREREG = 'docs/prereg/P4_DOMAIN_ALLOC.md'
OUT = ROOT / 'output/p4_domain_alloc_20261015'
BUNDLE = ROOT / 'replay_data/p4_domain_alloc'
ALPHA = ROOT / 'output/p4_alpha_20261012'
HYST = ROOT / 'output/p4_lowturn_20261013/report.json'
ENTRY_FRAC, EXIT_FRAC = 0.10, 0.50
MICRO_FULL = 40


def sha256(path):
    return T.sha256_file(path)


def ensure_registered():
    source = ('quantlab/alpha/expected.py lagged decile; quantlab/alpha/portfolio.py select_hysteresis counts; '
              'scripts/p4_domain_alloc.py')
    data = 'p4_domain_20261014 scores.parquet + samples label_raw; p4_ledger 800k; no retrain'
    segment = '2018-01-02..2026-09-22; no confirm segment'
    specs = (
        ('XDOM', 'trial', 'cross-domain lagged decile return, hysteresis depth = micro headcount'),
        ('XDOM_RAND', 'diagnostic', 'same eligible names and depths, random order'),
    )
    for trial_id, role, note in specs:
        try:
            T.register('P4_domain_alloc', trial_id, prereg=PREREG, source=source, data=data, segment=segment,
                       role=role, note=note)
        except ValueError as exc:
            if 'trial_already_registered' not in str(exc):
                raise


def load_scored():
    scores = pd.read_parquet(D.OUT / 'scores.parquet', columns=['date', 'code', 'domain', 'industry', 'GBDT'])
    raw = pd.read_parquet(D.OUT / 'samples.parquet', columns=['date', 'code', 'domain', 'label_raw', 'label_known_pos'])
    frame = scores.merge(raw, on=['date', 'code', 'domain'], how='inner')
    frame = frame.dropna(subset=['GBDT']).copy()
    frame['date'] = pd.to_datetime(frame['date'])
    frame = frame[frame.date >= pd.Timestamp(D.ANCHOR)]
    frame = assign_decile(frame)
    return frame


def band_counts(n_micro):
    return int(n_micro * ENTRY_FRAC), int(n_micro * EXIT_FRAC)


def _eligible(group, means):
    keys = pd.MultiIndex.from_arrays([group.domain.to_numpy(), group.decile.to_numpy()])
    expected = means.reindex(keys).to_numpy()
    keep = group.loc[np.isfinite(expected)].copy()
    keep['expected'] = expected[np.isfinite(expected)]
    return keep


def _book(rebalance, by_day, means_on, order_of):
    current, out, counts = [], {}, {name: 0 for name in universe.DOMAINS}
    days = 0
    for day in rebalance:
        group = by_day[day]
        eligible = _eligible(group, means_on(day))
        n_micro = int((group.domain == 'micro').sum())
        entry_n, exit_n = band_counts(n_micro)
        industries = dict(zip(group.code, group.industry))
        ordered = order_of(eligible)
        current = select_hysteresis(ordered, industries, current, entry_count=entry_n, exit_count=exit_n)
        out[day] = {code: GROSS / len(current) for code in current} if current else {}
        domain_of = dict(zip(group.code, group.domain))
        days += 1
        for code in current:
            domain = domain_of.get(code)
            if domain in counts:
                counts[domain] += 1
    mix = {domain: counts[domain] / days if days else 0.0 for domain in counts}
    return out, mix


def plans(frame=None):
    frame = load_scored() if frame is None else frame
    daily = daily_decile_totals(frame)
    known = frame.groupby('date')['label_known_pos'].first()
    by_day = {day: group for day, group in frame.groupby('date', sort=True)}
    days = sorted(by_day)

    def means_on(day):
        return lagged_decile_means(daily, int(known.loc[day]) - HORIZON - 1)

    def model_order(eligible):
        return list(eligible.sort_values(['expected', 'code'], ascending=[False, True], kind='mergesort').code)

    result, mix = {}, {}
    for phase in D.PHASES:
        rebalance = days[phase // D.STEP::D.REBALANCE // D.STEP]
        result[f'XDOM_p{phase}'], mix[f'XDOM_p{phase}'] = _book(rebalance, by_day, means_on, model_order)
        for seed in D.SEEDS:
            def rand_order(eligible, s=seed):
                return random_order(list(eligible.code), s)

            name = f'XDOM_RAND_p{phase}_s{seed}'
            result[name], mix[name] = _book(rebalance, by_day, means_on, rand_order)
    return result, mix


def stage_plans():
    """Write the books once. Replay processes read this file and do not rebuild the mapping."""
    OUT.mkdir(parents=True, exist_ok=True)
    built, mix = plans()
    payload = {name: {str(pd.Timestamp(day).date()): weights for day, weights in schedule.items()}
               for name, schedule in built.items()}
    (OUT / 'plans.json').write_text(json.dumps(payload))
    (OUT / 'holdings_mix.json').write_text(json.dumps(mix, indent=1))
    summary = {name: {domain: round(value, 2) for domain, value in counts.items()} for name, counts in mix.items()}
    print(json.dumps(dict(paths=len(payload), mix=summary), indent=1), flush=True)


def load_plans():
    path = BUNDLE / 'plans.json'
    if not path.exists():
        path = OUT / 'plans.json'
    payload = json.loads(path.read_text())
    return {name: {pd.Timestamp(day): weights for day, weights in schedule.items()} for name, schedule in payload.items()}


def stage_ledger(cost_mult, only=None):
    ensure_registered()
    folder = OUT / 'ledger'
    folder.mkdir(parents=True, exist_ok=True)
    built = load_plans() if (OUT / 'plans.json').exists() else plans()[0]
    if only:
        missing = [name for name in only if name not in built]
        if missing:
            raise SystemExit(json.dumps(dict(status='unknown_path', missing=missing)))
        built = {name: built[name] for name in only}
    targets = {name: p4_ledger.to_targets(schedule) for name, schedule in built.items()}
    codes = sorted({c for rows in targets.values() for row in rows for c in list(row['weights']) + row['sell']})
    manifest = BUNDLE / 'dataset/manifest.json'
    dataset = p4_ledger.load_dataset(codes, manifest_path=manifest if manifest.exists() else None) if codes else None
    rows = []
    for name, target in targets.items():
        if not target:
            row = dict(path=name, cost_mult=cost_mult, cagr=0.0, mdd=0.0, fees=0.0, slippage=0.0, fills=0,
                       checks_passed=False, empty=True)
        else:
            result = p4_ledger.replay(target, start=D.ANCHOR, end=D.END, cost_mult=cost_mult, dataset=dataset)
            summary = result['summary']
            result['nav'].to_frame('nav').to_csv(folder / f'nav_{name}_cost{cost_mult:g}.csv')
            row = dict(path=name, cost_mult=cost_mult, cagr=summary['cagr'], mdd=summary['max_drawdown'],
                       fees=summary['fees'], slippage=summary['slippage'], fills=summary['fills'],
                       checks_passed=summary['account_checks_passed'])
        (folder / f'part_{name}_cost{cost_mult:g}.json').write_text(json.dumps(row))
        rows.append(row)
        print(json.dumps(row), flush=True)
    if only is None:
        pd.DataFrame(rows).to_csv(folder / f'summary_cost{cost_mult:g}.csv', index=False)


def _summary(folder, cost):
    path = folder / f'summary_cost{cost:g}.csv'
    if path.exists():
        return pd.read_csv(path)
    parts = sorted(folder.glob(f'part_*_cost{cost:g}.json'))
    if not parts:
        raise SystemExit(json.dumps(dict(status='missing_ledger', cost=cost)))
    frame = pd.DataFrame(json.loads(part.read_text()) for part in parts)
    frame.to_csv(path, index=False)
    return frame


def _cagr(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def stage_report():
    folder = OUT / 'ledger'
    nav = lambda base, name, cost: pd.read_csv(base / f'nav_{name}_cost{cost}.csv', index_col=0, parse_dates=True)['nav']
    mine1, mine2 = _summary(folder, 1), _summary(folder, 2)
    expected = [f'XDOM_p{phase}' for phase in D.PHASES]
    expected += [f'XDOM_RAND_p{phase}_s{seed}' for phase in D.PHASES for seed in D.SEEDS]
    for cost, frame in ((1, mine1), (2, mine2)):
        missing = [name for name in expected if name not in set(frame.path)]
        if missing:
            raise SystemExit(json.dumps(dict(status='incomplete_ledger', cost=cost, missing=missing)))
    pool1 = pd.read_csv(ALPHA / 'ledger/summary_cost1.csv')
    pool2 = pd.read_csv(ALPHA / 'ledger/summary_cost2.csv')
    pool1, pool2 = pool1[pool1.path.str.startswith('POOL50_')], pool2[pool2.path.str.startswith('POOL50_')]
    model1 = mine1[mine1.path.str.startswith('XDOM_p')]
    model2 = mine2[mine2.path.str.startswith('XDOM_p')]
    rand1 = mine1[mine1.path.str.startswith('XDOM_RAND_')]

    def sub(base, frame, cost=1):
        series = [_cagr(nav(base, name, cost).loc[D.SUB_START:]) for name in frame.path]
        return float(np.median(series))

    control = nav(ALPHA / 'ledger', 'POOL50_p0_s1', 1).pct_change().dropna()
    excess = (nav(folder, 'XDOM_p0', 1).pct_change().dropna() - control).dropna()
    _, t_stat = newey_west(excess.to_numpy(), lags=20)
    nw_t = float(t_stat[0])
    mix_path = BUNDLE / 'holdings_mix.json'
    if not mix_path.exists():
        mix_path = OUT / 'holdings_mix.json'
    mix = json.loads(mix_path.read_text())
    micro = {name: mix[name]['micro'] for name in model1.path}
    collapsed = bool(micro) and all(count >= MICRO_FULL for count in micro.values()) and float(model1.cagr.median()) <= float(pool1.cagr.median())
    hyst = json.loads(HYST.read_text()) if HYST.exists() else {}
    gates = {'cagr_minus_pool_ge_2pp': float(model1.cagr.median()) - float(pool1.cagr.median()) >= 0.02,
             'nw_t_ge_2': nw_t >= 2.0,
             'cost2_above_pool': float(model2.cagr.median()) > float(pool2.cagr.median()),
             'sub_2022_above_pool': sub(folder, model1) > sub(ALPHA / 'ledger', pool1)}
    def flags_ok(frame):
        return bool(frame.checks_passed.map(lambda value: str(value).lower() == 'true').all())

    checks_ok = flags_ok(mine1) and flags_ok(mine2)
    passed = checks_ok and all(gates.values())
    report = dict(
        xdom=dict(cagr_cost1=float(model1.cagr.median()), cagr_cost2=float(model2.cagr.median()),
                  cagr_2022=sub(folder, model1), mdd_cost1=float(model1.mdd.median()), nw_t=nw_t,
                  by_phase=dict(zip(model1.path, model1.cagr.round(6))),
                  mean_names_by_domain={name: mix[name] for name in list(model1.path) + list(rand1.path)}),
        xdom_rand=dict(cagr_cost1=float(rand1.cagr.median()), cagr_cost2=float(mine2[mine2.path.str.startswith('XDOM_RAND_')].cagr.median())),
        pool50=dict(cagr_cost1=float(pool1.cagr.median()), cagr_cost2=float(pool2.cagr.median()),
                    cagr_2022=sub(ALPHA / 'ledger', pool1), mdd_cost1=float(pool1.mdd.median())),
        gbdt_hyst_reference=hyst.get('gbdt_hyst', {}).get('cagr_cost1', 0.1438),
        collapsed_to_micro=collapsed, checks_passed=checks_ok, gates=gates, passed=passed,
        equity_book='XDOM' if passed else 'POOL50',
        dsr_family=deflated_sharpe(excess, trials=1).get('value'),
        dsr_global=deflated_sharpe(excess, trials=T.count(include_diagnostics=False)).get('value'),
        prereg=PREREG, prereg_sha256=sha256(ROOT / PREREG), promotion_approved=False)
    text = json.dumps(report, indent=1, default=float)
    (OUT / 'report.json').write_text(text)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    (BUNDLE / 'report.json').write_text(text)
    for cost in (1, 2):
        summary = folder / f'summary_cost{cost}.csv'
        if summary.exists():
            (BUNDLE / summary.name).write_text(summary.read_text())
    if not any(r['trial_id'] == 'XDOM#outcome1' for r in T.rows()):
        note = (f"cost1 {report['xdom']['cagr_cost1']:.2%} vs POOL50 {report['pool50']['cagr_cost1']:.2%}; "
                f"cost2 {report['xdom']['cagr_cost2']:.2%} vs {report['pool50']['cagr_cost2']:.2%}; "
                f"2022+ {report['xdom']['cagr_2022']:.2%} vs {report['pool50']['cagr_2022']:.2%}; "
                f"NW t {nw_t:.2f}; collapsed_to_micro {collapsed}")
        outcome = 'pass' if passed else ('technical_failure' if not checks_ok else 'no_improvement')
        T.record_outcome('P4_domain_alloc', 'XDOM', outcome, note)
        T.record_outcome('P4_domain_alloc', 'XDOM_RAND', outcome, 'control for XDOM')
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('plans', 'ledger', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    parser.add_argument('--only', default='', help='comma-separated path names; omit to run every path')
    args = parser.parse_args()
    only = [item for item in args.only.split(',') if item] or None
    if args.stage == 'plans':
        stage_plans()
    elif args.stage == 'ledger':
        stage_ledger(args.cost, only)
    else:
        stage_report()
