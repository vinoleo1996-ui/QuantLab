#!/usr/bin/env python3
"""P2 large/mid-cap quality + dividend, reference only (docs/prereg/P2_QUALITY_DIVIDEND.md).

  python scripts/p2_quality_dividend.py select
  python scripts/p2_quality_dividend.py ledger --cost 1
  python scripts/p2_quality_dividend.py report

Writes output/p2_quality_dividend_20261008/.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import pandas as pd

from quantlab import research_statistics as RS
from quantlab import research_trials as T
from quantlab.broad import month_ends

PREREG = 'docs/prereg/P2_QUALITY_DIVIDEND.md'
OUT = ROOT / 'output/p2_quality_dividend_20261008'
START, END = '2018-01-02', '2026-09-22'
PHASES = {0: (1, 4, 7, 10), 1: (2, 5, 8, 11), 2: (3, 6, 9, 12)}
TOP, WEIGHT = 50, 0.018


def annual(financials):
    frame = financials[pd.to_datetime(financials.report_period).dt.month == 12].copy()
    frame['report_period'] = pd.to_datetime(frame.report_period)
    frame['available_date'] = pd.to_datetime(frame.available_date)
    return frame.pivot_table(index=['instrument_id', 'report_period', 'available_date'], columns='metric',
                             values='value', aggfunc='last').reset_index()


def factors(year_rows, day, codes):
    known = year_rows[year_rows.available_date <= day].sort_values('report_period')
    known = known[known.instrument_id.isin(codes)].groupby('instrument_id').tail(3)
    grouped = known.groupby('instrument_id')
    out = pd.DataFrame({'reports': grouped.size(), 'roe_mean': grouped.roe.mean(), 'roe_std': grouped.roe.std(),
                        'ocf': grouped.ocf_to_profit.last().clip(-5, 5), 'net_profit': grouped.net_profit.last()})
    return out[(out.reports >= 2) & (out.net_profit > 0)]


def select():
    import p2_event_flow as EF
    from neutral_alpha import sector_letters
    data = EF.Data()
    fin = pd.read_parquet(ROOT / f'data/smallcap_v4/{EF.VERSION}/financials.parquet')
    years = annual(fin)
    dy = pd.read_parquet(ROOT / 'data/broad/dividend_yield.parquet')
    days = month_ends(data.dates, START, END)
    days = days.append(month_ends(data.dates, '2017-10-01', '2017-12-31'))
    schedules = {phase: {} for phase in PHASES}
    rows = []
    for day in sorted(days):
        i = data.dates.get_loc(day)
        mask = data.eligible(i)
        codes = data.codes[mask]
        table = factors(years, day, set(codes))
        table['dividend'] = dy[dy.date == day].set_index('code').dividend_yield.reindex(table.index).fillna(0.0)
        table['industry'] = sector_letters(data.industries, day, table.index.to_numpy())
        z = lambda s: (s - s.mean()) / s.std()
        parts = pd.DataFrame({'roe': z(table.roe_mean), 'stable': z(-table.roe_std.fillna(table.roe_std.median())),
                              'ocf': z(table.ocf), 'dividend': z(table.dividend)})
        parts = parts - parts.groupby(table.industry).transform('mean')
        score = parts.mean(axis=1).dropna().sort_values(ascending=False)
        chosen = score.index[:TOP].tolist()
        pct = pd.Series(np.log(data.cap[i, mask]), index=codes).rank(pct=True)
        rows.append({'date': day, 'candidates': len(score), 'size_pct_median': float(pct.reindex(chosen).median())})
        for phase, months in PHASES.items():
            if day.month in months:
                schedules[phase][day] = {c: WEIGHT for c in chosen}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'selection.csv', index=False)
    for phase, schedule in schedules.items():
        before = [d for d in schedule if d < pd.Timestamp(START)]
        if before:
            opening = schedule[max(before)]
            schedule = {d: w for d, w in schedule.items() if d >= pd.Timestamp(START)}
            schedule[pd.Timestamp(START)] = opening
        targets = to_targets(schedule)
        (OUT / f'targets_phase{phase}.json').write_text(json.dumps({'targets': targets}, indent=1))
    print(json.dumps({'months': len(rows), 'phases': {p: len(s) for p, s in schedules.items()}}))


def to_targets(schedule):
    rows, ever = [], set()
    for day in sorted(schedule):
        weights = {ledger_code(c): w for c, w in schedule[day].items()}
        rows.append({'signal_date': str(pd.Timestamp(day).date()), 'weights': weights, 'active': list(weights),
                     'sell': sorted(ever - set(weights)), 'sleeve': 0})
        ever |= set(weights)
    return rows


def ledger_code(code):
    market, number = code.split('.')
    return f'{number}.{market.upper()}'


def ledger(cost_mult):
    import stable_book as book
    from quantlab.data.contracts import MarketDataset
    plans = {p: json.loads((OUT / f'targets_phase{p}.json').read_text())['targets'] for p in PHASES}
    root = book.MANIFEST.parent
    manifest = json.loads(book.MANIFEST.read_text())
    codes = sorted({c for rows in plans.values() for r in rows for c in list(r['weights']) + r['sell']})
    read = lambda entry: pd.read_parquet(io.BytesIO(book._read_checked(root, entry)))
    calendar = pd.DatetimeIndex(book._dated(read(manifest['calendar'])).session)
    panels = {book.canon(c): book._dated(read(manifest['panels'][book.canon(c)]))
              for c in codes if book.canon(c) in manifest['panels']}
    dataset = MarketDataset(panels, read(manifest['instruments']), calendar, read(manifest['actions']),
                            book._dated(read(manifest['benchmark'])), manifest=manifest)
    rows = []
    for phase, targets in plans.items():
        result = book.replay_account(dataset, targets, start=START, end=END, cost_mult=cost_mult)
        s = result['summary']
        result['nav'].to_frame('nav').to_csv(OUT / f'nav_phase{phase}_cost{cost_mult:g}.csv')
        rows.append({'phase': phase, 'cost_mult': cost_mult, 'cagr': s['cagr'], 'max_drawdown': s['max_drawdown'],
                     'fills': s['fills'], 'checks_passed': s['account_checks_passed'],
                     'missing_panels': len(s['missing_panels'])})
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'ledger_cost{cost_mult:g}.csv', index=False)


def report():
    import p2_event_flow as EF
    import smallcap_audit as AU
    data = EF.Data()
    first = data.dates.searchsorted(pd.Timestamp(START))
    last = data.dates.searchsorted(pd.Timestamp(END), side='right') - 1
    factor_rows = []
    for t in range(first, last + 1):
        mkt, smb, hml, live = EF.daily_factors(data, t)
        factor_rows.append({'date': data.dates[t], 'MKT': mkt, 'SMB': smb, 'HML': hml})
    fac = pd.DataFrame(factor_rows).set_index('date')
    bench = AU.bench_series('H20269').loc[START:END]
    years_b = (bench.index[-1] - bench.index[0]).days / 365.25
    bench_cagr = float((bench.iloc[-1] / bench.iloc[0]) ** (1 / years_b) - 1)
    m2 = pd.read_csv(ROOT / 'output/p1_m2_20261008/is/ledger_nav_0_1_cost1.csv', index_col=0, parse_dates=True)['nav']
    phases = {}
    for phase in PHASES:
        nav = pd.read_csv(OUT / f'nav_phase{phase}_cost1.csv', index_col=0, parse_dates=True)['nav']
        ret = nav.pct_change().dropna()
        excess = (ret - fac.MKT.reindex(ret.index)).dropna()
        years = excess.groupby(excess.index.year).sum()
        frame = pd.concat([ret.rename('y'), fac], axis=1).dropna()
        beta, t = RS.newey_west(frame.y.to_numpy(), frame[['MKT', 'SMB', 'HML']].to_numpy(), lags=20)
        span = (nav.index[-1] - nav.index[0]).days / 365.25
        phases[phase] = {'cagr': float((nav.iloc[-1] / nav.iloc[0]) ** (1 / span) - 1),
                         'max_drawdown': float((nav / nav.cummax() - 1).min()),
                         'excess_annual': float(excess.mean() * 252), 'positive_years': int((years > 0).sum()),
                         'years': {int(k): float(v) for k, v in years.items()},
                         'ff3_alpha_annual': float(beta[0] * 252), 'ff3_alpha_t': float(t[0]),
                         'betas': dict(zip(('MKT', 'SMB', 'HML'), map(float, beta[1:]))),
                         'corr_with_m2': float(pd.concat([ret, m2.pct_change()], axis=1).dropna().corr().iloc[0, 1])}
    med = lambda key: float(np.median([p[key] for p in phases.values()]))
    gates = {'cagr>H20269': med('cagr') > bench_cagr, 'excess_annual>0': med('excess_annual') > 0,
             'positive_years>=6': med('positive_years') >= 6, 'ff3_alpha_t>=2': med('ff3_alpha_t') >= 2}
    gates['reference_pass'] = all(gates.values())
    cost2 = pd.read_csv(OUT / 'ledger_cost2.csv')
    selection = pd.read_csv(OUT / 'selection.csv')
    out = {'status': 'reference_only_financials_unverified', 'phases': phases, 'h20269_cagr': bench_cagr,
           'median': {k: med(k) for k in ('cagr', 'max_drawdown', 'excess_annual', 'ff3_alpha_t')},
           'cost2_median_cagr': float(cost2.cagr.median()), 'size_pct_median': float(selection.size_pct_median.median()),
           'gates': gates, 'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG), 'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('median', 'h20269_cagr', 'cost2_median_cagr', 'gates')}, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('select', 'ledger', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    {'select': select, 'report': report}.get(args.stage, lambda: ledger(args.cost))()
