#!/usr/bin/env python3
"""Shenwan level-1 industry rotation, momentum + turnover share (docs/prereg/P3_INDUSTRY_ROTATION.md).

  python scripts/p3_industry_rotation.py index
  python scripts/p3_industry_rotation.py select
  python scripts/p3_industry_rotation.py ledger --cost 1
  python scripts/p3_industry_rotation.py report

Writes output/p3_industry_rotation_20261008/.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import pandas as pd

from quantlab import research_trials as T
from quantlab.broad import eligible, month_ends
from quantlab.smallcap.panel import Panel

PREREG = 'docs/prereg/P3_INDUSTRY_ROTATION.md'
OUT = ROOT / 'output/p3_industry_rotation_20261008'
VERSION = '20261006T152412Z'
DISCOVERY, CONFIRM = ('2018-01-02', '2026-09-22'), ('2015-01-05', '2017-12-29')
TOP_IND, PER_IND, WEIGHT, MIN_HIST, PHASE_SHIFT = 4, 12, 0.018, 252, 10
INDEX_COST = 0.0015


def sw():
    bars = pd.read_parquet(ROOT / 'data/sw/level1_bars.parquet')
    close = bars.pivot(index='date', columns='swindexcode', values='closeindex').sort_index()
    amount = bars.pivot(index='date', columns='swindexcode', values='bargainsum').sort_index()
    return close, amount


def signal_days(sessions, start, end, phase):
    ends = month_ends(sessions, start, end)
    if phase == 0:
        return list(ends)
    out = []
    for day in ends:
        i = sessions.get_loc(day) + PHASE_SHIFT
        if i < len(sessions) and sessions[i] <= pd.Timestamp(end):
            out.append(sessions[i])
    return out


def scores(close, amount, day):
    i = close.index.get_loc(day)
    history = close.iloc[:i + 1].notna().sum()
    ok = history[history >= MIN_HIST].index
    if i < 250:
        return pd.Series(dtype=float)
    mom = close.iloc[i - 21][ok] / close.iloc[i - 147][ok] - 1
    a = amount[ok].fillna(0.0)
    share20 = a.iloc[i - 19:i + 1].sum() / a.iloc[i - 19:i + 1].sum().sum()
    share250 = a.iloc[i - 249:i + 1].sum() / a.iloc[i - 249:i + 1].sum().sum()
    flow = share20 / share250
    return pd.concat([mom.rank(pct=True), flow.rank(pct=True)], axis=1).mean(axis=1).dropna()


def chosen_industries(close, amount, days):
    return {day: scores(close, amount, day).nlargest(TOP_IND).index.tolist() for day in days}


def index_backtest(close, picks, start, end):
    """Equal-weight held industries from the session after each signal; cost on weight changes."""
    ret = close.pct_change().loc[start:end]
    schedule = {}
    for day, inds in picks.items():
        j = ret.index.searchsorted(day, side='right')
        if j < len(ret.index):
            schedule[ret.index[j]] = inds
    nav, value, weights = [], 1.0, pd.Series(dtype=float)
    for day, row in ret.iterrows():
        if day in schedule:
            new = pd.Series(1 / len(schedule[day]), index=schedule[day])
            value *= 1 - (new.sub(weights, fill_value=0).abs().sum()) * INDEX_COST
            weights = new
        if len(weights):
            r = row.reindex(weights.index).fillna(0.0)
            value *= 1 + float((weights * r).sum())
            weights = weights * (1 + r) / (1 + float((weights * r).sum()))
        nav.append(value)
    return pd.Series(nav, index=ret.index)


def stats(nav):
    nav = nav.dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {'cagr': float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
            'max_drawdown': float((nav / nav.cummax() - 1).min())}


def index_stage():
    close, amount = sw()
    out = {}
    for label, (start, end) in (('discovery', DISCOVERY), ('confirm', CONFIRM)):
        for phase in (0, 1):
            days = signal_days(close.index, pd.Timestamp(start) - pd.Timedelta(days=45), end, phase)
            picks = chosen_industries(close, amount, days)
            everything = {d: [c for c in close.columns if close.iloc[:close.index.get_loc(d) + 1][c].notna().sum() >= MIN_HIST]
                          for d in days}
            rot, ew = index_backtest(close, picks, start, end), index_backtest(close, everything, start, end)
            out[f'{label}_phase{phase}'] = {'rotation': stats(rot), 'equal_weight': stats(ew)}
            if label == 'discovery':
                rot.to_csv(OUT / f'index_nav_phase{phase}.csv')
                pd.Series({str(d.date()): v for d, v in picks.items()}).to_json(OUT / f'picks_phase{phase}.json',
                                                                              force_ascii=False)
    (OUT / 'index_stage.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def panel_code(stock):
    stock = str(stock).zfill(6)
    return ('sh.' if stock.startswith('6') else 'bj.' if stock[0] in '489' else 'sz.') + stock


def ledger_code(code):
    market, number = code.split('.')
    return f'{number}.{market.upper()}'


def select():
    panel = Panel(VERSION)
    ok = eligible(panel, MIN_HIST)
    cap = panel.field('float_mktcap')
    members = pd.read_parquet(ROOT / 'data/sw/level1_members.parquet')
    members['code'] = members.stockcode.map(panel_code)
    by_ind = members.groupby('swindexcode').code.apply(list).to_dict()
    for phase in (0, 1):
        picks = json.loads((OUT / f'picks_phase{phase}.json').read_text())
        rows, ever, sizes = [], set(), []
        for day, inds in sorted(picks.items()):
            day = pd.Timestamp(day)
            if day not in cap.index:
                continue
            weights = {}
            for ind in inds:
                codes = [c for c in by_ind.get(ind, []) if c in cap.columns]
                values = cap.loc[day, codes].where(ok.loc[day, codes]).dropna()
                for c in values.nlargest(PER_IND).index:
                    weights[ledger_code(c)] = WEIGHT
            rows.append({'signal_date': str(day.date()), 'weights': weights, 'active': list(weights),
                         'sell': sorted(ever - set(weights)), 'sleeve': 0})
            ever |= set(weights)
            sizes.append(len(weights))
        if rows and pd.Timestamp(rows[0]['signal_date']) < pd.Timestamp(DISCOVERY[0]):
            opening = [r for r in rows if pd.Timestamp(r['signal_date']) < pd.Timestamp(DISCOVERY[0])][-1]
            rows = [dict(opening, signal_date=DISCOVERY[0])] + [r for r in rows if pd.Timestamp(r['signal_date']) >= pd.Timestamp(DISCOVERY[0])]
        (OUT / f'targets_phase{phase}.json').write_text(json.dumps({'targets': rows}, indent=1))
        print(json.dumps({'phase': phase, 'signals': len(rows), 'names_min': min(sizes), 'names_median': float(np.median(sizes))}))


def ledger(cost_mult):
    import stable_book as book
    from quantlab.data.contracts import MarketDataset
    plans = {p: json.loads((OUT / f'targets_phase{p}.json').read_text())['targets'] for p in (0, 1)}
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
        result = book.replay_account(dataset, targets, start=DISCOVERY[0], end=DISCOVERY[1], cost_mult=cost_mult)
        s = result['summary']
        result['nav'].to_frame('nav').to_csv(OUT / f'nav_phase{phase}_cost{cost_mult:g}.csv')
        rows.append({'phase': phase, 'cost_mult': cost_mult, 'cagr': s['cagr'], 'max_drawdown': s['max_drawdown'],
                     'fills': s['fills'], 'checks_passed': s['account_checks_passed'],
                     'missing_panels': len(s['missing_panels'])})
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'ledger_cost{cost_mult:g}.csv', index=False)


def report():
    idx = json.loads((OUT / 'index_stage.json').read_text())
    c1, c2 = pd.read_csv(OUT / 'ledger_cost1.csv'), pd.read_csv(OUT / 'ledger_cost2.csv')
    beats = {k: v['rotation']['cagr'] > v['equal_weight']['cagr'] for k, v in idx.items()}
    gate = {'ledger_cagr>=15%': bool(c1.cagr.mean() >= 0.15), 'ledger_mdd>=-30%': bool(c1.max_drawdown.mean() >= -0.30),
            'index_beats_ew_discovery': bool(beats['discovery_phase0'] and beats['discovery_phase1']),
            'index_beats_ew_confirm': bool(beats['confirm_phase0'] and beats['confirm_phase1']),
            'cost2_cagr>=12%': bool(c2.cagr.mean() >= 0.12)}
    gate['pass'] = all(gate.values())
    m2 = pd.read_csv(ROOT / 'output/p1_m2_20261008/is/ledger_nav_0_1_cost1.csv', index_col=0, parse_dates=True)['nav']
    extra = {}
    for phase in (0, 1):
        nav = pd.read_csv(OUT / f'nav_phase{phase}_cost1.csv', index_col=0, parse_dates=True)['nav']
        yearly = nav.groupby(nav.index.year).last()
        yearly = yearly / yearly.shift(1).fillna(nav.iloc[0]) - 1
        extra[phase] = {'years': {int(k): float(v) for k, v in yearly.items()},
                        'pre_2022_08': stats(nav.loc[:'2022-07-29']), 'post_2022_08': stats(nav.loc['2022-08-01':]),
                        'corr_m2_monthly': float(nav.resample('ME').last().pct_change().corr(
                            m2.resample('ME').last().pct_change()))}
    picks = json.loads((OUT / 'picks_phase0.json').read_text())
    freq = pd.Series([i for v in picks.values() for i in v]).value_counts(normalize=True).round(3).head(10).to_dict()
    out = {'ledger_cost1': c1.to_dict('records'), 'ledger_cost2': c2.to_dict('records'), 'index_stage': idx,
           'gate': gate, 'phases': extra, 'top_industry_share': freq, 'survivorship_note': 'current SW members only',
           'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG), 'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({'gate': gate, 'cost1': c1[['phase', 'cagr', 'max_drawdown']].to_dict('records'),
                      'cost2': c2[['phase', 'cagr']].to_dict('records')}, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('index', 'select', 'ledger', 'report'))
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'index': index_stage, 'select': select, 'report': report}.get(args.stage, lambda: ledger(args.cost))()
