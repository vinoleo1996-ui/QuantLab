#!/usr/bin/env python3
"""P1-B TURN50: low turnover within size quintiles, 50 names, 40-session rebalance
(docs/prereg/P1_B_TURN50_SIZE_MATCHED.md).

Stages:
  select            schedules for phases {0,10,20,30} plus daily benchmark and MKT/SMB/HML factors
  sim               fast simulator (report only)
  ledger --cost X   stable_book.replay_account (gate at cost 1.0)
  report            gates and report-only statistics
  forward           store the latest TURN cross-section and the conditional 50-name list; score matured ones
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import neutral_alpha as NA
from quantlab import research_statistics as RS
from quantlab import research_trials as T
from quantlab.smallcap.panel import Panel
from quantlab.smallcap.sim import Market, metrics, simulate

OUT = ROOT / 'output/p1_b_turn50_20261008'
FORWARD = ROOT / 'output/turn_forward'
PREREG = ROOT / 'docs/prereg/P1_B_TURN50_SIZE_MATCHED.md'
START, END = '2018-01-02', '2026-09-22'
PHASES = (0, 10, 20, 30)
STEP, GROUPS, PER_GROUP, WEIGHT = 40, 5, 10, 0.018
FORWARD_STEP = 20
FAMILY, TRIAL = 'B_price_volume', 'TURN50_SM'


def neutral_turn(data, i, mask):
    """-TURN residual on log cap, industry, small-30% dummy and z(log PB)."""
    codes = data.codes[mask]
    logcap = np.log(data.cap[i, mask])
    industry = NA.sector_letters(data.industries, data.dates[i], codes)
    small = (logcap <= np.quantile(logcap, NA.SMALL)).astype(float)
    value = NA.zscore(np.log(data.pb[i, mask]))
    z = NA.zscore(data.turn20[i, mask])
    ok = np.isfinite(z) & np.isfinite(logcap) & np.isfinite(value)
    out = np.full(len(z), np.nan)
    if ok.sum() < 100:
        return out, logcap
    dummies = pd.get_dummies(pd.Series(industry[ok])).to_numpy(dtype=float)
    X = np.hstack([logcap[ok][:, None], dummies, small[ok][:, None], value[ok][:, None]])
    beta, *_ = np.linalg.lstsq(X, z[ok], rcond=None)
    out[ok] = -(z[ok] - X @ beta)
    return out, logcap


def selectable(data, i):
    pb = data.pb[i]
    return data.eligible(i) & np.isfinite(pb) & (pb > 0)


def pick(data, i):
    mask = selectable(data, i)
    score, logcap = neutral_turn(data, i, mask)
    codes = data.codes[mask]
    ok = np.isfinite(score)
    group = pd.qcut(pd.Series(logcap[ok]).rank(method='first'), GROUPS, labels=False).to_numpy()
    chosen = []
    for g in range(GROUPS):
        idx = np.where(ok)[0][group == g]
        chosen += list(codes[idx[np.argsort(-score[idx])][:PER_GROUP]])
    return chosen, codes, score


def select():
    data = NA.Data()
    dates = data.dates
    first = dates.searchsorted(pd.Timestamp(START))
    last = dates.searchsorted(pd.Timestamp(END), side='right') - 1
    schedules, exposure = {}, []
    for phase in PHASES:
        schedule = {}
        for i in range(first + phase, last + 1, STEP):
            chosen, codes, _ = pick(data, i)
            schedule[dates[i]] = {c: WEIGHT for c in chosen}
            if phase == 0:
                mask = selectable(data, i)
                sel = np.isin(data.codes, chosen)
                exposure.append({'date': dates[i], 'names': len(chosen),
                                 'dlogcap': float(np.log(data.cap[i, sel]).mean() - np.log(data.cap[i, mask]).mean()),
                                 'dlogpb': float(np.log(data.pb[i, sel]).mean() - np.log(data.pb[i, mask]).mean())})
        schedules[phase] = schedule
    bench, hml = [], []
    for t in range(first, last + 1):
        prev = selectable(data, t - 1)
        r = data.ret[t, prev]
        bench.append(float(np.nanmean(r)))
        live = (data.status[t] == 1) & (data.st[t] != 1) & ~data.bj & np.isfinite(data.ret[t])
        pb = data.pb[t - 1]
        ok = live & np.isfinite(pb) & (pb > 0)
        lo, hi = np.quantile(pb[ok], [0.3, 0.7])
        hml.append(float(data.ret[t, ok & (pb <= lo)].mean() - data.ret[t, ok & (pb >= hi)].mean()))
    factors = pd.DataFrame({'bench': bench, 'mkt': data.mkt[first:last + 1], 'smb': data.smb[first:last + 1],
                            'hml': hml}, index=dates[first:last + 1])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'schedules.pkl', 'wb') as handle:
        pickle.dump(schedules, handle)
    factors.to_parquet(OUT / 'factors.parquet')
    pd.DataFrame(exposure).to_csv(OUT / 'exposure_phase0.csv', index=False)
    print(json.dumps({'stage': 'select', 'signals': {p: len(s) for p, s in schedules.items()},
                      'median_names': float(pd.DataFrame(exposure).names.median())}), flush=True)


def load_schedules():
    with open(OUT / 'schedules.pkl', 'rb') as handle:
        return pickle.load(handle)


def run_sim():
    panel = Panel(NA.VERSION)
    delist = panel.table('delistings')
    market = Market.from_panel(panel, START, END, delist=dict(zip(delist.code, delist.last_bar)))
    rows = []
    for phase, schedule in load_schedules().items():
        result = simulate(market, schedule)
        result['nav'].to_frame('nav').to_csv(OUT / f'sim_nav_{phase}.csv')
        rows.append({'phase': phase, **{k: v for k, v in metrics(result['nav']).items() if k != 'yearly'},
                     'median_gross': float(result['gross'][result['gross'] > 0].median()), 'fees': result['fees']})
        print(json.dumps(rows[-1], default=float), flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'sim.csv', index=False)


def to_targets(schedule):
    rows, ever = [], set()
    for day in sorted(schedule):
        weights = {f"{c.split('.')[1]}.{c.split('.')[0].upper()}": w for c, w in schedule[day].items()}
        rows.append({'signal_date': str(pd.Timestamp(day).date()), 'weights': weights, 'active': list(weights),
                     'sell': sorted(ever - set(weights)), 'sleeve': 0})
        ever |= set(weights)
    return rows


def ledger(cost_mult):
    import stable_book as book
    from quantlab.data.contracts import MarketDataset
    plans = {phase: to_targets(s) for phase, s in load_schedules().items()}
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
        result['daily_return'].to_frame('ret').to_csv(OUT / f'ledger_ret_{phase}_cost{cost_mult:g}.csv')
        rows.append({'phase': phase, 'cost_mult': cost_mult, 'cagr': s['cagr'], 'mdd': s['max_drawdown'],
                     'fees': s['fees'], 'fills': s['fills'], 'checks_passed': s['account_checks_passed'],
                     'missing_panels': len(s['missing_panels'])})
        print(json.dumps(rows[-1], default=float), flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'ledger_cost{cost_mult:g}.csv', index=False)


def evaluate_path(ret, factors):
    frame = pd.concat([ret.rename('y'), factors], axis=1, join='inner').dropna()
    beta, t = RS.newey_west(frame.y.to_numpy(), frame[['mkt', 'smb', 'hml']].to_numpy(), lags=20)
    years = (1 + frame[['y', 'bench']]).groupby(frame.index.year).prod() - 1
    excess = years.y - years.bench
    n_years = len(frame) / 252
    cagr = float((1 + frame.y).prod() ** (1 / n_years) - 1)
    bench = float((1 + frame.bench).prod() ** (1 / n_years) - 1)
    nav = (1 + frame.y).cumprod()
    return {'cagr': cagr, 'bench_cagr': bench, 'excess_cagr': cagr - bench,
            'mdd': float((nav / nav.cummax().clip(lower=1.0) - 1).min()),
            'alpha_annual': float(beta[0] * 252), 'alpha_t': float(t[0]),
            'beta_mkt': float(beta[1]), 'beta_smb': float(beta[2]), 'beta_hml': float(beta[3]),
            't_smb': float(t[2]), 't_hml': float(t[3]),
            'years_excess': {int(k): float(v) for k, v in excess.items()},
            'positive_years': int((excess > 0).sum()), 'n_years': int(len(excess)),
            **RS.win_rates(frame.y, frame.bench),
            'dsr': RS.deflated_sharpe(frame.y - frame.bench, trials=T.count(FAMILY, include_diagnostics=False))
            .get('value')}


def report():
    factors = pd.read_parquet(OUT / 'factors.parquet')
    out = {'prereg': str(PREREG.relative_to(ROOT)), 'prereg_sha256': T.sha256_file(PREREG),
           'family_trials': T.count(FAMILY, include_diagnostics=False),
           'global_trials': T.count(include_diagnostics=False), 'promotion_approved': False, 'ledger': {}}
    for cost in (1, 2):
        paths = {}
        for phase in PHASES:
            path = OUT / f'ledger_ret_{phase}_cost{cost}.csv'
            if path.exists():
                ret = pd.read_csv(path, index_col=0, parse_dates=True).ret
                paths[phase] = evaluate_path(ret, factors)
        if paths:
            frame = pd.DataFrame(paths).T
            median = {c: float(pd.to_numeric(frame[c]).median()) for c in frame.columns
                      if c != 'years_excess' and pd.api.types.is_numeric_dtype(pd.to_numeric(frame[c], errors='coerce'))}
            out['ledger'][f'cost{cost}'] = {'paths': paths, 'median': median}
    sim = {}
    for phase in PHASES:
        path = OUT / f'sim_nav_{phase}.csv'
        if path.exists():
            nav = pd.read_csv(path, index_col=0, parse_dates=True).nav
            sim[phase] = evaluate_path(nav.pct_change(fill_method=None).fillna(0.0), factors)
    out['sim'] = sim
    exposure = pd.read_csv(OUT / 'exposure_phase0.csv')
    out['exposure_phase0'] = {'dlogcap_mean': float(exposure.dlogcap.mean()), 'dlogpb_mean': float(exposure.dlogpb.mean()),
                              'names_median': float(exposure.names.median())}
    m = out['ledger'].get('cost1', {}).get('median')
    if m:
        gate = {'excess_cagr>0': m['excess_cagr'] > 0, 'alpha_t>=2': m['alpha_t'] >= 2,
                'positive_years>=6/9': m['positive_years'] >= 6}
        gate['pass'] = all(gate.values())
        out['gate'] = gate
    (OUT / 'report.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({'gate': out.get('gate'), 'median_cost1': m, 'exposure': out['exposure_phase0']}, indent=1,
                     default=float))


def forward():
    """Snapshot every FORWARD_STEP sessions at the latest panel date; score snapshots whose horizon has matured."""
    data = NA.Data()
    FORWARD.mkdir(parents=True, exist_ok=True)
    i = len(data.dates) - 1
    day = data.dates[i]
    existing = sorted(FORWARD.glob('snapshot_*.parquet'))
    last_day = pd.Timestamp(existing[-1].stem.split('_')[1]) if existing else None
    due = last_day is None or data.dates.searchsorted(day) - data.dates.searchsorted(last_day) >= FORWARD_STEP
    if due:
        mask = data.eligible(i)
        signals, _, _ = NA.neutral_set(data, i, mask)
        chosen, _, _ = pick(data, i)
        snap = pd.DataFrame({'code': data.codes[mask], 'turn_neutral': signals['TURN']})
        snap['turn50_conditional'] = snap.code.isin(chosen)
        snap.to_parquet(FORWARD / f'snapshot_{day.date()}.parquet')
    scores = []
    for path in sorted(FORWARD.glob('snapshot_*.parquet')):
        d = pd.Timestamp(path.stem.split('_')[1])
        k = data.dates.searchsorted(d)
        if k + 1 + NA.HORIZON >= len(data.dates):
            scores.append({'date': str(d.date()), 'status': 'pending'})
            continue
        snap = pd.read_parquet(path)
        mask = np.isin(data.codes, snap.code)
        fwd, blocked = data.forward(k, mask, NA.HORIZON)
        fwd = np.where(blocked, np.nan, fwd)
        series = pd.Series(fwd, index=data.codes[mask]).reindex(snap.code).to_numpy()
        scores.append({'date': str(d.date()), 'status': 'scored', 'ic': NA.rank_ic(snap.turn_neutral.to_numpy(), series),
                       'turn50_excess': float(np.nanmean(series[snap.turn50_conditional.to_numpy()]) - np.nanmean(series))})
    (FORWARD / 'scores.json').write_text(json.dumps({'panel_last_date': str(day.date()), 'new_snapshot': bool(due),
                                                     'scores': scores, 'note': 'TURN50 list is conditional on its trial'},
                                                    indent=1, default=float))
    print(json.dumps(scores[-3:], default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('select', 'sim', 'ledger', 'report', 'forward'))
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()
    {'select': select, 'sim': run_sim, 'report': report, 'forward': forward}.get(args.stage, lambda: ledger(args.cost))()
