#!/usr/bin/env python3
"""P8 PEAD/SUE Stage A: post-announcement drift inside the bottom-40% ADV pool.

Zero-cost, no ledger: the event portfolio and the pool are both marked with adjusted
closes, so this answers only whether the signal exists before costs. A high-SUE event is
known at the first session after ``max(available_at, ann_date 18:00)``; the pool is the
same relative-band ADV pool the book uses. Holding windows are the preregistered 20 and
60 sessions; overlapping events are averaged day by day, and the paired daily excess is
tested with a Newey-West t and a simultaneous block CI.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab.alpha.features import Fundamentals
from quantlab.research_statistics import deflated_sharpe, newey_west, simultaneous_block_ci
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

MANIFEST = '20261013T023232Z'
FIN = ROOT / 'data/cloud/financials_pit/bundle_20261012/part-all-all.parquet'
OUT = ROOT / 'output/p8_pead_stage_a_20261018'
START, END = '2018-01-02', '2026-09-22'
HOLDS = (20, 60)
SUE_LOOKBACK = 8
MIN_SEASONAL_DIFFS = 4
EVENT_LOOKBACK_SESSIONS = 5
VARIANTS = {20: 'SUE20', 60: 'SUE60'}
PREREG = ROOT / 'docs/prereg/P8_PEAD_SUE.md'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_financials():
    columns = ['ts_code', 'ann_date', 'end_date', 'report_type', 'update_flag',
               'available_at', *Fundamentals.COLUMNS]
    table = pd.read_parquet(FIN, columns=columns)
    known = Fundamentals(table).table
    known = known[known.report_type.astype(str).isin(['1', '2', '3', '4'])]
    known = known[known.n_income_attr_p.notna()]
    # the value known first: earliest availability, original disclosure before a restatement
    known = known.sort_values(['available', 'revision'], ascending=[True, True])
    return known.drop_duplicates(['code', 'end'], keep='first')


def sue_events(known):
    rows = []
    for code, frame in known.groupby('code', sort=False):
        frame = frame.sort_values('end')
        by_end = dict(zip(frame.end, frame.n_income_attr_p))
        events = dict(zip(frame.end, frame.available))
        diffs = {}
        for end, value in by_end.items():
            before = by_end.get(end - pd.DateOffset(years=1))
            if before is not None and np.isfinite(before):
                diffs[end] = value - before
        ends = sorted(diffs)
        for i, end in enumerate(ends):
            prior = [diffs[e] for e in ends[max(0, i - SUE_LOOKBACK):i]
                     if e - pd.DateOffset(years=1) in by_end]
            if len(prior) < MIN_SEASONAL_DIFFS:
                continue
            spread = float(np.std(prior, ddof=1))
            if not np.isfinite(spread) or spread <= 0:
                continue
            rows.append({'code': code, 'end': end, 'available': events[end],
                         'sue': float(diffs[end]) / spread})
    return pd.DataFrame(rows)


def pool_by_day(panel, features, sessions):
    pools = {}
    for day in sessions:
        pools[day] = set(U.pool_on(day, features, 'adv', band='relative', exclude_st=True))
    return pools


def daily_series(events, sessions, returns, pools, hold):
    position = {day: i for i, day in enumerate(sessions)}
    sums = np.zeros(len(sessions))
    counts = np.zeros(len(sessions))
    for row in events.itertuples():
        after = sessions[sessions > row.available.normalize()]
        if not len(after):
            continue
        signal = after[0]
        start = position[signal] + 1
        stop = min(start + hold, len(sessions))
        if start >= stop:
            continue
        values = returns.iloc[start:stop, returns.columns.get_indexer([row.code])].to_numpy().ravel()
        valid = np.isfinite(values)
        sums[start:stop] += np.where(valid, values, 0.0)
        counts[start:stop] += valid
    with np.errstate(invalid='ignore'):
        series = np.where(counts > 0, sums / counts, np.nan)
    return pd.Series(series, index=sessions)


def pool_series(sessions, returns, pools):
    values = []
    for i, day in enumerate(sessions):
        if i == 0:
            values.append(np.nan)
            continue
        codes = [c for c in pools[sessions[i - 1]] if c in returns.columns]
        row = returns.iloc[i].reindex(codes)
        values.append(float(row.mean()) if len(row) else np.nan)
    return pd.Series(values, index=sessions)


def evaluate(name, high, low, pool, events, hold):
    mask = high.notna() & pool.notna()
    excess = (high - pool)[mask]
    t = float(newey_west(excess.to_numpy(), lags=20)[1][0])
    p = float(2 * (1 - norm.cdf(abs(t))))
    ci = simultaneous_block_ci(excess.to_numpy(), block=20, samples=2000)
    dsr = deflated_sharpe(excess.dropna(), trials=len(HOLDS))
    low_mask = low.notna() & pool.notna()
    low_excess = (low - pool)[low_mask]
    low_t = float(newey_west(low_excess.to_numpy(), lags=20)[1][0]) if len(low_excess) > 30 else float('nan')
    years = events.end.dt.year.value_counts().sort_index().to_dict()
    ex2025 = excess[excess.index < '2025-01-01']
    return {'variant': name, 'hold_sessions': hold,
            'events': int(len(events)), 'fiscal_years': len(years),
            'years': {int(k): int(v) for k, v in years.items()},
            'days': int(len(excess)),
            'mean_daily_excess': float(excess.mean()),
            'annualised_excess': float(excess.mean() * 252),
            'nw_t': t, 'p_value': p,
            'block_ci_95': ci, 'dsr': dsr,
            'low_sue_nw_t': low_t,
            'ex_2025_mean_daily': float(ex2025.mean()) if len(ex2025) else float('nan'),
            'same_sign_ex_2025': bool(np.sign(ex2025.mean()) == np.sign(excess.mean()))
            if len(ex2025) else False}


def bh_fdr(pvalues):
    order = np.argsort(pvalues)
    ranked = np.empty(len(pvalues))
    m = len(pvalues)
    for rank, index in enumerate(order, start=1):
        ranked[index] = pvalues[index] * m / rank
    return np.minimum.accumulate(ranked[::-1])[::-1]


def main():
    panel = Panel(MANIFEST)
    sessions = panel.sessions
    sessions = sessions[(sessions >= pd.Timestamp(START)) & (sessions <= pd.Timestamp(END))]
    features = U.rolling_features(panel, with_st=True)
    returns = panel.field('adj_close').reindex(sessions).pct_change(fill_method=None)
    is_st = panel.field('is_st').reindex(sessions)

    known = load_financials()
    events = sue_events(known)
    events['available'] = pd.to_datetime(events['available']).dt.tz_localize(None)
    events = events[(events.available >= pd.Timestamp(START))
                      & (events.available <= pd.Timestamp(END) + pd.Timedelta(days=10))]

    pools = pool_by_day(panel, features, sessions)
    position = {day: i for i, day in enumerate(sessions)}
    kept = []
    for row in events.itertuples():
        after = sessions[sessions > row.available.normalize()]
        if not len(after):
            continue
        signal = after[0]
        pool = pools[signal]
        if row.code not in pool:
            continue
        recent = sessions[(sessions <= signal)][-EVENT_LOOKBACK_SESSIONS:]
        if isinstance(is_st.loc[signal, row.code], (bool, np.bool_)) and bool(is_st.loc[signal, row.code]):
            continue
        kept.append({'code': row.code, 'end': row.end, 'available': row.available,
                     'sue': row.sue, 'signal': signal})
    events = pd.DataFrame(kept)
    if events.empty:
        raise SystemExit('no_events_left_in_pool')

    # daily cross-section: high/low quartile among the events visible on each signal day
    events['signal'] = pd.to_datetime(events['signal'])
    high_rows, low_rows = [], []
    for signal, frame in events.groupby('signal'):
        frame = frame.sort_values('sue')
        n = len(frame)
        if n < 4:
            continue
        q = max(1, n // 4)
        low_rows.append(frame.iloc[:q])
        high_rows.append(frame.iloc[-q:])
    high_events = pd.concat(high_rows, ignore_index=True) if high_rows else events.iloc[:0]
    low_events = pd.concat(low_rows, ignore_index=True) if low_rows else events.iloc[:0]

    pool = pool_series(sessions, returns, pools)
    variants, reports = [], {}
    for hold in HOLDS:
        high = daily_series(high_events, sessions, returns, pools, hold)
        low = daily_series(low_events, sessions, returns, pools, hold)
        report = evaluate(VARIANTS[hold], high, low, pool, high_events, hold)
        variants.append(report)
        reports[hold] = report
    qvals = bh_fdr([v['p_value'] for v in variants])
    for value, q in zip(variants, qvals):
        value['bh_q'] = float(q)

    gates = {}
    for hold, report in reports.items():
        gates[str(hold)] = {
            'nw_t_ge_2': report['nw_t'] >= 2.0,
            'events_ge_200': report['events'] >= 200,
            'five_fiscal_years': report['fiscal_years'] >= 5,
            'ex_2025_same_sign': report['same_sign_ex_2025'],
            'bh_q_le_0_10': report['bh_q'] <= 0.10,
            'dsr_positive': bool(report['dsr'].get('value') is not None
                                 and report['dsr']['value'] > 0),
            'low_sue_weaker': not np.isfinite(report['low_sue_nw_t'])
                              or report['low_sue_nw_t'] < report['nw_t'],
        }
        gates[str(hold)]['passed'] = all(gates[str(hold)].values())

    OUT.mkdir(parents=True, exist_ok=True)
    result = {'strategy': 'P8_PEAD_SUE', 'stage': 'A', 'zero_cost': True,
              'prereg': str(PREREG.relative_to(ROOT)), 'prereg_sha256': sha256(PREREG),
              'manifest': MANIFEST, 'panel_close_sha256': sha256(
                  ROOT / f'data/smallcap_v4/{MANIFEST}/close.parquet'),
              'financials_sha256': sha256(FIN), 'financials_rows': int(len(known)),
              'window': [START, END], 'variants': variants, 'gates': gates,
              'control': {'low_sue_nw_t': [v['low_sue_nw_t'] for v in variants],
                          'note': 'low-SUE quartile on the same signal days must be weaker'},
              'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float))
    events.to_csv(OUT / 'events.csv', index=False)
    high_events.to_csv(OUT / 'high_sue_events.csv', index=False)
    pd.DataFrame({'pool': pool}).to_csv(OUT / 'pool_ew_daily.csv')
    for hold in HOLDS:
        pd.DataFrame({'high': daily_series(high_events, sessions, returns, pools, hold),
                      'pool': pool}).to_csv(OUT / f'daily_{hold}.csv')
    print(json.dumps(result, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
