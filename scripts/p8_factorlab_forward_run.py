#!/usr/bin/env python3
"""P8: first real run of the frozen FactorLab referee on the ten proposed candidates.

Historic screen only: the forward stream starts at submission (2026-10-12) and the trusted
panel ends there, so no forward session exists yet and the report must say ``forward_pending``.
The screen uses the referee's own frozen thresholds (IC 0.02, screen t 3.0, holdout t 2.0,
library correlation 0.5) and its H=20 non-overlapping observation days.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab.factor_lab import dsl
from quantlab.factor_lab.referee import screen
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

MANIFEST = '20261013T023232Z'
CANDIDATES = ROOT / 'output/factor_lab/candidates.jsonl'
OUT = ROOT / 'output/p8_factorlab_forward_20261018'
PREREG = ROOT / 'docs/prereg/P8_FACTORLAB_FORWARD.md'
START, END = '2015-01-01', '2026-10-12'
SAME_FAMILY = {
    'liquidity': ('amihud', 'amount', 'volume', 'turnover', 'float_cap'),
    'reversal': ('ret', 'delta(close', 'delay(close'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_days(sessions, step=20):
    window = sessions[(sessions >= pd.Timestamp(START)) & (sessions <= pd.Timestamp(END))]
    return window[::step]


def pool_members(panel, sessions, days):
    features = U.rolling_features(panel, with_st=True)
    adv = features['adv'].reindex(sessions)
    eligible = (features['history'].reindex(sessions).ge(253) & adv.notna() & adv.ge(5e6)
                & ~features['st'].reindex(sessions).eq(1))
    eligible = eligible.reindex(columns=adv.columns)
    rank = adv.where(eligible).rank(axis=1, ascending=False, method='first')
    selected = rank.gt(np.ceil(0.6 * eligible.sum(axis=1)), axis=0) & eligible
    members = []
    for day in days:
        codes = selected.columns[selected.loc[day].to_numpy()]
        for code in codes:
            members.append({'date': day, 'code': code})
    return pd.DataFrame(members)


def label_frame(open_prices, days, horizon=20):
    # signal at close t, entry at open t+1: the label must not include anything up to close t
    forward = open_prices.shift(-(horizon + 1)) / open_prices.shift(-1) - 1
    rows = []
    for day in days:
        series = forward.loc[day].dropna()
        for code, value in series.items():
            rows.append({'date': day, 'code': code, 'label': float(value)})
    return pd.DataFrame(rows)


def library_frame(panels, days, members):
    ret = panels['ret']
    close = panels['close']
    amount = panels['amount']
    volume = panels['volume']
    library = {'ret20': close.pct_change(20, fill_method=None),
               'vol20': ret.rolling(20, min_periods=10).std(),
               'turn20': panels['turnover'].rolling(20, min_periods=10).mean(),
               'amihud': (ret.abs() / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean(),
               'close_to_high': close / close.rolling(250, min_periods=60).max() - 1,
               'hl_range': ((panels['high'] - panels['low']) / close).rolling(20, min_periods=10).mean()}
    rows = []
    for day in days:
        frame = pd.DataFrame({name: value.loc[day] for name, value in library.items()})
        frame = frame.stack().unstack()
        frame.index.name = 'code'
        frame = frame.reset_index()
        frame['date'] = day
        rows.append(frame)
    long = pd.concat(rows, ignore_index=True)
    return long.merge(members, on=['date', 'code'], how='inner')


def main():
    panel = Panel(MANIFEST)
    sessions = panel.sessions[(panel.sessions >= pd.Timestamp(START)) &
                              (panel.sessions <= pd.Timestamp(END))]
    fields = {name: panel.field(name).reindex(sessions)
              for name in ('open', 'high', 'low', 'close', 'adj_close', 'volume', 'amount',
                           'turnover', 'float_mktcap', 'is_st')}
    days = observation_days(sessions)
    members = pool_members(panel, sessions, days)
    panels = dsl.panels_from_fields(fields)
    labels = label_frame(panels['open'], days)
    library = library_frame(panels, days, members)

    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line.strip()]
    results = []
    for candidate in candidates:
        try:
            values = dsl.evaluate(candidate['formula'], panels).loc[days]
        except Exception as exc:  # noqa: BLE001 - report, never hide a rejected formula
            results.append({'name': candidate['name'], 'id': candidate['id'],
                            'formula': candidate['formula'], 'status': 'formula_error',
                            'error': f'{type(exc).__name__}:{exc}'})
            continue
        long = values.stack().rename('value').reset_index()
        long.columns = ['date', 'code', 'value']
        long = long.merge(members, on=['date', 'code'], how='inner')
        outcome = screen(long, labels, library, days)
        family = [name for name, keys in SAME_FAMILY.items()
                  if any(key in candidate['formula'] for key in keys)]
        outcome.update({'name': candidate['name'], 'id': candidate['id'],
                        'formula': candidate['formula'], 'same_family': family})
        results.append(outcome)

    frame = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'screen.csv', index=False)
    report = {'strategy': 'P8_FACTORLAB_FORWARD', 'stage': 'historical_screen',
              'prereg': str(PREREG.relative_to(ROOT)), 'prereg_sha256': sha256(PREREG),
              'candidates_sha256': sha256(CANDIDATES),
              'manifest': MANIFEST, 'window': [START, END],
              'observation_days': int(len(days)), 'pool_members_observed': int(len(members)),
              'passed': [row['name'] for row in results if row.get('passed')],
              'same_family_passed': [row['name'] for row in results
                                     if row.get('passed') and row.get('same_family')],
              'formula_errors': [row['name'] for row in results
                                 if row.get('status') == 'formula_error'],
              'forward': {'submitted_at': '2026-10-12', 'observations': 0,
                          'status': 'forward_pending',
                          'reason': 'the trusted panel ends 2026-10-12; no forward session yet'},
              'results': results, 'promotion_approved': False}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, ensure_ascii=False,
                     indent=1, default=float))
    for row in results:
        print(row.get('name'), 'passed=', row.get('passed'), 'screen_ic=', row.get('screen_mean_ic'),
              'screen_t=', row.get('screen_t'), 'holdout_t=', row.get('holdout_t'),
              'corr=', row.get('max_library_corr'), 'family=', row.get('same_family'),
              row.get('error', ''))


if __name__ == '__main__':
    main()
