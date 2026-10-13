#!/usr/bin/env python3
"""P4 paper forward record (starts only after docs/prereg/P4_ALLOC.md passes; never trades).

  .venv2/bin/python scripts/p4_forward.py snapshot --asof 2026-10-30
  .venv2/bin/python scripts/p4_forward.py settle --to 2026-11-27

``snapshot`` freezes the 50-name equity basket (the P4_alpha equity book, refit on every label known
by ``asof``) and the ETF sleeve weights (P4_ALLOC rules) at a month end. ``settle`` books the period
from the latest snapshot to ``--to`` at closing prices with estimated costs; periods the panel does
not cover stay ``pending_data``. Output: output/p4_forward/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alloc as ALLOC_SCRIPT
from quantlab import p4_alloc as A
from quantlab.alpha import features as F
from quantlab.alpha.labels import HORIZON, forward_open_returns
from quantlab.alpha.models import GBDTEnsemble, LinearICIR, percentile_frame
from quantlab.alpha.portfolio import random_order, select, select_hysteresis
from quantlab.cloud import load_snapshot
from quantlab.p4_mandate import MANDATE
from quantlab.smallcap import universe
from quantlab.smallcap.panel import Panel, latest_version

OUT = ROOT / 'output/p4_forward'
ALPHA = ROOT / 'output/p4_alpha_20261012'
ALLOC = ROOT / 'output/p4_alloc_20261012/report.json'
EQUITY_TRADE_COST, ETF_TRADE_COST = 0.004, 0.0005
FIELDS = ('open', 'high', 'low', 'close', 'adj_close', 'amount', 'turnover', 'float_mktcap', 'tradestatus',
          'limit_up', 'is_st')


ETF_TS = {'gold': '518880.SH', 'nasdaq': '513100.SH', 'sp500': '513500.SH', 'treasury': '511010.SH'}
QDII = ('nasdaq', 'sp500')


def etf_returns_latest(sessions):
    """Daily returns of the ETF sleeves from the pushed bundle (close x adj_factor), aligned to ``sessions``."""
    bars = load_snapshot('etf_daily_cross_asset')
    bars['date'] = pd.to_datetime(bars['trade_date'].astype(str))
    bars['hfq'] = bars['close'] * bars['adj_factor']
    wide = bars.pivot(index='date', columns='ts_code', values='hfq').sort_index()
    wide = wide[list(ETF_TS.values())].rename(columns={v: k for k, v in ETF_TS.items()})
    wide = wide.reindex(wide.index.union(sessions)).ffill().reindex(sessions)
    return wide.pct_change(fill_method=None).fillna(0.0), wide


def qdii_premium(day):
    """Close on ``day`` over the latest unit NAV announced on or before ``day``, minus 1."""
    nav = load_snapshot('qdii_nav')
    nav['ann'] = pd.to_datetime(nav['ann_date'].astype(str))
    bars = load_snapshot('etf_daily_cross_asset')
    bars['date'] = pd.to_datetime(bars['trade_date'].astype(str))
    out = {}
    for sleeve in QDII:
        code = ETF_TS[sleeve]
        known = nav[(nav.ts_code == code) & (nav.ann <= day)].sort_values(['nav_date', 'ann'])
        close = bars[(bars.ts_code == code) & (bars.date <= day)].sort_values('date')
        out[sleeve] = (float(close.close.iloc[-1] / known.unit_nav.iloc[-1] - 1)
                       if len(known) and len(close) else None)
    return out


def apply_premium_pause(weights, previous, premium):
    """QDII sleeves above the pause threshold may not grow; the freed weight goes to treasury."""
    names = ['equity'] + list(A.ETF_CODES)
    out = dict(zip(names, map(float, weights)))
    paused = []
    for sleeve in QDII:
        value = premium.get(sleeve)
        if value is not None and value > MANDATE.qdii_premium_pause and previous is not None:
            cap = float(previous['sleeves'].get(sleeve, 0.0))
            if out[sleeve] > cap:
                out['treasury'] += out[sleeve] - cap
                out[sleeve] = cap
                paused.append(sleeve)
    return out, paused


BOOKS = ('POOL50', 'GBDT_HYST')
AUTHORIZATION = ('user 2026-10-13 17:20 CST: paper-record both books although neither passed its gates; '
                 'observation only, no trading, verdicts unchanged')


def require_pass(book=None, authorized=False):
    """Forward needs a passed allocation layer, or an explicit user authorization for a named book."""
    passed = ALLOC.exists() and json.loads(ALLOC.read_text()).get('passed')
    if not passed and not authorized:
        raise SystemExit(json.dumps(dict(status='not_started', reason='P4_ALLOC has not passed')))
    report = json.loads(ALLOC.read_text()) if ALLOC.exists() else {}
    chosen = book or report.get('equity_book')
    if chosen not in BOOKS:
        raise SystemExit(json.dumps(dict(status='not_started', reason=f'unknown book {chosen}')))
    return dict(equity_book=chosen, gates_passed=bool(passed), authorization=None if passed else AUTHORIZATION)


def book_dir(book):
    return OUT / book


def snapshots(book):
    return sorted(book_dir(book).glob('snapshot_*.json'))


def stage_snapshot(asof, book=None, authorized=False):
    alloc = require_pass(book, authorized)
    book = alloc['equity_book']
    panel = Panel(latest_version())
    asof = pd.Timestamp(asof)
    if asof not in panel.sessions:
        raise SystemExit(json.dumps(dict(status='pending_data', reason=f'panel ends {panel.sessions[-1].date()}')))
    fields = {name: panel.field(name) for name in FIELDS}
    pools = universe.rolling_features(panel, with_st=True)
    pool = [c for c in universe.pool_on(asof, pools, 'adv', band='relative', exclude_st=True, min_history=252)
            if not c.startswith('bj.')]
    sectors = F.Industries(load_snapshot('sw_l1_members'))
    industries = sectors.on(asof).reindex(pool).fillna('unclassified').to_dict()
    previous = json.loads(snapshots(book)[-1].read_text()) if snapshots(book) else None
    held = previous['equity']['names'] if previous else []
    if book == 'POOL50':
        ordered = random_order(pool, 1)
        model_note = dict(model='POOL50', seed=1)
    else:
        price = F.PriceVolume(fields)
        today = F.cross_section(asof, pool, price, F.Fundamentals(load_snapshot('financials_pit')), sectors,
                                fields['close'])
        samples = pd.read_parquet(ALPHA / 'samples.parquet')
        present = json.loads((ALPHA / 'samples_meta.json').read_text())['features']
        position = pd.Series(np.arange(len(panel.sessions)), index=panel.sessions)
        train = samples[(samples.label_known_pos <= position[asof]) & samples.label.notna()]
        model = GBDTEnsemble().fit(train, present) if book in ('GBDT', 'GBDT_HYST') else LinearICIR().fit(train, present)
        scored = percentile_frame(today, present)
        scored['score'] = model.predict(scored)
        ordered = list(scored.sort_values('score', ascending=False, kind='mergesort').code)
        model_note = dict(model=book, train_rows=len(train), last_label_known=str(panel.sessions[int(train.label_known_pos.max())].date()))
    names = (select_hysteresis if book == 'GBDT_HYST' else select)(ordered, industries, held)
    etf, _ = etf_returns_latest(panel.sessions[panel.sessions <= asof])
    nav_rows = _nav_rows(book)
    peak = max([1.0] + [r['nav'] for r in nav_rows])
    current = nav_rows[-1]['nav'] if nav_rows else 1.0
    stressed = bool(previous and previous['stressed'])
    drawdown = 1 - current / peak
    stressed = (drawdown >= A.DISASTER_DD) if not stressed else (drawdown > A.RECOVER_DD)
    weights = A.target_weights(etf, A.DISASTER_EQUITY if stressed else 0.60)
    premium = qdii_premium(asof)
    sleeves, paused = apply_premium_pause(weights, previous, premium)
    snapshot = dict(asof=str(asof.date()), created_at=datetime.now(timezone.utc).isoformat(), panel=latest_version(),
                    equity=dict(book=book, names=names, weight_each=1 / len(names) if names else 0.0, **model_note),
                    sleeves=sleeves, qdii_premium=premium, qdii_paused=paused,
                    stressed=stressed, mandate=MANDATE.version, capital_cny=MANDATE.capital_cny,
                    gates_passed=alloc['gates_passed'], authorization=alloc['authorization'],
                    promotion_approved=False, forward_decision=False)
    book_dir(book).mkdir(parents=True, exist_ok=True)
    (book_dir(book) / f'snapshot_{asof.date()}.json').write_text(json.dumps(snapshot, indent=1, ensure_ascii=False))
    print(json.dumps(dict(asof=snapshot['asof'], names=len(names), sleeves=snapshot['sleeves'], stressed=stressed)))


def _nav_rows(book):
    path = book_dir(book) / 'nav.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stage_settle(to, book=None, authorized=False):
    book = require_pass(book, authorized)['equity_book']
    if not snapshots(book):
        raise SystemExit(json.dumps(dict(status='no_snapshot')))
    snap = json.loads(snapshots(book)[-1].read_text())
    start, to = pd.Timestamp(snap['asof']), pd.Timestamp(to)
    panel = Panel(latest_version())
    if panel.sessions[-1] < to:
        row = dict(period=[str(start.date()), str(to.date())], status='pending_data', panel_end=str(panel.sessions[-1].date()))
        print(json.dumps(row))
        return
    window = panel.sessions[(panel.sessions > start) & (panel.sessions <= to)]
    adj = panel.field('adj_close')[snap['equity']['names']].loc[[start, *window]]
    basket = float(adj.iloc[-1].div(adj.iloc[0]).fillna(1.0).mean() - 1) if snap['equity']['names'] else 0.0
    etf, _ = etf_returns_latest(pd.DatetimeIndex([start, *window]))
    etf_growth = (1 + etf.iloc[1:]).prod() - 1
    sleeves = snap['sleeves']
    previous = json.loads(snapshots(book)[-2].read_text()) if len(snapshots(book)) > 1 else None
    turnover = 1.0 if previous is None else len(set(snap['equity']['names']) - set(previous['equity']['names'])) / 50
    gross = sleeves['equity'] * basket + sum(sleeves[k] * float(etf_growth[k]) for k in A.ETF_CODES)
    cost = sleeves['equity'] * turnover * 2 * EQUITY_TRADE_COST
    rows = _nav_rows(book)
    nav = (rows[-1]['nav'] if rows else 1.0) * (1 + gross - cost)
    row = dict(period=[str(start.date()), str(to.date())], status='settled', equity_return=basket,
               etf_returns={k: float(etf_growth[k]) for k in A.ETF_CODES}, gross=gross, cost=cost, nav=nav,
               settled_at=datetime.now(timezone.utc).isoformat(), approximation='close_to_close_estimated_costs')
    with (book_dir(book) / 'nav.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    print(json.dumps(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('snapshot', 'settle'))
    parser.add_argument('--asof')
    parser.add_argument('--to')
    parser.add_argument('--book', choices=BOOKS)
    parser.add_argument('--user-authorized', action='store_true')
    args = parser.parse_args()
    if args.stage == 'snapshot':
        stage_snapshot(args.asof, args.book, args.user_authorized)
    else:
        stage_settle(args.to, args.book, args.user_authorized)
