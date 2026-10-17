#!/usr/bin/env python3
"""Continuous two-leg staggered book. Research account only.

Does not change the frozen 6157c64 strategy, the production mandate, or any
previous experiment. Targets are frozen before the ledger runs.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import hashlib
import io
import json
import pickle
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab.backtest.ledger import Ledger, money
from quantlab.config import CONFIG
from quantlab.data.contracts import (
    InstrumentRules, MarketDataset, asof_records, decision_time, timestamp, timestamps_series,
)
from quantlab.data.local_adapter import canonical_code
from quantlab.data.market import universe_at
from quantlab.policy_comparison import account_checks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/stable_book_20260928'
MANIFEST = ROOT / 'output/account_2025_20260927/extension/dataset/manifest.json'
INPUTS = {
    '2018_2021': ROOT / 'output/reference_sustainability_20260927/2018_2021/decision_inputs.json',
    '2022_2025': ROOT / 'output/reference_sustainability_20260927/2022_2025/decision_inputs.json',
    '2025_recent': ROOT / 'output/reference_recent_20260927/inputs.json',
}
START, END = pd.Timestamp('2018-01-02'), pd.Timestamp('2026-09-22')
RECENT_START, RECENT_END = pd.Timestamp('2025-01-02'), pd.Timestamp('2026-09-22')
GROSS = 0.90
CASH_BUFFER = 0.10
LEG_A = GROSS * 0.70
LEG_B = GROSS * 0.30
SLEEVES = 5
HOLD_A, HOLD_B = 35, 15
STEP = 20
RETENTION_RANK = 150
PHASES = (0, 4, 8, 12, 16)
MIN_CAGR = 0.15
MAX_DRAWDOWN = 0.25
MAX_PHASE_GAP = 0.05
QUALITY_P2_CAGR = 0.006321
HOLD90_CAGR = 0.031791
INITIAL_CASH = 600_000.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    temporary.replace(path)


def prepare_fresh_output(output):
    """Create the only directory a new research preparation may write to.

    Research outputs are evidence, not scratch files.  In particular, the
    former default paths name archived experiments whose target and protocol
    identities must remain recoverable.  A new preparation therefore needs an
    explicitly empty output directory; it must never silently replace an
    archived protocol, target schedule, decision, or blocked record.
    """
    path = Path(output)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError("research_output_path_invalid")
        if any(path.iterdir()):
            raise ValueError("research_output_not_empty_refusing_archive_overwrite")
        return path
    path.mkdir(parents=True, exist_ok=False)
    return path


def set_explicit_output(value):
    """Select a new research destination without permitting path escape."""
    global OUT
    path = Path(value).expanduser().resolve()
    output_root = (ROOT / "output").resolve()
    if not path.is_relative_to(output_root):
        raise ValueError("research_output_must_be_below_output_root")
    OUT = path
    return OUT


def verify_protocol_sources(protocol):
    """Fail closed if an archived replay would run with changed source files."""
    sources = protocol.get("sources") if isinstance(protocol, dict) else None
    if not isinstance(sources, dict) or not sources:
        raise ValueError("protocol_sources_required")
    for path, digest in sources.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("protocol_source_identity_invalid")
        if sha256(path) != digest:
            raise ValueError("protocol_source_changed:" + path)


def liquidity_config():
    """Existing U5 qualification: 20-day amount, 60 sessions, exclude ST. Not a score."""
    cfg = copy.deepcopy(CONFIG)
    cfg.research.universe_mode = 'pit_liquidity'
    cfg.universe.a_exclude_st = True
    cfg.universe.a_min_listing_days = 60
    cfg.universe.a_min_avg_amount_wan = 5_000_000 / 10_000
    return cfg


def industry_slots():
    return max(1, int(np.ceil(HOLD_A * 0.15)))


def merge_band_inputs(parts):
    """Keep a date only when every source that has it agrees. Never fill a missing day."""
    heads, industries = {}, {}
    for name, part in parts.items():
        if set(part['heads']) != set(part['industries']):
            raise ValueError('head_industry_days_differ:' + name)
        for day, cross in part['heads'].items():
            group = part['industries'][day]
            if set(cross) != set(group):
                raise ValueError('head_industry_universe_mismatch:' + day)
            if day in heads and heads[day] != cross:
                raise ValueError('overlap_heads_mismatch:' + day)
            if day in industries and industries[day] != group:
                raise ValueError('overlap_industry_mismatch:' + day)
            heads[day] = cross
            industries[day] = group
    return heads, industries


def canon(code):
    text = str(code)
    if re.fullmatch(r'\d{6}\.(SH|SZ|BJ|sh|sz|bj)', text) or re.fullmatch(r'(SH|SZ|BJ|sh|sz|bj)\d{6}', text):
        return canonical_code(text)
    if re.fullmatch(r'(sh|sz|bj)\.\d{6}', text.lower()):
        return canonical_code(text)
    return text


def canonicalize_cross(cross):
    out = {}
    for code, value in cross.items():
        key = canon(code)
        if key in out and out[key] != value:
            raise ValueError('canonical_code_collision:' + key)
        out[key] = float(value)
    return out


def canonicalize_industry(group):
    out = {}
    for code, name in group.items():
        key = canon(code)
        if key in out and out[key] != name:
            raise ValueError('canonical_industry_collision:' + key)
        out[key] = name
    return out


def window_sessions(calendar, start=START, end=END):
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize()
    return calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]


def update_schedule(sessions, phase, step=STEP, sleeves=SLEEVES):
    if phase < 0 or phase >= step:
        raise ValueError('phase_outside_step')
    scheduled = []
    for index, day in enumerate(sessions):
        if index < phase or (index - phase) % step:
            continue
        scheduled.append((pd.Timestamp(day).normalize(), ((index - phase) // step) % sleeves))
    return scheduled


def latest_on_or_before(days, day):
    day = pd.Timestamp(day).normalize()
    prior = [item for item in days if pd.Timestamp(item) <= day]
    return max(prior, key=pd.Timestamp) if prior else None


def select_band_sleeve(scores, industries, current, locked, *, size=HOLD_A // SLEEVES,
                       retention_rank=RETENTION_RANK, slots=None):
    """Refresh one sleeve from the frozen binary tier. Other sleeves stay locked."""
    slots = industry_slots() if slots is None else slots
    if any(not np.isfinite(value) for value in scores.values()):
        raise ValueError('band_score_not_finite')
    # Equal scores keep the caller's order. Ticker text would lock the book
    # onto the lowest 000xxx.SZ codes.
    order = {code: index for index, code in enumerate(scores)}
    ranked = sorted(scores, key=lambda code: (-scores[code], order[code]))
    rank = {code: position for position, code in enumerate(ranked, start=1)}
    counts = Counter(industries.get(code, 'unclassified') for code in locked)
    chosen = []
    retained = [code for code in current if rank.get(code, retention_rank + 1) <= retention_rank]
    for code in list(retained) + ranked:
        if code in chosen or code in locked:
            continue
        industry = industries.get(code, 'unclassified')
        if counts[industry] >= slots:
            continue
        chosen.append(code)
        counts[industry] += 1
        if len(chosen) == size:
            break
    return tuple(chosen)


def select_liquid_sleeve(ranked, current, locked, *, size=HOLD_B // SLEEVES):
    """Highest already-qualified amount first. No score, low-vol, or calm filter."""
    del current
    chosen = []
    for code in ranked:
        if code in chosen or code in locked:
            continue
        chosen.append(code)
        if len(chosen) == size:
            break
    return tuple(chosen)


def _weights_for(names, budget):
    if not names:
        return {}
    share = budget / len(names)
    return {code: share for code in names}


class StaggeredBook:
    """Five sleeves per leg. Only the due sleeve changes membership or target weight."""

    def __init__(self):
        self.sleeves_a = [() for _ in range(SLEEVES)]
        self.sleeves_b = [() for _ in range(SLEEVES)]
        self.weights = {}
        self.industry = {}

    def _locked(self, sleeve, leg):
        groups = self.sleeves_a if leg == 'a' else self.sleeves_b
        other = self.sleeves_b if leg == 'a' else self.sleeves_a
        names = {code for index, group in enumerate(groups) if index != sleeve for code in group}
        names.update(code for group in other for code in group)
        return names

    def update(self, sleeve, heads, industries, liquid):
        if not 0 <= sleeve < SLEEVES:
            raise ValueError('sleeve_out_of_range')
        old_a, old_b = self.sleeves_a[sleeve], self.sleeves_b[sleeve]
        if heads is not None:
            self.industry.update(industries)
            remembered = {code: self.industry.get(code, 'unclassified') for code in self._locked(sleeve, 'a')}
            remembered.update(industries)
            self.sleeves_a[sleeve] = select_band_sleeve(
                heads, remembered, old_a, self._locked(sleeve, 'a'))
            for code in old_a:
                self.weights.pop(code, None)
            self.weights.update(_weights_for(self.sleeves_a[sleeve], LEG_A / SLEEVES))
        if liquid is not None:
            locked = self._locked(sleeve, 'b') | set(self.sleeves_a[sleeve])
            self.sleeves_b[sleeve] = select_liquid_sleeve(liquid, old_b, locked)
            for code in old_b:
                if code not in self.sleeves_a[sleeve]:
                    self.weights.pop(code, None)
            self.weights.update(_weights_for(self.sleeves_b[sleeve], LEG_B / SLEEVES))
        active = set()
        if heads is not None:
            active.update(self.sleeves_a[sleeve])
        if liquid is not None:
            active.update(self.sleeves_b[sleeve])
        sell = [code for code in (*old_a, *old_b) if code not in self.weights]
        if sum(self.weights.values()) > GROSS + 1e-9:
            raise ValueError('book_exceeds_executable_gross')
        return {
            'weights': dict(self.weights),
            'active': sorted(active),
            'sell': sorted(set(sell)),
            'holdings': len(self.weights),
        }


def build_targets(sessions, heads, industries, liquid_by_day, phase):
    book = StaggeredBook()
    head_days = list(heads)
    targets = []
    for day, sleeve in update_schedule(sessions, phase):
        key = str(day.date())
        snapshot = latest_on_or_before(head_days, day)
        band = industries_now = None
        if snapshot is not None:
            band = heads[snapshot]
            industries_now = industries[snapshot]
        liquid = liquid_by_day.get(key)
        if band is None and not liquid:
            continue
        state = book.update(sleeve, band, industries_now or {}, liquid)
        if not state['weights'] and not state['sell']:
            continue
        targets.append({
            'signal_date': key,
            'weights': state['weights'],
            'active': state['active'],
            'sell': state['sell'],
            'sell_only': False,
            'max_exposure': GROSS,
            'sleeve': sleeve,
            'phase': phase,
            'band_snapshot': None if snapshot is None else str(pd.Timestamp(snapshot).date()),
        })
    return targets


def execution_weights(intended, active, sell, quantities, marks, nav, gross=GROSS):
    """Share-lock sleeves that are not due. Size only the sleeve being updated."""
    if nav <= 0 or not np.isfinite(nav):
        raise ValueError('invalid_nav')
    active, sell = set(active), set(sell)
    frozen_value = 0.0
    weights = {}
    for code, quantity in quantities.items():
        if code in active or code in sell or quantity <= 0:
            continue
        price = marks.get(code)
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        weight = quantity * price / nav
        weights[code] = weight
        frozen_value += weight
    active_weights = {code: intended[code] for code in active if intended.get(code, 0.0) > 0}
    active_sum = sum(active_weights.values())
    room = max(0.0, gross - frozen_value)
    scale = 1.0 if active_sum <= room + 1e-12 else (room / active_sum if active_sum else 1.0)
    for code, weight in active_weights.items():
        weights[code] = weight * scale
    for code in sell:
        weights[code] = 0.0
    if sum(weight for weight in weights.values() if weight > 0) > gross + 1e-8:
        raise ValueError('execution_weights_exceed_gross')
    return weights, scale


def phase_names():
    return [f'phase{phase}' for phase in PHASES]


def judge(cost1, cost2):
    """Median of five stagger clocks. The best clock is not a result."""
    names = phase_names()
    if set(names) != set(cost1) or set(names) != set(cost2):
        raise ValueError('phase_set_mismatch')
    cagrs = [cost1[name]['cagr'] for name in names]
    ordered = sorted(cagrs)
    median = ordered[len(ordered) // 2]
    span = max(cagrs) - min(cagrs)
    reasons = []
    if median < MIN_CAGR:
        reasons.append('median_cagr')
    if span > MAX_PHASE_GAP + 1e-12:
        reasons.append('phase_range')
    for name in names:
        row = cost1[name]
        if -row['max_drawdown'] > MAX_DRAWDOWN + 1e-12:
            reasons.append(name + '_drawdown')
        if row['recent_return'] <= 0:
            reasons.append(name + '_recent')
    cost_ok = all(cost2[name]['cagr'] > 0 for name in names)
    if reasons:
        decision = 'combination_did_not_reach_stable_15'
    elif not cost_ok:
        decision = 'not_retained_cost_double_not_positive'
    else:
        decision = 'research_candidate_not_promoted'
    return {
        'decision': decision,
        'reasons': reasons,
        'median_cagr': median,
        'phase_range': span,
        'cost1_cagr': {name: cost1[name]['cagr'] for name in names},
        'cost_double_positive': cost_ok,
        'promotion_approved': False,
        'versus_quality_p2': median - QUALITY_P2_CAGR,
        'versus_hold90': median - HOLD90_CAGR,
    }


def _read_checked(root, entry):
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('dataset_path_escape')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry['sha256']:
        raise ValueError('checksum_mismatch:' + entry['path'])
    return raw


def _dated(frame):
    if 'event_time' in frame:
        frame = frame.set_index('event_time')
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    return frame.sort_index()


def load_qualification_view(manifest_path=MANIFEST):
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    calendar = pd.DatetimeIndex(_dated(pd.read_parquet(io.BytesIO(_read_checked(root, manifest['calendar'])))).session)
    instruments = pd.read_parquet(io.BytesIO(_read_checked(root, manifest['instruments'])))
    panels, amounts, late = {}, {}, 0
    columns = ['amount', 'close', 'available_at', 'quality_status', 'is_st_at_open']
    for code, entry in manifest['panels'].items():
        frame = _dated(pd.read_parquet(io.BytesIO(_read_checked(root, entry)), columns=columns))
        frame = frame[~frame.index.duplicated(keep='last')]
        available = timestamps_series(frame['available_at'])
        available.index = frame.index
        cutoff = pd.Series(pd.DatetimeIndex(frame.index).tz_localize('Asia/Shanghai') + pd.Timedelta(hours=18),
                           index=frame.index)
        bad = available.isna() | available.gt(cutoff) | frame['quality_status'].astype(str).ne('ok')
        late += int(bad.sum())
        amount = pd.to_numeric(frame.amount, errors='coerce').mask(bad)
        amount.index = frame.index
        amounts[code] = amount
        panels[code] = frame[['close', 'is_st_at_open']]
    amount_frame = pd.DataFrame(amounts).reindex(calendar)
    dataset = MarketDataset(panels, instruments, calendar, manifest=manifest)
    return dataset, amount_frame, late


def liquid_ranks(dataset, amount, days, cfg=None):
    cfg = cfg or liquidity_config()
    ranks = {}
    for index, day in enumerate(days):
        if index % 20 == 0:
            print(json.dumps({'stage': 'liquidity', 'day': str(pd.Timestamp(day).date()), 'index': index, 'total': len(days)}), flush=True)
        eligible, _records = universe_at(dataset, pd.Timestamp(day), amount, cfg)
        recent = amount.loc[:pd.Timestamp(day)].tail(20)
        average = recent.mean()
        ordered = sorted(eligible, key=lambda code: (-float(average.get(code, -np.inf) if np.isfinite(average.get(code, np.nan)) else -np.inf), code))
        ranks[str(pd.Timestamp(day).date())] = ordered[:200]
    return ranks


def _rules(records, codes=None):
    """Build execution rules for ``codes`` only. Building all ~5,500 instruments per day is the
    replay's dominant cost and the ledger only ever asks for held, pending or target names."""
    if codes is not None:
        wanted = [code for code in codes if code in records.index]
        records = records.loc[wanted]
    result = {}
    for code, row in records.iterrows():
        try:
            rule = InstrumentRules.from_record(row)
        except ValueError:
            continue
        if rule.asset_type == 'stock':
            result[code] = rule
    return result


def data_quality(snapshots, missing, data_issues, checks_passed, max_stale_days=5):
    """Summarize whether the replay is usable as evidence, not merely whether it ran."""
    issues_by_reason = dict(Counter(str(issue.get('reason', 'unknown')) for issue in data_issues))
    running, longest = {}, {}
    for state in snapshots:
        stale = set(state.get('stale') or [])
        for code in list(running):
            if code not in stale:
                running.pop(code)
        for code in stale:
            running[code] = running.get(code, 0) + 1
            longest[code] = max(longest.get(code, 0), running[code])
    stale_held_max_days = max(longest.values(), default=0)
    stale_held_codes = {code: int(days) for code, days in longest.items() if days > max_stale_days}
    unsettled = any(issue.get('reason') == 'delisting_settlement_unverified' for issue in data_issues)
    return dict(data_issues_by_reason=issues_by_reason, stale_held_max_days=int(stale_held_max_days),
                stale_held_codes=stale_held_codes,
                missing_panel_count=len(missing), unsettled_delisting=unsettled,
                data_quality_ok=bool(checks_passed and not missing and stale_held_max_days <= max_stale_days
                                     and not unsettled))


def replay_account(dataset, targets, *, start, end, cost_mult=1.0, delisting_haircut=0.0):
    if isinstance(targets, (str, Path)):
        payload = json.loads(Path(targets).read_text())
        rows = payload['targets']
    else:
        rows = targets
    schedule = {}
    for row in rows:
        day = pd.Timestamp(row['signal_date']).normalize()
        if day in schedule:
            raise ValueError('duplicate_signal_date')
        weights = {canon(code): float(weight) for code, weight in row['weights'].items()}
        if len(weights) > HOLD_A + HOLD_B or sum(weights.values()) > GROSS + 1e-8:
            raise ValueError('target_breaks_position_or_gross_cap')
        schedule[day] = dict(row, weights=weights,
                             active=[canon(code) for code in row['active']],
                             sell=[canon(code) for code in row['sell']])
    calendar = dataset.calendar
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if dates.empty:
        raise ValueError('empty_replay_window')
    codes = sorted({code for row in schedule.values() for code, weight in row['weights'].items() if weight > 0}
                   | {code for row in schedule.values() for code in row['sell']})
    missing = sorted(set(codes) - set(dataset.panels))
    amount = pd.DataFrame({code: dataset.panels[code].amount for code in codes if code in dataset.panels}).reindex(calendar)
    prior_adv = amount.rolling(20, min_periods=20).mean().shift(1)
    ledger = Ledger(INITIAL_CASH, slippage=0.003, cash_buffer=CASH_BUFFER, participation=0.01,
                    cost_mult=cost_mult, min_trade_value=1000.0, allow_open_assumption=True,
                    delistings=getattr(dataset, 'delistings', None), delisting_haircut=delisting_haircut)
    actions = dataset.actions
    actions_by_date = {}
    if actions is not None and not actions.empty:
        subset = actions[actions.instrument_id.isin(codes)]
        if not subset.empty:
            actions_by_date = dict(tuple(subset.groupby(pd.to_datetime(subset.ex_date).dt.normalize())))
    gaps_by_date = {}
    for gap in dataset.manifest.get('data_gaps', []):
        if gap.get('instrument_id') in set(codes):
            gaps_by_date.setdefault(pd.Timestamp(gap['date']).normalize(), []).append(gap)
    traces = []
    for day in dates:
        session = int(calendar.get_loc(day))
        relevant = set(ledger.quantities()) | {order.instrument_id for order in ledger.orders
                                                if order.status in ('pending', 'partial')}
        row = schedule.get(day)
        if row is not None:
            relevant.update(code for code, weight in row['weights'].items() if weight > 0)
            relevant.update(row['sell'])
        records = asof_records(dataset.instruments, timestamp(day) + pd.Timedelta(hours=9, minutes=25))
        rules = _rules(records, relevant)
        bars = {code: dataset.panels[code].loc[day].to_dict() for code in relevant
                if code in dataset.panels and day in dataset.panels[code].index}
        gap_issues = [dict(gap, held_shares=ledger.quantity(gap['instrument_id']))
                      for gap in gaps_by_date.get(day, []) if ledger.quantity(gap['instrument_id'])]
        ledger.corporate_actions(day, session, actions_by_date.get(day), calendar)
        adv = prior_adv.loc[day].to_dict() if day in prior_adv.index else {}

        def can_buy(code, quantity, price, fees, account, held_limit=HOLD_A + HOLD_B):
            if code not in records.index:
                return False
            active = records.loc[code].get('active')
            if pd.isna(active) or not bool(active):
                return False
            held = account.quantities()
            if code not in held and len(held) >= held_limit:
                return False
            values = {name: shares * account.marks[name] for name, shares in held.items()}
            values[code] = values.get(code, 0.0) + quantity * price
            return sum(values.values()) <= GROSS * (account.nav() - fees) + 1e-6

        ledger.execute(day, session, bars, rules, adv, can_buy=can_buy)
        for order in ledger.orders:
            if order.status == 'expired' and order.remaining > 0 and order.reason == 'stable_book_rebalance':
                order.status = 'partial' if order.remaining < order.quantity else 'pending'
        closing = {}
        for code, bar in bars.items():
            close = bar.get('close', np.nan)
            available = bar.get('available_at')
            if (bar.get('quality_status') == 'ok' and np.isfinite(close) and close > 0
                    and available is not None and not pd.isna(available)
                    and timestamp(available) <= decision_time(day)):
                closing[code] = close
        settled = set(ledger.settle_delistings(day, closing))
        for issue in gap_issues:
            if issue['instrument_id'] not in settled:
                ledger.data_issues.append(issue)
        ledger.close(day, session, closing)
        if row is None:
            continue
        close_rules = _rules(asof_records(dataset.instruments, decision_time(day)),
                             set(row['weights']) | set(row['sell']))
        weights, scale = execution_weights(row['weights'], row['active'], row['sell'],
                                           ledger.quantities(), ledger.marks, ledger.nav(), gross=GROSS)
        begin = len(ledger.orders)
        ledger.queue_targets(weights, day, 'stable_book_rebalance', close_rules)
        for order in ledger.orders[begin:]:
            order.risk_policy = 'none'
            order.candidate = 'stable_book'
            order.industry_policy = 'band_sleeve_only'
        traces.append({'date': str(day.date()), 'sleeve': row['sleeve'], 'active_scale': scale,
                       'target_holdings': row['holdings'] if 'holdings' in row else len(row['weights']),
                       'orders': len(ledger.orders) - begin})
    snapshots = ledger.snapshots
    nav = pd.Series({pd.Timestamp(state['date']): state['nav'] for state in snapshots}).sort_index()
    exposure = pd.Series({
        pd.Timestamp(state['date']): (sum(shares * state['marks'].get(code, 0.0)
                                          for code, shares in state['quantities'].items()) / state['nav']
                                      if state['nav'] else 0.0)
        for state in snapshots
    }).sort_index()
    returns = nav.pct_change(fill_method=None)
    returns.iloc[0] = nav.iloc[0] / INITIAL_CASH - 1
    elapsed = (nav.index[-1] - nav.index[0]).days
    base = nav.loc[nav.index < RECENT_START]
    end = nav.loc[nav.index <= RECENT_END]
    recent = float(end.iloc[-1] / base.iloc[-1] - 1) if len(base) and len(end) else float('nan')
    drawdown = float((nav / nav.cummax().clip(lower=INITIAL_CASH) - 1).min())
    checks = account_checks({'account': [
        {'date': state['date'], 'nav': state['nav'], 'cash': state['cash'], 'quantities': state['quantities'],
         'marks': state['marks'], 'receivables': state['receivables'], 'payables': 0.0}
        for state in snapshots
    ], 'fills': ledger.fills})
    quality = data_quality(snapshots, missing, ledger.data_issues, checks['passed'])
    delisting_fills = [fill for fill in ledger.fills if fill.get('reason') == 'delisting']
    summary = {
        'initial_cash': INITIAL_CASH,
        'final_equity': float(nav.iloc[-1]),
        'net_profit': float(nav.iloc[-1] - INITIAL_CASH),
        'cagr': float((nav.iloc[-1] / INITIAL_CASH) ** (365.25 / elapsed) - 1),
        'max_drawdown': drawdown,
        'recent_return': recent,
        'recent_base': str(base.index[-1].date()) if len(base) else None,
        'recent_end': str(end.index[-1].date()) if len(end) else None,
        'fees': float(sum(fill['fees'] for fill in ledger.fills)),
        'slippage': float(sum(fill['slippage_cost'] for fill in ledger.fills)),
        'fills': len(ledger.fills),
        'max_holdings': max((len(state['quantities']) for state in snapshots), default=0),
        'mean_exposure': float(exposure.mean()),
        'min_exposure': float(exposure.min()),
        'max_exposure': float(exposure.max()),
        'mean_cash_weight': float(1 - exposure.mean()),
        'data_issues': len(ledger.data_issues),
        **quality,
        'delisting_liquidations': len(delisting_fills),
        'account_checks_passed': checks['passed'],
        'account_check_errors': checks['errors'],
        'sessions': len(nav),
        'first': str(nav.index[0].date()),
        'last': str(nav.index[-1].date()),
        'cost_multiplier': cost_mult,
        'promotion_approved': False,
        'missing_panels': missing,
    }
    result = dict(summary=summary, nav=nav, exposure=exposure, daily_return=returns,
                  signal_trace=traces, **ledger.export())
    return result


def _band_maps(parts):
    raw_heads, raw_industries = merge_band_inputs(parts)
    heads, industries = {}, {}
    for day, cross in raw_heads.items():
        heads[day] = canonicalize_cross(cross)
        industries[day] = canonicalize_industry(raw_industries[day])
    return heads, industries


def prepare():
    prepare_fresh_output(OUT)
    parts = {name: json.loads(path.read_text()) for name, path in INPUTS.items()}
    try:
        heads, industries = _band_maps(parts)
    except ValueError as exc:
        write_json(OUT / 'BLOCKED.json', {'reason': str(exc), 'stage': 'band_inputs', 'promotion_approved': False})
        return 'blocked'
    known_panels = set(json.loads(MANIFEST.read_text())['panels'])
    mapped = [code in known_panels for cross in heads.values() for code in cross]
    if not mapped or sum(mapped) / len(mapped) < 0.5:
        write_json(OUT / 'BLOCKED.json', {
            'reason': 'band_codes_do_not_match_panel_ids',
            'coverage': 0 if not mapped else sum(mapped) / len(mapped),
            'promotion_approved': False,
        })
        return 'blocked'
    dataset, amount, late_rows = load_qualification_view()
    sessions = window_sessions(dataset.calendar)
    if sessions.empty or sessions[0] != START or sessions[-1] < END:
        write_json(OUT / 'BLOCKED.json', {
            'reason': 'window_not_covered_by_calendar',
            'first': None if sessions.empty else str(sessions[0].date()),
            'last': None if sessions.empty else str(sessions[-1].date()),
            'promotion_approved': False,
        })
        return 'blocked'
    if sessions[-1] != END:
        write_json(OUT / 'BLOCKED.json', {'reason': 'end_date_missing', 'last': str(sessions[-1].date()),
                                          'promotion_approved': False})
        return 'blocked'
    days = sorted({day for phase in PHASES for day, _sleeve in update_schedule(sessions, phase)})
    ranks = liquid_ranks(dataset, amount, days)
    covered = [day for day in days if pd.Timestamp(day) >= pd.Timestamp('2019-03-01')]
    ready = [day for day in covered if len(ranks[str(pd.Timestamp(day).date())]) >= HOLD_B]
    if len(ready) * 2 < len(covered):
        write_json(OUT / 'BLOCKED.json', {
            'reason': 'high_liquidity_eligibility_not_reconstructed',
            'update_days_after_2019_03_01': len(covered),
            'days_with_15_names': len(ready),
            'late_amount_rows': late_rows,
            'promotion_approved': False,
        })
        return 'blocked'
    write_json(OUT / 'liquid_ranks.json', ranks)
    targets = {}
    for phase in PHASES:
        rows = build_targets(sessions, heads, industries, ranks, phase)
        path = OUT / f'targets_phase{phase}.json'
        write_json(path, {'targets': rows, 'phase': phase, 'promotion_approved': False})
        targets[str(path)] = sha256(path)
    protocol = {
        'version': 'stable-book-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'question': 'Does this new two-leg book, not the old single basket, have a five-clock median net CAGR of at least 15 percent with a CAGR range of at most 5 percentage points?',
        'construction': 'new_book',
        'not_a_verification_of': 'continuous single broad basket, and not B_top1500_calm_s3',
        'window': [str(START.date()), str(END.date())],
        'initial_cash': INITIAL_CASH,
        'executable_gross': GROSS,
        'gross_reason': 'Shared ledger rejects exposure above 0.90. 70/30 splits that gross into 0.63 and 0.27. It is not a return parameter.',
        'legs': {
            'A': {'share_of_gross': 0.70, 'holdings': HOLD_A, 'sleeves': SLEEVES,
                  'pool': 'frozen ADV rank 3001-5000 heads',
                  'order': 'binary profit/PB then frozen input liquidity order',
                  'retention_rank': RETENTION_RANK, 'industry_slots': industry_slots()},
            'B': {'share_of_gross': 0.30, 'holdings': HOLD_B, 'sleeves': SLEEVES,
                  'pool': 'U5 qualified names ordered by 20-day mean amount',
                  'qualification': 'pit_liquidity, ADV>=5_000_000, listing_sessions>=60, exclude ST',
                  'not_the_external_leg': 'This is not B_top1500_calm_s3. That leg is top-1500 names with amount at least 20 million and a calm_core two-stage score in three sleeves. No external phase-sensitivity number applies here.'},
        },
        'stagger': {'step_sessions': STEP, 'phases': list(PHASES), 'hold_sessions': STEP * SLEEVES,
                    'decision': 'median of the five cost-1 CAGRs; their max-minus-min must be within 5 percentage points; the best phase is not used',
                    'immature_sleeves': 'share-locked until their own update',
                    'negative_history': 'no switch to cash; allocation ignores returns'},
        'risk': 'no P2, no drawdown scaler, no single-name stop',
        'costs': {'slippage': 0.003, 'min_trade_value': 1000, 'commission': 'instrument rule',
                  'cost_multipliers': [1, 2]},
        'gates': {'min_cagr': MIN_CAGR, 'max_drawdown': MAX_DRAWDOWN, 'max_phase_gap': MAX_PHASE_GAP,
                  'recent_positive': [str(RECENT_START.date()), str(RECENT_END.date())],
                  'cost_double_cagr_positive': True},
        'published_baselines': {'qualityP2_cagr': QUALITY_P2_CAGR, 'hold90_cagr': HOLD90_CAGR},
        'late_amount_rows_excluded': late_rows,
        'liquid_days': len(days),
        'liquid_days_with_15': len(ready),
        'band_days': len(heads),
        'inputs': {name: sha256(path) for name, path in INPUTS.items()},
        'manifest': sha256(MANIFEST),
        'sources': {str(path): sha256(path) for path in (Path(__file__), ROOT / 'tests/test_stable_book.py')},
        'targets': targets,
        'promotion_approved': False,
        'production_mandate_unchanged': True,
        'frozen_source_unchanged': '6157c64c84717dc8645dde3fe6228590b9f8538e',
    }
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'stage': 'prepared', 'band_days': len(heads), 'liquid_days': len(days),
                      'late_rows': late_rows}, ensure_ascii=False), flush=True)
    return 'prepared'


def _verify_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name, digest in protocol['inputs'].items():
        if sha256(INPUTS[name]) != digest:
            raise ValueError('protocol_input_changed:' + name)
    if sha256(MANIFEST) != protocol['manifest']:
        raise ValueError('protocol_manifest_changed')
    verify_protocol_sources(protocol)
    for path, digest in protocol['targets'].items():
        if sha256(path) != digest:
            raise ValueError('targets_changed_after_freeze:' + path)
    if protocol['stagger']['phases'] != list(PHASES):
        raise ValueError('protocol_phase_set_mismatch')
    return protocol


def run():
    if (OUT / 'BLOCKED.json').exists() and not (OUT / 'protocol.json').exists():
        print(json.dumps({'stage': 'run_skipped_blocked'}), flush=True)
        return 'blocked'
    protocol = _verify_protocol()
    root = MANIFEST.parent
    manifest = json.loads(MANIFEST.read_text())
    codes = sorted({canon(code) for path in protocol['targets']
                    for row in json.loads(Path(path).read_text())['targets']
                    for code in row['weights']})
    absent = sorted(set(codes) - set(manifest['panels']))
    if absent:
        write_json(OUT / 'BLOCKED.json', {'reason': 'target_panel_missing', 'count': len(absent),
                                          'sample': absent[:10], 'promotion_approved': False})
        return 'blocked'
    calendar = pd.DatetimeIndex(_dated(pd.read_parquet(io.BytesIO(_read_checked(root, manifest['calendar'])))).session)
    instruments = pd.read_parquet(io.BytesIO(_read_checked(root, manifest['instruments'])))
    actions = pd.read_parquet(io.BytesIO(_read_checked(root, manifest['actions'])))
    benchmark = _dated(pd.read_parquet(io.BytesIO(_read_checked(root, manifest['benchmark']))))
    panels = {code: _dated(pd.read_parquet(io.BytesIO(_read_checked(root, manifest['panels'][code])))) for code in codes}
    dataset = MarketDataset(panels, instruments, calendar, actions, benchmark, manifest=manifest)
    errors = dataset.validate('free_research')
    if errors:
        write_json(OUT / 'BLOCKED.json', {'reason': 'execution_projection_invalid', 'errors': errors,
                                          'promotion_approved': False})
        return 'blocked'
    for phase in PHASES:
        for multiplier in (1, 2):
            folder = OUT / f'phase{phase}_cost{multiplier}'
            print(json.dumps({'stage': 'replay', 'phase': phase, 'cost': multiplier}), flush=True)
            result = replay_account(dataset, OUT / f'targets_phase{phase}.json',
                                    start=START, end=END, cost_mult=multiplier)
            if not result['summary']['account_checks_passed']:
                write_json(folder / 'summary.json', result['summary'])
                write_json(OUT / 'BLOCKED.json', {'reason': 'account_identity_failed', 'phase': phase,
                                                  'cost': multiplier, 'errors': result['summary']['account_check_errors'],
                                                  'promotion_approved': False})
                return 'blocked'
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'result.pkl').open('wb') as handle:
                pickle.dump({'summary': result['summary'], 'fills': result['fills'],
                             'account': result['account'], 'data_issues': result['data_issues']}, handle, protocol=4)
            write_json(folder / 'summary.json', result['summary'])
            result['daily_return'].rename('daily_return').to_csv(folder / 'daily_returns.csv', index_label='date')
            print(json.dumps({'stage': 'replay_done', 'phase': phase, 'cost': multiplier,
                              'cagr': result['summary']['cagr'], 'max_drawdown': result['summary']['max_drawdown'],
                              'recent_return': result['summary']['recent_return']}), flush=True)
    return 'ran'


def decide():
    blocked = OUT / 'BLOCKED.json'
    if blocked.exists() and not (OUT / 'phase0_cost1' / 'summary.json').exists():
        payload = json.loads(blocked.read_text())
        text = _continuity_block(payload)
        _append_continuity(text)
        write_json(OUT / 'DECISION.json', dict(payload, decision='not_run_inputs_or_ledger', promotion_approved=False))
        (OUT / 'REVIEW.md').write_text(text)
        print(json.dumps({'stage': 'decided', 'decision': 'not_run'}, ensure_ascii=False), flush=True)
        return 'not_run'
    protocol = _verify_protocol()
    cost1, cost2 = {}, {}
    for phase in PHASES:
        cost1[f'phase{phase}'] = json.loads((OUT / f'phase{phase}_cost1' / 'summary.json').read_text())
        cost2[f'phase{phase}'] = json.loads((OUT / f'phase{phase}_cost2' / 'summary.json').read_text())
    verdict = judge(cost1, cost2)
    verdict['protocol'] = protocol['version']
    write_json(OUT / 'DECISION.json', verdict)
    review = _review(cost1, cost2, verdict)
    (OUT / 'REVIEW.md').write_text(review)
    _append_continuity(review)
    print(json.dumps({'stage': 'decided', 'decision': verdict['decision'], 'reasons': verdict['reasons']},
                     ensure_ascii=False), flush=True)
    return verdict['decision']


def _review(cost1, cost2, verdict):
    def line(name, row):
        return (f"- {name}：扣费年化 {row['cagr'] * 100:.2f}%，最大回撤 {-row['max_drawdown'] * 100:.2f}%，"
                f"2025-01-02 至 2026-09-22 累计 {row['recent_return'] * 100:.2f}%，"
                f"期末 {row['final_equity']:.2f} 元，费用 {row['fees']:.2f}，滑点 {row['slippage']:.2f}，"
                f"成交 {row['fills']}，最多 {row['max_holdings']} 只，数据问题 {row['data_issues']}。")
    rows = [line(name, cost1[name]) for name in phase_names()]
    doubled = [line(name + ' 成本翻倍', cost2[name]) for name in phase_names()]
    gaps = (f"五条成本 1 倍时钟的中位年化是 {verdict['median_cagr'] * 100:.2f}%，"
            f"相对已公布 qualityP2 的 0.6321% 高 {verdict['versus_quality_p2'] * 100:.2f} 个百分点，"
            f"相对 hold90 的 3.1791% 高 {verdict['versus_hold90'] * 100:.2f} 个百分点。"
            f"极差 {verdict['phase_range'] * 100:.2f} 个百分点。不采用五条里最高的那条。")
    outcome = {
        'research_candidate_not_promoted': '五条时钟的中位年化、极差、回撤、末段和成本翻倍都过了预先写死的线。这是研究候选，不是可交易许可。',
        'not_retained_cost_double_not_positive': '中位和极差过了，但五条成本翻倍年化没有全部为正。候选不保留。',
        'combination_did_not_reach_stable_15': '这条新组合没有同时达到中位年化 15% 和五条时钟极差不超过 5 个百分点。不改配比、只数或小组长度，也不改用其中最好的一条。',
    }[verdict['decision']]
    return '\n'.join([
        '# 连续双腿错峰账户',
        '',
        '这是一条新组合，不是把旧宽篮连续持有，也不是外部项目的 B_top1500_calm_s3。腿 B 只按 U5 合格股票的 20 日成交额从高到低取 15 只，没有 calm 过滤。外部相位敏感度数字不能引用到这里。',
        '',
        f"判决：`{verdict['decision']}`。{outcome}",
        f"未通过项：{('、'.join(verdict['reasons']) if verdict['reasons'] else '无')}。",
        gaps,
        '',
        '成本 1 倍',
        *rows,
        '',
        '成本 2 倍',
        *doubled,
        '',
        '账户是 60 万、单边 30bp、2018-01-02 至 2026-09-22 一条账。错峰时钟是 0、4、8、12、16 五个交易日。判决用五条年化的中位数和极差。腿 A 为冻结宽篮等权 35 只的 70%，腿 B 为 U5 合格股票里 20 日成交额最高的 15 只的 30%，比例相对可投资上限 90%。每 20 个交易日只换一个小组，未到期小组不重平衡。没有 P2、回撤缩仓和单股止损。生产委托和冻结源码没有改。promotion_approved=false。',
        '',
    ])


def _continuity_block(payload):
    return '\n'.join([
        '# 连续双腿错峰账户未运行',
        '',
        f"输入或账本核对没有通过，没有回放账户。原因：`{payload.get('reason')}`。",
        '不插补名单，不改生产委托。promotion_approved=false。',
        '',
    ])


def _append_continuity(text):
    path = ROOT / 'docs/RESEARCH_CONTINUITY.md'
    existing = path.read_text()
    marker = '## 连续双腿错峰账户（2026-09-28'
    if marker in existing:
        raise ValueError('continuity_section_already_appended')
    stamp = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M CST')
    section = (
        f"\n\n## 连续双腿错峰账户（2026-09-28，{stamp}，仅追加）\n\n"
        '研究账，不是生产策略。协议 `output/stable_book_20260928/protocol.json`，'
        '判决 `output/stable_book_20260928/DECISION.json`。冻结源码 6157c64 与生产委托未改。\n\n'
        + text
    )
    path.write_text(existing.rstrip() + section)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'decide'))
    parser.add_argument('--out', help='explicit fresh directory below output/ for a new research artifact')
    args = parser.parse_args()
    if args.out:
        set_explicit_output(args.out)
    globals()[args.action]()


if __name__ == '__main__':
    main()
