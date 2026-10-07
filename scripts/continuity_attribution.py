#!/usr/bin/env python3
"""Read-only historical attribution; never runs the account or optimizes a policy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'output/frozen_source/6157c64'
RESEARCH = ROOT / 'output/dynamic_risk/0e4aeb60bbd59c257851f53e'
MANIFEST = ROOT / 'data/local_market/1c85fd9aceaa03c3c4863bec/manifest.json'
OUT = ROOT / 'output/continuity_review_20260927/attribution'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def date(value):
    return str(pd.Timestamp(value).date())


def forward_outcome(prices, calendar, start, horizon, gaps=()):
    """Adjusted-feature-price outcome, not cash-account profit; no gap filling."""
    start = pd.Timestamp(start)
    i = calendar.get_indexer([start])[0]
    if i < 0:
        return None, 'non_session_start'
    if i + horizon >= len(calendar):
        return None, 'end_censored'
    end = calendar[i + horizon]
    if any(start < pd.Timestamp(g) <= end for g in gaps):
        return None, 'known_data_gap'
    path = prices.reindex(calendar[i:i + horizon + 1])
    if not np.isfinite(path).all() or (path <= 0).any():
        return None, 'missing_price_path'
    return float(path.iloc[-1] / path.iloc[0] - 1), 'observed'


def classify_fills(run):
    """Realized day/security transitions, not a reconstruction of intent.

    Partial fills on later days are classified by that day's beginning/end
    position. Risk reason has precedence; repurchases are exclusive additions.
    """
    states = {date(s['date']): s['quantities'] for s in run['account']}
    before, prior = {}, {}
    for d in sorted(states):
        before[d], prior = prior, states[d]
    grouped = defaultdict(list)
    for fill in run['fills']:
        grouped[(date(fill['date']), fill['instrument_id'])].append(fill)
    last_exit, rows = {}, []
    for (d, code), fills in sorted(grouped.items()):
        old, new = before[d].get(code, 0), states[d].get(code, 0)
        for fill in fills:
            side, reason = fill['side'], fill['reason']
            if reason == 'risk_exit':
                category = 'risk_exit'
            elif side == 'buy' and old == 0:
                category = 'repurchase' if code in last_exit else 'addition'
            elif side == 'buy':
                category = 'weight_increase'
            elif new == 0:
                category = 'removal'
            else:
                category = 'weight_decrease'
            previous = last_exit.get(code, {})
            rows.append(dict(date=d, instrument_id=code, side=side, category=category,
                             reason=reason, notional=float(fill['notional']),
                             fees=float(fill['fees']), slippage=float(fill.get('slippage_cost', 0)),
                             cost=float(fill['fees']) + float(fill.get('slippage_cost', 0)),
                             previous_exit_date=previous.get('date'),
                             previous_exit_reason=previous.get('reason')))
        if old > 0 and new == 0:
            last_exit[code] = dict(date=d, reason='risk_exit' if any(
                f['reason'] == 'risk_exit' for f in fills) else 'ordinary')
    return rows


def score_buckets(scores, count=5):
    # Equal scores must never gain an artificial ranking from security code.
    return np.ceil(scores.rank(pct=True, method='average') * count).astype(int).clip(1, count)


def risk_buybacks(trades, run, calendar):
    """Overlapping risk-sell -> first later-session buy links, never new costs."""
    states = {date(x['date']): x['quantities'] for x in run['account']}
    sessions = {date(d): i for i, d in enumerate(calendar)}
    rows = []
    for sell_id, sell in trades[trades.category.eq('risk_exit')].iterrows():
        buys = trades[trades.instrument_id.eq(sell.instrument_id) & trades.side.eq('buy')
                      & trades.date.gt(sell.date)].sort_values('date', kind='stable')
        lag = sessions[buys.date.iloc[0]] - sessions[sell.date] if len(buys) else None
        rows.append(dict(risk_event_id=int(sell_id), instrument_id=sell.instrument_id, risk_date=sell.date,
                         risk_exit_type='partial_reduction' if states[sell.date].get(sell.instrument_id, 0) > 0 else 'full_exit',
                         buy_id=int(buys.index[0]) if len(buys) else None,
                         buy_date=buys.date.iloc[0] if len(buys) else None,
                         buy_category=buys.category.iloc[0] if len(buys) else None,
                         first_buy_same_day_fill_count=int(buys.date.eq(buys.date.iloc[0]).sum()) if len(buys) else 0,
                         same_day_buy_excluded_count=int((trades.instrument_id.eq(sell.instrument_id) & trades.side.eq('buy') & trades.date.eq(sell.date)).sum()),
                         lag_sessions=lag, within_20=lag is not None and lag <= 20,
                         within_60=lag is not None and lag <= 60,
                         followup_sessions=len(calendar) - 1 - sessions[sell.date],
                         observed_20=lag is not None and lag <= 20 or len(calendar) - 1 - sessions[sell.date] >= 20,
                         observed_60=lag is not None and lag <= 60 or len(calendar) - 1 - sessions[sell.date] >= 60))
    return pd.DataFrame(rows)


def cash_labels(row, issues, data_block, eligible_count):
    labels = []
    if row['manual_review'] or row['level'] == 4:
        labels.append('L4_manual_stop')
    if row['flat'] and row['level'] == 3 and not row['allow_new_stock']:
        labels.append('L3_recovery_lock')
    if any('constraint' in x or 'solver' in x or 'covariance' in x for x in issues):
        labels.append('constraint_conflict')
    if data_block:
        labels.append('data_block')
    if eligible_count == 0 or 'no_candidates' in issues:
        labels.append('no_candidates')
    return labels


def load_archived(record):
    directory = RESEARCH / 'runs' / record['run_id']
    hashes = json.loads((directory / 'artifacts.json').read_text())['sha256']
    for name, expected in hashes.items():
        if sha(directory / name) != expected:
            raise ValueError('archived_integrity_failure:' + name)
    run = pickle.load((directory / 'result.pkl').open('rb'))
    return run, dict(directory=str(directory), verified_sha256=hashes)


def load_projection(manifest, calendar):
    """Only columns needed to recover membership/outcomes, not a new backtest."""
    from quantlab.data.contracts import decision_time, timestamps_series
    amount, prices, panels = {}, {}, {}
    cutoffs = pd.Series([decision_time(d) for d in calendar], index=calendar)
    for code, item in manifest['panels'].items():
        path = MANIFEST.parent / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('panel_integrity_failure:' + code)
        p = pd.read_parquet(path, columns=['close', 'feature_close', 'amount',
                                          'is_st_at_open', 'available_at', 'quality_status'])
        p = p.reindex(calendar)
        available = timestamps_series(p.available_at)
        known = available.notna() & available.le(cutoffs) & p.quality_status.eq('ok')
        amount[code] = p.amount.where(known)
        prices[code] = p.feature_close.where(known)
        panels[code] = p[['close', 'is_st_at_open']]
    def table(key):
        item = manifest[key]
        path = MANIFEST.parent / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('table_integrity_failure:' + key)
        return pd.read_parquet(path)
    dataset = SimpleNamespace(manifest=manifest, calendar=calendar, panels=panels,
                              instruments=table('instruments'), industry_records=table('industry_records'))
    return dataset, pd.DataFrame(amount), pd.DataFrame(prices)


def validate_scores(recovered, stored, expert_weights, stored_weights, day):
    aligned = recovered.reindex(stored.index)
    if not len(stored) or not np.isfinite(aligned).all() or not np.isfinite(stored).all():
        raise ValueError('score_reconstruction_coverage:' + str(day))
    error = float((aligned - stored).abs().max())
    if error > 1e-10 or expert_weights != stored_weights:
        raise ValueError('score_reconstruction_mismatch:' + str(day))
    return error


def reconstruct_scores(run, dataset, amount):
    from quantlab.policy_comparison import arm_config
    from quantlab.data.market import universe_at
    from quantlab.strategy.scoring import factor_crowding_penalty
    from quantlab.strategy.experts import experts_score
    from quantlab.data.contracts import InstrumentRules
    cfg = arm_config(dict(minimum_adv=5e6, risk_policy='P2', candidate='quality_pb', allocation_policy='legacy'))
    risk = {date(s['date']): s for s in run['risk_history']}
    states = {date(s['date']): s for s in run['account']}
    rows, checks, experts = [], [], []
    for trace in run['signal_trace']:
        if 'standardized_features' not in trace or 'selected' not in trace:
            continue
        d = pd.Timestamp(trace['date'])
        full = {k: pd.DataFrame([v], index=[d]) for k, v in trace['standardized_features'].items()}
        hist = run['ic_matrix'].loc[:d, list(full)]
        pen, _ = factor_crowding_penalty(hist, cfg.factors.crowding_corr_window, cfg.factors.crowding_corr_alert)
        if not pen.empty:
            pen = pen.reindex(pen.index.union([d])).sort_index().ffill().loc[[d]]
        weights = pd.DataFrame([trace['factor_weights']], index=[d])
        recovered, ew = experts_score(full, weights, pen, hist,
                                     lookback=cfg.factors.expert_ic_lookback, cfg=cfg.factors)
        all_scores = recovered.loc[d].dropna()
        stored = pd.Series(trace['scores'])
        error = validate_scores(all_scores, stored, ew, trace['expert_weights'], d)
        eligible, records = universe_at(dataset, d, amount, cfg)
        if len(eligible) != risk[date(d)]['eligible_count']:
            raise ValueError('eligible_count_mismatch:' + str(d))
        stocks = []
        for code in eligible:
            try:
                if InstrumentRules.from_record(records.loc[code]).asset_type == 'stock':
                    stocks.append(code)
            except ValueError:
                continue
        available = all_scores.reindex(stocks).dropna().drop(
            index=list(trace.get('unavailable_score_history', {})), errors='ignore')
        crowd = full.get('factor_price_crowding')
        if crowd is not None:
            available = available[crowd.loc[d].reindex(available.index).rank(pct=True) > (1 - cfg.factors.crowding_quantile)]
        labels = score_buckets(available)
        entries = trace.get('candidate_evidence', {}).get('entries', {})
        quality = stored[[bool(entries.get(c, {}).get('passes')) for c in stored.index]]
        quality_labels = score_buckets(quality) if len(quality) else pd.Series(dtype=int)
        selected = set(trace['selected'])
        held = states[date(d)]['weights']
        union = set(stored.index)
        recorded_union = set(c for names in trace['expert_candidates'].values() for c in names)
        if union != recorded_union:
            raise ValueError('stored_score_expert_union_mismatch:' + str(d))
        for code, score in available.items():
            rows.append(dict(date=date(d), instrument_id=code, score=float(score),
                             eligible_bucket=int(labels[code]), in_expert_union=code in union,
                             quality_pass=bool(entries.get(code, {}).get('passes')) if code in union else None,
                             quality_union_bucket=int(quality_labels[code]) if code in quality_labels else None,
                             selected=code in selected, held_weight=float(held.get(code, 0)),
                             sector=str(records.loc[code, 'sector'])))
        checks.append(dict(date=date(d), eligible_count=len(eligible), scored_after_filters=len(available),
                           stored_union=len(stored), selected=len(selected), max_score_error=error))
        experts.append(dict(date=date(d), **ew, leader=max(ew, key=ew.get)))
    return pd.DataFrame(rows), checks, pd.DataFrame(experts)


def attach_forward(frame, prices, calendar, gaps):
    unique = frame[['date', 'instrument_id']].drop_duplicates()
    results = []
    for row in unique.itertuples(index=False):
        values = dict(date=row.date, instrument_id=row.instrument_id)
        for horizon in (20, 60):
            value, reason = forward_outcome(prices[row.instrument_id], calendar, row.date,
                                            horizon, gaps.get(row.instrument_id, ()))
            values[f'return_{horizon}'] = value
            values[f'coverage_{horizon}'] = reason
        results.append(values)
    return frame.merge(pd.DataFrame(results), on=['date', 'instrument_id'], validate='many_to_one')


def aggregate_scores(frame, benchmark, calendar):
    records = []
    for d, block in frame.groupby('date'):
        for h in (20, 60):
            col = f'return_{h}'
            valid = block.dropna(subset=[col]).copy()
            benchmark_return, _ = forward_outcome(benchmark, calendar, d, h)
            industry_mean = valid.groupby('sector')[col].transform('mean')
            valid['sector_residual'] = valid[col] - industry_mean
            masks = {f'eligible_Q{q}': valid.eligible_bucket.eq(q) for q in range(1, 6)}
            masks.update({f'quality_union_Q{q}': valid.quality_union_bucket.eq(q) for q in range(1, 6)})
            masks.update(selected=valid.selected, actual_held=valid.held_weight.gt(0), all_eligible=valid.score.notna())
            for label, mask in masks.items():
                group = valid[mask]
                weights = group.held_weight if label == 'actual_held' else pd.Series(1., index=group.index)
                value = float(np.average(group[col], weights=weights)) if len(group) and weights.sum() else None
                residual = float(np.average(group.sector_residual, weights=weights)) if value is not None else None
                records.append(dict(date=d, horizon=h, group=label, valid_count=len(group),
                                    return_mean=value, sector_residual=residual,
                                    market_excess=value - benchmark_return if value is not None and benchmark_return is not None else None))
    return pd.DataFrame(records)


def requested_membership(run, scored, stock_daily):
    """Recover complete saved selections/holdings before forward-data exclusions."""
    states = {date(x['date']): x for x in run['account']}
    traces = {date(x['date']): x for x in run['signal_trace'] if 'selected' in x}
    industry = stock_daily[['date', 'instrument_id', 'industry']].copy()
    industry['date'] = industry.date.map(date)
    industry = industry.set_index(['date', 'instrument_id']).industry.to_dict()
    rows = []
    for d in sorted(scored.date.unique()):
        held = {c: w for c, w in states[d]['weights'].items() if w > 0}
        selected = set(traces[d]['selected'])
        for code in sorted(set(held) | selected):
            rows.append(dict(date=d, instrument_id=code, held_weight=float(held.get(code, 0)),
                             selected=code in selected, ledger_sector=industry.get((d, code), 'UNKNOWN')))
    columns = ['date', 'instrument_id', 'sector', 'return_20', 'coverage_20', 'return_60', 'coverage_60']
    frame = pd.DataFrame(rows).merge(scored[columns], on=['date', 'instrument_id'],
                                    how='left', indicator=True, validate='one_to_one')
    frame['in_scored'] = frame['_merge'].eq('both')
    frame['sector'] = frame.sector.fillna(frame.ledger_sector)
    return frame.drop(columns=['_merge', 'ledger_sector'])


def aggregate_requested(membership, scored, benchmark, calendar):
    records = []
    for d, block in membership.groupby('date'):
        for h in (20, 60):
            col = f'return_{h}'
            peers = scored[scored.date.eq(d)].dropna(subset=[col]).groupby('sector')[col].mean()
            benchmark_return, _ = forward_outcome(benchmark, calendar, d, h)
            for label, mask in [('actual_held', block.held_weight.gt(0)), ('selected', block.selected)]:
                group = block[mask].copy()
                valid = group.dropna(subset=[col]).copy()
                weights = valid.held_weight if label == 'actual_held' else pd.Series(1., index=valid.index)
                value = float(np.average(valid[col], weights=weights)) if len(valid) and weights.sum() else None
                valid['residual'] = valid[col] - valid.sector.map(peers)
                residual_valid = valid.residual.notna()
                residual = float(np.average(valid.loc[residual_valid, 'residual'], weights=weights[residual_valid])) if residual_valid.any() else None
                total_weight, observed_weight = float(group.held_weight.sum()), float(valid.held_weight.sum())
                record = dict(date=d, horizon=h, group=label, requested_count=len(group),
                              scored_count=int(group.in_scored.sum()), valid_count=len(valid),
                              unscored_count=int((~group.in_scored).sum()),
                              omitted_count=len(group) - len(valid), total_nav_weight=total_weight,
                              scored_nav_weight=float(group.loc[group.in_scored, 'held_weight'].sum()),
                              observed_nav_weight=observed_weight, omitted_nav_weight=total_weight - observed_weight,
                              observed_invested_weight_fraction=observed_weight / total_weight if total_weight else None,
                              sector_baseline_valid_count=int(residual_valid.sum()),
                              sector_baseline_valid_nav_weight=float(valid.loc[residual_valid, 'held_weight'].sum()),
                              return_mean=value, sector_residual=residual,
                              market_excess=value - benchmark_return if value is not None and benchmark_return is not None else None,
                              return_basis='observed_stock_weight_normalized_feature_return' if label == 'actual_held' else 'observed_selected_equal_weight_feature_return')
                for reason, part in group.groupby(f'coverage_{h}', dropna=False):
                    record['coverage_' + str(reason) + '_count'] = len(part)
                    record['coverage_' + str(reason) + '_nav_weight'] = float(part.held_weight.sum())
                records.append(record)
    return pd.DataFrame(records)


def registered_outcomes(outcomes, protocol):
    """Reference summaries are not registered runs; registered failures are fatal."""
    selected = {}
    for name in protocol['protocol']['arms']:
        if name not in outcomes:
            raise ValueError('missing_registered_arm:' + name)
        record = outcomes[name]
        if not isinstance(record, dict) or not isinstance(record.get('run_id'), str) or not record['run_id']:
            raise ValueError('malformed_registered_arm:' + name)
        selected[name] = record
    return selected


def cash_rows(run, name):
    """Describe archived states; inherited decision evidence is dated explicitly."""
    issues = defaultdict(list)
    for item in run['allocation_issues']:
        issues[date(item['date'])].append(item['reason'])
    traces = {date(x['date']): x for x in run['signal_trace']
              if 'scores' in x or x.get('status') in ('invalid', 'warmup', 'allocation_failed')}
    if len(run['account']) != len(run['risk_history']):
        raise ValueError('cash_calendar_mismatch:' + name)
    rows, counts = [], Counter()
    last_issue, last_trace, decision_date, decision_eligible = [], {}, None, None
    for state, risk in zip(run['account'], run['risk_history']):
        d = date(state['date'])
        if d != date(risk['date']):
            raise ValueError('cash_calendar_mismatch:' + name)
        if d in traces or d in issues:
            last_trace, last_issue = traces.get(d, {}), issues.get(d, [])
            decision_date, decision_eligible = d, risk.get('eligible_count')
        flat = not any(q > 0 for q in state['quantities'].values())
        evidence = []

        def add(label, source, evidence_date, detail):
            evidence.append(dict(label=label, source=source, date=evidence_date,
                                 inherited=evidence_date != d, detail=detail))

        if risk['manual_review'] or risk['level'] == 4:
            add('L4_manual_stop', 'current_risk', d, 'manual_review or level 4')
        if flat and risk['level'] == 3 and not risk['allow_new_stock']:
            add('L3_recovery_lock', 'current_risk_and_account', d, 'flat level 3; new stock forbidden')
        if any(any(k in x for k in ('constraint', 'solver', 'covariance')) for x in last_issue):
            add('constraint_conflict', 'last_decision.allocation_issues', decision_date, '|'.join(last_issue))
        if state['stale'] or 'data_stale' in risk.get('reasons', []):
            add('data_block', 'current_stale_marks_or_data_stale_risk', d, str(state['stale']))
        if last_trace.get('status') in ('invalid', 'warmup'):
            add('data_block', 'last_decision.score_status', decision_date, last_trace['status'])
        if risk.get('eligible_count') == 0:
            add('no_candidates', 'current_eligible_count', d, 'explicit zero')
        elif decision_eligible == 0:
            add('no_candidates', 'last_decision.eligible_count', decision_date, 'explicit zero')
        if 'no_candidates' in last_issue:
            add('no_candidates', 'last_decision.allocation_issues', decision_date, 'no_candidates')
        if 'scores' in last_trace and not last_trace['scores']:
            add('no_candidates', 'last_decision.empty_scores', decision_date, 'explicit empty score dictionary')
        if not evidence:
            label = ('invested_with_residual_cash' if not flat else
                     'waiting_for_first_decision' if decision_date is None else 'idle_after_decision_unattributed')
            add(label, 'current_account_and_decision_history', d, 'no recorded blocking cause')
        labels = list(dict.fromkeys(e['label'] for e in evidence))
        counts.update(labels)
        rows.append(dict(arm=name, date=d, cash=float(state['cash']), nav=float(state['nav']),
                         cash_fraction=state['cash'] / state['nav'], flat=flat, level=risk['level'],
                         labels='|'.join(labels), primary=labels[0], label_evidence=json.dumps(evidence),
                         last_decision_date=decision_date, last_allocation_issue='|'.join(last_issue),
                         eligible_count=risk.get('eligible_count'), last_decision_eligible_count=decision_eligible,
                         recovery_count=risk['recovery_count']))
    frame = pd.DataFrame(rows)
    locked = frame.labels.str.contains('L3_recovery_lock')
    episodes = []
    for _, block in frame[locked].groupby((locked != locked.shift()).cumsum()[locked]):
        episodes.append(dict(start=block.date.iloc[0], end=block.date.iloc[-1], sessions=len(block),
                             cash_min=float(block.cash.min()), cash_max=float(block.cash.max()),
                             nav_min=float(block.nav.min()), nav_max=float(block.nav.max()),
                             constant_cash=bool(block.cash.max() - block.cash.min() < 1e-8),
                             constant_nav=bool(block.nav.max() - block.nav.min() < 1e-8),
                             recovery_count_max=int(block.recovery_count.max())))
    return rows, dict(sessions=len(rows), label_counts=dict(counts),
                      primary_counts=frame.primary.value_counts().to_dict(), L3_flat_episodes=episodes,
                      flat_sessions=int(frame.flat.sum()), mean_cash_fraction=float(frame.cash_fraction.mean()),
                      rule='Labels overlap and are not additive lost-profit estimates. Last-decision evidence remains dated until the next decision. Generic data_issues do not imply trade blocks. No rule changed.')


def cash_audit(outcomes, protocol):
    all_rows, summaries, identities = [], {}, {}
    for name, record in registered_outcomes(outcomes, protocol).items():
        run, identities[name] = load_archived(record)
        rows, summaries[name] = cash_rows(run, name)
        all_rows.extend(rows)
        del run
        gc.collect()
    return pd.DataFrame(all_rows), summaries, identities


def phase(day):
    year = pd.Timestamp(day).year
    return '2020-21' if year <= 2021 else '2022-23' if year <= 2023 else '2024-25' if year <= 2025 else '2026H1'


def records(frame):
    # JSON roundtrip converts pandas missing values to JSON null.
    return json.loads(frame.to_json(orient='records', date_format='iso', double_precision=15))


def finalize_saved(source, output):
    """Resume only postprocessing; never reload the market projection or replay."""
    from quantlab.data.contracts import decision_time, timestamps_series
    if source.resolve() == output.resolve() or any(output.glob('*.parquet')) or (output / 'summary.json').exists():
        raise ValueError('final_output_must_be_new')
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: sha(p) for p in source.glob('*.parquet')}
    outcomes = json.loads((RESEARCH / 'outcomes.json').read_text())
    protocol = json.loads((RESEARCH / 'protocol.json').read_text())
    registered = registered_outcomes(outcomes, protocol)
    if len(registered) != 8:
        raise ValueError('expected_eight_registered_arms')
    manifest = json.loads(MANIFEST.read_text())
    run, main_source = load_archived(registered['qualityP2'])
    calendar = pd.DatetimeIndex(run['calendar'])
    calendar = calendar[calendar <= pd.Timestamp('2026-06-30')]
    scored = pd.read_parquet(source / 'scored_outcomes.parquet')
    trades = pd.read_parquet(source / 'trade_outcomes.parquet')
    groups = pd.read_parquet(source / 'score_date_groups.parquet')
    experts = pd.read_parquet(source / 'expert_weights.parquet')
    pairs = pd.read_parquet(source / 'trade_same_day_pairs.parquet')
    if scored.duplicated(['date', 'instrument_id']).any():
        raise ValueError('duplicate_saved_score')
    checks = []
    risk = {date(x['date']): x for x in run['risk_history']}
    traces = {date(x['date']): x for x in run['signal_trace'] if 'standardized_features' in x and 'selected' in x}
    if set(traces) != set(scored.date.unique()) or len(traces) != 74:
        raise ValueError('saved_decision_date_mismatch')
    for d, block in scored.groupby('date'):
        trace = traces[d]
        saved_union = block[block.in_expert_union].set_index('instrument_id').score
        expected_union = set(c for names in trace['expert_candidates'].values() for c in names)
        if set(saved_union.index) != set(trace['scores']) or set(saved_union.index) != expected_union:
            raise ValueError('saved_union_membership_mismatch:' + d)
        weights = experts.set_index('date').loc[d, list(trace['expert_weights'])].astype(float).to_dict()
        error = validate_scores(saved_union, pd.Series(trace['scores']), weights, trace['expert_weights'], d)
        checks.append(dict(date=d, eligible_count=risk[d]['eligible_count'], scored_after_filters=len(block),
                           stored_union=len(saved_union), selected_requested=len(trace['selected']),
                           max_score_error=error, union_membership_equal=True, expert_weights_equal=True,
                           validation='saved outcomes vs archived trace; earlier full-market reconstruction not repeated'))
    print('validated saved scores at74 decisions; recovering complete held/selected membership', flush=True)
    ledger_dir = ROOT / 'output/dynamic_risk/analysis_6157c64/qualityP2'
    ledger_report = json.loads((ledger_dir / 'report.json').read_text())
    for filename, expected in ledger_report['frame_sha256'].items():
        if sha(ledger_dir / filename) != expected:
            raise ValueError('ledger_attribution_integrity_failure:' + filename)
    if ledger_report['evidence']['run_files']['result.pkl']['sha256'] != main_source['verified_sha256']['result.pkl']:
        raise ValueError('ledger_run_mismatch')
    stock_daily = pd.read_parquet(ledger_dir / 'stock_daily.parquet')
    membership = requested_membership(run, scored, stock_daily)
    gaps = defaultdict(list)
    for gap in manifest.get('data_gaps', []) + manifest.get('accounting_gaps', []):
        if gap.get('instrument_id') and gap.get('date'):
            gaps[gap['instrument_id']].append(gap['date'])
    panel_sources = {}
    cutoffs = pd.Series([decision_time(d) for d in calendar], index=calendar)
    for code in sorted(membership.loc[~membership.in_scored, 'instrument_id'].unique()):
        item = manifest['panels'][code]
        path = MANIFEST.parent / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('panel_integrity_failure:' + code)
        panel_sources[code] = dict(path=str(path), sha256=item['sha256'])
        panel = pd.read_parquet(path, columns=['feature_close', 'available_at', 'quality_status']).reindex(calendar)
        available = timestamps_series(panel.available_at)
        prices = panel.feature_close.where(available.notna() & available.le(cutoffs) & panel.quality_status.eq('ok'))
        for idx in membership.index[~membership.in_scored & membership.instrument_id.eq(code)]:
            for h in (20, 60):
                value, reason = forward_outcome(prices, calendar, membership.at[idx, 'date'], h, gaps[code])
                membership.at[idx, f'return_{h}'] = value
                membership.at[idx, f'coverage_{h}'] = reason
    if membership[['coverage_20', 'coverage_60']].isna().any().any():
        raise ValueError('unresolved_requested_membership')
    requested = aggregate_requested(membership, scored, run['benchmark']['close'], calendar)
    groups = groups[~groups.group.isin(['actual_held', 'selected'])].copy()
    coverage_rows = []
    for d, block in scored.groupby('date'):
        masks = {f'eligible_Q{q}': block.eligible_bucket.eq(q) for q in range(1, 6)}
        masks.update({f'quality_union_Q{q}': block.quality_union_bucket.eq(q) for q in range(1, 6)})
        masks['all_eligible'] = block.score.notna()
        for label, mask in masks.items():
            part = block[mask]
            for h in (20, 60):
                coverage_rows.append(dict(date=d, horizon=h, group=label, requested_count=len(part),
                                          scored_count=len(part), omitted_count=int(part[f'return_{h}'].isna().sum()),
                                          **{'coverage_' + reason + '_count': int(count) for reason, count in part[f'coverage_{h}'].value_counts().items()}))
    groups = groups.merge(pd.DataFrame(coverage_rows), on=['date', 'horizon', 'group'], validate='one_to_one')
    groups = pd.concat([groups, requested], ignore_index=True)
    groups['phase'] = groups.date.map(phase)
    metrics = ['return_mean', 'market_excess', 'sector_residual']
    phase_groups = groups.groupby(['phase', 'horizon', 'group']).agg(
        decision_dates=('date', 'nunique'), observed_dates=('return_mean', 'count'),
        requested_count=('requested_count', 'sum'), valid_count=('valid_count', 'sum'),
        return_mean=('return_mean', 'mean'), market_excess=('market_excess', 'mean'),
        sector_residual=('sector_residual', 'mean')).reset_index()
    overall_groups = groups.groupby(['horizon', 'group']).agg(
        decision_dates=('date', 'nunique'), observed_dates=('return_mean', 'count'),
        requested_count=('requested_count', 'sum'), valid_count=('valid_count', 'sum'),
        return_mean=('return_mean', 'mean'), market_excess=('market_excess', 'mean'),
        sector_residual=('sector_residual', 'mean')).reset_index()
    trades['trade_id'] = np.arange(len(trades))
    session_map = {date(d): i for i, d in enumerate(calendar)}
    trades['repurchase_lag_sessions'] = trades.apply(lambda row: session_map[row.date] - session_map[row.previous_exit_date]
                                                     if row.category == 'repurchase' else None, axis=1)
    for h in (20, 60):
        trades[f'repurchase_within_{h}'] = trades.category.eq('repurchase') & trades.repurchase_lag_sessions.le(h)
    rebuilt = pd.DataFrame(classify_fills(run))
    for col in ['date', 'instrument_id', 'category', 'reason', 'side']:
        if not trades[col].equals(rebuilt[col]):
            raise ValueError('saved_fill_classification_mismatch:' + col)
    cost_checks = {col: float(trades[col].sum() - rebuilt[col].sum()) for col in ['notional', 'fees', 'slippage', 'cost']}
    cost_checks['summary_fees_residual'] = float(trades.fees.sum() - registered['qualityP2']['summary']['fees'])
    cost_checks['summary_slippage_residual'] = float(trades.slippage.sum() - registered['qualityP2']['summary']['slippage'])
    if any(abs(value) > 1e-6 for value in cost_checks.values()):
        raise ValueError('fill_cost_reconciliation_failure')
    trade_summary = {}
    for category, part in trades.groupby('category'):
        item = dict(fills=len(part), security_days=len(part[['date', 'instrument_id']].drop_duplicates()),
                    notional=float(part.notional.sum()), fees=float(part.fees.sum()),
                    slippage=float(part.slippage.sum()), cost=float(part.cost.sum()))
        for h in (20, 60):
            valid = part.dropna(subset=[f'return_{h}'])
            item[str(h)] = dict(coverage=part[f'coverage_{h}'].value_counts().to_dict(),
                               observed_notional=float(valid.notional.sum()), omitted_notional=float(part.notional.sum()-valid.notional.sum()),
                               notional_weighted_return=float(np.average(valid[f'return_{h}'], weights=valid.notional)) if len(valid) else None)
        trade_summary[category] = item
    repurchases = trades[trades.category.eq('repurchase')]
    repurchase_summary = repurchases.groupby('previous_exit_reason').agg(
        fills=('trade_id', 'count'), cost=('cost', 'sum'), within_20=('repurchase_within_20', 'sum'),
        within_60=('repurchase_within_60', 'sum'), mean_lag_sessions=('repurchase_lag_sessions', 'mean')).reset_index()
    buybacks = risk_buybacks(trades, run, calendar)
    buyback_summary = []
    for kind, part in buybacks.groupby('risk_exit_type'):
        buyback_summary.append(dict(risk_exit_type=kind, risk_events=len(part), with_later_buy=int(part.buy_id.notna().sum()),
                                    unique_later_buys=int(part.buy_id.nunique()), within_20=int(part.within_20.sum()),
                                    within_60=int(part.within_60.sum()), observed_20=int(part.observed_20.sum()), observed_60=int(part.observed_60.sum()),
                                    unique_buys_within_20=int(part.loc[part.within_20, 'buy_id'].nunique()),
                                    unique_buys_within_60=int(part.loc[part.within_60, 'buy_id'].nunique())))
    experts['phase'] = experts.date.map(phase)
    expert_phase = []
    numeric = ['momentum', 'value', 'lowvol_quality', 'flow', 'composite']
    for period, block in experts.groupby('phase', sort=False):
        expert_phase.append(dict(phase=period, decisions=len(block), leaders=block.leader.value_counts().to_dict(),
                                 switches=int(block.leader.ne(block.leader.shift()).sum()-1), weight_means=block[numeric].mean().to_dict()))
    realized = stock_daily.copy()
    realized['phase'] = realized.date.map(phase)
    ledger_phases = realized.groupby('phase')[['pnl', 'explicit_fees', 'reported_slippage', 'cash_dividend_accrual']].sum().reset_index()
    ledger_sectors = realized.groupby(['phase', 'industry'])[['pnl', 'explicit_fees', 'reported_slippage']].sum().reset_index()
    if abs(realized.pnl.sum() - ledger_report['net_profit']) > 1e-6:
        raise ValueError('realized_ledger_pnl_reconciliation')
    del run, stock_daily, realized
    gc.collect()
    print('auditing only eight registered archived cash paths', flush=True)
    daily, cash, identities = cash_audit(outcomes, protocol)
    frames = dict(complete_requested_outcomes=membership, requested_group_coverage=requested,
                  score_date_groups=groups, score_phase_groups=phase_groups, score_overall_groups=overall_groups,
                  trade_outcomes=trades, risk_sell_first_buy_links=buybacks, repurchase_summary=repurchase_summary,
                  expert_weights=experts, cash_daily_causes=daily,
                  realized_pnl_by_phase=ledger_phases, realized_pnl_by_phase_industry=ledger_sectors)
    for name, frame in frames.items():
        frame.to_parquet(output / (name + '.parquet'), index=False)
    membership_summary = []
    for label, mask in [('actual_held', membership.held_weight.gt(0)), ('selected', membership.selected)]:
        part = membership[mask]
        by_date = part.assign(omitted_scored_weight=part.held_weight.where(~part.in_scored, 0)).groupby('date').omitted_scored_weight.sum()
        membership_summary.append(dict(group=label, requested_rows=len(part), scored_rows=int(part.in_scored.sum()),
                                        unscored_rows=int((~part.in_scored).sum()), omitted_scored_dates=int((by_date > 0).sum()),
                                        largest_original_omitted_nav_weight=float(by_date.max()),
                                        largest_original_omitted_date=str(by_date.idxmax()),
                                        sum_snapshot_nav_weights=float(part.held_weight.sum()),
                                        sum_original_unscored_nav_weights=float(part.loc[~part.in_scored, 'held_weight'].sum()),
                                        coverage={str(h): part[f'coverage_{h}'].value_counts().to_dict() for h in (20, 60)}))
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), status='conditional_posthoc_attribution',
                   promotion_approved=False, resumed_postprocessing=True, account_replayed=False,
                   full_market_projection_reloaded=False, source=main_source,
                   original_parquets=dict(path=str(source), sha256=source_hashes),
                   manifest_sha256=sha(MANIFEST), protocol_sha256=sha(RESEARCH / 'protocol.json'),
                   outcomes_sha256=sha(RESEARCH / 'outcomes.json'), script_sha256=sha(__file__),
                   recovered_panel_sources=panel_sources, saved_evidence_checks=checks,
                   reconstruction_provenance='Original run verified all panels, full score reconstruction, expert weights and eligible counts before saved parquet publication. This resumed pass compares saved values/union membership/weights with compact archive and does not repeat that panel scan.',
                   interpretation=[
                       'All results remain same-sample conditional research; data, accounting, publication-date and industry proxy gaps block adoption.',
                       'Forward20/60 means use historical-adjustment feature prices and complete exchange-session paths; they are not cash-account or total-return outcomes. Missing/end/gap values are excluded, never zero.',
                       'Actual-held forward mean normalizes the observed stock weights, omitting cash and censored stock returns. Coverage exposes total/scored/observed/omitted NAV and invested weights per decision. Use separate recorded cash ledger for actual PnL.',
                       'all_eligible and eligible_Q mean the full after-filter scored eligible stock population, not all listed stocks or whole-market financial qualification. quality_union_Q only covers financially qualified recorded expert-union candidates.',
                       'Sector residual subtracts the horizon-valid scored-eligible industry peer mean at the same date; recovered noneligible names never change that baseline. Industry labels retain effective-date proxy gaps.',
                       'market_excess is feature-price stock return minus CSI300 sh.000300 raw-price index return; bases differ and this is neither total-return alpha nor causal alpha.',
                       'Trade categories are exclusive realized daily-security fill transitions. Repurchase means a prior fully exited position re-entered. Risk-sell to next-buy links additionally include partial reductions then weight increases; overlapping links never add costs.',
                       'Risk-buyback links use first later exchange-session buy, stable saved-fill order for same-day ties; same-session sells/buys are disclosed separately. Unobserved horizon follow-up is censored.',
                       'Same-date buy-minus-removed and phase/group means are descriptive overlapping windows, not executable counterfactuals, significance tests, annualized forecasts, independent OOS or additive lost-profit estimates.',
                       'Cash labels can overlap; inherited allocation/no-candidate/score-status evidence retains its decision date. Unknown eligible_count is not zero; generic data_issues are not automatic trade blocks. Original L3/L4 rules unchanged.'],
                   membership_coverage=membership_summary,
                   score_coverage={str(h): scored[f'coverage_{h}'].value_counts().to_dict() for h in (20, 60)},
                   trade_categories=trade_summary, cost_reconciliation=cost_checks,
                   trade_category_count_per_security_day_max=int(trades.groupby(['date','instrument_id']).category.nunique().max()),
                   repurchase_summary=records(repurchase_summary), risk_sell_first_buy_summary=buyback_summary,
                   same_day_trade_pair_summary=records(pairs.groupby('horizon').buy_minus_removed.agg(['count', 'mean', 'median']).reset_index()),
                   score_group_means=records(overall_groups), score_phase_means=records(phase_groups),
                   expert_leader_counts=experts.leader.value_counts().to_dict(),
                   expert_leader_switches=int(experts.leader.ne(experts.leader.shift()).sum()-1),
                   expert_weight_means=experts[numeric].mean().to_dict(), expert_phase=expert_phase,
                   realized_ledger=dict(report=str(ledger_dir / 'report.json'), report_sha256=sha(ledger_dir / 'report.json'),
                                        frame_sha256=ledger_report['frame_sha256'], status=ledger_report['status'],
                                        net_profit=ledger_report['net_profit'], fees=ledger_report['fees'],
                                        reported_slippage=ledger_report['reported_slippage'],
                                        concentration=ledger_report['concentration_removal_diagnostics'],
                                        basis=ledger_report['attribution_method'], limitations=ledger_report['limitations']),
                   cash_summary=cash, cash_sources=identities)
    if source_hashes != {p.name: sha(p) for p in source.glob('*.parquet')}:
        raise ValueError('original_parquets_changed')
    summary['artifacts'] = {p.name: sha(p) for p in output.glob('*.parquet')}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), recovered_panels=len(panel_sources),
                          membership=membership_summary, cash_arms=len(cash)), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--resume-from', type=Path, help='Finalize saved parquets into a NEW --output; no panel rescan or replay')
    args = parser.parse_args()
    sys.path.insert(0, str(FROZEN))
    if args.resume_from is not None:
        finalize_saved(args.resume_from, args.output)
        return
    if (args.output / 'summary.json').exists():
        raise SystemExit('existing completed attribution: use a new output path for a revised audit')
    args.output.mkdir(parents=True, exist_ok=True)
    outcomes = json.loads((RESEARCH / 'outcomes.json').read_text())
    manifest = json.loads(MANIFEST.read_text())
    run, evidence = load_archived(outcomes['qualityP2'])
    calendar = pd.DatetimeIndex(run['calendar'])
    calendar = calendar[calendar <= pd.Timestamp('2026-06-30')]
    print('verified main result; reading historical projection', flush=True)
    dataset, amount, prices = load_projection(manifest, calendar)
    print('recovering saved scores and historical membership', flush=True)
    scored, checks, experts = reconstruct_scores(run, dataset, amount)
    gaps = defaultdict(list)
    for gap in manifest.get('data_gaps', []) + manifest.get('accounting_gaps', []):
        if gap.get('instrument_id') and gap.get('date'):
            gaps[gap['instrument_id']].append(gap['date'])
    print('measuring post-hoc 20/60 session outcomes', flush=True)
    scored = attach_forward(scored, prices, calendar, gaps)
    trades = attach_forward(pd.DataFrame(classify_fills(run)), prices, calendar, gaps)
    assert abs(trades.cost.sum() - outcomes['qualityP2']['summary']['fees'] - outcomes['qualityP2']['summary']['slippage']) < 1e-6
    grouped = aggregate_scores(scored, run['benchmark']['close'], calendar)
    for name, frame in [('scored_outcomes', scored), ('trade_outcomes', trades), ('score_date_groups', grouped), ('expert_weights', experts)]:
        frame.to_parquet(args.output / f'{name}.parquet', index=False)
    trade_summary = {}
    for category, part in trades.groupby('category'):
        item = dict(fills=len(part), notional=float(part.notional.sum()), cost=float(part.cost.sum()))
        for h in (20, 60):
            valid = part.dropna(subset=[f'return_{h}'])
            item[str(h)] = dict(coverage=part[f'coverage_{h}'].value_counts().to_dict(),
                               notional_weighted_return=float(np.average(valid[f'return_{h}'], weights=valid.notional)) if len(valid) else None)
        trade_summary[category] = item
    pair_rows = []
    for h in (20, 60):
        ordinary = trades[trades.reason.eq('legacy_repaired')].dropna(subset=[f'return_{h}'])
        for d, part in ordinary.groupby('date'):
            buys = part[part.category.isin(['addition', 'repurchase'])]
            sells = part[part.category.eq('removal')]
            if len(buys) and len(sells):
                pair_rows.append(dict(date=d, horizon=h, buy_count=len(buys), removed_count=len(sells),
                                      buy_minus_removed=float(np.average(buys[f'return_{h}'], weights=buys.notional) - np.average(sells[f'return_{h}'], weights=sells.notional))))
    pairs = pd.DataFrame(pair_rows)
    pairs.to_parquet(args.output / 'trade_same_day_pairs.parquet', index=False)
    score_means = grouped.groupby(['horizon', 'group'])[['return_mean', 'market_excess', 'sector_residual']].mean().reset_index()
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), status='conditional_posthoc_attribution',
                   promotion_approved=False, source=evidence, manifest_sha256=sha(MANIFEST), script_sha256=sha(__file__),
                   interpretation=['No account replay; saved factors/scores reconstructed and exactly checked against archived candidates.',
                                   'Outcomes use historical-adjustment feature prices, not executable cash-account total returns; no fee addback.',
                                   'Exchange calendar endpoints, complete price path and known gaps checked. Missing/terminal observations are excluded and reported, not zero.',
                                   'Eligible scores before expert union; quality_union buckets only within recorded financial-entry candidates, not all-market financial coverage.',
                                   'Industry residual subtracts same-date eligible industry mean, not a causal or full multifactor alpha estimate.',
                                   'Overlapping horizon/date averages are descriptive and do not constitute significance or independent OOS.'],
                   reconstruction_checks=checks, score_coverage={str(h): scored[f'coverage_{h}'].value_counts().to_dict() for h in (20, 60)},
                   trade_categories=trade_summary,
                   same_day_trade_pair_summary=pairs.groupby('horizon').buy_minus_removed.agg(['count', 'mean', 'median']).reset_index().to_dict('records'),
                   score_group_means=score_means.to_dict('records'),
                   expert_leader_counts=experts.leader.value_counts().to_dict(),
                   expert_leader_switches=int(experts.leader.ne(experts.leader.shift()).sum() - 1),
                   expert_weight_means=experts.select_dtypes('number').mean().to_dict())
    del run, dataset, amount, prices, scored
    gc.collect()
    print('auditing eight archived cash paths', flush=True)
    daily, cash, identities = cash_audit(outcomes, json.loads((RESEARCH / 'protocol.json').read_text()))
    daily.to_parquet(args.output / 'cash_daily_causes.parquet', index=False)
    summary['cash_summary'], summary['cash_sources'] = cash, identities
    summary['artifacts'] = {p.name: sha(p) for p in args.output.glob('*.parquet')}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), scored_dates=len(checks), trades=trade_summary,
                          experts=summary['expert_leader_counts']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
