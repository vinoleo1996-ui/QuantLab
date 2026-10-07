#!/usr/bin/env python3
"""Fill the 2018 decision dates the frozen broad-basket file does not contain.

The saved heads start on 2019-03-01. The same 20-session grid already includes
2018-01-02. This run builds only those earlier memberships from the current
dataset, then carries one basket into the frozen heads. It does not replace
any frozen date and does not move the 2018-01-02 CAGR clock.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from datetime import datetime, timezone
import io
import json

import numpy as np
import pandas as pd

import continuous_band as band
import stable_book as book

OUT = ROOT / 'output/continuous_band_2018ext_20260928'
ANCHOR = pd.Timestamp('2019-03-01')
RANK_START, RANK_END = 3001, 5000
MIN_ADV = 5_000_000.0
MIN_OBSERVATIONS = 253  # current row plus the original 252-session price lag
ADV_WINDOW = 60


def to_exchange(panel_id):
    market, number = str(panel_id).split('.')
    return f'{number}.{market.upper()}'


def early_decision_days(calendar, anchor=ANCHOR, step=20, start=book.START):
    """Walk backward on the frozen cadence. The anchor itself stays frozen."""
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize()
    position = calendar.get_loc(pd.Timestamp(anchor))
    days = []
    step_count = 1
    while position - step * step_count >= 0:
        day = pd.Timestamp(calendar[position - step * step_count])
        if day < pd.Timestamp(start):
            break
        days.append(day)
        step_count += 1
    return tuple(reversed(days))


def liquidity_band(observations):
    """Ranks 3001-5000 by descending 60-observation amount, symbol as the tie break.

    ``observations`` is a sequence of ``(amount, symbol)`` for names that already
    have the original history. Fewer than 3001 names means the day is not bought.
    """
    if len(observations) < RANK_START:
        return None
    ordered = sorted(observations, key=lambda item: (-item[0], item[1]))
    chosen = []
    for rank, (amount, symbol) in enumerate(ordered, start=1):
        if rank < RANK_START:
            continue
        if rank > RANK_END:
            break
        if amount >= MIN_ADV:
            chosen.append(symbol)
    return tuple(chosen)


def _positive_amounts(frame):
    available = book.timestamps_series(frame['available_at'])
    available.index = frame.index
    cutoff = pd.Series(pd.DatetimeIndex(frame.index).tz_localize('Asia/Shanghai') + pd.Timedelta(hours=18),
                       index=frame.index)
    bad = available.isna() | available.gt(cutoff) | frame['quality_status'].astype(str).ne('ok')
    amount = pd.to_numeric(frame.amount, errors='coerce').mask(bad)
    amount = amount[amount > 0]
    return amount.index.to_numpy(dtype='datetime64[ns]'), amount.to_numpy(dtype=float)


def load_amount_history(manifest_path=book.MANIFEST, end=ANCHOR):
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    history = {}
    columns = ['amount', 'available_at', 'quality_status']
    panels = manifest['panels']
    for index, (code, entry) in enumerate(panels.items()):
        if index % 1000 == 0:
            print(json.dumps({'stage': 'amounts', 'index': index, 'total': len(panels)}), flush=True)
        frame = book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, entry)), columns=columns))
        frame = frame[~frame.index.duplicated(keep='last')]
        frame = frame.loc[:end]
        if frame.empty:
            continue
        dates, values = _positive_amounts(frame)
        if len(values) >= ADV_WINDOW:
            history[to_exchange(code)] = (dates, values)
    return history


def amount_on(dates, values, day):
    """Last 60 positive amounts ending on ``day``, if the original history exists."""
    position = np.searchsorted(dates, np.datetime64(pd.Timestamp(day)), side='right') - 1
    if position < MIN_OBSERVATIONS - 1 or dates[position] != np.datetime64(pd.Timestamp(day)):
        return None
    window = values[position - ADV_WINDOW + 1:position + 1]
    if len(window) < ADV_WINDOW:
        return None
    return float(window.mean())


def load_industry_history(manifest_path=book.MANIFEST):
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['industry_records'])))
    frame['effective_at'] = pd.to_datetime(frame.effective_at)
    grouped = {}
    for code, rows in frame.groupby('instrument_id', sort=False):
        rows = rows.sort_values(['effective_at', 'revision']).drop_duplicates('effective_at', keep='last')
        grouped[to_exchange(code)] = (
            rows.effective_at.to_numpy(dtype='datetime64[ns]'),
            rows.sector.astype(str).to_numpy(),
        )
    return grouped


def industry_on(history, code, day):
    pair = history.get(code)
    if pair is None:
        return 'unclassified'
    dates, sectors = pair
    position = np.searchsorted(dates, np.datetime64(pd.Timestamp(day)), side='right') - 1
    if position < 0:
        return 'unclassified'
    sector = sectors[position]
    return 'unclassified' if sector in ('', 'NA', 'UNKNOWN', 'nan') else sector


def early_heads(amount_history, industry_history, days):
    heads, industries, skipped = {}, {}, []
    for day in days:
        observations = []
        for code, (dates, values) in amount_history.items():
            amount = amount_on(dates, values, day)
            if amount is not None:
                observations.append((amount, code))
        members = liquidity_band(observations)
        key = str(pd.Timestamp(day).date())
        if not members:
            skipped.append({'date': key, 'qualified': len(observations)})
            continue
        heads[key] = {code: 1.0 for code in members}
        industries[key] = {code: industry_on(industry_history, code, day) for code in members}
    return heads, industries, skipped


def prepare():
    book.prepare_fresh_output(OUT)
    manifest = json.loads(book.MANIFEST.read_text())
    calendar = pd.DatetimeIndex(book._dated(pd.read_parquet(
        book.MANIFEST.parent / manifest['calendar']['path'])).session)
    days = early_decision_days(calendar)
    if len(days) != 14 or days[0] != book.START or days[-1] >= ANCHOR:
        book.write_json(OUT / 'BLOCKED.json', {
            'reason': 'early_grid_mismatch', 'days': [str(day.date()) for day in days],
            'promotion_approved': False})
        return 'blocked'
    amount_history = load_amount_history()
    industry_history = load_industry_history()
    added_heads, added_industries, skipped = early_heads(amount_history, industry_history, days)
    parts = {name: json.loads(path.read_text()) for name, path in book.INPUTS.items()}
    frozen_heads, frozen_industries = book.merge_band_inputs(parts)
    overlap = set(added_heads) & set(frozen_heads)
    if overlap:
        raise ValueError('early_dates_overlap_frozen_heads:' + ','.join(sorted(overlap)))
    heads = dict(frozen_heads)
    industries = dict(frozen_industries)
    heads.update(added_heads)
    industries.update(added_industries)
    sessions = book.window_sessions(calendar)
    targets = band.build_targets(sessions, heads, industries)
    first_frozen = next(row for row in targets if row['signal_date'] >= '2019-03-01')
    if not targets or targets[0]['signal_date'] > '2019-01-25':
        book.write_json(OUT / 'BLOCKED.json', {
            'reason': 'no_early_band', 'skipped': skipped, 'promotion_approved': False})
        return 'blocked'
    path = OUT / 'targets.json'
    book.write_json(path, {'targets': targets, 'promotion_approved': False,
                           'early_days': [str(day.date()) for day in days], 'skipped': skipped})
    unclassified = {
        day: sum(value == 'unclassified' for value in group.values()) / len(group)
        for day, group in added_industries.items()
    }
    protocol = {
        'version': 'continuous-band-2018ext-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'construction': 'frozen_heads_plus_earlier_same_adv_rule',
        'question': 'Does filling the missing 2018 decision dates with the same liquidity band raise the continuous 50-name account to 15 percent?',
        'early_rule': {
            'amount': 'mean of the last 60 positive-amount observations',
            'history': '253 positive-amount observations, matching the original 252-session lag',
            'ranks': [RANK_START, RANK_END],
            'minimum_average_amount': MIN_ADV,
            'below_3001_names': 'do not buy that day',
            'financials': 'none before 2019-03-01; one tier; score ties retain the input liquidity order (exact ADV ties use the explicit code rule)',
            'st': 'not excluded, matching the reference rebuild',
        },
        'carry': 'one basket into the frozen 2019-03-01 head; that head is not rebuilt',
        'first_target': targets[0]['signal_date'],
        'first_frozen_target': first_frozen['signal_date'],
        'skipped': skipped,
        'unclassified_share': unclassified,
        'cagr_gate': {
            'start': '2018-01-02', 'end': '2026-09-22',
            'formula': '(final_equity/600000)**(365.25/(last_session-first_session).days)-1',
            'from_first_fill_is_not_the_gate': True,
        },
        'gates': {'min_cagr': 0.15, 'max_drawdown': 0.25, 'recent_positive': True,
                  'cost_double_cagr_positive': True},
        'no_parameter_rescue': True,
        'inputs': {name: book.sha256(file) for name, file in book.INPUTS.items()},
        'manifest': book.sha256(book.MANIFEST),
        'sources': {str(file): book.sha256(file) for file in (
            Path(__file__), ROOT / 'tests/test_extend_2018_band.py',
            ROOT / 'scripts/continuous_band.py', ROOT / 'scripts/stable_book.py')},
        'targets': {str(path): book.sha256(path)},
        'promotion_approved': False,
    }
    book.write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'stage': 'prepared', 'early_days': len(added_heads), 'skipped': len(skipped),
                      'first': targets[0]['signal_date'], 'decisions': len(targets)}), flush=True)
    return 'prepared'


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    book.verify_protocol_sources(protocol)
    for name, digest in protocol['inputs'].items():
        if book.sha256(book.INPUTS[name]) != digest:
            raise ValueError('input_changed:' + name)
    for path, digest in protocol['targets'].items():
        if book.sha256(path) != digest:
            raise ValueError('targets_changed')
    root = book.MANIFEST.parent
    manifest = json.loads(book.MANIFEST.read_text())
    codes = sorted({book.canon(code) for row in json.loads((OUT / 'targets.json').read_text())['targets']
                    for code in list(row['weights']) + row['sell']})
    absent = sorted(set(codes) - set(manifest['panels']))
    if absent:
        book.write_json(OUT / 'BLOCKED.json', {'reason': 'target_panel_missing', 'count': len(absent),
                                               'promotion_approved': False})
        return 'blocked'
    calendar = pd.DatetimeIndex(book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['calendar'])))).session)
    instruments = pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['instruments'])))
    actions = pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['actions'])))
    benchmark = book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['benchmark']))))
    panels = {code: book._dated(pd.read_parquet(io.BytesIO(book._read_checked(root, manifest['panels'][code]))))
              for code in codes}
    from quantlab.data.contracts import MarketDataset
    dataset = MarketDataset(panels, instruments, calendar, actions, benchmark, manifest=manifest)
    errors = dataset.validate('free_research')
    if errors:
        book.write_json(OUT / 'BLOCKED.json', {'reason': 'projection_invalid', 'errors': errors,
                                               'promotion_approved': False})
        return 'blocked'
    for multiplier in (1, 2):
        print(json.dumps({'stage': 'replay', 'cost': multiplier}), flush=True)
        result = book.replay_account(dataset, OUT / 'targets.json', start=book.START, end=book.END,
                                     cost_mult=multiplier)
        summary = result['summary']
        if summary['first'] != '2018-01-02' or summary['last'] != '2026-09-22':
            raise ValueError('cagr_clock_moved')
        held = [state for state in result['account'] if state['quantities']]
        if held:
            start_held = pd.Timestamp(held[0]['date'])
            elapsed = (pd.Timestamp(summary['last']) - start_held).days
            summary['cagr_from_first_fill'] = float((summary['final_equity'] / book.INITIAL_CASH) ** (365.25 / elapsed) - 1)
            summary['first_fill'] = str(start_held.date())
        summary['cagr_clock'] = '2018-01-02_to_2026-09-22'
        folder = OUT / f'cost{multiplier}'
        folder.mkdir(parents=True, exist_ok=True)
        if not summary['account_checks_passed']:
            book.write_json(folder / 'summary.json', summary)
            book.write_json(OUT / 'BLOCKED.json', {'reason': 'account_identity_failed', 'cost': multiplier,
                                                   'errors': summary['account_check_errors'],
                                                   'promotion_approved': False})
            return 'blocked'
        book.write_json(folder / 'summary.json', summary)
        result['daily_return'].rename('daily_return').to_csv(folder / 'daily_returns.csv', index_label='date')
        print(json.dumps({'stage': 'replay_done', 'cost': multiplier, 'cagr': summary['cagr'],
                          'max_drawdown': summary['max_drawdown'],
                          'recent_return': summary['recent_return']}), flush=True)
    return 'ran'


def decide():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    book.verify_protocol_sources(protocol)
    cost1 = json.loads((OUT / 'cost1' / 'summary.json').read_text())
    cost2 = json.loads((OUT / 'cost2' / 'summary.json').read_text())
    if cost1.get('cagr_clock') != '2018-01-02_to_2026-09-22' or cost1.get('first') != '2018-01-02':
        raise ValueError('refusing_to_judge_a_moved_cagr_clock')
    reasons = []
    if cost1['cagr'] < 0.15:
        reasons.append('cagr')
    if -cost1['max_drawdown'] > 0.25 + 1e-12:
        reasons.append('drawdown')
    if cost1['recent_return'] <= 0:
        reasons.append('recent')
    if cost2['cagr'] <= 0:
        reasons.append('cost_double')
    if 'cagr' in reasons or 'drawdown' in reasons or 'recent' in reasons:
        decision = 'extended_2018_band_did_not_reach_15'
    elif 'cost_double' in reasons:
        decision = 'not_retained_cost_double_not_positive'
    else:
        decision = 'research_candidate_not_promoted'
    verdict = {'decision': decision, 'reasons': reasons, 'cagr': cost1['cagr'],
               'max_drawdown': cost1['max_drawdown'], 'recent_return': cost1['recent_return'],
               'cost2_cagr': cost2['cagr'], 'promotion_approved': False,
               'cagr_from_first_fill_not_used': cost1.get('cagr_from_first_fill'),
               'first_fill': cost1.get('first_fill')}
    book.write_json(OUT / 'DECISION.json', verdict)
    text = '\n'.join([
        '# 补上 2018 年宽篮',
        '',
        f"判决：`{decision}`。未通过项：{('、'.join(reasons) if reasons else '无')}。",
        f"扣费年化 {cost1['cagr'] * 100:.2f}%（时钟仍是 2018-01-02 至 2026-09-22）。"
        f"从首次持仓 {cost1.get('first_fill')} 起算是 {cost1.get('cagr_from_first_fill', float('nan')) * 100:.2f}%，不参与过线。",
        f"最大回撤 {-cost1['max_drawdown'] * 100:.2f}%，2024-12-31 至 2026-09-22 累计 {cost1['recent_return'] * 100:.2f}%，"
        f"期末 {cost1['final_equity']:.2f} 元。成本翻倍年化 {cost2['cagr'] * 100:.2f}%。",
        '2019-03-01 之前用同一成交额排名规则补决策日，没有财务快照的日期全体同一档并按代码排序。篮子带入冻结名单，不在边界重置。',
        'promotion_approved=false。',
        '',
    ])
    (OUT / 'REVIEW.md').write_text(text)
    path = ROOT / 'docs/RESEARCH_CONTINUITY.md'
    marker = '## 补上 2018 年宽篮（2026-09-28'
    existing = path.read_text()
    if marker not in existing:
        path.write_text(existing.rstrip() + '\n\n' + marker + '，仅追加）\n\n' + text)
    print(json.dumps({'stage': 'decided', 'decision': decision, 'cagr': cost1['cagr']}), flush=True)
    return decision


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'decide'))
    parser.add_argument('--out', help='explicit fresh directory below output/ for a new research artifact')
    args = parser.parse_args()
    if args.out:
        OUT = book.set_explicit_output(args.out)
    globals()[args.action]()
