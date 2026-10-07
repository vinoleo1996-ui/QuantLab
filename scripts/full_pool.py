#!/usr/bin/env python3
"""50 names drawn from the whole qualified pool, not ranks 3001-5000.

Every decision day can buy. The account still holds at most 50 names, spaced
through the full liquidity ranking so the book is not the old tail and not the
first codes in alphabetical order.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from collections import Counter
from datetime import datetime, timezone
import io
import json

import pandas as pd

import extend_2018_band as early
import stable_book as book

OUT = ROOT / 'output/full_pool_20260928'
HOLDINGS = 50
INDUSTRY_SLOTS = 15
GROSS = 0.90


def decision_days(calendar, start=book.START, end=book.END, step=20):
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize()
    window = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    return tuple(pd.Timestamp(day) for index, day in enumerate(window) if index % step == 0)


def spaced_indexes(count, holdings):
    if count <= 0 or holdings <= 0:
        return ()
    if count <= holdings:
        return tuple(range(count))
    raw = [round(i * (count - 1) / (holdings - 1)) for i in range(holdings)]
    chosen = []
    for index in raw:
        while index in chosen and index + 1 < count:
            index += 1
        if index not in chosen:
            chosen.append(index)
    return tuple(chosen)


def select_full_pool(ranked, industries, current, holdings=HOLDINGS, slots=INDUSTRY_SLOTS):
    """Keep names still in the pool, then fill from positions spread across it."""
    counts = Counter()
    chosen = []
    qualified = set(ranked)

    def add(code):
        if code in chosen or code not in qualified:
            return False
        industry = industries.get(code, 'unclassified')
        if counts[industry] >= slots:
            return False
        chosen.append(code)
        counts[industry] += 1
        return True

    for code in current:
        add(code)
        if len(chosen) == holdings:
            return tuple(chosen)
    for index in spaced_indexes(len(ranked), holdings):
        add(ranked[index])
        if len(chosen) == holdings:
            return tuple(chosen)
    for code in ranked:
        add(code)
        if len(chosen) == holdings:
            break
    return tuple(chosen)


def build_targets(sessions, heads, industries):
    by_day = {pd.Timestamp(day).normalize(): day for day in heads}
    basket = ()
    targets = []
    for session in sessions:
        key = by_day.get(pd.Timestamp(session).normalize())
        if key is None:
            continue
        order = list(heads[key])
        new = select_full_pool(order, industries[key], basket)
        previous = set(basket)
        basket = new
        weights = {code: GROSS / len(new) for code in new} if new else {}
        targets.append({
            'signal_date': str(pd.Timestamp(session).date()),
            'weights': weights,
            'active': list(new),
            'sell': sorted(previous - set(new)),
            'sell_only': False,
            'max_exposure': GROSS,
            'sleeve': 0,
        })
    return targets


def pool_on(amount_history, industry_history, day):
    observations = []
    for code, (dates, values) in amount_history.items():
        amount = early.amount_on(dates, values, day)
        if amount is not None:
            observations.append((amount, code))
    observations.sort(key=lambda item: (-item[0], item[1]))
    ranked = [code for _amount, code in observations]
    industries = {code: early.industry_on(industry_history, code, day) for code in ranked}
    return ranked, industries


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(book.MANIFEST.read_text())
    calendar = pd.DatetimeIndex(book._dated(pd.read_parquet(
        book.MANIFEST.parent / manifest['calendar']['path'])).session)
    sessions = book.window_sessions(calendar)
    days = decision_days(calendar)
    if not days or days[0] != book.START:
        book.write_json(OUT / 'BLOCKED.json', {'reason': 'grid_does_not_start', 'promotion_approved': False})
        return 'blocked'
    amount_history = early.load_amount_history(end=book.END)
    industry_history = early.load_industry_history()
    heads, industries = {}, {}
    sizes = {}
    for index, day in enumerate(days):
        if index % 20 == 0:
            print(json.dumps({'stage': 'pool', 'day': str(day.date()), 'index': index, 'total': len(days)}), flush=True)
        ranked, group = pool_on(amount_history, industry_history, day)
        if not ranked:
            continue
        key = str(day.date())
        heads[key] = ranked
        industries[key] = group
        sizes[key] = len(ranked)
    targets = build_targets(sessions, heads, industries)
    if not targets or targets[0]['signal_date'] != '2018-01-02' or not targets[0]['weights']:
        book.write_json(OUT / 'BLOCKED.json', {
            'reason': 'january_2018_still_empty',
            'first': None if not targets else targets[0]['signal_date'],
            'sizes': {day: sizes[day] for day in list(sizes)[:6]},
            'promotion_approved': False,
        })
        return 'blocked'
    path = OUT / 'targets.json'
    book.write_json(path, {'targets': targets, 'pool_sizes': sizes, 'promotion_approved': False})
    protocol = {
        'version': 'full-pool-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'construction': '50 names spaced across the whole history-qualified pool',
        'removed': 'ADV ranks 3001-5000 and the 3001-name minimum',
        'question': 'Does selecting 50 names from the full qualified pool, including January 2018, reach 15 percent net CAGR inside a 25 percent drawdown?',
        'pool': 'every name with 253 positive-amount observations and a trade on the decision day; no rank window and no 5 million floor',
        'selection': 'equal-spaced liquidity ranks, industry cap 15, keep names that remain in the pool',
        'holdings': HOLDINGS,
        'gross': GROSS,
        'first_target': targets[0]['signal_date'],
        'first_pool_size': sizes.get(targets[0]['signal_date']),
        'first_holdings': len(targets[0]['weights']),
        'decision_days': len(days),
        'cagr_gate': {
            'start': '2018-01-02', 'end': '2026-09-22',
            'from_first_fill_is_not_the_gate': True,
        },
        'gates': {'min_cagr': 0.15, 'max_drawdown': 0.25, 'recent_positive': True,
                  'cost_double_cagr_positive': True},
        'no_parameter_rescue': True,
        'inputs': {'manifest': book.sha256(book.MANIFEST)},
        'sources': {str(file): book.sha256(file) for file in (
            Path(__file__), ROOT / 'tests/test_full_pool.py',
            ROOT / 'scripts/extend_2018_band.py', ROOT / 'scripts/stable_book.py')},
        'targets': {str(path): book.sha256(path)},
        'promotion_approved': False,
    }
    book.write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'stage': 'prepared', 'first': targets[0]['signal_date'],
                      'holdings': len(targets[0]['weights']),
                      'pool': sizes.get(targets[0]['signal_date']),
                      'decisions': len(targets)}), flush=True)
    return 'prepared'


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
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
        book.write_json(folder / 'summary.json', summary)
        if not summary['account_checks_passed']:
            book.write_json(OUT / 'BLOCKED.json', {'reason': 'account_identity_failed', 'cost': multiplier,
                                                   'errors': summary['account_check_errors'],
                                                   'promotion_approved': False})
            return 'blocked'
        result['daily_return'].rename('daily_return').to_csv(folder / 'daily_returns.csv', index_label='date')
        print(json.dumps({'stage': 'replay_done', 'cost': multiplier, 'cagr': summary['cagr'],
                          'max_drawdown': summary['max_drawdown'],
                          'recent_return': summary['recent_return']}), flush=True)
    return 'ran'


def decide():
    cost1 = json.loads((OUT / 'cost1' / 'summary.json').read_text())
    cost2 = json.loads((OUT / 'cost2' / 'summary.json').read_text())
    if cost1.get('first') != '2018-01-02':
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
        decision = 'full_pool_did_not_reach_15'
    elif 'cost_double' in reasons:
        decision = 'not_retained_cost_double_not_positive'
    else:
        decision = 'research_candidate_not_promoted'
    verdict = {'decision': decision, 'reasons': reasons, 'cagr': cost1['cagr'],
               'max_drawdown': cost1['max_drawdown'], 'recent_return': cost1['recent_return'],
               'cost2_cagr': cost2['cagr'], 'first_fill': cost1.get('first_fill'),
               'cagr_from_first_fill_not_used': cost1.get('cagr_from_first_fill'),
               'promotion_approved': False}
    book.write_json(OUT / 'DECISION.json', verdict)
    text = '\n'.join([
        '# 全池 50 只',
        '',
        f"判决：`{decision}`。未通过项：{('、'.join(reasons) if reasons else '无')}。",
        f"扣费年化 {cost1['cagr'] * 100:.2f}%（2018-01-02 至 2026-09-22）。"
        f"首次持仓 {cost1.get('first_fill')}，从那天起算 {cost1.get('cagr_from_first_fill', float('nan')) * 100:.2f}%，不参与过线。",
        f"最大回撤 {-cost1['max_drawdown'] * 100:.2f}%，2024-12-31 至 2026-09-22 累计 {cost1['recent_return'] * 100:.2f}%，"
        f"期末 {cost1['final_equity']:.2f} 元。成本翻倍年化 {cost2['cagr'] * 100:.2f}%。",
        '候选池是当天全部历史够长的股票，不再使用第3001到第5000名。账户最多 50 只，按流动性排名均匀抽取，每行业最多 15 只。',
        'promotion_approved=false。',
        '',
    ])
    (OUT / 'REVIEW.md').write_text(text)
    path = ROOT / 'docs/RESEARCH_CONTINUITY.md'
    marker = '## 全池 50 只（2026-09-28'
    existing = path.read_text()
    if marker not in existing:
        path.write_text(existing.rstrip() + '\n\n' + marker + '，仅追加）\n\n' + text)
    print(json.dumps({'stage': 'decided', 'decision': decision, 'cagr': cost1['cagr']}), flush=True)
    return decision


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'decide'))
    args = parser.parse_args()
    globals()[args.action]()
