#!/usr/bin/env python3
"""One continuous 50-name broad basket. Research account only.

The two-leg book put most of the capital outside the liquidity band and
made about 4 percent. This account holds that band: 50 names, equal weight,
absolute top-150 retention, 15 industry slots, no large-cap sleeve, no
stagger, no score, no P2. One ledger from 2018-01-02 through 2026-09-22.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from collections import Counter
from datetime import datetime, timezone
import json

import pandas as pd

import stable_book as book
OUT = ROOT / 'output/continuous_band_20260928'
HOLDINGS = 50
BUFFER = 100
from quantlab.smallcap.universe import industry_slots

# derived from the mandate weight, not a fixed count (M1): one industry stays <= 25% of the book
INDUSTRY_SLOTS = industry_slots(HOLDINGS)
GROSS = 0.90


def select_band(scores, industries, current):
    """Top-150 retention and 15 names per industry, without a ticker-text tie break.

    Equal scores keep the head's existing order. Those heads are stored in
    liquidity-rank order. Sorting a tie by ticker text keeps only ``000xxx.SZ``
    once hundreds of names share the same financial score.
    """
    if any(not pd.notna(value) for value in scores.values()):
        raise ValueError('band_score_missing')
    order = {code: index for index, code in enumerate(scores)}
    ranked = sorted(scores, key=lambda code: (-float(scores[code]), order[code]))
    held = set(current)
    retained = [code for code in ranked[:HOLDINGS + BUFFER] if code in held]
    chosen = []
    counts = Counter()
    for code in retained + ranked:
        industry = industries.get(code, 'unclassified')
        if code in chosen or counts[industry] >= INDUSTRY_SLOTS:
            continue
        chosen.append(code)
        counts[industry] += 1
        if len(chosen) == HOLDINGS:
            break
    if len(chosen) > HOLDINGS:
        raise ValueError('band_exceeds_50')
    counts = Counter(industries.get(code, 'unclassified') for code in chosen)
    if counts and max(counts.values()) > INDUSTRY_SLOTS:
        raise ValueError('industry_cap_breached')
    return tuple(chosen)


def build_targets(sessions, heads, industries):
    """One basket for the whole window.

    The old segmented accounts created a new policy, and therefore an empty
    basket, at each period boundary. The closes dict was not that reset.
    """
    by_day = {pd.Timestamp(day).normalize(): day for day in heads}
    basket = ()
    targets = []
    for session in sessions:
        key = by_day.get(pd.Timestamp(session).normalize())
        if key is None:
            continue
        new = select_band(heads[key], industries[key], basket)
        previous = set(basket)
        basket = new
        weights = {code: GROSS / len(new) for code in new} if new else {}
        if sum(weights.values()) > GROSS + 1e-8:
            raise ValueError('gross_cap')
        targets.append({
            'signal_date': str(pd.Timestamp(session).date()),
            'weights': weights,
            'active': list(new),
            'sell': sorted(previous - set(new)),
            'sell_only': False,
            'max_exposure': GROSS,
            'sleeve': 0,
            'band_snapshot': str(pd.Timestamp(key).date()) if not isinstance(key, str) else key[:10],
        })
    return targets


def prepare():
    book.prepare_fresh_output(OUT)
    parts = {name: json.loads(path.read_text()) for name, path in book.INPUTS.items()}
    heads, industries = book.merge_band_inputs(parts)
    manifest = json.loads(book.MANIFEST.read_text())
    calendar = pd.DatetimeIndex(book._dated(pd.read_parquet(
        book.MANIFEST.parent / manifest['calendar']['path'])).session)
    sessions = book.window_sessions(calendar)
    if sessions.empty or sessions[0] != book.START or sessions[-1] != book.END:
        book.write_json(OUT / 'BLOCKED.json', {'reason': 'window_not_covered', 'promotion_approved': False})
        return 'blocked'
    targets = build_targets(sessions, heads, industries)
    path = OUT / 'targets.json'
    book.write_json(path, {'targets': targets, 'promotion_approved': False})
    protocol = {
        'version': 'continuous-band-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'construction': 'single_50_name_band_one_ledger',
        'removed': 'large-cap sleeve and five-sleeve slow deployment that produced about 4 percent',
        'question': 'Does the 50-name equal-weight band, held continuously, reach 15 percent net CAGR inside a 25 percent drawdown?',
        'window': [str(book.START.date()), str(book.END.date())],
        'initial_cash': book.INITIAL_CASH,
        'holdings': HOLDINGS,
        'retention_rank': HOLDINGS + BUFFER,
        'industry_slots': INDUSTRY_SLOTS,
        'gross': GROSS,
        'costs': {'slippage': 0.003, 'min_trade_value': 1000},
        'risk': 'no P2, no drawdown scaler, no single-name stop',
        'cagr_gate': {
            'start': '2018-01-02',
            'end': '2026-09-22',
            'formula': '(final_equity/600000)**(365.25/(last_session-first_session).days)-1',
            'includes_cash_before_the_first_fill': True,
            'not_the_gate': 'CAGR from the first filled session is reported and cannot replace this clock. Heads begin in 2019; starting the clock there changes the result by enough to cross 15 percent.',
        },
        'gates': {'min_cagr': 0.15, 'max_drawdown': 0.25, 'recent_positive': True, 'cost_double_cagr_positive': True,
                  'cagr_clock': '2018-01-02_to_2026-09-22'},
        'no_parameter_rescue': True,
        'inputs': {name: book.sha256(file) for name, file in book.INPUTS.items()},
        'manifest': book.sha256(book.MANIFEST),
        'sources': {str(file): book.sha256(file) for file in (Path(__file__), ROOT / 'tests/test_continuous_band.py',
                                                              ROOT / 'scripts/stable_book.py')},
        'targets': {str(path): book.sha256(path)},
        'promotion_approved': False,
    }
    book.write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({'stage': 'prepared', 'decisions': len(targets),
                      'first': targets[0]['signal_date'], 'last': targets[-1]['signal_date']}), flush=True)
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
    import io
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
        book.write_json(OUT / 'BLOCKED.json', {'reason': 'projection_invalid', 'errors': errors, 'promotion_approved': False})
        return 'blocked'
    for multiplier in (1, 2):
        print(json.dumps({'stage': 'replay', 'cost': multiplier}), flush=True)
        result = book.replay_account(dataset, OUT / 'targets.json', start=book.START, end=book.END, cost_mult=multiplier)
        summary = result['summary']
        if summary['first'] != '2018-01-02' or summary['last'] != '2026-09-22':
            raise ValueError('cagr_clock_moved:' + summary['first'] + ':' + summary['last'])
        held = [state for state in result['account'] if state['quantities']]
        if held:
            start_held = pd.Timestamp(held[0]['date'])
            elapsed = (pd.Timestamp(summary['last']) - start_held).days
            summary['cagr_from_first_fill'] = float((summary['final_equity'] / book.INITIAL_CASH) ** (365.25 / elapsed) - 1)
            summary['first_fill'] = str(start_held.date())
        summary['cagr_clock'] = '2018-01-02_to_2026-09-22'
        result['summary'] = summary
        folder = OUT / f'cost{multiplier}'
        folder.mkdir(parents=True, exist_ok=True)
        book.write_json(folder / 'summary.json', result['summary'])
        result['daily_return'].rename('daily_return').to_csv(folder / 'daily_returns.csv', index_label='date')
        if not result['summary']['account_checks_passed']:
            book.write_json(OUT / 'BLOCKED.json', {'reason': 'account_identity_failed', 'cost': multiplier,
                                                   'errors': result['summary']['account_check_errors'],
                                                   'promotion_approved': False})
            return 'blocked'
        print(json.dumps({'stage': 'replay_done', 'cost': multiplier, 'cagr': result['summary']['cagr'],
                          'max_drawdown': result['summary']['max_drawdown'],
                          'recent_return': result['summary']['recent_return']}), flush=True)
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
        decision = 'continuous_band_did_not_reach_15'
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
        '# 连续 50 只宽篮',
        '',
        f"判决：`{decision}`。未通过项：{('、'.join(reasons) if reasons else '无')}。",
        f"扣费年化 {cost1['cagr'] * 100:.2f}%（时钟 2018-01-02 至 2026-09-22，含建仓前现金）。"
        f"从首次持仓 {cost1.get('first_fill')} 起算是 {cost1.get('cagr_from_first_fill', float('nan')) * 100:.2f}%，这个数不参与过线。"
        f"最大回撤 {-cost1['max_drawdown'] * 100:.2f}%，"
        f"2024-12-31 至 2026-09-22 累计 {cost1['recent_return'] * 100:.2f}%，"
        f"期末 {cost1['final_equity']:.2f} 元。成本翻倍年化 {cost2['cagr'] * 100:.2f}%。",
        '一条账，50 只等权宽篮，前 150 名保留，每行业 15 只，没有大盘腿，没有错峰慢投入，没有 P2。',
        '这不是实盘许可。promotion_approved=false。',
        '',
    ])
    (OUT / 'REVIEW.md').write_text(text)
    path = ROOT / 'docs/RESEARCH_CONTINUITY.md'
    marker = '## 连续 50 只宽篮（2026-09-28'
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
