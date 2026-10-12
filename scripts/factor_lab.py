#!/usr/bin/env python3
"""AI factor lab driver.

  propose   one LLM call (OpenRouter free models), new formulas appended to candidates.jsonl
  screen    historical screen of proposed candidates; each screened candidate is a registered trial
            in family P4_factorlab (outcome pass / no_improvement)
  forward   daily next-day rank IC after each candidate's submission date on the latest panel;
            e-process update, online e-BH admission, e-detector retirement
  status    counts by state
Candidates and outcomes are append-only (output/factor_lab/).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantlab import research_trials as T
from quantlab.alpha import features as F
from quantlab.alpha.labels import adjusted_open, forward_open_returns
from quantlab.factor_lab import dsl, proposer, referee
from quantlab.smallcap import universe
from quantlab.smallcap.panel import Panel, latest_version

OUT = ROOT / 'output/factor_lab'
CANDIDATES, EVENTS = OUT / 'candidates.jsonl', OUT / 'events.jsonl'
from quantlab.llm import USAGE  # noqa: E402  shared across LLM callers
STATE = OUT / 'forward_state.json'
FIELDS = ('open', 'high', 'low', 'close', 'adj_close', 'amount', 'volume', 'turnover', 'float_mktcap',
          'tradestatus', 'limit_up', 'is_st')
OBSERVE_START, OBSERVE_STEP = '2015-01-05', 20


def now():
    return datetime.now(timezone.utc).isoformat()


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=float) + '\n')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def candidates():
    """Latest state per candidate id (candidates.jsonl holds proposals, events.jsonl later states)."""
    out = {row['id']: row for row in read(CANDIDATES)}
    for event in read(EVENTS):
        out[event['id']] = dict(out[event['id']], **event)
    return out


def stage_propose(n):
    known = candidates()
    seen = [row['formula'] for row in known.values()]
    result = proposer.propose(list(F.ALL_FEATURES), seen, USAGE, n=n)
    added = 0
    for item in result['candidates']:
        if item['id'] in known:
            continue
        append(CANDIDATES, dict(item, state='proposed', proposed_at=now(), model=result['model'],
                                prompt_sha=result['prompt_sha']))
        added += 1
    print(json.dumps(dict(model=result['model'], returned=len(result['candidates']), added=added,
                          invalid=result['invalid']), ensure_ascii=False))


def _panel():
    panel = Panel(latest_version())
    return panel, {name: panel.field(name) for name in FIELDS}


def stage_screen():
    todo = [row for row in candidates().values() if row['state'] == 'proposed']
    if not todo:
        print(json.dumps(dict(screened=0)))
        return
    panel, fields = _panel()
    pools = universe.rolling_features(panel, with_st=True)
    sessions = panel.sessions
    days = sessions[(sessions >= pd.Timestamp(OBSERVE_START))][::OBSERVE_STEP]
    days = days[days <= sessions[-22]]
    label = forward_open_returns(fields)
    price = F.PriceVolume(fields)
    members = {day: [c for c in universe.pool_on(day, pools, 'adv', band='relative', exclude_st=True,
                                                 min_history=252) if not c.startswith('bj.')] for day in days}
    long = lambda wide, name: pd.concat([pd.DataFrame(dict(date=d, code=m, **{name: wide.loc[d].reindex(m).to_numpy()}))
                                         for d, m in members.items() if m], ignore_index=True)
    labels = long(label, 'label')
    library = pd.concat([long(frame, name).set_index(['date', 'code']) for name, frame in price.m.items()], axis=1).reset_index()
    panels = dsl.panels_from_fields(fields)
    for row in todo:
        try:
            values = long(dsl.evaluate(row['formula'], panels), 'value')
            result = referee.screen(values, labels, library, set(days))
            state = 'screen_pass' if result['passed'] else 'screen_fail'
        except Exception as exc:
            result, state = dict(error=f'{type(exc).__name__}:{exc}'[:200]), 'technical_failure'
        trial = f'{row["id"]}'
        T.register('P4_factorlab', trial, prereg='docs/prereg/P4_FACTORLAB.md', source='scripts/factor_lab.py',
                   data=f'panel {latest_version()}', segment=f'screen {referee.SCREEN}; holdout {referee.HOLDOUT}',
                   outcome={'screen_pass': 'pass', 'screen_fail': 'no_improvement'}.get(state, 'technical_failure'),
                   note=f'{row["formula"][:120]} | {json.dumps({k: result.get(k) for k in ("screen_mean_ic", "screen_t", "holdout_t", "max_library_corr")}, default=float)}')
        append(EVENTS, dict(id=row['id'], state=state, screened_at=now(), screen=result,
                            submitted_at=str(sessions[-1].date()) if state == 'screen_pass' else None))
        print(json.dumps(dict(id=row['id'], formula=row['formula'], state=state,
                              **{k: result.get(k) for k in ('screen_mean_ic', 'screen_t', 'holdout_mean_ic',
                                                            'holdout_t', 'max_library_corr', 'closest_feature')}),
                         default=float, ensure_ascii=False), flush=True)


def stage_forward():
    live = {k: v for k, v in candidates().items() if v['state'] in ('screen_pass', 'admitted')}
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not live:
        print(json.dumps(dict(live=0)))
        return
    panel, fields = _panel()
    pools = universe.rolling_features(panel, with_st=True)
    adj_open = adjusted_open(fields)
    next_day = adj_open.shift(-2) / adj_open.shift(-1) - 1
    panels = dsl.panels_from_fields(fields)
    sessions = panel.sessions
    streams = {}
    for cid, row in live.items():
        saved = state.get(cid, {})
        stream = referee.ForwardStream(sign=float(row['screen']['sign']), submitted_at=row['submitted_at'],
                                       **{k: saved[k] for k in ('wealth', 'observations', 'retire_processes',
                                                                'admitted_at', 'retired_at') if k in saved})
        values = dsl.evaluate(row['formula'], panels)
        done = {d for d, _ in stream.observations}
        for day in sessions[(sessions > pd.Timestamp(stream.submitted_at)) & (sessions <= sessions[-3])]:
            if str(day) in done:
                continue
            members = [c for c in universe.pool_on(day, pools, 'adv', band='relative', exclude_st=True,
                                                   min_history=252) if not c.startswith('bj.')]
            ic = values.loc[day].reindex(members).corr(next_day.loc[day].reindex(members), method='spearman')
            if np.isfinite(ic):
                stream.update(day, ic)
        streams[cid] = stream
    admitted = referee.admit(streams, sessions[-1])
    STATE.write_text(json.dumps({k: vars(s) for k, s in streams.items()}, default=str, indent=1))
    print(json.dumps(dict(live=len(streams), admitted=sorted(admitted),
                          wealth={k: round(s.wealth, 3) for k, s in streams.items()},
                          observations={k: len(s.observations) for k, s in streams.items()})))


def stage_status():
    rows = candidates().values()
    counts = pd.Series([r['state'] for r in rows]).value_counts().to_dict() if rows else {}
    print(json.dumps(dict(states=counts, llm_calls_today=proposer.calls_today(USAGE),
                          family_trials=T.count('P4_factorlab', include_diagnostics=False))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('propose', 'screen', 'forward', 'status'))
    parser.add_argument('-n', type=int, default=10)
    args = parser.parse_args()
    {'propose': lambda: stage_propose(args.n), 'screen': stage_screen, 'forward': stage_forward,
     'status': stage_status}[args.stage]()
