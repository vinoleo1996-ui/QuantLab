#!/usr/bin/env python3
"""Forward event-risk reviews of the current book (record only; never changes the account).

  .venv2/bin/python scripts/event_agent.py --max 3
Holdings default to the RP3 forward snapshot M2 path 0_1; names reviewed within the last 20 sessions are
skipped so the daily LLM budget rotates through the book. Output: output/event_agent/reviews.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from quantlab import llm
from quantlab.agents.event_risk import evidence, review
from quantlab.smallcap.panel import Panel, latest_version

OUT = ROOT / 'output/event_agent/reviews.jsonl'
SNAPSHOT = ROOT / 'output/p2_forward_rp3/snapshot_2026-09-30.json'
TABLES = ('unlock', 'holder_change', 'pledge', 'forecast', 'billboard')


def holdings(path, key):
    snapshot = json.loads(Path(path).read_text())
    book = snapshot['m2'][key]
    return sorted(book['weights'] if isinstance(book, dict) and 'weights' in book else book)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=3)
    parser.add_argument('--snapshot', default=str(SNAPSHOT))
    parser.add_argument('--path', default='0_1')
    args = parser.parse_args(argv)
    panel = Panel(latest_version())
    fields = {name: panel.field(name) for name in ('adj_close', 'close', 'limit_down', 'tradestatus')}
    asof = panel.sessions[-1]
    tables = {name: pd.read_parquet(ROOT / f'data/smallcap_info/{name}.parquet') for name in TABLES}
    done = [json.loads(line) for line in OUT.read_text().splitlines()] if OUT.exists() else []
    recent = {row['code'] for row in done if pd.Timestamp(row['asof']) > asof - pd.Timedelta(days=28)}
    queue = [code for code in holdings(args.snapshot, args.path) if code not in recent][:args.max]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for code in queue:
        facts = evidence(code, asof, tables, fields)
        try:
            row = review(code, str(asof.date()), facts)
        except llm.LLMError as exc:
            row = dict(code=code, asof=str(asof.date()), evidence=facts, abstained=True, abstain_reason=str(exc),
                       verdict=None, rating=None, order=None)
        row.update(reviewed_at=datetime.now(timezone.utc).isoformat(), panel=latest_version(), source_book=f'{SNAPSHOT.name}:{args.path}')
        with OUT.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=float) + '\n')
        print(json.dumps(dict(code=code, verdict=row['verdict'], abstained=row['abstained'],
                              reason=row.get('abstain_reason')), ensure_ascii=False), flush=True)
        if row.get('abstain_reason') == 'daily_llm_call_cap_reached':
            break
    print(json.dumps(dict(reviewed=len(queue), llm_calls_today=llm.calls_today())))


if __name__ == '__main__':
    main()
