#!/usr/bin/env python3
"""P4 daily update after the close (research record only; never trades).

  .venv2/bin/python scripts/p4_daily.py --date 2026-10-13

1. pull the cloud tables needed after the v4 cutoff with explicit start/end (large tables never unbounded):
   daily bars and index from 2026-09-23, daily_basic from 2026-09-01 (seam window), dividends from 2026-09-01;
2. rebuild dataset v5 and the wide panel from that daily snapshot;
3. factor-lab forward update and event-agent reviews (shared LLM cap).
Each step logs to output/p4_daily/<date>.jsonl; a failed step stops the later ones.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab.cloud import CloudClient, snapshot_dataset  # noqa: E402

PY = str(ROOT / '.venv2/bin/python')
PULLS = (('daily_ohlcv_raw_pit', '20260923'), ('index_daily_pit', '20260923'), ('daily_basic_full', '20260901'),
         ('dividend_tushare', '20260901'))


def log(path, **row):
    row['at'] = datetime.now(timezone.utc).isoformat()
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(row, ensure_ascii=False, default=str), flush=True)


def run(cmd, path, step):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log(path, step=step, returncode=result.returncode, tail=(result.stdout or result.stderr)[-1500:])
    if result.returncode != 0:
        raise SystemExit(f'{step}_failed')
    return result.stdout


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    parser.add_argument('--reviews', type=int, default=8)
    args = parser.parse_args(argv)
    day = args.date.replace('-', '')
    out = ROOT / 'output/p4_daily'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.date}.jsonl'
    snapshot = f'daily_{day}'
    client = CloudClient()
    for dataset, start in PULLS:
        manifest = snapshot_dataset(client, dataset, snapshot, segment='all', start=start, end=day, progress=None,
                                    availability_rule='dividend notice next session' if dataset == 'dividend_tushare' else None)
        log(path, step='pull', dataset=dataset, rows=manifest['rows'], start=start, end=day)
    built = run([PY, 'scripts/build_dataset_v5.py', '--end', args.date, '--daily-snapshot', snapshot], path, 'dataset_v5')
    target = json.loads(built[built.index('{'):])['target']
    run([PY, '-c', f"from quantlab.smallcap import panel; panel.build(source_manifest='{target}/manifest.json', end='{args.date}')"],
        path, 'wide_panel')
    run([PY, 'scripts/factor_lab.py', 'forward'], path, 'factor_lab_forward')
    run([PY, 'scripts/event_agent.py', '--max', str(args.reviews)], path, 'event_agent')
    log(path, step='done', snapshot=snapshot, dataset=target)


if __name__ == '__main__':
    main()
