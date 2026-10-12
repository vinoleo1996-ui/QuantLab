"""Pull cloud datasets into versioned local snapshots (resumable; run again to continue).

    .venv2/bin/python scripts/cloud_pull.py --snapshot 20261012 sw_l1_members financials_pit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.cloud import CloudClient, CloudError, snapshot_dataset  # noqa: E402

AVAILABILITY_RULES = {
    'float_mktcap': 'trade_date 18:00 Asia/Shanghai (end-of-day daily_basic)',
    'hk_hold_ashare': 'trade_date + 1 session 09:00 Asia/Shanghai (HKEX publishes next morning)',
    'hsgt_top10': 'trade_date 18:00 Asia/Shanghai',
}
P4_DEFAULT = ['sw_l1_members', 'financials_pit', 'dividend_tushare', 'top_inst_labeled', 'margin_detail',
              'share_float_pit']


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='*', default=P4_DEFAULT)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--segment', default='auto', choices=('auto', 'month', 'all'))
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--wait', type=float, default=0, help='hours to keep retrying while the service is down')
    args = parser.parse_args(argv)
    client = CloudClient()
    deadline = time.time() + args.wait * 3600
    for dataset in args.datasets:
        while True:
            try:
                manifest = snapshot_dataset(client, dataset, args.snapshot, segment=args.segment, start=args.start,
                                            end=args.end, availability_rule=AVAILABILITY_RULES.get(dataset))
                break
            except CloudError as exc:
                if 'unreachable' not in str(exc) or time.time() > deadline:
                    raise
                print(json.dumps(dict(dataset=dataset, waiting=str(exc)[:120])), flush=True)
                time.sleep(120)
        print(json.dumps(dict(dataset=dataset, rows=manifest['rows'], catalog_rows=manifest['catalog_rows'],
                              complete=manifest['complete'], parts=len(manifest['parts']))), flush=True)


if __name__ == '__main__':
    main()
