"""Build dataset v5 from the frozen v4 extension snapshot and cloud snapshots.

    .venv2/bin/python scripts/build_dataset_v5.py --end 2026-10-12

Inputs (cloud snapshots; missing required tables abort, no fallback source):
  daily_ohlcv_raw_pit         raw bars + adj_factor + limits + tradestatus + ST (required)
  daily_basic_full            turnover/pb for new rows and closes for the seam check (required, window snapshot)
  stock_basic_incl_delisted   listing records for new codes (optional)
  index_daily_pit             benchmark extension (optional)
  dividend_tushare            implemented distributions after the cutoff (optional)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from quantlab import dataset_v5  # noqa: E402
from quantlab.cloud import CloudError, load_snapshot  # noqa: E402

BASE = ROOT / 'output/account_2025_20260927/extension/dataset/manifest.json'
REQUIRED = {'daily': ('daily_ohlcv_raw_pit', '20261012'), 'basic': ('daily_basic_full', '20261012w')}
OPTIONAL = {'stock_basic': ('stock_basic_incl_delisted', '20261012'), 'index_daily': ('index_daily_pit', '20261012'),
            'dividend': ('dividend_tushare', None)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--end', required=True)
    parser.add_argument('--overlap', nargs=2, default=('2026-09-01', '2026-09-22'))
    args = parser.parse_args(argv)
    frames, snapshots, missing = {}, {}, []
    for name, (dataset, snapshot) in {**REQUIRED, **OPTIONAL}.items():
        try:
            frames[name] = load_snapshot(dataset, snapshot)
            snapshots[name] = frames[name].attrs['cloud_manifest']
        except CloudError as exc:
            if name in REQUIRED:
                missing.append(f'{name}<-{dataset}:{exc}')
    if missing:
        print(json.dumps(dict(status='blocked_data', missing=missing), ensure_ascii=False, indent=1))
        return 2
    basic = frames['basic']
    basic_dates = pd.to_datetime(basic['trade_date']).dt.tz_localize(None).dt.normalize()
    seam = basic[basic_dates <= pd.Timestamp(args.overlap[1])]
    closes = dict(zip(zip(seam['instrument_id'], basic_dates[seam.index]), seam['close']))
    cloud = dataset_v5.prepare_combined(frames['daily'], basic, overlap_close=closes)
    cloud.update({k: frames.get(k) for k in OPTIONAL})
    target = dataset_v5.build(BASE, cloud, end=args.end, overlap=tuple(args.overlap), cloud_snapshots=snapshots)
    manifest = json.loads((target / 'manifest.json').read_text())
    print(json.dumps(dict(status='built', target=str(target), as_of=manifest['as_of'],
                          provenance={k: v for k, v in manifest['v5_provenance'].items() if k != 'new_sessions'}),
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
