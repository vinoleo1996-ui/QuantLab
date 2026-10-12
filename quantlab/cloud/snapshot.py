"""Versioned, append-only local snapshots of cloud datasets.

Layout: ``data/cloud/<dataset>/<snapshot_id>/part-<start>-<end>.parquet`` plus a
``.json`` receipt per part and ``manifest.json`` once every part is verified.
Parts already on disk are re-hashed and reused, so an interrupted pull resumes
by being run again. A finished snapshot is never rewritten.

Each snapshot must name how its rows become usable: either a column that holds
``available_at`` or an explicit ``availability_rule``. Without one it is refused.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quantlab.cloud.client import CloudError

ROOT = Path(__file__).resolve().parents[2]
CLOUD_ROOT = ROOT / 'data/cloud'
SINGLE_SEGMENT_MAX_ROWS = 300_000


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def month_windows(date_min, date_max):
    start, end = pd.Timestamp(str(date_min)), pd.Timestamp(str(date_max))
    edges = pd.date_range(start.to_period('M').to_timestamp(), end, freq='MS')
    windows = []
    for left in edges:
        right = min(left + pd.offsets.MonthEnd(0), end)
        windows.append((max(left, start).strftime('%Y%m%d'), right.strftime('%Y%m%d')))
    return windows


def plan_segments(entry, segment='auto', start=None, end=None):
    """Full table in one part when small or undated; otherwise monthly windows."""
    lo, hi = start or entry.get('date_min'), end or entry.get('date_max')
    if segment == 'all' or not entry.get('date_col') or not lo or not hi:
        return [(None, None)]
    if segment == 'auto' and start is None and end is None and int(entry.get('nrows') or 0) <= SINGLE_SEGMENT_MAX_ROWS:
        return [(None, None)]
    return month_windows(lo, hi)


def _availability(columns, availability_rule):
    timed = [c for c in columns if 'available_at' in c]
    if not timed and not availability_rule:
        raise CloudError('no_available_at_column_and_no_rule')
    return timed


def _part_name(lo, hi):
    return f'part-{lo or "all"}-{hi or "all"}'


def snapshot_dataset(client, dataset_id, snapshot_id, *, root=CLOUD_ROOT, segment='auto', start=None, end=None,
                     availability_rule=None, progress=print):
    target = Path(root) / dataset_id / snapshot_id
    manifest_path = target / 'manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    entry = client.catalog().get(dataset_id)
    if entry is None:
        raise CloudError(f'unknown_dataset:{dataset_id}')
    schema = client.schema(dataset_id)
    columns = [c['name'] for c in schema['columns']]
    timed = _availability(columns, availability_rule)
    target.mkdir(parents=True, exist_ok=True)
    parts = []
    windows = plan_segments(entry, segment, start, end)
    for index, (lo, hi) in enumerate(windows):
        name = _part_name(lo, hi)
        data_path, receipt_path = target / f'{name}.parquet', target / f'{name}.json'
        if data_path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if sha256_bytes(data_path.read_bytes()) == receipt['sha256']:
                parts.append(receipt)
                continue
        body, rows, headers = client.export(dataset_id, start=lo, end=hi)
        receipt = dict(part=name, start=lo, end=hi, rows=rows, bytes=len(body), sha256=sha256_bytes(body),
                       fetched_at=_now(), server_row_count=headers.get('x-row-count'))
        data_path.write_bytes(body)
        receipt_path.write_text(json.dumps(receipt, indent=2))
        parts.append(receipt)
        if progress:
            progress(json.dumps(dict(dataset=dataset_id, part=name, index=index + 1, of=len(windows), rows=rows)))
    total = sum(p['rows'] for p in parts)
    full_range = start is None and end is None
    expected = int(entry.get('nrows') or 0)
    manifest = dict(dataset_id=dataset_id, snapshot_id=snapshot_id, created_at=_now(), source=client.base_url,
                    catalog_entry=entry, columns=columns, available_at_columns=timed,
                    availability_rule=availability_rule, requested=dict(start=start, end=end, segment=segment),
                    parts=parts, rows=total, catalog_rows=expected,
                    complete=bool(full_range and total == expected) or not full_range,
                    rows_missing_vs_catalog=(expected - total) if full_range else None,
                    promotion_approved=bool(entry.get('promotion_approved', False)))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def latest_snapshot(dataset_id, root=CLOUD_ROOT):
    base = Path(root) / dataset_id
    done = sorted(p.name for p in base.iterdir() if (p / 'manifest.json').exists()) if base.exists() else []
    if not done:
        raise CloudError(f'no_complete_snapshot:{dataset_id}')
    return done[-1]


def load_snapshot(dataset_id, snapshot_id=None, *, root=CLOUD_ROOT, columns=None, require_complete=True):
    snapshot_id = snapshot_id or latest_snapshot(dataset_id, root)
    target = Path(root) / dataset_id / snapshot_id
    if not (target / 'manifest.json').exists():
        raise CloudError(f'no_complete_snapshot:{dataset_id}/{snapshot_id}')
    manifest = json.loads((target / 'manifest.json').read_text())
    if require_complete and not manifest['complete']:
        raise CloudError(f'incomplete_snapshot:{dataset_id}/{snapshot_id}:{manifest["rows_missing_vs_catalog"]}')
    frames = []
    for part in manifest['parts']:
        payload = (target / f'{part["part"]}.parquet').read_bytes()
        if sha256_bytes(payload) != part['sha256']:
            raise CloudError(f'part_hash_mismatch:{part["part"]}')
        frames.append(pq.read_table(pa_buffer(payload), columns=columns).to_pandas())
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or manifest['columns'])
    frame.attrs['cloud_manifest'] = dict(dataset_id=dataset_id, snapshot_id=snapshot_id, rows=manifest['rows'])
    return frame


def pa_buffer(payload):
    import pyarrow as pa
    return pa.BufferReader(payload)
