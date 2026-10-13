"""Register a pushed data bundle (``~/QuantLab/data_v2_bundle``) as local cloud snapshots.

Each bundle file is hashed, its row count is checked against the bundle manifest, and it is
hard-linked (never moved or rewritten) into ``data/cloud/<dataset>/<snapshot>/part-all-all.parquet``
with a snapshot manifest in the same layout as ``snapshot.snapshot_dataset``, so ``load_snapshot``
reads it unchanged. A row-count mismatch is registered as incomplete and refused by default loads.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from quantlab.cloud.snapshot import CLOUD_ROOT

BUNDLE_ROOT = Path.home() / 'QuantLab/data_v2_bundle'
DATASET_OF = {
    'bars/daily_ohlcv_raw_pit.parquet': 'daily_ohlcv_raw_pit',
    'corporate_actions/dividend_tushare.parquet': 'dividend_tushare',
    'corporate_actions/dividend_by_exdate_tushare.parquet': 'dividend_by_exdate',
    'dragon_tiger/top_inst_labeled.parquet': 'top_inst_labeled',
    'dragon_tiger/top_list.parquet': 'top_list',
    'ep/ep_pit_daily_tushare.parquet': 'ep_pit_daily',
    'financials_pit/balancesheet_pit_tushare.parquet': 'balancesheet_pit',
    'financials_pit/bs_cf_pit_tushare.parquet': 'bs_cf_pit',
    'financials_pit/cashflow_pit_tushare.parquet': 'cashflow_pit',
    'financials_pit/financials_pit_tushare.parquet': 'financials_pit',
    'float_mktcap/float_mktcap_daily_from2015_tushare.parquet': 'float_mktcap',
    'index/index_daily_pit.parquet': 'index_daily_pit',
    'margin/margin_detail_daily_2018plus.parquet': 'margin_detail',
    'margin/margin_exchange_daily_merged.parquet': 'margin_exchange',
    'meta/stock_basic_incl_delisted.parquet': 'stock_basic_incl_delisted',
    'northbound/hk_hold_ashare_daily_2018plus.parquet': 'hk_hold_ashare',
    'northbound/hsgt_top10_daily_2018plus.parquet': 'hsgt_top10',
    'northbound/moneyflow_hsgt_daily_2014plus.parquet': 'moneyflow_hsgt',
    'share_float/share_float_pit.parquet': 'share_float_pit',
    'sw_l1_pit/industry_members_sw_l1_tushare.parquet': 'sw_l1_members',
    'valuation/daily_basic_valuation_tushare.parquet': 'daily_basic_valuation',
    'windows/daily_basic_20250901_to_latest.parquet': 'daily_basic_window',
    'windows/etf_daily_cross_asset_5.parquet': 'etf_daily_cross_asset',
    'windows/qdii_nav_513100_513500.parquet': 'qdii_nav',
    'windows/share_float_window_m30_p90.parquet': 'share_float_window',
}


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def bundle_entries(root=BUNDLE_ROOT):
    """``{relative_path: expected_rows}`` from MANIFEST.json and the windows/*_manifest.json files."""
    root = Path(root)
    expected = {row['path']: row.get('nrows') for row in json.loads((root / 'MANIFEST.json').read_text())}
    for manifest in (root / 'windows').glob('*_manifest.json'):
        meta = json.loads(manifest.read_text())
        data = f'windows/{manifest.name.replace("_manifest.json", ".parquet")}'
        if (root / data).exists():
            expected[data] = meta.get('rows')
    return expected


def _link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def register_bundle(snapshot_id, root=BUNDLE_ROOT, cloud_root=CLOUD_ROOT, progress=print):
    root, cloud_root = Path(root), Path(cloud_root)
    report = []
    for relative, expected in sorted(bundle_entries(root).items()):
        source = root / relative
        dataset = DATASET_OF.get(relative, Path(relative).stem)
        target = cloud_root / dataset / snapshot_id
        manifest_path = target / 'manifest.json'
        if manifest_path.exists():
            report.append(json.loads(manifest_path.read_text()) | {'skipped': 'already_registered'})
            continue
        parquet = pq.ParquetFile(source)
        rows = parquet.metadata.num_rows
        columns = parquet.schema_arrow.names
        digest = sha256_file(source)
        target.mkdir(parents=True, exist_ok=True)
        _link(source, target / 'part-all-all.parquet')
        if sha256_file(target / 'part-all-all.parquet') != digest:
            raise RuntimeError(f'link_hash_mismatch:{relative}')
        part = dict(part='part-all-all', start=None, end=None, rows=rows, bytes=source.stat().st_size, sha256=digest,
                    fetched_at=datetime.now(timezone.utc).isoformat(), server_row_count=expected)
        complete = expected is None or int(expected) == rows
        manifest = dict(dataset_id=dataset, snapshot_id=snapshot_id, created_at=part['fetched_at'],
                        source=f'bundle:{source}', catalog_entry=dict(bundle_path=relative, bundle_rows=expected),
                        columns=columns, available_at_columns=[c for c in columns if 'available_at' in c],
                        availability_rule=None, requested=dict(start=None, end=None, segment='bundle'), parts=[part],
                        rows=rows, catalog_rows=expected if expected is not None else rows, complete=complete,
                        rows_missing_vs_catalog=None if expected is None else int(expected) - rows,
                        promotion_approved=False)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        report.append(manifest)
        if progress:
            progress(json.dumps(dict(dataset=dataset, rows=rows, expected=expected, complete=complete,
                                     sha256=digest[:12]), ensure_ascii=False))
    return report
