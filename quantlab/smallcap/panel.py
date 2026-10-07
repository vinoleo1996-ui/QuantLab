"""Canonical wide PIT panel (v4) built from the verified local dataset.

The source dataset is read, never modified. Every derived field is written to a
new version directory together with the source manifest digest and the rules
used. Fields are date x code float32 matrices so a whole-market daily
simulation fits in memory.

Rules (all decided before any account was run):
- ``adj_close`` = raw close x the stored adjustment factor. Return ratios from it
  do not depend on the factor base; price *levels* from it are not used.
- ``float_mktcap`` = close x volume / (turnover / 100). Baostock ``turn`` is volume
  over tradable shares, so this is the float market value at that close.
- Beijing exchange rows have no limit fields upstream; a 30 percent limit from the
  previous raw close is filled in. They also have no turnover, so no float cap.
- Financial values become usable at the open of the first session strictly after
  ``announced_on``. Values are the stored snapshot; later restatements are not
  reconstructed (recorded as a limitation, not hidden).
- Delisting: ``delist_date`` is the lifecycle date from the instrument records.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / 'output/account_2025_20260927/extension/dataset/manifest.json'
PANEL_ROOT = ROOT / 'data/smallcap_v4'
FIELDS = ('open', 'high', 'low', 'close', 'adj_close', 'amount', 'volume', 'turnover',
          'float_mktcap', 'is_st', 'tradestatus', 'limit_up', 'limit_down', 'pb_mrq')
BJ_LIMIT = 0.30


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _read(root, entry):
    payload = (root / entry['path']).read_bytes()
    if sha256_bytes(payload) != entry['sha256']:
        raise ValueError('source_hash_mismatch:' + entry['path'])
    return pd.read_parquet(io.BytesIO(payload))


def float_mktcap(close, volume, turnover_pct):
    """Float market value from close, share volume and turnover in percent."""
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    turnover = np.asarray(turnover_pct, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        value = close * volume / (turnover / 100.0)
    return np.where((turnover > 0) & (volume > 0) & np.isfinite(value), value, np.nan)


def bj_limits(prev_close):
    prev_close = np.asarray(prev_close, dtype=float)
    return np.round(prev_close * (1 + BJ_LIMIT), 2), np.round(prev_close * (1 - BJ_LIMIT), 2)


def financial_availability(announced_on, sessions):
    """First session strictly after the announcement day (trade at that open)."""
    sessions = pd.DatetimeIndex(sessions)
    announced = pd.to_datetime(pd.Series(announced_on)).dt.normalize()
    position = sessions.searchsorted(announced.to_numpy(), side='right')
    out = pd.Series(pd.NaT, index=announced.index, dtype='datetime64[ns]')
    valid = position < len(sessions)
    out[valid] = sessions[position[valid]]
    return out


def _panel_frame(root, entry, code):
    frame = _read(root, entry)
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize() if getattr(frame.index, 'tz', None) else pd.to_datetime(frame.index).normalize()
    frame = frame[~frame.index.duplicated(keep='last')]
    ok = frame['quality_status'].astype(str).eq('ok')
    out = pd.DataFrame(index=frame.index)

    def column(name):
        if name not in frame:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[name], errors='coerce')

    for name in ('open', 'high', 'low', 'close', 'amount', 'volume', 'turnover', 'tradestatus',
                 'limit_up', 'limit_down', 'pb_mrq'):
        out[name] = column(name).where(ok)
    out['is_st'] = column('is_st_at_open')
    factor = column('factor').ffill().bfill()
    out['adj_close'] = out['close'] * factor
    out['float_mktcap'] = float_mktcap(out['close'], out['volume'], out['turnover'])
    if code.startswith('bj.'):
        up, down = bj_limits(out['close'].shift(1))
        out['limit_up'] = out['limit_up'].fillna(pd.Series(up, index=out.index))
        out['limit_down'] = out['limit_down'].fillna(pd.Series(down, index=out.index))
    return out


def build(version=None, source_manifest=SOURCE_MANIFEST, end='2026-09-22'):
    source_manifest = Path(source_manifest)
    root = source_manifest.parent
    manifest = json.loads(source_manifest.read_text())
    version = version or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = PANEL_ROOT / version
    if target.exists():
        raise FileExistsError(target)
    calendar = _read(root, manifest['calendar'])
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar['session'])).normalize()
    sessions = sessions[(sessions >= pd.Timestamp('2014-01-02')) & (sessions <= pd.Timestamp(end))]
    codes = sorted(manifest['panels'])
    matrices = {name: np.full((len(sessions), len(codes)), np.nan, dtype=np.float32) for name in FIELDS}
    last_bar = {}
    for column, code in enumerate(codes):
        if column % 1000 == 0:
            print(json.dumps({'stage': 'panels', 'index': column, 'total': len(codes)}), flush=True)
        frame = _panel_frame(root, manifest['panels'][code], code).reindex(sessions)
        for name in FIELDS:
            matrices[name][:, column] = frame[name].to_numpy(dtype=np.float32)
        traded = frame['close'].dropna()
        last_bar[code] = str(traded.index[-1].date()) if len(traded) else None
    target.mkdir(parents=True)
    index = pd.Index(sessions, name='date')
    for name, values in matrices.items():
        pd.DataFrame(values, index=index, columns=codes).to_parquet(target / f'{name}.parquet')
    instruments = _read(root, manifest['instruments'])
    delisted = instruments[instruments.active.eq(False)][['instrument_id', 'effective_at']]
    delisted = delisted.rename(columns={'instrument_id': 'code', 'effective_at': 'delist_date'})
    delisted['delist_date'] = pd.to_datetime(delisted['delist_date']).dt.normalize()
    delisted['last_bar'] = delisted['code'].map(last_bar)
    delisted.to_parquet(target / 'delistings.parquet', index=False)
    industry = _read(root, manifest['industry_records'])[['instrument_id', 'effective_at', 'sector']]
    industry.to_parquet(target / 'industry.parquet', index=False)
    financials = _read(root, manifest['financials'])
    financials = financials[['instrument_id', 'metric', 'value', 'report_period', 'announced_on', 'quality_status']].copy()
    financials['available_date'] = financial_availability(financials['announced_on'], sessions).to_numpy()
    financials.to_parquet(target / 'financials.parquet', index=False)
    meta = {
        'version': version,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_manifest': str(source_manifest.relative_to(ROOT)),
        'source_manifest_sha256': sha256_bytes(source_manifest.read_bytes()),
        'sessions': [str(sessions[0].date()), str(sessions[-1].date()), len(sessions)],
        'codes': len(codes),
        'fields': list(FIELDS),
        'rules': {
            'adj_close': 'raw close x stored factor; only ratios are used',
            'float_mktcap': 'close*volume/(turnover/100)',
            'bj_limits': f'filled from previous raw close at +/-{BJ_LIMIT:.0%}; no turnover so no float cap',
            'financial_available_date': 'first session strictly after announced_on, trade at open',
            'financial_values': 'stored snapshot; restatement history not reconstructed',
            'quality': "price fields masked where quality_status != 'ok'",
        },
        'coverage': {name: float(np.isfinite(values).mean()) for name, values in matrices.items()},
        'financial_rows': int(len(financials)),
        'financial_available_rows': int(financials['available_date'].notna().sum()),
        'delistings': int(len(delisted)),
        'promotion_approved': False,
    }
    (target / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({'stage': 'built', 'target': str(target)}), flush=True)
    return target


class Panel:
    """Read-only accessor for one panel version."""

    def __init__(self, version):
        self.path = PANEL_ROOT / version
        self.meta = json.loads((self.path / 'meta.json').read_text())
        self._cache = {}

    def field(self, name):
        if name not in self._cache:
            self._cache[name] = pd.read_parquet(self.path / f'{name}.parquet')
        return self._cache[name]

    def table(self, name):
        return pd.read_parquet(self.path / f'{name}.parquet')

    @property
    def sessions(self):
        return self.field('close').index

    @property
    def codes(self):
        return self.field('close').columns


def latest_version():
    versions = sorted(path.name for path in PANEL_ROOT.iterdir() if (path / 'meta.json').exists())
    return versions[-1]


if __name__ == '__main__':
    build()
