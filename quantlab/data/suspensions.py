"""Verified suspension evidence for held stale runs.

Only the strict four-condition raw-cache record counts as suspension:
``tradestatus == 0``, ``volume == 0``, ``amount == 0`` and
``close == preclose``.  Missing files, missing rows, truncated coverage and
source disagreements remain unexplained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.local_adapter import canonical_code, verified_csv

RAW_ROOT = Path(__file__).resolve().parents[2] / 'data/local_supplement/raw'
RAW_EXT_ROOT = Path(__file__).resolve().parents[2] / 'data/local_supplement/raw_ext_20261018'
DEFAULT_ROOTS = (RAW_EXT_ROOT, RAW_ROOT)
PRICE_TOLERANCE = 1e-6
REQUIRED_COLUMNS = ('date', 'tradestatus', 'volume', 'amount', 'close', 'preclose')
_MISSING = object()


def daily_path(code, root=RAW_ROOT):
    return Path(root) / f'{canonical_code(code)}_daily.csv'


def _resolve_roots(root=None, roots=None):
    if roots is not None:
        return tuple(Path(item) for item in roots)
    if root is not None:
        return (Path(root),)
    return DEFAULT_ROOTS


def load_daily_evidence(code, *, root=None, roots=None):
    """Return verified daily rows for one code, or ``None`` when no cache exists.

    The supplemental cache is tried first; a present but unverifiable file fails
    closed instead of silently falling back to shorter primary coverage.
    """
    for candidate in _resolve_roots(root, roots):
        path = daily_path(code, candidate)
        if not path.exists() and not path.with_suffix('.json').exists():
            continue
        return _read_evidence(path)
    return None


def _read_evidence(path):
    try:
        frame, meta = verified_csv(path)
    except (ValueError, OSError) as exc:
        return dict(frame=pd.DataFrame(), meta={}, path=str(path), complete=False,
                    reason='raw_cache_unreadable:' + str(exc))
    if frame is None:
        return dict(frame=pd.DataFrame(), meta={}, path=str(path), complete=False,
                    reason='raw_cache_incomplete_files')
    meta = dict(meta or {})
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        return dict(frame=pd.DataFrame(), meta=meta, path=str(path), complete=False,
                    reason='missing_columns:' + ','.join(missing_columns))
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
    frame = frame.dropna(subset=['date']).sort_values('date')
    frame = frame[~frame['date'].duplicated(keep='last')].set_index('date')
    for column in ('tradestatus', 'volume', 'amount', 'close', 'preclose'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['suspended'] = (
        frame['tradestatus'].eq(0)
        # 4.7% of tradestatus=0 rows encode zero volume/amount as NaN.
        & frame['volume'].fillna(0).eq(0)
        & frame['amount'].fillna(0).eq(0)
        & frame['close'].notna()
        & frame['preclose'].notna()
        & frame['close'].sub(frame['preclose']).abs().le(PRICE_TOLERANCE)
    )
    declared_start = pd.Timestamp(meta['requested_start']).normalize() if meta.get('requested_start') else None
    declared_end = pd.Timestamp(meta['requested_end']).normalize() if meta.get('requested_end') else None
    if declared_start is None or declared_end is None:
        return dict(frame=frame, meta=meta, path=str(path), complete=False,
                    reason='sidecar_coverage_missing')
    expected_rows = meta.get('rows')
    if expected_rows is not None and int(expected_rows) != len(frame):
        return dict(frame=frame, meta=meta, path=str(path), complete=False,
                    reason='sidecar_row_count_mismatch')
    return dict(frame=frame, meta=meta, path=str(path), complete=True,
                coverage_start=declared_start, coverage_end=declared_end,
                sha256=meta.get('sha256'))


class SuspensionEvidence:
    """Lazy, cached access to verified raw-cache evidence for a set of codes."""

    def __init__(self, root=None, roots=None):
        self.roots = _resolve_roots(root, roots)
        self._cache = {}

    def get(self, code):
        code = canonical_code(code)
        cached = self._cache.get(code, _MISSING)
        if cached is _MISSING:
            cached = load_daily_evidence(code, roots=self.roots)
            self._cache[code] = cached
        return cached

    def session_status(self, code, day):
        """``suspended``, ``traded``, ``missing_row``, ``outside_coverage`` or ``no_evidence``."""
        evidence = self.get(code)
        if evidence is None or not evidence.get('complete'):
            return 'no_evidence'
        day = pd.Timestamp(day).normalize()
        if day < evidence['coverage_start'] or day > evidence['coverage_end']:
            return 'outside_coverage'
        frame = evidence['frame']
        if day not in frame.index:
            return 'missing_row'
        return 'suspended' if bool(frame.at[day, 'suspended']) else 'traded'

    def coverage(self, code):
        evidence = self.get(code)
        if evidence is None or not evidence.get('complete'):
            return None
        return evidence['coverage_start'], evidence['coverage_end']

    def metadata(self, code):
        evidence = self.get(code)
        if evidence is None:
            return dict(path=str(daily_path(code, self.roots[0])), present=False, complete=False,
                        searched=[str(daily_path(code, root)) for root in self.roots])
        return dict(path=evidence['path'], present=True, complete=bool(evidence.get('complete')),
                    reason=evidence.get('reason'), coverage_start=str(evidence['coverage_start'].date())
                    if evidence.get('coverage_start') is not None else None,
                    coverage_end=str(evidence['coverage_end'].date())
                    if evidence.get('coverage_end') is not None else None,
                    sha256=evidence.get('sha256'), rows=len(evidence['frame']))


def classify_run(code, days, evidence, delist_date=None, *, window_end=None,
                 open_ended_limit_sessions=20):
    """Classify one consecutive held-stale run. Fail closed on any missing evidence."""
    days = [pd.Timestamp(day).normalize() for day in days]
    if not days:
        return dict(status='empty', reason='empty_run', days=0, covered=0,
                    missing=[], contradictions=[])
    if delist_date is not None and pd.notna(delist_date):
        if days[-1] >= pd.Timestamp(delist_date).normalize():
            return dict(status='unexplained', reason='run_crosses_delist_date', days=len(days),
                        covered=0, missing=[], contradictions=[])
    # Suspension records that never end are indistinguishable from a delisting with no record.
    # A long run still open at the last session is therefore a supply gap, not a suspension.
    if (window_end is not None and pd.notna(window_end)
            and (delist_date is None or not pd.notna(delist_date))
            and days[-1] >= pd.Timestamp(window_end).normalize()
            and len(days) >= open_ended_limit_sessions):
        return dict(status='unexplained', reason='open_ended_supply_gap', days=len(days),
                    covered=len(days), missing=[], contradictions=[])
    statuses = [evidence.session_status(code, day) for day in days]
    missing = [str(day.date()) for day, status in zip(days, statuses)
               if status in ('no_evidence', 'outside_coverage', 'missing_row')]
    if missing:
        return dict(status='unexplained', reason='missing_evidence', days=len(days),
                    covered=len(days) - len(missing), missing=missing, contradictions=[])
    contradictions = [str(day.date()) for day, status in zip(days, statuses)
                      if status != 'suspended']
    if contradictions:
        return dict(status='unexplained', reason='not_suspended_on_session', days=len(days),
                    covered=len(days), missing=[], contradictions=contradictions)
    return dict(status='suspended', reason='all_sessions_suspended', days=len(days),
                covered=len(days), missing=[], contradictions=[])
