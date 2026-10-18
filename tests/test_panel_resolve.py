"""Panel versions must be resolved fail-closed, never by directory name order.

``p4_ledger`` used to take ``sorted(glob('*/delistings.parquet'))[-1]`` and the prestart preview
had ``20261018T012723Z`` hard-coded. That version carries newer prices but its amount and other
derived fields stop five sessions earlier and its adjustment factor is stale, so a consumer that
picked it would rank liquidity on a broken unit.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab.smallcap.panel import resolve_trusted_panel


def write_version(root, name, last, *, amount_last=None, declare_fields=True):
    path = root / name
    path.mkdir(parents=True)
    meta = {'version': name}
    if declare_fields:
        meta['fields'] = ['open', 'close', 'adj_close', 'amount']
    (path / 'meta.json').write_text(json.dumps(meta))
    for field in ('close', 'adj_close', 'amount'):
        day = amount_last if (field == 'amount' and amount_last is not None) else last
        pd.DataFrame({'x': [1.0]}, index=[pd.Timestamp(day)]).to_parquet(path / f'{field}.parquet')
    return path


def test_newest_consistent_version_wins(tmp_path):
    write_version(tmp_path, '20260101T000000Z', '2026-01-30')
    write_version(tmp_path, '20260201T000000Z', '2026-02-27')
    # newer but internally inconsistent: amount stops earlier
    write_version(tmp_path, '20260301T000000Z', '2026-03-31', amount_last='2026-03-20')
    resolved = resolve_trusted_panel(tmp_path)
    assert resolved['version'] == '20260201T000000Z'
    assert resolved['last_session'] == pd.Timestamp('2026-02-27')


def test_a_version_that_does_not_declare_fields_is_never_trusted(tmp_path):
    write_version(tmp_path, '20260101T000000Z', '2026-01-30', declare_fields=False)
    with pytest.raises(ValueError, match='no_trusted_panel'):
        resolve_trusted_panel(tmp_path)


def test_duplicate_snapshots_resolve_to_the_earliest_name(tmp_path):
    write_version(tmp_path, '20260201T000000Z', '2026-02-27')
    write_version(tmp_path, '20260201T010000Z', '2026-02-27')
    assert resolve_trusted_panel(tmp_path)['version'] == '20260201T000000Z'


def test_the_real_repository_resolves_the_documented_base():
    pytest.importorskip('pyarrow')
    if not (ROOT / 'data/smallcap_v4/20261013T023232Z/meta.json').exists():
        pytest.skip('trusted base not present')
    resolved = resolve_trusted_panel()
    assert resolved['version'] == '20261013T023232Z'
    assert resolved['close_sha256'].startswith('9341f8df')


def test_prestart_calendar_uses_the_resolver_not_a_hard_coded_version(monkeypatch, tmp_path):
    import importlib
    snapshot = importlib.import_module('p5_snapshot')
    version = write_version(tmp_path, '20260101T000000Z', '2026-01-30')
    pd.DataFrame({'sh.600000': [1.0]}, index=[pd.Timestamp('2026-01-30')]).to_parquet(
        version / 'close.parquet')
    monkeypatch.setattr(snapshot, 'resolve_trusted_panel',
                        lambda **kwargs: {'path': version, 'version': '20260101T000000Z',
                                          'last_session': pd.Timestamp('2026-01-30'),
                                          'fields': [], 'meta': {}, 'close_sha256': 'x'})
    calendar, source, provisional = snapshot.prestart_calendar('2026-02-10')
    assert str(version) in source
    assert calendar[-1] >= pd.Timestamp('2026-02-10')
    assert provisional is True
