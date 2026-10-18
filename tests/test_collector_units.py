"""The incremental collector must not write thousand-CNY amounts or pick an ad-hoc base."""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
if 'requests' not in sys.modules:
    import types
    sys.modules['requests'] = types.ModuleType('requests')  # import-time only; no network in tests
collector = importlib.import_module('collect_tushare_incremental')


class FakeAdapter:
    def trade_cal(self, exchange, start, end):
        return pd.DataFrame({'is_open': [True, True],
                             'cal_date': pd.to_datetime(['2026-10-13', '2026-10-14'])})

    def daily_trade_date(self, day):
        return pd.DataFrame([{'trade_date': pd.Timestamp(day), 'ts_code': '000001.SZ',
                              'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.0,
                              'pre_close': 10.0, 'vol': 1_000.0, 'amount': 1_000.0}])


def test_lots_and_thousand_cny_are_converted_to_shares_and_cny():
    frame = collector.fetch_daily_incremental(FakeAdapter(), '20261013', '20261013')
    assert len(frame) == 2  # the fake calendar returns both sessions
    row = frame.iloc[0]
    assert row['volume'] == pytest.approx(100_000.0)          # 1,000 lots -> 100,000 shares
    assert row['amount'] == pytest.approx(1_000_000.0)        # 1,000 thousand CNY -> CNY
    assert row['amount'] / (row['volume'] * row['close']) == pytest.approx(1.0)


def write_version(root, name, last, *, amount_last=None, declare_fields=True):
    path = root / name
    path.mkdir(parents=True)
    meta = {'version': name}
    if declare_fields:
        meta['fields'] = ['close', 'adj_close', 'amount']
    (path / 'meta.json').write_text(json.dumps(meta))
    for field in ('close', 'adj_close', 'amount'):
        day = amount_last if (field == 'amount' and amount_last is not None) else last
        pd.DataFrame({'x': [1.0]}, index=[pd.Timestamp(day)]).to_parquet(path / f'{field}.parquet')


def test_collector_never_selects_the_newer_inconsistent_panel(tmp_path, monkeypatch):
    write_version(tmp_path, '20261013T023232Z', '2026-10-12')
    write_version(tmp_path, '20261018T012723Z', '2026-10-16', amount_last='2026-10-09')
    monkeypatch.setattr(collector, 'PANEL_DIR', tmp_path)
    assert collector.latest_panel_dir().name == '20261013T023232Z'


def test_collector_returns_none_rather_than_a_broken_panel(tmp_path, monkeypatch):
    write_version(tmp_path, '20261018T012723Z', '2026-10-16', amount_last='2026-10-09')
    monkeypatch.setattr(collector, 'PANEL_DIR', tmp_path)
    assert collector.latest_panel_dir() is None
