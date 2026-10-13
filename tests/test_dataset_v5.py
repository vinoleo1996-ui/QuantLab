import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab import dataset_v5 as v5

BASE_DAYS = pd.bdate_range('2026-09-14', '2026-09-22')
NEW_DAYS = pd.bdate_range('2026-09-23', '2026-09-29')
CODES = {'sh.600000': 10.0, 'sz.000001': 20.0}


def _write(root, name, frame, index=False):
    path = root / f'{name}.parquet'
    frame.to_parquet(path, index=index)
    return dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def make_base(root):
    root.mkdir(parents=True, exist_ok=True)
    panels = {}
    for code, price in CODES.items():
        frame = pd.DataFrame(dict(event_time=BASE_DAYS.strftime('%Y-%m-%d'), open=price, high=price + .1, low=price - .1,
                                  close=price, volume=1000.0, amount=price * 1000.0, factor=0.5,
                                  feature_close=price * 0.5, quality_status='ok', tradestatus=1.0,
                                  listing_session_lower_bound=range(1, len(BASE_DAYS) + 1), is_st_at_open=0.0))
        frame['source'] = pd.Categorical(['baostock'] * len(frame))
        panels[code] = _write(root, f'panel_{code}', frame.assign(event_time=pd.to_datetime(frame.event_time)).set_index('event_time'), index=True)
    manifest = dict(as_of='2026-09-22', panels=panels, benchmark_id='sh.000300',
                    calendar=_write(root, 'calendar', pd.DataFrame({'session': BASE_DAYS.strftime('%Y-%m-%d')})),
                    instruments=_write(root, 'instruments', pd.DataFrame(dict(instrument_id=list(CODES), active=True))),
                    actions=_write(root, 'actions', pd.DataFrame(dict(action_id=['x'], instrument_id=['sh.600000'],
                                                                      ex_date=['2020-01-02']))),
                    benchmark=_write(root, 'benchmark', pd.DataFrame(dict(event_time=BASE_DAYS.strftime('%Y-%m-%d'),
                                                                          code='sh.000300', close=4000.0))),
                    data_gaps=[])
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def make_cloud(volume_lots=10.0, suspend_new=False):
    days = BASE_DAYS[-3:].append(NEW_DAYS)
    daily, adj, limits, basic, suspend = [], [], [], [], []
    for code, price in CODES.items():
        ts = f'{code.split(".")[1]}.{code.split(".")[0].upper()}'
        for day in days:
            key = day.strftime('%Y%m%d')
            if suspend_new and code == 'sz.000001' and day == NEW_DAYS[1]:
                suspend.append(dict(ts_code=ts, trade_date=key, suspend_type='S'))
            else:
                daily.append(dict(ts_code=ts, trade_date=key, open=price, high=price + .1, low=price - .1, close=price,
                                  pre_close=price, vol=volume_lots, amount=price))
            factor = 3.0 if (code == 'sh.600000' and day >= NEW_DAYS[2]) else 2.0
            adj.append(dict(ts_code=ts, trade_date=key, adj_factor=factor))
            limits.append(dict(ts_code=ts, trade_date=key, up_limit=round(price * 1.1, 2), down_limit=round(price * .9, 2)))
            basic.append(dict(ts_code=ts, trade_date=key, turnover_rate=1.0, pb=2.0))
    return dict(daily=pd.DataFrame(daily), adj_factor=pd.DataFrame(adj), stk_limit=pd.DataFrame(limits),
                daily_basic=pd.DataFrame(basic), stock_st=pd.DataFrame(columns=['ts_code', 'trade_date']),
                suspend_d=pd.DataFrame(suspend, columns=['ts_code', 'trade_date', 'suspend_type']))


class DatasetV5Tests(unittest.TestCase):
    def build(self, tmp, cloud):
        base = make_base(Path(tmp) / 'base')
        return base, v5.build(base, cloud, end='2026-09-29', overlap=('2026-09-18', '2026-09-22'),
                              version='t', out_root=Path(tmp) / 'out')

    def read(self, target, code):
        manifest = json.loads((target / 'manifest.json').read_text())
        return v5._read(target, manifest['panels'][code]), manifest

    def test_history_kept_and_new_rows_continue_the_factor(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, target = self.build(tmp, make_cloud())
            frame, manifest = self.read(target, 'sh.600000')
            old = pd.read_parquet(base.parent / 'panel_sh.600000.parquet')
            self.assertEqual(frame.index.name, 'event_time')
            pd.testing.assert_frame_equal(frame.iloc[:len(old)].astype(object), old.astype(object), check_dtype=False)
            new = frame.iloc[len(old):]
            self.assertEqual(len(new), len(NEW_DAYS))
            self.assertEqual(new.volume.iloc[0], 1000.0)
            self.assertAlmostEqual(new.factor.iloc[0], 0.5)
            self.assertAlmostEqual(new.factor.iloc[2], 0.75)
            self.assertEqual(manifest['as_of'], '2026-09-29')
            self.assertEqual(manifest['v5_provenance']['overlap_check']['rows_mismatched'], 0)
            self.assertTrue(np.all(new.listing_session_lower_bound.to_numpy() == np.arange(len(BASE_DAYS) + 1, len(BASE_DAYS) + 1 + len(NEW_DAYS))))

    def test_suspended_session_is_kept_as_non_trading(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, target = self.build(tmp, make_cloud(suspend_new=True))
            frame, _ = self.read(target, 'sz.000001')
            row = frame.loc[NEW_DAYS[1]]
            self.assertEqual(row.tradestatus, 0.0)
            self.assertEqual(row.volume, 0.0)

    def test_unit_error_fails_the_overlap_check(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(v5.BuildError):
            self.build(tmp, make_cloud(volume_lots=1000.0))

    def combined(self, unit='share', close_shift=0.0):
        rows, basic = [], []
        for code, price in CODES.items():
            for i, day in enumerate(NEW_DAYS):
                suspended = code == 'sz.000001' and i == 1
                rows.append(dict(instrument_id=code, ts_code='x', trade_date=day, open=None if suspended else price,
                                 high=price, low=price, close=None if suspended else price, pre_close=price,
                                 volume=1000.0, amount=price * 1000, adj_factor=3.0 if (code == 'sh.600000' and i >= 2) else 2.0,
                                 limit_up=price * 1.1, limit_down=price * .9, tradestatus=0 if suspended else 1,
                                 volume_unit=unit, amount_unit='CNY', is_st_at_open=code == 'sz.000001'))
            for day in BASE_DAYS[-3:].append(NEW_DAYS):
                basic.append(dict(instrument_id=code, trade_date=day, close=price + close_shift, turnover_rate=1.0, pb=2.0))
        daily = pd.DataFrame(rows)
        basic = pd.DataFrame(basic)
        closes = {(r.instrument_id, r.trade_date): r.close for r in basic.itertuples() if r.trade_date <= BASE_DAYS[-1]}
        return v5.prepare_combined(daily, basic, overlap_close=closes)

    def test_combined_table_builds_with_units_from_the_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = make_base(Path(tmp) / 'base')
            target = v5.build(base, self.combined(), end='2026-09-29', overlap=('2026-09-18', '2026-09-22'),
                              version='c', out_root=Path(tmp) / 'out')
            frame, manifest = self.read(target, 'sz.000001')
            new = frame.loc[NEW_DAYS]
            self.assertEqual(new.volume.iloc[0], 1000.0)
            self.assertEqual(new.tradestatus.iloc[1], 0.0)
            self.assertTrue(new.is_st_at_open.eq(1.0).all())
            self.assertEqual(manifest['v5_provenance']['overlap_check']['kind'], 'close_only_daily_basic')
            other, _ = self.read(target, 'sh.600000')
            self.assertAlmostEqual(other.loc[NEW_DAYS[2]].factor, 0.75)

    def test_seam_on_an_ex_date_first_session_uses_pre_close(self):
        cloud = self.combined()
        key = ('sh.600000', NEW_DAYS[0])
        cloud['adj'] = cloud['adj'].copy()
        cloud['daily'] = cloud['daily'].copy()
        cloud['adj'].loc[key] = 4.0
        cloud['daily'].loc[key, 'pre_close'] = 5.0
        with tempfile.TemporaryDirectory() as tmp:
            target = v5.build(make_base(Path(tmp) / 'base'), cloud, end='2026-09-29',
                              overlap=('2026-09-18', '2026-09-22'), version='e', out_root=Path(tmp) / 'out')
            frame, _ = self.read(target, 'sh.600000')
            first = frame.loc[NEW_DAYS[0]]
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame.index))
            self.assertAlmostEqual(first.factor, 0.5 * 4.0 / (4.0 * 5.0 / 10.0))
            self.assertAlmostEqual(first.feature_close / frame.iloc[len(BASE_DAYS) - 1].feature_close, 2.0)

    def test_combined_table_refuses_mixed_units_and_bad_seam(self):
        with self.assertRaises(v5.BuildError):
            self.combined(unit='furlong')
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(v5.BuildError):
            v5.build(make_base(Path(tmp) / 'base'), self.combined(close_shift=1.0), end='2026-09-29',
                     overlap=('2026-09-18', '2026-09-22'), version='c', out_root=Path(tmp) / 'out')

    def test_missing_cloud_table_is_refused(self):
        cloud = make_cloud()
        cloud.pop('stock_st')
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(v5.BuildError):
            self.build(tmp, cloud)


if __name__ == '__main__':
    unittest.main()
