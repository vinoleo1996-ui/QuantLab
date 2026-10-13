import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quantlab.cloud import CloudError, load_snapshot
from quantlab.cloud.bundle import register_bundle


def make_bundle(root, rows_in_manifest):
    (root / 'sw_l1_pit').mkdir(parents=True)
    (root / 'windows').mkdir()
    frame = pd.DataFrame(dict(instrument_id=['sh.600000', 'sz.000001'], available_at=['2020-01-01 18:00:00+08:00'] * 2))
    frame.to_parquet(root / 'sw_l1_pit/industry_members_sw_l1_tushare.parquet', index=False)
    frame.to_parquet(root / 'windows/etf_daily_cross_asset_5.parquet', index=False)
    (root / 'windows/etf_daily_cross_asset_5_manifest.json').write_text(json.dumps(dict(rows=2)))
    (root / 'MANIFEST.json').write_text(json.dumps([dict(path='sw_l1_pit/industry_members_sw_l1_tushare.parquet',
                                                         nrows=rows_in_manifest)]))


class BundleTests(unittest.TestCase):
    def test_registered_bundle_loads_through_snapshot_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, cloud = Path(tmp) / 'bundle', Path(tmp) / 'cloud'
            make_bundle(bundle, 2)
            register_bundle('b1', root=bundle, cloud_root=cloud, progress=None)
            self.assertEqual(len(load_snapshot('sw_l1_members', 'b1', root=cloud)), 2)
            self.assertEqual(len(load_snapshot('etf_daily_cross_asset', 'b1', root=cloud)), 2)
            self.assertTrue((bundle / 'sw_l1_pit/industry_members_sw_l1_tushare.parquet').exists())

    def test_row_count_mismatch_is_incomplete_and_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, cloud = Path(tmp) / 'bundle', Path(tmp) / 'cloud'
            make_bundle(bundle, 5)
            register_bundle('b1', root=bundle, cloud_root=cloud, progress=None)
            with self.assertRaises(CloudError):
                load_snapshot('sw_l1_members', 'b1', root=cloud)
            self.assertEqual(len(load_snapshot('sw_l1_members', 'b1', root=cloud, require_complete=False)), 2)


if __name__ == '__main__':
    unittest.main()
