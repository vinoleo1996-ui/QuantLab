import io
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantlab.smallcap import panel as P


class FactorFill(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        frame = pd.DataFrame({'quality_status': ['ok'] * 3, 'close': [10.0] * 3,
                              'factor': [float('nan'), 2.0, 2.0]},
                             index=pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']))
        target = self.root / 'panel.parquet'
        frame.to_parquet(target)
        self.entry = {'path': 'panel.parquet', 'sha256': P.sha256_bytes(target.read_bytes())}

    def test_ffill_keeps_leading_factor_missing(self):
        out = P._panel_frame(self.root, self.entry, 'sz.000001', factor_fill='ffill')
        self.assertTrue(pd.isna(out['adj_close'].iloc[0]))
        self.assertEqual(out['adj_close'].iloc[1], 20.0)
        self.assertEqual(out.attrs['factor_lead_missing'], 1)

    def test_legacy_backfills_the_head(self):
        out = P._panel_frame(self.root, self.entry, 'sz.000001', factor_fill='legacy')
        self.assertEqual(out['adj_close'].iloc[0], 20.0)
        self.assertEqual(out.attrs['factor_fill'], 'legacy')

    def test_unknown_fill_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            P._panel_frame(self.root, self.entry, 'sz.000001', factor_fill='bogus')


if __name__ == '__main__':
    unittest.main()
