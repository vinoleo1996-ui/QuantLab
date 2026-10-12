import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from quantlab.cloud.client import CloudClient, CloudError
from quantlab.cloud.snapshot import load_snapshot, month_windows, snapshot_dataset

ROWS = pd.DataFrame({'trade_date': ['20240102', '20240131', '20240201', '20240229', '20240301'],
                     'value': [1.0, 2.0, 3.0, 4.0, 5.0],
                     'available_at': pd.to_datetime(['2024-01-02 18:00'] * 5)})


def _parquet(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


class FakeServer:
    def __init__(self, frame=ROWS, truncate=False, wrong_count=False):
        self.frame, self.truncate, self.wrong_count, self.calls = frame, truncate, wrong_count, []

    def __call__(self, url, timeout):
        parsed = urlparse(url)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        self.calls.append((parsed.path, query))
        if parsed.path == '/datasets':
            return 200, {}, json.dumps([dict(id='toy', date_col='trade_date', date_min='20240102',
                                             date_max='20240301', nrows=len(self.frame))]).encode()
        if parsed.path.endswith('/schema'):
            return 200, {}, json.dumps(dict(columns=[dict(name=c) for c in self.frame.columns])).encode()
        frame = self.frame
        if 'start' in query:
            frame = frame[(frame.trade_date >= query['start']) & (frame.trade_date <= query['end'])]
        body = _parquet(frame)
        headers = {'x-byte-length': str(len(body)), 'x-row-count': str(len(frame) + (1 if self.wrong_count else 0))}
        return 200, headers, body[:-10] if self.truncate else body


class CloudSnapshotTests(unittest.TestCase):
    def client(self, server):
        return CloudClient('http://fake', transport=server, retries=1, backoff=0)

    def test_month_windows_cover_range_without_gaps(self):
        self.assertEqual(month_windows('20240115', '20240310'),
                         [('20240115', '20240131'), ('20240201', '20240229'), ('20240301', '20240310')])

    def test_monthly_snapshot_matches_catalog_and_resumes(self):
        server = FakeServer()
        with tempfile.TemporaryDirectory() as root:
            manifest = snapshot_dataset(self.client(server), 'toy', 's1', root=root, segment='month', progress=None)
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['rows'], 5)
            self.assertEqual(len(manifest['parts']), 3)
            frame = load_snapshot('toy', root=root)
            self.assertEqual(sorted(frame.value), [1.0, 2.0, 3.0, 4.0, 5.0])
            (Path(root) / 'toy/s1/manifest.json').unlink()
            before = len(server.calls)
            snapshot_dataset(self.client(server), 'toy', 's1', root=root, segment='month', progress=None)
            exports = [c for c in server.calls[before:] if c[0].endswith('export.parquet')]
            self.assertEqual(exports, [])

    def test_truncated_body_and_wrong_row_count_are_refused(self):
        for server in (FakeServer(truncate=True), FakeServer(wrong_count=True)):
            with tempfile.TemporaryDirectory() as root, self.assertRaises(CloudError):
                snapshot_dataset(self.client(server), 'toy', 's1', root=root, progress=None)

    def test_tampered_part_is_detected_on_load(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot_dataset(self.client(FakeServer()), 'toy', 's1', root=root, progress=None)
            part = next((Path(root) / 'toy/s1').glob('part-*.parquet'))
            part.write_bytes(_parquet(ROWS.assign(value=0.0)))
            with self.assertRaises(CloudError):
                load_snapshot('toy', root=root)

    def test_unfinished_named_snapshot_is_a_cloud_error(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'toy/s1').mkdir(parents=True)
            with self.assertRaises(CloudError):
                load_snapshot('toy', 's1', root=root)

    def test_dataset_without_time_column_needs_explicit_rule(self):
        frame = ROWS.drop(columns='available_at')
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(CloudError):
                snapshot_dataset(self.client(FakeServer(frame)), 'toy', 's1', root=root, progress=None)
            manifest = snapshot_dataset(self.client(FakeServer(frame)), 'toy', 's2', root=root, progress=None,
                                        availability_rule='trade_date 18:00 Asia/Shanghai')
            self.assertEqual(manifest['availability_rule'], 'trade_date 18:00 Asia/Shanghai')


if __name__ == '__main__':
    unittest.main()
