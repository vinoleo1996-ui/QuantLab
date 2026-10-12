"""HTTP client for the cloud data API (``/datasets``, ``/schema``, ``/export.parquet``).

The link is a Tailscale address, so the local HTTP proxy is bypassed. Every
export is checked against the server's ``x-byte-length`` and ``x-row-count``
headers and the parquet magic bytes before it is returned; a short or corrupt
body raises instead of being cached.
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request

import pyarrow.parquet as pq

DEFAULT_BASE = os.environ.get('QUANTLAB_CLOUD_URL', 'http://100.124.230.108:8787')
PARQUET_MAGIC = b'PAR1'


class CloudError(RuntimeError):
    pass


def _urllib_transport(url, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()


class CloudClient:
    def __init__(self, base_url=DEFAULT_BASE, *, transport=None, timeout=900, retries=3, backoff=5.0):
        self.base_url = base_url.rstrip('/')
        self.transport = transport or _urllib_transport
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _url(self, path, params=None):
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        return f'{self.base_url}{path}' + (f'?{query}' if query else '')

    def _get(self, path, params=None):
        last = None
        for attempt in range(self.retries):
            try:
                status, headers, body = self.transport(self._url(path, params), self.timeout)
                if status != 200:
                    raise CloudError(f'http_{status}:{path}')
                return headers, body
            except CloudError:
                raise
            except Exception as exc:  # network errors are retried, then surfaced
                last = exc
                time.sleep(self.backoff * (attempt + 1))
        raise CloudError(f'unreachable:{path}:{type(last).__name__}:{last}')

    def _json(self, path, params=None):
        return json.loads(self._get(path, params)[1])

    def health(self):
        return self._json('/health')

    def catalog(self):
        return {row['id']: row for row in self._json('/datasets')}

    def schema(self, dataset_id):
        return self._json(f'/datasets/{dataset_id}/schema')

    def export(self, dataset_id, start=None, end=None, symbol=None):
        """Return ``(bytes, rows, headers)`` for one verified parquet export."""
        last = None
        for attempt in range(self.retries):
            headers, body = self._get(f'/datasets/{dataset_id}/export.parquet',
                                      dict(start=start, end=end, symbol=symbol))
            try:
                rows = verify_parquet(body, headers)
                return body, rows, headers
            except CloudError as exc:
                last = exc
                time.sleep(self.backoff * (attempt + 1))
        raise last


def verify_parquet(body, headers):
    declared = headers.get('x-byte-length') or headers.get('content-length')
    if declared is not None and int(declared) != len(body):
        raise CloudError(f'short_body:{len(body)}/{declared}')
    if len(body) < 12 or body[:4] != PARQUET_MAGIC or body[-4:] != PARQUET_MAGIC:
        raise CloudError('not_parquet')
    rows = pq.ParquetFile(io.BytesIO(body)).metadata.num_rows
    declared_rows = headers.get('x-row-count')
    if declared_rows is not None and int(declared_rows) != rows:
        raise CloudError(f'row_count_mismatch:{rows}/{declared_rows}')
    return rows
