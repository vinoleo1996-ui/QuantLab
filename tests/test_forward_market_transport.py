"""Behavioural tests for the injected forward raw-market transport.

Every opener in this file is a small in-memory fake.  These tests must never
make an HTTP request, construct an account, or enter the historical replay
path.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

from quantlab.forward_market_transport import (
    FailedRawEastmoneyCollection,
    RawEastmoneyHttpsResponse,
    collect_registered_eastmoney_eod_pages,
)
from quantlab.forward_source_manifest import (
    PreparedRawEastmoneyCollection,
    SourceManifestJournal,
    fixed_eastmoney_eod_manifest_v1,
)


SESSION = "2026-10-01"
DECISION_AT = "2026-10-01 18:00:00+08:00"
CAPTURE_TIME = "2026-10-01 16:00:00+08:00"


def _raw_page(*, total: int, row_count: int) -> bytes:
    return json.dumps(
        {"data": {"total": total, "diff": [{"row": index} for index in range(row_count)]}},
        separators=(",", ":"),
    ).encode("utf-8")


class _FixedClock:
    def __init__(self, value: str):
        self.value = value
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self.value


class _FakeOpener:
    def __init__(self, responses, *, on_open=None):
        self._responses = list(responses)
        self._on_open = on_open
        self.urls: list[str] = []

    def __call__(self, url: str):
        self.urls.append(url)
        if self._on_open is not None:
            self._on_open(url)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _response(manifest, page_number: int, raw: bytes, *, response_url: str | None = None):
    expected_url = manifest.canonical_url(page_number)
    return RawEastmoneyHttpsResponse(
        request_url=expected_url,
        response_url=response_url or expected_url,
        raw=raw,
        content_type="application/json",
    )


class ForwardMarketTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.journal = SourceManifestJournal(Path(self.temp.name) / "source-manifests")
        self.manifest = fixed_eastmoney_eod_manifest_v1()

    def tearDown(self):
        self.temp.cleanup()

    def _collect(self, *, opener, clock=CAPTURE_TIME):
        return collect_registered_eastmoney_eod_pages(
            journal=self.journal,
            manifest=self.manifest,
            session=SESSION,
            decision_at=DECISION_AT,
            collection_id="forward-eod-20261001",
            opener=opener,
            clock=_FixedClock(clock),
        )

    def test_registers_before_fetch_and_fetches_only_pages_implied_by_valid_first_total(self):
        journal_records_visible_to_opener: list[int] = []
        opener = _FakeOpener(
            [
                _response(self.manifest, 1, _raw_page(total=101, row_count=100)),
                _response(self.manifest, 2, _raw_page(total=101, row_count=1)),
            ],
            on_open=lambda _url: journal_records_visible_to_opener.append(
                int(self.journal.verify()["records"])
            ),
        )

        result = self._collect(opener=opener)

        self.assertIsInstance(result, PreparedRawEastmoneyCollection)
        self.assertEqual(
            opener.urls,
            [self.manifest.canonical_url(1), self.manifest.canonical_url(2)],
        )
        self.assertEqual(journal_records_visible_to_opener, [1, 1])
        self.assertEqual([page.page_number for page in result.capture_pages], [1, 2])
        self.assertEqual(result.manifest_sha256, self.manifest.manifest_sha256)
        self.assertTrue(self.journal.verify()["verified"])

    def test_transport_continues_collecting_after_close_snapshot_window(self):
        opener = _FakeOpener([
            _response(self.manifest, 1, _raw_page(total=1, row_count=1)),
        ])

        result = self._collect(opener=opener, clock="2026-10-01 17:51:00+08:00")

        self.assertIsInstance(result, PreparedRawEastmoneyCollection)
        self.assertEqual(opener.urls, [self.manifest.canonical_url(1)])
        self.assertEqual(len(result.capture_pages), 1)
        self.assertEqual(str(result.capture_pages[0].captured_at), "2026-10-01 17:51:00+08:00")
        self.assertEqual(self.journal.verify()["records"], 1)

    def test_existing_registration_after_capture_clock_fails_before_calling_the_opener(self):
        self.journal.register(self.manifest, recorded_at="2026-10-01 16:30:00+08:00")
        opener = _FakeOpener([])

        result = self._collect(opener=opener)

        self.assertIsInstance(result, FailedRawEastmoneyCollection)
        self.assertEqual(result.failure_reason, "registration_after_capture")
        self.assertEqual(result.attempts, ())
        self.assertEqual(str(result.reported_at), "2026-10-01 16:00:00+08:00")
        self.assertEqual(opener.urls, [])

    def test_first_page_parse_failure_is_explicit_and_never_requests_page_two(self):
        opener = _FakeOpener([_response(self.manifest, 1, b"not-json")])

        result = self._collect(opener=opener)

        self.assertIsInstance(result, FailedRawEastmoneyCollection)
        self.assertEqual(result.failure_reason, "first_page_total_parse_failed")
        self.assertEqual(opener.urls, [self.manifest.canonical_url(1)])
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].raw, b"not-json")
        self.assertIsNone(result.attempts[0].fetch_error)

    def test_later_page_failure_is_preserved_without_cache_or_fallback(self):
        opener = _FakeOpener(
            [
                _response(self.manifest, 1, _raw_page(total=101, row_count=100)),
                TimeoutError("synthetic timeout"),
            ]
        )

        result = self._collect(opener=opener)

        self.assertIsInstance(result, FailedRawEastmoneyCollection)
        self.assertEqual(result.failure_reason, "page_fetch_failed")
        self.assertEqual(
            opener.urls,
            [self.manifest.canonical_url(1), self.manifest.canonical_url(2)],
        )
        self.assertEqual(len(result.attempts), 2)
        failed = result.attempts[-1]
        self.assertEqual(failed.page_number, 2)
        self.assertIsNone(failed.raw)
        self.assertIn("TimeoutError", failed.fetch_error or "")
        self.assertFalse(failed.cache_reused)
        self.assertFalse(failed.fallback_attempted)

    def test_rejects_mismatched_response_identity_before_any_later_page(self):
        opener = _FakeOpener(
            [
                _response(
                    self.manifest,
                    1,
                    _raw_page(total=101, row_count=100),
                    response_url="https://example.test/redirected-response",
                )
            ]
        )

        result = self._collect(opener=opener)

        self.assertIsInstance(result, FailedRawEastmoneyCollection)
        self.assertEqual(result.failure_reason, "response_source_identity_mismatch")
        self.assertEqual(opener.urls, [self.manifest.canonical_url(1)])
        self.assertEqual(len(result.attempts), 1)
        self.assertIsNone(result.attempts[0].raw)
        self.assertEqual(result.attempts[0].fetch_error, "response_source_identity_mismatch")

    def test_transport_module_has_no_network_client_or_account_dependency(self):
        module = Path(__file__).parents[1] / "quantlab" / "forward_market_transport.py"
        tree = ast.parse(module.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden_prefixes = (
            "requests",
            "httpx",
            "urllib.request",
            "socket",
            "quantlab.backtest",
            "quantlab.strategy",
            "quantlab.portfolio",
            "quantlab.webapp",
        )
        self.assertFalse(
            [name for name in imported if name.startswith(forbidden_prefixes)],
            imported,
        )


if __name__ == "__main__":
    unittest.main()
