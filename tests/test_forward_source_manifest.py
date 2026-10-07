"""Behavioural tests for the future raw Eastmoney source-manifest boundary.

These tests never make a network request or create a market-data sample.  They
exercise only the immutable request contract and its append-only registration
journal in a temporary directory.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

from quantlab.forward_source_manifest import (
    RawEastmoneyTransportAttempt,
    SourceManifestContractError,
    SourceManifestJournal,
    fixed_eastmoney_eod_manifest_v1,
)


class ForwardSourceManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.journal = SourceManifestJournal(Path(self.temp.name) / "source-manifests")

    def tearDown(self):
        self.temp.cleanup()

    def test_registers_one_fixed_eastmoney_manifest_in_an_append_only_chain(self):
        manifest = fixed_eastmoney_eod_manifest_v1()

        registered = self.journal.register(
            manifest,
            recorded_at="2026-10-01 14:00:00+08:00",
        )
        repeated = self.journal.register(
            manifest,
            recorded_at="2026-10-01 14:01:00+08:00",
        )

        self.assertEqual(registered.manifest_sha256, manifest.manifest_sha256)
        self.assertEqual(repeated.event_sha256, registered.event_sha256)
        self.assertEqual(self.journal.verify(), {
            "verified": True,
            "records": 1,
            "head_event_sha256": registered.event_sha256,
        })
        self.assertEqual(registered.manifest["source_id"], "eastmoney.push2.clist")
        self.assertEqual(registered.manifest["source_profile_version"], "eastmoney-push2-clist-eod-v2")
        self.assertEqual(registered.manifest["page_contract"]["page_size"], 100)
        self.assertEqual(registered.manifest["transport_contract"]["fallback_policy"], "forbidden")
        self.assertEqual(registered.manifest["clock_contract"]["operations_validation_target"], "17:55")
        self.assertNotIn("validation_deadline", registered.manifest["clock_contract"])

    def test_registration_binding_requires_the_exact_manifest_and_event_hashes(self):
        manifest = fixed_eastmoney_eod_manifest_v1()
        registered = self.journal.register(
            manifest,
            recorded_at="2026-10-01 14:00:00+08:00",
        )

        bound = self.journal.verify_registration_binding(
            manifest_sha256=manifest.manifest_sha256,
            manifest_event_sha256=registered.event_sha256,
        )

        self.assertEqual(bound, registered)
        with self.assertRaisesRegex(SourceManifestContractError, "registration_binding_not_found"):
            self.journal.verify_registration_binding(
                manifest_sha256=manifest.manifest_sha256,
                manifest_event_sha256="0" * 64,
            )

    def test_registered_transport_only_prepares_declared_raw_page_bytes(self):
        manifest = fixed_eastmoney_eod_manifest_v1()
        registered = self.journal.register(
            manifest,
            recorded_at="2026-10-01 14:00:00+08:00",
        )
        transport = self.journal.open_raw_transport(manifest)

        collection = transport.prepare_collection(
            session="2026-10-01",
            decision_at="2026-10-01 18:00:00+08:00",
            attempts=(
                RawEastmoneyTransportAttempt(
                    collection_id="forward-eod-20261001",
                    page_number=1,
                    canonical_url=manifest.canonical_url(1),
                    raw=b'{"fixture":"raw-response-only"}',
                    captured_at="2026-10-01 16:30:00+08:00",
                ),
            ),
        )

        self.assertEqual(collection.manifest_sha256, manifest.manifest_sha256)
        self.assertEqual(collection.manifest_event_sha256, registered.event_sha256)
        self.assertEqual(collection.collection_id, "forward-eod-20261001")
        self.assertEqual(len(collection.capture_pages), 1)
        self.assertEqual(collection.capture_pages[0].raw, b'{"fixture":"raw-response-only"}')
        self.assertEqual(collection.capture_pages[0].canonical_url, manifest.canonical_url(1))

    def test_failed_page_is_preserved_without_a_fallback_page(self):
        manifest = fixed_eastmoney_eod_manifest_v1()
        self.journal.register(manifest, recorded_at="2026-10-01 14:00:00+08:00")
        transport = self.journal.open_raw_transport(manifest)

        collection = transport.prepare_collection(
            session="2026-10-01",
            decision_at="2026-10-01 18:00:00+08:00",
            attempts=(
                RawEastmoneyTransportAttempt(
                    collection_id="forward-eod-20261001-failed",
                    page_number=1,
                    canonical_url=manifest.canonical_url(1),
                    raw=None,
                    fetch_error="timeout",
                    captured_at="2026-10-01 16:30:00+08:00",
                ),
            ),
        )

        self.assertEqual(len(collection.capture_pages), 1)
        self.assertIsNone(collection.capture_pages[0].raw)
        self.assertEqual(collection.capture_pages[0].fetch_error, "timeout")

    def test_transport_preserves_only_a_provider_clock_not_later_than_capture(self):
        manifest = fixed_eastmoney_eod_manifest_v1()
        self.journal.register(manifest, recorded_at="2026-10-01 14:00:00+08:00")
        transport = self.journal.open_raw_transport(manifest)
        attempt = RawEastmoneyTransportAttempt(
            collection_id="forward-eod-20261001-clock",
            page_number=1,
            canonical_url=manifest.canonical_url(1),
            raw=b'{"fixture":"provider-clock"}',
            captured_at="2026-10-01 16:30:00+08:00",
            provider_published_at="2026-10-01 16:29:00+08:00",
        )

        collection = transport.prepare_collection(
            session="2026-10-01",
            decision_at="2026-10-01 18:00:00+08:00",
            attempts=(attempt,),
        )

        self.assertEqual(
            str(collection.capture_pages[0].provider_published_at),
            "2026-10-01 16:29:00+08:00",
        )
        with self.assertRaisesRegex(Exception, "provider_published_after_capture"):
            transport.prepare_collection(
                session="2026-10-01",
                decision_at="2026-10-01 18:00:00+08:00",
                attempts=(
                    RawEastmoneyTransportAttempt(
                        collection_id="forward-eod-20261001-clock-late",
                        page_number=1,
                        canonical_url=manifest.canonical_url(1),
                        raw=b'{"fixture":"late-provider-clock"}',
                        captured_at="2026-10-01 16:30:00+08:00",
                        provider_published_at="2026-10-01 16:31:00+08:00",
                    ),
                ),
            )

    def test_transport_rejects_cache_fallback_and_url_deviations_but_not_late_evidence(self):
        manifest = fixed_eastmoney_eod_manifest_v1()
        self.journal.register(manifest, recorded_at="2026-10-01 14:00:00+08:00")
        transport = self.journal.open_raw_transport(manifest)
        base = dict(
            collection_id="forward-eod-20261001-reject",
            page_number=1,
            canonical_url=manifest.canonical_url(1),
            raw=b'{"fixture":"rejected-route"}',
            captured_at="2026-10-01 16:30:00+08:00",
        )
        invalid = {
            "cache": {**base, "cache_reused": True},
            "fallback": {**base, "fallback_attempted": True},
            "url": {**base, "canonical_url": "https://example.test/not-eastmoney"},
        }
        for label, values in invalid.items():
            with self.subTest(label=label):
                with self.assertRaises(SourceManifestContractError):
                    transport.prepare_collection(
                        session="2026-10-01",
                        decision_at="2026-10-01 18:00:00+08:00",
                        attempts=(RawEastmoneyTransportAttempt(**values),),
                    )

        late = transport.prepare_collection(
            session="2026-10-01",
            decision_at="2026-10-01 18:00:00+08:00",
            attempts=(
                RawEastmoneyTransportAttempt(
                    **{**base, "captured_at": "2026-10-01 17:51:00+08:00"}
                ),
            ),
        )
        self.assertEqual(str(late.capture_pages[0].captured_at), "2026-10-01 17:51:00+08:00")

    def test_transport_requires_manifest_to_precede_the_raw_capture(self):
        manifest = fixed_eastmoney_eod_manifest_v1()
        self.journal.register(manifest, recorded_at="2026-10-01 16:31:00+08:00")
        transport = self.journal.open_raw_transport(manifest)

        with self.assertRaisesRegex(SourceManifestContractError, "registration_after_capture"):
            transport.prepare_collection(
                session="2026-10-01",
                decision_at="2026-10-01 18:00:00+08:00",
                attempts=(
                    RawEastmoneyTransportAttempt(
                        collection_id="forward-eod-20261001-too-early",
                        page_number=1,
                        canonical_url=manifest.canonical_url(1),
                        raw=b'{"fixture":"before-manifest"}',
                        captured_at="2026-10-01 16:30:00+08:00",
                    ),
                ),
            )

    def test_tampered_manifest_journal_fails_closed_before_transport_can_open(self):
        manifest = fixed_eastmoney_eod_manifest_v1()
        self.journal.register(manifest, recorded_at="2026-10-01 14:00:00+08:00")
        record = json.loads(self.journal.path.read_text(encoding="utf-8"))
        record["manifest"]["page_contract"]["page_size"] = 99
        self.journal.path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

        with self.assertRaises(SourceManifestContractError):
            self.journal.verify()
        with self.assertRaises(SourceManifestContractError):
            self.journal.open_raw_transport(manifest)

    def test_contract_module_has_no_network_or_account_dependency(self):
        module = Path(__file__).parents[1] / "quantlab" / "forward_source_manifest.py"
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
        )
        self.assertFalse(
            [name for name in imported if name.startswith(forbidden_prefixes)],
            imported,
        )


if __name__ == "__main__":
    unittest.main()
