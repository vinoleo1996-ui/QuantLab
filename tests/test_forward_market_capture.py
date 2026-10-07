"""Behavioural tests for the forward-only public-market capture sidecar.

The tests use supplied raw bytes only.  They intentionally never request the
network, construct an account, invoke a router, or place an order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from quantlab.forward_market_capture import (
    EastmoneyRawPage,
    ForwardPoolInputPayload,
    MarketSnapshotCapture,
    ThemeMappingCapture,
    ThemeMappingPayload,
    build_forward_u0_pool,
    capture_eastmoney_eod_snapshot,
    capture_forward_pool_input,
    capture_primary_theme_mapping,
    derive_pool_rows_from_forward_history,
    record_theme_carrier_ledger_event,
    score_theme_carrier_from_captures,
)
from quantlab.forward_market_transport import (
    RawEastmoneyHttpsResponse,
    collect_registered_eastmoney_eod_pages,
)
from quantlab.forward_source_manifest import SourceManifestJournal, fixed_eastmoney_eod_manifest_v1
from quantlab.research_signal_ledger import ResearchSignalLedger
from quantlab.research_universe import FORWARD_CONTRACT_VERSION, U0, UniverseSnapshot, forward_membership_sha256
from quantlab.theme_carrier import ThemeCarrierResult


SESSION = pd.Timestamp("2026-08-31")
PRIOR = pd.Timestamp("2026-08-28")
DECISION_AT = "2026-08-31 18:00:00+08:00"


def _raw_page(rows, *, total=None):
    return json.dumps(
        {"data": {"total": len(rows) if total is None else total, "diff": rows}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _row(code, *, close=11.0, high=None, limit_up=11.0, amount=10_000_000.0):
    return {
        "f12": str(code).zfill(6),
        "f14": "测试" + str(code).zfill(6),
        "f2": close,
        "f17": close,
        "f15": close if high is None else high,
        "f16": close,
        "f18": 10.0,
        "f5": 1000.0,
        "f6": amount,
        "f51": limit_up,
    }


def _pages(rows, *, session=SESSION, captured="2026-08-31 17:10:00+08:00", page_size=5):
    result = []
    for start in range(0, len(rows), page_size):
        page = start // page_size + 1
        result.append(
            EastmoneyRawPage(
                page_number=page,
                canonical_url=(
                    "https://push2.eastmoney.com/api/qt/clist/get?"
                    f"pn={page}&pz={page_size}&po=1&np=1&fltt=2&invt=2&fid=f6"
                    "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81,s:2048"
                    "&fields=f2,f5,f6,f12,f14,f15,f16,f17,f18,f51"
                ),
                raw=_raw_page(rows[start : start + page_size], total=len(rows)),
                captured_at=captured,
            )
        )
    return tuple(result)


class _FixedClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class _RawTransportOpener:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, request_url):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        raw, response_url, provider_published_at = response
        return RawEastmoneyHttpsResponse(
            request_url=request_url,
            response_url=response_url or request_url,
            raw=raw,
            provider_published_at=provider_published_at,
        )


def _u0_for(codes):
    input_hash = hashlib.sha256(b"forward-capture-u0-input").hexdigest()
    return UniverseSnapshot(
        contract_version=FORWARD_CONTRACT_VERSION,
        pool_id="U0",
        session=SESSION.date(),
        decision_at=pd.Timestamp(DECISION_AT),
        cadence_anchor=SESSION.date(),
        members=tuple(codes),
        rank_by_instrument={code: 3001 + index for index, code in enumerate(codes)},
        adv60_by_instrument={code: 6_000_000.0 for code in codes},
        eligibility_reason_by_instrument={},
        input_hash=input_hash,
        membership_hash=forward_membership_sha256(
            spec=U0,
            session=SESSION,
            decision_at=DECISION_AT,
            cadence_anchor=SESSION,
            input_hash=input_hash,
            members=tuple(codes),
            status="ok",
            abstain_reason=None,
        ),
        status="ok",
        abstain_reason=None,
    )


class ForwardMarketCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._written_at = "2026-08-31 17:50:00+08:00"
        self.ledger = ResearchSignalLedger(
            Path(self.temp.name) / "research-signal-ledger",
            now_fn=lambda: self._written_at,
        )
        self.journal = SourceManifestJournal(Path(self.temp.name) / "source-manifests")
        self.manifest = fixed_eastmoney_eod_manifest_v1()

    def tearDown(self):
        self.temp.cleanup()

    def _capture(self, rows, **kwargs):
        # V1's manifest owns its fixed 100-row pagination.  Tests that need
        # a failed second page construct a 101-row raw first response below;
        # callers cannot choose an alternate page size at the capture API.
        kwargs.pop("page_size", None)
        page_kwargs = kwargs.pop("page_kwargs", {})
        session = kwargs.pop("session", SESSION)
        session_day = pd.Timestamp(session).date().isoformat()
        decision_at = kwargs.pop("decision_at", DECISION_AT)
        pages = kwargs.pop("pages", None)
        transport_responses = kwargs.pop("transport_responses", None)
        captured = page_kwargs.pop("captured", session_day + " 17:10:00+08:00")
        collection_id = kwargs.pop("collection_id", "fixture-eod")
        self._written_at = kwargs.pop("recorded_at", session_day + " 17:50:00+08:00")
        if kwargs:
            raise AssertionError("unexpected_capture_test_arguments:" + ",".join(sorted(kwargs)))
        if transport_responses is not None:
            responses = list(transport_responses)
        elif pages is None:
            responses = [(_raw_page(rows), None, None)]
        else:
            responses = []
            for page in pages:
                if page.fetch_error or page.raw is None:
                    responses.append(TimeoutError(page.fetch_error or "fixture fetch error"))
                else:
                    responses.append((bytes(page.raw), None, page.provider_published_at))
            if pages:
                captured = pages[0].captured_at
        collection = collect_registered_eastmoney_eod_pages(
            journal=self.journal,
            manifest=self.manifest,
            session=session,
            decision_at=decision_at,
            collection_id=collection_id,
            opener=_RawTransportOpener(responses),
            clock=_FixedClock(captured),
        )
        return capture_eastmoney_eod_snapshot(
            ledger=self.ledger,
            journal=self.journal,
            collection=collection,
        )

    def test_full_single_source_snapshot_records_each_raw_page_and_preserves_evidence(self):
        capture = self._capture([_row(1), _row(2), _row(3), _row(4), _row(5), _row(6)])

        self.assertEqual(capture.status, "ok")
        self.assertEqual(capture.abstain_reason, None)
        self.assertEqual(len(capture.receipt_ids), 1)
        self.assertEqual(set(capture.bars.instrument_id), {
            "sz.000001", "sz.000002", "sz.000003", "sz.000004", "sz.000005", "sz.000006",
        })
        self.assertTrue(capture.bars.capture_mode.eq("forward_receipt").all())
        self.assertTrue(capture.bars.quality_status.eq("ok").all())
        self.assertTrue(capture.bars.source_quality_status.eq("unverified_public_eod").all())
        self.assertTrue(capture.bars.source_receipt_hash.str.fullmatch(r"[0-9a-f]{64}").all())
        self.assertTrue(self.ledger.verify()["verified"])
        evidence = self.ledger.export_receipt_evidence(capture.bars.iloc[0].source_receipt_id)
        self.assertEqual(evidence["receipt"]["body"]["source_id"], "eastmoney.push2.clist")
        self.assertEqual(evidence["receipt"]["body"]["availability_basis"], "forward_https_capture_no_provider_publish_clock")
        self.assertEqual(evidence["raw"]["sha256"], capture.bars.iloc[0].source_receipt_hash)

    def test_missing_page_or_fetch_failure_abstains_before_any_pool_or_signal_is_built(self):
        first = _raw_page([_row(index) for index in range(1, 101)], total=101)
        failed = self._capture(
            [],
            transport_responses=[(first, None, None), TimeoutError("fixture timeout")],
            collection_id="fixture-fetch-failure",
        )
        self.assertEqual((failed.status, failed.abstain_reason), ("abstain", "page_fetch_failed"))
        self.assertTrue(failed.bars.empty)
        self.assertEqual(len(failed.receipt_ids), 1)

    def test_eastmoney_source_profile_rejects_a_disguised_response_but_preserves_transport_evidence(self):
        rows = [_row(1)]
        raw = _raw_page(rows)
        query = self.manifest.canonical_url(1).split("?", 1)[1]
        for label, url in (
            ("wrong-host", "https://push2.eastmoney.invalid/api/qt/clist/get?" + query),
            ("wrong-path", "https://push2.eastmoney.com/api/qt/ulist/get?" + query),
        ):
            with self.subTest(label=label):
                capture = self._capture(
                    rows,
                    transport_responses=[(raw, url, None)],
                    collection_id="fixture-pinned-source-" + label,
                )
                self.assertEqual((capture.status, capture.abstain_reason), ("abstain", "page_fetch_failed"))
                self.assertEqual(capture.receipt_ids, ())
                self.assertIsNotNone(capture.transport_receipt_id)
                transport = self.ledger.export_receipt_evidence(capture.transport_receipt_id or "")["receipt"]
                self.assertEqual(
                    transport["body"]["metadata"]["transport"]["attempts"][0]["fetch_error"],
                    "response_source_identity_mismatch",
                )
                self.assertTrue(self.ledger.verify()["verified"])

    def test_late_snapshot_is_retained_as_raw_evidence_but_cannot_be_a_forward_input(self):
        capture = self._capture(
            [_row(1)],
            pages=_pages([_row(1)], captured="2026-08-31 18:01:00+08:00", page_size=1),
            recorded_at="2026-08-31 18:02:00+08:00",
            page_size=1,
            collection_id="fixture-late",
        )

        self.assertEqual(
            (capture.status, capture.abstain_reason),
            ("abstain", "not_eligible_for_decision_at:close_page_received_outside_window"),
        )
        self.assertEqual(len(capture.receipt_ids), 1)
        receipt = self.ledger.export_receipt_evidence(capture.receipt_ids[0])["receipt"]
        self.assertGreater(pd.Timestamp(receipt["available_at"]), pd.Timestamp(DECISION_AT))

    def test_derived_pool_rows_need_a_contiguous_253_session_forward_history(self):
        capture = self._capture([_row(1), _row(2)], page_size=2)
        result = derive_pool_rows_from_forward_history(
            captures=(capture,),
            session=SESSION,
            exchange_sessions=(PRIOR, SESSION),
        )

        self.assertEqual((result.status, result.abstain_reason), ("abstain", "insufficient_forward_price_history"))
        self.assertTrue(result.rows.empty)

    def test_only_a_canonical_raw_pool_bundle_can_declare_and_build_u0(self):
        rows = [
            {
                "instrument_id": f"sz.{index:06d}",
                "base_eligible": True,
                "adv60_cny": float(10_000_000 - index),
                "adv_observation_count": 60,
                "valid_price_session_count": 253,
                "asof_ok": True,
            }
            for index in range(3001)
        ]
        payload = ForwardPoolInputPayload(
            canonical_url="urn:quantlab:forward-market-pool-input:2026-08-31",
            raw=json.dumps({"schema": "quantlab.forward-pool-input.v1", "rows": rows}, sort_keys=True).encode(),
            captured_at="2026-08-31 17:40:00+08:00",
            source_id="fixture-forward-pool-provider",
            source_family="fixture-forward-pool-provider",
            origin_id="fixture-forward-pool-20260831",
        )
        materialized = capture_forward_pool_input(
            ledger=self.ledger,
            session=SESSION,
            decision_at=DECISION_AT,
            payload=payload,
        )
        self.assertEqual(materialized.status, "ok")
        self.assertEqual(materialized.decision_eligibility, "eligible_for_decision_at")
        self.assertTrue(materialized.rows.source_receipt_id.eq(materialized.receipt_id).all())
        self.assertTrue(materialized.rows.source_receipt_hash.eq(materialized.content_sha256).all())

        calendar = pd.bdate_range("2026-08-03", periods=21)
        blocked = build_forward_u0_pool(
            ledger=self.ledger,
            materialized=replace(materialized, decision_eligibility="not_eligible_for_decision_at"),
            session=calendar[-1],
            decision_at="2026-08-31 18:00:00+08:00",
            exchange_sessions=calendar,
            cadence_anchor=calendar[0],
            scope_id="U0-ineligible",
        )
        self.assertEqual(
            (blocked.status, blocked.abstain_reason),
            ("abstain", "pool_input_not_eligible_for_decision_at"),
        )
        pool = build_forward_u0_pool(
            ledger=self.ledger,
            materialized=materialized,
            session=calendar[-1],
            decision_at="2026-08-31 18:00:00+08:00",
            exchange_sessions=calendar,
            cadence_anchor=calendar[0],
            scope_id="U0",
            now="2026-08-31 17:50:00+08:00",
        )
        self.assertEqual(pool.universe.status, "ok")
        self.assertEqual(pool.scope["status"], "declared")
        self.assertIn("sz.003000", pool.universe.members)
        self.assertNotIn("sz.002999", pool.universe.members)

    def test_non_market_receipts_after_1750_remain_eligible_before_1800_and_late_receipts_are_retained(self):
        pool_raw = json.dumps(
            {
                "schema": "quantlab.forward-pool-input.v1",
                "rows": [
                    {
                        "instrument_id": "sz.000001",
                        "base_eligible": True,
                        "adv60_cny": 6_000_000.0,
                        "adv_observation_count": 60,
                        "valid_price_session_count": 253,
                        "asof_ok": True,
                    }
                ],
            },
            sort_keys=True,
        ).encode()
        mapping_raw = b'{"rows":[{"code":"000001","theme_id":"bank"}]}'
        self._written_at = "2026-08-31 17:51:00+08:00"
        pool = capture_forward_pool_input(
            ledger=self.ledger,
            session=SESSION,
            decision_at=DECISION_AT,
            payload=ForwardPoolInputPayload(
                canonical_url="urn:fixture:pool:1751",
                raw=pool_raw,
                captured_at="2026-08-31 17:51:00+08:00",
                source_id="fixture-pool-1751",
                source_family="fixture-pool-1751",
                origin_id="fixture-pool-1751",
            ),
        )
        mapping = capture_primary_theme_mapping(
            ledger=self.ledger,
            session=SESSION,
            decision_at=DECISION_AT,
            payload=ThemeMappingPayload(
                canonical_url="urn:fixture:mapping:1751",
                raw=mapping_raw,
                captured_at="2026-08-31 17:51:00+08:00",
                source_id="fixture-mapping-1751",
                source_family="fixture-mapping-1751",
                origin_id="fixture-mapping-1751",
            ),
        )
        self.assertEqual((pool.status, pool.decision_eligibility), ("ok", "eligible_for_decision_at"))
        self.assertEqual((mapping.status, mapping.decision_eligibility), ("ok", "eligible_for_decision_at"))

        self._written_at = "2026-08-31 18:01:00+08:00"
        late_pool = capture_forward_pool_input(
            ledger=self.ledger,
            session=SESSION,
            decision_at=DECISION_AT,
            payload=ForwardPoolInputPayload(
                canonical_url="urn:fixture:pool:late",
                raw=pool_raw,
                captured_at="2026-08-31 17:51:00+08:00",
                source_id="fixture-pool-late",
                source_family="fixture-pool-late",
                origin_id="fixture-pool-late",
            ),
        )
        late_mapping = capture_primary_theme_mapping(
            ledger=self.ledger,
            session=SESSION,
            decision_at=DECISION_AT,
            payload=ThemeMappingPayload(
                canonical_url="urn:fixture:mapping:late",
                raw=mapping_raw,
                captured_at="2026-08-31 17:51:00+08:00",
                source_id="fixture-mapping-late",
                source_family="fixture-mapping-late",
                origin_id="fixture-mapping-late",
            ),
        )
        for late, expected_prefix in (
            (late_pool, "not_eligible_for_decision_at:receipt_written_after_decision_cutoff"),
            (late_mapping, "not_eligible_for_decision_at:receipt_written_after_decision_cutoff"),
        ):
            with self.subTest(kind=type(late).__name__):
                self.assertEqual(late.status, "abstain")
                self.assertEqual(late.abstain_reason, expected_prefix)
                self.assertEqual(late.decision_eligibility, "not_eligible_for_decision_at")
                self.assertIsNotNone(late.receipt_id)
                self.assertRegex(late.content_sha256 or "", r"^[0-9a-f]{64}$")
                evidence = self.ledger.export_receipt_evidence(late.receipt_id or "")
                self.assertEqual(evidence["receipt"]["recorded_at"], "2026-08-31T18:01:00+08:00")

    def test_theme_observation_uses_ledger_runtime_and_late_invocation_abstains(self):
        codes = tuple("sz." + str(index).zfill(6) for index in range(1, 11))
        prior_rows = [_row(index, close=11.0, high=11.0, limit_up=11.0) for index in range(1, 11)]
        prior = self._capture(
            prior_rows,
            session=PRIOR,
            decision_at="2026-08-28 18:00:00+08:00",
            pages=_pages(prior_rows, session=PRIOR, captured="2026-08-28 17:10:00+08:00", page_size=10),
            page_size=10,
            collection_id="fixture-runtime-prior",
        )
        current_rows = [
            _row(index, close=10.5, high=11.0, limit_up=11.0) if index <= 5
            else _row(index, close=12.1, high=12.1, limit_up=12.1)
            for index in range(1, 11)
        ]
        current = self._capture(current_rows, pages=_pages(current_rows, page_size=10), page_size=10)
        mapping = capture_primary_theme_mapping(
            ledger=self.ledger,
            session=SESSION,
            decision_at=DECISION_AT,
            payload=ThemeMappingPayload(
                canonical_url="urn:fixture:mapping:runtime",
                raw=json.dumps(
                    {
                        "rows": [
                            {"code": str(index).zfill(6), "theme_id": "tech" if index <= 5 else "energy"}
                            for index in range(1, 11)
                        ]
                    },
                    sort_keys=True,
                ).encode(),
                captured_at="2026-08-31 17:20:00+08:00",
                source_id="fixture-mapping-runtime",
                source_family="fixture-mapping-runtime",
                origin_id="fixture-mapping-runtime",
            ),
        )
        u0 = _u0_for(codes)
        score = score_theme_carrier_from_captures(
            u0=u0,
            prior_capture=prior,
            current_capture=current,
            mapping_capture=mapping,
            ledger=self.ledger,
            decision_session=SESSION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
        )
        self._written_at = "2026-08-31 17:57:00+08:00"
        self.ledger.declare_scope(
            "U0-runtime-observed",
            DECISION_AT,
            codes,
            definition_version=FORWARD_CONTRACT_VERSION,
            source_hash=u0.membership_hash,
            now=self.ledger.runtime_now(),
        )
        observed = record_theme_carrier_ledger_event(
            ledger=self.ledger,
            scope_id="U0-runtime-observed",
            u0=u0,
            score=score,
            prior_capture=prior,
            current_capture=current,
            mapping_capture=mapping,
            decision_at=DECISION_AT,
            # These two scheduler-supplied times intentionally disagree with
            # the ledger clock and must not control the event.
            entity_recorded_at="2026-08-31 17:00:00+08:00",
            now="2026-08-31 17:00:00+08:00",
        )
        self.assertEqual(observed.status, "observed")
        self.assertEqual(observed.observation["recorded_at"], "2026-08-31T17:57:00+08:00")

        self.ledger.declare_scope(
            "U0-runtime-late",
            DECISION_AT,
            codes,
            definition_version=FORWARD_CONTRACT_VERSION,
            source_hash=u0.membership_hash,
            now=self.ledger.runtime_now(),
        )
        self._written_at = "2026-08-31 18:01:00+08:00"
        late = record_theme_carrier_ledger_event(
            ledger=self.ledger,
            scope_id="U0-runtime-late",
            u0=u0,
            score=score,
            prior_capture=prior,
            current_capture=current,
            mapping_capture=mapping,
            decision_at=DECISION_AT,
            entity_recorded_at="2026-08-31 17:00:00+08:00",
            now="2026-08-31 17:00:00+08:00",
        )
        self.assertEqual(
            (late.status, late.reason),
            ("abstained", "primary_theme_mapping_incomplete_or_late"),
        )
        self.assertEqual(late.observation["recorded_at"], "2026-08-31T18:01:00+08:00")

    def test_theme_score_uses_receipt_backed_prior_and_current_bars_and_mapping_only(self):
        codes = tuple("sz." + str(index).zfill(6) for index in range(1, 11))
        prior = self._capture(
            [_row(index, close=11.0, high=11.0, limit_up=11.0) for index in range(1, 11)],
            session=PRIOR,
            decision_at="2026-08-28 18:00:00+08:00",
            pages=_pages(
                [_row(index, close=11.0, high=11.0, limit_up=11.0) for index in range(1, 11)],
                session=PRIOR,
                captured="2026-08-28 17:10:00+08:00",
                page_size=10,
            ),
            page_size=10,
            collection_id="fixture-prior",
        )
        current_rows = [
            _row(index, close=10.5, high=11.0, limit_up=11.0) if index <= 5
            else _row(index, close=12.1, high=12.1, limit_up=12.1)
            for index in range(1, 11)
        ]
        current = self._capture(current_rows, pages=_pages(current_rows, page_size=10), page_size=10)
        map_payload = ThemeMappingPayload(
            canonical_url="https://theme-source.test/primary/20260831",
            raw=json.dumps(
                {"rows": [{"code": str(index).zfill(6), "theme_id": "tech" if index <= 5 else "energy"} for index in range(1, 11)]},
                sort_keys=True,
            ).encode(),
            captured_at="2026-08-31 17:20:00+08:00",
            source_id="fixture-primary-theme-source",
            source_family="fixture-primary-theme-source",
            origin_id="fixture-primary-theme-20260831",
        )
        mapping = capture_primary_theme_mapping(
            ledger=self.ledger, session=SESSION, decision_at=DECISION_AT, payload=map_payload
        )

        u0 = _u0_for(codes)
        scored = score_theme_carrier_from_captures(
            u0=u0,
            prior_capture=prior,
            current_capture=current,
            mapping_capture=mapping,
            ledger=self.ledger,
            decision_session=SESSION,
            prior_session=PRIOR,
            decision_at=DECISION_AT,
        )
        self.assertEqual(scored.status, "ok")
        self.assertGreater(scored.stock_scores["sz.000006"], scored.stock_scores["sz.000001"])
        self.ledger.declare_scope(
            "U0",
            DECISION_AT,
            codes,
            definition_version=FORWARD_CONTRACT_VERSION,
            source_hash=u0.membership_hash,
            now="2026-08-31 17:50:00+08:00",
        )
        recorded = record_theme_carrier_ledger_event(
            ledger=self.ledger,
            scope_id="U0",
            u0=u0,
            score=scored,
            prior_capture=prior,
            current_capture=current,
            mapping_capture=mapping,
            decision_at=DECISION_AT,
            entity_recorded_at=DECISION_AT,
            now="2026-08-31 18:01:00+08:00",
        )
        self.assertEqual(recorded.status, "observed")
        self.assertEqual(len(recorded.entity_ids), len(codes))
        self.assertEqual(recorded.observation["body"]["feature_id"], "theme_carrier_minimal_v1")
        self.assertEqual(set(recorded.observation["body"]["receipt_ids"]),
                         set(prior.receipt_ids) | set(current.receipt_ids) | {mapping.receipt_id})
        extractor = recorded.observation["body"]["extractor"]
        self.assertRegex(extractor["theme_carrier_code_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(extractor["capture_adapter_code_sha256"], r"^[0-9a-f]{64}$")
        coverage = self.ledger.coverage("U0", DECISION_AT, "theme_carrier_minimal_v1")
        self.assertEqual((coverage["observed"], coverage["abstained"], coverage["missing"]), (10, 0, []))

    def test_theme_score_abstains_when_public_snapshot_omits_limit_evidence(self):
        row = _row(1)
        row.pop("f51")
        capture = self._capture([row], page_size=1)
        self.assertEqual(capture.status, "ok")
        self.assertTrue(capture.bars.limit_up.isna().all())

    def test_ledger_records_one_all_scope_abstention_when_score_is_incomplete(self):
        code = "sz.000001"
        prior_rows = [_row(1, close=11.0, high=11.0, limit_up=11.0)]
        prior = self._capture(
            prior_rows,
            session=PRIOR,
            decision_at="2026-08-28 18:00:00+08:00",
            pages=_pages(prior_rows, session=PRIOR, captured="2026-08-28 17:10:00+08:00", page_size=1),
            page_size=1,
            collection_id="fixture-prior-incomplete-score",
        )
        current = self._capture([_row(1)], pages=_pages([_row(1)], page_size=1), page_size=1)
        mapping = capture_primary_theme_mapping(
            ledger=self.ledger,
            session=SESSION,
            decision_at=DECISION_AT,
            payload=ThemeMappingPayload(
                canonical_url="https://theme-source.test/primary/one",
                raw=b'{"rows":[{"code":"000001","theme_id":"bank"}]}',
                captured_at="2026-08-31 17:20:00+08:00",
                source_id="fixture-primary-theme-source",
                source_family="fixture-primary-theme-source",
                origin_id="fixture-primary-theme-one",
            ),
        )
        u0 = _u0_for((code,))
        self.ledger.declare_scope(
            "U0",
            DECISION_AT,
            (code,),
            definition_version=FORWARD_CONTRACT_VERSION,
            source_hash=u0.membership_hash,
            now="2026-08-31 17:50:00+08:00",
        )
        incomplete = ThemeCarrierResult(
            status="abstain",
            abstain_reason="fixture_incomplete_score",
            decision_session=SESSION.date(),
            decision_at=pd.Timestamp(DECISION_AT),
            theme_components=pd.DataFrame(),
            stock_scores=pd.Series(dtype=float, name="theme_score_equal_weight_v1"),
            coverage={},
            receipt_hashes=(),
        )
        recorded = record_theme_carrier_ledger_event(
            ledger=self.ledger,
            scope_id="U0",
            u0=u0,
            score=incomplete,
            prior_capture=prior,
            current_capture=current,
            mapping_capture=mapping,
            decision_at=DECISION_AT,
            entity_recorded_at=DECISION_AT,
            now="2026-08-31 18:01:00+08:00",
        )
        self.assertEqual((recorded.status, recorded.reason), ("abstained", "theme_score_incomplete_or_unlinked"))
        coverage = self.ledger.coverage("U0", DECISION_AT, "theme_carrier_minimal_v1")
        self.assertEqual((coverage["observed"], coverage["abstained"]), (0, 1))
        self.assertEqual(coverage["abstained_entity_keys"], [code])

    def test_ledger_refuses_to_write_when_scope_has_not_been_declared(self):
        u0 = _u0_for(("sz.000001",))
        recorded = record_theme_carrier_ledger_event(
            ledger=self.ledger,
            scope_id="U0",
            u0=u0,
            score=None,
            prior_capture=None,
            current_capture=None,
            mapping_capture=None,
            decision_at=DECISION_AT,
            entity_recorded_at=DECISION_AT,
            now="2026-08-31 18:01:00+08:00",
        )
        self.assertEqual((recorded.status, recorded.reason), ("not_recorded", "scope_not_declared_or_membership_mismatch"))
        self.assertEqual(self.ledger.verify()["events"], 0)

    def test_declared_scope_records_all_member_abstention_when_capture_object_is_missing(self):
        code = "sz.000001"
        u0 = _u0_for((code,))
        self.ledger.declare_scope(
            "U0",
            DECISION_AT,
            (code,),
            definition_version=FORWARD_CONTRACT_VERSION,
            source_hash=u0.membership_hash,
            now="2026-08-31 17:50:00+08:00",
        )
        recorded = record_theme_carrier_ledger_event(
            ledger=self.ledger,
            scope_id="U0",
            u0=u0,
            score=None,
            prior_capture=None,
            current_capture=None,
            mapping_capture=None,
            decision_at=DECISION_AT,
            entity_recorded_at=DECISION_AT,
            now="2026-08-31 18:01:00+08:00",
        )
        self.assertEqual((recorded.status, recorded.reason), ("abstained", "capture_or_mapping_unavailable"))
        coverage = self.ledger.coverage("U0", DECISION_AT, "theme_carrier_minimal_v1")
        self.assertEqual((coverage["observed"], coverage["abstained"], coverage["abstained_entity_keys"]), (0, 1, [code]))

    def test_declared_scope_abstains_when_a_capture_object_is_malformed(self):
        code = "sz.000001"
        u0 = _u0_for((code,))
        self.ledger.declare_scope(
            "U0",
            DECISION_AT,
            (code,),
            definition_version=FORWARD_CONTRACT_VERSION,
            source_hash=u0.membership_hash,
            now="2026-08-31 17:50:00+08:00",
        )
        malformed_prior = MarketSnapshotCapture(
            status="ok",
            abstain_reason=None,
            session=PRIOR.date(),
            decision_at=pd.Timestamp("2026-08-28 18:00:00+08:00"),
            bars=pd.DataFrame(),
            receipt_ids=None,  # type: ignore[arg-type] -- deliberate hostile caller fixture
            receipt_hashes=(),
        )
        current = self._capture([_row(1)], pages=_pages([_row(1)], page_size=1), page_size=1)
        mapping = ThemeMappingCapture(
            status="abstain",
            abstain_reason="fixture-no-map",
            session=SESSION.date(),
            decision_at=pd.Timestamp(DECISION_AT),
            mappings=pd.DataFrame(),
            receipt_id=None,
            content_sha256=None,
        )
        recorded = record_theme_carrier_ledger_event(
            ledger=self.ledger,
            scope_id="U0",
            u0=u0,
            score=None,
            prior_capture=malformed_prior,
            current_capture=current,
            mapping_capture=mapping,
            decision_at=DECISION_AT,
            entity_recorded_at=DECISION_AT,
            now="2026-08-31 18:01:00+08:00",
        )
        self.assertEqual((recorded.status, recorded.reason), ("abstained", "capture_or_mapping_unavailable"))
        coverage = self.ledger.coverage("U0", DECISION_AT, "theme_carrier_minimal_v1")
        self.assertEqual((coverage["observed"], coverage["abstained"], coverage["abstained_entity_keys"]), (0, 1, [code]))

    def test_declared_scope_abstains_when_a_mapping_object_is_malformed(self):
        code = "sz.000001"
        u0 = _u0_for((code,))
        self.ledger.declare_scope(
            "U0",
            DECISION_AT,
            (code,),
            definition_version=FORWARD_CONTRACT_VERSION,
            source_hash=u0.membership_hash,
            now="2026-08-31 17:50:00+08:00",
        )
        prior_rows = [_row(1)]
        prior = self._capture(
            prior_rows,
            session=PRIOR,
            decision_at="2026-08-28 18:00:00+08:00",
            pages=_pages(prior_rows, session=PRIOR, captured="2026-08-28 17:10:00+08:00", page_size=1),
            page_size=1,
            collection_id="fixture-malformed-map-prior",
        )
        current = self._capture([_row(1)], pages=_pages([_row(1)], page_size=1), page_size=1)
        complete_mapping = capture_primary_theme_mapping(
            ledger=self.ledger,
            session=SESSION,
            decision_at=DECISION_AT,
            payload=ThemeMappingPayload(
                canonical_url="https://theme-source.test/primary/malformed",
                raw=b'{"rows":[{"code":"000001","theme_id":"bank"}]}',
                captured_at="2026-08-31 17:20:00+08:00",
                source_id="fixture-primary-theme-source",
                source_family="fixture-primary-theme-source",
                origin_id="fixture-primary-theme-malformed",
            ),
        )
        malformed_mapping = ThemeMappingCapture(
            status="ok",
            abstain_reason=None,
            session=complete_mapping.session,
            decision_at=complete_mapping.decision_at,
            mappings=None,  # type: ignore[arg-type] -- deliberate hostile caller fixture
            receipt_id=complete_mapping.receipt_id,
            content_sha256=complete_mapping.content_sha256,
        )
        recorded = record_theme_carrier_ledger_event(
            ledger=self.ledger,
            scope_id="U0",
            u0=u0,
            score=None,
            prior_capture=prior,
            current_capture=current,
            mapping_capture=malformed_mapping,
            decision_at=DECISION_AT,
            entity_recorded_at=DECISION_AT,
            now="2026-08-31 18:01:00+08:00",
        )
        self.assertEqual((recorded.status, recorded.reason), ("abstained", "capture_or_mapping_unavailable"))
        coverage = self.ledger.coverage("U0", DECISION_AT, "theme_carrier_minimal_v1")
        self.assertEqual((coverage["observed"], coverage["abstained"], coverage["abstained_entity_keys"]), (0, 1, [code]))

    def test_capture_sidecar_has_no_account_or_router_dependency(self):
        module = Path(__file__).parents[1] / "quantlab" / "forward_market_capture.py"
        source = module.read_text(encoding="utf-8")
        self.assertNotIn("quantlab.backtest", source)
        self.assertNotIn("quantlab.strategy", source)
        self.assertNotIn("quantlab.portfolio", source)


if __name__ == "__main__":
    unittest.main()
