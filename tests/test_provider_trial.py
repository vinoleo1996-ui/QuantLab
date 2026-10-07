"""Behaviour tests for fail-closed external-provider trial qualification."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from quantlab.data.l2_contract import L2_FEATURE_KEYS
from quantlab.data.provider_trial import (
    CapabilityDeclaration,
    HistoricalBarSeed,
    L2Evidence,
    L2FieldEvidence,
    PageReconciliation,
    PermissionDeclaration,
    ProviderTrialError,
    ProviderTrialSubmission,
    SampleReceipt,
    TermsIdentity,
    TextEvidence,
    TextSampleEvidence,
    ThemeMarketEvidence,
    acceptance_requirements,
    qualify_provider_trial,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


CAPABILITIES = {
    "theme_market": (
        "stock_theme_multi_mapping",
        "active_theme_semantics",
        "limit_up_events",
        "failed_board_events",
        "daily_bars",
        "paged_snapshot_reconciliation",
        "u0_membership_coverage",
    ),
    "text": (
        "canonical_content",
        "original_publisher_identity",
        "published_timestamp",
        "reprint_identity",
    ),
    "l2": (
        "fixed_14_field_dictionary",
        "per_field_available_at",
        "per_field_source_version",
        "per_field_raw_receipt",
        "source_as_of",
    ),
}


class ProviderTrialQualificationTests(unittest.TestCase):
    def capabilities(self, source_class: str):
        return tuple(
            CapabilityDeclaration(
                capability=name,
                available=True,
                evidence_locator="https://contract.example.test/capabilities#" + name,
                evidence_sha256=digest("capability:" + name),
            )
            for name in CAPABILITIES[source_class]
        )

    def receipts(self, *names: str):
        return tuple(
            SampleReceipt(
                receipt_id="receipt-" + name,
                raw_sha256=digest("raw:" + name),
                source_locator="https://provider.example.test/sample/" + name,
                captured_at="2026-09-29T17:20:00+08:00",
            )
            for name in names
        )

    def base_submission(self, source_class: str, receipts, **evidence):
        return ProviderTrialSubmission(
            source_class=source_class,
            provider_id="provider-example",
            trial_id="trial-20260929-a",
            source_dataset_id="dataset-example",
            source_release_id="release-20260929",
            source_release_sha256=digest("release-20260929"),
            terms=TermsIdentity(
                contract_id="MSA-trial-example",
                version="2026-09-29",
                locator="https://contract.example.test/terms",
                sha256=digest("terms-20260929"),
            ),
            automated_access=PermissionDeclaration(
                permission="automated_access",
                granted=True,
                scope="read-only research API access",
                evidence_locator="https://contract.example.test/terms#api",
                evidence_sha256=digest("permission:automated_access"),
            ),
            raw_retention=PermissionDeclaration(
                permission="raw_retention",
                granted=True,
                scope="retain immutable raw research receipts",
                evidence_locator="https://contract.example.test/terms#retention",
                evidence_sha256=digest("permission:raw_retention"),
            ),
            capabilities=self.capabilities(source_class),
            sample_receipts=receipts,
            **evidence,
        )

    def theme_submission(self):
        receipts = self.receipts("seed", "page-1", "page-2", "mapping", "limit-up", "failed-board", "bars")
        raw = {receipt.receipt_id.removeprefix("receipt-"): receipt.raw_sha256 for receipt in receipts}
        return self.base_submission(
            "theme_market",
            receipts,
            theme_market=ThemeMarketEvidence(
                historical_bar_seed=HistoricalBarSeed(
                    seed_id="u0-seed-20260929",
                    bar_set_sha256=digest("bar-set"),
                    calendar_sha256=digest("calendar"),
                    instrument_coverage_sha256=digest("instrument-coverage"),
                    raw_receipt_sha256=raw["seed"],
                    session_count=253,
                    first_session="2025-09-22",
                    last_session="2026-09-29",
                ),
                page_reconciliation=PageReconciliation(
                    snapshot_id="market-snapshot-20260929",
                    coverage_sha256=digest("market-coverage"),
                    expected_page_count=2,
                    observed_page_numbers=(1, 2),
                    page_receipt_sha256s=(raw["page-1"], raw["page-2"]),
                    expected_instrument_count=100,
                    observed_unique_instrument_count=100,
                    duplicate_instrument_count=0,
                    u0_membership_sha256=digest("u0-membership"),
                    expected_u0_member_count=5,
                    mapped_u0_member_count=5,
                    unmapped_u0_member_count=0,
                ),
                theme_mapping_receipt_sha256=raw["mapping"],
                limit_up_receipt_sha256=raw["limit-up"],
                failed_board_receipt_sha256=raw["failed-board"],
                market_bar_receipt_sha256=raw["bars"],
            ),
        )

    def text_submission(self):
        receipts = self.receipts("article-a", "article-b", "reprint-a")
        raw = {receipt.receipt_id.removeprefix("receipt-"): receipt.raw_sha256 for receipt in receipts}
        original_a = TextSampleEvidence(
            document_id="origin-a",
            raw_receipt_sha256=raw["article-a"],
            content_sha256=digest("article-a-content"),
            canonical_url="https://publisher-a.example.test/story/a",
            publisher_id="publisher-a",
            original_publisher_id="publisher-a",
            published_at="2026-09-29T15:40:00+08:00",
            original_published_at="2026-09-29T15:40:00+08:00",
            is_reprint=False,
        )
        original_b = TextSampleEvidence(
            document_id="origin-b",
            raw_receipt_sha256=raw["article-b"],
            content_sha256=digest("article-b-content"),
            canonical_url="https://publisher-b.example.test/story/b",
            publisher_id="publisher-b",
            original_publisher_id="publisher-b",
            published_at="2026-09-29T15:45:00+08:00",
            original_published_at="2026-09-29T15:45:00+08:00",
            is_reprint=False,
        )
        reprint = TextSampleEvidence(
            document_id="reprint-a",
            raw_receipt_sha256=raw["reprint-a"],
            content_sha256=digest("reprint-a-content"),
            canonical_url="https://publisher-c.example.test/reprint/a",
            publisher_id="publisher-c",
            original_publisher_id="publisher-a",
            published_at="2026-09-29T16:00:00+08:00",
            original_published_at="2026-09-29T15:40:00+08:00",
            is_reprint=True,
            reprint_of_document_id="origin-a",
            reprint_of_canonical_url=original_a.canonical_url,
        )
        return self.base_submission("text", receipts, text=TextEvidence((original_a, original_b, reprint)))

    def l2_submission(self):
        receipts = self.receipts("l2-field-dictionary")
        raw = receipts[0].raw_sha256
        fields = tuple(
            L2FieldEvidence(
                key=key,
                unit="ratio",
                source_field="provider." + key,
                available_at_field="available_at",
                source_version="release-20260929",
                raw_receipt_sha256=raw,
                source_as_of_field="source_as_of",
            )
            for key in L2_FEATURE_KEYS
        )
        return self.base_submission(
            "l2",
            receipts,
            l2=L2Evidence(
                dataset_id="l2-trial-dataset",
                source_release_id="release-20260929",
                field_dictionary_sha256=digest("l2-field-dictionary"),
                sample_matrix_sha256=digest("l2-sample-matrix"),
                fields=fields,
            ),
        )

    def test_theme_market_qualification_records_u0_seed_pages_and_all_component_receipts(self):
        qualified = qualify_provider_trial(self.theme_submission())
        record = qualified.to_record()
        self.assertEqual(record["status"], "qualified")
        self.assertEqual(record["source_class"], "theme_market")
        self.assertEqual(record["submission"]["theme_market"]["historical_bar_seed"]["session_count"], 253)
        self.assertIn("u0_bar_seed", record["acceptance_requirement_ids"])
        self.assertEqual(len(record["qualification_sha256"]), 64)
        self.assertEqual(qualified.sha256, qualify_provider_trial(self.theme_submission()).sha256)

    def test_theme_market_rejects_a_non_253_seed_and_unreconciled_page_coverage(self):
        submission = self.theme_submission()
        bad_seed = replace(submission.theme_market.historical_bar_seed, session_count=252)
        with self.assertRaisesRegex(ProviderTrialError, "exactly_253"):
            qualify_provider_trial(
                replace(submission, theme_market=replace(submission.theme_market, historical_bar_seed=bad_seed))
            )
        bad_pages = replace(submission.theme_market.page_reconciliation, observed_page_numbers=(1,))
        with self.assertRaisesRegex(ProviderTrialError, "page_numbers_not_exactly_reconciled"):
            qualify_provider_trial(
                replace(submission, theme_market=replace(submission.theme_market, page_reconciliation=bad_pages))
            )
        bad_u0 = replace(
            submission.theme_market.page_reconciliation,
            mapped_u0_member_count=4,
            unmapped_u0_member_count=1,
        )
        with self.assertRaisesRegex(ProviderTrialError, "u0_theme_coverage_not_reconciled"):
            qualify_provider_trial(
                replace(submission, theme_market=replace(submission.theme_market, page_reconciliation=bad_u0))
            )

    def test_text_qualification_requires_two_independent_originals_and_an_auditable_reprint_link(self):
        qualified = qualify_provider_trial(self.text_submission())
        self.assertEqual(qualified.submission.source_class, "text")
        self.assertIn("text_reprint_identity", qualified.requirement_ids)

        submission = self.text_submission()
        first, second, reprint = submission.text.samples
        no_independence = replace(second, publisher_id="publisher-a", original_publisher_id="publisher-a")
        with self.assertRaisesRegex(ProviderTrialError, "independent_original_publishers"):
            qualify_provider_trial(replace(submission, text=TextEvidence((first, no_independence, reprint))))
        duplicate_content = replace(second, content_sha256=first.content_sha256)
        with self.assertRaisesRegex(ProviderTrialError, "independent_original_content_not_proven"):
            qualify_provider_trial(replace(submission, text=TextEvidence((first, duplicate_content, reprint))))
        bad_link = replace(reprint, reprint_of_document_id="not-in-sample")
        with self.assertRaisesRegex(ProviderTrialError, "reprint_original_not_in_trial_samples"):
            qualify_provider_trial(replace(submission, text=TextEvidence((first, second, bad_link))))

    def test_text_rejects_date_only_or_unknown_reprint_status(self):
        submission = self.text_submission()
        first, second, reprint = submission.text.samples
        date_only = replace(first, published_at="2026-09-29", original_published_at="2026-09-29")
        with self.assertRaisesRegex(ProviderTrialError, "timezone_required_text_published_at"):
            qualify_provider_trial(replace(submission, text=TextEvidence((date_only, second, reprint))))
        unknown_status = replace(reprint, is_reprint="yes")
        with self.assertRaisesRegex(ProviderTrialError, "boolean_required_text_is_reprint"):
            qualify_provider_trial(replace(submission, text=TextEvidence((first, second, unknown_status))))

    def test_l2_qualification_requires_all_fixed_fields_with_available_at_version_and_receipt_identity(self):
        qualified = qualify_provider_trial(self.l2_submission())
        self.assertEqual(qualified.submission.source_class, "l2")
        self.assertIn("l2_fixed_fields", qualified.requirement_ids)

        submission = self.l2_submission()
        with self.assertRaisesRegex(ProviderTrialError, "fixed_14_key_order"):
            qualify_provider_trial(replace(submission, l2=replace(submission.l2, fields=submission.l2.fields[:-1])))
        bad_field = replace(submission.l2.fields[0], available_at_field="published_at")
        with self.assertRaisesRegex(ProviderTrialError, "available_at_field_must_be_available_at"):
            qualify_provider_trial(
                replace(submission, l2=replace(submission.l2, fields=(bad_field, *submission.l2.fields[1:])))
            )
        missing_receipt = replace(submission.l2.fields[0], raw_receipt_sha256=digest("not-a-sample"))
        with self.assertRaisesRegex(ProviderTrialError, "receipt_not_in_trial_samples"):
            qualify_provider_trial(
                replace(submission, l2=replace(submission.l2, fields=(missing_receipt, *submission.l2.fields[1:])))
            )

    def test_common_permissions_terms_capabilities_and_cross_class_evidence_are_fail_closed(self):
        submission = self.theme_submission()
        denied = replace(submission.automated_access, granted=False)
        with self.assertRaisesRegex(ProviderTrialError, "permission_not_granted:automated_access"):
            qualify_provider_trial(replace(submission, automated_access=denied))
        ambiguous_terms = replace(submission.terms, sha256="unknown")
        with self.assertRaisesRegex(ProviderTrialError, "missing_or_ambiguous_terms_sha256"):
            qualify_provider_trial(replace(submission, terms=ambiguous_terms))
        missing_capability = submission.capabilities[:-1]
        with self.assertRaisesRegex(ProviderTrialError, "capability_declarations_must_match_exactly"):
            qualify_provider_trial(replace(submission, capabilities=missing_capability))
        with self.assertRaisesRegex(ProviderTrialError, "cross_class_evidence_forbidden"):
            qualify_provider_trial(replace(submission, l2=self.l2_submission().l2))

    def test_requirements_are_exact_and_class_specific(self):
        theme_ids = [item.requirement_id for item in acceptance_requirements("theme_market")]
        l2_ids = [item.requirement_id for item in acceptance_requirements("l2")]
        self.assertIn("u0_bar_seed", theme_ids)
        self.assertNotIn("l2_fixed_fields", theme_ids)
        self.assertIn("l2_fixed_fields", l2_ids)
        with self.assertRaisesRegex(ProviderTrialError, "unknown_source_class"):
            acceptance_requirements("market")


if __name__ == "__main__":
    unittest.main()
