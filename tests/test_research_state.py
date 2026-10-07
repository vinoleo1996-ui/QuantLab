import json
import tempfile
import unittest
from pathlib import Path

from quantlab.report.research_state import ALLOWLIST, load_research_catalog


PAYLOADS = {
    "continuity_ledger": "# QuantLab 研究接续台账\n当前状态仅供研究接续。\n",
    "market_data_manifest": {
        "snapshot_id": "fixture-snapshot",
        "as_of": "2026-06-30",
        "panels": {"sh.600000": "fixture"},
        "accounting_gaps": [{"reason": "fixture_gap"}],
        "data_gaps": [{"reason": "fixture_gap"}],
    },
    "cadence_comparison": {
        "status": "same_sample_descriptive_only",
        "promotion_approved": False,
        "arms": {
            "qP2_monthly": {
                "status": "unverified_data_gaps",
                "promotion_approved": False,
                "start": "2020-01-02",
                "end": "2026-06-30",
                "net_cagr": 0.006321,
                "max_drawdown": -0.210086,
                "observations": 1571,
            },
            "qP2_monthly_hold90": {
                "status": "unverified_data_gaps",
                "promotion_approved": False,
                "start": "2020-01-02",
                "end": "2026-06-30",
                "net_cagr": 0.031791,
                "max_drawdown": -0.188879,
                "observations": 1571,
            },
            "cadence13": {
                "status": "unverified_data_gaps",
                "promotion_approved": False,
                "start": "2020-01-02",
                "end": "2026-06-30",
                "net_cagr": 0.027670,
                "max_drawdown": -0.171411,
                "observations": 1571,
            },
        },
    },
    "continuity_attribution": {
        "status": "conditional_posthoc_attribution",
        "promotion_approved": False,
        "account_replayed": False,
    },
    "reference_pool_diagnostic": {
        "status": "complete_conditional_diagnostic",
        "promotion_approved": False,
        "ruling": "A pool-level price-behaviour signal, not portfolio alpha.",
        "unresolved_gates": ["strict PIT", "full corporate actions"],
    },
    "expert_router_delivery": {
        "model_adopted": False,
        "automatic_trading": False,
        "result": "router_improved_original_experts_but_did_not_pass_profitable_reference_and_risk_gates",
    },
    "expert_selection_delivery": {
        "status": "completed_conditional_research_no_promotion",
        "promotion_approved": False,
        "frozen_ledger": "6157c64",
        "candidate_decisions": {
            "A1": {
                "decision": "shadow_improvement_evidence_incomplete",
                "promotion_approved": False,
            },
            "A2": {
                "decision": "close_frozen_configuration_no_parameter_rescue",
                "promotion_approved": False,
            },
            "S": {
                "decision": "close_frozen_configuration_no_parameter_rescue",
                "promotion_approved": False,
            },
        },
    },
    "reference_challenge_delivery": {
        "status": "completed_conditional",
        "promotion_approved": False,
        "account_runs": 9,
    },
    "account_2025_delivery": {
        "status": "completed_conditional_accounts_and_evidence_review",
        "promotion_approved": False,
        "main_new_accounts_run": 1,
        "reference_new_accounts_run": 1,
        "strategy_changed": False,
    },
    "stable_book_decision": {
        "decision": "combination_did_not_reach_stable_15",
        "promotion_approved": False,
        "reasons": ["median_cagr", "phase0_drawdown"],
    },
    "continuous_band_decision": {
        "decision": "continuous_band_did_not_reach_15",
        "promotion_approved": False,
        "reasons": ["cagr"],
    },
    "extended_2018_band_decision": {
        "decision": "extended_2018_band_did_not_reach_15",
        "promotion_approved": False,
        "reasons": ["cagr"],
    },
    "full_pool_decision": {
        "decision": "full_pool_did_not_reach_15",
        "promotion_approved": False,
        "reasons": ["cagr", "drawdown"],
    },
}


class ResearchCatalogTests(unittest.TestCase):
    def write_allowlisted_artifacts(self, root, overrides=None, missing=()):
        payloads = dict(PAYLOADS)
        payloads.update(overrides or {})
        for spec in ALLOWLIST:
            if spec.artifact_id in missing:
                continue
            path = root / spec.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = payloads[spec.artifact_id]
            if spec.kind == "markdown":
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(json.dumps(payload), encoding="utf-8")

    def test_catalog_is_allowlisted_and_never_promotes_conditional_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_allowlisted_artifacts(root)
            catalog = load_research_catalog(root).to_dict()

        self.assertEqual(catalog["source_policy"]["mode"], "explicit_allowlist_only")
        self.assertEqual(
            {reference["id"] for reference in catalog["evidence_references"]},
            {spec.artifact_id for spec in ALLOWLIST},
        )
        self.assertTrue(all(
            "conditional" in reference["evidence_label"]
            or "provenance" in reference["evidence_label"]
            or "ledger" in reference["evidence_label"]
            for reference in catalog["evidence_references"]
        ))
        self.assertEqual(catalog["current_execution_state"]["observation"], "not_observed_by_catalog")
        self.assertFalse(catalog["adoption"]["promotion_approved"])
        self.assertEqual(catalog["adoption"]["status"], "not_adoptable")
        self.assertTrue(all(gate["outcome"] != "passed" for gate in catalog["adoption"]["gates"]))
        self.assertTrue(all(not candidate["promotion_approved"] for candidate in catalog["candidates"]))
        self.assertTrue(all(
            "conditional" in candidate["evidence_label"]
            or "unvalidated" in candidate["evidence_label"]
            or "unavailable" in candidate["evidence_label"]
            for candidate in catalog["candidates"]
        ))
        candidates = {candidate["id"]: candidate for candidate in catalog["candidates"]}
        self.assertEqual(candidates["quality_pb_hold90"]["metrics"]["net_cagr"], 0.031791)
        self.assertEqual(candidates["nonlinear_a1"]["recorded_decision"], "shadow_improvement_evidence_incomplete")
        self.assertTrue(all(question["evidence_label"] == "unverified_open_question"
                            for question in catalog["open_questions"]))

    def test_catalog_does_not_require_or_read_the_legacy_latest_run_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_allowlisted_artifacts(root)
            pointer = root / "output/runs/latest.json"
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(json.dumps({"status": "blocked_data", "promotion_approved": True}), encoding="utf-8")
            fake = root / "output/untrusted/DELIVERY.json"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.write_text(json.dumps({"promotion_approved": True}), encoding="utf-8")

            catalog = load_research_catalog(root).to_dict()

        self.assertNotIn("latest_run", {reference["id"] for reference in catalog["evidence_references"]})
        self.assertNotIn("untrusted", {reference["id"] for reference in catalog["evidence_references"]})
        self.assertEqual(catalog["current_execution_state"]["legacy_latest_run_pointer"], "intentionally_not_read")
        self.assertFalse(catalog["adoption"]["promotion_approved"])

    def test_missing_or_conflicting_inputs_degrade_to_unavailable_never_to_adoptable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conflict = dict(PAYLOADS["expert_selection_delivery"])
            conflict["promotion_approved"] = True
            conflict["candidate_decisions"] = {
                "A1": {"decision": "approved_by_untrusted_source", "promotion_approved": True}
            }
            self.write_allowlisted_artifacts(
                root,
                overrides={"expert_selection_delivery": conflict},
                missing=("cadence_comparison",),
            )
            catalog = load_research_catalog(root).to_dict()

        references = {reference["id"]: reference for reference in catalog["evidence_references"]}
        candidates = {candidate["id"]: candidate for candidate in catalog["candidates"]}
        self.assertEqual(references["cadence_comparison"]["availability"], "missing")
        self.assertEqual(candidates["quality_pb_hold90"]["evidence_label"], "unavailable_evidence_not_adoptable")
        self.assertFalse(candidates["nonlinear_a1"]["promotion_approved"])
        self.assertIn("expert_selection_delivery:source_claimed_promotion", catalog["integrity_warnings"])
        self.assertFalse(catalog["adoption"]["promotion_approved"])


if __name__ == "__main__":
    unittest.main()
