import copy
import unittest
from unittest.mock import patch

import quantlab.webapp.app as webapp


class _CatalogStub:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return copy.deepcopy(self._payload)


def _catalog_payload():
    return {
        "source_policy": {
            "mode": "explicit_allowlist_only",
            "allowlisted_paths": (
                "docs/RESEARCH_CONTINUITY.md",
                "output/continuity_review_20260927/cadence13/comparison_with13/comparison.json",
            ),
            "excluded_inputs": ("output/runs/**",),
        },
        "current_execution_state": {
            "observation": "not_observed_by_catalog",
            "legacy_latest_run_pointer": "intentionally_not_read",
        },
        "adoption": {
            "status": "not_adoptable",
            "promotion_approved": False,
            "decision": "No allowlisted artifact grants strategy adoption.",
            "gates": (
                {
                    "id": "net_performance_target",
                    "label": "扣费后长期年化 15%",
                    "outcome": "blocked",
                    "source_artifacts": ("cadence_comparison",),
                },
            ),
        },
        "candidates": (
            {
                "id": "quality_pb_hold90",
                "label": "quality/PB 月度 + 最低持有 90 日",
                "recorded_status": "unverified_data_gaps",
                "promotion_approved": False,
                "adoptable": False,
                "source_artifact": "cadence_comparison",
                "source_path": "output/continuity_review_20260927/cadence13/comparison_with13/comparison.json",
            },
        ),
        "evidence_references": (
            {
                "id": "continuity_ledger",
                "label": "研究接续台账",
                "path": "docs/RESEARCH_CONTINUITY.md",
                "availability": "available",
                "integrity": "current_ledger_text",
            },
        ),
        "open_questions": (
            {
                "id": "data_acceptance",
                "question": "能否补齐严格 PIT 数据验收？",
                "evidence_label": "unverified_open_question",
                "source_artifacts": ("continuity_ledger",),
            },
        ),
        "integrity_warnings": (),
    }


class ResearchHubTests(unittest.TestCase):
    def setUp(self):
        webapp.app.config.update(TESTING=True)
        self.client = webapp.app.test_client()

    def test_dashboard_and_research_have_separate_shells(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('/static/dashboard.css', html)
        self.assertIn('/static/dashboard.js', html)
        self.assertIn('/static/echarts.min.js', html)
        self.assertIn('id="dashboard-main"', html)
        self.assertIn('id="dashboardStatus"', html)
        self.assertIn('id="overview-tab"', html)
        self.assertIn('href="/portfolio"', html)
        self.assertIn('href="/research"', html)
        self.assertNotIn('data-research-hub', html)

        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn('/static/dashboard.js', response.get_data(as_text=True))

        response = self.client.get("/research")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('data-research-hub', html)
        self.assertIn('/static/research_hub.css', html)
        self.assertIn('/static/research_hub.js', html)
        self.assertNotIn('dashboard.js', html)
        self.assertNotIn('echarts.min.js', html)

    def test_research_endpoints_use_a_consistent_read_only_envelope(self):
        expected = {
            "/api/research/decision": "not_adoptable",
            "/api/research/candidates": "not_adoptable",
            "/api/research/evidence": "unverified",
            "/api/research/open-questions": "pending",
        }
        with patch.object(webapp, "load_research_catalog", return_value=_CatalogStub(_catalog_payload())):
            for path, status in expected.items():
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, path)
                payload = response.get_json()
                self.assertEqual(payload["status"], status)
                self.assertEqual(set(payload), {"value", "status", "provenance", "as_of", "caveats", "drilldown"})
                self.assertIsInstance(payload["provenance"], list)
                self.assertIsInstance(payload["caveats"], list)

    def test_legacy_holdings_endpoint_never_promotes_weight_history_to_holdings(self):
        with patch.object(webapp, "snapshot", return_value=[]), \
             patch.object(webapp, "summary", return_value={"status": "blocked_data"}), \
             patch.object(webapp, "ytd_return", return_value=None):
            response = self.client.get("/api/holdings")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["account_status"], "blocked_data")
        self.assertEqual(payload["status"], "blocked_data")
        self.assertFalse(payload["legacy_snapshot"])
        self.assertEqual(payload["rows"], [])
        self.assertTrue(any("历史回放权重" in item for item in payload["caveats"]))

    def test_decision_fails_closed_and_never_exposes_legacy_run_state_as_holdings(self):
        catalog = _catalog_payload()
        catalog["adoption"]["status"] = "adopted"
        catalog["adoption"]["promotion_approved"] = True
        with patch.object(webapp, "load_research_catalog", return_value=_CatalogStub(catalog)):
            response = self.client.get("/api/research/decision")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "not_adoptable")
        self.assertFalse(payload["value"]["promotion_approved"])
        self.assertNotIn("holdings", payload["value"])
        self.assertNotIn("signals", payload["value"])
        self.assertNotIn("latest.json", response.get_data(as_text=True))
        self.assertEqual(
            payload["drilldown"]["current_execution_state"]["legacy_latest_run_pointer"],
            "intentionally_not_read",
        )


if __name__ == "__main__":
    unittest.main()
