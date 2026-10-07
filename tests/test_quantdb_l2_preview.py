"""The L2 preview check stays closed when time or receipt identity is absent."""

from __future__ import annotations

import unittest

from quantlab.data.l2_contract import L2_FEATURE_KEYS
from quantlab.data.quantdb_l2_preview import assess_l2_preview, preview_column_names


class QuantDBL2PreviewTests(unittest.TestCase):
    def test_named_columns_without_time_or_receipt_are_not_qualified(self):
        result = assess_l2_preview(columns=L2_FEATURE_KEYS, preview_response_sha256="ab" * 32)
        self.assertEqual(result["status"], "not_qualified")
        self.assertFalse(result["training_allowed"])
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["missing_fields"], ())
        self.assertIn("available_at_not_declared", result["reasons"])
        self.assertIn("per_field_raw_receipt_missing", result["reasons"])

    def test_missing_keys_are_listed(self):
        result = assess_l2_preview(columns=("micro_pin", "available_at"))
        self.assertEqual(result["present_fields"], ("micro_pin",))
        self.assertIn("l2_fields_missing", result["reasons"])
        self.assertNotIn("available_at_not_declared", result["reasons"])

    def test_column_names_come_from_the_header_not_the_rows(self):
        names = preview_column_names({"columns": ["micro_pin"], "data": [{"secret": 1}]})
        self.assertEqual(names, ("micro_pin",))


if __name__ == "__main__":
    unittest.main()
