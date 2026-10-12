import dataclasses
import unittest

from quantlab.p4_mandate import MANDATE, P4Mandate


class P4MandateTests(unittest.TestCase):
    def test_user_decision_values(self):
        self.assertEqual(MANDATE.capital_cny, 800_000.0)
        self.assertEqual(MANDATE.max_stocks, 50)
        self.assertEqual(MANDATE.max_drawdown, 0.25)
        self.assertIn('convertible_bond', MANDATE.excluded_assets)
        self.assertFalse(MANDATE.promotion_approved)

    def test_frozen_and_fingerprint_changes_with_values(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            MANDATE.capital_cny = 1.0
        self.assertNotEqual(MANDATE.fingerprint(), P4Mandate(capital_cny=600_000.0).fingerprint())
        self.assertEqual(MANDATE.fingerprint(), P4Mandate().fingerprint())


if __name__ == '__main__':
    unittest.main()
