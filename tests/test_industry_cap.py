import unittest
from collections import Counter

from quantlab.alpha.portfolio import (INDUSTRY_MAX_WEIGHT, industry_cap_names, select,
                                      select_hysteresis)


class IndustryCapTest(unittest.TestCase):
    def test_cap_follows_mandate_weight(self):
        # 50 equal-weight names: floor(0.25 * 50) = 12, unchanged from the old fixed count.
        self.assertEqual(industry_cap_names(50), 12)
        # 40 names: the old fixed 12 would be 30% of the book; the mandate says 25%.
        self.assertEqual(industry_cap_names(40), 10)
        self.assertEqual(industry_cap_names(30), 7)
        self.assertEqual(industry_cap_names(4), 1)

    def test_select_never_breaks_25pct_at_40_names(self):
        # eight industries, 25 names each: the book can fill while every industry stays <= 25%
        ordered = [f'c{i}' for i in range(200)]
        industries = {f'c{i}': f'ind{i % 8}' for i in range(200)}
        chosen = select(ordered, industries, [], holdings=40)
        counts = Counter(industries[c] for c in chosen)
        self.assertEqual(len(chosen), 40)
        for industry, count in counts.items():
            self.assertLessEqual(count / len(chosen), INDUSTRY_MAX_WEIGHT + 1e-9,
                                 f'{industry} weighs {count / len(chosen):.2%}')

    def test_hysteresis_never_breaks_25pct_at_40_names(self):
        ordered = [f'c{i}' for i in range(200)]
        industries = {f'c{i}': f'ind{i % 8}' for i in range(200)}
        # entry_frac 10% of a 200-name pool caps entries at 20 names, so hold 20 here
        chosen = select_hysteresis(ordered, industries, [], holdings=20)
        counts = Counter(industries[c] for c in chosen)
        self.assertEqual(len(chosen), 20)
        for industry, count in counts.items():
            self.assertLessEqual(count / len(chosen), INDUSTRY_MAX_WEIGHT + 1e-9)

    def test_two_industry_pool_underfills_rather_than_breaking_the_mandate(self):
        # 2 industries x cap 10 can only reach 20 names at 40 holdings: the mandate wins,
        # the book stays short (same convention as select_domain_book's empty slots).
        ordered = [f'c{i}' for i in range(200)]
        industries = {f'c{i}': 'bank' if i < 100 else 'tech' for i in range(200)}
        chosen = select(ordered, industries, [], holdings=40)
        self.assertEqual(len(chosen), 20)

    def test_explicit_cap_still_wins(self):
        ordered = [f'c{i}' for i in range(100)]
        industries = {c: 'bank' for c in ordered}
        self.assertEqual(len(select(ordered, industries, [], holdings=40, cap=3)), 3)

    def test_universe_select_uses_the_same_mandate_derived_cap(self):
        # quantlab/smallcap/universe.py used a hard-coded 15 names, which is 37.5% of a 40-name book.
        from quantlab.smallcap import universe
        ordered = [f'c{i}' for i in range(200)]
        # ranked so one industry dominates the head of the list: the cap has to bind
        industries = {f'c{i}': f'ind{i // 50}' for i in range(200)}
        chosen = universe.select(ordered, industries, [], holdings=40)
        self.assertEqual(len(chosen), 40)
        counts = Counter(industries[c] for c in chosen)
        for industry, count in counts.items():
            self.assertLessEqual(count / len(chosen), INDUSTRY_MAX_WEIGHT + 1e-9,
                                 f'{industry} weighs {count / len(chosen):.2%}')

    def test_both_selection_paths_derive_the_same_slot_count(self):
        from quantlab.smallcap import universe
        for holdings in (50, 40, 30, 4):
            self.assertEqual(universe.industry_slots(holdings), industry_cap_names(holdings))


if __name__ == '__main__':
    unittest.main()
