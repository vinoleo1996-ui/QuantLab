import unittest

from quantlab.data.public_desk import (
    observe_public_industry,
    parse_industry_boards,
    parse_suggest,
    twenty_day_return,
)


def _closes(code: str) -> list[tuple[str, float]]:
    base = {"600001": 1.0, "600002": 2.0, "600003": 0.5}[code]
    return [(f"2026-01-{index + 1:02d}", base + index * 0.01) for index in range(21)]


class _Desk:
    def boards(self, secid: str) -> dict:
        return {"data": {"diff": [
            {"f12": "BK0002", "f14": "宽行业", "f19": 2, "f3": 1.5, "f128": "宽龙头", "f140": "600100"},
            {"f12": "BK0001", "f14": "窄行业", "f19": 2, "f3": -0.4, "f128": "丙", "f140": "600003"},
            {"f12": "BK0009", "f14": "概念", "f19": 3, "f3": 9.9},
        ]}}

    def members(self, board: str, page: int, size: int) -> dict:
        if board == "0002":
            rows = [{"SECURITY_CODE": f"60010{index}", "SECURITY_NAME_ABBR": f"宽{index}"} for index in range(5)]
            count = 5
        else:
            rows = [
                {"SECURITY_CODE": "600001", "SECURITY_NAME_ABBR": "甲"},
                {"SECURITY_CODE": "600002", "SECURITY_NAME_ABBR": "乙"},
                {"SECURITY_CODE": "600003", "SECURITY_NAME_ABBR": "丙"},
            ]
            count = 3
        shown = rows[:1] if size == 1 else rows
        return {"result": {"count": count, "data": shown if page == 1 else []}}

    def daily_closes(self, code: str):
        return _closes(code)


class PublicDeskTests(unittest.TestCase):
    def test_suggest_keeps_a_shares_and_drops_funds(self):
        rows = parse_suggest({"QuotationCodeTable": {"Data": [
            {"Code": "600519", "Name": "贵州茅台", "PinYin": "GZMT", "Classify": "AStock", "SecurityTypeName": "沪A"},
            {"Code": "510300", "Name": "沪深300ETF", "Classify": "Fund", "SecurityTypeName": "基金"},
        ]}})

        self.assertEqual(rows, [{"code": "600519", "name": "贵州茅台", "pinyin": "GZMT", "market": "沪A"}])

    def test_industry_parser_ignores_concepts(self):
        boards = parse_industry_boards({"data": {"diff": [
            {"f12": "BK1532", "f14": "金属包装", "f19": 2},
            {"f12": "BK0896", "f14": "白酒", "f19": 3},
        ]}})

        self.assertEqual([board["name"] for board in boards], ["金属包装"])

    def test_twenty_day_return_needs_twenty_one_closes(self):
        closes = [(f"2026-01-{index + 1:02d}", 10.0 + index) for index in range(21)]

        self.assertIsNone(twenty_day_return(closes[:20]))
        value, day = twenty_day_return(closes)
        self.assertAlmostEqual(value, 30.0 / 10.0 - 1.0)
        self.assertEqual(day, "2026-01-21")

    def test_observation_ranks_the_narrow_industry_not_the_concept(self):
        payload = observe_public_industry("600001", client=_Desk())

        self.assertEqual(payload["sector"]["name"], "窄行业")
        self.assertEqual(payload["sector"]["constituent_count"], 3)
        self.assertEqual([row["name"] for row in payload["industry_top3_1m"]["top"]], ["丙", "甲", "乙"])
        self.assertAlmostEqual(payload["industry_top3_1m"]["top"][0]["return_20d"], 0.40)
        context = payload["context"]
        self.assertEqual(context["focus"]["name"], "窄行业")
        self.assertAlmostEqual(context["focus"]["today_pct"], -0.004)
        self.assertEqual(context["focus"]["today_leader"], {"code": "600003", "name": "丙"})
        self.assertEqual([item["name"] for item in context["chain"]], ["宽行业"])
        self.assertEqual(context["breadth"], {"up": 3, "down": 0, "flat": 0, "total": 3})
        self.assertEqual(context["selected_rank_20d"], 2)
        self.assertEqual(context["selected_count"], 3)
        self.assertLess(context["versus_board"], 0)

    def test_missing_history_keeps_the_sector_and_withholds_the_ranking(self):
        class _Short(_Desk):
            def daily_closes(self, code: str):
                return _closes(code)[:5] if code == "600002" else _closes(code)

        payload = observe_public_industry("600001", client=_Short())

        self.assertEqual(payload["sector"]["name"], "窄行业")
        self.assertFalse(payload["industry_top3_1m"]["available"])
        self.assertEqual(payload["industry_top3_1m"]["top"], [])
        self.assertIn("不发布 Top 3", payload["industry_top3_1m"]["message"])


if __name__ == "__main__":
    unittest.main()
