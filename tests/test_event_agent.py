import json
import unittest

import numpy as np
import pandas as pd

from quantlab.agents.event_risk import evidence, parse_verdict, review

DAYS = pd.bdate_range('2026-06-01', '2026-09-22')
CODE = 'sz.000001'


def fields():
    close = pd.DataFrame({CODE: np.linspace(10, 12, len(DAYS))}, index=DAYS)
    return dict(adj_close=close, close=close, limit_down=close * .9, tradestatus=close * 0 + 1)


def tables():
    return dict(unlock=pd.DataFrame(dict(FREE_DATE=['2026-10-15', '2027-03-01'], SECURITY_CODE=['000001', '000001'],
                                         FREE_RATIO=[0.12, 0.30])),
                holder_change=pd.DataFrame(dict(NOTICE_DATE=['2026-09-01', '2026-10-01'], SECURITY_CODE=['000001'] * 2,
                                                DIRECTION=['减持', '减持'], CHANGE_FREE_RATIO=[0.5, 9.0])))


class EventAgentTests(unittest.TestCase):
    def test_evidence_uses_only_facts_known_at_asof(self):
        facts = evidence(CODE, '2026-09-22', tables(), fields())
        self.assertAlmostEqual(facts['unlock.next_60d_ratio'], 0.12)
        self.assertEqual(facts['unlock.next_date'], '2026-10-15')
        self.assertEqual(facts['insider.reduction_notices_90d'], 1)
        self.assertAlmostEqual(facts['insider.reduction_ratio_90d'], 0.5)
        self.assertEqual(facts['price.last_session'], '2026-09-22')

    def test_veto_must_cite_real_evidence(self):
        keys = {'unlock.next_60d_ratio'}
        ok, _ = parse_verdict('{"veto": true, "severity": 2, "reasons": ["unlock"], "cited_facts": ["[unlock.next_60d_ratio]"]}', keys)
        self.assertTrue(ok['veto'])
        bad, why = parse_verdict('{"veto": true, "severity": 2, "reasons": ["vibes"], "cited_facts": ["news.rumor"]}', keys)
        self.assertIsNone(bad)
        self.assertEqual(why, 'veto_without_evidence')
        self.assertEqual(parse_verdict('no json here', keys)[1], 'no_json')

    def test_review_never_emits_rating_or_order(self):
        replies = iter(['bull note', 'bear note', json.dumps(dict(veto=False, severity=0, reasons=[], cited_facts=[]))])
        out = review(CODE, '2026-09-22', {'unlock.next_60d_ratio': 0.0},
                     chat=lambda prompt, **kw: ('m', next(replies), 'sha'))
        self.assertFalse(out['abstained'])
        self.assertIsNone(out['rating'])
        self.assertIsNone(out['order'])


if __name__ == '__main__':
    unittest.main()
