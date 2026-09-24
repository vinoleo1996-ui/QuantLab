import unittest
import pandas as pd
from scripts.collect_corporate_actions import normalize_matches


class CorporateActionMatching(unittest.TestCase):
    def setUp(self):
        self.calendar=pd.bdate_range('2024-01-01',periods=80)
        self.east=[dict(raw=dict(SECUCODE='600001.SH',ASSIGN_PROGRESS='实施分配',EX_DIVIDEND_DATE='2024-02-05',
            EQUITY_RECORD_DATE='2024-02-02',PRETAX_BONUS_RMB=3.,BONUS_RATIO=1.,IT_RATIO=2.,
            NOTICE_DATE='2024-01-29',D30_CLOSE_ADJCHRATE=1000),source_hash='em1',fetched_at='2026-09-24')]
        self.record=dict(F020D='2024-02-05',F018D='2024-02-02',F012N=3.,F010N=1.,F011N=2.,
                         F023D='2024-02-06',F025D='2024-02-07',F006D='2024-01-29')

    def run_match(self,record):
        cn=[('sh.600001',dict(records=[record]),dict(sha256='cn1',fetched_at='2026-09-24'))]
        return normalize_matches(self.east,cn,self.calendar,'2024-01-01','2024-03-01')

    def test_per_ten_units_and_real_payment_dates_are_preserved(self):
        actions,issues=self.run_match(self.record)
        self.assertEqual(issues,[]);self.assertEqual(len(actions),1)
        action=actions[0]
        self.assertAlmostEqual(action['cash_per_share'],.3)
        self.assertAlmostEqual(action['share_ratio'],.3)
        self.assertEqual(action['pay_date'],'2024-02-06')
        self.assertEqual(action['share_date'],'2024-02-07')
        self.assertEqual(action['available_at'],'2024-01-30 00:00:00')
        self.assertNotIn('D30_CLOSE_ADJCHRATE',action)

    def test_missing_payment_never_becomes_an_assumed_ex_date_payment(self):
        record=dict(self.record,F023D=None)
        actions,issues=self.run_match(record)
        self.assertEqual(actions,[])
        self.assertEqual(issues[0]['reason'],'distribution_payment_or_share_date_missing')

    def test_source_disagreement_is_not_silently_resolved(self):
        actions,issues=self.run_match(dict(self.record,F012N=30.))
        self.assertEqual(actions,[])
        self.assertEqual(issues[0]['reason'],'distribution_cross_source_missing_or_conflicting')

    def test_two_distributions_on_the_same_day_are_not_replaced_by_the_first(self):
        duplicate=dict(self.east[0]);duplicate['raw']=dict(duplicate['raw'],PRETAX_BONUS_RMB=6.)
        self.east.append(duplicate)
        actions,issues=self.run_match(self.record)
        self.assertEqual(actions,[])
        self.assertEqual(issues[0]['reason'],'multiple_distribution_records_same_day_not_reconciled')


if __name__=='__main__':unittest.main()
