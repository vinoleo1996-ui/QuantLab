"""Account-profile changes must reach actual fills, not just report labels."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reference_challenge_replay as m
from tests.fixtures_v2 import dataset
import reference_challenge as study


class ReferenceChallengeTests(unittest.TestCase):
    def test_original_profile_preserves_entire_fills_including_identity_tags(self):
        from quantlab.reference_cash_replay import replay_reference
        ds=dataset(50,2)
        ds.panels={c.lower():p for c,p in ds.panels.items()}
        ds.instruments['instrument_id']=ds.instruments.instrument_id.str.lower()
        day=ds.calendar[25]
        targets={'targets':[{'signal_date':str(day.date()),'weights':{'s000':.45,'s001':.45}}]}
        original=replay_reference(ds,targets,start=day,end=ds.calendar[35])
        adapted=m.replay_reference(ds,targets,start=day,end=ds.calendar[35])
        self.assertEqual(original['account'],adapted['account'])
        self.assertEqual(original['fills'],adapted['fills'])

    def test_capital_and_cost_profile_change_actual_cash_shares_and_fill(self):
        ds = dataset(days=50, stocks=2)
        ds.panels = {c.lower(): p for c, p in ds.panels.items()}
        ds.instruments['instrument_id'] = ds.instruments.instrument_id.str.lower()
        day = ds.calendar[25]
        target = {'targets': [{'signal_date': str(day.date()), 'weights': {'S000': .45, 'S001': .45}}]}
        low = m.replay_reference(ds, target, start=day, end=ds.calendar[35], profile=m.ReplayProfile())
        common = m.replay_reference(ds, target, start=day, end=ds.calendar[35],
                                  profile=m.ReplayProfile.common(50))
        self.assertEqual(common['account'][0]['nav'], 600000.)
        self.assertEqual(low['account'][0]['nav'], 500000.)
        fill = common['fills'][0]
        price = ds.panels[fill['instrument_id']].loc[fill['date'], 'open']
        self.assertAlmostEqual(fill['price'], m.money(price * 1.003))
        self.assertAlmostEqual(fill['commission'], m.money(max(5., fill['notional'] * .00025)))
        self.assertGreater(common['fills'][0]['quantity'], low['fills'][0]['quantity'])
        self.assertFalse(common['promotion_approved'])
        self.assertLessEqual(max(len(s['quantities']) for s in common['account']), 50)

    def test_count_adaptation_preserves_absolute_retention_and_sector_slots(self):
        codes=[f'{i:06d}.SZ' for i in range(200)]
        inputs={'heads':{'2023-01-31':dict.fromkeys(codes,1.),
                         '2023-02-28':{c:float(i>=100) for i,c in enumerate(codes)}},
                'industries':{d:{c:str(i%10) for i,c in enumerate(codes)}
                              for d in ('2023-01-31','2023-02-28')}}
        targets=study.build_targets(inputs,['2023-01-31','2023-02-28'],50)
        self.assertEqual(len(targets[0]['weights']),50)
        # Initially selected names now rank 101..150, so all must be retained.
        self.assertEqual(targets[0]['weights'],targets[1]['weights'])
        for groups in inputs['industries'].values():
            groups.update({c:'one_industry' for c in codes[:40]})
        target=study.build_targets(inputs,['2023-01-31','2023-02-28'],50)[0]
        self.assertEqual(sum(c in codes[:40] for c in target['weights']),15)

    def test_future_score_append_does_not_change_past_targets(self):
        codes=[f'{i:06d}.SZ' for i in range(160)]
        inputs={'heads':{'2023-01-31':dict.fromkeys(codes,1.)},
                'industries':{'2023-01-31':{c:str(i%15) for i,c in enumerate(codes)}}}
        before=study.build_targets(inputs,['2023-01-31'],50)
        inputs['heads']['2023-02-28']={c:float(i%2) for i,c in enumerate(codes)}
        inputs['industries']['2023-02-28']=dict.fromkeys(codes,'changed')
        after=study.build_targets(inputs,['2023-01-31','2023-02-28'],50)
        self.assertEqual(before,after[:1])

    def test_position_cap_rejects_invalid_target_and_no_silent_profile_override(self):
        with self.assertRaises(ValueError): m.ReplayProfile.common(60)
        with self.assertRaises(ValueError): m.ReplayProfile(initial_cash=-1.)
        with self.assertRaises(ValueError):
            m.replay_reference(dataset(50,2),{'targets':[{'signal_date':'2023-02-06',
                'weights':{f'S{i:03}':.9/51 for i in range(51)}}]},
                start='2023-02-06',end='2023-02-07',profile=m.ReplayProfile.common(50))


if __name__ == '__main__':
    unittest.main()
