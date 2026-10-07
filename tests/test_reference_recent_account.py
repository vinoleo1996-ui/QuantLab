import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reference_recent_account',ROOT/'scripts/reference_recent_account.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ReferenceRecentTests(unittest.TestCase):
    def test_fresh_account_uses_only_latest_prior_signal_for_initial_order(self):
        codes=[f'{i:06d}.SZ' for i in range(150)]
        heads={'2024-11-01':{s:float(i<50) for i,s in enumerate(codes)},
               '2024-12-11':{s:float(i>=50) for i,s in enumerate(codes)},
               '2025-01-09':dict.fromkeys(codes,1.)}
        industries={d:{s:str(i%20) for i,s in enumerate(codes)} for d in heads}
        sessions=['2024-12-31','2025-01-02','2025-01-09']
        targets=m.recent_schedule(heads,industries,sessions)
        self.assertEqual([t['signal_date'] for t in targets],['2024-12-31','2025-01-09'])
        self.assertEqual(set(targets[0]['weights']),set(codes[50:]))
        heads['2024-11-01']=dict.fromkeys(codes,1.)
        heads['2025-01-09']=dict.fromkeys(codes,0.)
        self.assertEqual(m.recent_schedule(heads,industries,sessions)[0],targets[0])


if __name__=='__main__':unittest.main()
