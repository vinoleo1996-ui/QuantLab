import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.factor_lab import proposer, referee


class RefereeTests(unittest.TestCase):
    def test_null_candidates_rarely_cross_the_admission_bar(self):
        rng = np.random.default_rng(0)
        admitted = 0
        for trial in range(200):
            stream = referee.ForwardStream(sign=1.0, submitted_at='2026-01-01')
            for t in range(250):
                stream.update(t, rng.normal(0.0, 0.10))
            admitted += bool(referee.online_ebh({'c': stream.wealth}, slots=1))
        self.assertLessEqual(admitted, 20)

    def test_true_factor_accumulates_wealth(self):
        rng = np.random.default_rng(1)
        stream = referee.ForwardStream(sign=1.0, submitted_at='2026-01-01')
        for t in range(500):
            stream.update(t, rng.normal(0.05, 0.10))
        self.assertGreater(stream.wealth, 1000)
        self.assertEqual(referee.online_ebh({'good': stream.wealth, 'bad': 0.5}, slots=10), {'good'})

    def test_admitted_factor_that_decays_is_retired(self):
        stream = referee.ForwardStream(sign=1.0, submitted_at='2026-01-01')
        streams = {'f': stream}
        for t in range(300):
            stream.update(t, 0.05)
            referee.admit(streams, t)
        self.assertIsNotNone(stream.admitted_at)
        for t in range(300, 500):
            stream.update(t, -0.05)
        self.assertIsNotNone(stream.retired_at)

    def test_screen_requires_holdout_agreement_and_low_library_corr(self):
        rng = np.random.default_rng(2)
        days = pd.bdate_range('2015-01-05', '2026-06-30')[::20]
        rows, labels, library = [], [], []
        for day in days:
            signal = rng.normal(size=300)
            label = signal * 0.3 + rng.normal(size=300)
            codes = [f'c{i}' for i in range(300)]
            rows.append(pd.DataFrame(dict(date=day, code=codes, value=signal)))
            labels.append(pd.DataFrame(dict(date=day, code=codes, label=label)))
            library.append(pd.DataFrame(dict(date=day, code=codes, other=rng.normal(size=300), same=signal)))
        values, labels = pd.concat(rows), pd.concat(labels)
        library = pd.concat(library)
        unique = referee.screen(values, labels, library.drop(columns='same'), set(days))
        self.assertTrue(unique['passed'])
        duplicate = referee.screen(values, labels, library, set(days))
        self.assertFalse(duplicate['passed'])
        self.assertEqual(duplicate['closest_feature'], 'same')


class ProposerTests(unittest.TestCase):
    def test_rotates_on_rate_limit_and_logs_every_attempt(self):
        replies = iter([{'error': {'code': 429}},
                        {'choices': [{'message': {'content': json.dumps([
                            {'name': 'rev', 'formula': 'neg(ts_sum(ret, 5))', 'rationale': 'reversal'},
                            {'name': 'bad', 'formula': 'delay(close, -3)', 'rationale': 'peek'}])}}]}])
        with tempfile.TemporaryDirectory() as tmp:
            usage = Path(tmp) / 'usage.jsonl'
            out = proposer.propose(['ret_20'], [], usage, n=2, post=lambda body, key: next(replies), key='SECRET-KEY')
            self.assertEqual(out['model'], proposer.MODELS[1])
            self.assertEqual([c['formula'] for c in out['candidates']], ['neg(ts_sum(ret, 5))'])
            self.assertEqual(len(out['invalid']), 1)
            lines = [json.loads(line) for line in usage.read_text().splitlines()]
            self.assertEqual([line['status'] for line in lines], ['error', 'ok'])
            self.assertNotIn('SECRET-KEY', usage.read_text())

    def test_daily_cap_is_enforced_from_the_usage_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            usage = Path(tmp) / 'usage.jsonl'
            now = pd.Timestamp.now(tz='UTC').isoformat()
            usage.write_text(''.join(json.dumps(dict(at=now, status='ok')) + '\n' for _ in range(proposer.DAILY_CAP)))
            with self.assertRaises(RuntimeError):
                proposer.propose([], [], usage, post=lambda body, key: {}, key='k')


if __name__ == '__main__':
    unittest.main()
