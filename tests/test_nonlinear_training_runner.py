"""Runner identity and recovery checks with fake models, no market training."""
import contextlib
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT/'scripts'))
SPEC = importlib.util.spec_from_file_location('nonlinear_runner_under_test', ROOT/'scripts/train_nonlinear_experts.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeTree:
    def model_to_string(self):
        return 'deterministic fake tree'


class FakeModel:
    fits = 0

    def __init__(self, kind, policy):
        self.models = [FakeTree() for _ in range(policy.num_models)]
        self.evidence = {'kind': kind, 'version': 'fixture'}

    def fit(self, frame, as_of):
        type(self).fits += 1
        self.evidence['as_of'] = str(as_of)
        return self

    def predict(self, frame, *, as_of):
        return pd.Series([.5, 1.], index=frame.index)


class NonlinearTrainingRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.source = self.out/'model_source.py'
        self.source.write_text('fixture source\n')
        day = '2020-01-31'
        self.day = day
        frame = pd.DataFrame({'instrument_id': ['a', 'b'], 'date': [day]*2})
        train = self.out/'training.parquet'
        scored = self.out/'scored.parquet'
        frame.to_parquet(train, index=False)
        frame.to_parquet(scored, index=False)
        entry = lambda p: dict(path=str(p), sha256=runner.shared.sha256(p))
        (self.out/'features_complete.json').write_text(json.dumps({'files': {day: entry(train)}, 'router_scores': entry(scored)}))
        (self.out/'random_diagnostic').mkdir()
        (self.out/'random_diagnostic/summary.json').write_text('{"completed": true}')
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, 'OUT', self.out))
        self.stack.enter_context(patch.object(runner, 'MODULE', self.source))
        self.stack.enter_context(patch.object(runner.nonlinear, 'MonthlyEnsemble', FakeModel))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        FakeModel.fits = 0

    def test_partial_publication_is_preserved_then_resumed_and_completed_is_idempotent(self):
        publish = runner.shared.publish

        def interrupt(path, writer, **kwargs):
            if Path(path).name == 'model1.txt':
                raise RuntimeError('simulated interruption')
            return publish(path, writer, **kwargs)

        with patch.object(runner.shared, 'publish', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                runner.train('plain')
        folder = self.out/'models/plain'
        partial = (folder/self.day/'predictions.parquet').read_bytes()
        self.assertFalse((folder/self.day/'complete.json').exists())
        runner.train('plain')
        attempts = list((folder/'incomplete_attempts').iterdir())
        self.assertEqual(len(attempts), 1)
        self.assertEqual((attempts[0]/'predictions.parquet').read_bytes(), partial)
        self.assertTrue((folder/'complete.json').exists())
        self.assertEqual(FakeModel.fits, 2)
        final = (folder/'complete.json').read_bytes()
        runner.train('plain')
        self.assertEqual(FakeModel.fits, 2)
        self.assertEqual((folder/'complete.json').read_bytes(), final)

    def test_prediction_input_corruption_is_rejected_before_training(self):
        with (self.out/'scored.parquet').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact_checksum_mismatch'):
            runner.train('plain')
        self.assertEqual(FakeModel.fits, 0)

    def test_completed_output_corruption_is_rejected_without_retraining(self):
        runner.train('plain')
        (self.out/'models/plain'/self.day/'model0.txt').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'artifact_checksum_mismatch'):
            runner.train('plain')
        self.assertEqual(FakeModel.fits, 1)

    def test_resume_rejects_changed_dependency_and_model_source(self):
        import lightgbm
        runner.train('plain')
        with patch.object(lightgbm, '__version__', 'fixture-different'):
            with self.assertRaisesRegex(ValueError, 'training_resume_identity_changed:lightgbm'):
                runner.train('plain')
        self.source.write_text('changed source\n')
        with self.assertRaisesRegex(ValueError, 'training_resume_identity_changed:source_sha256'):
            runner.train('plain')
        self.assertEqual(FakeModel.fits, 1)

    def test_same_kind_concurrent_run_is_rejected(self):
        folder = self.out/'models/plain'
        folder.mkdir(parents=True)
        with (folder/'training.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'training_already_running:plain'):
                runner.train('plain')
        self.assertEqual(FakeModel.fits, 0)


if __name__ == '__main__':
    unittest.main()
