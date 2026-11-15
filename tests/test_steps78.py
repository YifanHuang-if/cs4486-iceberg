"""Guard baseline compatibility, leakage boundaries, ablations and deterministic resume."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.train_baseline import BaselineCNN
from src.experiments.data import Preprocessor, read_folds, fold_hash
from src.experiments.model import ExperimentCNN, augment_flips
from src.experiments.metrics import log_loss
from src.experiments.train import fit_fold

torch.set_num_threads(2)


class Steps78Tests(unittest.TestCase):
    def test_baseline_parameters_and_outputs_match(self):
        torch.manual_seed(42); reference = BaselineCNN()
        torch.manual_seed(42); candidate = ExperimentCNN()
        self.assertEqual(list(reference.state_dict()), list(candidate.state_dict()))
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]))
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 60433)
        x = torch.randn(4, 2, 75, 75)
        rng = torch.get_rng_state()
        a = reference(x); torch.set_rng_state(rng); b = candidate(x)
        self.assertTrue(torch.equal(a, b))

    def test_shared_image_initialization_is_preserved(self):
        torch.manual_seed(42); baseline = ExperimentCNN()
        torch.manual_seed(42); fusion = ExperimentCNN(angle_features=2)
        for name, tensor in fusion.state_dict().items():
            if name.startswith('network.'):
                self.assertTrue(torch.equal(tensor, baseline.state_dict()[name]))

    def test_training_only_preprocessing(self):
        images = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
        fitted = Preprocessor.fit(images, np.array([20., 40.]))
        self.assertEqual(fitted.channel_mean, [5.5, 9.5])
        x, meta = fitted.transform(np.ones((1, 2, 2, 2), dtype=np.float32)*1000, np.array([np.nan]))
        np.testing.assert_array_equal(meta, [[0., 1.]])
        self.assertEqual(fitted.angle_median, 30.)
        self.assertTrue(np.isfinite(x).all())

    def test_augmentation_alignment_and_rng_isolation(self):
        x = torch.arange(48, dtype=torch.float32).reshape(3, 1, 4, 4)
        paired = torch.cat([x, x+100], dim=1)
        before = torch.get_rng_state().clone()
        actual = augment_flips(paired, torch.Generator().manual_seed(13))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(actual[:, 1]-actual[:, 0], torch.full_like(x[:, 0], 100)))

    def test_missing_indicator_ablation(self):
        model = ExperimentCNN(angle_features=1).eval()
        x = torch.randn(3, 2, 75, 75); a = torch.randn(3, 2); b = a.clone(); b[:, 1] += 100
        self.assertTrue(torch.equal(model(x, a), model(x, b)))
        for features in [0, 1, 2]:
            model = ExperimentCNN(angle_features=features)
            logits = model(x, a)
            self.assertEqual(tuple(logits.shape), (3,))
            logits.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_probability_loss_and_weighted_pooling(self):
        self.assertAlmostEqual(log_loss([0, 1], [.5, .5]), np.log(2))
        self.assertAlmostEqual(log_loss([0, 1], [.1, .9]), -np.log(.9))
        y = np.array([0, 1, 0]); p = np.array([.1, .5, .6])
        self.assertAlmostEqual(log_loss(y, p), (log_loss(y[:1], p[:1])+2*log_loss(y[1:], p[1:]))/3)
        self.assertTrue(np.isfinite(log_loss([0, 1], [1, 0])))
        with self.assertRaises(ValueError): log_loss([0], [1.1])

    def test_fold_id_alignment_and_line_endings(self):
        ids = np.array([f'id_{i}' for i in range(10)])
        data = {'ids': ids, 'labels': np.array([0]*5+[1]*5), 'angles': np.arange(10.)}
        frame = pd.DataFrame({'id': ids, 'fold': np.arange(10)%5, 'label': data['labels'], 'angle_missing': 0})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'folds.csv'; frame.iloc[::-1].to_csv(path, index=False)
            self.assertEqual(read_folds(data, path).id.tolist(), ids.tolist())
            digest = fold_hash(path)
            path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
            self.assertEqual(fold_hash(path), digest)
            broken = frame.copy(); broken.loc[0, 'id'] = 'wrong'; broken.to_csv(path, index=False)
            with self.assertRaises(ValueError): read_folds(data, path)

    def test_interrupted_training_reproduces_continuous_run(self):
        rng = np.random.default_rng(7); n = 50
        data = {'ids': np.array([f'fixture_{i}' for i in range(n)]),
                'images': rng.normal(size=(n, 2, 75, 75)).astype(np.float32),
                'labels': (np.arange(n)%2).astype(np.float32), 'angles': rng.uniform(25, 45, n)}
        data['angles'][::7] = np.nan
        mapping = pd.DataFrame({'id': data['ids'], 'fold': np.arange(n)%5, 'label': data['labels'], 'angle_missing': (~np.isfinite(data['angles'])).astype(int)})
        config = {'name': 'synthetic_resume', 'seed': 42, 'batch_size': 8, 'max_epochs': 3, 'patience': 10,
                  'dropout': .25, 'learning_rate': .001, 'weight_decay': .0001, 'angle_features': 2, 'augmentation': True}
        signature = {'synthetic_test': True}
        with tempfile.TemporaryDirectory() as temporary:
            full = Path(temporary)/'full'; resumed = Path(temporary)/'resumed'; full.mkdir(); resumed.mkdir()
            fit_fold(data, mapping, config, 0, full, signature)
            self.assertIsNone(fit_fold(data, mapping, config, 0, resumed, signature, stop_after=1))
            fit_fold(data, mapping, config, 0, resumed, signature)
            a = torch.load(full/'fold_0.pt', weights_only=True); b = torch.load(resumed/'fold_0.pt', weights_only=True)
            for name, tensor in a['state_dict'].items(): self.assertTrue(torch.equal(tensor, b['state_dict'][name]))
            self.assertEqual((full/'oof_fold_0.csv').read_bytes(), (resumed/'oof_fold_0.csv').read_bytes())
            self.assertEqual((full/'history_fold_0.csv').read_bytes(), (resumed/'history_fold_0.csv').read_bytes())
            saved = pd.read_csv(full/'oof_fold_0.csv')
            measured = json.loads((full/'metrics_fold_0.json').read_text())
            self.assertAlmostEqual(log_loss(saved.label, saved.probability), measured['oof_log_loss'], places=12)


if __name__ == '__main__': unittest.main()
