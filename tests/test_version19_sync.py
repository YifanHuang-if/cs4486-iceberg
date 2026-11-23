"""Small synthetic checks for the imported version19 source changes.

These tests verify CLI/coefficient behavior, not full training or CSV replay.
"""
import ast
import contextlib
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from work import v5_submit
from src.v3 import train


class SubmissionScalingTest(unittest.TestCase):
    def run_fixture(self, options, expected_weights):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / 'work/predictions'
            cache.mkdir(parents=True)
            (root / 'work/cache').mkdir()
            (root / 'work/submissions').mkdir()
            (root / 'data/raw').mkdir(parents=True)
            ids = np.array(['a', 'b', 'c', 'd'])
            angles = np.array([30., 31., 32., 33.])
            np.savez(root / 'work/cache/test.npz', ids=ids, angles=angles)
            np.savez(root / 'work/cache/train.npz', angles=angles)
            image = np.array([.2, .4, .6, .8])
            for fold in range(5):
                np.save(cache / f'fixture_fold{fold}.npy', image)
            with (root / 'data/raw/sample_submission.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'is_iceberg'])
                writer.writerows((i, 0.5) for i in ids[::-1])
            prior = np.array([.7, .3, .8, .2])
            exact = np.array([True, False, True, False])
            labels = np.array([0., 0., 1., 1.])
            fitted = np.array([.8, 1.2, .6, -.1])
            original = fitted.copy()
            with patch.object(v5_submit, 'ROOT', root), \
                 patch.object(v5_submit, 'CACHE', cache), \
                 patch.object(v5_submit, 'training_features', return_value=(labels, np.ones(4, dtype=bool), np.zeros((4, 3)), {})), \
                 patch.object(v5_submit, 'fit_logistic', return_value=fitted), \
                 patch.object(v5_submit, 'kernel_prior', return_value=(prior, np.ones(4, dtype=bool), exact)), \
                 patch('sys.argv', ['v5_submit', '--tags', 'fixture', '--name', 'fixture', *options]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(v5_submit.main(), 0)
            np.testing.assert_array_equal(fitted, original)
            with (root / 'work/submissions/fixture.csv').open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r['id'] for r in rows], list(ids[::-1]))
            image_logit = np.log(image / (1 - image))
            prior_logit = np.log(prior / (1 - prior))
            w = np.asarray(expected_weights)
            logits = w[0] * image_logit + np.where(exact, w[1], w[2]) * prior_logit + w[3]
            expected = np.clip(1 / (1 + np.exp(-logits)), .001, .999)
            actual = np.array([float(r['is_iceberg']) for r in rows])
            np.testing.assert_allclose(actual, expected[::-1], atol=5.1e-11, rtol=0)
            manifest = json.loads((root / 'work/submissions/fixture.json').read_text())
            np.testing.assert_allclose(manifest['coefficients'], expected_weights, atol=1e-12)

    def test_default_preserves_unscaled_coefficients(self):
        self.run_fixture([], [.8, 1.2, .6, -.1])

    def test_version19_scales_both_angle_terms(self):
        self.run_fixture(['--angle-scale', '0.90'], [.8, 1.08, .54, -.1])

    def test_exact_override_only_changes_exact_term(self):
        self.run_fixture(['--angle-scale', '.90', '--exact-scale', '.80'], [.8, .96, .54, -.1])


class WarmupCompatibilityTest(unittest.TestCase):
    def test_default_schedule_matches_previous_cosine_schedule(self):
        parsed = ast.parse(Path(train.__file__).read_text(encoding='utf-8'))
        nodes = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'main_scheduler' for t in node.targets):
                nodes.append(node)
            elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Attribute) and node.test.left.attr == 'warmup':
                nodes.append(node)
        self.assertEqual(len(nodes), 2)
        nodes.sort(key=lambda n: n.lineno)
        new_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=.003)
        old_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=.003)
        scope = {'torch': torch, 'opt': new_opt, 'args': SimpleNamespace(epochs=20, warmup=0, lr=.003)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<imported scheduler>', 'exec'), scope)
        old_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_opt, T_max=20, eta_min=.003 * .01)
        for _ in range(21):
            self.assertEqual(new_opt.param_groups[0]['lr'], old_opt.param_groups[0]['lr'])
            new_opt.step()
            old_opt.step()
            scope['scheduler'].step()
            old_scheduler.step()


if __name__ == '__main__':
    unittest.main()
