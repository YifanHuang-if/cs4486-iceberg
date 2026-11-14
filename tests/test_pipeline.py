"""Checks for leakage boundaries, model inputs and probability evaluation."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import Preprocessor, aligned_folds, fold_hash
from src.evaluate import log_loss
from src.models.cnn import IcebergCNN, augment_flips


class PipelineTests(unittest.TestCase):
    def test_preprocessing_is_fitted_on_training_only(self):
        train = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
        angles = np.array([30., np.nan])
        p = Preprocessor.fit(train, angles)
        self.assertEqual(p.angle_median, 30.)
        self.assertTrue(np.allclose(p.channel_mean, [5.5, 9.5]))
        x, m = p.transform(np.full((1, 2, 2, 2), 100., dtype=np.float32), np.array([np.nan]))
        self.assertGreater(x.mean(), 10.)
        np.testing.assert_allclose(m, [[0., 1.]])
        self.assertEqual(p.angle_median, 30.)

    def test_probabilities_and_known_log_loss(self):
        self.assertAlmostEqual(log_loss([0, 1], [.5, .5]), np.log(2))
        self.assertAlmostEqual(log_loss([0, 1], [.1, .9]), -np.log(.9))
        self.assertTrue(np.isfinite(log_loss([1, 0], [0., 1.])))
        with self.assertRaises(ValueError): log_loss([0], [1.2])
        with self.assertRaises(ValueError): log_loss([0], [np.nan])

    def test_pooled_score_is_weighted_not_unweighted_fold_mean(self):
        a = log_loss([0], [.1]); b = log_loss([1, 1, 0], [.7, .8, .3])
        pooled = log_loss([0, 1, 1, 0], [.1, .7, .8, .3])
        self.assertAlmostEqual(pooled, (a + 3*b)/4)
        self.assertNotAlmostEqual(pooled, (a+b)/2)

    def test_models_return_logits_and_backward_is_finite(self):
        torch.set_num_threads(2)
        for inputs in [0, 1, 2]:
            model = IcebergCNN(inputs, .3)
            logits = model(torch.randn(4, 2, 75, 75), torch.randn(4, 2))
            self.assertEqual(tuple(logits.shape), (4,))
            loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0.,1.,0.,1.]))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_augmentation_keeps_bands_aligned(self):
        first = torch.arange(16.).reshape(1, 1, 4, 4).repeat(8, 1, 1, 1)
        x = torch.cat((first, first + 100), dim=1)
        out = augment_flips(x, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(out[:,1] - out[:,0], torch.full((8,4,4),100.)))
        self.assertTrue(torch.equal(torch.sort(out[:,0].flatten(1)).values,
                                    torch.sort(first[:,0].flatten(1)).values))

    def test_mask_ablation_really_ignores_mask(self):
        model = IcebergCNN(1).eval()
        image = torch.randn(2,2,75,75)
        m = torch.tensor([[.1,0.],[.2,0.]])
        changed = m.clone(); changed[:,1] = 1
        with torch.no_grad():
            self.assertTrue(torch.equal(model(image,m), model(image,changed)))

    def test_fold_alignment_uses_ids_not_row_order(self):
        ids = np.array([f'id{i}' for i in range(10)])
        y = np.tile([0,1], 5)
        table = pd.DataFrame({'id':ids,'fold':np.repeat(np.arange(5),2),'label':y})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'folds.csv'
            table.iloc[::-1].to_csv(path,index=False)
            folds = aligned_folds({'ids':ids,'labels':y},path)
            np.testing.assert_array_equal(folds,np.repeat(np.arange(5),2))
            table.loc[0,'id']='wrong';table.to_csv(path,index=False)
            with self.assertRaises(ValueError):aligned_folds({'ids':ids,'labels':y},path)

    def test_fold_hash_is_cross_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a',Path(directory)/'b'
            a.write_bytes(b'id,fold\na,0\n');b.write_bytes(b'id,fold\r\na,0\r\n')
            self.assertEqual(fold_hash(a),fold_hash(b))


if __name__ == '__main__': unittest.main()
