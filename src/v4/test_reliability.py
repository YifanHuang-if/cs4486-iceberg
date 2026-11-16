"""Non-leakage and regression checks for angle reliability."""
import unittest
import numpy as np
from src.v4.reliability import gated_prior, gated_fuse, fold_gated_fusion

class ReliabilityTests(unittest.TestCase):
    def test_missing_query_keeps_original(self):
        ap, c = gated_prior([30., 31.], [0, 1], [np.nan])
        self.assertEqual(c[0], 0)
        self.assertAlmostEqual(gated_fuse([.73], ap, c)[0], .73)

    def test_holdout_labels_cannot_affect_own_prediction(self):
        angles = np.array([30., 30., 31., 31., 32., 32.])
        labels = np.array([0, 1, 0, 1, 0, 1])
        folds = np.array([0, 0, 1, 1, 2, 2])
        probabilities = np.array([.2, .7, .3, .6, .4, .5])
        a, _ = fold_gated_fusion(probabilities, angles, labels, folds)
        changed = labels.copy()
        changed[folds == 0] = 1 - changed[folds == 0]
        b, _ = fold_gated_fusion(probabilities, angles, changed, folds)
        np.testing.assert_array_equal(a[folds == 0], b[folds == 0])

    def test_zero_evidence_preserves_model(self):
        ap, c = gated_prior([29.], [1], [40.], sigma=.02)
        self.assertEqual(c[0], 0)
        self.assertAlmostEqual(gated_fuse([.23], ap, c)[0], .23)

    def test_stronger_exact_support_increases_gate(self):
        a1, c1 = gated_prior([30.], [1], [30.], tau=5)
        a2, c2 = gated_prior([30.]*10, [1]*10, [30.], tau=5)
        self.assertGreater(c2[0], c1[0])
        self.assertGreater(a2[0], a1[0])

    def test_tiny_reliability_is_continuous(self):
        model = np.array([.9, .1])
        angle = np.array([.2, .8])
        got = gated_fuse(model, angle, np.array([1e-10, 1e-10]))
        np.testing.assert_allclose(got, model, atol=1e-9, rtol=0)

    def test_gate_one_matches_v3_full_fusion(self):
        from src.v3.fusion import logit, sigmoid
        model = np.array([.7, .2])
        angle = np.array([.9, .3])
        got = gated_fuse(model, angle, np.ones(2))
        expected = sigmoid(.75 * logit(model) + logit(angle))
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_gate_interpolates_in_log_odds(self):
        model = np.array([.8])
        angle = np.array([.25])
        low = gated_fuse(model, angle, np.array([0.]))[0]
        mid = gated_fuse(model, angle, np.array([.5]))[0]
        high = gated_fuse(model, angle, np.array([1.]))[0]
        self.assertLess(high, mid)
        self.assertLess(mid, low)

    def test_exact_preservation_and_neighbor_gating(self):
        angles = np.array([30., 30., 30.4])
        labels = np.array([1, 1, 0])
        query = np.array([30., 30.01])
        prior, preserved = gated_prior(angles, labels, query, preserve_exact=True)
        normal_prior, default = gated_prior(angles, labels, query)
        self.assertAlmostEqual(preserved[0], 1.)
        self.assertLess(default[0], 1.)
        self.assertLess(preserved[1], 1.)
        np.testing.assert_allclose(prior, normal_prior)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            gated_prior([30.], [1], [30.], tau=0)
        with self.assertRaises(ValueError):
            gated_fuse([.2], [.3], [1.2])

if __name__ == "__main__":
    unittest.main()
