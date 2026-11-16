"""Test label-free evidence groups against fixed fold definitions."""
import unittest
import numpy as np
from src.v4.groups import evidence_groups

class GroupTests(unittest.TestCase):
    def test_partition_and_fold_separation(self):
        angles = np.array([30.,30.,30.01,30.03,40.,np.nan,45.])
        folds = np.array([0,1,0,1,0,1,1])
        g = evidence_groups(angles, folds)
        self.assertTrue(g["exact_angle_match"][0])
        self.assertTrue(g["exact_angle_match"][1])
        self.assertTrue(g["near_angle_only"][2])
        self.assertTrue(g["near_angle_only"][3])
        self.assertTrue(g["missing_angle"][5])
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in g.values()), np.ones(7))
    def test_no_reference_is_far(self):
        g = evidence_groups([30.,31.,np.nan], [0,0,1])
        self.assertTrue(g["far_angle"][0] and g["far_angle"][1])
    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            evidence_groups([30.], [0], near_threshold=0)
        with self.assertRaises(ValueError):
            evidence_groups([30.,31.], [0])

if __name__ == "__main__":
    unittest.main()
