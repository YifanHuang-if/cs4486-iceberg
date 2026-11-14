"""Small synthetic fixtures exercise data-quality failures, not model scores."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.eda import check_submission, scan_json


def record(sample_id, label=0, angle=38.0, pixel=1.0):
    return {"id": sample_id, "band_1": [pixel] * 5625, "band_2": [pixel + 1] * 5625,
            "inc_angle": angle, "is_iceberg": label}


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def scan(self, rows):
        path = self.root / "train.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return scan_json(path, labeled=True)

    def test_missingness_and_duplicate_images_are_reported(self):
        result = self.scan([record("a", angle="na"), record("b", label=1), record("c", pixel=3)])
        self.assertEqual(result["summary"]["missing_angle_label_counts"], {"0": 1})
        self.assertEqual(result["summary"]["exact_duplicate_image_excess_rows"], 1)
        self.assertEqual(result["summary"]["mixed_class_repeated_angle_groups"], 1)

    def test_bad_band_length_rejected(self):
        row = record("a")
        row["band_2"].pop()
        with self.assertRaisesRegex(ValueError, "5625"):
            self.scan([row])

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            self.scan([record("a"), record("a", pixel=2)])

    def test_invalid_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "label must"):
            self.scan([record("a", label=2)])

    def test_invalid_angle_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid inc_angle"):
            self.scan([record("a", angle="unknown")])

    def test_nonfinite_angle_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite inc_angle"):
            self.scan([record("a", angle="inf")])

    def test_submission_order_is_distinguished_from_id_membership(self):
        path = self.root / "sample.csv"
        pd.DataFrame({"id": ["b", "a"], "is_iceberg": [.5, .5]}).to_csv(path, index=False)
        result = check_submission(path, ["a", "b"])
        self.assertTrue(result["test_id_set_match"])
        self.assertFalse(result["test_id_order_match"])

    def test_submission_id_mismatch_rejected(self):
        path = self.root / "sample.csv"
        pd.DataFrame({"id": ["a", "c"], "is_iceberg": [.5, .5]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            check_submission(path, ["a", "b"])

    def test_out_of_range_probability_rejected(self):
        path = self.root / "sample.csv"
        pd.DataFrame({"id": ["a"], "is_iceberg": [1.2]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "probabilities"):
            check_submission(path, ["a"])


if __name__ == "__main__":
    unittest.main()
