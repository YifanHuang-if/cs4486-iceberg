"""Per-tag pooled OOF summary plus final submission validation."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.fusion import fold_fusion, loss
from src.v3.train import ROOT

TAGS = ["s101", "s202", "s404", "stu1", "stu2", "v5_shape", "d50a", "d50b", "g50"]


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    labels = data["labels"].astype(np.float64)
    angles = data["angles"].astype(np.float64)
    folds = read_folds(data["ids"], data["labels"])["fold"]

    print(f"{'tag':10s} {'image':>9s} {'fixed075':>9s}")
    for tag in TAGS:
        probability = np.full(len(ids), np.nan)
        complete = True
        for fold in range(5):
            path = ROOT / "outputs/v3" / tag / f"fold{fold}_predictions.npz"
            if not path.exists():
                complete = False
                break
            blob = np.load(path, allow_pickle=False)
            rows = np.flatnonzero(folds == fold)
            probability[rows] = blob["probability"]
        if not complete:
            print(f"{tag:10s} incomplete")
            continue
        fused, covered = fold_fusion(probability, angles, labels, folds)
        result = np.where(covered, fused, probability)
        print(f"{tag:10s} {loss(labels, probability):9.5f} {loss(labels, result):9.5f}")

    # Final submission validation.
    for name in ("v5_mix5", "v5_mix4"):
        path = ROOT / "work/submissions" / f"{name}.csv"
        if not path.exists():
            continue
        with io.open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        sample = list(csv.reader(io.open(ROOT / "data/raw/sample_submission.csv", encoding="utf-8")))
        header, body = rows[0], rows[1:]
        values = np.array([float(row[1]) for row in body])
        ok_header = [c.strip().lower() for c in header] == ["id", "is_iceberg"]
        ok_rows = len(body) == 8424
        ok_order = [row[0] for row in body] == [row[0] for row in sample[1:]]
        ok_range = bool(np.all((values >= 0) & (values <= 1)) and np.all(np.isfinite(values)))
        print(f"{name}: header={ok_header} rows={len(body)}({ok_rows}) order={ok_order} "
              f"range={ok_range} min={values.min():.6f} max={values.max():.6f} mean={values.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
