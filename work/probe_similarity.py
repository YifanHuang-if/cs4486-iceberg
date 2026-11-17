"""Is image similarity informative, and does it sharpen the angle prior?

Builds a flip-invariant 15x15 two-band thumbnail descriptor, computes cosine
similarity between held-out rows and training-fold rows only, and asks two
questions:

1. As a standalone k-nearest-neighbour classifier, how much log loss does the
   descriptor achieve?
2. Does similarity separate the "same acquisition pass" rows from merely
   nearby-angle rows? If it does, it can be used to up-weight matching
   neighbours inside the angle prior.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def log_loss(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def thumbnails(images, size=15):
    x = images.reshape(len(images), 2, size, 75 // size, size, 75 // size).mean(axis=(3, 5))
    flat = x.reshape(len(x), -1).astype(np.float64)
    flipped = x[:, :, ::-1, :].reshape(len(x), -1).astype(np.float64)
    return flat, flipped


def normalise(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-9)


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    images = data["images"]
    angles = data["angles"].astype(np.float64)
    labels = data["labels"].astype(np.float64)
    fold_of = {}
    with io.open(ROOT / "data/splits/folds.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fold_of[row["id"]] = int(row["fold"])
    folds = np.array([fold_of[i] for i in ids], dtype=np.int64)

    flat, flipped = thumbnails(images)
    u = normalise(flat)
    v = normalise(flipped)
    report = {"rows": int(len(labels))}

    # Fold-isolated nearest neighbours.
    best_similarity = np.full(len(labels), -np.inf)
    knn_probability = np.full(len(labels), 0.5)
    knn_k = 8
    for fold in range(5):
        heldout = np.flatnonzero(folds == fold)
        reference = np.flatnonzero(folds != fold)
        similarity = np.maximum(u[heldout] @ u[reference].T, v[heldout] @ u[reference].T)
        report_slice = similarity
        order = np.argsort(-report_slice, axis=1)[:, :knn_k]
        weights = np.take_along_axis(report_slice, order, axis=1).clip(0.0)
        neighbour_labels = labels[reference][order]
        knn_probability[heldout] = (np.sum(weights * neighbour_labels, axis=1) + 0.5) / (weights.sum(axis=1) + 1.0)
        best_similarity[heldout] = report_slice.max(axis=1)

    report["knn"] = {"k": knn_k, "log_loss": round(log_loss(labels, knn_probability), 6),
                     "accuracy": round(float(np.mean((knn_probability > 0.5) == labels)), 6)}
    print(f"kNN(k={knn_k}) loss={report['knn']['log_loss']:.4f} acc={report['knn']['accuracy']:.3f}")

    # Does similarity separate exact-match rows from near-only rows?
    exact = np.zeros(len(labels), dtype=bool)
    for fold in range(5):
        heldout = np.flatnonzero(folds == fold)
        reference = np.flatnonzero((folds != fold) & np.isfinite(angles))
        set_angles = np.sort(angles[reference])
        value = angles[heldout]
        finite = np.isfinite(value)
        pos = np.clip(np.searchsorted(set_angles, value[finite]), 0, len(set_angles) - 1)
        match = np.abs(set_angles[pos] - value[finite]) < 1e-12
        exact[heldout[finite]] = match
    report["similarity_by_group"] = {}
    for name, mask in (("exact", exact), ("observed_other", np.isfinite(angles) & ~exact)):
        if not mask.any():
            continue
        values = best_similarity[mask]
        report["similarity_by_group"][name] = {
            "rows": int(mask.sum()),
            "median": round(float(np.median(values)), 4),
            "p90": round(float(np.quantile(values, 0.9)), 4),
            "p99": round(float(np.quantile(values, 0.99)), 4),
        }
        print(f"{name:15s} rows={mask.sum():4d} median={np.median(values):.3f} p99={np.quantile(values,0.99):.3f}")

    # Is kNN complementary to the angle prior? Correlation of the two errors.
    report["knn_by_angle_group"] = {}
    for name, mask in (("exact", exact), ("observed_other", np.isfinite(angles) & ~exact)):
        if not mask.any():
            continue
        report["knn_by_angle_group"][name] = round(log_loss(labels[mask], knn_probability[mask]), 6)

    np.save(ROOT / "work/leak/knn_probability.npy", knn_probability)
    np.save(ROOT / "work/leak/nearest_similarity.npy", best_similarity)
    (ROOT / "work/leak/similarity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
