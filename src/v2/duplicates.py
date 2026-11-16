"""Near-duplicate screening between the machine-generated and real test rows.

Statoil added 5,000 machine-generated test images. If a synthetic image is a
transformed copy of a real image, the pair can be matched, and if that real image
is a *training* row the label is known. This module measures that structure with
flip-invariant thumbnail similarity before anything is concluded from it.

Screening heuristic only: similarity is not proof of a shared scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"
OUT = ROOT / "work" / "duplicates"


def thumbnails(images: np.ndarray, size: int = 15) -> np.ndarray:
    x = images.reshape(len(images), 2, size, 75 // size, size, 75 // size).mean(axis=(3, 5))
    flat = x.reshape(len(x), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


def flip_invariant(query: np.ndarray, reference: np.ndarray, block: int = 512):
    """Best cosine similarity over the eight dihedral views of each query row."""
    best = np.full(len(query), -np.inf)
    best_index = np.zeros(len(query), dtype=np.int64)
    for code in range(8):
        view = query.reshape(len(query), 2, 15, 15).copy()
        if code & 1:
            view = view.transpose(0, 1, 3, 2)
        if code & 2:
            view = view[:, :, :, ::-1]
        if code & 4:
            view = view[:, :, ::-1, :]
        flat = view.reshape(len(query), -1)
        flat = flat - flat.mean(axis=1, keepdims=True)
        flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
        for start in range(0, len(query), block):
            part = flat[start:start + block]
            scores = part @ reference.T
            index = scores.argmax(axis=1)
            values = scores[np.arange(len(part)), index]
            keep = values > best[start:start + block]
            best[start:start + block] = np.where(keep, values, best[start:start + block])
            best_index[start:start + block] = np.where(keep, index, best_index[start:start + block])
    return best, best_index


def decimals(angles: np.ndarray) -> np.ndarray:
    out = np.zeros(len(angles), dtype=np.int64)
    for index, value in enumerate(angles):
        scaled, count = float(value), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        out[index] = count
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    train = np.load(CACHE / "train.npz", allow_pickle=False)
    test = np.load(CACHE / "test.npz", allow_pickle=False)
    machine = decimals(test["angles"]) > 4
    real = ~machine

    train_thumbs = thumbnails(train["images"])
    test_thumbs = thumbnails(test["images"])

    scores, index = flip_invariant(test_thumbs[machine], train_thumbs)
    labels = train["labels"].astype(int)
    report = {
        "machine_rows": int(machine.sum()),
        "real_rows": int(real.sum()),
        "threshold": args.threshold,
        "machine_to_train_similarity": {
            "max": float(scores.max()), "p99": float(np.percentile(scores, 99)),
            "p95": float(np.percentile(scores, 95)), "median": float(np.median(scores)),
        },
        "machine_rows_above_threshold": int((scores >= args.threshold).sum()),
        "matched_train_label_share_iceberg": float(labels[index[scores >= args.threshold]].mean())
        if (scores >= args.threshold).any() else None,
    }

    # Similarity of the synthetic rows to *real test* rows, split by angle equality.
    scores_real, index_real = flip_invariant(test_thumbs[machine], test_thumbs[real])
    machine_angles = np.round(test["angles"][machine], 6)
    real_angles = np.round(test["angles"][real], 6)
    same_angle = machine_angles == real_angles[index_real]
    report["machine_to_real_test"] = {
        "max": float(scores_real.max()), "p99": float(np.percentile(scores_real, 99)),
        "above_threshold": int((scores_real >= args.threshold).sum()),
        "above_threshold_same_angle": int(((scores_real >= args.threshold) & same_angle).sum()),
        "above_0.99": int((scores_real >= 0.99).sum()),
        "above_0.99_same_angle": int(((scores_real >= 0.99) & same_angle).sum()),
    }

    # Within-split duplicate check for the real rows and across train/test.
    train_self, train_self_index = flip_invariant(train_thumbs, train_thumbs)
    self_match = train_self_index == np.arange(len(train_thumbs))
    train_self = np.where(self_match, -np.inf, train_self)
    report["train_self"] = {"above_threshold": int((train_self >= args.threshold).sum()),
                            "max": float(train_self.max())}
    real_to_train, real_to_train_index = flip_invariant(test_thumbs[real], train_thumbs)
    report["real_test_to_train"] = {
        "above_threshold": int((real_to_train >= args.threshold).sum()),
        "above_0.99": int((real_to_train >= 0.99).sum()),
        "max": float(real_to_train.max()),
        "index_of_max": int(real_to_train_index[real_to_train.argmax()]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    pairs = []
    for row in np.flatnonzero(scores_real >= args.threshold):
        pairs.append({
            "machine_id": str(test["ids"][np.flatnonzero(machine)[row]]),
            "real_id": str(test["ids"][np.flatnonzero(real)[index_real[row]]]),
            "similarity": float(scores_real[row]),
            "same_angle": bool(same_angle[row]),
            "machine_angle": float(machine_angles[row]),
            "real_angle": float(real_angles[index_real[row]]),
            "nearest_train_id": str(train["ids"][index[row]]),
            "nearest_train_similarity": float(scores[row]),
            "nearest_train_label": int(labels[index[row]]),
        })
    import csv
    with (OUT / "machine_real_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]) if pairs else ["machine_id"])
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
