"""Centred thumbnail descriptors: can image similarity identify the same pass?

The first attempt compared raw pooled thumbnails and every cosine similarity
saturated near 0.998 because the shared radar background dominates. This version
standardises each image before normalising, so the descriptor encodes structure
rather than brightness, and then asks whether the most similar training image is
a better label witness than the nearest-angle training row.
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


def descriptor(images, size=15):
    x = images.reshape(len(images), 2, size, 75 // size, size, 75 // size).mean(axis=(3, 5))
    flat = x.reshape(len(x), -1).astype(np.float64)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat = flat / np.maximum(flat.std(axis=1, keepdims=True), 1e-6)
    flip = x[:, :, ::-1, :].reshape(len(x), -1).astype(np.float64)
    flip = flip - flip.mean(axis=1, keepdims=True)
    flip = flip / np.maximum(flip.std(axis=1, keepdims=True), 1e-6)
    return flat, flip


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    angles = data["angles"].astype(np.float64)
    labels = data["labels"].astype(np.float64)
    fold_of = {}
    with io.open(ROOT / "data/splits/folds.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fold_of[row["id"]] = int(row["fold"])
    folds = np.array([fold_of[i] for i in ids], dtype=np.int64)

    flat, flip = descriptor(data["images"])
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
    flip /= np.maximum(np.linalg.norm(flip, axis=1, keepdims=True), 1e-9)

    exact = np.zeros(len(labels), dtype=bool)
    near = np.zeros(len(labels), dtype=bool)
    similarity = np.full(len(labels), np.nan)
    image_label = np.full(len(labels), np.nan)
    angle_label = np.full(len(labels), np.nan)
    for fold in range(5):
        heldout = np.flatnonzero(folds == fold)
        reference = np.flatnonzero(folds != fold)
        sim = np.maximum(flat[heldout] @ flat[reference].T, flip[heldout] @ flat[reference].T)
        best = np.argmax(sim, axis=1)
        similarity[heldout] = sim[np.arange(len(heldout)), best]
        image_label[heldout] = labels[reference][best]

        ref_angles = angles[reference]
        for position, row in enumerate(heldout):
            value = angles[row]
            if not np.isfinite(value):
                continue
            finite = np.isfinite(ref_angles)
            same = np.flatnonzero(finite & (ref_angles == value))
            if len(same):
                exact[row] = True
                angle_label[row] = labels[reference][same].mean()
                continue
            near[row] = True
            order = np.argsort(np.abs(ref_angles - value))
            angle_label[row] = labels[reference][order[0]]

    report = {}
    for name, mask in (("exact", exact), ("near", near)):
        if not mask.any():
            continue
        report[name] = {
            "rows": int(mask.sum()),
            "image_nn_agreement": round(float(np.mean(image_label[mask] == labels[mask])), 4),
            "angle_nn_agreement": round(float(np.mean(angle_label[mask] == labels[mask])), 4),
            "similarity_median": round(float(np.nanmedian(similarity[mask])), 4),
            "angle_only_loss": round(log_loss(labels[mask], np.where(angle_label[mask] > 0.5, 0.9, 0.1)), 4),
            "image_only_loss": round(log_loss(labels[mask], np.where(image_label[mask] > 0.5, 0.9, 0.1)), 4),
        }
        print(f"{name:6s} rows={mask.sum():4d} image_nn={report[name]['image_nn_agreement']:.3f} "
              f"angle_nn={report[name]['angle_nn_agreement']:.3f} "
              f"sim_median={report[name]['similarity_median']:.3f}")

    # Do the two neighbours agree with each other, and is the union better?
    both = near & np.isfinite(angle_label) & np.isfinite(image_label)
    if both.any():
        agree = np.mean(angle_label[both] == image_label[both])
        confident = both & (angle_label == image_label)
        report["near_neighbour_agreement"] = round(float(agree), 4)
        report["near_both_confident_rows"] = int(confident.sum())
        if confident.any():
            report["near_both_confident_accuracy"] = round(
                float(np.mean(angle_label[confident] == labels[confident])), 4)
        print(f"near: image/angle neighbours agree {agree:.3f}; "
              f"when they agree ({confident.sum()} rows) accuracy "
              f"{report.get('near_both_confident_accuracy')}")

    (ROOT / "work/leak/similarity_centred.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
