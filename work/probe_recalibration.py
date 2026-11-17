"""Does a global recalibration of the fused probability reduce log loss?

The V3 error analysis reports that 25 rows carry 45% of the pooled loss, so the
tail of confident mistakes may dominate. Temperature scaling and probability
clipping are fitted on four folds and scored on the fifth, which keeps the
comparison honest.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def log_loss(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def prior_for(reference_angles, reference_labels, query_angles, sigma=0.012, alpha=0.1):
    a = np.asarray(reference_angles, dtype=np.float64)
    y = np.asarray(reference_labels, dtype=np.float64)
    q = np.asarray(query_angles, dtype=np.float64)
    valid = np.isfinite(a)
    a, y = a[valid], y[valid]
    out = np.full(len(q), 0.5)
    covered = np.zeros(len(q), dtype=bool)
    for start in range(0, len(q), 256):
        stop = min(start + 256, len(q))
        value = q[start:stop]
        distance = np.abs(value[:, None] - a[None, :])
        has_exact = (distance == 0.0).any(axis=1)
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
        weights = np.where(has_exact[:, None], (distance == 0.0).astype(np.float64), weights)
        weights = np.where(np.isfinite(value)[:, None], weights, 0.0)
        mass = weights.sum(axis=1)
        out[start:stop] = (weights @ y + alpha) / (mass + 2 * alpha)
        covered[start:stop] = (mass > 1e-3) & np.isfinite(value)
    return out, covered


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

    image = np.clip(np.load(ROOT / "work/submissions/v1_fast.oof.npy").astype(np.float64), 1e-6, 1 - 1e-6)
    prior = np.full(len(labels), 0.5)
    covered = np.zeros(len(labels), dtype=bool)
    for fold in np.unique(folds):
        heldout = folds == fold
        p, mask = prior_for(angles[~heldout], labels[~heldout], angles[heldout])
        prior[heldout] = p
        covered[heldout] = mask
    fused = image.copy()
    fused[covered] = sigmoid(0.75 * logit(image[covered]) + logit(prior[covered]))

    base = log_loss(labels, fused)
    # Where does the loss live?
    row_loss = -(labels * np.log(np.clip(fused, 1e-7, 1 - 1e-7))
                 + (1 - labels) * np.log1p(-np.clip(fused, 1e-7, 1 - 1e-7)))
    order = np.argsort(row_loss)[::-1]
    total = row_loss.sum()
    concentration = {str(k): round(float(row_loss[order[:k]].sum() / total), 4) for k in (5, 10, 25, 50, 100)}

    temperatures = np.round(np.arange(0.6, 1.85, 0.05), 2)
    clips = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

    # Nested: pick (T, clip) on the four training folds, apply to the held-out fold.
    prediction = np.full(len(labels), np.nan)
    chosen = []
    for fold in np.unique(folds):
        outer_valid = folds == fold
        outer_train = ~outer_valid
        best = None
        for temperature in temperatures:
            for clip in clips:
                candidate = sigmoid(logit(fused[outer_train]) / temperature)
                if clip > 0:
                    candidate = np.clip(candidate, clip, 1 - clip)
                value = log_loss(labels[outer_train], candidate)
                if best is None or value < best[0]:
                    best = (value, temperature, clip)
        _, temperature, clip = best
        chosen.append({"fold": int(fold), "temperature": float(temperature), "clip": float(clip)})
        candidate = sigmoid(logit(fused[outer_valid]) / temperature)
        if clip > 0:
            candidate = np.clip(candidate, clip, 1 - clip)
        prediction[outer_valid] = candidate

    report = {
        "base_fused": round(base, 6),
        "loss_concentration": concentration,
        "nested_recalibrated": round(log_loss(labels, prediction), 6),
        "chosen": chosen,
        "fixed_grid_best_in_sample": None,
    }
    best_fixed = None
    for temperature in temperatures:
        for clip in clips:
            candidate = sigmoid(logit(fused) / temperature)
            if clip > 0:
                candidate = np.clip(candidate, clip, 1 - clip)
            value = log_loss(labels, candidate)
            if best_fixed is None or value < best_fixed[0]:
                best_fixed = (value, float(temperature), float(clip))
    report["fixed_grid_best_in_sample"] = {"loss": round(best_fixed[0], 6),
                                           "temperature": best_fixed[1], "clip": best_fixed[2]}
    print(json.dumps(report, indent=2))
    (ROOT / "work/leak/recalibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
