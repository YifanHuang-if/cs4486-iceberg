"""Distance-binned, calibrated angle prior instead of one Gaussian kernel.

Nearest-neighbour agreement is not flat in distance: exact groups are 98.6%
pure, the 1e-4 to 5e-4 band is about 89%, the 5e-4 to 2e-3 band about 80%, and
2e-3 to 6e-2 about 85%. A single sigma cannot express that.

This builds the prior as an empirically calibrated reliability per distance bin.
Reliability is estimated only from the outer training rows (leave-one-out inside
them) and applied to the untouched outer validation fold, so the comparison with
the Gaussian kernel is honest.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.train import OUT, ROOT
from work.v5_fuse import (apply_logistic, fit_logistic, kernel_prior, logit,
                          log_loss, sigmoid)

TAGS = ["s101", "s202", "s404", "v5_shape", "stu1", "stu5"]
EDGES = [0.0, 5e-4, 2e-3, 6e-2]


def nearest_neighbour(angles, labels, query_rows, reference_rows, exclude_self=False):
    """Label and distance of the closest reference angle for every query row."""
    reference = reference_rows[np.isfinite(angles[reference_rows])]
    order = np.argsort(angles[reference], kind="stable")
    reference = reference[order]
    values = angles[reference]
    result_label = np.full(len(query_rows), 0.5)
    result_distance = np.full(len(query_rows), np.inf)
    result_bin = np.full(len(query_rows), -1, dtype=np.int64)
    for index, row in enumerate(query_rows):
        value = angles[row]
        if not np.isfinite(value):
            continue
        position = np.searchsorted(values, value)
        candidates = [p for p in (position - 1, position) if 0 <= p < len(values)]
        if exclude_self and 0 <= position < len(values) and values[position] == value:
            candidates = [p for p in candidates if reference[p] != row]
        if not candidates:
            continue
        best = min(candidates, key=lambda p: abs(values[p] - value))
        if exclude_self and reference[best] == row:
            continue
        distance = abs(values[best] - value)
        result_distance[index] = distance
        result_label[index] = labels[reference[best]]
        if distance == 0.0:
            result_bin[index] = 0
        elif distance < EDGES[1]:
            result_bin[index] = 1
        elif distance < EDGES[2]:
            result_bin[index] = 2
        elif distance < EDGES[3]:
            result_bin[index] = 3
    return result_label, result_distance, result_bin


def binned_prior(query_label, query_bin, reliability, shrinkage=12.0):
    """Map a neighbour label and its distance bin to a calibrated prior."""
    prior = np.full(len(query_label), 0.5)
    for index, (label, bin_index) in enumerate(zip(query_label, query_bin)):
        if bin_index < 0:
            continue
        rate = reliability[bin_index]
        prior[index] = rate if label > 0.5 else 1.0 - rate
    return np.clip(prior, 0.02, 0.98)


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    labels = data["labels"].astype(np.float64)
    angles = data["angles"].astype(np.float64)
    folds = read_folds(data["ids"], data["labels"])["fold"]

    stack = []
    for tag in TAGS:
        probability = np.full(len(ids), np.nan)
        for fold in range(5):
            blob = np.load(OUT / tag / f"fold{fold}_predictions.npz", allow_pickle=False)
            probability[folds == fold] = blob["probability"]
        stack.append(probability)
    image = np.mean(stack, axis=0)
    image_logit = logit(np.clip(image, 1e-6, 1 - 1e-6))

    kernel = np.full(len(labels), 0.5)
    kernel_exact = np.zeros(len(labels), dtype=bool)
    binned = np.full(len(labels), 0.5)
    binned_exact = np.zeros(len(labels), dtype=bool)
    reliability_per_fold = {}

    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        # Calibrate reliability on the outer training rows only.
        label_in, _, bin_in = nearest_neighbour(angles, labels, train, train, exclude_self=True)
        counts = np.zeros(len(EDGES))
        correct = np.zeros(len(EDGES))
        for label, bin_index in zip(label_in, bin_in):
            if bin_index < 0:
                continue
            counts[bin_index] += 1.0
            correct[bin_index] += 1.0 if label > 0.5 else 0.0
        reliability = (correct + 12.0 * 0.5) / (counts + 12.0)
        reliability_per_fold[str(fold)] = {
            "counts": counts.tolist(), "reliability": reliability.round(4).tolist()}
        # Apply to the untouched outer validation fold.
        label_out, _, bin_out = nearest_neighbour(angles, labels, valid, train)
        binned[valid] = binned_prior(label_out, bin_out, reliability)
        binned_exact[valid] = bin_out == 0
        prior_kernel, covered, flag = kernel_prior(angles[train], labels[train], angles[valid], 0.012, 0.1)
        kernel[valid] = prior_kernel
        kernel_exact[valid] = flag

    def evaluate(prior_values, exact_flags, name):
        prior_logit = logit(np.clip(prior_values, 1e-6, 1 - 1e-6))
        features = np.column_stack([
            image_logit,
            prior_logit * exact_flags,
            prior_logit * (~exact_flags),
        ])
        prediction = np.full(len(labels), np.nan)
        for fold in range(5):
            train = folds != fold
            weights = fit_logistic(features[train], labels[train])
            prediction[folds == fold] = apply_logistic(features[folds == fold], weights)
        prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
        total = log_loss(labels, prediction)
        covered = np.isfinite(prior_values) & (prior_values != 0.5)
        groups = {}
        for group, mask in (("exact", exact_flags),
                            ("near", covered & ~exact_flags),
                            ("other", ~covered & ~exact_flags)):
            if mask.any():
                groups[group] = [int(mask.sum()), round(log_loss(labels[mask], prediction[mask]), 5)]
        print(f"{name:14s} loss={total:.6f}  {groups}")
        return {"loss": round(total, 6), "groups": groups}

    print(json.dumps(reliability_per_fold, indent=2)[:600])
    report = {"kernel": evaluate(kernel, kernel_exact, "kernel_012"),
              "binned": evaluate(binned, binned_exact, "binned"),
              "reliability_per_fold": reliability_per_fold,
              "edges": EDGES}
    (ROOT / "work/leak/binned_prior.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
