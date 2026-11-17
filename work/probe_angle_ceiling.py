"""Measure the angle-only information ceiling on the held-out rows.

Answers two questions that decide where the remaining log loss can come from:

1. For rows with no exact angle match in the training fold, how far is the
   nearest training angle, and how often does it carry the same label?
2. What is the best log loss an angle-only predictor can reach on those rows,
   and therefore how much of the current 0.207 loss is removable?

Everything uses only training-fold labels for each held-out row (leave-one-out
within the fold), so the numbers are honest five-fold estimates.
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


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=True)
    ids = data["ids"].astype(str)
    angles = data["angles"].astype(np.float64)
    labels = data["labels"].astype(np.float64)

    fold_of = {}
    with io.open(ROOT / "data/splits/folds.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fold_of[row["id"]] = int(row["fold"])
    folds = np.array([fold_of[i] for i in ids], dtype=np.int64)
    assert len(folds) == len(labels) == 1604

    observed = np.isfinite(angles)
    nn_distance = np.full(len(labels), np.inf)
    nn_label = np.full(len(labels), np.nan)
    nn_count = np.zeros(len(labels), dtype=np.int64)

    # Leave-one-out inside each training fold.
    for fold in np.unique(folds):
        heldout = np.flatnonzero(folds == fold)
        reference = np.flatnonzero((folds != fold) & observed)
        ref_angles = angles[reference]
        ref_labels = labels[reference]
        order = np.argsort(ref_angles, kind="stable")
        sorted_angles = ref_angles[order]
        sorted_labels = ref_labels[order]
        for row in heldout:
            if not observed[row]:
                continue
            value = angles[row]
            position = np.searchsorted(sorted_angles, value)
            candidates = []
            for offset in (position - 1, position):
                if 0 <= offset < len(sorted_angles):
                    candidates.append(offset)
            # Exact matches first, otherwise the nearest angle.
            matches = np.flatnonzero(sorted_angles == value)
            if len(matches):
                nn_distance[row] = 0.0
                nn_count[row] = len(matches)
                nn_label[row] = float(sorted_labels[matches].mean())
                continue
            if not candidates:
                continue
            best = min(candidates, key=lambda i: abs(sorted_angles[i] - value))
            nn_distance[row] = abs(sorted_angles[best] - value)
            nn_count[row] = 1
            nn_label[row] = float(sorted_labels[best])

    exact = observed & (nn_distance == 0.0)
    near = observed & (nn_distance > 0.0) & (nn_distance < 0.06)
    far = observed & (nn_distance >= 0.06)
    missing = ~observed

    report = {
        "rows": {
            "exact": int(exact.sum()),
            "near": int(near.sum()),
            "far": int(far.sum()),
            "missing": int(missing.sum()),
        }
    }

    # Distribution of the nearest-angle distance for the near group.
    near_d = nn_distance[near]
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    report["near_distance_quantiles"] = {
        str(q): float(np.quantile(near_d, q)) for q in quantiles
    }
    bins = [0.0, 1e-6, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 6e-2]
    histogram, _ = np.histogram(near_d, bins=bins)
    report["near_distance_histogram"] = [
        {"lo": bins[i], "hi": bins[i + 1], "rows": int(histogram[i])}
        for i in range(len(histogram))
    ]

    # How predictive is the nearest angle's label?
    def agreement(mask):
        if not mask.any():
            return None
        neighbour = np.round(nn_label[mask])
        return float(np.mean(neighbour == labels[mask]))

    report["nearest_label_agreement"] = {
        "exact": agreement(exact),
        "near": agreement(near),
        "near_d_le_2e3": agreement(near & (nn_distance <= 2e-3)),
        "near_d_le_5e3": agreement(near & (nn_distance <= 5e-3)),
        "near_d_gt_5e3": agreement(near & (nn_distance > 5e-3)),
    }

    # Best angle-only log loss per group when predicting the neighbour label
    # with a fixed confidence c. This is an optimistic ceiling, not a model.
    grid = {}
    for name, mask in (("exact", exact), ("near", near), ("far", far)):
        if not mask.any():
            continue
        y = labels[mask]
        neighbour = np.round(nn_label[mask])
        best = None
        for c in np.linspace(0.5, 0.999, 250):
            p = np.where(neighbour == 1, c, 1 - c)
            value = log_loss(y, p)
            if best is None or value < best[1]:
                best = (float(c), value)
        grid[name] = {"confidence": best[0], "log_loss": best[1], "rows": int(mask.sum())}
    report["angle_only_ceiling"] = grid

    # If the near rows could be brought down to a target loss, what would the
    # whole pooled score become? Current per-group losses are from the V3 run.
    current = {
        "exact": (0.024792585821466, 764),
        "near": (0.20687782007126573, 697),
        "far": (0.272635206900713, 10),
        "missing": (0.0005074864528228376, 133),
    }
    total_rows = sum(v[1] for v in current.values())
    fixed_sum = sum(v[0] * v[1] for k, v in current.items() if k != "near")
    report["pooled_if_near_reaches"] = {
        str(round(target, 3)): round((fixed_sum + target * 697) / total_rows, 6)
        for target in (0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08)
    }
    report["pooled_at_1e3"] = round((fixed_sum + 0.001 * 697) / total_rows, 6)
    report["current_pooled"] = round(sum(v[0] * v[1] for v in current.values()) / total_rows, 6)

    out = ROOT / "work/leak/angle_ceiling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
