"""Does pooling same-angle rows inside the query set help the rows that lack a
training-side exact match?

At test time 70% of the scored rows share an exact angle with another test row,
so a group whose pass never appears in training can still be pooled. Cross
validation has far smaller query groups, which makes the CV estimate
conservative; this measures the effect on exactly the subgroup that pooling is
meant to help.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.fusion import fold_fusion, loss
from src.v3.train import ROOT


def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def load_ensemble(tags):
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    folds = read_folds(data["ids"], data["labels"])["fold"]
    stack = []
    for tag in tags:
        probability = np.full(len(ids), np.nan)
        for fold in range(5):
            path = ROOT / "outputs/v3" / tag / f"fold{fold}_predictions.npz"
            blob = np.load(path, allow_pickle=False)
            rows = np.flatnonzero(folds == fold)
            probability[rows] = blob["probability"]
        stack.append(probability)
    return data, folds, np.mean(stack, axis=0)


def main() -> int:
    tags = ["s101", "s202", "s404", "v5_shape"]
    data, folds, image = load_ensemble(tags)
    labels = data["labels"].astype(np.float64)
    angles = data["angles"].astype(np.float64)

    fused, covered = fold_fusion(image, angles, labels, folds)
    fused = np.where(covered, fused, image)

    # Which held-out rows have no training-fold exact match but do have a
    # same-angle mate inside their own validation fold?
    poolable = np.zeros(len(labels), dtype=bool)
    mate_count = np.zeros(len(labels), dtype=np.int64)
    for fold in range(5):
        heldout = np.flatnonzero(folds == fold)
        reference = np.flatnonzero((folds != fold) & np.isfinite(angles))
        set_angles = np.sort(angles[reference])
        for row in heldout:
            value = angles[row]
            if not np.isfinite(value):
                continue
            position = np.searchsorted(set_angles, value)
            has_train_match = False
            if position < len(set_angles) and abs(set_angles[position] - value) < 1e-12:
                has_train_match = True
            if has_train_match:
                continue
            mates = np.flatnonzero((folds == fold) & (angles == value))
            mate_count[row] = len(mates)
            poolable[row] = len(mates) > 1

    report = {
        "rows_without_train_match": int(np.sum(np.isfinite(angles) & ~covered)),
        "rows_poolable_in_fold": int(poolable.sum()),
        "labels": labels.tolist(),
    }

    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        pooled = fused.copy()
        for fold in range(5):
            heldout = np.flatnonzero(folds == fold)
            for value in np.unique(angles[heldout][np.isfinite(angles[heldout])]):
                rows = heldout[angles[heldout] == value]
                if len(rows) > 1:
                    pooled[rows] = (1 - weight) * fused[rows] + weight * fused[rows].mean()
        entry = {"all": round(loss(labels, pooled), 6)}
        if poolable.any():
            entry["poolable_only"] = round(loss(labels[poolable], pooled[poolable]), 6)
            entry["poolable_baseline"] = round(loss(labels[poolable], fused[poolable]), 6)
        report[f"weight_{weight}"] = entry
        print(f"weight={weight:.2f}  all={entry['all']:.6f}  "
              f"poolable={entry.get('poolable_only')} (base {entry.get('poolable_baseline')})")

    print(json.dumps({k: v for k, v in report.items() if k != "labels"}, indent=2))
    (ROOT / "work/leak/query_pooling.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "labels"}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
