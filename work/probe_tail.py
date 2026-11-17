"""Inspect the rows that carry most of the pooled log loss.

For each of the worst rows this prints the image probability, the angle prior,
the fused probability, the truth, the angle group size and the nearest-angle
distance. This identifies whether the tail is an image failure, a prior failure,
or a fusion failure.
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


def prior_for(reference_angles, reference_labels, query_angles, sigma=0.012, alpha=0.1):
    a = np.asarray(reference_angles, dtype=np.float64)
    y = np.asarray(reference_labels, dtype=np.float64)
    q = np.asarray(query_angles, dtype=np.float64)
    valid = np.isfinite(a)
    a, y = a[valid], y[valid]
    out = np.full(len(q), 0.5)
    covered = np.zeros(len(q), dtype=bool)
    exact = np.zeros(len(q), dtype=bool)
    nearest = np.full(len(q), np.inf)
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
        exact[start:stop] = has_exact & np.isfinite(value)
        nearest[start:stop] = np.where(np.isfinite(value), distance.min(axis=1), np.inf)
    return out, covered, exact, nearest


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
    exact = np.zeros(len(labels), dtype=bool)
    nearest = np.full(len(labels), np.inf)
    for fold in np.unique(folds):
        heldout = folds == fold
        p, mask, flag, dist = prior_for(angles[~heldout], labels[~heldout], angles[heldout])
        prior[heldout] = p
        covered[heldout] = mask
        exact[heldout] = flag
        nearest[heldout] = dist
    fused = image.copy()
    fused[covered] = sigmoid(0.75 * logit(image[covered]) + logit(prior[covered]))

    row_loss = -(labels * np.log(np.clip(fused, 1e-7, 1 - 1e-7))
                 + (1 - labels) * np.log1p(-np.clip(fused, 1e-7, 1 - 1e-7)))
    order = np.argsort(row_loss)[::-1]
    total = row_loss.sum()

    # How much loss comes from rows where image and prior disagree in direction?
    disagree = covered & ((image > 0.5) != (prior > 0.5))
    agree = covered & ~disagree
    report = {
        "total_rows": int(len(labels)),
        "pooled": round(float(row_loss.mean()), 6),
        "groups": {
            "agree": {"rows": int(agree.sum()), "loss_sum": round(float(row_loss[agree].sum()), 3),
                      "share": round(float(row_loss[agree].sum() / total), 4)},
            "disagree": {"rows": int(disagree.sum()), "loss_sum": round(float(row_loss[disagree].sum()), 3),
                         "share": round(float(row_loss[disagree].sum() / total), 4)},
            "uncovered": {"rows": int((~covered).sum()), "loss_sum": round(float(row_loss[~covered].sum()), 3),
                          "share": round(float(row_loss[~covered].sum() / total), 4)},
        },
        "top": [],
    }
    for index in order[:30]:
        report["top"].append({
            "id": ids[index], "fold": int(folds[index]), "label": int(labels[index]),
            "image": round(float(image[index]), 5), "prior": round(float(prior[index]), 5),
            "fused": round(float(fused[index]), 5), "loss": round(float(row_loss[index]), 4),
            "group": "exact" if exact[index] else ("near" if covered[index] else "far"),
            "nearest": None if not np.isfinite(nearest[index]) else round(float(nearest[index]), 6),
        })
    text = json.dumps(report, indent=2)
    print(text)
    (ROOT / "work/leak/tail_analysis.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
