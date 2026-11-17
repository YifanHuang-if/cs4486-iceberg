"""Nested logistic stacking of image probability and angle prior.

The fixed V3 rule uses logit(p) = 0.75*logit(image) + logit(prior). That assumes
the prior has the same reliability everywhere. Exact-angle groups are ~98.6%
pure while nearest-angle neighbours agree only ~84%, so a per-group fitted
combination may extract more.

Level-2 features are built only from fold-isolated quantities: the image
probability is an existing OOF file and the angle prior never sees a held-out
label. Stacker coefficients are fitted inside each outer training split and
applied to the untouched outer validation fold.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def log_loss(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def prior_for(reference_angles, reference_labels, query_angles, sigma, alpha):
    a = np.asarray(reference_angles, dtype=np.float64)
    y = np.asarray(reference_labels, dtype=np.float64)
    q = np.asarray(query_angles, dtype=np.float64)
    valid = np.isfinite(a)
    a, y = a[valid], y[valid]
    out = np.full(len(q), 0.5)
    covered = np.zeros(len(q), dtype=bool)
    exact = np.zeros(len(q), dtype=bool)
    nearest = np.full(len(q), np.inf)
    if not len(a):
        return out, covered, exact, nearest
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


def fit_logistic(features, labels, l2=1.0, epochs=4000, lr=0.05):
    """Plain Newton/gradient logistic regression with an intercept (last column)."""
    x = np.column_stack([features, np.ones(len(features))])
    w = np.zeros(x.shape[1])
    for _ in range(epochs):
        p = sigmoid(x @ w)
        grad = x.T @ (p - labels) / len(labels)
        grad[:-1] += l2 * w[:-1] / len(labels)
        hessian = (x * (p * (1 - p))[:, None]).T @ x / len(labels)
        hessian[:-1, :-1] += l2 * np.eye(x.shape[1] - 1) / len(labels)
        hessian += 1e-8 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def apply_logistic(features, w):
    x = np.column_stack([features, np.ones(len(features))])
    return sigmoid(x @ w)


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

    oof_path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "work/submissions/v1_fast.oof.npy")
    image = np.clip(np.load(oof_path).astype(np.float64), 1e-6, 1 - 1e-6)

    sigma, alpha = 0.012, 0.1
    prior = np.full(len(labels), 0.5)
    covered = np.zeros(len(labels), dtype=bool)
    exact = np.zeros(len(labels), dtype=bool)
    nearest = np.full(len(labels), np.inf)
    for fold in np.unique(folds):
        heldout = folds == fold
        p, mask, flag, dist = prior_for(angles[~heldout], labels[~heldout], angles[heldout], sigma, alpha)
        prior[heldout] = p
        covered[heldout] = mask
        exact[heldout] = flag
        nearest[heldout] = dist

    image_logit = logit(image)
    prior_logit = logit(prior)
    distance_feature = np.where(np.isfinite(nearest), np.log1p(nearest * 1000.0), 0.0)

    schemes = {
        "fixed_w0.75": lambda idx: None,
        "stack_all": lambda idx: np.column_stack([image_logit[idx], prior_logit[idx]]),
        "stack_groupwise": lambda idx: np.column_stack([
            image_logit[idx], prior_logit[idx] * exact[idx], prior_logit[idx] * (~exact[idx]),
            exact[idx].astype(float)]),
        "stack_with_distance": lambda idx: np.column_stack([
            image_logit[idx], prior_logit[idx], distance_feature[idx] * covered[idx]]),
        "stack_groupwise_distance": lambda idx: np.column_stack([
            image_logit[idx], prior_logit[idx] * exact[idx], prior_logit[idx] * (~exact[idx]),
            exact[idx].astype(float), distance_feature[idx] * covered[idx]]),
    }

    report = {"image_only": log_loss(labels, image), "sigma": sigma, "alpha": alpha,
              "exact_rows": int(exact.sum()), "near_rows": int((covered & ~exact).sum())}
    for name, builder in schemes.items():
        prediction = np.full(len(labels), np.nan)
        for fold in np.unique(folds):
            outer_train = folds != fold
            outer_valid = folds == fold
            if builder(outer_train) is None:
                valid_prior = prior[outer_valid]
                prediction[outer_valid] = np.where(
                    covered[outer_valid],
                    sigmoid(0.75 * image_logit[outer_valid] + logit(valid_prior)),
                    image[outer_valid])
            else:
                w = fit_logistic(builder(outer_train), labels[outer_train])
                prediction[outer_valid] = apply_logistic(builder(outer_valid), w)
        prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
        entry = {"loss": round(log_loss(labels, prediction), 6)}
        for group_name, mask in (("exact", exact), ("near", covered & ~exact),
                                 ("far", ~covered & np.isfinite(angles)),
                                 ("missing", ~np.isfinite(angles))):
            if mask.any():
                entry[group_name] = [int(mask.sum()), round(log_loss(labels[mask], prediction[mask]), 6)]
        report[name] = entry
        print(f"{name:26s} loss={entry['loss']:.6f}  exact={entry.get('exact')}  near={entry.get('near')}")

    out = ROOT / "work/leak/stacker.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
