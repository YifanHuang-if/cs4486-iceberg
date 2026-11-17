"""Stack image probabilities with fold-isolated angle evidence, strictly nested.

Two layers:

* Level 1 - image model. Out-of-fold probabilities from trained v3 tags.
* Level 2 - angle evidence plus a small logistic combiner. Exact-angle groups are
  about 98.6% pure while nearest-angle neighbours agree only about 84%, so the
  prior weight is allowed to differ between the two regimes and is fitted.

No held-out label is ever used to build another row's level-2 feature:

* For an outer training row the prior is a leave-one-out estimate inside the
  outer training folds, so it cannot see the outer validation labels either.
* For an outer validation row the prior comes from the outer training labels only.

Coefficients are fitted on outer training rows and applied unchanged to the
outer validation fold, so the reported loss is a genuine five-fold estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.train import OUT, ROOT


def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def log_loss(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def kernel_prior(reference_angles, reference_labels, query_angles, sigma, alpha, exclude_columns=None):
    """Two-tier prior: exact group if it exists, otherwise a Gaussian kernel.

    ``exclude_columns`` gives, for every query row, the column of the reference
    array that must be dropped (its own label, for a leave-one-out estimate).
    Columns refer to the array after non-finite reference angles are removed.
    """
    a = np.asarray(reference_angles, dtype=np.float64)
    y = np.asarray(reference_labels, dtype=np.float64)
    q = np.asarray(query_angles, dtype=np.float64)
    valid = np.isfinite(a)
    a, y = a[valid], y[valid]
    out = np.full(len(q), 0.5)
    covered = np.zeros(len(q), dtype=bool)
    exact_flag = np.zeros(len(q), dtype=bool)
    if not len(a):
        return out, covered, exact_flag
    if exclude_columns is None:
        exclude_columns = np.full(len(q), -1, dtype=np.int64)
    else:
        exclude_columns = np.asarray(exclude_columns, dtype=np.int64)
    for start in range(0, len(q), 256):
        stop = min(start + 256, len(q))
        value = q[start:stop]
        distance = np.abs(value[:, None] - a[None, :])
        exact = distance == 0.0
        rows = np.arange(stop - start)
        own = exclude_columns[start:stop]
        active = own >= 0
        if active.any():
            exact[rows[active], own[active]] = False
        has_exact = exact.any(axis=1)
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
        if active.any():
            weights[rows[active], own[active]] = 0.0
        weights = np.where(has_exact[:, None], exact.astype(np.float64), weights)
        weights = np.where(np.isfinite(value)[:, None], weights, 0.0)
        mass = weights.sum(axis=1)
        out[start:stop] = (weights @ y + alpha) / (mass + 2 * alpha)
        covered[start:stop] = (mass > 1e-3) & np.isfinite(value)
        exact_flag[start:stop] = has_exact & np.isfinite(value)
    return out, covered, exact_flag


def thumbnail_descriptor(images, size=15):
    """Centred, flip-invariant thumbnail descriptor; structure rather than brightness."""
    x = images.reshape(len(images), 2, size, 75 // size, size, 75 // size).mean(axis=(3, 5))
    flat = x.reshape(len(x), -1).astype(np.float64)
    flip = x[:, :, ::-1, :].reshape(len(x), -1).astype(np.float64)
    for matrix in (flat, flip):
        matrix -= matrix.mean(axis=1, keepdims=True)
        matrix /= np.maximum(matrix.std(axis=1, keepdims=True), 1e-6)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    return flat, flip


def witness_labels(flat, flip, labels, query_rows, reference_rows, exclude_self=False):
    """Label of the most similar reference image for each query row."""
    similarity = np.maximum(flat[query_rows] @ flat[reference_rows].T,
                           flip[query_rows] @ flat[reference_rows].T)
    if exclude_self:
        rows = np.arange(len(query_rows))
        similarity[rows, rows] = -np.inf
    best = np.argmax(similarity, axis=1)
    return labels[reference_rows][best], similarity[np.arange(len(query_rows)), best]


def fit_logistic(features, labels, l2=1.0, max_iter=100):
    x = np.column_stack([features, np.ones(len(features))])
    w = np.zeros(x.shape[1])
    for _ in range(max_iter):
        p = sigmoid(x @ w)
        grad = x.T @ (p - labels) / len(labels)
        grad[:-1] += l2 * w[:-1] / len(labels)
        hessian = (x * (p * (1 - p))[:, None]).T @ x / len(labels)
        hessian[:-1, :-1] += l2 * np.eye(x.shape[1] - 1) / len(labels)
        hessian += 1e-9 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def apply_logistic(features, w):
    return sigmoid(np.column_stack([features, np.ones(len(features))]) @ w)


def load_tags(tags):
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"]
    labels = data["labels"].astype(np.float64)
    angles = data["angles"].astype(np.float64)
    folds = read_folds(ids, data["labels"])["fold"]
    stack, seen = [], None
    for tag in tags:
        probability = np.full(len(ids), np.nan)
        present = np.zeros(len(ids), dtype=bool)
        for fold in range(5):
            path = OUT / tag / f"fold{fold}_predictions.npz"
            if not path.exists():
                continue
            blob = np.load(path, allow_pickle=False)
            rows = np.flatnonzero(folds == fold)
            if not np.array_equal(blob["valid_ids"], ids[rows]):
                raise ValueError(f"{tag} fold {fold} ID order mismatch")
            probability[rows] = blob["probability"]
            present[rows] = True
        if not present.any():
            raise ValueError(f"{tag} has no completed folds")
        stack.append(np.where(present, probability, np.nan))
        seen = present if seen is None else (seen & present)
    return ids, labels, angles, folds, np.vstack(stack), seen


def combine(stack, seen, mode):
    """Combine level-1 models. Logit averaging needs every member present."""
    if mode == "prob":
        return np.nanmean(stack, axis=0)
    if mode == "logit":
        values = logit(np.clip(stack, 1e-6, 1 - 1e-6))
        finite = np.isfinite(values).all(axis=0)
        out = np.full(values.shape[1], np.nan)
        out[finite] = sigmoid(np.mean(values[:, finite], axis=0))
        return out
    raise ValueError(mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--name", default="v5")
    parser.add_argument("--sigma", type=float, default=0.012)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--image-witness", action="store_true")
    args = parser.parse_args()

    ids, labels, angles, folds, stack, seen = load_tags(args.tags)
    report = {"tags": args.tags, "sigma": args.sigma, "alpha": args.alpha,
              "rows": int(seen.sum()), "per_tag": {}}
    for index, tag in enumerate(args.tags):
        rows = np.isfinite(stack[index])
        report["per_tag"][tag] = {"rows": int(rows.sum()),
                                  "image": round(log_loss(labels[rows], stack[index][rows]), 6)}

    for mode in ("prob", "logit"):
        image = combine(stack, seen, mode)
        rows = np.isfinite(image)
        # Level-2 features, built without touching any outer validation label.
        prior = np.full(len(labels), np.nan)
        exact = np.zeros(len(labels), dtype=bool)
        covered_flag = np.zeros(len(labels), dtype=bool)
        for fold in range(5):
            outer_valid = folds == fold
            outer_train = ~outer_valid
            # Outer validation rows: prior from outer training labels.
            p, covered, flag = kernel_prior(angles[outer_train], labels[outer_train],
                                            angles[outer_valid], args.sigma, args.alpha)
            prior[outer_valid] = p
            exact[outer_valid] = flag
            covered_flag[outer_valid] = covered
            # Outer training rows: leave-one-out prior inside outer training only.
            train_angles = angles[outer_train]
            finite_train = np.isfinite(train_angles)
            column = np.full(len(train_angles), -1, dtype=np.int64)
            column[finite_train] = np.arange(int(finite_train.sum()))
            p2, covered2, flag2 = kernel_prior(train_angles, labels[outer_train],
                                               train_angles, args.sigma, args.alpha,
                                               exclude_columns=column)
            rows_train = np.flatnonzero(outer_train)
            prior[rows_train] = p2
            exact[rows_train] = flag2
            covered_flag[rows_train] = covered2

        image_logit = logit(np.clip(image, 1e-6, 1 - 1e-6))
        prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))

        # Optional image-witness feature: the label of the most similar training
        # image. Only the nearby rows use it, because exact-angle groups are far
        # more accurate than a thumbnail neighbour.
        witness_direction = np.zeros(len(labels))
        witness_agree = np.zeros(len(labels))
        if args.image_witness:
            cache = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
            flat, flip = thumbnail_descriptor(cache["images"])
            for fold in range(5):
                outer_valid = np.flatnonzero(folds == fold)
                outer_train = np.flatnonzero(folds != fold)
                label_valid, _ = witness_labels(flat, flip, labels, outer_valid, outer_train)
                witness_direction[outer_valid] = 2.0 * label_valid - 1.0
                label_train, _ = witness_labels(flat, flip, labels, outer_train, outer_train,
                                                exclude_self=True)
                witness_direction[outer_train] = 2.0 * label_train - 1.0
            prior_direction = np.where(prior > 0.5, 1.0, -1.0)
            witness_agree = ((witness_direction > 0) == (prior_direction > 0)).astype(float)
            print("image witness: nearby rows where witness and angle agree = "
                  f"{int(np.sum((~exact) & (witness_agree > 0)))}")

        def features(index_rows, groupwise):
            nearby = (~exact[index_rows]).astype(float)
            base = [image_logit[index_rows]]
            if groupwise:
                base.extend([prior_logit[index_rows] * exact[index_rows],
                             prior_logit[index_rows] * (~exact[index_rows])])
            else:
                base.append(prior_logit[index_rows])
            if args.image_witness:
                base.extend([witness_direction[index_rows] * nearby,
                             witness_agree[index_rows] * nearby,
                             witness_direction[index_rows] * witness_agree[index_rows] * nearby])
            return np.column_stack(base)

        results = {}
        # Fixed rule as a reference: uncovered rows keep the image probability.
        fixed = image.copy()
        covered_rows = rows & covered_flag
        fixed[covered_rows] = sigmoid(0.75 * image_logit[covered_rows] + prior_logit[covered_rows])
        results["fixed_w0.75"] = {"loss": round(log_loss(labels[rows], fixed[rows]), 6)}
        results["image_only"] = {"loss": round(log_loss(labels[rows], image[rows]), 6)}

        for groupwise in (False, True):
            prediction = np.full(len(labels), np.nan)
            for fold in range(5):
                outer_valid = np.flatnonzero(folds == fold)
                outer_train = np.flatnonzero((folds != fold) & rows)
                w = fit_logistic(features(outer_train, groupwise), labels[outer_train])
                prediction[outer_valid] = apply_logistic(features(outer_valid, groupwise), w)
            prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
            key = "stack_groupwise" if groupwise else "stack_shared"
            results[key] = {"loss": round(log_loss(labels[rows], prediction[rows]), 6)}

        # Group breakdown for the best stacking rule.
        prediction = np.full(len(labels), np.nan)
        for fold in range(5):
            outer_valid = np.flatnonzero(folds == fold)
            outer_train = np.flatnonzero((folds != fold) & rows)
            w = fit_logistic(features(outer_train, True), labels[outer_train])
            prediction[outer_valid] = apply_logistic(features(outer_valid, True), w)
        prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
        covered_mask = rows & covered_flag
        groups = {
            "exact": rows & exact,
            "near": covered_mask & ~exact,
            "far": rows & ~covered_mask & np.isfinite(angles),
            "missing": rows & ~np.isfinite(angles),
        }
        row_loss = -(labels * np.log(prediction) + (1 - labels) * np.log1p(-prediction))
        total = row_loss[rows].sum()
        results["groups"] = {}
        for name, mask in groups.items():
            if not mask.any():
                continue
            results["groups"][name] = {
                "rows": int(mask.sum()),
                "fused": round(log_loss(labels[mask], prediction[mask]), 6),
                "share": round(float(row_loss[mask].sum() / total), 4),
            }
        report[mode] = results
        print(f"[{mode}] " + "  ".join(f"{k}={v['loss']:.6f}" for k, v in results.items()
                                       if isinstance(v, dict) and "loss" in v))
        for name, row in results["groups"].items():
            print(f"    {name:8s} rows={row['rows']:4d} fused={row['fused']:.5f} share={row['share']:.3f}")

    path = ROOT / "work/leak" / f"{args.name}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
