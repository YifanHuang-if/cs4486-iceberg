"""Ground truth for the document and deck: CV, per-fold, and loss decomposition."""
from __future__ import annotations

import numpy as np

from work.v5_fuse import apply_logistic, fit_logistic, log_loss, logit
from work.v5_greedy import build_prior, load

TAGS = ["v5_shape", "hp_star", "hp_b", "hp_d", "pt25a", "pt25b", "stu5"]


def main() -> int:
    ids, labels, angles, folds, matrix = load(TAGS)
    prior, exact = build_prior(angles, labels, folds)
    image = np.clip(matrix.mean(axis=0), 1e-6, 1 - 1e-6)
    prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))
    features = np.column_stack([logit(image), prior_logit * exact, prior_logit * (~exact)])

    prediction = np.zeros(len(labels))
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        prediction[valid] = apply_logistic(features[valid], fit_logistic(features[train], labels[train]))

    pooled = log_loss(labels, prediction)
    print(f"pooled CV = {pooled:.6f}")
    for fold in range(5):
        mask = folds == fold
        print(f"  fold {fold}: {log_loss(labels[mask], prediction[mask]):.6f}  n={int(mask.sum())}")

    observed = np.isfinite(angles) & (angles != 0)
    missing = ~observed
    covered = observed & (prior != 0.5)
    near = covered & ~exact
    far = observed & ~covered

    print()
    total = pooled * len(labels)
    for name, mask in (("exact", exact & ~missing), ("near", near), ("far", far), ("missing", missing)):
        if mask.sum() == 0:
            continue
        value = log_loss(labels[mask], prediction[mask])
        share = value * mask.sum() / total
        print(f"{name:8s} rows={int(mask.sum()):4d}  loss={value:.6f}  share={share:.1%}")

    print()
    print(f"labels: ship={int((labels == 0).sum())} iceberg={int((labels == 1).sum())}")
    print(f"missing-angle rows: {int(missing.sum())}, iceberg rate={labels[missing].mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
