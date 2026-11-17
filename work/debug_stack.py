"""Compare the two stacking harnesses fold by fold on one member."""
from __future__ import annotations

import numpy as np

from src.v2.train import read_folds
from src.v3.train import OUT, ROOT
from work.v5_fuse import (apply_logistic, fit_logistic, kernel_prior, log_loss,
                          logit)

TAG = "stu1"


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"]
    labels = data["labels"].astype(np.float64)
    angles = data["angles"].astype(np.float64)
    folds = read_folds(ids, data["labels"])["fold"]
    probability = np.full(len(ids), np.nan)
    for fold in range(5):
        blob = np.load(OUT / TAG / f"fold{fold}_predictions.npz", allow_pickle=False)
        probability[folds == fold] = blob["probability"]
    image = np.clip(probability, 1e-6, 1 - 1e-6)

    # Path A: exactly what v5_fuse does.
    prior = np.zeros(len(labels))
    exact = np.zeros(len(labels), dtype=bool)
    for fold in range(5):
        outer_valid = folds == fold
        outer_train = ~outer_valid
        p, _, flag = kernel_prior(angles[outer_train], labels[outer_train],
                                  angles[outer_valid], 0.012, 0.1)
        prior[outer_valid] = p
        exact[outer_valid] = flag
        train_angles = angles[outer_train]
        finite = np.isfinite(train_angles)
        column = np.full(len(train_angles), -1, dtype=np.int64)
        column[finite] = np.arange(int(finite.sum()))
        p2, _, flag2 = kernel_prior(train_angles, labels[outer_train], train_angles,
                                    0.012, 0.1, exclude_columns=column)
        index = np.flatnonzero(outer_train)
        prior[index] = p2
        exact[index] = flag2
    prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))
    image_logit = logit(image)

    prediction = np.full(len(labels), np.nan)
    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        features_train = np.column_stack([image_logit[train],
                                          prior_logit[train] * exact[train],
                                          prior_logit[train] * (~exact[train])])
        features_valid = np.column_stack([image_logit[valid],
                                          prior_logit[valid] * exact[valid],
                                          prior_logit[valid] * (~exact[valid])])
        w = fit_logistic(features_train, labels[train])
        prediction[valid] = apply_logistic(features_valid, w)
        print(f"fold {fold}: A={log_loss(labels[valid], prediction[valid]):.6f} "
              f"prior_mean={prior[valid].mean():.4f} exact={int(exact[valid].sum())}")
    print(f"A pooled = {log_loss(labels, prediction):.6f}")

    # How much do the outer-training priors differ from a leave-one-out built on
    # the full data? A large gap means the outer-training features are the leak.
    print(f"prior range on train rows: {prior[folds == 0].min():.3f}..{prior[folds == 0].max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
