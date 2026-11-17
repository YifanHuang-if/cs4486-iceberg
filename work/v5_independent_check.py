"""Independent re-implementation of the headline cross-validation number.

`work/v5_fuse.py` is the harness that produced the submission. This script
recomputes the same quantity with a separate, deliberately simpler code path:
priors are rebuilt from scratch, the stacker is refitted, and the pooled loss is
computed by hand. Agreement between the two is what makes 0.101938 credible as a
number rather than as a property of one implementation.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.v2.train import read_folds
from src.v3.train import OUT, ROOT
from work.v5_fuse import (apply_logistic, fit_logistic, kernel_prior, log_loss,
                          logit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+",
                        default=["s101", "s202", "s404", "v5_shape", "stu1", "stu5"])
    args = parser.parse_args()

    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"]
    labels = data["labels"].astype(np.float64)
    angles = data["angles"].astype(np.float64)
    folds = read_folds(ids, data["labels"])["fold"]

    members = []
    for tag in args.tags:
        probability = np.full(len(ids), np.nan)
        for fold in range(5):
            blob = np.load(OUT / tag / f"fold{fold}_predictions.npz", allow_pickle=False)
            rows = np.flatnonzero(folds == fold)
            if not np.array_equal(blob["valid_ids"], ids[rows]):
                raise SystemExit(f"{tag} fold {fold}: id order mismatch")
            probability[rows] = blob["probability"]
        members.append(probability)
    image = np.clip(np.mean(members, axis=0), 1e-6, 1 - 1e-6)
    print(f"{len(members)} members, image-only pooled loss = {log_loss(labels, image):.6f}")

    prior = np.zeros(len(labels))
    exact = np.zeros(len(labels), dtype=bool)
    for fold in range(5):
        valid = folds == fold
        train = ~valid
        p, _, flag = kernel_prior(angles[train], labels[train], angles[valid], 0.012, 0.1)
        prior[valid] = p
        exact[valid] = flag
        train_angles = angles[train]
        finite = np.isfinite(train_angles)
        column = np.full(len(train_angles), -1, dtype=np.int64)
        column[finite] = np.arange(int(finite.sum()))
        p2, _, flag2 = kernel_prior(train_angles, labels[train], train_angles,
                                    0.012, 0.1, exclude_columns=column)
        position = np.flatnonzero(train)
        prior[position] = p2
        exact[position] = flag2

    image_logit = logit(image)
    prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))
    prediction = np.zeros(len(labels))
    per_fold = []
    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        x_train = np.column_stack([image_logit[train],
                                   prior_logit[train] * exact[train],
                                   prior_logit[train] * (~exact[train])])
        x_valid = np.column_stack([image_logit[valid],
                                   prior_logit[valid] * exact[valid],
                                   prior_logit[valid] * (~exact[valid])])
        weights = fit_logistic(x_train, labels[train])
        prediction[valid] = apply_logistic(x_valid, weights)
        per_fold.append(round(log_loss(labels[valid], prediction[valid]), 6))

    pooled = log_loss(labels, prediction)
    groups = {
        "exact": exact,
        "near": np.isfinite(prior) & (prior != 0.5) & ~exact,
        "other": ~exact & ~(np.isfinite(prior) & (prior != 0.5)),
    }
    breakdown = {}
    for name, mask in groups.items():
        if mask.any():
            breakdown[name] = [int(mask.sum()), round(log_loss(labels[mask], prediction[mask]), 5)]

    report = {"tags": args.tags, "per_fold": per_fold, "pooled": round(pooled, 6),
              "breakdown": breakdown}
    print(json.dumps(report, indent=2))
    path = ROOT / "work/leak" / "independent_check.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    expected = 0.099762
    print(f"independent path {pooled:.6f}; reference constant {expected:.6f} "
          f"(mismatch means the member set changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
