"""Build the final submission: ensemble, angle prior, fitted group stacking.

The level-2 combiner is fitted on all 1,604 training rows using out-of-fold image
probabilities and a leave-one-out angle prior, then applied unchanged to the test
rows. Test rows use all 1,604 training labels as reference evidence, which is
allowed because no test label exists anywhere in the pipeline.

Outputs a CSV with a header plus 8,424 rows in sample-submission order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.train import ROOT
from work.v5_fuse import (apply_logistic, fit_logistic, kernel_prior, load_tags,
                          log_loss, logit, sigmoid)

CACHE = ROOT / "work" / "predictions"


def training_features(tags, sigma, alpha):
    """Level-2 features for the training rows, built without self labels."""
    ids, labels, angles, folds, stack, seen = load_tags(tags)
    image = np.nanmean(stack, axis=0)
    rows = np.isfinite(image)
    prior = np.zeros(len(labels))
    exact = np.zeros(len(labels), dtype=bool)
    covered_flag = np.zeros(len(labels), dtype=bool)
    for fold in range(5):
        outer_valid = folds == fold
        outer_train = ~outer_valid
        p, covered, flag = kernel_prior(angles[outer_train], labels[outer_train],
                                        angles[outer_valid], sigma, alpha)
        prior[outer_valid] = p
        exact[outer_valid] = flag
        covered_flag[outer_valid] = covered
        train_angles = angles[outer_train]
        finite = np.isfinite(train_angles)
        column = np.full(len(train_angles), -1, dtype=np.int64)
        column[finite] = np.arange(int(finite.sum()))
        p2, covered2, flag2 = kernel_prior(train_angles, labels[outer_train], train_angles,
                                           sigma, alpha, exclude_columns=column)
        index = np.flatnonzero(outer_train)
        prior[index] = p2
        exact[index] = flag2
        covered_flag[index] = covered2
    image_logit = logit(np.clip(image, 1e-6, 1 - 1e-6))
    prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))
    features = np.column_stack([
        image_logit,
        prior_logit * exact,
        prior_logit * (~exact),
    ])
    return labels, rows, features, dict(image=image, prior=prior, exact=exact,
                                        covered=covered_flag, folds=folds, ids=ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--sigma", type=float, default=0.012)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=0.001)
    parser.add_argument("--real-only", action="store_true", default=True)
    parser.add_argument("--no-real-only", dest="real_only", action="store_false")
    parser.add_argument("--pool-weight", type=float, default=0.0,
                        help="blend each test row toward its exact-angle group mean")
    parser.add_argument("--pool-non-exact-only", dest="pool_non_exact_only",
                        action="store_true",
                        help="pool only rows without a training exact match")
    args = parser.parse_args()

    labels, rows, features, info = training_features(args.tags, args.sigma, args.alpha)
    weights = fit_logistic(features[rows], labels[rows], l2=args.l2)
    oof = apply_logistic(features[rows], weights)
    oof_loss = log_loss(labels[rows], oof)
    print(f"level-2 fitted on {int(rows.sum())} rows, in-sample OOF loss={oof_loss:.6f}")
    print(f"coefficients (image, prior_exact, prior_near, intercept) = {np.round(weights, 4).tolist()}")

    # Test side.
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    test_ids = test["ids"].astype(str)
    test_angles = test["angles"].astype(np.float64)
    sample = list(csv.reader(io.open(ROOT / "data/raw/sample_submission.csv", encoding="utf-8")))
    order = [row[0] for row in sample[1:]]
    position = {identifier: index for index, identifier in enumerate(test_ids)}
    if set(order) != set(test_ids):
        raise SystemExit("sample submission ids do not match the test cache")

    per_tag = []
    for tag in args.tags:
        folds = []
        for fold in range(5):
            path = CACHE / f"{tag}_fold{fold}.npy"
            if not path.exists():
                raise SystemExit(f"missing test predictions {path}; run work/v5_predict.py first")
            folds.append(np.load(path))
        per_tag.append(np.mean(folds, axis=0))
        print(f"{tag}: test mean={per_tag[-1].mean():.4f}")
    image_test = np.mean(per_tag, axis=0)

    train_cache = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    prior_test, covered_test, exact_test = kernel_prior(
        train_cache["angles"], labels, test_angles, args.sigma, args.alpha)

    # The competition states that machine-generated images are not scored, and
    # those rows carry random many-decimal angles. Only rows whose angle looks
    # like a real acquisition are allowed to receive angle evidence.
    decimals = np.zeros(len(test_angles), dtype=np.int64)
    for index, value in enumerate(test_angles):
        scaled, count = float(value), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        decimals[index] = count
    real = decimals <= 4
    if args.real_only:
        covered_test = covered_test & real
        exact_test = exact_test & real
        prior_test = np.where(covered_test, prior_test, 0.5)
    print(f"real-angle rows: {int(real.sum())} of {len(real)}; "
          f"angle evidence applied to {int(covered_test.sum())}")
    image_logit = logit(np.clip(image_test, 1e-6, 1 - 1e-6))
    prior_logit = logit(np.clip(prior_test, 1e-6, 1 - 1e-6))
    test_features = np.column_stack([
        image_logit,
        prior_logit * exact_test,
        prior_logit * (~exact_test),
    ])
    probability = np.clip(apply_logistic(test_features, weights), args.clip, 1 - args.clip)
    print(f"test: mean={probability.mean():.4f} min={probability.min():.6f} max={probability.max():.6f} "
          f"exact_share={exact_test.mean():.4f} covered_share={covered_test.mean():.4f}")

    if args.pool_weight > 0:
        # Rows sharing an exact incidence angle belong to the same acquisition
        # pass, so their labels agree almost always. Blending each member toward
        # its group mean reduces variance for groups that have no training match.
        pooled = probability.copy()
        poolable = 0
        for value in np.unique(test_angles[real]):
            group = np.flatnonzero(real & (test_angles == value))
            if len(group) < 2:
                continue
            # The group mean uses every real member, but with
            # --pool-non-exact-only only the unmatched rows are modified.
            target = group
            if args.pool_non_exact_only:
                target = group[~exact_test[group]]
            if len(target) == 0:
                continue
            poolable += len(target)
            group_mean = probability[group].mean()
            pooled[target] = (1 - args.pool_weight) * probability[target] + args.pool_weight * group_mean
        probability = np.clip(pooled, args.clip, 1 - args.clip)
        print(f"pooled {poolable} rows in exact-angle groups at weight {args.pool_weight}")

    destination = ROOT / "work" / "submissions" / f"{args.name}.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "is_iceberg"])
        for identifier in order:
            writer.writerow([identifier, f"{probability[position[identifier]]:.10f}"])

    rows_written = sum(1 for _ in io.open(destination, encoding="utf-8")) - 1
    record = {
        "name": args.name,
        "tags": args.tags,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "l2": args.l2,
        "clip": args.clip,
        "pool_weight": args.pool_weight,
        "pool_non_exact_only": bool(args.pool_non_exact_only),
        "level2_rows": int(rows.sum()),
        "level2_in_sample_oof": round(oof_loss, 6),
        "coefficients": np.round(weights, 6).tolist(),
        "test_rows": int(rows_written),
        "test_mean": round(float(probability.mean()), 6),
        "test_exact_share": round(float(exact_test.mean()), 6),
        "test_covered_share": round(float(covered_test.mean()), 6),
        "per_tag_test_mean": {tag: round(float(value.mean()), 6) for tag, value in zip(args.tags, per_tag)},
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "note": ("Development CV submission. The level-2 fit uses training labels only. "
                 "No Kaggle score is implied."),
    }
    (ROOT / "work/submissions" / f"{args.name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
