"""Non-linear level-2 combiner: boosted trees instead of logistic regression.

The fusion layer currently assumes a linear relationship in log-odds. That may be
too rigid: the right trust in the angle prior plausibly depends on how confident
the image model is and whether the two signals agree. A small boosted-tree
stacker can express such interactions.

Feature construction and the fold-isolated prior are copied from
`work/v5_greedy.py`, which is calibrated against `v5_fuse` to the last digit, so
only the combiner changes.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.v3.train import ROOT
from work.train_tree import bin_features, sigmoid, train_gbdt
from work.v5_fuse import log_loss, logit
from work.v5_greedy import build_prior, complete_tags, load

SIGMA = 0.012
ALPHA = 0.1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+",
                        default=["s101", "s202", "s404", "v5_shape", "stu1", "stu5"])
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    ids, labels, angles, folds, matrix = load(args.tags)
    prior, exact = build_prior(angles, labels, folds)
    image = np.clip(matrix.mean(axis=0), 1e-6, 1 - 1e-6)
    image_logit = logit(image)
    prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))

    features = np.column_stack([
        image_logit,
        prior_logit * exact,
        prior_logit * (~exact),
        exact.astype(np.float64),
        np.abs(image_logit),
        np.where(exact, 0.0, np.abs(prior_logit)),
    ])
    names = ["image_logit", "prior*exact", "prior*near", "exact", "|image_logit|", "|prior|*near"]

    report = {"tags": args.tags, "trees": args.trees, "depth": args.depth,
              "lr": args.lr, "features": names, "folds": []}
    logistic_like = np.zeros(len(labels))
    tree_like = np.zeros(len(labels))
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        binned_train, binned_valid = bin_features(features[train], features[valid])
        forest, base = train_gbdt(binned_train, labels[train], trees=args.trees,
                                  depth=args.depth, lr=args.lr, seed=700 + fold)
        score = np.full(len(valid), base)
        for tree in forest:
            score += args.lr * tree.predict(binned_valid)
        tree_like[valid] = sigmoid(score)
        entry = {"fold": fold,
                 "tree_loss": round(log_loss(labels[valid], tree_like[valid]), 6)}
        report["folds"].append(entry)
        print(f"fold {fold}: tree stacker {entry['tree_loss']:.6f}")

    report["pooled_tree"] = round(log_loss(labels, tree_like), 6)
    report["reference_logistic"] = 0.101938
    print(f"\npooled tree stacker = {report['pooled_tree']:.6f} "
          f"(logistic reference 0.101938)")
    path = ROOT / "work/leak/stack_gbdt.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
