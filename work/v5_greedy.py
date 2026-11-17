"""Nested greedy ensemble selection, built on the verified fusion code path.

The prior and feature construction here is a line-for-line copy of the block in
`work/debug_stack.py`, which reproduces the `v5_fuse` pooled score exactly for a
single member. A `--calibrate` run checks that property before any multi-member
number is trusted.

Greedy selection adds members one at a time with replacement; each candidate is
scored by refitting the group-wise logistic stacker on the outer training folds
only. The chosen weights are then applied untouched to the outer validation fold,
so the reported score covers the whole select-then-fuse procedure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.train import OUT, ROOT
from work.v5_fuse import (apply_logistic, fit_logistic, kernel_prior, log_loss,
                          logit)

SIGMA = 0.012
ALPHA = 0.1


def complete_tags() -> list[str]:
    tags = []
    for directory in sorted(OUT.iterdir()):
        if not directory.is_dir():
            continue
        if all((directory / f"fold{k}_predictions.npz").exists() for k in range(5)):
            blob = np.load(directory / "fold0_predictions.npz", allow_pickle=False)
            if "probability" in blob.files:
                tags.append(directory.name)
    return tags


def load(tags):
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"]
    labels = data["labels"].astype(np.float64)
    angles = data["angles"].astype(np.float64)
    folds = read_folds(ids, data["labels"])["fold"]
    matrix = np.empty((len(tags), len(ids)))
    for index, tag in enumerate(tags):
        probability = np.full(len(ids), np.nan)
        for fold in range(5):
            blob = np.load(OUT / tag / f"fold{fold}_predictions.npz", allow_pickle=False)
            probability[folds == fold] = blob["probability"]
        if not np.isfinite(probability).all():
            raise ValueError(f"{tag} incomplete")
        matrix[index] = np.clip(probability, 1e-6, 1 - 1e-6)
    return ids, labels, angles, folds, matrix


def build_prior(angles, labels, folds):
    """Identical to the verified block: fold-isolated, leave-one-out inside the outer train."""
    prior = np.zeros(len(labels))
    exact = np.zeros(len(labels), dtype=bool)
    for fold in range(5):
        outer_valid = folds == fold
        outer_train = ~outer_valid
        p, _, flag = kernel_prior(angles[outer_train], labels[outer_train],
                                  angles[outer_valid], SIGMA, ALPHA)
        prior[outer_valid] = p
        exact[outer_valid] = flag
        index = np.flatnonzero(outer_train)
        train_angles = angles[outer_train]
        finite = np.isfinite(train_angles)
        column = np.full(len(train_angles), -1, dtype=np.int64)
        column[finite] = np.arange(int(finite.sum()))
        p2, _, flag2 = kernel_prior(train_angles, labels[outer_train], train_angles,
                                    SIGMA, ALPHA, exclude_columns=column)
        prior[index] = p2
        exact[index] = flag2
    return prior, exact


def features(matrix, weights, prior_logit, exact, rows):
    image = weights @ matrix[:, rows]
    image_logit = logit(np.clip(image, 1e-6, 1 - 1e-6))
    return np.column_stack([image_logit,
                            prior_logit[rows] * exact[rows],
                            prior_logit[rows] * (~exact[rows])])


def score(matrix, weights, prior_logit, exact, labels, fit_rows, score_rows):
    w = fit_logistic(features(matrix, weights, prior_logit, exact, fit_rows), labels[fit_rows])
    x = features(matrix, weights, prior_logit, exact, score_rows)
    return log_loss(labels[score_rows], apply_logistic(x, w))


def greedy(matrix, prior_logit, exact, labels, rows, steps):
    counts = np.zeros(matrix.shape[0])
    history = []
    current = np.inf
    for _ in range(steps):
        best = (np.inf, -1)
        for index in range(matrix.shape[0]):
            trial = counts.copy()
            trial[index] += 1.0
            value = score(matrix, trial / trial.sum(), prior_logit, exact, labels, rows, rows)
            if value < best[0]:
                best = (value, index)
        if best[0] > current - 1e-7:
            break
        counts[best[1]] += 1.0
        current = best[0]
        history.append(round(float(current), 6))
    return counts, history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--name", default="greedy2")
    args = parser.parse_args()

    tags = args.only if args.only else complete_tags()
    print(f"{len(tags)} members: {tags}")
    ids, labels, angles, folds, matrix = load(tags)
    prior, exact = build_prior(angles, labels, folds)
    prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))

    uniform = np.ones(len(matrix)) / len(matrix)
    report = {"tags": tags, "steps": args.steps, "folds": []}
    pooled_greedy, pooled_uniform = [], []
    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        counts, history = greedy(matrix, prior_logit, exact, labels, train, args.steps)
        weights = counts / counts.sum()
        greedy_loss = score(matrix, weights, prior_logit, exact, labels, train, valid)
        uniform_loss = score(matrix, uniform, prior_logit, exact, labels, train, valid)
        pooled_greedy.append((greedy_loss, len(valid)))
        pooled_uniform.append((uniform_loss, len(valid)))
        entry = {"fold": fold, "greedy": round(greedy_loss, 6), "uniform": round(uniform_loss, 6),
                 "weights": {tags[i]: round(float(weights[i]), 4) for i in np.flatnonzero(counts)}}
        report["folds"].append(entry)
        print(f"fold {fold}: greedy={greedy_loss:.6f} uniform={uniform_loss:.6f} {entry['weights']}")

    def pooled(pairs):
        return float(sum(v * n for v, n in pairs) / sum(n for _, n in pairs))

    report["pooled_greedy"] = round(pooled(pooled_greedy), 6)
    report["pooled_uniform"] = round(pooled(pooled_uniform), 6)
    print(f"\npooled greedy={report['pooled_greedy']:.6f}  uniform={report['pooled_uniform']:.6f}")
    path = ROOT / "work/leak" / f"{args.name}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
