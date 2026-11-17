"""Nested greedy ensemble selection over every available member.

The first-place team combined many models with greedy blending. The risk is that
greedy selection on out-of-fold predictions overfits those same predictions, so
here the whole selection loop runs inside each outer training split and the
resulting weights are applied untouched to the outer validation fold. The
reported number is therefore a genuine five-fold estimate of the whole
"select then fuse" procedure.

Members are added with replacement, so a member selected twice carries twice the
weight. The objective is the same group-wise logistic stacking used by the final
submission, so selection and deployment share one recipe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.train import OUT, ROOT
from work.v5_fuse import (apply_logistic, fit_logistic, kernel_prior, log_loss,
                          logit, sigmoid)

TAGS = ["s101", "s202", "s404", "v50_shape", "v5_shape", "v5_shape_glcm",
        "stu1", "stu2", "stu3", "stu4", "stu5", "stu9", "stu_ls", "v5_tree",
        "d50a", "d50b", "g50"]


def available_tags():
    found = []
    for tag in TAGS:
        directory = OUT / tag
        if all((directory / f"fold{k}_predictions.npz").exists() for k in range(5)):
            found.append(tag)
    extra = sorted(p.name for p in OUT.iterdir()
                   if p.is_dir() and p.name.startswith("stu_b")
                   and all((p / f"fold{k}_predictions.npz").exists() for k in range(5)))
    return found + [t for t in extra if t not in found]


def load_matrix(tags):
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
            raise ValueError(f"{tag} is incomplete")
        matrix[index] = np.clip(probability, 1e-6, 1 - 1e-6)
    return labels, angles, folds, matrix


def build_prior(angles, labels, folds, rows, sigma, alpha):
    """Fold-isolated prior for the given rows: reference is always the other folds."""
    prior = np.zeros(len(labels))
    exact = np.zeros(len(labels), dtype=bool)
    for fold in range(5):
        valid = folds == fold
        train = ~valid
        p, _, flag = kernel_prior(angles[train], labels[train], angles[valid], sigma, alpha)
        prior[valid] = p
        exact[valid] = flag
    return prior[rows], exact[rows]


def features_for(weights, matrix, prior_logit, exact, rows):
    image = weights @ matrix[:, rows]
    image_logit = logit(np.clip(image, 1e-6, 1 - 1e-6))
    return np.column_stack([image_logit,
                            prior_logit[rows] * exact[rows],
                            prior_logit[rows] * (~exact[rows])])


def greedy(matrix, prior_logit, exact, labels, rows, steps, verbose=False):
    """Forward selection with replacement, scored inside the given rows only."""
    counts = np.zeros(matrix.shape[0])
    chosen = []
    best_history = []
    current = None
    for _ in range(steps):
        best = (np.inf, None)
        for index in range(matrix.shape[0]):
            trial = counts.copy()
            trial[index] += 1.0
            weights = trial / trial.sum()
            features = features_for(weights, matrix, prior_logit, exact, rows)
            w = fit_logistic(features, labels[rows])
            loss = log_loss(labels[rows], apply_logistic(features, w))
            if loss < best[0]:
                best = (loss, index)
        if current is not None and best[0] > current - 1e-9:
            break
        counts[best[1]] += 1.0
        chosen.append(int(best[1]))
        current = best[0]
        best_history.append(round(best[0], 6))
        if verbose:
            print(f"    +{best[1]}: in-sample {best[0]:.6f}")
    return counts, chosen, best_history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.012)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--name", default="greedy")
    parser.add_argument("--only", nargs="+", default=None,
                        help="restrict to these tags (debugging the harness)")
    args = parser.parse_args()

    tags = available_tags()
    if args.only:
        tags = args.only
    print(f"{len(tags)} complete members: {tags}")
    labels, angles, folds, matrix = load_matrix(tags)
    report = {"tags": tags, "steps": args.steps, "folds": [], "per_tag_uniform": {}}

    uniform = np.ones(len(tags)) / len(tags)
    for fold in range(5):
        rows_all = np.arange(len(labels))
        prior = np.zeros(len(labels))
        exact = np.zeros(len(labels), dtype=bool)
        train = folds != fold
        p, _, flag = kernel_prior(angles[train], labels[train], angles[folds == fold],
                                  args.sigma, args.alpha)
        prior[folds == fold] = p
        exact[folds == fold] = flag
        p2, _, _ = kernel_prior(angles[folds == fold], labels[folds == fold],
                                angles[folds == fold], args.sigma, args.alpha)
        # For outer-training rows use a leave-one-out prior inside the outer
        # training folds so validation labels never enter the search.
        inner_prior = np.zeros(len(labels))
        inner_exact = np.zeros(len(labels), dtype=bool)
        index = np.flatnonzero(train)
        finite = np.isfinite(angles[index])
        column = np.full(len(index), -1, dtype=np.int64)
        column[finite] = np.arange(int(finite.sum()))
        pi, _, fi = kernel_prior(angles[index], labels[index], angles[index],
                                 args.sigma, args.alpha, exclude_columns=column)
        inner_prior[index] = pi
        inner_exact[index] = fi
        prior[train] = inner_prior[train]
        exact[train] = inner_exact[train]

        prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))
        rows_train = np.flatnonzero(train)
        rows_valid = np.flatnonzero(~train)
        counts, chosen, history = greedy(matrix, prior_logit, exact, labels,
                                         rows_train, args.steps, verbose=True)
        weights = counts / counts.sum()
        weights_valid = features_for(weights, matrix, prior_logit, exact, rows_valid)
        w = fit_logistic(features_for(weights, matrix, prior_logit, exact, rows_train), labels[rows_train])
        prediction = apply_logistic(weights_valid, w)
        uniform_features = features_for(uniform, matrix, prior_logit, exact, rows_valid)
        uniform_w = fit_logistic(features_for(uniform, matrix, prior_logit, exact, rows_train), labels[rows_train])
        uniform_prediction = apply_logistic(uniform_features, uniform_w)
        entry = {
            "fold": fold,
            "selected": [tags[i] for i in chosen],
            "weights": {tags[i]: round(float(weights[i]), 4) for i in np.flatnonzero(counts)},
            "history": history,
            "valid_loss_greedy": round(log_loss(labels[rows_valid], prediction), 6),
            "valid_loss_uniform": round(log_loss(labels[rows_valid], uniform_prediction), 6),
        }
        report["folds"].append(entry)
        print(f"fold {fold}: greedy={entry['valid_loss_greedy']:.6f} "
              f"uniform={entry['valid_loss_uniform']:.6f} | {entry['weights']}", flush=True)

    pooled_greedy = np.mean([f["valid_loss_greedy"] for f in report["folds"]])
    pooled_uniform = np.mean([f["valid_loss_uniform"] for f in report["folds"]])
    report["pooled_greedy"] = round(float(pooled_greedy), 6)
    report["pooled_uniform"] = round(float(pooled_uniform), 6)
    print(f"\npooled (mean of fold losses) greedy={pooled_greedy:.6f} uniform={pooled_uniform:.6f}")
    path = ROOT / "work/leak" / f"{args.name}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
