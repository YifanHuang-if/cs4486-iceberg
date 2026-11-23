"""What does softening the final probability cost on cross validation?

The leaderboard splits move in opposite directions as the model sharpens, which
suggests the public half punishes confident mistakes harder. Softening trades a
little accuracy on easy rows for less damage on confidently wrong ones. This
measures the cost before spending a submission on it.
"""
from __future__ import annotations

import numpy as np

from src.v3.train import ROOT
from work.v5_greedy import build_prior, load
from work.v5_fuse import apply_logistic, fit_logistic, log_loss, logit

TAGS = ["v5_shape", "hp_star", "hp_b", "hp_d", "e40a", "pt25b", "stu5"]


def main() -> int:
    ids, labels, angles, folds, matrix = load(TAGS)
    prior, exact = build_prior(angles, labels, folds)
    image = np.clip(matrix.mean(axis=0), 1e-6, 1 - 1e-6)
    features = np.column_stack([logit(image),
                                logit(np.clip(prior, 1e-6, 1 - 1e-6)) * exact,
                                logit(np.clip(prior, 1e-6, 1 - 1e-6)) * (~exact)])
    prediction = np.zeros(len(labels))
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        prediction[valid] = apply_logistic(features[valid], fit_logistic(features[train], labels[train]))

    print(f"{'temperature':>12s} {'pooled':>9s} {'exact':>8s} {'near':>8s} {'mean_abs_shift':>15s}")
    near = np.isfinite(prior) & (prior != 0.5) & ~exact
    for temperature in (1.0, 1.1, 1.2, 1.3, 1.5, 2.0):
        logits = logit(np.clip(prediction, 1e-9, 1 - 1e-9)) / temperature
        softened = 1.0 / (1.0 + np.exp(-logits))
        shift = float(np.abs(softened - prediction).mean())
        print(f"{temperature:12.1f} {log_loss(labels, softened):9.6f} "
              f"{log_loss(labels[exact], softened[exact]):8.6f} "
              f"{log_loss(labels[near], softened[near]):8.6f} {shift:15.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
