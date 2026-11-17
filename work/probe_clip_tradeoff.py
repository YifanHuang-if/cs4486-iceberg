"""What does clipping cost on cross validation?

The submission clips probabilities to a floor. A tighter floor limits the damage
of a confidently wrong row but also caps the reward of a confidently right one.
This measures the trade-off on the honest five-fold predictions so that a
leaderboard result can be interpreted.
"""
from __future__ import annotations

import numpy as np

from src.v3.train import ROOT
from work.v5_fuse import apply_logistic, fit_logistic, log_loss
from work.v5_greedy import build_prior, load

from work.v5_greedy import SIGMA, ALPHA

TAGS = ["v5_shape", "hp_star", "hp_b", "hp_d", "pt25a", "pt25b", "stu5"]


def main() -> int:
    ids, labels, angles, folds, matrix = load(TAGS)
    prior, exact = build_prior(angles, labels, folds)
    from work.v5_fuse import logit
    image = np.clip(matrix.mean(axis=0), 1e-6, 1 - 1e-6)
    image_logit = logit(image)
    prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))
    features = np.column_stack([image_logit,
                                prior_logit * exact,
                                prior_logit * (~exact)])
    prediction = np.zeros(len(labels))
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        weights = fit_logistic(features[train], labels[train])
        prediction[valid] = apply_logistic(features[valid], weights)

    print(f"{'floor':>8s} {'pooled':>10s} {'exact':>9s} {'near':>9s} "
          f"{'rows_at_floor':>14s}")
    near = np.isfinite(prior) & (prior != 0.5) & ~exact
    for floor in (1e-6, 0.001, 0.005, 0.01, 0.02, 0.05):
        clipped = np.clip(prediction, floor, 1 - floor)
        at_floor = int(np.sum((clipped <= floor + 1e-12) | (clipped >= 1 - floor - 1e-12)))
        print(f"{floor:8.4f} {log_loss(labels, clipped):10.6f} "
              f"{log_loss(labels[exact], clipped[exact]):9.6f} "
              f"{log_loss(labels[near], clipped[near]):9.6f} {at_floor:14d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
