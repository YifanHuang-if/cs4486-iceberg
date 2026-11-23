"""Cross-validation cost of scaling the angle coefficients.

Scaling trades private for public: the public half carries less angle evidence, so
a smaller angle weight should help there and hurt here. This measures the "here"
side so the leaderboard result can be read correctly.
"""
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

    print(f"{'angle_scale':>12s} {'pooled':>9s} {'exact':>8s} {'near':>8s}")
    near = np.isfinite(prior) & (prior != 0.5) & ~exact
    for scale in (1.0, 0.9, 0.8, 0.65, 0.4, 0.0):
        prediction = np.zeros(len(labels))
        for fold in range(5):
            train = np.flatnonzero(folds != fold)
            valid = np.flatnonzero(folds == fold)
            weights = fit_logistic(features[train], labels[train]).copy()
            weights[1] *= scale
            weights[2] *= scale
            prediction[valid] = apply_logistic(features[valid], weights)
        print(f"{scale:12.2f} {log_loss(labels, prediction):9.6f} "
              f"{log_loss(labels[exact], prediction[exact]):8.6f} "
              f"{log_loss(labels[near], prediction[near]):8.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
