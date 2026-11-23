"""Verify the specific claims used in the report and the slides."""
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

    total = log_loss(labels, prediction) * len(labels)
    observed = np.isfinite(angles) & (angles != 0)

    # disagreement between image and prior on rows that have prior evidence
    evidence = observed & (prior != 0.5)
    image_side = image > 0.5
    prior_side = prior > 0.5
    disagree = evidence & (image_side != prior_side)
    loss_disagree = log_loss(labels[disagree], prediction[disagree])
    print(f"disagreement rows = {int(disagree.sum())}, "
          f"share of OOF loss = {loss_disagree * disagree.sum() / total:.1%}")
    print(f"  image accuracy on them = {(image_side[disagree] == labels[disagree]).mean():.1%}")
    print(f"  prior accuracy on them = {(prior_side[disagree] == labels[disagree]).mean():.1%}")

    # exact-angle group purity (unseen-group definition: all rows sharing that angle)
    pure_rows = 0
    total_rows = 0
    for value in np.unique(angles[observed]):
        group = np.flatnonzero(observed & (angles == value))
        if len(group) < 2:
            continue
        majority = 1.0 if labels[group].mean() >= 0.5 else 0.0
        pure_rows += int((labels[group] == majority).sum())
        total_rows += len(group)
    print(f"exact-group agreement = {pure_rows / total_rows:.3%} over {total_rows} rows")

    # nearest-neighbour agreement
    reference = np.sort(angles[observed])
    agree = 0
    counted = 0
    for row in np.flatnonzero(observed):
        value = angles[row]
        others = np.flatnonzero(observed & (angles != value))
        if len(others) == 0:
            continue
        nearest = others[np.argmin(np.abs(angles[others] - value))]
        agree += int(labels[nearest] == labels[row])
        counted += 1
    print(f"nearest-angle agreement = {agree / counted:.3%} over {counted} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
