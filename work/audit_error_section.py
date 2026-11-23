"""Numbers for the error section: disagreement definitions and nearby agreement."""
from __future__ import annotations

import numpy as np

from work.v5_fuse import apply_logistic, fit_logistic, log_loss, logit
from work.v5_greedy import build_prior, load


def main() -> int:
    ids, labels, angles, folds, matrix = load(
        ["v5_shape", "hp_star", "hp_b", "hp_d", "pt25a", "pt25b", "stu5"])
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
    evidence = observed & (prior != 0.5)

    # definition 1: the two sides point to different classes
    d1 = evidence & ((image > 0.5) != (prior > 0.5))
    loss1 = log_loss(labels[d1], prediction[d1])
    print(f"[sign] rows={int(d1.sum())} share={loss1 * d1.sum() / total:.1%} "
          f"image_acc={(image[d1] > 0.5).eq if False else ((image[d1] > 0.5) == labels[d1]).mean():.1%} "
          f"prior_acc={((prior[d1] > 0.5) == labels[d1]).mean():.1%}")

    # definition 2: both sides confident but opposite (>0.6 / <0.4)
    d2 = evidence & (np.abs(image - 0.5) > 0.1) & (np.abs(prior - 0.5) > 0.1) & \
         ((image > 0.5) != (prior > 0.5))
    if d2.sum():
        loss2 = log_loss(labels[d2], prediction[d2])
        print(f"[conf] rows={int(d2.sum())} share={loss2 * d2.sum() / total:.1%} "
              f"image_acc={((image[d2] > 0.5) == labels[d2]).mean():.1%} "
              f"prior_acc={((prior[d2] > 0.5) == labels[d2]).mean():.1%}")

    # nearby (no exact match) angle agreement
    rows = np.flatnonzero(observed)
    values = angles[rows]
    order = np.argsort(values)
    sv, sr = values[order], rows[order]
    near_same, near_total = 0, 0
    for i in range(len(sv)):
        cand = [c for c in (i - 1, i + 1) if 0 <= c < len(sv)]
        if not cand:
            continue
        best = min(cand, key=lambda c: abs(sv[c] - sv[i]))
        if abs(sv[best] - sv[i]) > 0:
            near_total += 1
            near_same += int(labels[sr[best]] == labels[sr[i]])
    print(f"nearby-only (no exact match) agreement = {near_same / near_total:.2%} over {near_total} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
