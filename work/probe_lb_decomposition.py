"""What would the leaderboard look like if the near rows improved?

Decomposes the measured private score by angle-evidence group on the real test
rows, then solves for the near-row loss that 0.083 would require. This replaces
the earlier CV-based projection, which was pessimistic because the test side has
higher exact-angle coverage and a stronger prior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.fusion import prior
from src.v3.train import ROOT
from work.v5_fuse import kernel_prior

MEASURED_PRIVATE = 0.08944
TARGET = 0.083
CV_EXACT = 0.0223
# CV five-fold, pooled, from work/leak/fuse_6final.json
CV_NEAR = 0.2038


def main() -> int:
    train = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    angles_train = train["angles"].astype(np.float64)
    labels_train = train["labels"].astype(np.float64)
    test_angles = test["angles"].astype(np.float64)

    decimals = np.zeros(len(test_angles), dtype=np.int64)
    for index, value in enumerate(test_angles):
        scaled, count = float(value), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        decimals[index] = count
    real = decimals <= 4

    prior_test, covered, exact = kernel_prior(angles_train, labels_train, test_angles, 0.012, 0.1)
    covered = covered & real
    exact = exact & real
    near = covered & ~exact

    # Exact-angle groups among real rows, which is what the pooling variant touches.
    group_rows = 0
    group_uneXact = 0
    sizes = []
    for value in np.unique(test_angles[real]):
        rows = np.flatnonzero(real & (test_angles == value))
        if len(rows) < 2:
            continue
        group_rows += len(rows)
        group_uneXact += int((~exact[rows]).sum())
        sizes.append(len(rows))

    rows = {"real": int(real.sum()), "exact": int(exact.sum()),
            "near": int(near.sum()), "groups_touched": group_rows,
            "groups_touched_non_exact": group_uneXact}
    share_exact = rows["exact"] / rows["real"]
    share_near = rows["near"] / rows["real"]

    # Back out the near-row loss implied by the measured private score.
    implied_near = (MEASURED_PRIVATE - share_exact * CV_EXACT) / share_near
    required_near = (TARGET - share_exact * CV_EXACT) / share_near

    report = {
        "test_rows": rows,
        "share_exact": round(share_exact, 4),
        "share_near": round(share_near, 4),
        "measured_private": MEASURED_PRIVATE,
        "cv_exact_loss": CV_EXACT,
        "cv_near_loss": CV_NEAR,
        "implied_test_near_loss": round(implied_near, 4),
        "required_near_loss_for_target": round(required_near, 4),
        "relative_improvement_needed": round((implied_near - required_near) / implied_near, 4),
        "pooling_group_mean_size": round(float(np.mean(sizes)), 3) if sizes else None,
        "pooling_group_max": int(max(sizes)) if sizes else None,
        "pooling_share_of_near_rows": round(group_uneXact / max(rows["near"], 1), 4),
    }
    print(json.dumps(report, indent=2))
    (ROOT / "work/leak/lb_decomposition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
