"""Label-free evidence groups based only on each outer training fold."""
from __future__ import annotations
import numpy as np

def evidence_groups(angles, folds, near_threshold=.06):
    """Partition held-out samples by nearest complementary-training angle."""
    if near_threshold <= 0:
        raise ValueError("near_threshold must be positive")
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    f = np.asarray(folds).reshape(-1)
    if len(a) != len(f):
        raise ValueError("angles/folds length mismatch")
    missing = ~np.isfinite(a)
    exact = np.zeros(len(a), dtype=bool)
    near = np.zeros(len(a), dtype=bool)
    far = np.zeros(len(a), dtype=bool)
    for fold in np.unique(f):
        train_angles = a[(f != fold) & np.isfinite(a)]
        held = np.flatnonzero((f == fold) & np.isfinite(a))
        if not len(held):
            continue
        if not len(train_angles):
            far[held] = True
            continue
        distances = np.min(np.abs(a[held, None] - train_angles[None, :]), axis=1)
        exact[held] = distances == 0
        near[held] = (distances > 0) & (distances < near_threshold)
        far[held] = distances >= near_threshold
    if not np.all(sum(mask.astype(int) for mask in (exact, near, far, missing)) == 1):
        raise ValueError("Evidence groups are not a complete partition")
    return {"exact_angle_match": exact, "near_angle_only": near,
            "far_angle": far, "missing_angle": missing}
