"""Pin down the exact-group and near-angle statistics under explicit definitions."""
from __future__ import annotations

import numpy as np

from work.v5_greedy import load

TAGS = ["v5_shape", "hp_star", "hp_b", "hp_d", "pt25a", "pt25b", "stu5"]


def main() -> int:
    ids, labels, angles, folds, matrix = load(TAGS)
    observed = np.isfinite(angles) & (angles != 0)

    # --- exact groups: leave-one-out majority agreement (own label excluded)
    hit = 0
    total = 0
    for value in np.unique(angles[observed]):
        group = np.flatnonzero(observed & (angles == value))
        if len(group) < 2:
            continue
        for row in group:
            others = group[group != row]
            majority = 1.0 if labels[others].mean() >= 0.5 else 0.0
            hit += int(labels[row] == majority)
            total += 1
    print(f"exact-group leave-one-out agreement = {hit / total:.3%} over {total} rows")

    # --- nearest neighbour, all rows
    order = np.argsort(angles[observed])
    values = angles[observed]
    rows = np.flatnonzero(observed)
    same = 0
    counted = 0
    distances = []
    for position, row in enumerate(rows):
        left = position - 1
        right = position + 1
        candidates = []
        if left >= 0:
            candidates.append(left)
        if right < len(values):
            candidates.append(right)
        if not candidates:
            continue
        best = min(candidates, key=lambda c: abs(values[c] - values[position]))
        same += int(labels[rows[best]] == labels[row])
        distances.append(abs(values[best] - values[position]))
        counted += 1
    print(f"nearest-angle agreement = {same / counted:.3%} over {counted} rows")
    distances = np.array(distances)
    print(f"  median nearest distance = {np.median(distances):.5f}")
    for threshold in (0.002, 0.012, 0.05):
        mask = distances <= threshold
        if mask.sum():
            # recompute agreement within the threshold
            agree = 0
            counted_t = 0
            for position, row in enumerate(rows):
                left, right = position - 1, position + 1
                candidates = [c for c in (left, right) if 0 <= c < len(values)]
                if not candidates:
                    continue
                best = min(candidates, key=lambda c: abs(values[c] - values[position]))
                distance = abs(values[best] - values[position])
                if distance <= threshold:
                    agree += int(labels[rows[best]] == labels[row])
                    counted_t += 1
            print(f"  within {threshold}: {agree / counted_t:.3%} over {counted_t} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
