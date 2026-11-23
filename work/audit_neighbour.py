"""Final audit numbers: neighbour agreement overall and by distance band."""
from __future__ import annotations

import numpy as np

from work.v5_greedy import load


def main() -> int:
    ids, labels, angles, folds, matrix = load(
        ["v5_shape", "hp_star", "hp_b", "hp_d", "pt25a", "pt25b", "stu5"])
    observed = np.isfinite(angles) & (angles != 0)
    rows = np.flatnonzero(observed)
    values = angles[rows]
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_rows = rows[order]

    records = []
    for index in range(len(sorted_values)):
        candidates = []
        if index > 0:
            candidates.append(index - 1)
        if index + 1 < len(sorted_values):
            candidates.append(index + 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda c: abs(sorted_values[c] - sorted_values[index]))
        distance = abs(sorted_values[best] - sorted_values[index])
        same = labels[sorted_rows[best]] == labels[sorted_rows[index]]
        records.append((distance, bool(same)))

    distances = np.array([r[0] for r in records])
    same = np.array([r[1] for r in records])
    print(f"overall nearest-angle agreement = {same.mean():.2%} over {len(same)} rows")
    print(f"  exact duplicates (distance 0) = {int((distances == 0).sum())} rows, "
          f"agreement {same[distances == 0].mean():.2%}")

    bands = [(0, 0, "0"), (1e-9, 5e-4, "(0, 5e-4]"), (5e-4, 2e-3, "(5e-4, 2e-3]"),
             (2e-3, 6e-2, "(2e-3, 6e-2]"), (6e-2, 1e9, "> 6e-2")]
    for low, high, name in bands:
        mask = (distances >= low) & (distances <= high)
        if mask.sum():
            print(f"  band {name:14s} rows={int(mask.sum()):4d} agreement={same[mask].mean():.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
