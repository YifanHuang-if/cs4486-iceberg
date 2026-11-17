"""Confirm the all-rows variant differs from the best submission only on machine rows.

If the two files agree on every real-angle row, then any change in the public
score between them is attributable solely to how the 5,000 long-decimal rows are
handled, which makes the comparison a clean diagnostic.
"""
from __future__ import annotations

import csv
import io

import numpy as np

from src.v3.train import ROOT


def load(name):
    rows = list(csv.reader(io.open(ROOT / "work/submissions" / f"{name}.csv",
                                   encoding="utf-8", newline="")))
    ids = [row[0] for row in rows[1:]]
    values = np.array([float(row[1]) for row in rows[1:]])
    return ids, values


def main() -> int:
    ids_a, a = load("v5_best")
    ids_b, b = load("v5_best_allrows")
    assert ids_a == ids_b, "id order differs"
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    angles = test["angles"].astype(np.float64)
    order = {identifier: index for index, identifier in enumerate(test["ids"].astype(str))}
    decimals = np.zeros(len(ids_a), dtype=np.int64)
    for position, identifier in enumerate(ids_a):
        value = angles[order[identifier]]
        scaled, count = float(value), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        decimals[position] = count
    real = decimals <= 4
    changed = np.abs(a - b) > 1e-12
    print(f"rows differing: {int(changed.sum())}")
    print(f"of those, machine rows: {int((changed & ~real).sum())}")
    print(f"of those, real rows:    {int((changed & real).sum())}")
    print(f"real rows unchanged:    {int((~changed & real).sum())}")
    print(f"max |difference| on real rows: {np.abs(a - b)[real].max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
