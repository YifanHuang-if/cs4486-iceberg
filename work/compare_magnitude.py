"""How large was the machine-row change between the two submissions?

The earlier comparison counted rows that differ but not by how much. If the
change is tiny, an unchanged public score does not prove the machine rows are
unscored, so the magnitude decides whether that conclusion is safe.
"""
from __future__ import annotations

import csv
import io

import numpy as np

from src.v3.train import ROOT


def load(name):
    rows = list(csv.reader(io.open(ROOT / "work/submissions" / f"{name}.csv",
                                   encoding="utf-8", newline="")))
    return [row[0] for row in rows[1:]], np.array([float(row[1]) for row in rows[1:]])


def main() -> int:
    ids, a = load("v5_best")
    _, b = load("v5_best_allrows")
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    angles = test["angles"].astype(np.float64)
    order = {identifier: index for index, identifier in enumerate(test["ids"].astype(str))}
    decimals = np.zeros(len(ids), dtype=np.int64)
    for position, identifier in enumerate(ids):
        value = angles[order[identifier]]
        scaled, count = float(value), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        decimals[position] = count
    machine = decimals > 4
    difference = np.abs(a - b)
    print(f"machine rows: {int(machine.sum())}")
    print(f"  mean |change|: {difference[machine].mean():.4f}")
    print(f"  median |change|: {np.median(difference[machine]):.4f}")
    print(f"  p90 |change|: {np.quantile(difference[machine], 0.9):.4f}")
    print(f"  max |change|: {difference[machine].max():.4f}")
    print(f"  rows changed by >0.05: {int((difference[machine] > 0.05).sum())}")
    print(f"  rows changed by >0.20: {int((difference[machine] > 0.20).sum())}")
    print(f"real rows mean |change|: {difference[~machine].mean():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
