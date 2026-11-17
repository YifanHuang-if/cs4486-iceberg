"""Find a large, label-pure subgroup of the training data.

The first-place team's account says their decisive step was an unsupervised
grouping of the data followed by retraining on one subgroup, because the other
subgroup was entirely one class and biased the model. That idea has not been
tried here.

This searches for the same structure in our data with our own definition: group
rows by exact incidence angle, keep groups that are label-pure, and measure how
much of the training set sits in a pure-iceberg or pure-ship block. The result
decides what a "specialist" model should be trained on.
"""
from __future__ import annotations

import json

import numpy as np

from src.v2.train import read_folds
from src.v3.train import ROOT


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    angles = data["angles"].astype(np.float64)
    labels = data["labels"].astype(np.float64)
    folds = read_folds(data["ids"], data["labels"])["fold"]
    observed = np.isfinite(angles)

    report = {"rows": int(len(labels)), "observed": int(observed.sum())}
    pure_iceberg, pure_ship, mixed, singleton = [], [], [], []
    for value in np.unique(angles[observed]):
        rows = np.flatnonzero(observed & (angles == value))
        rate = labels[rows].mean()
        if len(rows) == 1:
            singleton.append(rows)
        elif rate == 1.0:
            pure_iceberg.append(rows)
        elif rate == 0.0:
            pure_ship.append(rows)
        else:
            mixed.append(rows)

    def summarise(groups, name):
        rows = np.concatenate(groups) if groups else np.array([], dtype=int)
        entry = {"groups": len(groups), "rows": int(len(rows)),
                 "share_of_all": round(len(rows) / len(labels), 4)}
        if len(rows):
            entry["label_rate"] = round(float(labels[rows].mean()), 4)
            entry["folds"] = {int(k): int(v) for k, v in
                              zip(*np.unique(folds[rows], return_counts=True))}
        report[name] = entry
        print(f"{name:16s} groups={entry['groups']:4d} rows={entry['rows']:4d} "
              f"share={entry['share_of_all']:.3f}")
        return rows

    iceberg_rows = summarise(pure_iceberg, "pure_iceberg")
    ship_rows = summarise(pure_ship, "pure_ship")
    summarise(mixed, "mixed")
    singleton_rows = summarise(singleton, "singleton")

    # What the specialist would see: everything except the pure-iceberg block.
    complement = np.ones(len(labels), dtype=bool)
    complement[iceberg_rows] = False
    report["complement_of_pure_iceberg"] = {
        "rows": int(complement.sum()),
        "label_rate": round(float(labels[complement].mean()), 4),
        "folds": {int(k): int(v) for k, v in zip(*np.unique(folds[complement], return_counts=True))},
    }
    print(f"complement rows={complement.sum()} "
          f"label_rate={labels[complement].mean():.3f}")

    # And a specialist trained only on the purity-free middle.
    both = np.zeros(len(labels), dtype=bool)
    both[ship_rows] = True
    both[iceberg_rows] = True
    report["pure_only"] = {"rows": int(both.sum()),
                           "label_rate": round(float(labels[both].mean()), 4)}
    print(f"pure-only rows={both.sum()} label_rate={labels[both].mean():.3f}")

    (ROOT / "work/leak/pure_groups.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
