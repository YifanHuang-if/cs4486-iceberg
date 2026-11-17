"""Compare the angle-evidence density at test time with the cross-validation setup.

The kernel width (sigma 0.012) was chosen on cross validation, where each query
row sees four fifths of the labels. On the test side every query sees all 1,604
labels, and the test angles are distributed differently from the training angles.
If the test side is much denser, the same sigma may be too wide or too narrow.

This measures, for the rows that actually need the prior (no exact match), how
much kernel mass and how many contributing neighbours they get on each side.
"""
from __future__ import annotations

import json

import numpy as np

from src.v2.train import read_folds
from src.v3.train import ROOT

SIGMA = 0.012


def kernel_stats(query_angles, reference_angles, reference_labels):
    query = np.asarray(query_angles, dtype=np.float64)
    reference = np.asarray(reference_angles, dtype=np.float64)
    labels = np.asarray(reference_labels, dtype=np.float64)
    valid = np.isfinite(reference)
    reference, labels = reference[valid], labels[valid]
    mass, neighbours, purity = [], [], []
    for value in query:
        if not np.isfinite(value):
            continue
        difference = np.abs(reference - value)
        if np.any(difference == 0.0):
            continue
        weight = np.exp(-0.5 * (difference / SIGMA) ** 2)
        total = weight.sum()
        if total <= 1e-3:
            continue
        mass.append(total)
        neighbours.append(int((difference < 3 * SIGMA).sum()))
        purity.append(weight @ labels / total)
    mass = np.array(mass)
    neighbours = np.array(neighbours)
    purity = np.array(purity)
    return {
        "rows": int(len(mass)),
        "mass_median": round(float(np.median(mass)), 4),
        "mass_p90": round(float(np.quantile(mass, 0.9)), 4),
        "neighbours_median": int(np.median(neighbours)),
        "neighbours_p90": int(np.quantile(neighbours, 0.9)),
        "purity_median_distance_to_half": round(float(np.median(np.abs(purity - 0.5))), 4),
    }


def main() -> int:
    train = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    angles_train = train["angles"].astype(np.float64)
    labels_train = train["labels"].astype(np.float64)
    folds = read_folds(train["ids"], train["labels"])["fold"]
    test_angles = test["angles"].astype(np.float64)

    decimals = np.zeros(len(test_angles), dtype=np.int64)
    for index, value in enumerate(test_angles):
        scaled, count = float(value), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        decimals[index] = count
    real = decimals <= 4

    report = {"sigma": SIGMA, "cv": {}, "test": {}}
    cv_mass = {"mass": [], "neighbours": [], "purity": []}
    for fold in range(5):
        train_mask = folds != fold
        query = angles_train[folds == fold]
        reference = angles_train[train_mask]
        labels = labels_train[train_mask]
        stats = kernel_stats(query, reference, labels)
        report["cv"][f"fold{fold}"] = stats
        print(f"CV fold {fold}: {stats}")
    report["test"] = kernel_stats(test_angles[real], angles_train, labels_train)
    print(f"TEST: {report['test']}")

    cv_mass_median = np.mean([report["cv"][f"fold{k}"]["mass_median"] for k in range(5)])
    report["cv_mass_median_mean"] = round(float(cv_mass_median), 4)
    report["test_over_cv_mass"] = round(report["test"]["mass_median"] / cv_mass_median, 4)
    cv_nb = np.mean([report["cv"][f"fold{k}"]["neighbours_median"] for k in range(5)])
    report["test_over_cv_neighbours"] = round(report["test"]["neighbours_median"] / cv_nb, 4)
    print(f"\ntest/CV kernel mass ratio {report['test_over_cv_mass']}, "
          f"neighbour ratio {report['test_over_cv_neighbours']}")
    (ROOT / "work/leak/prior_density.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
