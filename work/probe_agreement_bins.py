"""Nearest-angle agreement as a function of distance, in fine bins.

If agreement jumps at very small non-zero distances then nearby angles still
identify the same pass and a distance-aware prior is justified. If it is flat
then nearby angles are different passes and no weighting can help.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    angles = data["angles"].astype(np.float64)
    labels = data["labels"].astype(np.float64)
    fold_of = {}
    with io.open(ROOT / "data/splits/folds.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fold_of[row["id"]] = int(row["fold"])
    folds = np.array([fold_of[i] for i in ids], dtype=np.int64)

    distance = np.full(len(labels), np.nan)
    neighbour = np.full(len(labels), np.nan)
    for fold in range(5):
        heldout = np.flatnonzero(folds == fold)
        reference = np.flatnonzero((folds != fold) & np.isfinite(angles))
        ref_angles = angles[reference]
        ref_labels = labels[reference]
        order = np.argsort(ref_angles)
        sorted_angles = ref_angles[order]
        sorted_labels = ref_labels[order]
        for row in heldout:
            value = angles[row]
            if not np.isfinite(value):
                continue
            same = np.flatnonzero(sorted_angles == value)
            if len(same):
                distance[row] = 0.0
                neighbour[row] = sorted_labels[same].mean()
                continue
            position = np.searchsorted(sorted_angles, value)
            candidates = [p for p in (position - 1, position) if 0 <= p < len(sorted_angles)]
            if not candidates:
                continue
            best = min(candidates, key=lambda p: abs(sorted_angles[p] - value))
            distance[row] = abs(sorted_angles[best] - value)
            neighbour[row] = sorted_labels[best]

    edges = [0.0, 1e-12, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 6e-2]
    report = {"bins": []}
    for low, high in zip(edges[:-1], edges[1:]):
        if low == 0.0:
            mask = distance == 0.0
        else:
            mask = (distance >= low) & (distance < high)
        if mask.sum() < 3:
            continue
        agreement = float(np.mean(np.round(neighbour[mask]) == labels[mask]))
        entry = {"low": low, "high": high, "rows": int(mask.sum()),
                 "agreement": round(agreement, 4)}
        report["bins"].append(entry)
        print(f"[{low:g}, {high:g})  rows={entry['rows']:4d}  agreement={agreement:.3f}")

    (ROOT / "work/leak/agreement_by_distance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
