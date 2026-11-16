"""Build pseudo-labels for the unlabelled test images from a blended prediction.

Only confident rows are used, and their soft targets are clipped away from 0/1 so
a wrong pseudo-label cannot dominate the loss. The 5,000 machine-generated test
rows are kept on purpose: they depict the same scenes under a different
simulator, so they act as extra augmentation for the convolutional models.

Usage:
    python -m src.v2.pseudo --submission work/submissions/blend.csv --threshold 0.03 --out work/pseudo/p1.npz
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--soft-floor", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "pseudo" / "p1.npz")
    parser.add_argument("--exclude-machine", action="store_true")
    args = parser.parse_args()

    test = np.load(CACHE / "test.npz", allow_pickle=False)
    ids = test["ids"]
    with args.submission.open(newline="") as handle:
        mapping = {row["id"]: float(row["is_iceberg"]) for row in csv.DictReader(handle)}
    if set(mapping) != set(ids.tolist()):
        raise ValueError("submission ids do not match the test cache")
    probability = np.array([mapping[i] for i in ids], dtype=np.float64)

    angles = test["angles"]
    decimals = np.zeros(len(angles), dtype=int)
    for index, value in enumerate(angles):
        scaled = float(value)
        count = 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        decimals[index] = count
    machine = decimals > 4

    confident = (probability <= args.threshold) | (probability >= 1 - args.threshold)
    if args.exclude_machine:
        confident &= ~machine
    rows = np.flatnonzero(confident)
    labels = np.where(probability[rows] >= 0.5, 1.0, 0.0)
    soft = np.where(labels > 0.5, np.maximum(probability[rows], 1 - args.soft_floor),
                    np.minimum(probability[rows], args.soft_floor))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, rows=rows.astype(np.int64), labels=soft.astype(np.float32))
    summary = {
        "source": str(args.submission), "threshold": args.threshold,
        "rows": int(len(rows)), "share_of_test": float(len(rows) / len(ids)),
        "share_machine_rows_selected": float(machine[rows].mean()) if len(rows) else 0.0,
        "iceberg_share": float(labels.mean()) if len(rows) else 0.0,
        "out": str(args.out),
    }
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
