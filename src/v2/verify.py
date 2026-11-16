"""Validate a submission file against the official sample submission.

Checks row count, exact id order, and that every probability is finite and inside
the open interval, then reports the distribution.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "raw" / "sample_submission.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with SAMPLE.open(newline="") as handle:
        sample = list(csv.DictReader(handle))
    with args.submission.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    issues = []
    if list(rows[0].keys()) != ["id", "is_iceberg"]:
        issues.append(f"header is {list(rows[0].keys())}")
    if len(rows) != len(sample):
        issues.append(f"row count {len(rows)} != sample {len(sample)}")
    if [row["id"] for row in rows] != [row["id"] for row in sample]:
        issues.append("id order differs from sample_submission")
    values = np.array([float(row["is_iceberg"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        issues.append("non-finite probabilities")
    if values.min() <= 0 or values.max() >= 1:
        issues.append(f"probabilities outside (0,1): [{values.min()}, {values.max()}]")

    report = {
        "submission": str(args.submission),
        "rows": len(rows),
        "ids_match_order": not any("id order" in issue for issue in issues),
        "issues": issues,
        "valid": not issues,
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "median": float(np.median(values)),
        "share_below_0.1": float((values < 0.1).mean()),
        "share_above_0.9": float((values > 0.9).mean()),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
