"""Validate the final submission file against the competition sample."""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "v5_mix6"
    path = ROOT / "work/submissions" / f"{name}.csv"
    rows = list(csv.reader(io.open(path, encoding="utf-8", newline="")))
    sample = list(csv.reader(io.open(ROOT / "data/raw/sample_submission.csv", encoding="utf-8", newline="")))
    header, body = rows[0], rows[1:]
    values = np.array([float(row[1]) for row in body])
    checks = {
        "header": [c.strip().lower() for c in header] == ["id", "is_iceberg"],
        "rows_8424": len(body) == 8424,
        "id_order": [row[0] for row in body] == [row[0] for row in sample[1:]],
        "range_0_1": bool(np.all((values >= 0) & (values <= 1)) and np.all(np.isfinite(values))),
    }
    print(f"{name}: " + "  ".join(f"{k}={v}" for k, v in checks.items()))
    print(f"  min={values.min():.6f} max={values.max():.6f} mean={values.mean():.4f} "
          f"share_gt_0.9={float((values > 0.9).mean()):.4f} share_lt_0.1={float((values < 0.1).mean()):.4f}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
