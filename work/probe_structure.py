"""Structural probe of the raw test file order and identifier space.

The winning team's account says the decisive step was understanding the data
rather than the network. Several structural questions have not been checked yet:

1. Are the real-angle rows contiguous in the test file order, or interleaved
   with the machine-generated ones?
2. Does angle precision drift with row position?
3. Do identifier prefixes carry any signal in the training set?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def decimals_of(values):
    out = np.zeros(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        scaled, count = float(value), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        out[index] = count
    return out


def main() -> int:
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    train = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    test_ids = test["ids"].astype(str)
    train_ids = train["ids"].astype(str)
    labels = train["labels"].astype(np.float64)
    test_decimals = decimals_of(test["angles"])
    real = test_decimals <= 4

    positions = np.flatnonzero(real)
    contiguous = bool(len(positions) and positions[-1] - positions[0] == len(positions) - 1)
    runs = int(1 + np.sum(np.diff(positions) > 1)) if len(positions) else 0
    report = {
        "test_rows": int(len(real)),
        "real_rows": int(real.sum()),
        "machine_rows": int((~real).sum()),
        "real_rows_contiguous": contiguous,
        "number_of_real_runs": runs,
        "first_real_position": int(positions[0]) if len(positions) else None,
        "last_real_position": int(positions[-1]) if len(positions) else None,
        "precision_by_quarter": [],
    }
    for quarter in range(4):
        low = quarter * len(real) // 4
        high = (quarter + 1) * len(real) // 4
        report["precision_by_quarter"].append({
            "quarter": quarter,
            "mean_decimals": round(float(test_decimals[low:high].mean()), 3),
            "real_share": round(float(real[low:high].mean()), 4),
        })
        print(f"quarter {quarter}: mean decimals {test_decimals[low:high].mean():.3f} "
              f"real share {real[low:high].mean():.3f}")

    # Identifier prefixes in the training set.
    prefixes = [identifier[0] for identifier in train_ids]
    unique = sorted(set(prefixes))
    per_prefix = {}
    for prefix in unique:
        mask = np.array([p == prefix for p in prefixes])
        if mask.sum() >= 20:
            per_prefix[prefix] = {"rows": int(mask.sum()),
                                  "iceberg_rate": round(float(labels[mask].mean()), 4)}
    report["train_id_prefix"] = per_prefix
    print(f"train id first-character groups (n>=20): {len(per_prefix)}")
    rates = [v["iceberg_rate"] for v in per_prefix.values()]
    if rates:
        print(f"  iceberg rate across prefixes: {min(rates):.3f}..{max(rates):.3f}")

    # Is the machine/real split stable across the file, or clustered at an end?
    half = len(real) // 2
    report["real_share_first_half"] = round(float(real[:half].mean()), 4)
    report["real_share_second_half"] = round(float(real[half:].mean()), 4)
    print(f"real share first half {report['real_share_first_half']:.3f} "
          f"second half {report['real_share_second_half']:.3f}")

    (ROOT / "work/leak/structure_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "precision_by_quarter"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
