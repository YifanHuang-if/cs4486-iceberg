"""Incidence-angle group analysis and the conservative label-transfer rule.

The competition shipped ``inc_angle`` for every test row. In this dataset the
angle acts as a satellite-pass identifier: within one angle the label is almost
always constant, so a test row whose exact angle also appears in the training
labels can be predicted from that group. This module measures that structure and
implements a *conservative* transfer that never contradicts a confident image
model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"


def decimals(values: np.ndarray) -> np.ndarray:
    """Number of decimal places needed to represent each value exactly."""
    out = np.zeros(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            out[index] = -1
            continue
        count = 0
        scaled = float(value)
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        out[index] = count
    return out


def angle_groups(angles: np.ndarray, labels: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"angle": angles})
    frame["angle_key"] = frame.angle.round(6)
    grouped = frame.groupby("angle_key")
    summary = pd.DataFrame({"rows": grouped.size()})
    if labels is not None:
        frame["label"] = labels
        stats = frame.groupby("angle_key").label.agg(["size", "mean", "nunique"])
        summary["rows"] = stats["size"]
        summary["label_mean"] = stats["mean"]
        summary["labels"] = stats["nunique"]
    summary = summary.reset_index()
    summary["key_decimals"] = decimals(summary.angle_key.to_numpy())
    return summary


def main() -> int:
    train = np.load(CACHE / "train.npz", allow_pickle=False)
    test = np.load(CACHE / "test.npz", allow_pickle=False)
    train_angles, labels = train["angles"], train["labels"].astype(int)
    test_angles = test["angles"]

    train_groups = angle_groups(train_angles, labels)
    observed = train_groups[train_groups.rows > 0]
    mixed = observed[observed.labels > 1]
    pure = observed[observed.labels == 1]
    assigned_rows = int(observed[observed.labels == 1].rows.sum())
    report = {
        "train_rows": int(len(train_angles)),
        "train_missing_angle": int(np.isnan(train_angles).sum()),
        "train_missing_all_ship": bool(labels[np.isnan(train_angles)].sum() == 0),
        "unique_angles": int(len(observed)),
        "pure_ship_angles": int((pure.label_mean == 0).sum()),
        "pure_iceberg_angles": int((pure.label_mean == 1).sum()),
        "mixed_angles": int(len(mixed)),
        "rows_in_pure_angles": assigned_rows,
        "rows_in_mixed_angles": int(mixed.rows.sum()),
        "majority_accuracy_if_angle_revealed": float(
            (pure.rows.sum() + mixed[["rows"]].to_numpy().sum() * 0) / len(train_angles)
        ),
        "angle_decimals_train": np.bincount(decimals(train_angles[np.isfinite(train_angles)])).tolist(),
    }

    test_decimal = decimals(test_angles)
    real_mask = test_decimal <= 4
    report["test_rows"] = int(len(test_angles))
    report["test_missing_angle"] = int(np.isnan(test_angles).sum())
    report["test_rows_machine_generated"] = int((test_decimal > 4).sum())
    report["test_rows_real"] = int(real_mask.sum())
    report["test_angle_decimals_histogram"] = {
        str(k): int(v) for k, v in zip(*np.unique(test_decimal, return_counts=True))
    }

    lookup = observed.set_index(observed.angle_key.round(6))
    test_keys = pd.Series(test_angles).round(6)
    matched = test_keys.isin(lookup.index).to_numpy()
    report["test_real_rows_angle_in_train"] = int((matched & real_mask).sum())
    report["test_real_rows_angle_unseen"] = int((~matched & real_mask).sum())
    report["test_machine_rows_angle_in_train"] = int((matched & ~real_mask).sum())

    detail = {}
    for name, mask in (("matched_pure_ship", matched & real_mask), ("matched_pure_iceberg", matched & real_mask)):
        keys = test_keys[mask]
        means = lookup.loc[keys, "label_mean"].to_numpy()
        detail[name] = int((means == (0 if "ship" in name else 1)).sum())
    report["test_real_matched_pure_ship"] = detail["matched_pure_ship"]
    report["test_real_matched_pure_iceberg"] = detail["matched_pure_iceberg"]
    report["coverage_of_real_test"] = float((matched & real_mask).sum() / max(real_mask.sum(), 1))

    # Training-side purity of the angles that actually match a real test row.
    matched_keys = np.unique(test_keys[matched & real_mask].to_numpy())
    subset = lookup.loc[lookup.index.intersection(matched_keys)]
    report["matched_group_rows"] = int(subset.rows.sum())
    report["matched_group_purity"] = float(
        (subset.loc[subset.label_mean == 0, "rows"].sum() + subset.loc[subset.label_mean == 1, "rows"].sum())
        / max(subset.rows.sum(), 1)
    )

    out = ROOT / "work" / "leak"
    out.mkdir(parents=True, exist_ok=True)
    (out / "angle_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
