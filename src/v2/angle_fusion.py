"""Incidence-angle prior estimation, leave-one-out calibration and Bayes fusion.

A test row whose exact angle also occurs in the training labels is not only
"allowed" to use them: within an angle group the label is almost constant, so the
group itself carries a probability. This module estimates that probability with
shrinkage, measures it honestly by leave-one-out on the training rows, and fuses
it with an image model in log-odds space.

Label convention: ``is_iceberg`` = 0 for ship, 1 for iceberg.
Numpy only, so it runs without the heavier analysis stack.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"


def keys(angles: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(angles, dtype=np.float64), 6)


def group_table(train_angles: np.ndarray, labels: np.ndarray, alpha: float = 0.5):
    """Return (angle keys, smoothed P(iceberg)) for every observed training angle."""
    angle_key = keys(train_angles)
    labels = np.asarray(labels, dtype=np.float64)
    observed = np.isfinite(angle_key)
    angle_key, labels = angle_key[observed], labels[observed]
    order = np.argsort(angle_key, kind="stable")
    angle_key, labels = angle_key[order], labels[order]
    unique, starts, counts = np.unique(angle_key, return_index=True, return_counts=True)
    icebergs = np.add.reduceat(labels, starts)
    p_iceberg = (icebergs + alpha) / (counts + 2 * alpha)
    return unique, p_iceberg


def lookup(angles: np.ndarray, unique: np.ndarray, p_iceberg: np.ndarray) -> np.ndarray:
    """Map angles to their group probability; NaN where no group is observed."""
    angle_key = keys(angles)
    position = np.searchsorted(unique, np.nan_to_num(angle_key, nan=-1e9))
    position = np.clip(position, 0, len(unique) - 1)
    matched = np.isfinite(angle_key) & (unique[position] == angle_key)
    out = np.full(len(angle_key), np.nan)
    out[matched] = p_iceberg[position[matched]]
    return out


def loo_prior(train_angles: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Each row's prior uses the *other* rows of its angle group, never itself."""
    angle_key = keys(train_angles)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(angle_key, kind="stable")
    sorted_angles = angle_key[order]
    unique, starts, counts = np.unique(sorted_angles, return_index=True, return_counts=True)
    prior = np.full(len(labels), np.nan)
    for value, start, count in zip(unique, starts, counts):
        if not np.isfinite(value) or count < 2:
            continue
        index = order[start:start + count]
        total = float(labels[index].sum())
        prior[index] = (total - labels[index] + alpha) / (count - 1 + 2 * alpha)
    return prior


def bayes_fusion(model_probability: np.ndarray, prior_iceberg: np.ndarray, weight: float = 1.0,
                 floor: float = 1e-6) -> np.ndarray:
    """Combine image-model odds with angle-group odds; rows without a prior are untouched."""
    p = np.clip(np.asarray(model_probability, dtype=np.float64), floor, 1 - floor)
    prior = np.asarray(prior_iceberg, dtype=np.float64)
    logit = lambda q: np.log(np.clip(q, floor, 1 - floor) / (1 - np.clip(q, floor, 1 - floor)))
    combined = weight * logit(p) + np.where(np.isfinite(prior), logit(prior), 0.0)
    return 1.0 / (1.0 + np.exp(-combined))


def log_loss(labels, probability, floor: float = 1e-7) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), floor, 1 - floor)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def main() -> int:
    train = np.load(CACHE / "train.npz", allow_pickle=False)
    labels = train["labels"].astype(int)
    observed = np.isfinite(train["angles"])
    unique, p_iceberg = group_table(train["angles"], labels)
    report = {}
    for name, mask in (("all_rows", np.ones(len(labels), dtype=bool)), ("observed_angles_only", observed)):
        prior = loo_prior(train["angles"][mask], labels[mask])
        usable = np.isfinite(prior)
        confidence = np.maximum(prior[usable], 1 - prior[usable])
        correct = np.where(prior[usable] >= 0.5, labels[mask][usable] == 1, labels[mask][usable] == 0)
        buckets = []
        for low, high in ((0.5, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)):
            inside = (confidence >= low) & (confidence < high)
            if inside.any():
                buckets.append({"bucket": f"[{low},{high})", "rows": int(inside.sum()),
                                "accuracy": float(correct[inside].mean()),
                                "mean_confidence": float(confidence[inside].mean())})
        entropy = float(-np.mean(np.where(labels[mask][usable] == 1, np.log(np.clip(prior[usable], 1e-9, 1)),
                                          np.log(np.clip(1 - prior[usable], 1e-9, 1)))))
        report[name] = {
            "rows_with_group_mate": int(usable.sum()),
            "rows_without_group_mate": int((~usable).sum()),
            "accuracy_when_confident_above_0.9": float(correct[confidence >= 0.9].mean())
            if (confidence >= 0.9).any() else None,
            "row_entropy_nats": entropy,
            "buckets": buckets,
        }
    report["groups"] = {
        "count": int(len(unique)),
        "sizes": {str(k): int(v) for k, v in zip(*np.unique(
            np.bincount(np.searchsorted(unique, keys(train["angles"])[observed])), return_counts=True)) if k > 0},
    }
    out = ROOT / "work" / "leak"
    out.mkdir(parents=True, exist_ok=True)
    (out / "loo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
