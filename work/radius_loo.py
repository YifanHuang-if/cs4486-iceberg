"""Leave-one-out study: how far can the angle evidence be stretched?

Exact-angle transfer covers 57% of real test rows. This measures whether a
tolerance window around a row's angle still predicts its label, which would raise
coverage. Every row's prior uses the *other* training rows only.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
train = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
angles = train["angles"]
labels = train["labels"].astype(int)
observed = np.isfinite(angles)
a = angles[observed]
y = labels[observed]
order = np.argsort(a)
a_sorted, y_sorted = a[order], y[order]


def prior_radius(row_index, epsilon, alpha=0.5):
    value = a_sorted[row_index]
    lo = np.searchsorted(a_sorted, value - epsilon, side="left")
    hi = np.searchsorted(a_sorted, value + epsilon, side="right")
    index = np.arange(lo, hi)
    index = index[index != row_index]
    if len(index) == 0:
        return np.nan
    return (y_sorted[index].sum() + alpha) / (len(index) + 2 * alpha)


def evaluate(epsilon, alpha=0.5):
    priors = np.array([prior_radius(i, epsilon, alpha) for i in range(len(a_sorted))])
    usable = np.isfinite(priors)
    p = priors[usable]
    truth = y_sorted[usable]
    confidence = np.maximum(p, 1 - p)
    correct = np.where(p >= 0.5, truth == 1, truth == 0)
    entropy = float(-np.mean(np.where(truth == 1, np.log(np.clip(p, 1e-9, 1)),
                                     np.log(np.clip(1 - p, 1e-9, 1)))))
    return {
        "epsilon": epsilon,
        "coverage_rows": float(usable.mean()),
        "accuracy": float(correct.mean()),
        "accuracy_confident_0.9": float(correct[confidence >= 0.9].mean()) if (confidence >= 0.9).any() else None,
        "share_confident_0.9": float((confidence >= 0.9).mean()),
        "entropy_nats": entropy,
    }


report = [evaluate(e) for e in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)]

# Test-side coverage under each tolerance, split into real (<=4 decimals) rows.
test_angles = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)["angles"]


def decimals(values):
    out = np.zeros(len(values), dtype=int)
    for i, v in enumerate(values):
        scaled, count = float(v), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        out[i] = count
    return out


real = decimals(test_angles) <= 4
coverage = {}
for epsilon in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
    lo = np.searchsorted(a_sorted, test_angles[real] - epsilon, side="left")
    hi = np.searchsorted(a_sorted, test_angles[real] + epsilon, side="right")
    coverage[str(epsilon)] = float(np.mean(hi > lo))
print(json.dumps({"loo": report, "test_real_coverage": coverage, "real_rows": int(real.sum())}, indent=2))
(ROOT / "work/leak/radius_report.json").write_text(
    json.dumps({"loo": report, "test_real_coverage": coverage, "real_rows": int(real.sum())}, indent=2),
    encoding="utf-8")
