"""Angle evidence beyond exact matches: tolerance windows and two-tier priors.

Exact-angle transfer covers 57% of the scored test rows. A tolerance window around
the query angle raises coverage above 90% but with weaker evidence, so the two are
combined: an exact group when one exists, otherwise the window. Everything is
measured by leave-one-out on the training rows before it is applied to test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.v2.angle_fusion import bayes_fusion, log_loss

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"


class AngleIndex:
    """Sorted training angles with prefix sums for O(log n) window priors."""

    def __init__(self, angles: np.ndarray, labels: np.ndarray):
        self.angles = np.asarray(angles, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=np.float64)
        observed = np.isfinite(self.angles)
        self.angles = self.angles[observed]
        self.labels = self.labels[observed]
        self.original_index = np.flatnonzero(observed)
        order = np.argsort(self.angles, kind="stable")
        self.angles = self.angles[order]
        self.labels = self.labels[order]
        self.original_index = self.original_index[order]
        self.prefix = np.concatenate([[0.0], np.cumsum(self.labels)])
        self.position = np.full(len(observed), -1, dtype=np.int64)
        self.position[self.original_index] = np.arange(len(self.original_index))

    def window(self, values: np.ndarray, epsilon: float):
        values = np.asarray(values, dtype=np.float64)
        lo = np.searchsorted(self.angles, values - epsilon, side="left")
        hi = np.searchsorted(self.angles, values + epsilon, side="right")
        return lo, hi

    def prior(self, values: np.ndarray, epsilon: float, alpha: float, exclude: np.ndarray | None = None):
        lo, hi = self.window(values, epsilon)
        total = self.prefix[hi] - self.prefix[lo]
        count = hi - lo
        if exclude is not None:
            own = self.labels[exclude]
            total = total - own
            count = count - 1
        prior = (total + alpha) / (count + 2 * alpha)
        prior = np.where(count > 0, prior, np.nan)
        return prior

    def exact_prior(self, values: np.ndarray, alpha: float, exclude: np.ndarray | None = None):
        return self.prior(values, 0.0, alpha, exclude=exclude)

    def kernel_prior(self, values: np.ndarray, sigma: float, alpha: float, exclude: np.ndarray | None = None):
        """Gaussian-weighted label average over the training angles."""
        values = np.asarray(values, dtype=np.float64)
        difference = values[:, None] - self.angles[None, :]
        weight = np.exp(-0.5 * (difference / sigma) ** 2)
        if exclude is not None:
            weight[np.arange(len(values)), exclude] = 0.0
        total = weight @ self.labels
        mass = weight.sum(axis=1)
        prior = (total + alpha) / (mass + 2 * alpha)
        return np.where(mass > 1e-3, prior, np.nan)


def evaluate(index: AngleIndex, oof: np.ndarray, train_labels: np.ndarray, model_rows: np.ndarray):
    """Fuse the image OOF with each angle scheme on the rows used by the model."""
    truth = train_labels[model_rows]
    model = oof[model_rows]
    base = log_loss(truth, model)
    position = index.position[model_rows]
    has_angle = np.flatnonzero(position >= 0)
    plans, priors = [], {}
    for alpha in (0.1, 0.25, 0.5, 1.0):
        exact = np.full(len(model_rows), np.nan)
        exact[has_angle] = index.exact_prior(index.angles[position[has_angle]], alpha=alpha,
                                             exclude=position[has_angle])
        for epsilon in (0.005, 0.01, 0.02, 0.03, 0.05):
            window = np.full(len(model_rows), np.nan)
            window[has_angle] = index.prior(index.angles[position[has_angle]], epsilon, alpha,
                                            exclude=position[has_angle])
            priors[f"window|{epsilon}|{alpha}"] = window
            priors[f"two_tier|{epsilon}|{alpha}"] = np.where(np.isfinite(exact), exact, window)
        for sigma in (0.01, 0.02, 0.05):
            kernel = np.full(len(model_rows), np.nan)
            kernel[has_angle] = index.kernel_prior(index.angles[position[has_angle]], sigma, alpha,
                                                   exclude=position[has_angle])
            priors[f"kernel|{sigma}|{alpha}"] = np.where(np.isfinite(exact), exact, kernel)
    for key, prior in priors.items():
        scheme, spread, alpha = key.split("|")
        covered = np.isfinite(prior)
        for weight in (0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
            fused = bayes_fusion(model, prior, weight)
            plans.append({"scheme": scheme, "spread": float(spread), "alpha": float(alpha), "weight": weight,
                          "coverage": float(covered.mean()), "loss": log_loss(truth, fused),
                          "loss_covered": log_loss(truth[covered], fused[covered]),
                          "loss_uncovered": log_loss(truth[~covered], fused[~covered]) if (~covered).any() else None})
    plans.sort(key=lambda row: row["loss"])
    return {"base": base, "best": plans[0], "top": plans[:10]}


def main() -> int:
    train = np.load(CACHE / "train.npz", allow_pickle=False)
    labels = train["labels"].astype(np.float64)
    index = AngleIndex(train["angles"], labels)
    rows = np.arange(len(labels))
    oof = np.load(ROOT / "work/submissions/fast_s0.oof.npy")
    report = evaluate(index, oof, labels, rows)

    # Test-side coverage under each tolerance, real rows only (<=4 decimals).
    test = np.load(CACHE / "test.npz", allow_pickle=False)
    angles = test["angles"]
    decimals = np.zeros(len(angles), dtype=int)
    for i, v in enumerate(angles):
        scaled, count = float(v), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        decimals[i] = count
    real = decimals <= 4
    coverage = {}
    for epsilon in (0.0, 0.005, 0.01, 0.02, 0.05):
        lo, hi = index.window(angles[real], epsilon)
        coverage[str(epsilon)] = float(np.mean(hi > lo))
    report["test_real_coverage"] = coverage

    out = ROOT / "work" / "leak" / "angle_schemes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
