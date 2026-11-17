"""Which signal wins where image and angle prior disagree?

Fits the best image weight separately for the agreement and disagreement
regimes. This tests whether a regime-specific fusion weight is better than the
single fixed 0.75 used so far.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def log_loss(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def prior_for(reference_angles, reference_labels, query_angles, sigma=0.012, alpha=0.1):
    a = np.asarray(reference_angles, dtype=np.float64)
    y = np.asarray(reference_labels, dtype=np.float64)
    q = np.asarray(query_angles, dtype=np.float64)
    valid = np.isfinite(a)
    a, y = a[valid], y[valid]
    out = np.full(len(q), 0.5)
    covered = np.zeros(len(q), dtype=bool)
    for start in range(0, len(q), 256):
        stop = min(start + 256, len(q))
        value = q[start:stop]
        distance = np.abs(value[:, None] - a[None, :])
        has_exact = (distance == 0.0).any(axis=1)
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
        weights = np.where(has_exact[:, None], (distance == 0.0).astype(np.float64), weights)
        weights = np.where(np.isfinite(value)[:, None], weights, 0.0)
        mass = weights.sum(axis=1)
        out[start:stop] = (weights @ y + alpha) / (mass + 2 * alpha)
        covered[start:stop] = (mass > 1e-3) & np.isfinite(value)
    return out, covered


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

    image = np.clip(np.load(ROOT / "work/submissions/v1_fast.oof.npy").astype(np.float64), 1e-6, 1 - 1e-6)
    prior = np.full(len(labels), 0.5)
    covered = np.zeros(len(labels), dtype=bool)
    for fold in range(5):
        heldout = folds == fold
        p, mask = prior_for(angles[~heldout], labels[~heldout], angles[heldout])
        prior[heldout] = p
        covered[heldout] = mask
    fused = image.copy()
    fused[covered] = sigmoid(0.75 * logit(image[covered]) + logit(prior[covered]))

    disagree = covered & ((image > 0.5) != (prior > 0.5))
    agree = covered & ~disagree
    report = {}
    for name, mask in (("agree", agree), ("disagree", disagree)):
        entry = {
            "rows": int(mask.sum()),
            "image": round(log_loss(labels[mask], image[mask]), 6),
            "prior": round(log_loss(labels[mask], prior[mask]), 6),
            "fused_075": round(log_loss(labels[mask], fused[mask]), 6),
            "image_accuracy": round(float(np.mean((image[mask] > 0.5) == labels[mask])), 6),
            "prior_accuracy": round(float(np.mean((prior[mask] > 0.5) == labels[mask])), 6),
        }
        best = None
        for weight in np.arange(0.2, 1.61, 0.05):
            candidate = image.copy()
            candidate[mask] = sigmoid(weight * logit(image[mask]) + logit(prior[mask]))
            value = log_loss(labels[mask], candidate[mask])
            if best is None or value < best[1]:
                best = (float(weight), value)
        entry["best_weight"] = round(best[0], 3)
        entry["best_loss"] = round(best[1], 6)
        report[name] = entry
        print(f"{name:9s} rows={entry['rows']:4d} image={entry['image']:.4f} prior={entry['prior']:.4f} "
              f"fused.75={entry['fused_075']:.4f} | acc img={entry['image_accuracy']:.3f} "
              f"prior={entry['prior_accuracy']:.3f} | best w={entry['best_weight']:.2f} -> {entry['best_loss']:.4f}")

    # Regime-specific weights applied together, still with fold-isolated priors.
    combined = image.copy()
    combined[agree] = sigmoid(report["agree"]["best_weight"] * logit(image[agree]) + logit(prior[agree]))
    combined[disagree] = sigmoid(report["disagree"]["best_weight"] * logit(image[disagree]) + logit(prior[disagree]))
    report["combined_regime_weights"] = round(log_loss(labels, combined), 6)
    print(f"\ncombined regime weights (in-sample optimum): {report['combined_regime_weights']:.6f}")
    (ROOT / "work/leak/regimes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
