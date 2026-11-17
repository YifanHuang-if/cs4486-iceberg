"""Honest grid over the angle-fusion hyper-parameters, reported per subgroup.

Uses fold-isolated priors: a held-out row only ever sees labels from its
training folds. The image probabilities come from an existing OOF file, so the
comparison isolates the fusion layer and does not retrain anything.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def log_loss(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def prior_for(reference_angles, reference_labels, query_angles, sigma, alpha, top_k=None):
    """Two-tier prior: exact group when it exists, else Gaussian kernel (or kNN)."""
    a = np.asarray(reference_angles, dtype=np.float64)
    y = np.asarray(reference_labels, dtype=np.float64)
    q = np.asarray(query_angles, dtype=np.float64)
    valid = np.isfinite(a)
    a, y = a[valid], y[valid]
    out = np.full(len(q), 0.5)
    covered = np.zeros(len(q), dtype=bool)
    exact_flag = np.zeros(len(q), dtype=bool)
    if not len(a):
        return out, covered, exact_flag

    for start in range(0, len(q), 256):
        stop = min(start + 256, len(q))
        value = q[start:stop]
        distance = np.abs(value[:, None] - a[None, :])
        exact = distance == 0.0
        has_exact = exact.any(axis=1)
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
        if top_k is not None and top_k < len(a):
            keep = np.zeros_like(weights)
            for row in range(distance.shape[0]):
                order = np.argsort(distance[row])[:top_k]
                keep[row, order] = weights[row, order]
            weights = keep
        weights = np.where(has_exact[:, None], exact.astype(np.float64), weights)
        finite = np.isfinite(value)[:, None]
        weights = np.where(finite, weights, 0.0)
        mass = weights.sum(axis=1)
        out[start:stop] = (weights @ y + alpha) / (mass + 2 * alpha)
        covered[start:stop] = (mass > 1e-3) & np.isfinite(value)
        exact_flag[start:stop] = has_exact & np.isfinite(value)
    return out, covered, exact_flag


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=True)
    ids = data["ids"].astype(str)
    angles = data["angles"].astype(np.float64)
    labels = data["labels"].astype(np.float64)
    fold_of = {}
    with io.open(ROOT / "data/splits/folds.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fold_of[row["id"]] = int(row["fold"])
    folds = np.array([fold_of[i] for i in ids], dtype=np.int64)

    oof_path = ROOT / sys.argv[1] if len(sys.argv) > 1 else ROOT / "work/submissions/v1_fast.oof.npy"
    model = np.load(oof_path).astype(np.float64)
    model = np.clip(model, 1e-6, 1 - 1e-6)
    assert len(model) == len(labels)

    # Fold-isolated priors for every (sigma, alpha, top_k) combination once.
    sigmas = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.04]
    alphas = [0.03, 0.1, 0.3, 1.0]
    top_ks = [None, 6, 12]

    baseline = log_loss(labels, model)
    print(f"image-only OOF: {baseline:.6f}")

    results = []
    for sigma in sigmas:
        for alpha in alphas:
            for top_k in top_ks:
                prior = np.full(len(labels), 0.5)
                covered = np.zeros(len(labels), dtype=bool)
                exact_flag = np.zeros(len(labels), dtype=bool)
                for fold in np.unique(folds):
                    heldout = folds == fold
                    reference = ~heldout
                    p, mask, flag = prior_for(angles[reference], labels[reference],
                                              angles[heldout], sigma, alpha, top_k)
                    prior[heldout] = p
                    covered[heldout] = mask
                    exact_flag[heldout] = flag
                for weight in (0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
                    fused = model.copy()
                    fused[covered] = sigmoid(weight * logit(model[covered]) + logit(prior[covered]))
                    results.append({
                        "sigma": sigma, "alpha": alpha, "top_k": top_k, "weight": weight,
                        "loss": log_loss(labels, fused),
                        "covered": int(covered.sum()), "exact": int(exact_flag.sum()),
                    })
    results.sort(key=lambda r: r["loss"])
    print("\ntop 12 fusion settings:")
    for row in results[:12]:
        print(f"  loss={row['loss']:.6f}  sigma={row['sigma']} alpha={row['alpha']} "
              f"k={row['top_k']} w={row['weight']} covered={row['covered']} exact={row['exact']}")

    best = results[0]
    prior = np.full(len(labels), 0.5)
    covered = np.zeros(len(labels), dtype=bool)
    exact_flag = np.zeros(len(labels), dtype=bool)
    for fold in np.unique(folds):
        heldout = folds == fold
        p, mask, flag = prior_for(angles[~heldout], labels[~heldout], angles[heldout],
                                  best["sigma"], best["alpha"], best["top_k"])
        prior[heldout] = p
        covered[heldout] = mask
        exact_flag[heldout] = flag
    fused = model.copy()
    fused[covered] = sigmoid(best["weight"] * logit(model[covered]) + logit(prior[covered]))

    groups = {
        "exact": exact_flag,
        "near": covered & ~exact_flag,
        "far": ~covered & np.isfinite(angles),
        "missing": ~np.isfinite(angles),
    }
    def row_losses(truth, probability):
        probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
        return -(truth * np.log(probability) + (1 - truth) * np.log1p(-probability))

    total = row_losses(labels, fused).sum()
    breakdown = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        breakdown[name] = {
            "rows": int(mask.sum()),
            "image": round(log_loss(labels[mask], model[mask]), 6),
            "fused": round(log_loss(labels[mask], fused[mask]), 6),
            "share": round(float(row_losses(labels[mask], fused[mask]).sum() / total), 4),
        }
    print(f"\nbest fusion {best['loss']:.6f} with sigma={best['sigma']} alpha={best['alpha']} "
          f"k={best['top_k']} w={best['weight']}")
    for name, row in breakdown.items():
        print(f"  {name:8s} rows={row['rows']:4d} image={row['image']:.5f} fused={row['fused']:.5f} share={row['share']:.3f}")

    out = ROOT / "work/leak/fusion_grid.json"
    out.write_text(json.dumps({"image_only": baseline, "best": best, "breakdown": breakdown,
                               "top": results[:30]}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
