"""Apply the tuned angle evidence to a blended prediction and write a submission.

The angle layer is a two-tier prior: the exact-angle group when it exists,
otherwise a Gaussian kernel over the training angles. Its shrinkage, bandwidth and
fusion weight were tuned by leave-one-out on training rows only (see
``src/v2/angle_radius.py``), so nothing here touches test labels.

Usage:
    python -m src.v2.submit --submission work/submissions/blend.csv --oof work/submissions/blend.oof.npy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.v2.angle_fusion import bayes_fusion, log_loss
from src.v2.angle_radius import AngleIndex, evaluate

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"


def read_submission(path: Path) -> dict:
    with path.open(newline="") as handle:
        return {row["id"]: float(row["is_iceberg"]) for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "submissions" / "final.csv")
    parser.add_argument("--clip", type=float, default=1e-3)
    parser.add_argument("--spread", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--weight", type=float, default=None)
    args = parser.parse_args()

    train = np.load(CACHE / "train.npz", allow_pickle=False)
    test = np.load(CACHE / "test.npz", allow_pickle=False)
    labels = train["labels"].astype(np.float64)
    oof = np.load(args.oof)
    if len(oof) != len(labels):
        raise ValueError("OOF length does not match the training cache")

    index = AngleIndex(train["angles"], labels)
    tuning = evaluate(index, oof, labels, np.arange(len(labels)))
    best = tuning["best"]
    spread = args.spread if args.spread is not None else best["spread"]
    alpha = args.alpha if args.alpha is not None else best["alpha"]
    weight = args.weight if args.weight is not None else best["weight"]
    scheme = best["scheme"]
    print(f"tuned scheme={scheme} spread={spread} alpha={alpha} weight={weight}: "
          f"OOF {tuning['base']:.5f} -> {best['loss']:.5f} (coverage {best['coverage']:.3f})")

    # Test-side prior with the same rule.
    position_free = np.arange(len(test["angles"]))
    exact = np.full(len(test["angles"]), np.nan)
    kernel = np.full(len(test["angles"]), np.nan)
    valid = np.isfinite(test["angles"])
    exact[valid] = index.exact_prior(np.asarray(test["angles"])[valid], alpha=alpha)
    kernel[valid] = index.kernel_prior(np.asarray(test["angles"])[valid], spread, alpha)
    prior = np.where(np.isfinite(exact), exact, kernel)

    probabilities = read_submission(args.submission)
    base = np.array([probabilities[i] for i in test["ids"]], dtype=np.float64)
    fused = bayes_fusion(base, prior, weight)
    final = np.clip(fused, args.clip, 1 - args.clip)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "is_iceberg"])
        for sample_id, probability in zip(test["ids"].tolist(), final.tolist()):
            writer.writerow([sample_id, f"{probability:.10f}"])

    summary = {
        "source": str(args.submission), "rows": int(len(final)), "clip": args.clip,
        "scheme": scheme, "spread": spread, "alpha": alpha, "weight": weight,
        "oof_model_loss": tuning["base"], "oof_fused_loss": best["loss"],
        "oof_coverage": best["coverage"], "tuning_top": tuning["top"][:3],
        "rows_with_prior": int(np.isfinite(prior).sum()),
        "share_with_prior": float(np.isfinite(prior).mean()),
        "share_changed": float(np.mean(np.abs(final - np.clip(base, args.clip, 1 - args.clip)) > 1e-6)),
        "mean_before": float(np.mean(base)), "mean_after": float(np.mean(final)),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "tuning_top"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
