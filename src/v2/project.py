"""Estimate the expected leaderboard score from held-out components.

The scored test rows split into two populations: those whose exact angle occurs in
the training labels (57.0% of the 3,424 real rows) and the rest. The fused loss on
the first population and the model loss on the second are both measurable on
training rows through leave-one-out, so the expected total can be projected
instead of guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.angle_fusion import bayes_fusion, loo_prior, log_loss

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--coverage", type=float, default=0.5695, help="share of real test rows with an angle match")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "submissions" / "projection.json")
    args = parser.parse_args()

    train = np.load(CACHE / "train.npz", allow_pickle=False)
    labels = train["labels"].astype(np.float64)
    oof = np.load(args.oof)
    prior = loo_prior(train["angles"], labels, args.alpha)
    fused = bayes_fusion(oof, prior, args.weight)
    has_mate = np.isfinite(prior)

    report = {
        "rows_with_angle_mate": int(has_mate.sum()),
        "rows_without_angle_mate": int((~has_mate).sum()),
        "model_loss_all": log_loss(labels, oof),
        "model_loss_with_mate": log_loss(labels[has_mate], oof[has_mate]),
        "model_loss_without_mate": log_loss(labels[~has_mate], oof[~has_mate]),
        "fused_loss_with_mate": log_loss(labels[has_mate], fused[has_mate]),
        "coverage_assumed": args.coverage,
        "alpha": args.alpha, "weight": args.weight,
    }
    residual = report["model_loss_without_mate"]
    for name, value in (("residual_as_is", residual), ("residual_15pct_better", residual * 0.85),
                        ("residual_25pct_better", residual * 0.75), ("residual_35pct_better", residual * 0.65)):
        report[f"projected_{name}"] = (
            args.coverage * report["fused_loss_with_mate"] + (1 - args.coverage) * value
        )
    report["projected"] = report["projected_residual_as_is"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
