"""Evaluate a prespecified V4 reliability rule using completed V3 OOF files.

Requires local V3 cache and fold prediction artifacts. Does not train new CNNs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.v3.evaluate import load_metadata, load_tag
from src.v3.fusion import fold_fusion, loss
from src.v4.reliability import fold_gated_fusion
from src.v4.groups import evidence_groups

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--name", default="v4_reliability_fixed")
    p.add_argument("--folds-file", type=Path)
    p.add_argument("--sigma", type=float, default=.02)
    p.add_argument("--alpha", type=float, default=.1)
    p.add_argument("--tau", type=float, default=8.)
    p.add_argument("--distance-scale", type=float, default=.02)
    p.add_argument("--preserve-exact", action="store_true", help="Preserve full V3 fusion when training angle exactly matches")
    args = p.parse_args()
    data = load_metadata(args.folds_file)
    models = [load_tag(tag, data, require_test=False)[0] for tag in args.tags]
    model = np.mean(models, axis=0)
    angles, labels, folds = data["angles"], data["labels"], data["folds"]
    control, _ = fold_fusion(model, angles, labels, folds)
    candidate, confidence = fold_gated_fusion(
        model, angles, labels, folds, sigma=args.sigma, alpha=args.alpha,
        tau=args.tau, distance_scale=args.distance_scale, preserve_exact=args.preserve_exact)
    baseline = loss(labels, control)
    score = loss(labels, candidate)
    eps = 1e-7
    v3c = np.clip(control, eps, 1-eps)
    v4c = np.clip(candidate, eps, 1-eps)
    v3_row_loss = -(labels*np.log(v3c)+(1-labels)*np.log1p(-v3c))
    v4_row_loss = -(labels*np.log(v4c)+(1-labels)*np.log1p(-v4c))
    evidence = evidence_groups(angles, folds)
    evidence_report = {
        key: {
            "rows": int(mask.sum()),
            "v3_log_loss": loss(labels[mask], control[mask]) if mask.any() else None,
            "v4_log_loss": loss(labels[mask], candidate[mask]) if mask.any() else None,
            "v3_total_loss_share": float(v3_row_loss[mask].sum() / v3_row_loss.sum()),
            "v4_total_loss_share": float(v4_row_loss[mask].sum() / v4_row_loss.sum()),
            "mean_reliability": float(confidence[mask].mean()) if mask.any() else None
        } for key, mask in evidence.items()
    }
    report = {
        "name": args.name, "tags": args.tags,
        "metric": "pooled development five-fold OOF log loss",
        "split": str(args.folds_file or "data/splits/folds.csv"),
        "official_kaggle_score": None, "target": .083,
        "method": "Effective sample size and nearest-distance gated angle prior",
        "parameters": {"sigma": args.sigma, "alpha": args.alpha,
                       "tau": args.tau, "distance_scale": args.distance_scale, "preserve_exact": args.preserve_exact},
        "baseline_v3_model_log_loss": loss(labels, model),
        "baseline_v3_fused_log_loss": baseline,
        "candidate_gated_log_loss": score,
        "improvement_over_v3": baseline - score,
        "target_met_in_development_cv": bool(score < .083),
        "mean_reliability": float(confidence.mean()),
        "angle_evidence_groups": evidence_report,
        "by_fold": [
            {"fold": int(f), "rows": int(np.count_nonzero(folds == f)),
             "v3": loss(labels[folds == f], control[folds == f]),
             "gated": loss(labels[folds == f], candidate[folds == f])}
            for f in sorted(np.unique(folds))],
        "subgroups": {
            key: {"rows": int(mask.sum()),
                  "v3": loss(labels[mask], control[mask]),
                  "gated": loss(labels[mask], candidate[mask])}
            for key, mask in [("observed_angle", np.isfinite(angles)),
                              ("missing_angle", ~np.isfinite(angles))]
            if mask.any()},
        "limitations": [
            "No outer fold labels are used in its own angle prior.",
            "Hyperparameters are fixed for this diagnostic, not fitted to outer labels.",
            "The frozen five-fold split is development CV, not an untouched test.",
            "The experiment makes no official Kaggle submission."
        ]}
    root = Path(__file__).resolve().parents[2]
    out = root / "results/v4"
    out.mkdir(parents=True, exist_ok=True)
    (out / (args.name + ".json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
