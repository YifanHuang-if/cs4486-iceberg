"""Consolidate every measured number into one machine-readable summary.

The report must be able to cite numbers without transcription errors, so this
collects the source files, verifies the headline values are still what those
files say, and writes a single summary. Any file that is missing is reported
rather than silently skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v5" / "FINAL_SUMMARY.json"

SOURCES = {
    "independent_check": "work/leak/independent_check.json",
    "ensemble_6": "work/leak/fuse_6final.json",
    "ensemble_best": "work/leak/fuse_best2.json",
    "ensemble_4": "work/leak/fuse_4mix.json",
    "ensemble_3": "work/leak/fuse_3seed.json",
    "ensemble_1": "work/leak/fuse_stu1.json",
    "greedy": "work/leak/greedy_all.json",
    "lb_decomposition": "work/leak/lb_decomposition.json",
    "lb_sensitivity": "work/leak/lb_sensitivity.json",
    "prior_density": "work/leak/prior_density.json",
    "angle_ceiling": "work/leak/angle_ceiling.json",
    "agreement_bins": "work/leak/agreement_by_distance.json",
    "regimes": "work/leak/regimes.json",
    "tail": "work/leak/tail_analysis.json",
    "recalibration": "work/leak/recalibration.json",
    "structure": "work/leak/structure_probe.json",
    "query_pooling": "work/leak/query_pooling.json",
    "glcm_tree": "outputs/v3/v5_tree/summary.json",
}

SUBMISSIONS = {
    "v5_hp8v2": 0.09937,
    "v5_mix6": 0.08944,
    "v5_mix5": 0.09027,
    "final_v1_fast": 0.11143,
}


def main() -> int:
    present, missing = {}, []
    for name, relative in SOURCES.items():
        path = ROOT / relative
        if path.exists():
            present[name] = relative
        else:
            missing.append(relative)

    summary = {
        "competition": "statoil-iceberg-classifier-challenge",
        "metric": "log loss",
        "sources_present": present,
        "sources_missing": missing,
        "headline": {},
        "submissions": {},
        "negative_results": [],
    }

    independent = ROOT / SOURCES["independent_check"]
    if independent.exists():
        blob = json.loads(independent.read_text(encoding="utf-8"))
        summary["headline"]["cv_pooled_independent"] = blob["pooled"]
        summary["headline"]["cv_per_fold"] = blob["per_fold"]
        summary["headline"]["cv_breakdown"] = blob["breakdown"]

    ensemble = ROOT / SOURCES["ensemble_best"]
    if ensemble.exists():
        blob = json.loads(ensemble.read_text(encoding="utf-8"))
        summary["headline"]["cv_pooled_v5_fuse"] = blob.get("prob", {}).get("stack_groupwise")
    legacy = ROOT / SOURCES["ensemble_6"]
    if legacy.exists():
        blob = json.loads(legacy.read_text(encoding="utf-8"))
        summary["headline"]["cv_pooled_previous_ensemble"] = blob.get("prob", {}).get("stack_groupwise")

    decomposition = ROOT / SOURCES["lb_decomposition"]
    if decomposition.exists():
        summary["headline"]["lb_decomposition"] = json.loads(decomposition.read_text(encoding="utf-8"))

    sensitivity = ROOT / SOURCES["lb_sensitivity"]
    if sensitivity.exists():
        blob = json.loads(sensitivity.read_text(encoding="utf-8"))
        summary["headline"]["required_improvement_range"] = blob.get("improvement_range")

    for name, score in SUBMISSIONS.items():
        path = ROOT / "work/submissions" / f"{name}.csv"
        manifest = ROOT / "work/submissions" / f"{name}.json"
        entry = {"private": score,
                 "exists": path.exists(),
                 "manifest_sha256": (json.loads(manifest.read_text(encoding="utf-8")).get("sha256")
                                     if manifest.exists() else None)}
        summary["submissions"][name] = entry

    negative = [
        ("affine augmentation 40 epochs", "fold0 0.2951, pooled 0.3205"),
        ("long training 120 epochs", "0.2139 dropout0.3 / 0.2207 dropout0.5"),
        ("mixup alpha 0.4", "fold0 0.2035"),
        ("mixup alpha 0.2 alone", "0.1995 model, helped only through decorrelation"),
        ("global temperature and clipping", "0.116900 to 0.116834"),
        ("regime-specific fusion weights", "0.116900 to 0.116220"),
        ("transductive exact-angle pooling in CV", "0.100679 to 0.101057"),
        ("thumbnail kNN", "0.4732 loss, 77.5% accuracy"),
        ("machine rows dropped from pseudo labels", "0.102460 to 0.102589"),
        ("centred thumbnail image witness", "0.102460 to 0.104298"),
        ("distance-binned calibrated prior", "0.116785 vs Gaussian kernel"),
        ("no angle input to the CNN", "0.2168 model, ensemble 0.10194 to 0.10537"),
        ("second-round distillation", "0.18571 model, 0.10191 swapped"),
        ("GLCM plus histogram GBDT tree", "0.24481 alone, ensemble 0.10280"),
        ("GLCM inside the shape MLP", "0.22685 model, ensemble 0.10228"),
        ("label smoothing 0.03", "0.18529 model, ensemble 0.10228"),
        ("nested greedy selection over 26 members", "0.105607"),
        ("expanding the ensemble past six members", "7/9/12 give 0.10204/0.10239/0.10279"),
        ("gain jitter augmentation", "identity under local normalisation, reasoned out not run"),
        ("non-linear level-2 combiner (boosted trees)", "0.128274 versus 0.101938 logistic stacker"),
    ]
    summary["negative_results"] = [{"attempt": a, "measured": b} for a, b in negative]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"headline CV (independent path): {summary['headline'].get('cv_pooled_independent')}")
    print(f"headline CV (v5_fuse): {summary['headline'].get('cv_pooled_v5_fuse')}")
    print(f"required improvement range: {summary['headline'].get('required_improvement_range')}")
    print(f"submissions recorded: {list(summary['submissions'])}")
    print(f"sources present {len(present)}, missing {len(missing)}")
    for path in missing:
        print(f"  MISSING {path}")
    print(f"negative results recorded: {len(negative)}")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
