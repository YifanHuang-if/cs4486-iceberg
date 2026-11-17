"""Prove a submission file is model output, not hand-written.

Four independent checks:

A. The file hash matches the manifest written when the pipeline produced it.
B. Re-running the submission generator from the same inputs reproduces the file
   byte for byte.
C. The cached test predictions really come from the trained checkpoints: each
   (tag, fold) checkpoint is reloaded and re-run on a fixed block of test rows,
   and its output is compared with the cached array on those rows.
D. Statistical sanity: the values cannot be a constant or a hand-typed pattern.

Nothing here retrains. If a check fails the script exits non-zero.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.v2.model import RadarV1, predict
from src.v2.train import transform_images
from src.v3.train import OUT, ROOT, meta_transform

CACHE = ROOT / "work" / "predictions"
BLOCK = 512


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_test_block():
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    stats = np.load(ROOT / "work/cache/v3_image_stats.npz", allow_pickle=False)
    identifiers = test["ids"].astype(str)
    metadata = np.column_stack([test["angles"],
                                test["angle_missing"].astype(np.float32),
                                stats["test"]]).astype(np.float32)
    return identifiers, test["images"], metadata


def replay_checkpoint(tag: str, fold: int, identifiers, images, metadata, block: int):
    """Recompute one checkpoint's test probabilities on a fixed block of rows."""
    checkpoint = torch.load(OUT / tag / f"fold{fold}.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") == "shape_mlp":
        from src.v3.shape_model import network
        features = np.load(ROOT / "work/cache/v3_shape_features.npz", allow_pickle=False)["test"]
        fitted = checkpoint["metadata_fit"]
        x = np.clip(meta_transform(features[:block], fitted), -8, 8)
        outputs = []
        for state in checkpoint["states"]:
            model = network(features.shape[1])
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                outputs.append(torch.sigmoid(model(torch.from_numpy(x)).squeeze(1)).numpy().astype(np.float64))
        return np.mean(outputs, axis=0), "shape_mlp"
    model = RadarV1(dropout=checkpoint["dropout"], stats_dim=checkpoint.get("stats_dim", 15),
                    widths=tuple(checkpoint.get("widths", [16, 32, 64, 96])))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = transform_images(images[:block], checkpoint["image_normalisation"], checkpoint["image_fit"])
    s = meta_transform(metadata[:block], checkpoint["metadata_fit"])
    return predict(model, torch.from_numpy(x), torch.from_numpy(s), "cpu", tta=True, batch=128), "cnn"


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "v5_mix6"
    path = ROOT / "work/submissions" / f"{name}.csv"
    manifest_path = ROOT / "work/submissions" / f"{name}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {"submission": str(path.relative_to(ROOT)), "checks": {}}
    failures = []

    # A. Manifest hash.
    actual = sha256(path)
    expected = manifest["sha256"]
    report["checks"]["A_manifest_hash"] = {
        "expected": expected, "actual": actual, "pass": actual == expected}
    if actual != expected:
        failures.append("A")
    print(f"A manifest hash: {'PASS' if actual == expected else 'FAIL'} ({actual[:16]}...)")

    # B. Regenerate and compare bytes.
    audit = f"{name}_audit"
    command = [sys.executable, str(ROOT / "work/v5_submit.py"),
               "--tags", *manifest["tags"], "--name", audit,
               "--sigma", str(manifest["sigma"]), "--alpha", str(manifest["alpha"]),
               "--l2", str(manifest["l2"]), "--clip", str(manifest["clip"]),
               "--pool-weight", str(manifest.get("pool_weight", 0.0))]
    if manifest.get("pool_non_exact_only"):
        command.append("--pool-non-exact-only")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        report["checks"]["B_regenerate"] = {"pass": False, "error": result.stderr[-800:]}
        failures.append("B")
        print("B regenerate: FAIL (generator error)")
    else:
        regenerated = ROOT / "work/submissions" / f"{audit}.csv"
        same = sha256(regenerated) == actual
        report["checks"]["B_regenerate"] = {"pass": same, "hash": sha256(regenerated)}
        if not same:
            failures.append("B")
        print(f"B regenerate byte-identical: {'PASS' if same else 'FAIL'}")
        # Leave no duplicate submission behind: it would pollute the inventory.
        for leftover in (regenerated, ROOT / "work/submissions" / f"{audit}.json"):
            if leftover.exists():
                leftover.unlink()

    # C. Cached arrays come from the checkpoints.
    identifiers, images, metadata = load_test_block()
    cache_rows = []
    for tag in manifest["tags"]:
        for fold in range(5):
            cached = np.load(CACHE / f"{tag}_fold{fold}.npy")
            replayed, kind = replay_checkpoint(tag, fold, identifiers, images, metadata, BLOCK)
            difference = float(np.max(np.abs(cached[:BLOCK] - replayed)))
            cache_rows.append({"tag": tag, "fold": fold, "kind": kind,
                               "max_abs_diff": difference,
                               "pass": difference < 2e-6})
            print(f"C {tag} fold {fold} ({kind}): max|diff| on first {BLOCK} rows = {difference:.2e}")
    worst = max(row["max_abs_diff"] for row in cache_rows)
    report["checks"]["C_checkpoint_replay"] = {
        "rows_per_checkpoint": BLOCK, "worst_max_abs_diff": worst,
        "pass": worst < 2e-6, "detail": cache_rows}
    if worst >= 2e-6:
        failures.append("C")
    print(f"C checkpoint replay: {'PASS' if worst < 2e-6 else 'FAIL'} (worst {worst:.2e})")

    # D. Sanity of the values themselves.
    import csv as csv_module
    import io
    rows = list(csv_module.reader(io.open(path, encoding="utf-8", newline="")))
    values = np.array([float(row[1]) for row in rows[1:]])
    per_tag = {tag: np.mean([np.load(CACHE / f"{tag}_fold{k}.npy") for k in range(5)], axis=0)
               for tag in manifest["tags"]}
    correlation = {tag: float(np.corrcoef(values, per_tag[tag])[0, 1]) for tag in manifest["tags"]}
    at_clip = int(np.sum((values <= manifest["clip"] + 1e-12) | (values >= 1 - manifest["clip"] - 1e-12)))
    sanity = {
        "rows": int(len(values)),
        "distinct_values": int(len(np.unique(values))),
        "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
        "share_at_clip": at_clip / len(values),
        "correlation_with_each_model": correlation,
        "constant_column": bool(np.ptp(values) == 0),
        "pass": bool(np.ptp(values) > 0 and len(np.unique(values)) > 1000
                     and min(correlation.values()) > 0.5
                     and 0.0 <= values.min() and values.max() <= 1.0),
    }
    report["checks"]["D_value_sanity"] = sanity
    if not sanity["pass"]:
        failures.append("D")
    print(f"D value sanity: {'PASS' if sanity['pass'] else 'FAIL'} "
          f"distinct={sanity['distinct_values']} mean={sanity['mean']:.4f} "
          f"at_clip={sanity['share_at_clip']:.4f} "
          f"min_corr={min(correlation.values()):.3f}")

    report["verdict"] = "PASS" if not failures else f"FAIL {failures}"
    out = ROOT / "work/submissions" / f"{name}_provenance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {report['verdict']}  ->  {out.relative_to(ROOT)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
