"""Test-set inference for saved v3 checkpoints.

Rebuilds each fold model from its checkpoint (architecture, dropout and the
exact per-fold preprocessing are all stored), predicts the 8,424 test images
with eight-view test-time augmentation, and caches one array per (tag, fold).

Run with the repository root on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.v2.model import RadarV1, predict
from src.v2.train import transform_images
from src.v3.train import OUT, ROOT, meta_transform

CACHE = ROOT / "work" / "predictions"


def load_test():
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    stats = np.load(ROOT / "work/cache/v3_image_stats.npz", allow_pickle=False)
    identifiers = test["ids"].astype(str)
    angles = test["angles"].astype(np.float64)
    missing = test["angle_missing"].astype(np.float32)
    metadata = np.column_stack([angles, missing, stats["test"]]).astype(np.float32)
    return identifiers, test["images"], metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    identifiers, images, metadata = load_test()
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {"test_rows": int(len(identifiers)),
              "test_ids_sha256": hashlib.sha256("\n".join(identifiers).encode()).hexdigest(),
              "tags": {}}

    for tag in args.tags:
        directory = OUT / tag
        probabilities = []
        for fold in args.folds:
            checkpoint_path = directory / f"fold{fold}.pt"
            if not checkpoint_path.exists():
                raise SystemExit(f"missing checkpoint {checkpoint_path}")
            target = CACHE / f"{tag}_fold{fold}.npy"
            if target.exists():
                values = np.load(target)
                if len(values) != len(identifiers):
                    raise SystemExit(f"cached {target} has the wrong length")
                probabilities.append(values)
                print(f"{tag} fold {fold}: cached done", flush=True)
                continue
            # Some trainers already saved test probabilities alongside the fold.
            saved = directory / f"fold{fold}_predictions.npz"
            if saved.exists():
                blob = np.load(saved, allow_pickle=False)
                if "test_probability" in blob.files and len(blob["test_probability"]) == len(identifiers):
                    if "test_ids" in blob.files and not np.array_equal(blob["test_ids"].astype(str), identifiers):
                        raise SystemExit(f"{saved} test IDs do not match the cache")
                    values = blob["test_probability"].astype(np.float64)
                    np.save(target, values)
                    probabilities.append(values)
                    print(f"{tag} fold {fold}: reused saved test predictions", flush=True)
                    continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = RadarV1(dropout=checkpoint["dropout"], stats_dim=checkpoint.get("stats_dim", 15),
                            widths=tuple(checkpoint.get("widths", [16, 32, 64, 96])))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            normalisation = checkpoint["image_normalisation"]
            fitted = checkpoint["image_fit"]
            meta_fit = checkpoint["metadata_fit"]
            chunks = []
            for start in range(0, len(identifiers), args.chunk):
                stop = start + args.chunk
                x = transform_images(images[start:stop], normalisation, fitted)
                s = meta_transform(metadata[start:stop], meta_fit)
                chunks.append(predict(model, torch.from_numpy(x), torch.from_numpy(s),
                                      "cpu", tta=True, batch=128))
            values = np.concatenate(chunks)
            np.save(target, values.astype(np.float64))
            probabilities.append(values)
            print(f"{tag} fold {fold}: mean={values.mean():.4f} done", flush=True)
        stacked = np.vstack(probabilities)
        report["tags"][tag] = {
            "folds": list(args.folds),
            "mean": round(float(stacked.mean()), 6),
            "min": round(float(stacked.min()), 6),
            "max": round(float(stacked.max()), 6),
        }

    path = CACHE / "test_predictions.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
