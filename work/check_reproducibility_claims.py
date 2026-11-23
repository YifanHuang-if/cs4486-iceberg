"""Verify the reproducibility checklist claims against the actual artefacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

ROOT = Path(r"C:\Users\16018\cs4486-iceberg")


def param_count(tag="e40a", fold=0):
    from src.v2.model import RadarV1

    checkpoint = torch.load(ROOT / "outputs/v3" / tag / f"fold{fold}.pt",
                            map_location="cpu", weights_only=True)
    widths = tuple(checkpoint.get("widths", [16, 32, 64, 96]))
    model = RadarV1(dropout=checkpoint["dropout"],
                    stats_dim=checkpoint.get("stats_dim", 15), widths=widths)
    total = sum(p.numel() for p in model.parameters())
    print(f"RadarV1 widths={widths} params={total:,}")
    print(f"checkpoint keys: {sorted(checkpoint.keys())}")
    return total


def fold_seed():
    text = (ROOT / "src/v3/prepare.py").read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "StratifiedKFold" in line or re.search(r"seed\s*=\s*\d", line) or "random_state" in line:
            print("  prepare.py:", line.strip()[:100])
    folds = list((ROOT / "data/splits/folds.csv").read_text(encoding="utf-8").splitlines())
    print(f"  folds.csv rows={len(folds) - 1}")
    counts = {}
    for row in folds[1:]:
        fold = row.split(",")[1]
        counts[fold] = counts.get(fold, 0) + 1
    print(f"  per-fold sizes: {counts}")


def seeds_used():
    seeds = set()
    for path in (ROOT / "outputs/v3").glob("*/fold0_metrics.json"):
        blob = json.loads(path.read_text(encoding="utf-8"))
        seeds.add(blob.get("config", {}).get("seed"))
    print(f"  seeds actually used: {sorted(s for s in seeds if s is not None)}")


if __name__ == "__main__":
    print("== model size ==")
    param_count()
    print("== folds ==")
    fold_seed()
    print("== seeds ==")
    seeds_used()
