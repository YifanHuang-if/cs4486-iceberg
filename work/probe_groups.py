"""Group-level diagnostics for any v3 tag, including partially trained runs.

Loads whatever folds exist for a tag, builds the OOF vector on the rows that are
covered, and reports image and fused log loss separately for exact / near / far /
missing-angle rows. Fusion uses fold-isolated priors only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.v3.fusion import fold_fusion, loss, pool_queries
from src.v3.train import OUT, ROOT
from src.v2.train import read_folds


def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def load_metadata():
    a = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    return dict(ids=a["ids"], labels=a["labels"].astype(np.float64), angles=a["angles"],
                folds=read_folds(a["ids"], a["labels"])["fold"])


def load_tag_partial(tag, data):
    probability = np.full(len(data["ids"]), np.nan)
    seen = np.zeros(len(data["ids"]), dtype=bool)
    for fold in range(5):
        path = OUT / tag / f"fold{fold}_predictions.npz"
        if not path.exists():
            continue
        blob = np.load(path, allow_pickle=False)
        rows = np.flatnonzero(data["folds"] == fold)
        if not np.array_equal(blob["valid_ids"], data["ids"][rows]):
            raise ValueError(f"{tag} fold {fold} ID order mismatch")
        probability[rows] = blob["probability"]
        seen[rows] = True
    if not seen.any():
        raise ValueError(f"{tag} has no completed folds")
    return probability, seen


def group_masks(data, subset):
    """Exact / near / far / missing, computed only on the requested rows."""
    angles = data["angles"]
    labels = data["labels"]
    folds = data["folds"]
    exact = np.zeros(len(angles), dtype=bool)
    for fold in range(5):
        heldout = (folds == fold) & subset
        if not heldout.any():
            continue
        reference = (folds != fold) & np.isfinite(angles)
        a = angles[reference]
        value = angles[heldout]
        finite = np.isfinite(value)
        exact_rows = np.zeros(len(value), dtype=bool)
        if len(a) and finite.any():
            sorted_a = np.sort(a)
            pos = np.clip(np.searchsorted(sorted_a, value[finite]), 0, len(sorted_a) - 1)
            exact_rows[np.flatnonzero(finite)] = np.abs(sorted_a[pos] - value[finite]) < 1e-12
        exact[np.flatnonzero(heldout)] = exact_rows
    # The coverage mask does not depend on the image probabilities.
    _, covered = fold_fusion(np.full(len(angles), 0.5), angles, labels, folds)
    covered &= subset
    return {
        "exact": exact & subset,
        "near": covered & ~exact & subset,
        "far": ~covered & np.isfinite(angles) & subset,
        "missing": ~np.isfinite(angles) & subset,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    data = load_metadata()
    y, angles, folds = data["labels"], data["angles"], data["folds"]
    report = {"tags": args.tags}
    matrix, seen_all = [], None
    for tag in args.tags:
        probability, seen = load_tag_partial(tag, data)
        seen_all = seen if seen_all is None else (seen_all & seen)
        matrix.append(np.where(seen, probability, np.nan))
        report.setdefault("per_tag", {})[tag] = {
            "rows": int(seen.sum()),
            "image": round(loss(y[seen], probability[seen]), 6),
        }
    matrix = np.vstack(matrix)
    subset = seen_all
    image = np.nanmean(matrix, axis=0)
    fused, covered = fold_fusion(image, angles, y, folds)
    pooled = pool_queries(image, angles)
    fused_pooled = np.where(covered, fused, image)

    report["rows"] = int(subset.sum())
    report["image"] = round(loss(y[subset], image[subset]), 6)
    report["fused"] = round(loss(y[subset], fused[subset]), 6)
    report["groups"] = {}
    masks = group_masks(data, subset)
    total = -np.sum(y[subset] * np.log(np.clip(fused[subset], 1e-7, 1 - 1e-7))
                    + (1 - y[subset]) * np.log(np.clip(1 - fused[subset], 1e-7, 1 - 1e-7)))
    for name, mask in masks.items():
        if not mask.any():
            continue
        row_loss = -(y[mask] * np.log(np.clip(fused[mask], 1e-7, 1 - 1e-7))
                     + (1 - y[mask]) * np.log(np.clip(1 - fused[mask], 1e-7, 1 - 1e-7)))
        report["groups"][name] = {
            "rows": int(mask.sum()),
            "image": round(loss(y[mask], image[mask]), 6),
            "fused": round(loss(y[mask], fused[mask]), 6),
            "share": round(float(row_loss.sum() / total), 4),
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.name:
        path = ROOT / "work/leak" / f"{args.name}.json"
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
