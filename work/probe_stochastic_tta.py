"""Does stochastic TTA beat the fixed 8-view dihedral TTA?

The 16th-place solution averages several predictions taken through random
augmentation, not just the deterministic dihedral views used here. That is a
post-hoc change: the checkpoint is unchanged, so it can be evaluated directly on
a held-out fold without retraining.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.v3.fusion import loss as fused_loss
from src.v3.train import OUT, ROOT, fit_meta, meta_transform
from src.v2.model import RadarV1
from src.v2.train import fit_normalisation, read_folds, transform_images


def dihedral_views(x, code):
    """The eight deterministic views already used during inference."""
    if code & 1:
        x = x.flip(-1)
    if code & 2:
        x = x.flip(-2)
    if code & 4:
        x = x.transpose(-1, -2)
    return x


def predict_fixed(model, images, stats, batch=128):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = images[start:start + batch]
            s = stats[start:start + batch]
            views = [torch.sigmoid(model(dihedral_views(x, code), s)).numpy().astype(np.float64)
                     for code in range(8)]
            outputs.append(np.mean(views, axis=0))
    return np.concatenate(outputs)


def predict_stochastic(model, images, stats, repeats, noise, batch=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = images[start:start + batch]
            s = stats[start:start + batch]
            views = []
            for code in range(8):
                for _ in range(repeats):
                    view = dihedral_views(x, code)
                    view = view + torch.randn(view.shape, generator=generator) * noise
                    views.append(torch.sigmoid(model(view, s)).numpy().astype(np.float64))
            outputs.append(np.mean(views, axis=0))
    return np.concatenate(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="e40a")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.02)
    args = parser.parse_args()

    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"]
    labels = data["labels"].astype(np.float64)
    folds = read_folds(ids, data["labels"])["fold"]
    stats_cache = np.load(ROOT / "work/cache/v3_image_stats.npz", allow_pickle=False)

    train = folds != args.fold
    valid = ~train
    checkpoint = torch.load(OUT / args.tag / f"fold{args.fold}.pt", map_location="cpu", weights_only=True)
    model = RadarV1(dropout=checkpoint["dropout"],
                    stats_dim=checkpoint.get("stats_dim", 15),
                    widths=tuple(checkpoint.get("widths", [16, 32, 64, 96])))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    normalisation = checkpoint["image_normalisation"]
    fitted = fit_normalisation(data["images"][train])
    meta_fit = fit_meta(np.column_stack([data["angles"][train],
                                         data["angle_missing"][train],
                                         stats_cache["train"][train]]).astype(np.float32))

    def prepare(index):
        x = transform_images(data["images"][index], normalisation, fitted)
        raw = np.column_stack([data["angles"][index],
                               data["angle_missing"][index],
                               stats_cache["train"][index]]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(meta_transform(raw, meta_fit))

    x_valid, s_valid = prepare(np.flatnonzero(valid))
    fixed = predict_fixed(model, x_valid, s_valid)
    print(f"fixed 8-view TTA      loss={fused_loss(labels[valid], fixed):.6f}")

    for repeats in (1, 2, 4):
        probability = predict_stochastic(model, x_valid, s_valid, repeats, args.noise)
        print(f"stochastic x{repeats:2d}         loss={fused_loss(labels[valid], probability):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
