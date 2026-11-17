"""Adaptation at test time by refreshing BatchNorm statistics (AdaBN).

The model contains BatchNorm layers. Their running statistics were fitted on the
training folds, but the test distribution differs from training (the two
leaderboards differ by 0.021, which is already evidence of that). AdaBN simply
re-estimates those statistics from the unlabelled target images before
predicting. No labels and no gradient steps are involved.

The same operation can be measured on a held-out fold: adapting on the fold's
images is exactly what would happen on the test set, and the fold's labels are
used only to score the result.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.v2.model import RadarV1
from src.v2.train import fit_normalisation, read_folds, transform_images
from src.v3.fusion import loss as fused_loss
from src.v3.train import OUT, ROOT, fit_meta, meta_transform


def load_fold(tag, fold, data, stats_cache):
    ids = data["ids"]
    folds = read_folds(ids, data["labels"])["fold"]
    train = folds != fold
    valid = ~train
    checkpoint = torch.load(OUT / tag / f"fold{fold}.pt", map_location="cpu", weights_only=True)
    model = RadarV1(dropout=checkpoint["dropout"],
                    stats_dim=checkpoint.get("stats_dim", 15),
                    widths=tuple(checkpoint.get("widths", [16, 32, 64, 96])))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    fitted = fit_normalisation(data["images"][train])
    raw = np.column_stack([data["angles"][train], data["angle_missing"][train],
                           stats_cache["train"][train]]).astype(np.float32)
    meta_fit = fit_meta(raw)

    def prepare(index):
        x = transform_images(data["images"][index], checkpoint["image_normalisation"], fitted)
        meta = np.column_stack([data["angles"][index], data["angle_missing"][index],
                                stats_cache["train"][index]]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(meta_transform(meta, meta_fit))

    x_valid, s_valid = prepare(np.flatnonzero(valid))
    return model, x_valid, s_valid, valid


def predict_dihedral(model, images, stats, batch=128):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = images[start:start + batch]
            s = stats[start:start + batch]
            views = []
            for code in range(8):
                view = x
                if code & 1:
                    view = view.flip(-1)
                if code & 2:
                    view = view.flip(-2)
                if code & 4:
                    view = view.transpose(-1, -2)
                views.append(torch.sigmoid(model(view, s)).numpy().astype(np.float64))
            outputs.append(np.mean(views, axis=0))
    return np.concatenate(outputs)


def adapt_bn(model, images, stats, passes=1, batch=64):
    """Run the images through the network with only BatchNorm in training mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()
            module.momentum = 0.1
    with torch.no_grad():
        for _ in range(passes):
            for start in range(0, len(images), batch):
                model(images[start:start + batch], stats[start:start + batch])
    model.eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="e40a")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    labels = data["labels"].astype(np.float64)
    stats_cache = np.load(ROOT / "work/cache/v3_image_stats.npz", allow_pickle=False)

    for passes in (0, 1, 2):
        model, x_valid, s_valid, valid = load_fold(args.tag, args.fold, data, stats_cache)
        if passes:
            adapt_bn(model, x_valid, s_valid, passes=passes)
        probability = predict_dihedral(model, x_valid, s_valid)
        print(f"passes={passes}: loss={fused_loss(labels[valid], probability):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
