"""Train radar CNN folds, saving held-out and test predictions for ensembling.

Usage:
    python -m src.v2.train --tag s0_medium --arch medium --seed 0

Deliberately depends only on numpy, torch and the standard library so it can run
before the rest of the analysis stack is installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.v2.model import STAT_NAMES, augment, build_model, image_statistics, predict

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"
MODELS = ROOT / "work" / "models"

ARCHITECTURES = {
    "v1": dict(widths=(16, 32, 64, 96), dropout=0.3),
    "fast": dict(widths=(32, 64, 128, 192), dropout=0.3, stem_stride=2),
    "small": dict(widths=(16, 32, 64, 128), dropout=0.3),
    "medium": dict(widths=(24, 48, 96, 160), dropout=0.3),
    "wide": dict(widths=(32, 64, 128, 192), dropout=0.35),
}


def log_loss(labels, probability, floor: float = 1e-7) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), floor, 1 - floor)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def read_folds(ids: np.ndarray, labels: np.ndarray, path: Path | None = None):
    path = path or (ROOT / "data" / "splits" / "folds.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != set(ids.tolist()):
        raise ValueError("fold ids do not exactly cover the training cache")
    fold = np.array([int(by_id[i]["fold"]) for i in ids], dtype=np.int64)
    fold_labels = np.array([float(by_id[i]["label"]) for i in ids], dtype=np.float32)
    if not np.array_equal(fold_labels, labels):
        raise ValueError("fold labels differ from cache labels")
    if set(fold.tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("invalid fold assignment")
    return {"fold": fold, "label": fold_labels, "path": str(path)}


def load_cache(folds_path: Path | None = None):
    train = np.load(CACHE / "train.npz", allow_pickle=False)
    test = np.load(CACHE / "test.npz", allow_pickle=False)
    folds = read_folds(train["ids"], train["labels"], folds_path)
    cached = CACHE / "stats.npz"
    if cached.exists():
        payload = np.load(cached, allow_pickle=False)
        train_stats, test_stats = payload["train"], payload["test"]
    else:
        median_angle = float(np.nanmedian(train["angles"])) if np.isfinite(train["angles"]).any() else 0.0
        train_angle = np.where(np.isfinite(train["angles"]), train["angles"], median_angle)
        test_angle = np.where(np.isfinite(test["angles"]), test["angles"], median_angle)
        train_stats = np.column_stack([
            train_angle, train["angle_missing"].astype(np.float32), image_statistics(train["images"])
        ]).astype(np.float32)
        test_stats = np.column_stack([
            test_angle, test["angle_missing"].astype(np.float32), image_statistics(test["images"])
        ]).astype(np.float32)
        np.savez(cached, train=train_stats, test=test_stats)
    if train_stats.shape[1] != len(STAT_NAMES):
        raise ValueError(f"metadata width {train_stats.shape[1]} does not match {len(STAT_NAMES)} names")
    return {
        "train_images": train["images"], "train_labels": train["labels"], "train_ids": train["ids"],
        "test_images": test["images"], "test_ids": test["ids"],
        "train_stats": train_stats, "test_stats": test_stats, "folds": folds,
    }


def fit_normalisation(images: np.ndarray) -> dict:
    return {
        "mean": images.mean(axis=(0, 2, 3), dtype=np.float64).tolist(),
        "std": images.std(axis=(0, 2, 3), dtype=np.float64).clip(1e-6).tolist(),
    }


def transform_images(images: np.ndarray, kind: str, fitted: dict) -> np.ndarray:
    x = images.astype(np.float32)
    if kind == "local":
        mean = x.mean(axis=(2, 3), keepdims=True, dtype=np.float64).astype(np.float32)
        std = x.std(axis=(2, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    elif kind == "local_clip":
        # Radar speckle produces extreme pixels that dominate the z-score of a
        # small patch. Clip to the central 98% before standardising so the
        # normalisation reflects the target rather than the tail.
        low = np.percentile(x, 1, axis=(2, 3), keepdims=True).astype(np.float32)
        high = np.percentile(x, 99, axis=(2, 3), keepdims=True).astype(np.float32)
        x = np.clip(x, low, high)
        mean = x.mean(axis=(2, 3), keepdims=True, dtype=np.float64).astype(np.float32)
        std = x.std(axis=(2, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    elif kind == "local_robust":
        # Median and interquartile range instead of mean and standard deviation.
        median = np.median(x, axis=(2, 3), keepdims=True).astype(np.float32)
        q75 = np.percentile(x, 75, axis=(2, 3), keepdims=True).astype(np.float32)
        q25 = np.percentile(x, 25, axis=(2, 3), keepdims=True).astype(np.float32)
        x = np.clip(x, q25 - 3.0 * (q75 - q25), q75 + 3.0 * (q75 - q25))
        mean = median
        std = np.maximum(q75 - q25, 1e-6)
    else:
        mean = np.asarray(fitted["mean"], dtype=np.float32)[None, :, None, None]
        std = np.asarray(fitted["std"], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((x - mean) / np.maximum(std, 1e-6), dtype=np.float32)


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def train_fold(args, data, fold, extra=None):
    seed = args.seed * 100 + fold
    torch.manual_seed(seed)
    train_mask = data["folds"]["fold"] != fold
    valid_mask = ~train_mask
    fitted = fit_normalisation(data["train_images"][train_mask])
    train_x = transform_images(data["train_images"][train_mask], args.normalisation, fitted)
    valid_x = transform_images(data["train_images"][valid_mask], args.normalisation, fitted)
    test_x = transform_images(data["test_images"], args.normalisation, fitted)

    stats = data["train_stats"]
    mean = stats[train_mask].mean(0)
    std = stats[train_mask].std(0)
    std = np.where(std < 1e-6, 1.0, std)
    train_s = ((stats[train_mask] - mean) / std).astype(np.float32)
    valid_s = ((stats[valid_mask] - mean) / std).astype(np.float32)
    test_s = ((data["test_stats"] - mean) / std).astype(np.float32)

    train_y = data["train_labels"][train_mask].astype(np.float32)
    valid_y = data["train_labels"][valid_mask].astype(np.float32)

    tensors = [torch.from_numpy(train_x), torch.from_numpy(train_s), torch.from_numpy(train_y),
               torch.ones(int(train_mask.sum()))]
    if extra is not None:
        pseudo_x = transform_images(extra["images"], args.normalisation, fitted)
        pseudo_s = ((extra["stats"] - mean) / std).astype(np.float32)
        tensors = [
            torch.cat([tensors[0], torch.from_numpy(pseudo_x)]),
            torch.cat([tensors[1], torch.from_numpy(pseudo_s)]),
            torch.cat([tensors[2], torch.from_numpy(extra["labels"].astype(np.float32))]),
            torch.cat([tensors[3], torch.full((len(extra["labels"]),), float(args.pseudo_weight))]),
        ]
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0, drop_last=False)
    aug = torch.Generator().manual_seed(seed + 7919)
    model = build_model(args.arch, in_channels=2, **ARCHITECTURES[args.arch]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(math.ceil(len(loader.dataset) / args.batch), 1) * args.epochs
    warmup = max(math.ceil(0.06 * total_steps), 1)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - 1e-3) + 1e-3

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    history, best, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        started = time.perf_counter()
        for images, meta, target, weight in loader:
            if args.augment:
                images = augment(images.to(args.device), aug)
            else:
                images = images.to(args.device)
            meta = meta.to(args.device)
            target = target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = (criterion(model(images, meta), target) * weight.to(args.device)).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(target)
        probability = predict(model, torch.from_numpy(valid_x), torch.from_numpy(valid_s), args.device, tta=False)
        score = log_loss(valid_y, probability)
        history.append({"fold": fold, "epoch": epoch, "train_loss": total / len(loader.dataset),
                        "val_log_loss": score, "lr": optimizer.param_groups[0]["lr"]})
        if score < best - 1e-6:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            print(f"  [{args.tag}] fold {fold} epoch {epoch}: val {score:.5f} best {best:.5f} "
                  f"(ep {best_epoch}) {time.perf_counter() - started:.1f}s/epoch", flush=True)
        if args.patience and epoch - best_epoch >= args.patience and epoch >= args.min_epochs:
            print(f"  [{args.tag}] fold {fold} early stop at {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    valid_plain = predict(model, torch.from_numpy(valid_x), torch.from_numpy(valid_s), args.device, tta=False)
    valid_tta = predict(model, torch.from_numpy(valid_x), torch.from_numpy(valid_s), args.device, tta=True)
    test_tta = predict(model, torch.from_numpy(test_x), torch.from_numpy(test_s), args.device, tta=True)

    out = MODELS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arch": args.arch, "fold": fold, "seed": args.seed,
                "best_epoch": best_epoch, "normalisation": args.normalisation, "fitted": fitted},
               out / f"fold{fold}.pt")
    valid_ids = data["train_ids"][valid_mask]
    write_csv(out / f"oof_fold{fold}.csv", ["id", "label", "probability", "plain_probability"],
              zip(valid_ids.tolist(), valid_y.tolist(), valid_tta.tolist(), valid_plain.tolist()))
    np.save(out / f"test_fold{fold}.npy", test_tta)
    write_csv(out / f"history_fold{fold}.csv", ["fold", "epoch", "train_loss", "val_log_loss", "lr"],
              [[h["fold"], h["epoch"], h["train_loss"], h["val_log_loss"], h["lr"]] for h in history])
    record = {"fold": fold, "val_log_loss_tta": log_loss(valid_y, valid_tta), "val_log_loss_plain": best,
              "best_epoch": best_epoch, "epochs_run": len(history),
              "params": int(sum(p.numel() for p in model.parameters()))}
    (out / f"metrics_fold{fold}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"DONE {args.tag} fold {fold}: tta {record['val_log_loss_tta']:.5f} plain {record['val_log_loss_plain']:.5f}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--arch", default="medium", choices=sorted(ARCHITECTURES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--normalisation", choices=["local", "global"], default="local")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pseudo", type=Path, default=None, help="npz with rows/labels of pseudo-labelled test rows")
    parser.add_argument("--pseudo-weight", type=float, default=0.5)
    parser.add_argument("--folds-file", type=Path, default=None)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = load_cache(args.folds_file)
    extra = None
    if args.pseudo is not None:
        payload = np.load(args.pseudo, allow_pickle=False)
        rows = payload["rows"]
        extra = {"images": data["test_images"][rows], "stats": data["test_stats"][rows], "labels": payload["labels"]}
    records = [train_fold(args, data, fold, extra) for fold in args.folds]
    (MODELS / args.tag / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (MODELS / args.tag / "summary.json").write_text(
        json.dumps({"tag": args.tag, "arch": args.arch, "seed": args.seed, "records": records,
                    "mean_fold_log_loss": float(np.mean([r["val_log_loss_tta"] for r in records]))}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
