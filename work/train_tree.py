"""GLCM texture features and a compact histogram gradient-boosted tree.

Why this exists: every ensemble member so far is a CNN or an MLP over the same
pixel statistics, so their errors are correlated. A boosted tree over classical
radar texture descriptors is a genuinely different function class, which is the
one lever that has kept paying on this data.

Textures are grey-level co-occurrence matrices per band, quantised per image,
averaged over the four undirected directions so the descriptor is rotation
robust. The tree is a small histogram GBDT fitted with the exact XGBoost gain
formula; no external ML library is available in this environment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work" / "cache"

LEVELS = 16
DISTANCES = (1, 2)
OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def glcm_features(images: np.ndarray) -> np.ndarray:
    """Return (N, features) co-occurrence descriptors for both bands."""
    count = len(images)
    features = []
    for band in range(images.shape[1]):
        raw = images[:, band].astype(np.float64)
        flat = raw.reshape(count, -1)
        low = np.percentile(flat, 1, axis=1)[:, None, None]
        high = np.percentile(flat, 99, axis=1)[:, None, None]
        scaled = (raw - low) / np.maximum(high - low, 1e-6)
        quantised = np.clip((scaled * LEVELS).astype(np.int64), 0, LEVELS - 1)
        for distance in DISTANCES:
            accumulators = {name: np.zeros(count) for name in
                            ("contrast", "homogeneity", "energy", "correlation")}
            for dy, dx in OFFSETS:
                dy *= distance
                dx *= distance
                y_from, y_to = (0, 75 - dy) if dy >= 0 else (-dy, 75)
                x_from, x_to = (0, 75 - dx) if dx >= 0 else (-dx, 75)
                left = quantised[:, y_from:y_to, x_from:x_to].reshape(count, -1)
                right = quantised[:, y_from + dy:y_to + dy, x_from + dx:x_to + dx].reshape(count, -1)
                index = left * LEVELS + right
                matrix = np.zeros((count, LEVELS * LEVELS))
                for row in range(count):
                    matrix[row] = np.bincount(index[row], minlength=LEVELS * LEVELS)
                matrix = matrix.reshape(count, LEVELS, LEVELS)
                total = matrix.sum(axis=(1, 2), keepdims=True) + 1e-9
                probability = matrix / total
                i, j = np.meshgrid(np.arange(LEVELS), np.arange(LEVELS), indexing="ij")
                difference = (i - j).astype(np.float64)
                accumulators["contrast"] += (probability * difference ** 2).sum(axis=(1, 2))
                accumulators["homogeneity"] += (probability / (1.0 + difference ** 2)).sum(axis=(1, 2))
                accumulators["energy"] += (probability ** 2).sum(axis=(1, 2))
                mean_i = (probability * i).sum(axis=(1, 2))
                mean_j = (probability * j).sum(axis=(1, 2))
                std_i = np.sqrt(np.maximum((probability * (i - mean_i[:, None, None]) ** 2).sum(axis=(1, 2)), 1e-9))
                std_j = np.sqrt(np.maximum((probability * (j - mean_j[:, None, None]) ** 2).sum(axis=(1, 2)), 1e-9))
                covariance = (probability * (i - mean_i[:, None, None]) * (j - mean_j[:, None, None])).sum(axis=(1, 2))
                accumulators["correlation"] += covariance / (std_i * std_j)
            features.extend([accumulators[name] / len(OFFSETS) for name in
                             ("contrast", "homogeneity", "energy", "correlation")])
    return np.column_stack(features).astype(np.float32)


def load_glcm() -> tuple[np.ndarray, np.ndarray]:
    target = CACHE / "glcm_features.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as blob:
            return blob["train"], blob["test"]
    train = np.load(CACHE / "train.npz", allow_pickle=False)["images"]
    test = np.load(CACHE / "test.npz", allow_pickle=False)["images"]
    result = []
    for split, images in (("train", train), ("test", test)):
        pieces = [glcm_features(images[start:start + 512]) for start in range(0, len(images), 512)]
        result.append(np.concatenate(pieces))
        print(f"GLCM {split}: {result[-1].shape}", flush=True)
    np.savez(target, train=result[0], test=result[1])
    return result[0], result[1]


def bin_features(reference: np.ndarray, *others: np.ndarray, bins: int = 64):
    """Quantile binning fitted on the reference split only."""
    edges = np.empty((reference.shape[1], bins + 1))
    for column in range(reference.shape[1]):
        values = reference[:, column]
        quantiles = np.linspace(0, 1, bins + 1)
        edges[column] = np.quantile(values, quantiles)
        edges[column] = np.maximum.accumulate(edges[column] + np.arange(bins + 1) * 1e-9)
    outputs = []
    for matrix in (reference, *others):
        binned = np.empty(matrix.shape, dtype=np.int64)
        for column in range(matrix.shape[1]):
            binned[:, column] = np.clip(np.searchsorted(edges[column], matrix[:, column], side="right") - 1,
                                        0, bins - 1)
        outputs.append(binned)
    return outputs


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def log_loss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


class Tree:
    """One depth-limited tree stored as node arrays."""

    def __init__(self, depth, bins):
        self.feature = np.full(2 ** (depth + 1), -1, dtype=np.int64)
        self.threshold = np.zeros(2 ** (depth + 1))
        self.value = np.zeros(2 ** (depth + 1))
        self.bins = bins

    def fit(self, binned, gradient, hessian, depth, min_child_weight, l2, rng, feature_fraction):
        self._grow(binned, gradient, hessian, 0, 0, depth, min_child_weight, l2, rng, feature_fraction)

    def _grow(self, binned, g, h, node, level, depth, min_child_weight, l2, rng, feature_fraction):
        total_g, total_h = g.sum(), h.sum()
        if level >= depth or len(g) < 2 * min_child_weight or total_h < 1e-9:
            self.value[node] = -total_g / (total_h + l2)
            return
        rows = len(g)
        n_features = binned.shape[1]
        keep = max(1, int(n_features * feature_fraction))
        candidates = rng.choice(n_features, size=keep, replace=False)
        parent_score = total_g ** 2 / (total_h + l2)
        best_gain, best_feature, best_threshold = 0.0, -1, 0
        for feature in candidates:
            column = binned[:, feature]
            grad_sum = np.bincount(column, weights=g, minlength=self.bins)
            hess_sum = np.bincount(column, weights=h, minlength=self.bins)
            count = np.bincount(column, minlength=self.bins).astype(np.float64)
            left_g = np.cumsum(grad_sum)[:-1]
            left_h = np.cumsum(hess_sum)[:-1]
            left_n = np.cumsum(count)[:-1]
            right_g = total_g - left_g
            right_h = total_h - left_h
            right_n = rows - left_n
            valid = (left_n >= min_child_weight) & (right_n >= min_child_weight) & (left_h > 1e-9) & (right_h > 1e-9)
            if not valid.any():
                continue
            gain = left_g ** 2 / (left_h + l2) + right_g ** 2 / (right_h + l2) - parent_score
            gain = np.where(valid, gain, -np.inf)
            index = int(np.argmax(gain))
            if gain[index] > best_gain:
                best_gain = float(gain[index])
                best_feature = int(feature)
                best_threshold = index + 1
        if best_feature < 0:
            self.value[node] = -total_g / (total_h + l2)
            return
        self.feature[node] = best_feature
        self.threshold[node] = best_threshold
        mask = binned[:, best_feature] < best_threshold
        self._grow(binned[mask], g[mask], h[mask], 2 * node + 1, level + 1,
                   depth, min_child_weight, l2, rng, feature_fraction)
        self._grow(binned[~mask], g[~mask], h[~mask], 2 * node + 2, level + 1,
                   depth, min_child_weight, l2, rng, feature_fraction)

    def predict(self, binned):
        node = np.zeros(len(binned), dtype=np.int64)
        active = self.feature[node] >= 0
        while active.any():
            rows = np.flatnonzero(active)
            feature = self.feature[node[rows]]
            go_left = binned[rows, feature] < self.threshold[node[rows]]
            node[rows] = np.where(go_left, 2 * node[rows] + 1, 2 * node[rows] + 2)
            active = self.feature[node] >= 0
        return self.value[node]


def train_gbdt(binned, y, trees=250, depth=3, lr=0.05, l2=1.0, min_child_weight=8.0,
               subsample=0.8, feature_fraction=0.5, seed=0, bins=64):
    rng = np.random.default_rng(seed)
    prior = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    base = np.log(prior / (1 - prior))
    scores = np.full(len(y), base)
    forest = []
    for _ in range(trees):
        probability = sigmoid(scores)
        gradient = probability - y
        hessian = np.maximum(probability * (1 - probability), 1e-9)
        rows = rng.choice(len(y), size=max(int(len(y) * subsample), 16), replace=False)
        tree = Tree(depth, bins)
        tree.fit(binned[rows], gradient[rows], hessian[rows], depth,
                 min_child_weight, l2, rng, feature_fraction)
        scores += lr * tree.predict(binned)
        forest.append(tree)
    return forest, base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v5_tree")
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    from src.v2.train import read_folds
    from src.v3.train import OUT, digest_ids

    shape = np.load(CACHE / "v3_shape_features.npz", allow_pickle=False)
    glcm_train, glcm_test = load_glcm()
    features_train = np.column_stack([shape["train"], glcm_train]).astype(np.float32)
    features_test = np.column_stack([shape["test"], glcm_test]).astype(np.float32)
    print(f"features: train {features_train.shape} test {features_test.shape}")

    data = np.load(CACHE / "train.npz", allow_pickle=False)
    ids = data["ids"]
    labels = data["labels"].astype(np.float64)
    folds = read_folds(ids, data["labels"])["fold"]
    test_ids = np.load(CACHE / "test.npz", allow_pickle=False)["ids"]

    target = OUT / args.tag
    target.mkdir(parents=True, exist_ok=True)
    report = {"tag": args.tag, "features": int(features_train.shape[1]),
              "glcm_features": int(glcm_train.shape[1]), "trees": args.trees,
              "depth": args.depth, "lr": args.lr, "folds": []}
    oof = np.full(len(labels), np.nan)
    test_accumulator = []
    for fold in range(5):
        train = folds != fold
        valid = ~train
        binned_train, binned_valid, binned_test = bin_features(
            features_train[train], features_train[valid], features_test)
        start = time.perf_counter()
        forest, base = train_gbdt(binned_train, labels[train], trees=args.trees,
                                  depth=args.depth, lr=args.lr, seed=100 + fold)
        score_valid = np.full(len(binned_valid), base)
        score_test = np.full(len(binned_test), base)
        for tree in forest:
            score_valid += args.lr * tree.predict(binned_valid)
            score_test += args.lr * tree.predict(binned_test)
        probability = sigmoid(score_valid)
        test_probability = sigmoid(score_test)
        oof[valid] = probability
        test_accumulator.append(test_probability)
        # Same on-disk contract as the v3 trainers, so the fusion tools can read it.
        np.savez(target / f"fold{fold}_predictions.npz",
                 valid_ids=ids[valid], probability=probability,
                 plain_probability=probability, fused_probability=probability,
                 test_ids=test_ids, test_probability=test_probability)
        entry = {"fold": fold, "train_ids_sha256": digest_ids(ids[train]),
                 "valid_rows": int(valid.sum()),
                 "val_log_loss": log_loss(labels[valid], probability),
                 "seconds": time.perf_counter() - start}
        report["folds"].append(entry)
        print(f"fold {fold}: loss={entry['val_log_loss']:.5f} ({entry['seconds']:.0f}s)", flush=True)
    report["oof_log_loss"] = log_loss(labels, oof)
    print(f"pooled OOF = {report['oof_log_loss']:.6f}")

    np.savez(target / "oof.npz", train_ids=ids, test_ids=test_ids,
             probability=oof, test_probability=np.mean(test_accumulator, axis=0))
    (target / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
