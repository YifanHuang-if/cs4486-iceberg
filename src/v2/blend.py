"""Blend fold/seed ensembles and write a submission-ready probability file.

Dependency-free on purpose (numpy + standard library): the analysis stack is
large and slow to install on this machine, while the blend must stay runnable at
any time.

Reads ``work/models/<tag>`` outputs, aligns them on the frozen training order,
compares mean-probability, mean-logit and Platt-scaled blends on the held-out
predictions, and applies the selected rule to the test predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"
MODELS = ROOT / "work" / "models"
SUBMISSIONS = ROOT / "work" / "submissions"
FLOOR = 1e-7


def log_loss(labels, probability, floor: float = FLOOR) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), floor, 1 - floor)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def read_oof(path: Path) -> dict:
    with path.open(newline="") as handle:
        return {row["id"]: float(row["probability"]) for row in csv.DictReader(handle)}


def load_tag(tag: str, ids: np.ndarray):
    folder = MODELS / tag
    oof_files = sorted(folder.glob("oof_fold*.csv"))
    if not oof_files:
        raise ValueError(f"{tag} has no OOF files")
    mapping = {}
    for path in oof_files:
        mapping.update(read_oof(path))
    missing = [i for i in ids if i not in mapping]
    if missing:
        raise ValueError(f"{tag} is missing {len(missing)} training ids (folds incomplete)")
    probability = np.array([mapping[i] for i in ids], dtype=np.float64)
    test_files = sorted(folder.glob("test_fold*.npy"))
    if not test_files:
        raise ValueError(f"{tag} has no test predictions")
    test = np.mean([np.load(path) for path in test_files], axis=0)
    return probability, test


def platt(logits: np.ndarray, labels: np.ndarray, iterations: int = 100):
    """Newton iterations for a two-parameter logistic calibration on logits."""
    a, b = 1.0, 0.0
    for _ in range(iterations):
        z = a * logits + b
        p = 1.0 / (1.0 + np.exp(-z))
        weight = p * (1 - p)
        gradient = np.array([np.sum((p - labels) * logits), np.sum(p - labels)])
        hessian = np.array([
            [np.sum(weight * logits * logits), np.sum(weight * logits)],
            [np.sum(weight * logits), np.sum(weight)],
        ])
        step = np.linalg.solve(hessian + np.eye(2) * 1e-9, gradient)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(a), float(b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--name", default="blend")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    train = np.load(CACHE / "train.npz", allow_pickle=False)
    test = np.load(CACHE / "test.npz", allow_pickle=False)
    ids, labels, test_ids = train["ids"], train["labels"].astype(np.float64), test["ids"]

    oof_prob, oof_test, report = [], [], {}
    for tag in args.tags:
        probability, test_probability = load_tag(tag, ids)
        report[tag] = {"oof_log_loss": log_loss(labels, probability)}
        print(f"{tag}: oof {report[tag]['oof_log_loss']:.5f}", flush=True)
        oof_prob.append(probability)
        oof_test.append(test_probability)

    logit = lambda p: np.log(np.clip(p, FLOOR, 1 - FLOOR) / (1 - np.clip(p, FLOOR, 1 - FLOOR)))
    prob_mean = np.mean(oof_prob, axis=0)
    test_prob_mean = np.mean(oof_test, axis=0)
    logit_mean = np.mean([logit(p) for p in oof_prob], axis=0)
    test_logit_mean = np.mean([logit(p) for p in oof_test], axis=0)

    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    candidates = {
        "prob_mean": (log_loss(labels, prob_mean), prob_mean, test_prob_mean),
        "logit_mean": (log_loss(labels, sigmoid(logit_mean)), sigmoid(logit_mean), sigmoid(test_logit_mean)),
    }
    if args.calibrate:
        a, b = platt(logit_mean, labels)
        candidates["logit_mean_platt"] = (
            log_loss(labels, sigmoid(a * logit_mean + b)),
            sigmoid(a * logit_mean + b),
            sigmoid(a * test_logit_mean + b),
        )
        report["platt"] = {"a": a, "b": b}
    best = min(candidates, key=lambda key: candidates[key][0])
    loss, oof_best, test_best = candidates[best]
    print(f"\nbest blend: {best}  oof log loss {loss:.5f}")
    for key, (value, _, _) in candidates.items():
        print(f"  {key}: {value:.5f}")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    path = SUBMISSIONS / f"{args.name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "is_iceberg"])
        for sample_id, probability in zip(test_ids.tolist(), np.clip(test_best, 1e-6, 1 - 1e-6).tolist()):
            writer.writerow([sample_id, f"{probability:.10f}"])
    summary = {
        "name": args.name, "tags": args.tags, "per_tag": report, "selected": best,
        "candidates": {key: value[0] for key, value in candidates.items()},
        "oof_log_loss": loss, "rows": int(len(test_ids)),
        "prediction_mean": float(np.mean(test_best)), "oof_prediction_mean": float(np.mean(oof_best)),
    }
    (SUBMISSIONS / f"{args.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(SUBMISSIONS / f"{args.name}.oof.npy", oof_best)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
