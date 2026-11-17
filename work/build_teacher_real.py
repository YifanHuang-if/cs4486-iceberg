"""How much of the pseudo-label set is machine-generated test imagery?

The teacher selects test rows whose fused probability is extreme and whose image
probability agrees. Machine-generated rows carry random many-decimal angles, so
their labels are not real acquisitions. This measures how many such rows enter
the student's training set and writes a teacher variant that excludes them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.fusion import fuse, prior
from src.v3.train import OUT, ROOT

CACHE = ROOT / "work" / "predictions"
TAGS = ["s101", "s202", "s404", "v5_shape", "stu1"]
SOURCE_TEACHER = "v5_teacher"
TARGET_TEACHER = "v5_teacher_real"


def decimals(values):
    out = np.zeros(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        scaled, count = float(value), 0
        while count < 15 and abs(scaled - round(scaled)) > 1e-9:
            scaled *= 10.0
            count += 1
        out[index] = count
    return out


def main() -> int:
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"]
    labels = data["labels"].astype(np.float64)
    angles = data["angles"].astype(np.float64)
    folds = read_folds(ids, data["labels"])["fold"]
    test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)
    test_ids = test["ids"]
    test_angles = test["angles"].astype(np.float64)
    real = decimals(test_angles) <= 4

    target = OUT / TARGET_TEACHER
    target.mkdir(parents=True, exist_ok=True)
    report = {"test_rows": int(len(test_ids)), "real_rows": int(real.sum()),
              "machine_rows": int((~real).sum()), "folds": {}}

    for fold in range(5):
        train = folds != fold
        members = [np.load(CACHE / f"{tag}_fold{fold}.npy") for tag in TAGS]
        probability = np.mean(members, axis=0)
        ap, coverage = prior(angles[train], labels[train], test_angles)
        fused = fuse(probability, ap, coverage)
        selected = (fused < 0.03) | (fused > 0.97)
        selected &= (probability < 0.1) | (probability > 0.9)
        selected &= (fused > 0.5) == (probability > 0.5)

        selected_real = selected & real
        selected_machine = selected & ~real
        report["folds"][str(fold)] = {
            "selected_total": int(selected.sum()),
            "selected_real": int(selected_real.sum()),
            "selected_machine": int(selected_machine.sum()),
            "share_machine_of_selected": round(float(selected_machine.sum() / max(selected.sum(), 1)), 4),
        }
        print(f"fold {fold}: selected={selected.sum()} real={selected_real.sum()} "
              f"machine={selected_machine.sum()} "
              f"machine_share={selected_machine.sum() / max(selected.sum(), 1):.2%}")

        # Teacher variant: machine rows are pinned to 0.5 so they are never picked.
        restricted = probability.copy()
        restricted[~real] = 0.5
        valid = ~train
        valid_ids = ids[valid]
        np.savez(target / f"fold{fold}_predictions.npz",
                 valid_ids=valid_ids,
                 probability=np.full(len(valid_ids), 0.5),
                 plain_probability=np.full(len(valid_ids), 0.5),
                 fused_probability=np.full(len(valid_ids), 0.5),
                 test_ids=test_ids,
                 test_probability=restricted)
        metrics = {
            "fold": fold,
            "config": {"tag": TARGET_TEACHER, "architecture": "ensemble", "teacher": "",
                       "members": TAGS, "machine_rows_excluded": True},
            "train_ids_sha256": json.loads(
                (OUT / SOURCE_TEACHER / f"fold{fold}_metrics.json").read_text())["train_ids_sha256"],
            "val_log_loss": None,
            "val_fused_log_loss": None,
            "valid_rows": int(valid.sum()),
            "pseudo_rows": 0,
            "seconds": 0.0,
            "note": "Ensemble teacher restricted to real-angle test rows.",
        }
        (target / f"fold{fold}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (ROOT / "work/leak/pseudo_machine_share.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
