"""Assemble a per-fold teacher tag from an existing ensemble of image models.

The V3 student trainer expects a teacher directory holding, for each fold, the
test-set probabilities produced by a model trained on that fold's training rows.
This script averages the fold-k test predictions of several tags into exactly
that layout, so heavy pseudo-label distillation can use the current ensemble
without retraining anything.

The teacher still only encodes fold-k information: every member's fold-k model
was fitted on the same training rows, so no outer validation label is involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.v2.train import read_folds
from src.v3.train import OUT, ROOT, digest_ids

CACHE = ROOT / "work" / "predictions"


def main() -> int:
    import sys
    tags = ["s101", "s202", "s404", "v5_shape", "stu1", "stu5"]
    target_name = "v5_teacher6"
    if len(sys.argv) > 1:
        target_name = sys.argv[1]
    if len(sys.argv) > 2:
        tags = sys.argv[2:]
    data = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
    ids = data["ids"]
    folds = read_folds(ids, data["labels"])["fold"]
    test_ids = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)["ids"]
    target = OUT / target_name
    target.mkdir(parents=True, exist_ok=True)

    for fold in range(5):
        members = []
        for tag in tags:
            path = CACHE / f"{tag}_fold{fold}.npy"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            members.append(np.load(path))
        probability = np.mean(members, axis=0)
        train = folds != fold
        valid = ~train
        valid_ids = ids[valid]
        # The teacher's own OOF vector is not used by the student, but the format
        # expects it; store the teacher's mean held-out probability per member.
        np.savez(target / f"fold{fold}_predictions.npz",
                 valid_ids=valid_ids,
                 probability=np.full(len(valid_ids), 0.5),
                 plain_probability=np.full(len(valid_ids), 0.5),
                 fused_probability=np.full(len(valid_ids), 0.5),
                 test_ids=test_ids,
                 test_probability=probability)
        metrics = {
            "fold": fold,
            "config": {"tag": "v5_teacher", "architecture": "ensemble", "teacher": "",
                       "members": tags},
            "train_ids_sha256": digest_ids(ids[train]),
            "val_log_loss": None,
            "val_fused_log_loss": None,
            "valid_rows": int(valid.sum()),
            "pseudo_rows": 0,
            "seconds": 0.0,
            "note": "Synthetic teacher assembled from ensemble fold predictions.",
        }
        (target / f"fold{fold}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"fold {fold}: teacher mean={probability.mean():.4f} rows={len(probability)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
