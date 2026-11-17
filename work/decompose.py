"""Decompose the fused loss: how much does the angle prior and the image model each buy?

Reports the loss of the prior alone, the model alone, and the fused prediction on
the same leave-one-out rows, so the ceiling of the angle layer is explicit.
"""
import json
from pathlib import Path

import numpy as np

from src.v2.angle_fusion import bayes_fusion, log_loss
from src.v2.angle_radius import AngleIndex

ROOT = Path(__file__).resolve().parents[1]
train = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
labels = train["labels"].astype(np.float64)
index = AngleIndex(train["angles"], labels)

rows = np.arange(len(labels))
position = index.position
has_angle = position >= 0

exact = np.full(len(labels), np.nan)
kernel = np.full(len(labels), np.nan)
exact[has_angle] = index.exact_prior(index.angles[position[has_angle]], alpha=0.1, exclude=position[has_angle])
kernel[has_angle] = index.kernel_prior(index.angles[position[has_angle]], 0.02, 0.1, exclude=position[has_angle])
prior = np.where(np.isfinite(exact), exact, kernel)
covered = np.isfinite(prior)

report = {
    "rows": int(len(labels)),
    "coverage": float(covered.mean()),
    "prior_alone_loss_all": log_loss(labels[covered], prior[covered]),
    "prior_alone_loss_covered": log_loss(labels[covered], prior[covered]),
}
for name in ("fast_s0",):
    oof = np.load(ROOT / f"work/submissions/{name}.oof.npy")
    fused = bayes_fusion(oof, prior, 0.6)
    report[name] = {
        "model_loss_all": log_loss(labels, oof),
        "model_loss_covered": log_loss(labels[covered], oof[covered]),
        "model_loss_uncovered": log_loss(labels[~covered], oof[~covered]),
        "fused_loss_covered": log_loss(labels[covered], fused[covered]),
        "fused_loss_uncovered": log_loss(labels[~covered], fused[~covered]),
        "fused_loss_all": log_loss(labels, fused),
    }
print(json.dumps(report, indent=2))
(ROOT / "work/leak/decomposition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
