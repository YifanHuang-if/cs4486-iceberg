"""List every completed fold-0 metric with its tag, sorted by image loss."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    rows = []
    for path in sorted((ROOT / "outputs/v3").glob("*/fold0_metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        image = record.get("val_log_loss")
        fused = record.get("val_fused_log_loss")
        if image is None:
            continue
        rows.append((image, path.parent.name, fused, record.get("config", {}).get("epochs"),
                     record.get("config", {}).get("dropout"), record.get("config", {}).get("augmentation"),
                     record.get("config", {}).get("mixup", 0.0)))
    rows.sort()
    print(f"{'tag':14s} {'image':>8s} {'fused':>8s} {'epochs':>7s} {'drop':>5s} {'aug':>7s} {'mixup':>6s}")
    for image, tag, fused, epochs, dropout, augmentation, mixup in rows:
        fused_text = f"{fused:8.5f}" if isinstance(fused, float) else "     n/a"
        print(f"{tag:14s} {image:8.5f} {fused_text} {str(epochs):>7s} {str(dropout):>5s} "
              f"{str(augmentation):>7s} {str(mixup):>6s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
