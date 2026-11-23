"""List the fold seed and the seeds of the seven members in the reported pipeline."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\16018\cs4486-iceberg")
MEMBERS = ["v5_shape", "hp_star", "hp_b", "hp_d", "pt25a", "pt25b", "stu5"]


def main() -> int:
    print("final ensemble members (version19):")
    for tag in MEMBERS:
        path = ROOT / "outputs/v3" / tag / "fold0_metrics.json"
        if not path.exists():
            print(f"  {tag:10s} (no fold0 metrics)")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        config = blob.get("config", {})
        print(f"  {tag:10s} seed={config.get('seed')}  epochs={config.get('epochs')} "
              f"dropout={config.get('dropout')} teacher={config.get('teacher')}")

    shape = ROOT / "src/v3/shape_model.py"
    if shape.exists():
        for line in shape.read_text(encoding="utf-8").splitlines():
            if "seed" in line.lower() and ("=" in line):
                print("  shape_model:", line.strip()[:110])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
