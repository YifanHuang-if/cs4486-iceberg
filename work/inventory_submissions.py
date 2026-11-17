"""Inventory every submission produced in this project, in creation order."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if (Path.cwd() / "work").exists() is False else Path.cwd()
ROOT = Path(r"C:\Users\16018\cs4486-iceberg")
SUB = ROOT / "work" / "submissions"

# Measured leaderboard scores, transcribed from the submission history the user shared.
LEADERBOARD = {
    "final_v1_fast": (0.11143, 0.12862),
    "v5_mix6": (0.08944, 0.10726),
    "v5_mix5": (0.09027, 0.10739),
    "v5_hp8": (0.08784, 0.10746),
    "v5_hp8v2": (0.08767, 0.11005),
    "v5_mix6_poolne": (0.08954, 0.10566),
    "v5_mix6_pool50": (0.09029, 0.10630),
    "v5_best": (0.08752, 0.10899),
    "v5_best_allrows": (0.08752, 0.10899),
}

# Nested cross-validation of the ensemble behind each file, where it was evaluated.
CV = {
    "v5_mix4": 0.103074,
    "v5_mix5": 0.102460,
    "v5_mix6": 0.101938,
    "v5_mix6_pool50": 0.101938,
    "v5_mix6_poolne": 0.101938,
    "v5_hp8": 0.099762,
    "v5_hp8v2": 0.099368,
    "v5_best": 0.098287,
    "v5_best_allrows": 0.098287,
    "v5_best_c001": 0.098287,
    "v5_best_c005": 0.098287,
}


def main() -> int:
    rows = []
    for path in sorted(SUB.glob("*.csv")):
        if path.stem.startswith("_"):
            continue
        manifest = SUB / f"{path.stem}.json"
        tags = None
        if manifest.exists():
            blob = json.loads(manifest.read_text(encoding="utf-8"))
            tags = blob.get("tags")
        rows.append((path.stat().st_mtime, path.stem, tags, path.stat().st_size))
    rows.sort()
    print(f"{'#':>2s} {'created':16s} {'file':22s} {'CV':>8s} {'private':>8s} {'public':>8s}")
    for index, (mtime, stem, tags, size) in enumerate(rows, start=1):
        from datetime import datetime
        stamp = datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")
        cv = CV.get(stem)
        lb = LEADERBOARD.get(stem)
        cv_text = f"{cv:.6f}" if cv is not None else "-"
        priv = f"{lb[0]:.5f}" if lb else "-"
        pub = f"{lb[1]:.5f}" if lb else "-"
        print(f"{index:2d} {stamp:16s} {stem:22s} {cv_text:>8s} {priv:>8s} {pub:>8s}")
    print(f"\ntotal csv files: {len(rows)}")
    print(f"submitted to Kaggle: {sum(1 for _, s, _, _ in rows if s in LEADERBOARD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
