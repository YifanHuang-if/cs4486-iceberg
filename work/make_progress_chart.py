"""Progress chart for the results slide: private loss by submitted version."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(r"C:\Users\16018\cs4486-iceberg\report\fig_private_progress.png")

VERSIONS = ["v1", "v6", "v7", "v8", "v11", "v13", "v19"]
PRIVATE = [0.23022, 0.09106, 0.09027, 0.08944, 0.08784, 0.08752, 0.08708]
LABELS = {
    "v1": "first CNN\n+ angle prior off",
    "v6": "stacked fusion",
    "v11": "tuned recipe",
    "v13": "soft labels",
    "v19": "angle calibration",
}


def main() -> int:
    figure, axes = plt.subplots(figsize=(9.0, 3.2), dpi=200)
    axes.plot(VERSIONS, PRIVATE, marker="o", linewidth=2.2, color="#1f4e79")
    for x, y in zip(VERSIONS, PRIVATE):
        axes.annotate(f"{y:.5f}", (x, y), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=8.5, color="#1f4e79")
    for key, note in LABELS.items():
        index = VERSIONS.index(key)
        axes.annotate(note, (VERSIONS[index], PRIVATE[index]), textcoords="offset points",
                      xytext=(0, -26), ha="center", fontsize=7.5, color="#666666")
    axes.axhline(0.08883, linestyle="--", linewidth=1.0, color="#b3373f")
    axes.text(len(VERSIONS) - 0.55, 0.08883 + 0.012, "historical Top-5  0.08883",
              fontsize=8, color="#b3373f", ha="right")
    axes.set_ylabel("Private Log Loss", fontsize=9)
    axes.set_ylim(0.076, 0.245)
    axes.tick_params(labelsize=8.5)
    axes.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
