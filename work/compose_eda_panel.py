"""Compose one EDA panel image so the template's single slot is used well."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

EDA = Path(r"C:\Users\16018\cs4486-iceberg\results\eda")
OUT = Path(r"C:\Users\16018\cs4486-iceberg\report\fig_eda_panel.png")


def main() -> int:
    top = Image.open(EDA / "distributions.png").convert("RGB")
    bottom = Image.open(EDA / "missingness_and_band2.png").convert("RGB")

    width = max(top.width, bottom.width)
    bottom = bottom.resize((width, int(bottom.height * width / bottom.width)), Image.LANCZOS)

    gap = 24
    canvas = Image.new("RGB", (width, top.height + gap + bottom.height), "white")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height + gap))
    canvas.save(OUT, optimize=True)
    print(f"wrote {OUT} {canvas.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
