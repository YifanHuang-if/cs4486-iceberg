"""Draw a simple progress chart without matplotlib."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(r"C:\Users\16018\cs4486-iceberg\report\fig_progress.png")

POINTS = [("v6", 0.09106), ("v7", 0.09027), ("v8", 0.08944),
          ("v11", 0.08784), ("v13", 0.08752), ("v19", 0.08708)]
LO, HI = 0.0865, 0.0920


def main() -> int:
    width, height = 2200, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    left, right, top, bottom = 210, width - 70, 70, height - 120

    def y_of(value: float) -> float:
        return bottom - (value - LO) / (HI - LO) * (bottom - top)

    # horizontal grid + labels
    for step in range(6):
        value = LO + (HI - LO) * step / 5
        y = y_of(value)
        draw.line([(left, y), (right, y)], fill=(226, 230, 236), width=2)
        draw.text((60, y - 10), f"{value:.3f}", fill=(90, 100, 115))

    step_x = (right - left) / (len(POINTS) - 1)
    coords = []
    for index, (label, value) in enumerate(POINTS):
        x = left + index * step_x
        coords.append((x, y_of(value)))

    draw.line(coords, fill=(31, 78, 121), width=5)
    for (x, y), (label, value) in zip(coords, POINTS):
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(31, 78, 121))
        draw.text((x - 26, y - 36), f"{value:.5f}", fill=(31, 78, 121))
        draw.text((x - 12, bottom + 18), label, fill=(60, 70, 85))

    # reference line for the historical top-5
    target_y = y_of(0.08883)
    for x in range(left, right, 18):
        draw.line([(x, target_y), (x + 9, target_y)], fill=(179, 55, 63), width=3)
    draw.text((right - 320, target_y - 26), "historical Top-5  0.08883", fill=(179, 55, 63))

    draw.text((left, 18), "Private Log Loss by submitted version", fill=(25, 38, 62))
    image.save(OUT, optimize=True)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

