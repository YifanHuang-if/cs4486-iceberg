"""Measure how much text each slide carries, as a proxy for density."""
from __future__ import annotations

import re

from pptx import Presentation

PATH = r"C:\Users\16018\Downloads\CS4486_Iceberg_Final_6min.pptx"


def main() -> int:
    presentation = Presentation(PATH)
    total = 0
    print(f"{'slide':>5s} {'words':>6s} {'boxes':>6s}")
    for index, slide in enumerate(presentation.slides, start=1):
        words = 0
        boxes = 0
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                boxes += 1
                words += len(re.findall(r"[A-Za-z0-9]+", shape.text_frame.text))
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        words += len(re.findall(r"[A-Za-z0-9]+", cell.text))
        total += words
        print(f"{index:5d} {words:6d} {boxes:6d}")
    print(f"\ntotal words: {total}")
    print(f"reading time at 130 wpm: {total / 130:.1f} min (budget is 6 min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
