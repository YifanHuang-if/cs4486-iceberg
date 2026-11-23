"""Dump every piece of text on every slide, in reading order (top-to-bottom, left-to-right)."""
from __future__ import annotations

from pptx import Presentation

PATH = r"D:\Desktop\Group12_Project3.pptx"


def main() -> int:
    presentation = Presentation(PATH)
    for index, slide in enumerate(presentation.slides, start=1):
        items = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                items.append((round(shape.top / 914400, 2), round(shape.left / 914400, 2),
                              shape.text_frame.text.strip().replace("\n", " ⏎ ")))
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [" | ".join(cell.text.strip() for cell in row.cells)
                        for row in shape.table.rows]
                items.append((round(shape.top / 914400, 2), round(shape.left / 914400, 2),
                              "TABLE:: " + " ;; ".join(rows)))
        items.sort()
        print(f"\n===== slide {index} ({len(items)} text blocks) =====")
        for top, left, text in items:
            print(f"  [T{top:5.2f} L{left:5.2f}] {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
