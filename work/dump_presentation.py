"""Dump every slide of the presentation so the summary can cover it all."""
from __future__ import annotations

from pptx import Presentation

PATH = r"D:\Desktop\Group12_Project3.pptx"


def main() -> int:
    presentation = Presentation(PATH)
    for index, slide in enumerate(presentation.slides, start=1):
        print(f"\n===== slide {index} =====")
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    print(text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    print(" | ".join(cell.text.strip() for cell in row.cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
