"""Inventory the text shapes on the two improvement slides."""
from __future__ import annotations

from pptx import Presentation
from pptx.util import Emu

PATH = r"C:\Users\16018\Downloads\CS4486_Iceberg_Final_6min.pptx"


def main() -> int:
    presentation = Presentation(PATH)
    for index in (4, 5):
        slide = presentation.slides[index]
        print(f"===== slide {index + 1} =====")
        for shape in slide.shapes:
            if not shape.has_text_frame or not shape.text_frame.text.strip():
                continue
            top = shape.top / 914400
            left = shape.left / 914400
            first = shape.text_frame.paragraphs[0]
            size = None
            color = None
            if first.runs:
                size = first.runs[0].font.size.pt if first.runs[0].font.size else None
                try:
                    color = first.runs[0].font.color.rgb
                except Exception:
                    color = None
            print(f"  id={shape.shape_id:3d} L={left:5.2f} T={top:5.2f} "
                  f"sz={size} color={color} runs={len(first.runs)} | "
                  f"{shape.text_frame.text.strip()[:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
