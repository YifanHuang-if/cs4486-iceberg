"""List every placeholder shape in the pristine template, with ids and text."""
from __future__ import annotations

from pptx import Presentation

SRC = r"C:\Users\16018\cs4486-iceberg\report\Final_Presentation_Template_CS4486.pptx"


def main() -> int:
    presentation = Presentation(SRC)
    for index, slide in enumerate(presentation.slides, start=1):
        print(f"\n### slide {index}")
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            text = shape.text_frame.text.strip()
            if not text:
                continue
            flag = "*" if "[" in text else " "
            print(f"{flag} id={shape.shape_id:3d} L={shape.left / 914400:5.2f} "
                  f"T={shape.top / 914400:5.2f} para={len(shape.text_frame.paragraphs)} | "
                  f"{text.replace(chr(10), ' ⏎ ')[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
