"""Dump every template text box with its guide text so each slot can be filled."""
from __future__ import annotations

from pptx import Presentation

PATH = r"D:\Desktop\3semesterA\CS4886\projet\Final_Presentation_Template_CS4486.pptx"


def main() -> int:
    presentation = Presentation(PATH)
    for index, slide in enumerate(presentation.slides, start=1):
        print(f"\n===== slide {index} =====")
        for shape in slide.shapes:
            if not shape.has_text_frame or not shape.text_frame.text.strip():
                continue
            text = shape.text_frame.text.strip().replace("\n", " ⏎ ")
            if len(text) > 120:
                text = text[:120] + "…"
            print(f"  id={shape.shape_id:3d} T={shape.top / 914400:5.2f} "
                  f"L={shape.left / 914400:5.2f} | {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
