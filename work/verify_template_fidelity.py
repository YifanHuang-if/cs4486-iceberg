"""Prove the output keeps the template's format and has no placeholders left."""
from __future__ import annotations

import re

from pptx import Presentation

TEMPLATE = r"C:\Users\16018\cs4486-iceberg\report\Final_Presentation_Template_CS4486.pptx"
OUTPUT = r"C:\Users\16018\cs4486-iceberg\report\CS4486_Iceberg_Final.pptx"


def geometry(slide):
    return {
        shape.shape_id: (shape.left, shape.top, shape.width, shape.height, shape.shape_type)
        for shape in slide.shapes
    }


def main() -> int:
    template = Presentation(TEMPLATE)
    output = Presentation(OUTPUT)

    print(f"slides: template={len(template.slides)} output={len(output.slides)}")

    moved = 0
    added = 0
    for index, (source, target) in enumerate(zip(template.slides, output.slides), start=1):
        before = geometry(source)
        after = geometry(target)
        for shape_id, values in before.items():
            if shape_id not in after:
                print(f"  slide {index}: shape {shape_id} MISSING")
                moved += 1
            elif after[shape_id][:4] != values[:4]:
                print(f"  slide {index}: shape {shape_id} moved/resized")
                moved += 1
        extra = [sid for sid in after if sid not in before]
        added += len(extra)
    print(f"shapes moved or missing: {moved}")
    print(f"shapes added (pictures only): {added}")

    leftovers = []
    for index, slide in enumerate(output.slides, start=1):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            text = shape.text_frame.text
            if "[" in text or "Replace all gray" in text or "FINAL PRESENTATION TEMPLATE" in text:
                leftovers.append((index, text.strip()[:60]))
    print(f"placeholder text left: {len(leftovers)}")
    for item in leftovers:
        print("  ", item)

    words = 0
    for slide in output.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                words += len(re.findall(r"[A-Za-z0-9]+", shape.text_frame.text))
    print(f"word count: {words} (~{words / 130:.1f} min at 130 wpm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
