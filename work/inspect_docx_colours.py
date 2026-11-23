"""Show every paragraph with the colours used, so the red text can be identified."""
from __future__ import annotations

from docx import Document

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"


def colour_of(run):
    try:
        rgb = run.font.color.rgb
    except Exception:
        return None
    return str(rgb) if rgb is not None else None


def main() -> int:
    document = Document(PATH)
    for index, paragraph in enumerate(document.paragraphs):
        if not paragraph.text.strip():
            continue
        colours = []
        for run in paragraph.runs:
            if not run.text.strip():
                continue
            colours.append((colour_of(run), run.bold, run.text[:40]))
        reds = [c for c in colours if c[0] and c[0] not in ("000000", "172238", "19263E")]
        marker = " <<RED" if reds else ""
        style = paragraph.style.name[:14]
        print(f"{index:3d} [{style:14s}]{marker} {paragraph.text.strip()[:90]}")
        if reds:
            for colour, bold, text in reds[:3]:
                print(f"        colour={colour} bold={bold} | {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
