"""Inspect the newly uploaded document: structure, colours, and red passages."""
from __future__ import annotations

from collections import Counter

from docx import Document

PATH = r"D:\Desktop\CS4486_Project_Full_Process.docx"


def colour(run):
    try:
        rgb = run.font.color.rgb
    except Exception:
        return "AUTO"
    return str(rgb) if rgb is not None else "AUTO"


def main() -> int:
    document = Document(PATH)
    print(f"paragraphs={len(document.paragraphs)} tables={len(document.tables)}")
    counter = Counter()
    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            if run.text.strip():
                counter[colour(run)] += 1
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if run.text.strip():
                            counter[colour(run)] += 1
    print("colours:", dict(counter))
    print()
    for index, paragraph in enumerate(document.paragraphs):
        if not paragraph.text.strip():
            continue
        colours = {colour(run) for run in paragraph.runs if run.text.strip()}
        flag = "RED " if any(c not in ("AUTO", "000000", "595959") for c in colours) else "    "
        style = paragraph.style.name[:12]
        print(f"{index:3d} {flag}[{style:12s}] {paragraph.text.strip()[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
