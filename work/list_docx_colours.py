"""List every distinct run colour in the document, paragraphs and tables alike."""
from __future__ import annotations

from collections import Counter

from docx import Document

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"


def run_colour(run):
    try:
        rgb = run.font.color.rgb
    except Exception:
        return "AUTO"
    return str(rgb) if rgb is not None else "AUTO"


def main() -> int:
    document = Document(PATH)
    counter = Counter()
    samples = {}

    def scan(paragraph, where):
        for run in paragraph.runs:
            if not run.text.strip():
                continue
            colour = run_colour(run)
            counter[colour] += 1
            samples.setdefault(colour, (where, run.text.strip()[:60]))

    for index, paragraph in enumerate(document.paragraphs):
        scan(paragraph, f"para {index}")
    for t_index, table in enumerate(document.tables):
        for r_index, row in enumerate(table.rows):
            for c_index, cell in enumerate(row.cells):
                for paragraph in cell.paragraphs:
                    scan(paragraph, f"table {t_index} r{r_index} c{c_index}")

    for colour, count in counter.most_common():
        where, text = samples[colour]
        print(f"{colour:10s} {count:4d}  first@ {where}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
