"""Move the version19 subsections so they sit directly after section 2.3."""
from __future__ import annotations

from docx import Document

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"


def main() -> int:
    document = Document(PATH)
    paragraphs = list(document.paragraphs)

    start = next(i for i, p in enumerate(paragraphs)
                 if p.text.strip().startswith("2.3.4 version19"))
    stop = next(i for i, p in enumerate(paragraphs)
                if p.text.strip().startswith("3. 关键诊断"))
    block = paragraphs[start:stop]

    target = next(p for p in paragraphs if p.text.strip().startswith("2.4 突破性进展"))
    for paragraph in block:
        target._p.addprevious(paragraph._p)

    document.save(PATH)
    print(f"moved {len(block)} paragraphs before section 2.4")

    check = Document(PATH)
    print([p.text.strip() for p in check.paragraphs if p.style.name.startswith("Heading")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
