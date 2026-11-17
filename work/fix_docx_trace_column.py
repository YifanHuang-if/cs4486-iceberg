"""Restore the original filename column and correct the section intro.

The relabelling pass compared table proxies by identity, which does not hold for
python-docx table objects, so it also rewrote the trace column. The mapping file
is the source of truth for the original names.
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.shared import Pt

DOC = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"
MAPPING = Path(r"C:\Users\16018\cs4486-iceberg\work\submissions\_version_mapping.json")

INTRO = ("下表是本项目产出的全部 16 个可提交版本，按生成时间排序。其中 11 个用于提分或阶段推进，"
         "5 个（#9、#10、#14、#15、#16）是刻意设计的诊断文件：目标不是提高分数，"
         "而是验证假设并避免在错误方向上继续投入。")


def main() -> int:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    document = Document(DOC)

    table = document.tables[1]
    restored = 0
    for row in table.rows[1:]:
        name = row.cells[2].text.strip()
        original = mapping.get(name)
        if original and row.cells[3].text.strip() != original:
            cell = row.cells[3]
            cell.text = ""
            run = cell.paragraphs[0].add_run(original)
            run.font.size = Pt(7.8)
            restored += 1
    print(f"restored {restored} original file names")

    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("下表是本项目产出的全部"):
            paragraph.runs[0].text = INTRO
            for run in paragraph.runs[1:]:
                run.text = ""
            print("intro updated")
            break

    document.save(DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
