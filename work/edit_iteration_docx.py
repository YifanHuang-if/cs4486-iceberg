"""Append the round 14-18 findings to the iteration document.

The generator script was corrupted by an encoding round trip, so the document is
edited in place instead of being rebuilt. Existing formatting and the paragraphs
that are still correct are left untouched.
"""
from __future__ import annotations

from copy import deepcopy

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

NEW_ROWS = [
    ("权重 EMA（衰减 0.999 / 0.995 / 0.990）",
     "0.19222 / 0.16864 / 0.16891，均不如无 EMA 的 0.16631；余弦调度已把学习率退火到 1%"),
    ("SGD + momentum 0.9（lr 0.03 / 0.06）",
     "0.2423 / 0.2478，远差于 AdamW；20 epoch 预算内无法收敛"),
    ("网络宽度加宽 24-48-96-128 / 收窄 12-24-48-72",
     "0.16724 / 0.17538，默认 16-32-64-96 最优"),
    ("更强教师（由调参后 8 模型重建）",
     "学生 0.17696 vs 0.16631，更强教师产出更差学生，学进了教师系统性错误"),
    ("伪标签权重 0.5",
     "0.17280 vs 0.16631"),
]


def set_cell(cell, text: str, size: float = 9.0) -> None:
    paragraph = cell.paragraphs[0]
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
        paragraph.runs[0].font.size = Pt(size)
    else:
        run = paragraph.add_run(text)
        run.font.size = Pt(size)


def main() -> int:
    document = Document(PATH)

    # 1. Append the new negative rows, cloning the last row's formatting.
    table = document.tables[3]
    for attempt, result in NEW_ROWS:
        template = table.rows[-1]._tr
        new_tr = deepcopy(template)
        table._tbl.append(new_tr)
        row = table.rows[-1]
        set_cell(row.cells[0], attempt)
        set_cell(row.cells[1], result)
    print(f"negative table now has {len(table.rows) - 1} rows")

    # 2. Update the V5 stage row in the iteration table.
    stages = document.tables[0]
    for row in stages.rows[1:]:
        if row.cells[0].text.strip().startswith("V5"):
            set_cell(row.cells[2], "0.0994")
            set_cell(row.cells[3], "当前最佳（超参调优后）")
            print("updated V5 stage row")

    # 3. Add the tuned submission to the score table.
    scores = document.tables[5]
    template = scores.rows[-1]._tr
    scores._tbl.append(deepcopy(template))
    row = scores.rows[-1]
    set_cell(row.cells[0], "v5_hp8v2（调参后 8 模型）")
    set_cell(row.cells[1], "待测")
    set_cell(row.cells[2], "待测")
    print(f"score table now has {len(scores.rows) - 1} rows")

    document.save(PATH)
    print(f"saved {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
