"""Append the round 19-22 findings to the iteration document."""
from __future__ import annotations

from copy import deepcopy

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

NEW_ROWS = [
    ("伪标签置信阈值 0.03（原）",
     "行数仅约 3,500；放宽到 0.25 后 6,260 行，集成 0.099368 → 0.098287"),
    ("训练预算 20 epoch（阈值放宽后）",
     "40 epoch 在 folds 0-1 为 0.15191，优于 20 epoch 的 0.16004；但五折仅换来集成 −0.0001"),
    ("去掉纯冰山角度组块的专家模型（借鉴第一名思路）",
     "单模型 0.1773 vs 0.1614；加入集成 0.098188 → 0.098435，无效"),
    ("分位数截断归一化（local_clip）",
     "fold0 0.17016 vs 0.15940，明显更差；极亮像素正是船的判别信号"),
    ("阈值 0.10 的多样性成员 d10",
     "单模型 0.16458，但加入集成 0.098188 → 0.098349"),
    ("贪心集成选择（26 成员，校准后）",
     "0.105607，劣于手工组合 0.101938"),
    ("机器生成行是否计分（控制实验）",
     "改动 3,659 行机器图预测（均值 0.087，最大 0.74），公私榜分数完全不变 → 不计分"),
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
    table = document.tables[3]
    existing = {row.cells[0].text.strip() for row in table.rows[1:]}
    added = 0
    for attempt, result in NEW_ROWS:
        if attempt in existing:
            continue
        table._tbl.append(deepcopy(table.rows[-1]._tr))
        row = table.rows[-1]
        set_cell(row.cells[0], attempt)
        set_cell(row.cells[1], result)
        added += 1
    print(f"added {added} rows; negative table now has {len(table.rows) - 1}")

    stages = document.tables[0]
    for row in stages.rows[1:]:
        if row.cells[0].text.strip().startswith("V5"):
            set_cell(row.cells[2], "0.0982")
            set_cell(row.cells[3], "当前最佳（调参+阈值+40epoch）")
    scores = document.tables[5]
    for row in scores.rows[1:]:
        if "v5_hp8v2" in row.cells[0].text:
            set_cell(row.cells[1], "0.08767")
            set_cell(row.cells[2], "0.11005")
    document.save(PATH)
    print("saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
