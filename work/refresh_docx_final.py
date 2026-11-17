"""Refresh the final-configuration and score sections of the iteration document."""
from __future__ import annotations

from copy import deepcopy

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

CONFIG = [
    "五个调参 CNN 成员：hp_star / hp_b / hp_d / e40a / pt25b（RadarV1 93k 参数）",
    "统一配方：batch 32、lr 3e-3、weight decay 1e-2、dropout 0.4、40 epoch、AdamW、余弦调度",
    "伪标签蒸馏：教师由 6 模型集成给出，置信阈值 0.25（约 6,260 行），权重 0.25",
    "一个 shape/statistics MLP（90 维辐照与几何特征，3 seed）与一个 mixup 学生（stu5）提供异类多样性",
    "融合：图像概率等权平均 → 折隔离两级角度先验（σ=0.012, α=0.1）→ 分组加权 logistic 堆叠",
]

SCORES = [
    ("v5_best（7 模型，当前最佳）", "0.08752", "0.10899"),
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

    # Replace the configuration bullets in place.
    targets = [p for p in document.paragraphs
               if p.style.name == "List Bullet" and p.text.strip() and
               (p.text.strip().startswith(("三个", "一个", "融合"))
                or "shape/statistics" in p.text or "mixup 学生" in p.text)]
    for paragraph, text in zip(targets, CONFIG):
        if paragraph.runs:
            paragraph.runs[0].text = text
            for run in paragraph.runs[1:]:
                run.text = ""
        else:
            paragraph.add_run(text)
    print(f"rewrote {min(len(targets), len(CONFIG))} configuration bullets")

    # Refresh the score table.
    scores = document.tables[5]
    existing = {row.cells[0].text.strip() for row in scores.rows[1:]}
    for name, private, public in SCORES:
        if name in existing:
            for row in scores.rows[1:]:
                if row.cells[0].text.strip() == name:
                    set_cell(row.cells[1], private)
                    set_cell(row.cells[2], public)
            continue
        scores._tbl.append(deepcopy(scores.rows[-1]._tr))
        row = scores.rows[-1]
        set_cell(row.cells[0], name)
        set_cell(row.cells[1], private)
        set_cell(row.cells[2], public)
    for row in scores.rows[1:]:
        if row.cells[0].text.strip().startswith("v5_mix4"):
            set_cell(row.cells[0], "v5_hp8（调参后 8 模型）")
            set_cell(row.cells[1], "0.08784")
            set_cell(row.cells[2], "0.10746")
    print(f"score table rows: {len(scores.rows) - 1}")

    document.save(PATH)
    print("saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
