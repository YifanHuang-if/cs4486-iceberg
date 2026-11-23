"""Align the report's diagnosis section with the deck's figures.

The deck reports the six-member ensemble figures (CV 0.1019). Those are restored
here so the two documents agree, with one footnote recording which ensemble the
numbers belong to so the report stays internally consistent with its own CV.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.shared import Pt

PATH = Path(r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx")
BACKUP = Path(r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.backup2.docx")

TABLE_ROWS = [
    ("exact", "训练集存在完全相同角度", "756", "0.0223", "10.3%"),
    ("near", "只有邻近角度，无完全相同角度", "700", "0.2038", "87.2%"),
    ("far", "邻近角度也很远", "15", "0.2202", "2.0%"),
    ("missing", "角度缺失", "133", "0.0054", "0.4%"),
]

BULLETS = {
    "89% 的损失集中在":
        "87% 的损失集中在 700 行 near 样本，这是唯一的战场。",
    "训练集里，同角度样本的一致率":
        "最近角度的邻居标签一致率 83.9%，且与距离基本无关；仅靠角度在这 700 行上的理论下限约 0.44，"
        "比现状 0.20 还差。所以角度层已被榨干，瓶颈是图像模型在「没有同过境伙伴」样本上的泛化。",
    "损失高度集中：最差 25 行占 44.8%":
        "损失高度集中：最差 25 行占 43%，最差 100 行占 80%。",
    "图像与角度先验判断相反的 181 行":
        "图像与角度先验方向冲突的 231 行占 50.7% 损失；在这些行上图像模型准确率仅 40.7%，"
        "而先验为 59.3%，说明图像模型恰在困难样本上系统性犯错。",
    "角度层已被用尽":
        "角度层已被测死：最近邻居标签一致率 83.9% 且与距离基本无关，单靠角度的理论下限约 0.44，"
        "而现状是 0.20。核宽 σ=0.012 经扫描确认最优，先验层没有剩余空间。",
}

FOOTNOTE = ("注：本节分组与占比按 6 成员集成的折外预测统计（CV 0.1019），"
            "与最终提交的 7 成员集成（CV 0.0983）存在约 1–2 个百分点的差异，结论方向不变。")


def set_text(paragraph, text) -> None:
    runs = paragraph.runs
    if not runs:
        runs = [paragraph.add_run()]
    runs[0].text = text
    for run in list(runs)[1:]:
        run._r.getparent().remove(run._r)


def main() -> int:
    shutil.copy2(PATH, BACKUP)
    document = Document(PATH)

    table = None
    for candidate in document.tables:
        if candidate.rows[0].cells[0].text.strip() == "分组":
            table = candidate
            break
    if table is None:
        raise SystemExit("table not found")
    for row, values in zip(table.rows[1:], TABLE_ROWS):
        for cell, text in zip(row.cells, values):
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(9.5)
    print("table restored to deck figures")

    changed = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        for prefix, replacement in BULLETS.items():
            if text.startswith(prefix):
                set_text(paragraph, replacement)
                changed += 1
                break
    print(f"bullets restored: {changed}")

    # One footnote so the report does not contradict its own CV.
    anchor = None
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("把 1,604 行按角度证据强弱分为四组"):
            anchor = paragraph
            break
    if anchor is not None:
        note = anchor.insert_paragraph_before(FOOTNOTE)
        for run in note.runs:
            run.font.size = Pt(9)
            run.italic = True
        print("footnote added")

    document.save(PATH)
    print(f"saved: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
