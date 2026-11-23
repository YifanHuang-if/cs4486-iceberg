"""Replace the stale error-diagnosis numbers in the report with verified ones.

All values below were recomputed from the seven members of the submitted
ensemble (CV 0.098287) and are reproducible with work/ground_truth.py,
work/audit_neighbour.py and work/audit_error_section.py.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt

TARGET = Path(r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx")
BACKUP = Path(r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.backup.docx")

TABLE_ROWS = [
    ("exact", "训练集存在完全相同角度", "756", "0.0213", "10.2%"),
    ("near", "只有邻近角度，无完全相同角度", "709", "0.1987", "89.3%"),
    ("far", "邻近角度也很远", "6", "0.0367", "0.1%"),
    ("missing", "角度缺失", "133", "0.0035", "0.3%"),
]

BULLETS = {
    "87% 的损失集中在 700 行 near 样本":
        "89% 的损失集中在 709 行 near 样本（几乎无精确匹配），这是唯一的战场。",
    "最近角度的邻居标签一致率只有 83.9%":
        "同角度样本的一致率是 98.5%（810 行），而只有邻近角度的 661 行降到 81.9%。"
        "在这 709 行上：只用角度损失 0.371，只用图像 0.255，融合后 0.199——两边都必需，单靠任一侧都更差。",
    "损失高度集中：最差 25 行占 43%":
        "损失高度集中：最差 25 行占 44.8%，最差 100 行占 80.8%。",
    "图像与角度先验方向冲突的 231 行":
        "图像与角度先验判断相反的 181 行占 49.0% 的损失。在这些行上图像更准（60.2% vs 39.8%），"
        "这正是 version19 把角度权重下调到 0.90 就能改善两个榜的依据。",
    "最近邻居标签一致率 83.9%":
        "同角度一致率 98.5%、仅邻近角度 81.9%；单靠角度在 near 行的损失是 0.371，现状是 0.199，",
    "已测 48 种干预":
        "已测约 38 种干预",
}


def set_text(paragraph, text) -> None:
    runs = paragraph.runs
    if not runs:
        runs = [paragraph.add_run()]
    runs[0].text = text
    for run in list(runs)[1:]:
        run._r.getparent().remove(run._r)


def main() -> int:
    import shutil

    shutil.copy2(TARGET, BACKUP)
    document = Document(TARGET)

    # 1. Decomposition table.
    table = None
    for candidate in document.tables:
        if candidate.rows[0].cells[0].text.strip() == "分组":
            table = candidate
            break
    if table is None:
        raise SystemExit("decomposition table not found")
    for row, values in zip(table.rows[1:], TABLE_ROWS):
        for cell, text in zip(row.cells, values):
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(9.5)
    print(f"decomposition table updated ({len(TABLE_ROWS)} rows)")

    # 2. Bullets that carry stale numbers.
    changed = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        for prefix, replacement in BULLETS.items():
            if text.startswith(prefix):
                set_text(paragraph, replacement)
                changed += 1
                break
    print(f"bullets rewritten: {changed}")

    document.save(TARGET)
    print(f"saved: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
