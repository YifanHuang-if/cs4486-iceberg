"""Add a breakthrough-only summary of the version history as section 2.4."""
from __future__ import annotations

from docx import Document
from docx.shared import Inches, Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

STEPS = [
    ("起点", "version1", "0.23022", "—", "首发 CNN：4 残差挤圧激励块 + 8 视角 TTA"),
    ("角度层 v1", "version2", "0.11970", "−0.11052",
     "把 inc_angle 当证据而非特征：按角度组用训练标签估计 P(冰山)，log-odds 融合"),
    ("角度层 v2", "version5", "0.11143", "−0.00827",
     "覆盖率从 57% 拉到 96.8%：无精确匹配时改用高斯核先验"),
    ("分组加权堆叠", "version6", "0.09106", "−0.02037",
     "把固定融合权重换成可学习的第二层，并按「有/无精确匹配」分组给系数"),
    ("蒸馏 + mixup", "version7–8", "0.08944", "−0.00162",
     "软伪标签蒸馏改善无角度行；mixup 学生单独更差但误差去相关"),
    ("超参调优", "version11", "0.08784", "−0.00160",
     "batch 64→32、lr 1.5e-3→3e-3、weight decay 1e-4→1e-2，此前从未调过"),
    ("阈值 + 预算 + 去弱成员", "version12–13", "0.08752", "−0.00032",
     "伪标签阈值 0.03→0.25（3,500→6,260 行）、训练 20→40 epoch、剔除旧弱基模型"),
    ("角度权重标定", "version19", "0.08708", "−0.00044",
     "按排行榜把两个先验系数缩放 0.90，两榜同时改善"),
]

NOTES = [
    "30 次提交里只有这 8 步真正改变了分数，其余或是诊断、或是无效尝试。"
    "累计从 0.23022 降到 0.08708，降幅 62%；其中角度层与堆叠两步贡献 0.131，占全部改善的 92%。",
    "贡献高度不平衡是本项目最重要的结论之一：真正起作用的是「发现数据结构」"
    "（inc_angle 近似过境批次编号）与「利用该结构的融合形式」，"
    "而不是模型容量、集成规模或训练技巧。",
    "后四步合计只贡献 0.00396（2.8%），且每一步的代价都在上升——"
    "这是收敛的典型特征，也是第 9 节判定无法继续提高的依据。",
]


def main() -> int:
    document = Document(PATH)
    anchor = next(p for p in document.paragraphs if p.text.strip().startswith("3. 关键诊断"))

    anchor.insert_paragraph_before("2.4 突破性进展：八步改变了分数", style="Heading 2")
    anchor.insert_paragraph_before(
        "下表只列出真正推动分数的版本，诊断性提交与无效尝试不在此列（它们记录在第 5 节）。"
        "每一步给出对应版本、private 分数与相对上一步的增量。")

    table = document.add_table(rows=len(STEPS) + 1, cols=5)
    table.style = "Light Grid Accent 1"
    header = ["步骤", "版本", "private", "增量", "关键做法"]
    for index, text in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(8)
    for row_index, values in enumerate(STEPS, start=1):
        for column, text in enumerate(values):
            cell = table.rows[row_index].cells[column]
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(7.8)
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    for element in tbl_pr.findall(qn("w:tblLayout")):
        tbl_pr.remove(element)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    for index, width in enumerate([Inches(1.15), Inches(0.72), Inches(0.62),
                                   Inches(0.62), Inches(3.39)]):
        table.columns[index].width = width
        for cell in table.columns[index].cells:
            cell.width = width
    anchor._p.addprevious(table._tbl)

    for note in NOTES:
        paragraph = anchor.insert_paragraph_before(note, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(2)

    document.save(PATH)
    print("added section 2.4 with", len(STEPS), "steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
