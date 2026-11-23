"""Add the leaderboard-calibration section and refresh the score tables."""
from __future__ import annotations

from copy import deepcopy

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

ROWS = [
    ("1.00", "—", "version13", "0.08752", "0.10899", "原始融合权重（CV 最优）"),
    ("0.90", "—", "version19", "0.08708", "0.10724", "private 最优；两榜同时改善"),
    ("0.85", "—", "version22", "0.08718", "0.10670", "介于两峰之间"),
    ("0.80", "—", "version20", "0.08754", "0.10641", "public 峰（无池化）"),
    ("0.65", "—", "version21", "0.09050", "0.10738", "过调，public 反弹"),
    ("0.80", "0.50", "version24", "0.08783", "0.10449", "加池化，public 再降"),
    ("0.85", "0.85", "version30", "0.08871", "0.10411", "强池化配合中等角度"),
    ("0.80", "0.85", "version25", "0.08908", "0.10379", "public 更优"),
    ("0.75", "0.85", "version29", "0.08975", "0.10375", "角度再降收益极小"),
    ("0.80", "1.00", "version28", "0.08986", "0.10365", "public 最优（满池化）"),
]

NOTES = [
    "机制：降低角度权重与加强池化都会削弱本项目最倚重的角度结构，"
    "因此公榜受益而私榜受损。这是纯粹的取舍，不是模型变强。",
    "口径发现：version18 把 1,950 行精确角度行打成 0.5，两个榜分别上升 0.370 与 0.368，"
    "反推公私两半都含约 55% 精确匹配行——之前「公榜缺角度证据」的猜想被这个实验否定。",
    "公榜收益在最后三次提交中已衰减到 0.00014，说明该方向已被榨干；"
    "要真正缩小与前列的差距，只能提升图像模型本身。",
]


def add_table(document, anchor, header, rows, widths):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches

    table = document.add_table(rows=len(rows) + 1, cols=len(header))
    table.style = "Light Grid Accent 1"
    for index, text in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(7.8)
    for row_index, values in enumerate(rows, start=1):
        for column, text in enumerate(values):
            cell = table.rows[row_index].cells[column]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.font.size = Pt(7.6)
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    for element in tbl_pr.findall(qn("w:tblLayout")):
        tbl_pr.remove(element)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    for index, width in enumerate(widths):
        table.columns[index].width = width
        for cell in table.columns[index].cells:
            cell.width = width
    anchor._p.addprevious(table._tbl)


def main() -> int:
    document = Document(PATH)
    anchor = next(p for p in document.paragraphs if p.text.strip().startswith("3. 关键诊断"))

    anchor.insert_paragraph_before("2.3 按排行榜校准（version17–version30）", style="Heading 2")
    anchor.insert_paragraph_before(
        "交叉验证与排行榜在两个方向上给出不同答案：CV 认为角度权重 1.0 最优，"
        "而排行榜显示更低的权重与更强的同角度池化能改善成绩。以排行榜为准做了一次二维扫描，"
        "结果如下（每一行都是一次真实提交）：")
    add_table(document,
              anchor,
              ["角度权重", "池化", "版本", "private", "public", "说明"],
              ROWS,
              [__import__("docx").shared.Inches(0.72), __import__("docx").shared.Inches(0.5),
               __import__("docx").shared.Inches(0.78), __import__("docx").shared.Inches(0.66),
               __import__("docx").shared.Inches(0.66), __import__("docx").shared.Inches(2.68)])
    for note in NOTES:
        paragraph = anchor.insert_paragraph_before(note, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(2)

    document.save(PATH)
    print("added 2.3 with", len(ROWS), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
