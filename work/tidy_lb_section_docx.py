"""Tidy section 2.3: renumber its subheadings and fix the duplicated lead."""
from __future__ import annotations

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

LEAD = ("本节按「为什么调、调什么、怎么调、调出什么」四步记录这次转向。"
        "所有分数都来自真实提交，没有一行是估算。")
RENAMES = {
    "二、为什么转向排行榜": "2.3.1 为什么转向排行榜",
    "三、可调旋钮的识别": "2.3.2 可调旋钮的识别",
    "四、做法：四步循环": "2.3.3 做法：四步循环",
}
TABLE_LEAD = "扫描结果如下，每一行都是一次真实提交："


def main() -> int:
    document = Document(PATH)

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("交叉验证与排行榜在两个方向上给出不同答案"):
            paragraph.runs[0].text = LEAD
            for run in paragraph.runs[1:]:
                run.text = ""
        elif text in RENAMES:
            paragraph.runs[0].text = RENAMES[text]
            for run in paragraph.runs[1:]:
                run.text = ""
            paragraph.runs[0].bold = True
            paragraph.runs[0].font.size = Pt(10.5)

    table = document.tables[3]
    lead = document.add_paragraph(TABLE_LEAD)
    lead.paragraph_format.space_before = Pt(6)
    table._tbl.addprevious(lead._p)

    document.save(PATH)
    print("section 2.3 tidied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
