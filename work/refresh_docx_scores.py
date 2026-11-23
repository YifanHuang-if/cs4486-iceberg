"""Refresh the score table and the honest-boundary numbers after the LB sweep."""
from __future__ import annotations

from copy import deepcopy

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

SCORES = [
    ("version5（final_v1_fast，M5 角度层 v2）", "0.11143", "0.12862"),
    ("version7（v5_mix5）", "0.09027", "0.10739"),
    ("version8（v5_mix6）", "0.08944", "0.10726"),
    ("version11（v5_hp8，超参调优）", "0.08784", "0.10746"),
    ("version12（v5_hp8v2）", "0.08767", "0.11005"),
    ("version13（v5_best，angle 1.00）", "0.08752", "0.10899"),
    ("version19（angle 0.90）", "0.08708", "0.10724"),
    ("version22（angle 0.85）", "0.08718", "0.10670"),
    ("version20（angle 0.80）", "0.08754", "0.10641"),
    ("version24（angle 0.80 + 池化 0.50）", "0.08783", "0.10449"),
    ("version25（angle 0.80 + 池化 0.85）", "0.08908", "0.10379"),
    ("version28（angle 0.80 + 池化 1.00）", "0.08986", "0.10365"),
]

REPLACE = {
    "目标 private < 0.083 尚未达到，缺口 0.0045；public < 0.086 的缺口为 0.023，":
        "目标 private < 0.083 尚未达到，当前最优 0.08708，缺口 0.00408；"
        "public < 0.086 的当前最优为 0.10365，缺口 0.01765，",
    "CV 0.0982 与 private 0.0875 差 0.0108，而 public 0.1090 高出 0.0107——":
        "CV 0.0982 与 private 0.0871 差 0.0111，而 public 0.1037 高出 0.0055——",
    "两条目标线的算术。五折 CV 与两个榜的关系近似对称：CV 0.0982、private 0.0875（CV−0.0108）、"
    "public 0.1090（CV+0.0107）。因此 private < 0.083 需要 CV ≈ 0.094，"
    "而 public < 0.086 需要 CV ≈ 0.075——后者比本项目全部历史进展加起来还要大四倍。":
        "两条目标线的算术。按当前最优提交，private 0.08708 距 0.083 差 0.00408，"
        "public 0.10365 距 0.086 差 0.01765。按榜校准已把公榜从 0.10899 压到 0.10365（−0.0053），"
        "但最后三次提交只再赚 0.00014，说明该方向已被榨干；剩下的差距只能靠图像模型本身。",
}


def main() -> int:
    document = Document(PATH)
    table = document.tables[-1]
    while len(table.rows) > len(SCORES) + 1:
        table._tbl.remove(table.rows[-1]._tr)
    while len(table.rows) < len(SCORES) + 1:
        table.add_row()
    for index, values in enumerate(SCORES, start=1):
        cells = table.rows[index].cells
        for column, text in enumerate(values):
            cells[column].text = ""
            run = cells[column].paragraphs[0].add_run(text)
            run.font.size = Pt(7.8)
    print(f"score table now {len(table.rows) - 1} rows")

    changed = 0
    for paragraph in document.paragraphs:
        for old, new in REPLACE.items():
            if old in paragraph.text:
                if paragraph.runs:
                    paragraph.runs[0].text = new
                    for run in paragraph.runs[1:]:
                        run.text = ""
                changed += 1
                break
    print(f"updated {changed} boundary statements")
    document.save(PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
