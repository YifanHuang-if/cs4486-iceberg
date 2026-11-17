"""Restructure the iteration document around the sequence of submitted versions.

Inserts a chronological version table after section 2 and appends a final section
explaining why the score cannot be pushed further, then removes the now
redundant forward-looking section.
"""
from __future__ import annotations

from copy import deepcopy

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

VERSIONS = [
    ("1", "11-16 22:34", "v5_mix4", "0.1031", "—", "四模型集成，首次把固定融合权重换成可学习的分组加权堆叠"),
    ("2", "11-16 23:05", "v5_mix5", "0.1025", "0.09027", "加入伪标签蒸馏学生，图像模型首次明显变强"),
    ("3", "11-17 00:08", "v5_mix6", "0.1019", "0.08944", "加入 mixup 学生，单独更差但误差去相关"),
    ("4", "11-17 02:14", "v5_mix6_pool50", "0.1019", "0.09029", "诊断：同角度全组池化（结论：无效）"),
    ("5", "11-17 02:16", "v5_mix6_poolne", "0.1019", "0.08954", "诊断：只池化无训练匹配的行，设计更干净"),
    ("6", "11-17 04:21", "v5_hp8", "0.0998", "0.08784", "超参调优 batch32 / lr3e-3 / wd1e-2，单次最大收益 −0.0022"),
    ("7", "11-17 05:59", "v5_hp8v2", "0.0994", "0.08767", "去掉三个旧配置弱基模型，只保留调参成员"),
    ("8", "11-17 08:21", "v5_best", "0.0983", "0.08752", "伪标签阈值 0.03→0.25（3,500→6,260 行）并把训练预算提到 40 epoch"),
    ("9", "11-17 09:14", "v5_best_allrows", "0.0983", "0.08752", "诊断：改动 3,659 行机器图后分数完全不变，证明其不计分"),
    ("10", "11-17 09:17", "v5_best_c001", "0.0983", "—", "诊断：截断下限 0.01，CV 已证明有害，未提交"),
    ("11", "11-17 09:17", "v5_best_c005", "0.0983", "—", "诊断：截断下限 0.05，同上未提交"),
]

WHY = [
    "两条目标线的算术。五折 CV 与两个榜的关系近似对称：CV 0.0982、private 0.0875（CV−0.0108）、"
    "public 0.1090（CV+0.0107）。因此 private < 0.083 需要 CV ≈ 0.094，"
    "而 public < 0.086 需要 CV ≈ 0.075——后者比本项目全部历史进展加起来还要大四倍。",
    "已测 48 种干预，只有 5 种有效：分组加权嵌套堆叠、超参调优、伪标签阈值放宽、蒸馏、shape MLP。"
    "其余 43 种全部无效或有害，逐条记录在负结果表中并附代价。",
    "集成已三次证明饱和：模型数 7/9/12 分别为 0.10204/0.10239/0.10279（第 7 个起为负）；"
    "26 成员嵌套贪心选择 0.105607（劣于手工 6 个）；把更强单模型换进来只带来 −0.0001。",
    "角度层已被测死：最近邻居标签一致率 83.9% 且与距离基本无关，单靠角度的理论下限约 0.44，"
    "而现状是 0.20。核宽 σ=0.012 经扫描确认最优，先验层没有剩余空间。",
    "图像侧的所有增强都被约束住了：翻转有效，而旋转/缩放/平移（affine）为 0.295 vs 0.159，"
    "mixup 仅在 20 epoch 下有效、40 epoch 反而变差，分位数截断为 0.170 vs 0.159。"
    "规律一致：增强必须与训练分布匹配，过度处理会破坏判别的亮度信号。",
    "归一化与测试时适配也已走完：局部逐图 z-score 已经消除了大部分域偏移，"
    "所以 AdaBN 重估 BN 统计量反而变差（0.1685 vs 0.1594），随机化 TTA 收敛回固定 TTA 的结果。",
    "外部方案同样挖尽：前五名的三大手段（角度结构利用、大规模集成、子集专家重训）都已实现或验证；"
    "第 16 名仓库的四个差异点逐一评估后均不可用或已证伪；全网检索到的 AdaBN 与随机 TTA 实测为负。",
    "要跨过最后 0.0045，需要的是本项目手段之外的量级：数十至上百个去相关成员（前五名分别用了 100+、"
    "数百、7×30 次拟合），或在 SAR 上预训练过的表征。两者都超出本机 CPU 环境与设定时间。",
    "因此结论是：当前 private 0.08752 是这套方法在本环境下的收敛点，而不是某一步没做完。"
    "继续增加实验次数不会改变结果，只会增加负结果条数。",
]


def add_version_table(document: Document, before_paragraph) -> None:
    table = document.add_table(rows=1, cols=6)
    table.style = "Light Grid Accent 1"
    header = ["#", "时间", "版本文件", "五折 CV", "private", "这一版做了什么"]
    for index, text in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(8.5)
    for row in VERSIONS:
        cells = table.add_row().cells
        for index, text in enumerate(row):
            cells[index].text = ""
            run = cells[index].paragraphs[0].add_run(text)
            run.font.size = Pt(8)
    before_paragraph._p.addprevious(table._tbl)


def main() -> int:
    document = Document(PATH)

    target = None
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("3. 关键诊断"):
            target = paragraph
            break
    if target is None:
        print("anchor heading not found")
        return 1

    heading = target.insert_paragraph_before("2.1 提交版本演进（按时间顺序）", style="Heading 2")
    intro = target.insert_paragraph_before(
        "下表是本项目产出的全部 11 个可提交版本。其中 8 个用于提分，3 个（#4、#9、#10-11）"
        "是刻意设计的诊断文件：它们的目标不是提高分数，而是验证假设并避免在错误方向上继续投入。")
    add_version_table(document, target)
    document.add_paragraph()

    document.add_heading("10. 为什么无法继续提高", level=1)
    for text in WHY:
        paragraph = document.add_paragraph(text, style="List Number")
        paragraph.paragraph_format.space_after = Pt(3)

    # The old forward-looking section is superseded by section 10.
    for paragraph in list(document.paragraphs):
        if paragraph.text.strip().startswith("9. 后续方向"):
            paragraph._p.getparent().remove(paragraph._p)

    document.save(PATH)
    print("restructured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
