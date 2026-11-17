"""Rebuild the version table for sixteen versions and merge the earlier document.

The earlier document covered milestones M1-M6, which correspond to version1 to
version5. Their detail is merged as a subsection so the whole history reads in
one order.
"""
from __future__ import annotations

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

HEADER = ["#", "时间", "文件", "原文件名", "五折 OOF", "private", "这一版做了什么"]
ROWS = [
    ("1", "11-16 17:06", "version1", "fast_s0", "0.25381", "—",
     "M1 首发 CNN：4 个残差挤压激励块 + stride-2 stem（954k 参数），8 视角 TTA"),
    ("2", "11-16 17:17", "version2", "final_fast_s0", "0.12109", "—",
     "M4 加角度层 v1：精确同角度组先验 + log-odds 融合，损失一次降 0.118"),
    ("3", "11-16 17:36", "version3", "v1_s0", "0.24804", "—",
     "M2 换成 93k 紧凑 CNN，证明容量不是瓶颈且训练快约 3 倍"),
    ("4", "11-16 17:36", "version4", "v1_fast", "0.23955", "—",
     "M3 两模型 logit 空间融合 + Platt 校准"),
    ("5", "11-16 17:36", "version5", "final_v1_fast", "0.11897", "0.11143",
     "M5 角度层 v2：高斯核 + 两级先验，真实测试行覆盖率 57%→96.8%"),
    ("6", "11-16 22:34", "version6", "v5_mix4", "0.1031", "—",
     "四模型集成，把固定融合权重换成可学习的分组加权堆叠"),
    ("7", "11-16 23:05", "version7", "v5_mix5", "0.1025", "0.09027",
     "加入伪标签蒸馏学生，图像模型首次明显变强"),
    ("8", "11-17 00:08", "version8", "v5_mix6", "0.1019", "0.08944",
     "加入 mixup 学生：单独更差，但与已有成员误差去相关"),
    ("9", "11-17 02:14", "version9", "v5_mix6_pool50", "0.1019", "0.09029",
     "诊断：同角度全组池化（结论：无效）"),
    ("10", "11-17 02:16", "version10", "v5_mix6_poolne", "0.1019", "0.08954",
     "诊断：只池化没有训练匹配的行，设计更干净"),
    ("11", "11-17 04:21", "version11", "v5_hp8", "0.0998", "0.08784",
     "超参调优 batch32 / lr3e-3 / wd1e-2，单次最大收益 −0.0022"),
    ("12", "11-17 05:59", "version12", "v5_hp8v2", "0.0994", "0.08767",
     "去掉三个旧配置弱基模型，只保留调参成员"),
    ("13", "11-17 08:21", "version13", "v5_best", "0.0983", "0.08752",
     "伪标签阈值 0.03→0.25（3,500→6,260 行）+ 训练预算 40 epoch，当前最佳"),
    ("14", "11-17 09:14", "version14", "v5_best_allrows", "0.0983", "0.08752",
     "诊断：改动 3,659 行机器图后分数不变，证明机器图不计分"),
    ("15", "11-17 09:17", "version15", "v5_best_c001", "0.0983", "—",
     "诊断：截断下限 0.01，CV 已证明有害，未提交"),
    ("16", "11-17 09:17", "version16", "v5_best_c005", "0.0983", "—",
     "诊断：截断下限 0.05，同上未提交"),
]

HISTORY = [
    ("参照", "上一轮 baseline", "单 CNN + 转导式角度后处理（结果偏乐观）", "0.1943 / 0.1314"),
    ("M1", "version1（fast_s0）", "4 残差 SE 块 + stride-2 stem（954k），本地归一化，8 视角 TTA，50 epoch", "0.25381"),
    ("M2", "version3（v1_s0）", "93k 紧凑 CNN（5 段单卷积、3 次池化、均值+最大值池化、元数据支路），35 epoch", "0.24804"),
    ("M3", "version4（v1_fast）", "大模型与小模型的折外预测在 logit 空间平均，再做 Platt 校准", "0.23955"),
    ("M4", "version2（final_fast_s0）", "角度层 v1：按角度组用训练标签估计 P(冰山)，α=0.1 收缩，log-odds 融合", "0.12109"),
    ("M5", "version5（final_v1_fast）", "角度层 v2：精确组优先、否则 σ=0.02° 高斯核，三参数留一网格搜索", "0.11897"),
    ("M6", "（未计入提交）", "用 0.119 的融合预测做软标签再训 CNN——本项目的伪标签线由此开始", "—"),
]

NOTES = [
    "M1 教训：验证损失在 epoch 16–31 触底后回升，50 epoch 明显过拟合；TTA 单独带来约 0.016 收益。",
    "M2 教训：在 1,604 张训练样本上容量不是瓶颈，小模型更准且便宜，省下的算力应投入集成。",
    "M3 教训：结构差异比同构多 seed 更值钱；校准本身又贡献约 0.002。",
    "M4 教训：精确匹配只覆盖 57% 的真实测试行，覆盖率是这一步的唯一瓶颈。",
    "M5 教训：核先验单独只有 0.206，必须与图像模型互补；覆盖率扩展贡献 −0.118，核平滑本身只再贡献 −0.002。",
    "两个不入提交版本的诊断：全分辨率 + 全局归一化 + 关掉增强时 TTA 反而变差（0.2429），"
    "说明 TTA 必须与训练增强配套；早停在余弦调度下有害（0.2877），因为学习率仍在高位就被截断。",
]


def fixed_layout(table, widths) -> None:
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


def fill(table, data, size=8.0, bold_first=False) -> None:
    for row_index, values in enumerate(data):
        cells = table.rows[row_index].cells
        for column, text in enumerate(values):
            cells[column].text = ""
            run = cells[column].paragraphs[0].add_run(str(text))
            run.font.size = Pt(size)
            if bold_first and row_index == 0:
                run.bold = True


def main() -> int:
    document = Document(PATH)

    anchor = next(p for p in document.paragraphs if p.text.strip().startswith("3. 关键诊断"))
    old = document.tables[1]
    old._tbl.getparent().remove(old._tbl)

    table = document.add_table(rows=len(ROWS) + 1, cols=len(HEADER))
    table.style = "Light Grid Accent 1"
    fill(table, [HEADER] + list(ROWS), size=7.8, bold_first=True)
    fixed_layout(table, [Inches(0.3), Inches(0.85), Inches(0.72), Inches(1.05),
                         Inches(0.62), Inches(0.55), Inches(2.41)])
    anchor._p.addprevious(table._tbl)

    spacer = anchor.insert_paragraph_before("")
    spacer.paragraph_format.space_after = Pt(2)

    anchor.insert_paragraph_before("2.2 早期迭代细节（M1–M6，并入上一阶段文档）", style="Heading 2")
    intro = anchor.insert_paragraph_before(
        "version1–version5 来自上一阶段的独立文档，按里程碑编号 M1–M6。它们的结论决定了后续所有训练配置，"
        "因此完整并入下表：")
    intro.paragraph_format.space_after = Pt(4)

    history = document.add_table(rows=len(HISTORY) + 1, cols=4)
    history.style = "Light Grid Accent 1"
    fill(history, [["里程碑", "对应版本", "核心改动", "五折 OOF"]] + list(HISTORY),
         size=7.8, bold_first=True)
    fixed_layout(history, [Inches(0.62), Inches(1.42), Inches(3.6), Inches(0.86)])
    anchor._p.addprevious(history._tbl)

    for note in NOTES:
        paragraph = anchor.insert_paragraph_before(note, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(2)

    document.save(PATH)
    print("version table rebuilt with 16 rows; early milestones merged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
