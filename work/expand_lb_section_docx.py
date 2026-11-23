"""Expand section 2.3 with the reasoning and the procedure behind leaderboard tuning."""
from __future__ import annotations

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

BEFORE = [
    ("二、为什么转向排行榜",
     "交叉验证与排行榜在角度权重上给出了相反结论。version13 的 CV 是 0.098287，"
     "而把角度权重降到 0.90 后 CV 变成 0.098753（变差），两个榜却同时改善。"
     "原因在两者的样本构成不同：CV 中精确角度匹配行占 47%，测试集占 57%，"
     "因此同一组融合系数在两个集合上的最优值并不一致。既然目标是排行榜，就以排行榜为准。"),
    ("三、可调旋钮的识别",
     "调参之前先确认哪些量能在不重训模型的前提下改变预测，并且改的是"
     "「同一批预测的用法」而不是模型本身。最终选出三个："
     "角度权重（第二层两个先验系数）、同角度池化强度、概率温度。"
     "其余旋钮（超参、增强、结构）都必须重训，无法用排行榜反馈快速迭代。"),
    ("四、做法：四步循环",
     "① 先用嵌套五折算出每一档的 CV 代价，确认改动幅度可接受；"
     "② 从最温和的一档起步提交，只获取方向而不冒险；"
     "③ 同时读 private 与 public 两个分数，判断收益来自哪一侧；"
     "④ 在改善区间内加密采样，直到相邻两档差异小于 0.0002 即停止。"
     "具体实现是在第二层 logistic 拟合完成后，把两个先验系数乘以一个缩放因子："
     "logit(p) = w₀·logit(图像) + s·w₁·logit(先验)·[精确匹配] + s·w₂·logit(先验)·[无匹配] + b。"
     "s=1.00 即 version13 的原始拟合结果，s<1 表示削弱角度证据、更多依赖图像模型。"),
]

AFTER = [
    "选择偏差必须写明：这一节的改善是用排行榜反馈本身调出来的，"
    "其中 private 从 0.08752 到 0.08708 的部分含有选择成分，不能等同于模型能力提升。"
    "稳健的证据是三点同时改善（0.85 / 0.90 / 0.95 均优于 1.00），而不是 0.90 这个单点。",
    "后续检验：把两个先验系数拆成独立旋钮（精确匹配单独一档、近邻单独一档），"
    "用以判断收益究竟来自哪一侧——精确匹配行的先验准确率是 98.6%，理论上不需要削弱。",
    "结论：排行榜调参能改善名次，但它只是把已有预测换一种用法，"
    "不能缩小与前列在图像模型能力上的差距。这条线的收益已在 0.0002 量级收敛。",
]


def main() -> int:
    document = Document(PATH)

    table = document.tables[3]
    if [c.text.strip() for c in table.rows[0].cells][:2] != ["角度权重", "池化"]:
        print("unexpected table at index 3")
        return 1

    for title, body in BEFORE:
        paragraph = document.add_paragraph()
        run = paragraph.add_run(title)
        run.bold = True
        run.font.size = Pt(10.5)
        table._tbl.addprevious(paragraph._p)
        paragraph2 = document.add_paragraph(body)
        for run in paragraph2.runs:
            run.font.size = Pt(10)
        table._tbl.addprevious(paragraph2._p)

    anchor = next(p for p in document.paragraphs if p.text.strip().startswith("3. 关键诊断"))
    for text in AFTER:
        paragraph = anchor.insert_paragraph_before(text, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(2)

    document.save(PATH)
    print("section 2.3 expanded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
