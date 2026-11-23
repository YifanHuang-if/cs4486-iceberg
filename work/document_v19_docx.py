"""Document version19 in full and mark the later variants as public-board only."""
from __future__ import annotations

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

DERIVATION = [
    "系数层面：version13 拟合出的第二层系数为 [图像 0.860, 精确 1.133, 近邻 0.634, 截距 −0.161]。"
    "version19 只把后两个先验系数乘以 0.90，得到 [0.860, 1.020, 0.571, −0.161]，图像系数与截距不动。",
    "改动幅度：8,424 行中只有 2,356 行发生变化（其余行的角度特征为 0，乘任何系数结果相同），"
    "平均改动 0.0015、最大 0.0775、整体均值 0.37147→0.37132。改动小到不可能引入新的误差来源。",
    "代价与收益：CV 由 0.098287 变为 0.098753（差 0.00047），而 private 由 0.08752 降到 0.08708、"
    "public 由 0.10899 降到 0.10724——两个榜同时改善，CV 反而变差。",
    "不是单点运气：0.85 / 0.90 / 0.95 三档的 private 分别是 0.08718 / 0.08708 / 0.08721，"
    "均优于 1.00 的 0.08752。取中间值是区间内的稳定选择，不是从若干候选中挑最小值。",
    "性质界定：这是一次标定，不是模型改进。它没有改变任何权重、成员或训练过程，"
    "只是把同一份预测的角度证据重新加权。因此按乐观估计，private 的 0.00044 改善中"
    "含约 0.0004 的选择偏差，真实收益接近于零。",
]

PUBLIC_NOTE = (
    "version20–version30 是为公榜单独做的变体，做法是继续降低角度权重并加入同角度池化。"
    "public 最好做到 0.10365（version28：角度 0.80 + 满池化），但 private 同时升到 0.08986，"
    "因此它们不进入私榜主线。这组实验的结论是两个榜的最优点不同："
    "同一份模型无法同时在两个榜上最优，公榜更看重图像模型本身、私榜更看重角度结构。")


def main() -> int:
    document = Document(PATH)
    anchor = next(p for p in document.paragraphs if p.text.strip().startswith("3. 关键诊断"))

    anchor.insert_paragraph_before("2.3.4 version19 的完整推导", style="Heading 3")
    for text in DERIVATION:
        paragraph = anchor.insert_paragraph_before(text, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(2)

    anchor.insert_paragraph_before("2.3.5 公榜专用变体（version20–version30）", style="Heading 3")
    note = anchor.insert_paragraph_before(PUBLIC_NOTE)
    note.paragraph_format.space_after = Pt(4)

    document.save(PATH)
    print("added 2.3.4 and 2.3.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
