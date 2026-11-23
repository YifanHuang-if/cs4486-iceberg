"""Make section 7's bullet list exactly the five intended lines."""
from __future__ import annotations

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

EXPECTED = [
    ("七个成员：五个 CNN（hp_star / hp_b / hp_d / pt25a / pt25b，均为 93,521 参数的 RadarV1）、"
     "一个 90 维几何统计 MLP（v5_shape，3 个种子）、一个 mixup 学生（stu5）"),
    ("CNN 公共设置：AdamW + 余弦退火（无 warmup、无 EMA）、BCEWithLogits 配软标签、"
     "伪标签权重 0.25、逐图局部 z-score、二面体翻转 + 高斯噪声 σ=0.02、batch 32、"
     "weight decay 1e-2、20 epoch、推理时 8 视角 TTA"),
    ("成员的刻意差异（用于去相关）：hp_star 学习率 3e-3 / dropout 0.5 / 教师 v5_teacher；"
     "hp_b 学习率 5e-3；hp_d 与 pt25a / pt25b 为 dropout 0.4 且教师为 v5_teacher6，"
     "其中 pt25a / pt25b 把伪标签阈值从 0.03 放宽到 0.25（约 3,500→6,260 行）；"
     "stu5 保留旧配置（lr 1.5e-3 / batch 64）并加 mixup α=0.2"),
    ("种子：形状 MLP 用 71 / 83 / 97，CNN 用 909（hp_star、stu5）、1201（hp_b）、"
     "1401（hp_d 与 pt25a）、2601（pt25b）；折划分固定为 seed 42"),
    ("第二层：特征为 [logit(图像均值), logit(先验)·精确匹配, logit(先验)·无匹配]，L2=1.0，"
     "在每个外层折的训练部分内拟合；两级角度先验 σ=0.012、α=0.1；"
     "version19 另把两个先验系数缩放 0.90"),
]


def main() -> int:
    document = Document(PATH)
    paragraphs = list(document.paragraphs)
    start = next(i for i, p in enumerate(paragraphs) if p.text.strip().startswith("7. 最终配置"))
    stop = next(i for i, p in enumerate(paragraphs)
                if p.text.strip().startswith("复现命令") and i > start)
    block = [p for p in paragraphs[start:stop] if p.style.name == "List Bullet"]
    print(f"found {len(block)} bullets in section 7")

    for paragraph, text in zip(block, EXPECTED):
        if paragraph.runs:
            paragraph.runs[0].text = text
            for run in paragraph.runs[1:]:
                run.text = ""
        else:
            paragraph.add_run(text)

    while len(block) > len(EXPECTED):
        extra = block.pop()
        extra._p.getparent().remove(extra._p)
        print("removed surplus bullet")
    while len(block) < len(EXPECTED):
        anchor = block[-1]
        new = anchor.insert_paragraph_before(EXPECTED[len(block)], style="List Bullet")
        new.paragraph_format.space_after = Pt(2)
        block.append(new)

    document.save(PATH)
    check = Document(PATH)
    hit = False
    for paragraph in check.paragraphs:
        if paragraph.text.strip().startswith("7. 最终配置"):
            hit = True
            continue
        if hit and paragraph.text.strip().startswith("8."):
            break
        if hit and paragraph.style.name == "List Bullet":
            print(" -", paragraph.text.strip()[:95])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
