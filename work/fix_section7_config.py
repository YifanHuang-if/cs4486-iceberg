"""Correct section 7 so the documented configuration matches version19 exactly."""
from __future__ import annotations

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

BULLETS = [
    "七个成员：五个 CNN（hp_star / hp_b / hp_d / pt25a / pt25b，均为 93,521 参数的 RadarV1）、"
    "一个 90 维几何统计 MLP（v5_shape，3 个种子）、一个 mixup 学生（stu5）",
    "CNN 公共设置：AdamW + 余弦退火（无 warmup、无 EMA）、BCEWithLogits 配软标签、"
    "伪标签权重 0.25、逐图局部 z-score、二面体翻转 + 高斯噪声 σ=0.02、batch 32、"
    "weight decay 1e-2、20 epoch、推理时 8 视角 TTA",
    "成员的刻意差异（用于去相关）：hp_star 学习率 3e-3/dropout 0.5/教师 v5_teacher；"
    "hp_b 学习率 5e-3；hp_d 与 pt25a/pt25b 为 dropout 0.4 且教师为 v5_teacher6，"
    "其中 pt25a/pt25b 把伪标签阈值从 0.03 放宽到 0.25（约 3,500→6,260 行）；"
    "stu5 保留旧配置（lr 1.5e-3 / batch 64）并加 mixup α=0.2",
    "种子：形状 MLP 用 71/83/97，CNN 用 909（hp_star、stu5）、1201（hp_b）、"
    "1401（hp_d 与 pt25a）、2601（pt25b）；折划分固定为 seed 42",
    "第二层：特征为 [logit(图像均值), logit(先验)·精确匹配, logit(先验)·无匹配]，"
    "L2=1.0，在每个外层折的训练部分内拟合；两级角度先验 σ=0.012、α=0.1；"
    "version19 另把两个先验系数缩放 0.90",
]

COMMANDS = (
    "python -m src.v3.prepare --raw data/raw\n"
    "python work/build_teacher.py v5_teacher6 s101 s202 s404 v5_shape stu1 stu5\n"
    "python -m src.v3.train --tag hp_star --seed 909  --epochs 20 --dropout 0.5 --lr 0.003 "
    "--weight-decay 0.01 --batch 32 --teacher v5_teacher\n"
    "python -m src.v3.train --tag hp_b    --seed 1201 --epochs 20 --dropout 0.5 --lr 0.005 "
    "--weight-decay 0.01 --batch 32 --teacher v5_teacher\n"
    "python -m src.v3.train --tag hp_d    --seed 1401 --epochs 20 --dropout 0.4 --lr 0.003 "
    "--weight-decay 0.01 --batch 32 --teacher v5_teacher6\n"
    "python -m src.v3.train --tag pt25a   --seed 1401 --epochs 20 --dropout 0.4 --lr 0.003 "
    "--weight-decay 0.01 --batch 32 --pseudo-threshold 0.25 --teacher v5_teacher6\n"
    "python -m src.v3.train --tag pt25b   --seed 2601 --epochs 20 --dropout 0.4 --lr 0.003 "
    "--weight-decay 0.01 --batch 32 --pseudo-threshold 0.25 --teacher v5_teacher6\n"
    "python -m src.v3.shape_model --tag v5_shape\n"
    "python work/v5_predict.py --tags v5_shape hp_star hp_b hp_d pt25a pt25b stu5\n"
    "python work/v5_submit.py  --tags v5_shape hp_star hp_b hp_d pt25a pt25b stu5 "
    "--angle-scale 0.90 --name version19\n"
    "python work/validate_submission.py version19\n"
    "python work/verify_submission_provenance.py version19"
)


def main() -> int:
    document = Document(PATH)
    bullets = [p for p in document.paragraphs
               if p.style.name == "List Bullet" and p.text.strip()
               and any(k in p.text for k in ("小时", "成员", "CNN 成员", "shape/statistics",
                                             "伪标签蒸馏", "融合：图像", "个调参"))]
    target = bullets[:5] if len(bullets) >= 5 else bullets
    for paragraph, text in zip(target, BULLETS):
        if paragraph.runs:
            paragraph.runs[0].text = text
            for run in paragraph.runs[1:]:
                run.text = ""
        else:
            paragraph.add_run(text)
    if len(target) < len(BULLETS):
        last = target[-1]
        for text in BULLETS[len(target):]:
            new = last.insert_paragraph_before(text, style="List Bullet")
            new.paragraph_format.space_after = Pt(2)
    print(f"rewrote {len(target)} config bullets")

    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("python -m src.v3.prepare"):
            for run in list(paragraph.runs)[1:]:
                run._element.getparent().remove(run._element)
            paragraph.runs[0].text = COMMANDS
            paragraph.runs[0].font.size = Pt(8)
            print("reproduce block refreshed")
            break

    document.save(PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
