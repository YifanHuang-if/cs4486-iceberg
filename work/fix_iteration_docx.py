"""Restore from backup, then fix the four gaps correctly.

Section 7 already has exactly five bullets, so their text is rewritten in place
rather than cloned - the earlier attempt cloned first and updated the wrong
paragraphs.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from docx import Document
from docx.shared import Pt

TARGET = Path(r"D:\Desktop\CS4486_冰山模型迭代流程总结.docx")
BACKUP = Path(r"D:\Desktop\CS4486_冰山模型迭代流程总结.backup.docx")

NEW_SECTION7 = [
    "七个成员：五个 CNN（hp_star / hp_b / hp_d / pt25a / pt25b，均为 93,521 参数的 RadarV1）、"
    "一个 90 维几何统计 MLP（v5_shape，3 个种子）、一个 mixup 学生（stu5）",
    "CNN 公共设置：AdamW + 余弦退火（无 warmup、无 EMA）、BCEWithLogits 配软标签、伪标签权重 0.25、"
    "逐图局部 z-score、二面体翻转 + 高斯噪声 σ=0.02、batch 32、weight decay 1e-2、20 epoch、8 视角 TTA",
    "成员的刻意差异：hp_star 学习率 3e-3 / dropout 0.5 / 教师 v5_teacher；hp_b 学习率 5e-3；"
    "hp_d 与 pt25a / pt25b 为 dropout 0.4 且教师 v5_teacher6，其中 pt25a / pt25b 把伪标签阈值从 0.03 "
    "放宽到 0.25（约 3,500→6,260 行）；stu5 保留旧配置（lr 1.5e-3 / batch 64）并加 mixup α=0.2",
    "种子：形状 MLP 用 71 / 83 / 97，CNN 用 909（hp_star、stu5）、1201（hp_b）、"
    "1401（hp_d 与 pt25a）、2601（pt25b）；折划分固定为 seed 42",
    "第二层：特征为 [logit(图像均值), logit(先验)·精确匹配, logit(先验)·无匹配]，L2=1.0，"
    "在每个外层折的训练部分内拟合；两级角度先验 σ=0.012、α=0.1；version19 另把两个先验系数缩放 0.90",
]

NEW_COMMANDS = (
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

WARMUP_ROW = ("预热 3 epoch（warmup）",
              "folds 0-1 均值 0.15567 vs 无预热 0.15191，净变差 +0.0038；两折方向相反")

V31_NOTE = ("后续检验（已生成未提交）：把两个先验系数拆成独立旋钮，version31 只降近邻（0.90）、"
            "version32 只降精确匹配（0.85），用于判断 version19 的收益来自哪一侧。"
            "因精确匹配行的先验准确率为 98.6%，理论上不需要削弱。")


def set_text(paragraph, text) -> None:
    runs = paragraph.runs
    if not runs:
        runs = [paragraph.add_run()]
    runs[0].text = text
    for run in list(runs)[1:]:
        run._r.getparent().remove(run._r)


def main() -> int:
    shutil.copy2(BACKUP, TARGET)
    print("restored from backup")

    document = Document(TARGET)
    paragraphs = list(document.paragraphs)

    # 1. Section 7 bullets, rewritten in place (5 -> 5).
    start = next(i for i, p in enumerate(paragraphs) if p.text.strip().startswith("7. 最终配置"))
    end = next(i for i, p in enumerate(paragraphs)
               if p.text.strip().startswith("复现命令") and i > start)
    bullets = [p for p in paragraphs[start:end] if p.style.name == "List Bullet"]
    print(f"section 7 bullets: {len(bullets)}")
    if len(bullets) != len(NEW_SECTION7):
        raise SystemExit(f"expected {len(NEW_SECTION7)} bullets, found {len(bullets)}")
    for paragraph, text in zip(bullets, NEW_SECTION7):
        set_text(paragraph, text)
    print("section 7 rewritten")

    # 2. Reproduce commands.
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("python -m src.v3.prepare"):
            for run in list(paragraph.runs)[1:]:
                run._element.getparent().remove(run._element)
            set_text(paragraph, NEW_COMMANDS)
            paragraph.runs[0].font.size = Pt(8)
            print("commands refreshed")
            break

    # 3. Warmup row in the negative table.
    negatives = document.tables[7]
    if WARMUP_ROW[0] not in {row.cells[0].text.strip() for row in negatives.rows[1:]}:
        negatives._tbl.append(copy.deepcopy(negatives.rows[-1]._tr))
        row = negatives.rows[-1]
        for column, text in enumerate(WARMUP_ROW):
            cell = row.cells[column]
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(9)
    print(f"negative rows: {len(negatives.rows) - 1}")

    # 4. version31/32 note after 2.3.5.
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("version20–version30 是为公榜单独做的变体"):
            new = paragraph.insert_paragraph_before(V31_NOTE)
            new.paragraph_format.space_after = Pt(3)
            print("v31/32 note added")
            break

    # 5. Section 8 comparison number.
    for paragraph in document.paragraphs:
        if "我们相差 0.00061" in paragraph.text:
            for run in paragraph.runs:
                if "0.00061" in run.text:
                    run.text = run.text.replace("0.00061", "0.00175")
            print("section 8 number corrected")
            break

    document.save(TARGET)
    print(f"saved: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
