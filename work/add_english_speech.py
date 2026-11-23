"""Add an English version under every Chinese paragraph and estimate speaking time."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

PATH = Path(r"D:\Desktop\CS4486_6分钟演讲稿.docx")

GREY = RGBColor(0x50, 0x50, 0x50)
CN_CHARS_PER_MIN = 220
EN_WORDS_PER_MIN = 140

ENGLISH = {
    "大家下午好，我们组做的是 Statoil / C-CORE 冰山分类。":
        "Good afternoon everyone. Our team's project is the Statoil / C-CORE iceberg classification challenge.",
    "整个项目分两段。": "The project has two stages.",
    "第一段：看懂数据、固定五折、训练一个能复现的小 CNN。":
        "Stage one: understand the data, fix five folds, and train a small CNN that anyone can reproduce.",
    "第二段：把角度变成证据、做模型融合、用软标签再训，最后调参与误差分析。":
        "Stage two: turn the angle into evidence, blend the models, retrain on soft labels, then tune and analyse the remaining errors.",
    "输入是两个波段的雷达图，加上一个入射角；输出是这张图属于冰山的概率；指标是 Log Loss。":
        "The input is two radar bands plus an incidence angle. The output is the probability that the object is an iceberg. The metric is Log Loss.",
    "Log Loss 的特点是：答错又自信，罚得最重。所以模型不但要准，还要知道自己有多确定。":
        "Log Loss punishes confident mistakes the hardest. So the model must be accurate and also know how certain it is.",
    "有三个数字决定了后面的所有决定。": "Three numbers decided every choice we made afterwards.",
    "第一，训练集只有 1,604 条，测试集 8,424 条，两边类别比较均衡。":
        "First, the training set has only 1,604 rows and the test set 8,424, with fairly balanced classes.",
    "第二，训练集里有 133 条没有角度，全部是船；而测试集的角度是完整的。这意味着任何依赖「缺失」的收益都带不走。":
        "Second, 133 training rows have no angle, and all of them are ships - while the test set has no missing angles. "
        "So any gain that relies on missingness cannot be carried over.",
    "第三，也是最重要的：角度完全相同的一组图，标签一致率 98.6%；而只有角度接近时，一致率只有 84%。":
        "Third, and most important: rows with exactly the same angle share the same label 98.6% of the time; "
        "when the angle is only nearby, agreement drops to 84%.",
    "所以我们知道：角度可以用，但必须分成「精确匹配」和「近邻」两种情况来用。":
        "So the angle is usable, but exact matches and near matches must be treated separately.",
    "基线是一个 6 万参数的小 CNN，从零训练，五折固定。折外分数 0.26063。公开参考是 0.24638，差异来自运行环境，所以后面所有改进都跟 0.26063 比。":
        "The baseline is a small 60k-parameter CNN trained from scratch on fixed five folds. Its out-of-fold score is 0.26063. "
        "The published reference is 0.24638; the gap comes from the environment, so every later improvement is measured against 0.26063.",
    "我们还做了六组只改一个因素的对照：加翻转、加 dropout、加角度输入、加缺失标记。":
        "We also ran six controlled comparisons that change one factor at a time: flips, dropout, an angle input, and a missing-angle flag.",
    "分数最低的是缺失标记 0.22683，但它的收益几乎全部来自那 133 条缺失样本，测试集没有这类行。所以这个「最好」是不能交的。":
        "The lowest score was the missing flag at 0.22683, but almost all of that gain came from those 133 missing rows, "
        "which the test set does not contain. So that best score could not be submitted.",
    "接下来是我们认为最有价值的三步。": "Next are the three steps we consider most valuable.",
    "A，两级先验。有完全相同的角度，就用这一组的标签估一个概率；没有，就用高斯核按角度距离借附近样本的信息。所有统计量只使用当前训练折算。":
        "A, a two-level prior. When an exact angle match exists, we estimate a probability from that group's labels; "
        "when it does not, a Gaussian kernel borrows information from nearby angles. Every statistic is fitted on the current training fold only.",
    "B，分组融合。精确匹配和近邻匹配的可信度不同，所以第二层给它们两个不同的系数，而且每折只在该折的训练部分拟合。":
        "B, grouped fusion. Exact and near matches deserve different trust, so the second layer gives them two separate coefficients, "
        "and it is fitted only inside each training fold.",
    "C，学生模型。用我们已经训好的模型给测试图打软标签，再训一个学生；另外加一个只看几何统计的小网络和一个 mixup 学生，让它们的错误模式不一样。":
        "C, student models. We use our own trained models to give the test images soft labels and train a student; "
        "we also add a small network that only sees geometric statistics and a mixup student, so their error patterns differ.",
    "效果：A 把折外从 0.25381 拉到 0.12109，一步贡献 0.13；B 再降 0.008；C 让图像模型从 0.1959 改善到 0.1833。":
        "The effect: A moved the out-of-fold score from 0.25381 to 0.12109, a single step worth 0.13; B took off another 0.008; "
        "C improved the image model from 0.1959 to 0.1833.",
    "D，调参。把批大小、学习率、权重衰减重新扫一遍，折外从 0.10194 降到 0.09976。":
        "D, tuning. We re-scanned batch size, learning rate and weight decay, taking the out-of-fold score from 0.10194 to 0.09976.",
    "E，放宽伪标签。门槛从 0.03 放到 0.25，可用的测试图从约 3,500 张变成 6,260 张，再降 0.0011。":
        "E, broader soft labels. The threshold went from 0.03 to 0.25, so usable test images grew from about 3,500 to 6,260, worth another 0.0011.",
    "F，角度标定。模型完全不动，只把两个角度系数乘 0.90。私榜从 0.08752 降到 0.08708，但交叉验证反而略微变差。我们如实说明：这是标定，不是模型变强。":
        "F, angle calibration. The models are untouched; we simply multiply the two angle coefficients by 0.90. "
        "The private score went from 0.08752 to 0.08708, while cross-validation became slightly worse. "
        "We say it plainly: this is calibration, not a stronger model.",
    "最终私榜 0.08708，比第一次提交的 0.23022 低约 62%。对照材料里历史第五的 0.08883，我们低 0.00175。":
        "The final private score is 0.08708, about 62% lower than our first submission of 0.23022. "
        "Against the historical fifth place of 0.08883 in the materials, we are 0.00175 lower.",
    "必须说明：这是截止之后的迟交，没有正式排名；公榜和私榜用的是测试集的不同子集。":
        "One caveat: these are late submissions with no official ranking, and the public and private scores come from different subsets of the test set.",
    "误差也很集中：只有近邻角度匹配的行，承担了 87.2% 的损失。":
        "The errors are also concentrated: rows that only have a near angle match carry 87.2% of the loss.",
    "有效的三件事：角度证据、分组权重、学生多样性。":
        "Three things worked: angle evidence, group-specific weights, and student diversity.",
    "无效的两件事：更强的图像增强、更大的集成，两者都更差。":
        "Two things did not: stronger image augmentation and larger ensembles - both were worse.",
    "局限有两个：没有精确角度匹配时图像模型偏弱；以及部分分数是调参数调出来的。":
        "There are two limits: without an exact angle match the image model is weak, and part of the score comes from tuning.",
    "所有模型都从零训练，没有使用任何预训练权重。":
        "Every model was trained from scratch, with no pretrained weights.",
    "折种子 42，模型种子全部写进对应权重文件；环境是 Python 3.12 加 PyTorch CPU 版；代码在 GitHub。":
        "The fold seed is 42, every model seed is saved in its checkpoint; the environment is Python 3.12 with CPU PyTorch; the code is on GitHub.",
    "AI 只用于调试代码和起草文字，研究设计与验证方案由我们小组自己定、自己核。":
        "AI was used only for debugging and drafting; the study design and validation protocol were set and checked by our own team.",
    "最后一页是我们的提交截图，左边私榜、右边公榜。":
        "The last page is our submission screenshot: private score on the left, public on the right.",
    "以上是我们组的工作，谢谢各位，欢迎提问。":
        "That is our project. Thank you - we are happy to take questions.",
}


def spoken_units(text: str) -> tuple[int, int, int]:
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    numbers = len(re.findall(r"\d[\d,\.]*", text))
    latin_words = len(re.findall(r"[A-Za-z][A-Za-z\-]+", text))
    return chinese, numbers, latin_words


def main() -> int:
    document = Document(PATH)
    paragraphs = list(document.paragraphs)

    inserted = 0
    for index, paragraph in enumerate(paragraphs):
        text = paragraph.text.strip()
        if text not in ENGLISH:
            continue
        following = paragraphs[index + 1] if index + 1 < len(paragraphs) else None
        english = ENGLISH[text]
        if following is not None:
            new = following.insert_paragraph_before()
        else:
            new = document.add_paragraph()
        new.paragraph_format.left_indent = Pt(6)
        new.paragraph_format.space_after = Pt(10)
        new.paragraph_format.line_spacing = 1.25
        tag = new.add_run("EN  ")
        tag.bold = True
        tag.font.size = Pt(8.5)
        tag.font.color.rgb = GREY
        body = new.add_run(english)
        body.italic = True
        body.font.size = Pt(10.5)
        body.font.color.rgb = GREY
        inserted += 1

    # Time estimate over the script body only.
    chinese_chars = numbers = latin_words = 0
    english_words = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("EN  "):
            english_words += len(text[4:].split())
        elif text in ENGLISH:
            c, n, w = spoken_units(text)
            chinese_chars += c
            numbers += n
            latin_words += w

    chinese_seconds = (chinese_chars + 3.0 * numbers + latin_words) / CN_CHARS_PER_MIN * 60
    english_seconds = english_words / EN_WORDS_PER_MIN * 60

    def fmt(seconds: float) -> str:
        return f"{int(seconds // 60)} 分 {int(round(seconds % 60))} 秒"

    print(f"inserted EN paragraphs: {inserted}")
    print(f"Chinese: {chinese_chars} chars, {numbers} numbers, {latin_words} latin words -> {fmt(chinese_seconds)}")
    print(f"English: {english_words} words -> {fmt(english_seconds)}")

    summary = document.paragraphs[1]
    extra = summary.insert_paragraph_before(
        f"口播时长测算：中文版约 {fmt(chinese_seconds)}（{chinese_chars} 个汉字 + {numbers} 处数字，按 220 字/分钟）；"
        f"英文版约 {fmt(english_seconds)}（{english_words} 词，按 140 词/分钟）。两版选其一讲，不要都念。")
    for run in extra.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = GREY

    document.save(PATH)
    print(f"saved: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
