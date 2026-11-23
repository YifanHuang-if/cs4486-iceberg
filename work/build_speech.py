"""Build the six-minute speech script for Group12_Project3.pptx."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

OUT = Path(r"D:\Desktop\CS4486_6分钟演讲稿.docx")

ACCENT = RGBColor(0xC0, 0x00, 0x00)
GREY = RGBColor(0x40, 0x40, 0x40)
DARK = RGBColor(0x1F, 0x38, 0x64)

SCRIPT = [
    ("第 1 页 · 开场（约 15 秒）", [
        "各位老师好，我们组做的是 Statoil / C-CORE 冰山分类。",
        "先用一句话概括：我们没有把模型做大，而是发现了数据里的一个结构，并把它变成了分数。",
    ]),
    ("第 2 页 · 路线图（约 20 秒）", [
        "整个项目分两段。",
        "第一段：看懂数据、固定五折、训练一个能复现的小 CNN。",
        "第二段：把角度变成证据、做模型融合、用软标签再训，最后调参与误差分析。",
    ]),
    ("第 3 页 · 任务与指标（约 30 秒）", [
        "输入是两个波段的雷达图，加上一个入射角；输出是这张图属于冰山的概率；指标是 Log Loss。",
        "Log Loss 的特点是：答错又自信，罚得最重。所以模型不但要准，还要知道自己有多确定。",
    ]),
    ("第 4 页 · 数据与挑战（约 50 秒）", [
        "有三个数字决定了后面的所有决定。",
        "第一，训练集只有 1,604 条，测试集 8,424 条，两边类别比较均衡。",
        "第二，训练集里有 133 条没有角度，全部是船；而测试集的角度是完整的。"
        "这意味着任何依赖「缺失」的收益都带不走。",
        "第三，也是最重要的：角度完全相同的一组图，标签一致率 98.6%；而只有角度接近时，一致率只有 84%。",
        "所以我们知道：角度可以用，但必须分成「精确匹配」和「近邻」两种情况来用。",
    ]),
    ("第 5 页 · 基线（约 35 秒）", [
        "基线是一个 6 万参数的小 CNN，从零训练，五折固定。折外分数 0.26063。"
        "公开参考是 0.24638，差异来自运行环境，所以后面所有改进都跟 0.26063 比。",
        "我们还做了六组只改一个因素的对照：加翻转、加 dropout、加角度输入、加缺失标记。",
        "分数最低的是缺失标记 0.22683，但它的收益几乎全部来自那 133 条缺失样本，"
        "测试集没有这类行。所以这个「最好」是不能交的。",
    ]),
    ("第 6 页 · 三个改进 A / B / C（约 70 秒）", [
        "接下来是我们认为最有价值的三步。",
        "A，两级先验。有完全相同的角度，就用这一组的标签估一个概率；没有，就用高斯核按角度距离借附近样本的信息。"
        "所有统计量只使用当前训练折算。",
        "B，分组融合。精确匹配和近邻匹配的可信度不同，所以第二层给它们两个不同的系数，"
        "而且每折只在该折的训练部分拟合。",
        "C，学生模型。用我们已经训好的模型给测试图打软标签，再训一个学生；"
        "另外加一个只看几何统计的小网络和一个 mixup 学生，让它们的错误模式不一样。",
        "效果：A 把折外从 0.25381 拉到 0.12109，一步贡献 0.13；B 再降 0.008；"
        "C 让图像模型从 0.1959 改善到 0.1833。",
    ]),
    ("第 7 页 · 三步精修 D / E / F（约 55 秒）", [
        "D，调参。把批大小、学习率、权重衰减重新扫一遍，折外从 0.10194 降到 0.09976。",
        "E，放宽伪标签。门槛从 0.03 放到 0.25，可用的测试图从约 3,500 张变成 6,260 张，再降 0.0011。",
        "F，角度标定。模型完全不动，只把两个角度系数乘 0.90。私榜从 0.08752 降到 0.08708，"
        "但交叉验证反而略微变差。我们如实说明：这是标定，不是模型变强。",
    ]),
    ("第 8 页 · 最终结果（约 40 秒）", [
        "最终私榜 0.08708，比第一次提交的 0.23022 低约 62%。"
        "对照材料里历史第五的 0.08883，我们低 0.00175。",
        "必须说明：这是截止之后的迟交，没有正式排名；公榜和私榜用的是测试集的不同子集。",
        "误差也很集中：只有近邻角度匹配的行，承担了 87.2% 的损失。",
    ]),
    ("第 9 页 · 讨论与反思（约 35 秒）", [
        "有效的三件事：角度证据、分组权重、学生多样性。",
        "无效的两件事：更强的图像增强、更大的集成，两者都更差。",
        "局限有两个：没有精确角度匹配时图像模型偏弱；以及部分分数是调参数调出来的。",
    ]),
    ("第 10 页 · AI 使用与复现（约 30 秒）", [
        "所有模型都从零训练，没有使用任何预训练权重。",
        "折种子 42，模型种子全部写进对应权重文件；环境是 Python 3.12 加 PyTorch CPU 版；代码在 GitHub。",
        "AI 只用于调试代码和起草文字，研究设计与验证方案由我们小组自己定、自己核。",
    ]),
    ("第 11 页 · 收尾（约 10 秒）", [
        "最后一页是我们的提交截图，左边私榜、右边公榜。",
        "以上是我们组的工作，谢谢各位，欢迎提问。",
    ]),
]

TIPS = [
    ("重点强调的位置", "第 4 页的 98.6% 与 84%、第 6 页的 0.25381 → 0.12109、第 8 页的 0.08708。"
                        "这三个数字是全场最该被记住的，说到时可以放慢一点。"),
    ("创新点怎么说", "不要只说「我们用了先验」，要说「我们从 98.6% 这个数据事实出发，"
                      "设计了精确与近邻分开的两级证据，并且每一层都只在训练折内拟合」。"),
    ("主动交代边界", "第 7、8 页要主动说明「迟交没有排名」「标定不等于改进」，"
                      "这比被评委问出来更可信。"),
    ("超时怎么砍", "如果只剩 5 分钟：第 2 页压成一句、第 10 页只留「从零训练、种子与代码已留」、"
                   "第 9 页只留两条局限。"),
]


def main() -> int:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(12)

    title = document.add_heading("CS4486 冰山分类 · 6 分钟演讲稿", level=0)

    meta = document.add_paragraph()
    run = meta.add_run("对应 Group12_Project3.pptx 全部 11 页 · 按每分钟 220 字左右口播 · 正文约 1,450 字")
    run.font.color.rgb = GREY
    run.font.size = Pt(10)

    for heading_text, paragraphs in SCRIPT:
        heading = document.add_heading(heading_text, level=1)
        heading.runs[0].font.color.rgb = DARK
        for text in paragraphs:
            paragraph = document.add_paragraph(text)
            paragraph.paragraph_format.space_after = Pt(6)
            paragraph.paragraph_format.line_spacing = 1.4

    document.add_page_break()
    tip_heading = document.add_heading("口播提示", level=1)
    tip_heading.runs[0].font.color.rgb = ACCENT
    for name, text in TIPS:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.line_spacing = 1.35
        tag = paragraph.add_run(f"{name}：")
        tag.bold = True
        tag.font.color.rgb = ACCENT
        body = paragraph.add_run(text)
        body.font.size = Pt(11)

    document.save(OUT)
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
