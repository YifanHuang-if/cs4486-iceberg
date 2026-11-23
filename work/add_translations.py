"""Insert a Chinese translation under each 原文 line, leaving everything else intact."""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

PATH = Path(r"D:\Desktop\CS4486_PPT_逐页通俗解释.docx")
ORIGINALS = Path(r"C:\Users\16018\cs4486-iceberg\work\originals.json")

ACCENT = RGBColor(0xC0, 0x00, 0x00)

TRANSLATIONS = [
    "冰山分类：使用雷达图像与角度",
    "从一个小型 CNN 基线，到角度先验、模型融合与学生模型",
    "黄亦凡 72400438（组长）／郑鸿宇 72400229／吴坤泽 72400463",
    "Statoil／C-CORE 冰山分类挑战赛",
    "2026 年 11 月 24 日",
    "用更低的 Log Loss 预测冰山概率",
    "第一阶段：分析数据、固定五折、训练 CNN 基线；再分别试翻转、dropout、角度输入与缺失角度标记",
    "第二阶段：加入角度先验、模型融合与软标签训练；调参、比分数、分析残余错误",
    "第一步：任务与指标",
    "第二步：数据与挑战——检查数据与缺失角度；设定五折并保持不变",
    "第三步：基线——训练一个小 CNN；每一行都由没见过它的模型打一次分；公平比较",
    "第四步：模型改进",
    "第五步：结果与反思",
    "比赛快照：Statoil／C-CORE 挑战赛；对卫星雷达图像分类；服务于更安全的航行",
    "预测设置：分类冰山或船；输入 HH、HV 与入射角；输出冰山概率；提交列为 id 与 is_iceberg",
    "上排是船、下排是冰山，各自展示 HH 与 HV 两个波段",
    "官方指标：Log Loss；越低越好；海上一次自信的错误代价高，因此指标惩罚过度自信",
    "项目目标：把入射角当作证据，超过一个可复现的 CNN 基线；对照历史前五，冲刺私榜低于 0.08883",
    "训练标签：851 条船、753 条冰山，占 53.1% 与 46.9%，类别基本均衡",
    "数据概况：训练 1,604 条、测试 8,424 条；每条两张 75×75 图像加角度；用各自训练折算归一化",
    "主要挑战：数据小且噪声大；训练集 133 条角度缺失；测试集角度完整",
    "关键结论：角度完全相同的行几乎总是同标签（98.6%）；仅邻近角度时一致率约 84%；"
    "因此标记缺失角度并把两类分开处理",
    "缺失角度的偏移：训练集 133 条缺失角度全部是船",
    "使用同一套五折：每折约 321 条验证样本",
    "额外检查：按角度分组——把同角度样本放进同一折",
    "基线流程：60,433 个参数，从零开始训练",
    "预处理：在每一折内部做缩放",
    "雷达输入：HH + HV，尺寸 75×75",
    "模型：3 个卷积块，通道 16／32／64",
    "损失：二元交叉熵",
    "训练设置：Adam、学习率 0.001、批大小 32；最多 30 轮，验证 7 轮不升就停；CPU；权重衰减 0.0001",
    "基线分数：折外 Log Loss 0.26063；公开参考值 0.24638，说明运行环境略有差异",
    "第一版消融实验：折外 Log Loss——V0 0.26063、V2 0.26703、V3 0.25407、V4 0.22683",
    "V0–V5 是第一版实验编号；之后的提交文件用另一套编号体系",
    "A 两级先验；假设：同角度通常同标签，但完全相同的角度匹配很少",
    "A 实现：先使用完全相同角度的组；没有就用高斯核；只在训练折上拟合",
    "A 效果：嵌套折外分数 0.25381 → 0.12109；折外提升 −0.13272",
    "B 分组融合；假设：精确匹配与近邻匹配需要不同程度的信任",
    "B 实现：对精确行与近邻行分别给权重，并在每折训练部分拟合",
    "B 效果：嵌套折外分数 0.11556 → 0.10721；折外提升 −0.00835",
    "C 学生模型多样性；假设：单个 CNN 会重复自己的错误，无标签数据被浪费",
    "C 实现：蒸馏软标签，再加入 shape MLP 与 mixup 学生，制造不同的错误模式",
    "C 效果：蒸馏后图像折外分数 0.1959 → 0.1833；折外提升 −0.0126",
    "角度证据带来的分数变化最大（−0.13）；后续步骤小得多；各步增益不能相加",
    "D 训练配方；假设：调好训练设置比加大模型容量更有效",
    "D 实现：AdamW、批大小 32、学习率 0.003、权重衰减 0.01、余弦调度、约 9.3 万参数的小 CNN",
    "D 效果：调参后集成折外分数 0.10194 → 0.09976；私榜变化 −0.0016",
    "E 更宽的软标签；假设：过严的置信门槛丢掉了大部分无标签行",
    "E 实现：阈值 0.03 → 0.25；约 6,260 条伪标签行；权重 0.25；训练 20 轮",
    "E 效果：阈值调整后折外分数 0.09937 → 0.09829；私榜变化 −0.0003",
    "F 角度标定；假设：交叉验证与排行榜偏好不同的角度权重",
    "F 实现：拟合后把两个角度权重同乘 0.90；模型不变，依据来自排行榜反馈",
    "F 效果：私榜 0.08708，交叉验证略变差；私榜变化 −0.0004",
    "调参带来最大的一步（折外 −0.0021）；最后两步只增加约 0.0005",
    "最佳私榜分数，来自 version19.csv",
    "私榜 0.08708",
    "目标核对：达到前五水平（第四）；冲刺目标 < 0.08883 已达成",
    "核心结论：私榜损失比首个提交低 62%；收益来自角度证据，而不是更大的模型",
    "部分提交的私榜 Log Loss（折线图）",
    "错误仍在哪里：仅近邻角度匹配的行承担 87.2% 的损失；图像与角度判断相反的 231 行占 50.7%",
    "迟交没有官方排名；公榜与私榜使用测试集的不同子集",
    "有效的做法：角度信息提供上下文（0.25381→0.12109）；分组权重有帮助（0.11556→0.10721）；"
    "不同学生模型让融合更强（0.1959→0.1833）",
    "无效的做法：更强增强与更大集成往往更差；可能因雷达细节被破坏、成员错误相似；"
    "下次应一次只改一个因素并让增强匹配训练",
    "两个失败的实验：几何增强第 0 折 Log Loss 0.2951；26 成员贪心集成折外 0.10561，"
    "对手工 6 成员 0.10194",
    "局限与下一步：没有精确角度匹配时图像模型偏弱；所有分数都部分受调参选择影响",
    "只有 1,604 条带标签样本；分数部分反映调参；下一步改进雷达图像特征",
    "ChatGPT／DeepSeek 协助调试代码与起草文字；研究设计、验证方案与建模选择由小组制定并核验",
    "折种子 42（固定划分文件）；模型种子 71、83、97、909、1201、1401、2601，均存入各权重文件",
    "Python 3.12、PyTorch 2.15（CPU）、NumPy 2.5；16 线程，无 GPU",
    "AdamW、20 轮、批大小 32、学习率 3e-3 或 5e-3、dropout 0.4–0.5、权重衰减 1e-2、8 视角 TTA",
    "代码仓库地址",
    "指标用折外预测算 Log Loss；第二层只在各折训练部分拟合；提交前核对 ID",
    "未使用预训练权重；全部 CNN 从零训练：基线 60,433 参数，最终 93,521 参数",
    "最终方案中没有预训练权重",
    "黄：数据分析；吴：超参数调整建议；郑：误差分析与预测",
    "提交证据：左为私榜分数，右为公榜分数",
    "核心思路：把雷达预测与角度预测结合起来；最佳私榜 0.08708（原文 rar 为 radar 笔误）",
]


def main() -> int:
    originals = json.loads(ORIGINALS.read_text(encoding="utf-8"))
    if len(originals) != len(TRANSLATIONS):
        raise SystemExit(f"count mismatch: {len(originals)} originals vs {len(TRANSLATIONS)} translations")

    document = Document(PATH)
    paragraphs = list(document.paragraphs)

    inserted = 0
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        text = paragraph.text.strip()
        if text.startswith("原文") and index + 1 < len(paragraphs):
            if paragraphs[index + 1].text.strip().startswith("解释"):
                target = paragraphs[index + 1]
                new = target.insert_paragraph_before()
                new.paragraph_format.left_indent = Pt(12)
                new.paragraph_format.space_after = Pt(1)
                tag = new.add_run("翻译  ")
                tag.bold = True
                tag.font.size = Pt(9)
                tag.font.color.rgb = ACCENT
                body = new.add_run(TRANSLATIONS[inserted])
                body.font.size = Pt(9.5)
                inserted += 1
                index += 1
        index += 1

    print(f"translations inserted: {inserted}")
    document.save(PATH)
    print(f"saved: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
