"""Build the slide-by-slide English/Chinese companion.

Each text block on a slide is quoted whole (line breaks kept), followed by a
Chinese translation and a note saying which data values it refers to.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

OUT = Path(r"D:\Desktop\CS4486_PPT_英中对照.docx")

ACCENT = RGBColor(0xC0, 0x00, 0x00)
GREY = RGBColor(0x40, 0x40, 0x40)
DARK = RGBColor(0x1F, 0x38, 0x64)

C = ("__CHROME__", "", "")

SLIDES = [
    ("第 1 页 · 封面", [
        ("CS4486 • PROJECT", C[0], C[1]),
        ("Iceberg Classification\nUsing Radar Images and Angles",
         "冰山分类：使用雷达图像与角度", "无。只有题目，不读数据。"),
        ("From a small CNN baseline to angle priors, model blending and student models",
         "从一个小型 CNN 基线，到角度先验、模型融合与学生模型", "无。方法演进顺序。"),
        ("TEAM", "团队", "无（栏目标签）。"),
        ("Huang Yifan   72400438   Team leader\nZheng Hongyu   72400229\nWu Kunze   72400463",
         "黄亦凡 72400438（组长）／郑鸿宇 72400229／吴坤泽 72400463", "无。组员与学号。"),
        ("COMPETITION", "比赛", "无（栏目标签）。"),
        ("Statoil/C-CORE\nIceberg Classifier Challenge",
         "Statoil／C-CORE 冰山分类挑战赛", "无。比赛名称。"),
        ("DATE", "日期", "无（栏目标签）。"),
        ("24 November 2026", "2026 年 11 月 24 日", "无。展示日期。"),
        ("TAGLINE", "一句话主张", "无（栏目标签）。"),
        ("Predict iceberg probabilities with lower Log Loss",
         "用更低的 Log Loss 预测冰山概率", "目标：让测试集 8,424 行的预测更准。"),
        ("CITYU-DG\nFinal Presentation\nCS4486\n01", C[0], C[1]),
    ]),
    ("第 2 页 · 项目路线图", [
        ("CS4486 PROJECT\nOur Project Roadmap\nPROJECT OVERVIEW",
         "CS4486 项目 · 我们的项目路线图 · 项目总览", "无。目录页。"),
        ("FIRST VERSION\nAnalyze the data, fix five folds and train a CNN baseline.\n"
         "Test flips, dropout, angle input and a missing-angle flag.",
         "第一阶段：分析数据、固定五折、训练 CNN 基线；再分别试翻转、dropout、角度输入、缺失角度标记。",
         "只读训练集 1,604 行：band_1/band_2 作输入、is_iceberg 作答案、按 inc_angle 分五折。"
         "测试集不参与。"),
        ("LATER UPDATES\nAdd angle priors, model blending and soft-label training.\n"
         "Tune the model, compare scores and study remaining errors.",
         "第二阶段：加入角度先验、模型融合与软标签训练；调参、比分数、分析残余错误。",
         "仍用训练集；新增把测试集 8,424 行图像送进模型取概率作软标签。"),
        ("01\nTask &\nmetric", "第一步：任务与指标", "无。"),
        ("Predict iceberg probability\nfrom radar images.\nMetric: Log Loss.",
         "从雷达图像预测冰山概率，指标为 Log Loss。", "分数由训练集 1,604 行的五折折外预测算出。"),
        ("02\nData &\nchallenges", "第二步：数据与挑战", "无。"),
        ("Check the data and missing angles.\nSet five validation folds and keep them fixed.",
         "检查数据与缺失角度；设定五折并保持不变。",
         "用训练集 inc_angle 数出 133 行为空，再分层切五折存成 folds.csv。"),
        ("03\nBaseline", "第三步：基线", "无。"),
        ("Train a small CNN.\nScore every row once, on data the model never saw.\nCompare changes fairly.",
         "训练一个小 CNN；每一行都由「没见过它」的模型打一次分；公平比较。",
         "训练集五折：每折 4 份训练、1 份预测，1,604 行各得折外概率，得 OOF 0.26063。"),
        ("04\nModel\nimprovements", "第四步：模型改进", "无。"),
        ("Add angle priors, blend predictions\nand retrain models on soft labels.",
         "加入角度先验、融合预测，并用软标签重训模型。",
         "先验来自训练集 inc_angle+is_iceberg；软标签来自测试集图像概率（无标签）。"),
        ("05\nResults &\nreflection", "第五步：结果与反思", "无。"),
        ("Compare scores and\nthe Top-5 target.\nExplain errors and limits.",
         "比较分数与历史前五目标；解释错误与局限。",
         "分数是测试集私榜值；错误分析用训练集折外预测。"),
        ("CS4486 Final Presentation\n02", C[0], C[1]),
    ]),
    ("第 3 页 · 任务与指标", [
        ("CS4486 PROJECT\n1. Task Description & Metric\nRADAR CLASSIFICATION",
         "CS4486 项目 · 1. 任务描述与指标 · 雷达分类", "无。"),
        ("COMPETITION SNAPSHOT", "比赛快照", "无（栏目标签）。"),
        ("Statoil / C-CORE challenge\nClassify satellite radar images\nSupport safer navigation",
         "Statoil／C-CORE 挑战赛；对卫星雷达图像分类；服务于更安全航行。", "无。背景说明。"),
        ("Two bands of the same object", "同一目标的两种波段", "无（图注标题）。"),
        ("PREDICTION SETUP", "预测设置", "无（栏目标签）。"),
        ("Classify: iceberg or ship\nInputs: HH, HV and incidence angle\nOutput: iceberg probability\n"
         "CSV columns: id, is_iceberg",
         "分类冰山或船；输入 HH、HV 与入射角；输出冰山概率；提交列为 id 与 is_iceberg。",
         "模型读训练集的 band_1（HH）、band_2（HV）、inc_angle，答案取 is_iceberg。"),
        ("OFFICIAL METRIC", "官方指标", "无（栏目标签）。"),
        ("Log Loss\nLower is better\nA confident wrong call is costly at sea, so the metric punishes overconfidence.",
         "Log Loss 越低越好；海上一次自信的错误代价高，所以指标惩罚过度自信。",
         "用预测概率与真实标签算（训练集有 is_iceberg，测试集看不到）。"),
        ("PROJECT GOAL", "项目目标", "无（栏目标签）。"),
        ("Beat a reproducible CNN baseline by treating incidence angle as evidence.\n"
         "Compare against the historical Top-5. Stretch goal: private < 0.08883.",
         "把入射角当证据，超过可复现的 CNN 基线；对照历史前五，冲刺私榜低于 0.08883。",
         "要超过的基线是训练集上的 0.26063；对照线是历史第五 0.08883。"),
        ("Top: a ship. Bottom: an iceberg.\nEach shown in both radar bands (HH, HV).",
         "上排是船、下排是冰山，各自展示 HH 与 HV 两个波段。",
         "从训练集取 1 条 is_iceberg=0 与 1 条 =1 的行，画它们的 band_1、band_2。"),
        ("CS4486 Final Presentation\n03", C[0], C[1]),
    ]),
    ("第 4 页 · 数据分析与挑战", [
        ("CS4486 PROJECT\n2. Data Analysis & Challenges\nDATA AND VALIDATION",
         "CS4486 项目 · 2. 数据分析与挑战 · 数据与验证", "无。"),
        ("Training labels", "训练标签分布", "指训练集 is_iceberg 列。"),
        ("53.1% ships / 46.9% icebergs\nThe classes are fairly balanced.",
         "船 53.1%、冰山 46.9%，类别基本均衡。",
         "训练集 1,604 行中 is_iceberg=0 有 851 行、=1 有 753 行。"),
        ("DATASET PROFILE", "数据概况", "无（栏目标签）。"),
        ("1,604 train / 8,424 test samples\nTwo 75 × 75 images + angle\nNormalize using each training fold",
         "训练 1,604 条、测试 8,424 条；每条两张 75×75 图加角度；用各自训练折算归一化。",
         "1,604/8,424 是两份文件行数；归一化只用当前折训练行像素。"),
        ("MAIN CHALLENGES", "主要挑战", "无（栏目标签）。"),
        ("Small, noisy radar dataset\n133 missing train angles\nTest angles are complete",
         "数据小且噪声大；训练集 133 条角度缺失；测试集角度完整。",
         "133 = 训练集 inc_angle 为空的行数；完整 = 测试集 8,424 行 inc_angle 无空值。"),
        ("TAKEAWAY", "关键结论", "无（栏目标签）。"),
        ("Exact-angle rows almost always share a label (98.6%).\nNearby angles agree only 84% of the time.\n"
         "So we flag missing angles and treat the two cases separately.",
         "角度完全相同的行几乎总是同标签（98.6%）；只有邻近角度时一致率约 84%。"
         "因此标记缺失角度并把两类分开处理。",
         "用训练集 inc_angle+is_iceberg：同角度分组一致率 98.6%；最近但不同角度约 84%。"),
        ("Missing-angle shift", "缺失角度的偏移", "无（小卡标题）。"),
        ("133 missing train angles, all ships", "训练集 133 条缺失角度全部是船",
         "查那 133 行的 is_iceberg，全部为 0；测试集没有缺失角度。"),
        ("Use the same five folds", "使用同一套五折", "无（小卡标题）。"),
        ("About 321 validation samples per fold", "每折约 321 条验证样本",
         "训练集 1,604 行分成 321/321/321/321/320。"),
        ("Extra check: group by angle", "额外检查：按角度分组", "无（小卡标题）。"),
        ("Put equal-angle samples in the same fold", "把同角度样本放进同一折",
         "用训练集 inc_angle 把同角度行绑在一起再分折，重训检验是否靠同角度泄漏。"),
        ("CS4486 Final Presentation\n04", C[0], C[1]),
    ]),
    ("第 5 页 · 基线模型", [
        ("CS4486 PROJECT\n3. Baseline Model\nFIVE-FOLD BASELINE",
         "CS4486 项目 · 3. 基线模型 · 五折基线", "无。"),
        ("BASELINE PIPELINE", "基线流程", "无（栏目标签）。"),
        ("60,433 parameters, trained from scratch", "60,433 个参数，从零训练",
         "参数个数，不是数据量。"),
        ("Preprocess\nScale within\neach fold", "预处理：在每折内部做缩放",
         "用当前折训练行的像素均值与标准差缩放。"),
        ("Radar input\nHH + HV\n75 × 75", "雷达输入 HH + HV，尺寸 75×75",
         "训练集每行 band_1 与 band_2 还原成 75×75 两通道。"),
        ("Model\n3 conv blocks\n16 / 32 / 64", "模型：3 个卷积块，通道 16/32/64", "无。结构描述。"),
        ("Loss\nBinary\ncross-entropy", "损失：二元交叉熵", "用输出概率与训练行 is_iceberg 计算。"),
        ("WHY THIS BASELINE", "为什么用这个基线", "无（栏目标签）。"),
        ("Start with a small, repeatable CNN.\nKeep the folds and budget fixed\nto compare changes fairly.",
         "从小而可重复的 CNN 开始；固定折与训练预算才能公平比较。",
         "固定同一训练集五折与同一轮数，改动才可归因。"),
        ("TRAINING SETUP", "训练设置", "无（栏目标签）。"),
        ("Adam, LR 0.001, batch size 32\nUp to 30 epochs; early stop after 7\nCPU; weight decay 0.0001",
         "Adam、学习率 0.001、批大小 32；最多 30 轮，验证 7 轮不升就停；CPU；权重衰减 0.0001。",
         "每批 32 条取自当前折训练行；早停只看验证行分数。"),
        ("BASELINE SCORE", "基线分数", "无（栏目标签）。"),
        ("OOF Log Loss 0.26063.\nPublished reference 0.24638, so this environment differs slightly.",
         "折外 Log Loss 0.26063；公开参考 0.24638，说明环境略有差异。",
         "0.26063 是五折验证预测与训练集 1,604 行 is_iceberg 算出的 Log Loss。"),
        ("First-version ablations: OOF Log Loss", "第一版消融：折外 Log Loss", "四组都只用训练集，同一五折。"),
        ("Configuration | Loss\nV0: baseline | 0.26063\nV2: higher dropout | 0.26703\n"
         "V3: angle branch | 0.25407\nV4: missing flag | 0.22683",
         "V0 基线 0.26063；V2 提高 dropout 0.26703；V3 加角度分支 0.25407；V4 加缺失标记 0.22683。",
         "V2 把 dropout 0.25→0.40；V3 把 inc_angle 作输入；V4 再加缺失标记；均用训练集同一五折。"),
        ("V0–V5 are first-version experiments. Later submission files have a separate version numbering.",
         "V0–V5 是第一版实验编号；提交文件用另一套编号。", "无。编号说明。"),
        ("CS4486 Final Presentation\n05", C[0], C[1]),
    ]),
    ("第 6 页 · 模型改进 A / B / C", [
        ("CS4486 PROJECT\n4. Model Improvements\nCHANGE AND EVIDENCE",
         "CS4486 项目 · 4. 模型改进 · 改动与证据", "无。"),
        ("A  Two-level prior", "A 两级先验", "无（卡片标题）。"),
        ("HYPOTHESIS", "假设", "无（小标题）。"),
        ("Same angle, same label.\nExact matches are rare.",
         "同角度通常同标签；但完全相同的角度匹配很少。",
         "依据是训练集 inc_angle+is_iceberg 的组内一致率 98.6%。"),
        ("IMPLEMENTATION", "实现", "无（小标题）。"),
        ("Use the exact group first.\nElse a Gaussian kernel.\nFit on training folds only.",
         "先用完全相同角度的组；没有就用高斯核；只在训练折上拟合。",
         "每折用该折训练行的 inc_angle+is_iceberg 建先验；验证行只查表。"),
        ("EFFECT", "效果", "无（小标题）。"),
        ("Nested OOF\n0.25381 → 0.12109", "嵌套折外分数 0.25381 → 0.12109",
         "训练集折外分数：只用图像 0.25381，加角度先验后 0.12109。"),
        ("OOF gain: −0.13272", "折外提升 −0.13272", "两个训练集折外分数之差。"),
        ("B  Grouped fusion", "B 分组融合", "无（卡片标题）。"),
        ("Exact and near matches\nneed different trust.",
         "精确匹配与近邻匹配需要不同程度的信任。", "依据：训练集 exact 98.6%、near 约 84%。"),
        ("Separate weights for exact\nand near rows.\nFit in each training fold.",
         "对精确行与近邻行分别给权重；在每折训练部分拟合。",
         "用该折训练行 is_iceberg 拟合两个系数，验证行只打分。"),
        ("Nested OOF\n0.11556 → 0.10721", "嵌套折外分数 0.11556 → 0.10721",
         "训练集折外分数：固定权重 0.11556，分组权重 0.10721。"),
        ("OOF gain: −0.00835", "折外提升 −0.00835", "上两个训练集折外分数之差。"),
        ("C  Student diversity", "C 学生模型多样性", "无（卡片标题）。"),
        ("One CNN repeats\nits own mistakes.\nUnlabelled data unused.",
         "单个 CNN 会重复自己的错误；无标签数据被浪费。", "指测试集 8,424 行图像未被用于训练。"),
        ("Distil soft labels, then add a\nshape MLP and a mixup student for different error patterns.",
         "蒸馏软标签，再加入 shape MLP 与 mixup 学生，制造不同错误模式。",
         "把测试集 8,424 行图像送入教师得概率（无标签）作软标签；shape MLP 只用训练集 90 维统计量。"),
        ("Distilled image OOF\n0.1959 → 0.1833", "蒸馏后图像折外分数 0.1959 → 0.1833",
         "训练集折外分数：加软标签前 0.1959、后 0.1833。"),
        ("OOF gain: −0.0126", "折外提升 −0.0126", "上两个训练集折外分数之差。"),
        ("Angle evidence changed the score most (−0.13); the later steps added far less. "
         "Gains are per step, not additive.",
         "角度证据带来的变化最大（−0.13）；后续步骤小得多；各步增益不能相加。",
         "全部为训练集折外分数之差，起点不同。"),
        ("CS4486 Final Presentation\n05", C[0], C[1]),
    ]),
    ("第 7 页 · 模型改进 D / E / F", [
        ("CS4486 PROJECT\n4. Model Improvements: Refinement\nCHANGE AND EVIDENCE",
         "CS4486 项目 · 4. 模型改进：精修 · 改动与证据", "无。"),
        ("D  Training recipe", "D 训练配方", "无（卡片标题）。"),
        ("HYPOTHESIS", "假设", "无（小标题）。"),
        ("Tuning the setup beats\nadding capacity.", "调好训练设置比加大容量更有效。",
         "依据：加宽与收窄网络都没有改善。"),
        ("IMPLEMENTATION", "实现", "无（小标题）。"),
        ("AdamW, batch 32.\nLR 0.003, decay 0.01.\nCosine schedule.\nSmall 93k CNNs.",
         "AdamW、批大小 32、学习率 0.003、权重衰减 0.01、余弦调度、约 9.3 万参数小 CNN。",
         "仍在训练集上训练，只改超参数。"),
        ("EFFECT", "效果", "无（小标题）。"),
        ("Tuned ensemble OOF 0.10194 → 0.09976", "调参后集成折外 0.10194 → 0.09976",
         "训练集折外分数，降 0.00218。"),
        ("−0.0016", "−0.0016", "私榜分数变化（0.08944→0.08784），与折外差口径不同。"),
        ("E  Broader soft labels", "E 更宽的软标签", "无（卡片标题）。"),
        ("A strict cutoff throws\naway most unlabelled rows.", "过严的门槛丢掉大部分无标签行。",
         "旧门槛只从测试集 8,424 行挑出约 3,500 行。"),
        ("Threshold 0.03 → 0.25.\nAbout 6,260 pseudo rows.\nWeight 0.25.\n20 epochs.",
         "阈值 0.03→0.25；约 6,260 条伪标签；权重 0.25；训练 20 轮。",
         "用教师对测试集 8,424 行给概率，保留 p<0.25 或 p>0.75 的约 6,260 行，答案即该概率。"),
        ("Threshold OOF 0.09937 → 0.09829", "阈值调整后折外 0.09937 → 0.09829",
         "训练集折外分数，降 0.00108。"),
        ("−0.0003", "−0.0003", "私榜分数变化（0.08784→0.08752）。"),
        ("F  Angle calibration", "F 角度标定", "无（卡片标题）。"),
        ("CV and the leaderboard\nprefer different weights.", "交叉验证与排行榜偏好不同权重。",
         "训练集最优 1.00，私榜最优约 0.90。"),
        ("Scale both angle weights\nby 0.90 after fitting.\nSame CNNs, LB feedback.",
         "拟合后把两个角度权重同乘 0.90；模型不变，依据是排行榜反馈。",
         "不动数据不重训，只改系数后重新对测试集 8,424 行预测。"),
        ("Private 0.08708, CV slightly worse", "私榜 0.08708，交叉验证略变差",
         "私榜由 Kaggle 用测试集标签算；CV 指训练集折外 0.098287→0.098753。"),
        ("−0.0004", "−0.0004", "私榜分数变化（0.08752→0.08708）。"),
        ("Recipe tuning gave the largest step (−0.0021 OOF); the last two steps added only 0.0005..",
         "调参贡献最大（−0.0021）；最后两步只增加约 0.0005。",
         "D 降 0.00218、E 降 0.00108，故该句偏保守。"),
        ("CS4486 Final Presentation\n06", C[0], C[1]),
    ]),
    ("第 8 页 · 最终结果", [
        ("CS4486 PROJECT\n5. Final Results\nKAGGLE LATE SUBMISSIONS",
         "CS4486 项目 · 5. 最终结果 · Kaggle 迟交提交", "无。"),
        ("BEST PRIVATE SCORE", "最佳私榜分数", "无（栏目标签）。"),
        ("0.08708\nversion19.csv", "0.08708，来自 version19.csv",
         "该 CSV 是对测试集 8,424 行的预测，分数由 Kaggle 用测试集标签算。"),
        ("TARGET CHECK", "目标核对", "无（栏目标签）。"),
        ("Reached the level of the top five: fourth", "达到前五水平：第四",
         "由 0.08883 与 0.08708 相减推算，无官方排名。"),
        ("Stretch goal < 0.08883:  met", "冲刺目标 < 0.08883：已达成", "0.08708 低于 0.08883，差 0.00175。"),
        ("KEY TAKEAWAY", "核心结论", "无（栏目标签）。"),
        ("62% lower private loss than our first submission.\n"
         "The gain came from angle evidence, not from a bigger model.",
         "私榜损失比首个提交低 62%；收益来自角度证据而非更大模型。",
         "首个提交 0.23022、最终 0.08708，均来自测试集评分。"),
        ("Private Log Loss: selected submissions", "部分提交的私榜 Log Loss",
         "每个点是该次提交在测试集私榜上的分数。"),
        ("Where errors remain", "错误仍在哪里", "无（栏目标签）。"),
        ("Rows with only a near angle match carry 87.2% of the loss.\n"
         "Where image and angle disagree (231 rows), 50.7% of the loss sits.",
         "仅近邻角度匹配的行承担 87.2% 损失；图像与角度相反的 231 行占 50.7%。",
         "用训练集 1,604 行折外预测分组统计（六成员开发模型），不是测试集分析。"),
        ("Late submissions have no official rank. Public and Private scores use different test subsets.",
         "迟交没有官方排名；公榜与私榜用测试集不同子集。", "无。"),
        ("CS4486 Final Presentation\n08", C[0], C[1]),
    ]),
    ("第 9 页 · 讨论与反思", [
        ("CS4486 PROJECT\n6. Discussion & Reflection\nLESSONS AND NEXT STEPS",
         "CS4486 项目 · 6. 讨论与反思 · 经验与下一步", "无。"),
        ("WHAT WORKED", "有效的做法", "无（栏目标签）。"),
        ("Angle information added context: OOF 0.25381 → 0.12109.\n"
         "Group-specific weights helped: 0.11556 → 0.10721.\n"
         "Different student models made the blend stronger: image OOF 0.1959 → 0.1833.",
         "角度信息提供上下文（0.25381→0.12109）；分组权重有帮助（0.11556→0.10721）；"
         "不同学生模型让融合更强（0.1959→0.1833）。",
         "三条都是训练集折外分数，来自三组对照。"),
        ("WHAT DID NOT WORK", "无效的做法", "无（栏目标签）。"),
        ("Stronger augmentation and larger ensembles often hurt.\n"
         "Possible causes: lost radar details and similar model errors.\n"
         "Next time: change one factor at a time and keep augmentation matched to training.",
         "更强增强与更大集成往往更差；可能因雷达细节被破坏、成员错误相似；"
         "下次应一次只改一个因素并让增强匹配训练。",
         "依据：几何增强第 0 折 0.2951；贪心 26 成员 0.10561。"),
        ("Two failed experiments", "两个失败的实验", "无（小卡标题）。"),
        ("Affine augmentation, fold 0:\nLog Loss 0.2951\n \n26-member greedy ensemble:\nOOF 0.10561 vs 0.10194",
         "几何增强第 0 折 0.2951；26 成员贪心集成 0.10561，对手工 6 成员 0.10194。",
         "0.2951 用训练集第 0 折 321 行验证样本算；后两者为训练集全体折外分数。"),
        ("LIMITATIONS & NEXT STEPS", "局限与下一步", "无（栏目标签）。"),
        ("Two limits dominate: the image model is weak without an exact angle match, \n"
         "and every score was partly chosen by tuning.",
         "两个主要局限：无精确角度匹配时图像模型偏弱；所有分数部分受调参影响。",
         "前者来自训练集折外预测分组统计；后者指部分后处理由私榜挑选。"),
        ("Only 1,604 labelled examples", "只有 1,604 条带标签样本", "训练集行数。"),
        ("Scores partly reflect tuning", "分数部分反映调参", "指后处理与超参经多轮比较选出。"),
        ("Next: improve radar image features", "下一步：改进雷达图像特征",
         "针对训练集中「无精确角度伙伴」的行。"),
        ("CS4486 Final Presentation\n09", C[0], C[1]),
    ]),
    ("第 10 页 · AI 使用与复现", [
        ("CS4486 PROJECT\n7. AI Usage & Reproducibility\nTOOLS AND TEAMWORK",
         "CS4486 项目 · 7. AI 使用与复现 · 工具与分工", "无。"),
        ("AI TOOL USAGE", "AI 工具使用", "无（栏目标签）。"),
        ("ChatGPT / Deepseek helped with debug code and drafting; the design decisions, "
         "the validation protocol and modeling choices were set and verified by the group.",
         "ChatGPT／DeepSeek 协助调试代码与起草文字；设计、验证方案与建模选择由小组制定并核验。",
         "无数据取值。"),
        ("Tools: ChatGPT and Deepseek", "工具：ChatGPT 与 DeepSeek", "无。"),
        ("Used for: Debugging and explaining results", "用途：调试代码与解释结果", "无。"),
        ("The design decisions,validation protocol and modeling choices were independent",
         "设计、验证方案与建模选择由小组独立完成", "无。"),
        ("AI support was used across the roles listed below", "AI 支持贯穿下列各角色", "无。"),
        ("Huang: data analysis   |   Wu : hyperparameter changes suggestion\nZheng: error analysis, predictions",
         "黄：数据分析；吴：超参数调整建议；郑：误差分析与预测", "分工说明。"),
        ("HOW TO REPRODUCE", "如何复现", "无（栏目标签）。"),
        ("Fold seed 42 (fixed split file); model seeds 71, 83, 97, 909,\n   1201, 1401, 2601, saved in every checkpoint.",
         "折种子 42（固定划分文件）；模型种子 71、83、97、909、1201、1401、2601，存入各权重文件。",
         "42 决定训练集 1,604 行如何分五折；其余种子决定初始化与打乱。"),
        ("Python 3.12, PyTorch 2.15 (CPU), NumPy 2.5; 16 threads, no GPU;",
         "Python 3.12、PyTorch 2.15（CPU）、NumPy 2.5；16 线程，无 GPU。", "环境，无数据取值。"),
        ("AdamW, 20 epochs, batch 32, LR 3e-3 / 5e-3, dropout 0.4-0.5,\n   weight decay 1e-2, 8-view TTA.",
         "AdamW、20 轮、批大小 32、学习率 3e-3 或 5e-3、dropout 0.4–0.5、权重衰减 1e-2、8 视角 TTA。",
         "训练用训练集；8 视角 TTA 作用于测试集每行图像。"),
        ("github.com/YifanHuang-if/cs4486-iceberg", "代码仓库地址", "无。"),
        ("Log Loss on OOF predictions; level-2 fitted inside each training \n   fold; submission IDs checked.",
         "指标用折外预测算 Log Loss；第二层只在各折训练部分拟合；提交前核对 ID。",
         "折外预测来自训练集；ID 核对针对测试集 8,424 行。"),
        ("None used. All CNNs trained from scratch: 60,433 baseline, \n   93,521 final.",
         "未用预训练权重；全部 CNN 从零训练：基线 60,433 参数，最终 93,521 参数。", "参数个数。"),
        ("No pretrained weights in the reported final pipeline", "最终方案中没有预训练权重", "无。"),
        ("CS4486 Final Presentation\n10", C[0], C[1]),
    ]),
    ("第 11 页 · 问答页", [
        ("Q&A", "问答", "无。"),
        ("Submission evidence", "提交证据", "无（小标题）。"),
        ("Private score: left   |   Public score: right", "左为私榜分数，右为公榜分数",
         "两个分数都来自测试集，只是不同子集。"),
        ("Main idea\n \nCombine rar\nand angle predictions\n \nBest Private\n0.08708",
         "核心思路：把雷达预测与角度预测结合；最佳私榜 0.08708（原文 rar 为 radar 笔误）。",
         "测试集图像给概率，训练集角度给先验，二者结合。"),
        ("CS4486 Final Presentation\n11", C[0], C[1]),
    ]),
]


def add_block(document, original: str, translation: str, note: str) -> None:
    chrome = translation == "__CHROME__"
    source = document.add_paragraph()
    source.paragraph_format.space_after = Pt(1)
    tag = source.add_run("原文  ")
    tag.bold = True
    tag.font.size = Pt(9)
    tag.font.color.rgb = ACCENT
    text = source.add_run(original)
    text.font.size = Pt(9.5)
    text.bold = True

    if chrome:
        line = document.add_paragraph()
        line.paragraph_format.left_indent = Pt(12)
        line.paragraph_format.space_after = Pt(6)
        run = line.add_run("（版式文字：页眉、页脚、页码或机构标识，不参与内容）")
        run.font.size = Pt(9)
        run.font.color.rgb = GREY
        return

    translated = document.add_paragraph()
    translated.paragraph_format.left_indent = Pt(12)
    translated.paragraph_format.space_after = Pt(1)
    tag2 = translated.add_run("翻译  ")
    tag2.bold = True
    tag2.font.size = Pt(9)
    tag2.font.color.rgb = ACCENT
    body = translated.add_run(translation)
    body.font.size = Pt(9.5)

    data_line = document.add_paragraph()
    data_line.paragraph_format.left_indent = Pt(12)
    data_line.paragraph_format.space_after = Pt(8)
    tag3 = data_line.add_run("数据  ")
    tag3.bold = True
    tag3.font.size = Pt(9)
    tag3.font.color.rgb = GREY
    note_run = data_line.add_run(note)
    note_run.font.size = Pt(9)
    note_run.font.color.rgb = GREY


def main() -> int:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(10.5)

    document.add_heading("Group12 · PPT 逐页英中对照", level=0)

    intro = document.add_paragraph()
    run = intro.add_run(
        "按 11 页顺序列出每一个文字块：英文整块照抄（保留原换行，不与其他块拼接），下面是中文翻译，"
        "再说明它引用了哪份数据的哪一列。\n"
        "训练集 1,604 行含 band_1/band_2、inc_angle、is_iceberg；测试集 8,424 行只有前三项，没有标签。")
    run.font.color.rgb = GREY
    run.font.size = Pt(9.5)

    total = 0
    for slide_title, blocks in SLIDES:
        heading = document.add_heading(slide_title, level=1)
        heading.runs[0].font.color.rgb = DARK
        for original, translation, note in blocks:
            add_block(document, original, translation, note)
            total += 1

    document.save(OUT)
    print(f"saved: {OUT} | blocks: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
