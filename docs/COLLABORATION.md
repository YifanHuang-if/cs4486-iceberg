# 三人协作约定（角色待认领）

| 角色 | 主要工作 | 主要文件 |
|---|---|---|
| A | 数据分析、缺失值、固定验证划分、统一评估 | src/data.py、src/make_folds.py、src/evaluate.py |
| B | 基础 CNN、训练与预测流程 | src/models/baseline.py、src/train.py、src/predict.py |
| C | 角度融合、改进实验、结果分析 | src/models/fusion.py、configs/、src/summarize_results.py |

文件名是分工约定，模型代码尚未实现。每个人负责对应报告部分，并理解共同流程。

## 同步流程
1. GitHub Desktop 切到 main，Fetch origin / Pull origin 获取更新。
2. 从最新 main 创建任务分支，如 data-validation、baseline-training、fusion-model。
3. 修改自己负责的文件；共同文件提前沟通。
4. 检查改动并 Commit，再 Push origin；本地保存不会自动同步。
5. 创建 Pull Request，请另一成员检查后合并 main。
6. 其他成员同步 main 后再开始依赖这些修改的工作。

## 开始实验前共同确认
- 统一相对数据路径 data/raw/。
- 检查数据后确定验证策略并固定划分；同一组对比使用相同划分。
- 统一图像、角度、缺失标记的输入格式和输出 logit 格式。
- 预处理只在训练部分拟合；增强不能先于划分产生跨集合重复样本。
- 统一 Log Loss 计算；记录随机种子、代码提交版本、配置和运行环境。
- 每个实验独立配置及输出目录；不要多人修改同一个 Notebook。
- 依赖版本和 PyTorch CPU/GPU 安装方案确认后再添加环境清单。

## 共享范围
仓库设为 Private，邀请已确认的两位队友。数据由各自接受规则后下载；不上传密钥、原始数据、模型权重或环境目录。
