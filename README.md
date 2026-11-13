# CS4486 Iceberg Classification

三人小组课程项目：Statoil/C-CORE Iceberg Classifier Challenge。
比赛：https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge

## 开始使用
1. 每位成员接受比赛规则并自行下载数据。
2. 将实际文件 train.json、test.json、sample_submission.csv 放入 data/raw/。不要放入同名文件夹或压缩包。
3. 使用 VS Code 打开本仓库。环境版本由团队确认后统一记录；当前尚未安装依赖或实现模型。
4. 阅读 docs/COLLABORATION.md，领取任务后创建分支。

## 目录
- data/raw/：本地原始数据，不上传。
- data/splits/：固定验证划分（样本 ID 与折号）。
- notebooks/：各成员独立的数据分析 Notebook。
- src/：数据处理、训练和评估代码；models/ 放模型定义。
- configs/：实验配置。
- results/：小型结果表和报告图表。
- outputs/：本地模型和预测输出，不上传。
- report/：报告与展示材料。
- docs/：分工、贡献和 AI 使用记录。

## 课程交付
报告 PDF 与课堂展示各占 50%；使用课程模板，披露引用与 AI 用途，记录实际贡献小时数，提供可复现代码和环境。
主要指标为 Log Loss（越低越好）。Top-5 的比较口径须与老师确认，不预设分数门槛。
