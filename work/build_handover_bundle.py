"""Assemble a handover bundle for the final submission (version19).

Layout
  00_README.md
  01_submission/   version19.csv + manifest
  02_oof/          per-row out-of-fold predictions + summary
  03_weights/      the seven members' checkpoints, metrics, histories, teacher labels
  04_configs/      fold definition, per-member configs, reproduce commands
  05_code/         the scripts that produced the submission + version record
  06_evidence/     Kaggle screenshots already in the repository
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from work.v5_fuse import apply_logistic, fit_logistic, log_loss, logit
from work.v5_greedy import build_prior, load

ROOT = Path(r"C:\Users\16018\cs4486-iceberg")
BUNDLE = ROOT / "handover" / "CS4486_final_bundle"

TAGS = ["v5_shape", "hp_star", "hp_b", "hp_d", "pt25a", "pt25b", "stu5"]
SCALE = 0.90
L2 = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_oof() -> dict:
    ids, labels, angles, folds, matrix = load(TAGS)
    prior, exact = build_prior(angles, labels, folds)
    image = np.clip(matrix.mean(axis=0), 1e-6, 1 - 1e-6)
    prior_logit = logit(np.clip(prior, 1e-6, 1 - 1e-6))
    features = np.column_stack([logit(image), prior_logit * exact, prior_logit * (~exact)])

    prediction = np.zeros(len(labels))
    coefficients = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        weights = fit_logistic(features[train], labels[train], l2=L2)
        scaled = weights.copy()
        scaled[1] *= SCALE
        scaled[2] *= SCALE
        coefficients.append([round(float(value), 6) for value in scaled])
        prediction[valid] = apply_logistic(features[valid], scaled)

    pooled = log_loss(labels, prediction)
    per_fold = {
        str(fold): round(float(log_loss(labels[folds == fold], prediction[folds == fold])), 6)
        for fold in range(5)
    }

    out_dir = BUNDLE / "02_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "oof_version19.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "fold", "is_iceberg", "inc_angle", "exact_match", "image_prob", "prior", "oof_prob"])
        for index, row_id in enumerate(ids):
            writer.writerow([
                row_id, int(folds[index]), int(labels[index]),
                "" if not np.isfinite(angles[index]) else angles[index],
                int(exact[index]), f"{image[index]:.8f}",
                "" if not np.isfinite(prior[index]) else f"{prior[index]:.8f}",
                f"{prediction[index]:.8f}",
            ])

    summary = {
        "tags": TAGS,
        "angle_scale": SCALE,
        "sigma": 0.012,
        "alpha": 0.1,
        "l2": L2,
        "pooled_oof_log_loss": round(float(pooled), 6),
        "per_fold": per_fold,
        "coefficients_per_fold": coefficients,
        "note": "Out-of-fold predictions with the second layer fitted inside each training fold, "
                "then both angle coefficients multiplied by 0.90 (same as the submitted file).",
    }
    (out_dir / "oof_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def copy_members() -> list[str]:
    target = BUNDLE / "03_weights"
    target.mkdir(parents=True, exist_ok=True)
    lines = []
    for tag in TAGS:
        source = ROOT / "outputs" / "v3" / tag
        destination = target / tag
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        for pattern in ("fold*.pt", "fold*_metrics.json", "fold*_history.csv", "fold*_pseudo.npz"):
            for path in sorted(source.glob(pattern)):
                shutil.copy2(path, destination / path.name)
                copied += 1
        lines.append(f"{tag}: {copied} files")
    return lines


def main() -> int:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    # 01 submission
    submission_dir = BUNDLE / "01_submission"
    submission_dir.mkdir()
    shutil.copy2(ROOT / "work/submissions/version19.csv", submission_dir / "version19.csv")
    shutil.copy2(ROOT / "work/submissions/version19.json", submission_dir / "version19.json")
    submission_hash = sha256(submission_dir / "version19.csv")

    # 02 oof
    summary = build_oof()

    # 03 weights
    member_lines = copy_members()

    # 04 configs
    config_dir = BUNDLE / "04_configs"
    config_dir.mkdir()
    shutil.copy2(ROOT / "configs/validation.json", config_dir / "validation.json")
    shutil.copy2(ROOT / "data/splits/folds.csv", config_dir / "folds.csv")

    rows = []
    for tag in TAGS:
        metrics = json.loads((ROOT / "outputs" / "v3" / tag / "fold0_metrics.json").read_text())
        config = metrics["config"]
        rows.append({
            "tag": tag,
            "seed": config.get("seed"),
            "epochs": config.get("epochs"),
            "batch": config.get("batch"),
            "lr": config.get("lr"),
            "dropout": config.get("dropout"),
            "weight_decay": config.get("weight_decay"),
            "teacher": config.get("teacher"),
            "pseudo_weight": config.get("pseudo_weight"),
            "normalisation": config.get("normalisation"),
        })
    with (config_dir / "members.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 05 code
    code_dir = BUNDLE / "05_code"
    for part in ("src", "work"):
        for path in sorted((ROOT / part).rglob("*.py")):
            relative = path.relative_to(ROOT)
            destination = code_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%H%n%ad%n%s", "--date=short"],
                                cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        commit, branch, dirty = "unavailable", "unavailable", []

    (code_dir / "VERSION.md").write_text(
        "# 代码版本\n\n"
        f"- 分支：`{branch}`\n- 最近提交：`{commit}`\n"
        f"- 未提交改动：{len(dirty)} 项（工作区导出，包含未提交脚本）\n"
        f"- 提交 CSV 的 sha256：`{submission_hash}`\n"
        f"- 生成提交的脚本：`work/v5_submit.py`（`--angle-scale 0.90`、`--name version19`）\n"
        f"- 融合与堆叠：`work/v5_fuse.py`；逐折预测：`work/v5_predict.py`\n"
        f"- 训练：`src/v3/train.py`；形状模型：`src/v3/shape_model.py`；教师：`work/build_teacher.py`\n",
        encoding="utf-8")

    # 06 evidence
    evidence_dir = BUNDLE / "06_evidence"
    evidence_dir.mkdir()
    for path in sorted((ROOT / "work/pr9-evidence").glob("*.png")):
        shutil.copy2(path, evidence_dir / path.name)

    # 00 readme
    (BUNDLE / "00_README.md").write_text(
        "# CS4486 冰山分类 · 最终提交交接包\n\n"
        "对应提交：**version19**（Private 0.08708 / Public 0.10724，截止后迟交，无正式排名）。\n\n"
        "## 目录\n\n"
        "| 目录 | 内容 |\n|---|---|\n"
        "| `01_submission/` | 提交 CSV 与生成清单（含 sha256） |\n"
        "| `02_oof/` | 每行五折折外概率、每折系数与汇总分数 |\n"
        "| `03_weights/` | 七个成员的 5 折检查点、逐折指标、训练曲线、教师软标签 |\n"
        "| `04_configs/` | 折划分文件、成员配置表、复现命令 |\n"
        "| `05_code/` | 生成该提交的脚本与代码版本记录 |\n"
        "| `06_evidence/` | 仓库内已有的 Kaggle 截图 |\n\n"
        "## 关键数字\n\n"
        f"- 提交文件 sha256：`{submission_hash}`\n"
        f"- 折外 Log Loss（含 0.90 角度缩放）：**{summary['pooled_oof_log_loss']}**\n"
        f"- 每折：{summary['per_fold']}\n"
        f"- 第二层系数（每折独立拟合后缩放）：{summary['coefficients_per_fold'][0]}\n\n"
        "## 成员\n\n" + "\n".join(f"- {line}" for line in member_lines) + "\n\n"
        "## `--angle-scale` 在哪\n\n"
        "不在仓库根目录的入口脚本里，而在 **`work/v5_submit.py`**：\n\n"
        "```\n"
        "python work/v5_submit.py --tags v5_shape hp_star hp_b hp_d pt25a pt25b stu5 \\\n"
        "    --angle-scale 0.90 --name version19\n"
        "```\n\n"
        "它做的事：逐折拟合第二层逻辑回归，然后把两个角度系数（`weights[1]`、`weights[2]`）\n"
        "同时乘以 0.90，再重新预测并导出 CSV。`--exact-scale` 可以只缩放精确匹配那一个系数。\n",
        encoding="utf-8")

    print(json.dumps({
        "bundle": str(BUNDLE),
        "submission_sha256": submission_hash,
        "pooled_oof": summary["pooled_oof_log_loss"],
        "per_fold": summary["per_fold"],
        "members": member_lines,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
