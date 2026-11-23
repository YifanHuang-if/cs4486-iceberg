"""Repair section 9: restore the truncated sentence and refresh its numbers."""
from __future__ import annotations

from docx import Document

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

REPLACEMENTS = {
    "已测约 38 种干预":
        "已测约 38 种干预，只有 5 种有效：分组加权嵌套堆叠、超参调优、伪标签阈值放宽、"
        "蒸馏、shape MLP。其余全部无效或有害，逐条记录在负结果表中并附代价。",
    "角度层已被测死":
        "角度层已被用尽：同角度样本一致率 98.5%，只有邻近角度的样本降到 81.9%；"
        "在这 709 行 near 样本上，只用角度的损失是 0.371、只用图像 0.255、融合后 0.199。"
        "核宽 σ=0.012 经扫描确认最优，先验层没有剩余空间，瓶颈在图像模型。",
    "要跨过最后 0.0045":
        "要跨过最后 0.0041，需要的是本项目手段之外的量级：数十至上百个去相关成员"
        "（前五名分别用了 100+、数百、7×30 次拟合），或在 SAR 上预训练过的表征。"
        "两者都超出本机 CPU 环境与设定时间。",
    "因此结论是：当前 private 0.08752":
        "因此结论是：当前 private 0.08708 是这套方法在本环境下的收敛点，而不是某一步没做完。"
        "继续增加实验次数不会改变结果，只会增加负结果条数。",
}


def main() -> int:
    document = Document(PATH)
    changed = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        for prefix, replacement in REPLACEMENTS.items():
            if text.startswith(prefix):
                runs = paragraph.runs
                if not runs:
                    runs = [paragraph.add_run()]
                runs[0].text = replacement
                for run in list(runs)[1:]:
                    run._r.getparent().remove(run._r)
                changed += 1
                break
    print(f"section 9 paragraphs rewritten: {changed}")
    document.save(PATH)
    print(f"saved: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
