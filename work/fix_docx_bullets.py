"""Correct the stale closing bullets in the iteration document."""
from __future__ import annotations

from docx import Document

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

REPLACE = {
    "CV 0.1019 与 private 0.0894 差 0.0125":
        "CV 0.0982 与 private 0.0875 差 0.0108，而 public 0.1090 高出 0.0107——"
        "两个榜几乎对称分布在 CV 两侧，说明 3,424 行被切成了一难一易两半。",
    "目标 0.083 尚未达到，缺口 0.0064。":
        "目标 private < 0.083 尚未达到，缺口 0.0045；public < 0.086 的缺口为 0.023，"
        "按当前换算需要 CV≈0.075，短期不可达。",
    "引入非 CNN 成员（GLCM 纹理 + 树模型）":
        "GLCM 纹理 + 树模型已实测：树单独 0.24481、加入集成 0.10280，无效（见负结果表）。",
}


def main() -> int:
    document = Document(PATH)
    changed = 0
    for paragraph in document.paragraphs:
        text = paragraph.text
        for old, new in REPLACE.items():
            if old in text:
                if paragraph.runs:
                    paragraph.runs[0].text = new
                    for run in paragraph.runs[1:]:
                        run.text = ""
                else:
                    paragraph.add_run(new)
                changed += 1
                break
    document.save(PATH)
    print(f"updated {changed} bullets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
