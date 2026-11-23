"""Fix the minute/second formatting on the duration line."""
from __future__ import annotations

from pathlib import Path

from docx import Document

PATH = Path(r"D:\Desktop\CS4486_6分钟演讲稿.docx")

TEXT = ("口播时长测算：中文正文 979 个汉字 + 35 处数字，按 220 字/分钟约 5 分 00 秒，"
        "加上 11 次换页与停顿，实际约 5 分 40 秒至 6 分 00 秒；"
        "英文版 734 词，按 140 词/分钟约 5 分 15 秒，加停顿约 5 分 45 秒。"
        "两版任选其一讲，不要都念。")


def main() -> int:
    document = Document(PATH)
    changed = 0
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("口播时长测算"):
            runs = paragraph.runs
            if runs:
                runs[0].text = TEXT
                for run in list(runs)[1:]:
                    run.text = ""
            changed += 1
            break
    print(f"lines rewritten: {changed}")
    document.save(PATH)
    print(f"saved: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
