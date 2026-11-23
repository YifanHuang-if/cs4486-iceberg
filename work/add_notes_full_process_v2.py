"""Insert the term notes again, matching step headings exactly.

The previous run matched by plain prefix, so "步骤 1" also matched "步骤 12",
"步骤 20" and so on. Matching now requires a following space or end of string,
and the longest key wins.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

from work.add_notes_full_process import NOTES

PATH = Path(r"D:\Desktop\CS4486_Project_Full_Process.docx")
BACKUP = Path(r"D:\Desktop\CS4486_Project_Full_Process.backup.docx")

GREY = RGBColor(0x40, 0x40, 0x40)
ACCENT = RGBColor(0xC0, 0x00, 0x00)

KEYS = sorted(NOTES, key=len, reverse=True)
PATTERNS = [(key, re.compile(rf"^{re.escape(key)}(?=[\s\u3000]|$)")) for key in KEYS]


def match_key(text: str) -> str | None:
    for key, pattern in PATTERNS:
        if pattern.match(text):
            return key
    return None


def insert_notes(anchor, entries) -> None:
    def make(runs, indent=12):
        paragraph = anchor.insert_paragraph_before()
        for text, bold, colour, size in runs:
            run = paragraph.add_run(text)
            run.bold = bold
            run.font.size = Pt(size)
            run.font.color.rgb = colour
        paragraph.paragraph_format.left_indent = Pt(indent)
        paragraph.paragraph_format.space_after = Pt(2)

    make([("术语说明（通俗版）", True, ACCENT, 9.5)], indent=0)
    for term, explanation in entries:
        make([(f"{term}：", True, GREY, 9), (explanation, False, GREY, 9)])


def main() -> int:
    shutil.copy2(BACKUP, PATH)
    document = Document(PATH)

    pending = None
    done = 0
    for paragraph in list(document.paragraphs):
        if not paragraph.style.name.startswith("Heading"):
            continue
        text = paragraph.text.strip()
        if pending is not None:
            insert_notes(paragraph, NOTES[pending])
            done += 1
            pending = None
        key = match_key(text)
        if key is not None:
            pending = key

    print(f"blocks inserted: {done}/{len(NOTES)}")
    document.save(PATH)
    print(f"saved: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
