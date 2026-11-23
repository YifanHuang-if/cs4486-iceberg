"""Audit the deck's numbers against the verified values."""
from __future__ import annotations

import re

from pptx import Presentation

PATH = r"C:\Users\16018\cs4486-iceberg\report\Group12_Project3.pptx"

VERIFIED = {
    "near_rows": 709,
    "near_share": "89.3%",
    "disagree_rows": 181,
    "disagree_share": "49.0%",
    "image_acc": "60.2%",
    "prior_acc": "39.8%",
    "exact_purity": "98.5%",
    "near_agree": "81.9%",
    "cv": "0.098287",
    "private_best": "0.08708",
    "public_best": "0.10365",
}


def main() -> int:
    presentation = Presentation(PATH)
    print(f"slides: {len(presentation.slides)}")
    numbers = []
    for index, slide in enumerate(presentation.slides, start=1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text
            elif getattr(shape, "has_table", False) and shape.has_table:
                text = "\n".join(
                    " | ".join(cell.text.strip() for cell in row.cells)
                    for row in shape.table.rows)
            else:
                continue
            for match in re.findall(r"\d+\.?\d*%|\d+\.\d{3,5}|\b\d{2,4}\b", text):
                numbers.append((index, match, text.strip().replace("\n", " / ")[:70]))

    suspicious = [
        ("87.2%", "应为 89.3%（near 组损失占比）"),
        ("231", "应为 181（图像与先验判相反的行数）"),
        ("50.7%", "应为 49.0%"),
        ("98.6%", "应为 98.5%（同角度一致率）"),
        ("84%", "应为 81.9%（仅邻近角度一致率）"),
        ("0.00061", "应为 0.00175"),
    ]
    print("\n--- 可疑数字 ---")
    for needle, note in suspicious:
        hits = [(i, t) for i, m, t in numbers if m == needle or needle in m or needle in t]
        if hits:
            for index, text in hits[:2]:
                print(f"  slide {index}: {needle}  -> {note}")
        else:
            print(f"  slide ?: {needle} not found (already fixed?)")

    print("\n--- 关键数字出现位置 ---")
    for key, value in VERIFIED.items():
        hits = sorted({i for i, m, t in numbers if m == value or value in t})
        print(f"  {key:16s} {value:9s} slides={hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
