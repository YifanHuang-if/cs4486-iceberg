"""Rewrite slides 5-6 with short lines that fit the existing boxes.

Each card has room for three lines under HYPOTHESIS, four under IMPLEMENTATION and
three under EFFECT at 13.5 pt, and a line must stay near 28 characters to avoid
wrapping. The text below respects those limits.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation

SOURCE = Path(r"C:\Users\16018\Downloads\CS4486_Iceberg_Final_6min.pptx")
TARGET_DIR = Path(r"C:\Users\16018\cs4486-iceberg\report")
TARGET = TARGET_DIR / "CS4486_Iceberg_Final_6min_v2.pptx"

SLIDE5 = {
    15: ["Same angle, same label.", "Exact matches are rare."],
    17: ["Use the exact group first.", "Else a Gaussian kernel.",
         "Fit on training folds only."],
    19: ["Nested OOF", "0.25381 → 0.12109"],
    21: ["OOF gain: −0.13272"],
    26: ["Exact and near matches", "need different trust."],
    28: ["Separate weights for exact", "and near rows.",
         "Fit in each training fold."],
    30: ["Nested OOF", "0.11556 → 0.10721"],
    45: ["Our contribution: fuse radar predictions with angle evidence, weighted by trust. "
         "Gains are per step and not additive."],
    37: ["One CNN repeats", "its own mistakes.", "Unlabelled data unused."],
    39: ["Distil soft labels.", "Add a shape MLP", "and a mixup student."],
    41: ["Distilled image OOF", "0.1959 → 0.1833"],
}

SLIDE6 = {
    15: ["Tuning the setup beats", "adding capacity."],
    17: ["AdamW, batch 32.", "LR 0.003, decay 0.01.", "Cosine schedule.",
         "Small 93k CNNs."],
    19: ["Tuned ensemble OOF", "0.10194 → 0.0998"],
    26: ["A strict cutoff throws", "away most unlabelled rows."],
    28: ["Threshold 0.03 → 0.25.", "About 6,260 pseudo rows.", "Weight 0.25.",
         "20 epochs."],
    30: ["Threshold OOF", "0.09937 → 0.09829"],
    37: ["CV and the leaderboard", "prefer different weights."],
    39: ["Scale both angle weights", "by 0.90 after fitting.",
         "Same CNNs, LB feedback."],
    41: ["Private −0.00044.", "CV +0.00047."],
    45: ["version19 is a leaderboard-tuned calibration, "
         "so its score is not an untouched test estimate."],
}


def set_lines(shape, lines) -> None:
    frame = shape.text_frame
    first = frame.paragraphs[0]
    template = first.runs[0]
    size = template.font.size
    bold = template.font.bold
    name = template.font.name
    try:
        color = template.font.color.rgb
    except Exception:
        color = None

    for paragraph in list(frame.paragraphs)[1:]:
        paragraph._p.getparent().remove(paragraph._p)
    for run in list(first.runs)[1:]:
        run._r.getparent().remove(run._r)
    first.runs[0].text = lines[0]

    for line in lines[1:]:
        paragraph = frame.add_paragraph()
        run = paragraph.add_run()
        run.text = line
        if size is not None:
            run.font.size = size
        if bold is not None:
            run.font.bold = bold
        if name:
            run.font.name = name
        if color is not None:
            run.font.color.rgb = color


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, TARGET)
    presentation = Presentation(TARGET)

    applied = 0
    for slide_index, mapping in ((4, SLIDE5), (5, SLIDE6)):
        for shape in presentation.slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id in mapping:
                set_lines(shape, mapping[shape.shape_id])
                applied += 1
    print(f"rewrote {applied} text boxes")

    presentation.save(TARGET)
    longest = max((len(line) for group in (SLIDE5, SLIDE6)
                   for lines in group.values() for line in lines))
    print(f"longest line: {longest} characters -> {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
