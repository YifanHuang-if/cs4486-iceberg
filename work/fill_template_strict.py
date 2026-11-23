"""Fill the course template without touching a single formatting attribute.

Every replacement happens inside an existing shape: no slide is added, removed or
reordered, no shape is moved or resized, and no font, size or colour is changed.
Each line becomes its own paragraph cloned from the original first paragraph, so
paragraph spacing and run formatting are inherited exactly.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

SRC = Path(r"C:\Users\16018\cs4486-iceberg\report\Final_Presentation_Template_CS4486.pptx")
OUT = Path(r"C:\Users\16018\cs4486-iceberg\report\CS4486_Iceberg_Final.pptx")
FIG = Path(r"C:\Users\16018\cs4486-iceberg\results")
DOT = "\u00b7"          # middle dot
BULLET = "\u2022"       # bullet
ARROW = "\u2192"        # right arrow
MINUS = "\u2212"        # minus sign
DASH = "\u2013"         # en dash

TEXT: dict[int, dict[int, list[str]]] = {
    0: {
        6: ["Iceberg", "Classification"],
        7: ["Radar images plus angle evidence, fused by how much each is trusted."],
        10: [f"Group 12 {DOT} Huang Yifan {DOT}", f"Hachiware {DOT} Zheng"],
        12: [f"Kaggle {DOT} Statoil / C-CORE Iceberg Classifier"],
        14: ["24 November 2026"],
        16: ["Angle-aware radar fusion."],
        19: ["CS4486"],
    },
    1: {
        6: ["Project Roadmap"],
        43: ["Full pipeline: data understanding, validation design, baseline, improvements, results."],
        45: ["OUR PATH"],
        46: ["Five-fold baseline, then angle evidence and grouped fusion, then a calibrated submission."],
    },
    2: {
        13: ["Statoil / C-CORE Iceberg Classifier",
             "Satellite radar near offshore platforms",
             "Ice versus ships matters for safety"],
        16: [f"{BULLET} Binary classification",
             f"{BULLET} Bands HH + HV, 75x75",
             f"{BULLET} Plus incidence angle",
             f"{BULLET} Output: probability of iceberg"],
        22: ["Log Loss",
             "Lower is better",
             "Confident wrong answers cost more"],
        25: ["Reach the historical Top-5 private score of 0.08883.",
             "Stretch goal: below 0.083."],
        19: [""],
    },
    3: {
        13: [""],
        16: [f"{BULLET} Train 1,604 / test 8,424",
             f"{BULLET} Two bands, 75x75, plus angle",
             f"{BULLET} Per-fold normalization"],
        19: [f"{BULLET} Small, noisy radar data",
             f"{BULLET} 133 training angles missing",
             f"{BULLET} Test angles are complete"],
        22: ["Angle is evidence, not a feature.",
             "Same-angle rows usually share a label, so exact and near",
             "matches are trusted differently."],
        25: ["Class balance and angle spread"],
        28: ["Samples: ship versus iceberg"],
        31: ["Missing angles and band level"],
    },
    4: {
        13: ["A small CNN, trained from scratch."],
        16: ["Per-fold scaling"],
        20: ["HH + HV, 75x75"],
        24: ["3 conv blocks, 60,433 params"],
        28: ["BCE loss, Adam"],
        31: ["30 epochs, LR 0.001, batch 32, CPU"],
        34: ["OOF Log Loss 0.26063",
             "(historical 0.24638)"],
        37: ["Simple and reproducible.",
             "It isolates what each later change adds."],
        40: [""],
    },
    5: {
        13: ["A  Angle prior"],
        15: ["Exact angles often share a label,", "but coverage is low."],
        17: ["Use the exact-match group first,", "else a Gaussian kernel."],
        19: [f"OOF 0.25381 {ARROW} 0.12109"],
        21: [f"{MINUS}0.13272"],
        24: ["B  Grouped fusion"],
        26: ["Exact and near matches need", "different levels of trust."],
        28: ["Separate angle weights, fitted", "inside each training fold."],
        30: [f"OOF 0.11556 {ARROW} 0.10721"],
        32: [f"{MINUS}0.00835"],
        35: ["C  Student diversity"],
        37: ["One CNN repeats its own", "mistakes; unlabelled data is unused."],
        39: ["Soft-label distillation, a shape", "MLP and a mixup student."],
        41: [f"Image OOF 0.1959 {ARROW} 0.1833"],
        43: [f"{MINUS}0.0126"],
        45: ["Angle evidence gave the largest gain. Later steps: tuned recipe 0.08784, "
             "soft labels 0.08752, angle calibration 0.08708."],
    },
    6: {
        13: ["0.08708"],
        14: ["Private Log Loss (version19)"],
        20: ["Reached Top-5 level.", "0.083 stretch not met."],
        21: ["Historical Top-5: 0.08883.", "We are 0.00175 lower."],
        24: ["62% lower private loss", "than our first submission."],
        27: [""],
        30: ["Nearby-angle rows carry 87% of the loss.",
             "Image and prior disagree on 231 rows: 51% of loss."],
    },
    7: {
        13: [f"{BULLET} Angle evidence gave context",
             f"{BULLET} Grouped fusion set the trust",
             f"{BULLET} Diverse students cut shared errors"],
        16: [f"{BULLET} Strong augmentation lost radar cues",
             f"{BULLET} Bigger ensembles added correlated errors",
             f"{BULLET} Fixed by tuning the training recipe"],
        19: ["Wrong rows cluster where no exact angle match exists."],
        22: ["Small data, a leaderboard-tuned calibration, and a weak image model."],
        24: ["Only 1,604 labels"],
        26: ["Leaderboard-tuned calibration"],
        28: ["Next: stronger SAR features"],
    },
    8: {
        13: ["AI tools helped write code, run experiments and draft slides. "
             "Design choices stayed with us."],
        15: ["ChatGPT / Deepseek"],
        17: ["Code, debugging, analysis"],
        19: ["Problem framing, validation design, model choice"],
        21: ["AI use is reported per task."],
        25: ["Fold seed 42; member seeds 71, 83, 97, 909, 1201, 1401, 2601, saved per checkpoint."],
        27: [f"Python 3.12 {DOT} PyTorch 2.15 (CPU) {DOT} NumPy 2.5 {DOT} 16 threads, no GPU"],
        29: ["AdamW, 20 epochs, batch 32, LR 3e-3 / 5e-3, dropout 0.4-0.5, 8-view TTA"],
        31: ["github.com/YifanHuang-if/cs4486-iceberg"],
        33: ["OOF Log Loss; nested fusion; test IDs checked"],
        35: ["None used. CNNs trained from scratch, 93,521 params."],
        37: ["No pretrained weights in the reported pipeline."],
    },
    9: {
        5: ["Backup materials"],
        8: ["github.com/YifanHuang-if/cs4486-iceberg"],
        10: [f"CS4486 {BULLET} Final Presentation"],
    },
}

PICTURES = [
    (2, FIG / "eda/representative_images.png", 4.60, 2.15, 4.10),
    (3, Path(r"C:\Users\16018\cs4486-iceberg\report\fig_eda_panel.png"), 1.05, 2.02, 4.10),
    (4, FIG / "baseline_v1/learning_curves.png", 8.28, 4.55, 3.95),
    (6, Path(r"C:\Users\16018\cs4486-iceberg\report\fig_progress.png"), 1.00, 4.32, 6.40),
]


def fill(shape, lines: list[str]) -> None:
    """Rewrite the text using cloned paragraphs so formatting is inherited."""
    frame = shape.text_frame
    original = list(frame.paragraphs)
    template = copy.deepcopy(original[0]._p)

    for paragraph in original:
        paragraph._p.getparent().remove(paragraph._p)

    body = frame._txBody
    for _ in lines:
        body.append(copy.deepcopy(template))

    for paragraph, line in zip(frame.paragraphs, lines):
        runs = paragraph.runs
        if not runs:
            runs = [paragraph.add_run()]
        runs[0].text = line
        for run in list(runs)[1:]:
            run._r.getparent().remove(run._r)


def main() -> int:
    shutil.copy2(SRC, OUT)
    presentation = Presentation(OUT)

    for slide in presentation.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            text = shape.text_frame.text.strip()
            if text == "FINAL PRESENTATION TEMPLATE":
                fill(shape, [f"CS4486 {BULLET} PROJECT"])
            elif text.startswith(f"CS4486{BULLET} Project {BULLET} Replace all gray"):
                fill(shape, ["CS4486 Final Presentation"])

    for slide_index, mapping in TEXT.items():
        for shape in presentation.slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id in mapping:
                fill(shape, mapping[shape.shape_id])

    for slide_index, path, left, top, width in PICTURES:
        presentation.slides[slide_index].shapes.add_picture(
            str(path), Inches(left), Inches(top), width=Inches(width))

    presentation.save(OUT)
    print(f"saved {OUT} | slides={len(presentation.slides)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
