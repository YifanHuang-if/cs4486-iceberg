"""Build the final deck from the course template.

Order matters: text is filled on the original indices first, then the improvements
slide is duplicated, the optional roadmap slide is dropped, and finally the
duplicate receives the D/E/F cards. Pictures are placed last, using the final
indices.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Inches, Pt

TEMPLATE = Path(r"D:\Desktop\3semesterA\CS4886\projet\Final_Presentation_Template_CS4486.pptx")
OUT_DIR = Path(r"C:\Users\16018\cs4486-iceberg\report")
OUT = OUT_DIR / "CS4486_Iceberg_Final_6min.pptx"
FIG = Path(r"C:\Users\16018\cs4486-iceberg\results")

SLIDE_TEXT: dict[int, dict[int, list[str]]] = {
    0: {  # title
        10: ["Group 12  ·  Huang Yifan, Hachiware, Zheng"],
        12: ["Kaggle  ·  Statoil / C-CORE Iceberg Classifier"],
        14: ["24 November 2026"],
        16: ["Radar images plus angle evidence", "to tell ships from icebergs."],
        6: ["Iceberg Classification"],
        7: ["Fusing radar images with angle evidence, weighted by trust."],
    },
    2: {  # 1. task
        13: ["Statoil / C-CORE Iceberg Classifier",
             "Satellite radar near offshore platforms",
             "Ice versus ships matters for safety"],
        16: ["• Binary classification",
             "• Two radar bands (HH, HV), 75 × 75",
             "• Plus incidence angle",
             "• Output: probability of iceberg"],
        22: ["Log Loss",
             "Lower is better",
             "Confident wrong answers cost more"],
        25: ["Beat the historical Top-5 private score of 0.08883.",
             "Stretch goal: below 0.083."],
    },
    3: {  # 2. data
        16: ["• Train 1,604 / test 8,424",
             "• Two bands, 75 × 75, plus angle",
             "• Per-fold normalization"],
        19: ["• Small, noisy radar data",
             "• 133 training angles missing",
             "• Test angles are complete"],
        22: ["Angle is evidence, not a feature.",
             "Same-angle rows usually share a label.",
             "Exact and near matches need different trust."],
        13: [""],
        25: ["Class balance and angle spread"],
        28: ["Missing angles and band level"],
        31: ["No exact duplicates found"],
    },
    4: {  # 3. baseline
        13: ["A small CNN, trained from scratch."],
        16: ["Per-fold scaling"],
        20: ["HH + HV, 75 × 75"],
        24: ["3 conv blocks, 60,433 params"],
        28: ["BCE loss, Adam"],
        31: ["30 epochs, LR 0.001, batch 32, CPU"],
        34: ["OOF Log Loss 0.26063",
             "(historical reference 0.24638)"],
        37: ["Simple and reproducible.",
             "It isolates what each later change adds."],
        40: [""],
    },
    5: {  # 4. improvements A/B/C
        13: ["A  Angle prior"],
        15: ["Same angle, same label.", "Exact matches are rare."],
        17: ["Use the exact group first.", "Else a Gaussian kernel."],
        19: ["OOF 0.25381 → 0.12109"],
        21: ["−0.13272"],
        24: ["B  Grouped fusion"],
        26: ["Exact and near matches", "need different trust."],
        28: ["Separate weights for exact", "and near rows. Fit per fold."],
        30: ["OOF 0.11556 → 0.10721"],
        32: ["−0.00835"],
        35: ["C  Student diversity"],
        37: ["One CNN repeats", "its own mistakes."],
        39: ["Distil soft labels. Add a", "shape MLP and a mixup student."],
        41: ["Image OOF 0.1959 → 0.1833"],
        43: ["−0.0126"],
        45: ["Angle evidence gave the largest gain. Grouped fusion set how much to trust it."],
    },
    6: {  # 5. results
        13: ["0.08708"],
        14: ["Private Log Loss (version19)"],
        21: ["Historical Top-5: 0.08883.", "We are 0.00175 lower."],
        20: ["Reached Top-5 level.", "0.083 stretch not met."],
        24: ["62% lower private loss", "than our first submission."],
        27: [""],
        30: ["Nearby-angle rows carry 87% of the loss.",
             "Image and prior disagree on 231 rows: 51% of loss."],
    },
    7: {  # 6. discussion
        13: ["• Angle evidence gave context",
             "• Grouped fusion set the trust",
             "• Diverse students cut shared errors"],
        16: ["• Strong augmentation lost radar cues",
             "• Bigger ensembles added correlated errors",
             "• Fixed by tuning the recipe instead"],
        18: ["Where it still fails"],
        19: ["Wrong rows cluster where no exact angle match exists."],
        22: ["Small data, a leaderboard-tuned calibration, and a weak image model."],
        24: ["Only 1,604 labels"],
        26: ["Leaderboard-tuned calibration"],
        28: ["Next: stronger SAR features"],
    },
    8: {  # 7. AI and reproducibility
        13: ["AI tools helped write code, run experiments and draft slides. "
             "Design choices stayed with us."],
        15: ["ChatGPT / Deepseek"],
        17: ["Code, debugging, analysis"],
        19: ["Problem framing, validation design, model choice"],
        21: ["AI use is reported per task."],
        25: ["Fold seed 42; member seeds 71, 83, 97, 909, 1201, 1401, 2601, saved per checkpoint."],
        27: ["Python 3.12 · PyTorch 2.15 (CPU) · NumPy 2.5 · 16 threads, no GPU"],
        29: ["AdamW, 20 epochs, batch 32, LR 3e-3 / 5e-3, dropout 0.4–0.5, 8-view TTA"],
        31: ["github.com/YifanHuang-if/cs4486-iceberg"],
        33: ["OOF Log Loss; nested fusion; test IDs checked"],
        35: ["None used. CNNs trained from scratch, 93,521 params."],
        37: ["No pretrained weights in the reported pipeline."],
    },
    9: {  # Q&A
        6: ["• Ablation details", "• Fold-by-fold results", "• Full version history"],
        8: ["github.com/YifanHuang-if/cs4486-iceberg"],
    },
}

SECOND_IMPROVEMENTS = {
    13: ["D  Training recipe"],
    15: ["Tuning beats capacity."],
    17: ["AdamW, batch 32, LR 0.003, decay 0.01."],
    19: ["OOF 0.10194 → 0.0998"],
    21: ["v11 · 0.08784"],
    24: ["E  Broader soft labels"],
    26: ["A strict cutoff wastes data."],
    28: ["Threshold 0.03 → 0.25, 6,260 rows, 20 epochs."],
    30: ["OOF 0.09937 → 0.09829"],
    32: ["v13 · 0.08752"],
    35: ["F  Angle calibration"],
    37: ["CV and leaderboard disagree."],
    39: ["Scale both angle weights by 0.90."],
    41: ["Private −0.00044, CV +0.00047"],
    43: ["v19 · 0.08708"],
    45: ["Gains are per step, not additive. version19 is a leaderboard-tuned calibration."],
}


def set_lines(shape, lines) -> None:
    frame = shape.text_frame
    if not lines or (len(lines) == 1 and not lines[0]):
        frame.text = ""
        return
    first = frame.paragraphs[0]
    if not first.runs:
        first.add_run()
    template = first.runs[0]
    size, bold, name = template.font.size, template.font.bold, template.font.name
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


def duplicate_slide(presentation, index):
    source = presentation.slides[index]
    new = presentation.slides.add_slide(source.slide_layout)
    for shape in list(new.shapes):
        shape._element.getparent().remove(shape._element)
    for shape in source.shapes:
        new.shapes._spTree.append(copy.deepcopy(shape._element))
    return new


def move_slide(presentation, old_index, new_index):
    ids = presentation.slides._sldIdLst
    element = list(ids)[old_index]
    ids.remove(element)
    ids.insert(new_index, element)


def drop_slide(presentation, index):
    from pptx.oxml.ns import qn

    ids = presentation.slides._sldIdLst
    element = list(ids)[index]
    r_id = element.get(qn("r:id"))
    if r_id:
        presentation.part.drop_rel(r_id)
    ids.remove(element)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE, OUT)
    presentation = Presentation(OUT)

    # 1. Fill text on the original layout.
    for slide_index, mapping in SLIDE_TEXT.items():
        for shape in presentation.slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id in mapping:
                set_lines(shape, mapping[shape.shape_id])

    # 2. Duplicate the improvements slide, then drop the optional roadmap.
    duplicate_slide(presentation, 5)
    move_slide(presentation, len(presentation.slides) - 1, 6)
    drop_slide(presentation, 1)

    # 3. The duplicate now sits at index 5; give it the D/E/F cards.
    for shape in presentation.slides[5].shapes:
        if shape.has_text_frame and shape.shape_id in SECOND_IMPROVEMENTS:
            set_lines(shape, SECOND_IMPROVEMENTS[shape.shape_id])

    # 4. Pictures, using the final indices.
    def place(slide_index, path, left, top, width):
        presentation.slides[slide_index].shapes.add_picture(
            str(path), Inches(left), Inches(top), width=Inches(width))

    place(1, FIG / "eda/representative_images.png", 4.42, 1.80, 4.55)
    place(2, FIG / "eda/distributions.png", 0.94, 1.70, 5.45)
    place(2, FIG / "eda/missingness_and_band2.png", 0.94, 5.92, 3.5)
    place(3, FIG / "baseline_v1/learning_curves.png", 8.20, 3.95, 4.15)

    # 5. Native progress chart on the results slide.
    chart_data = CategoryChartData()
    chart_data.categories = ["v1", "v6", "v7", "v8", "v11", "v13", "v19"]
    chart_data.add_series("Private Log Loss",
                          (0.23022, 0.09106, 0.09027, 0.08944, 0.08784, 0.08752, 0.08708))
    frame = presentation.slides[6].shapes.add_chart(
        XL_CHART_TYPE.LINE_MARKERS, Inches(0.94), Inches(4.30),
        Inches(7.3), Inches(2.25), chart_data)
    chart = frame.chart
    chart.has_legend = False
    chart.font.size = Pt(9)
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(8)
    plot.data_labels.number_format = "0.00000"
    plot.data_labels.number_format_is_linked = False
    chart.value_axis.minimum_scale = 0.080
    chart.value_axis.maximum_scale = 0.240
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.tick_labels.font.size = Pt(8)
    chart.category_axis.tick_labels.font.size = Pt(9)

    presentation.save(OUT)
    print(f"saved {OUT} with {len(presentation.slides)} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
