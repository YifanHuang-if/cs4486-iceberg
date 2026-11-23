"""Fix the template leftovers and the layout problems found in the render."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

PATH = Path(r"C:\Users\16018\cs4486-iceberg\report\CS4486_Iceberg_Final_6min.pptx")

REPLACE_TEXT = {
    5: "CS4486 • PROJECT",
    8: "CS4486 Final Presentation",
}

SHORT_TEXT = {
    (0, 16): ["Angle-aware radar fusion."],
    (1, 13): ["Statoil / C-CORE", "Satellite radar", "Ice versus ships"],
    (1, 16): ["• Binary classification",
              "• Bands HH + HV, 75×75",
              "• Plus incidence angle",
              "• Output: iceberg probability"],
    (3, 20): ["HH + HV, 75×75"],
    (9, 5): ["Backup slides"],
}


def set_lines(shape, lines) -> None:
    frame = shape.text_frame
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


def main() -> int:
    presentation = Presentation(PATH)
    slides = list(presentation.slides)

    # 1. Template branding and footer, matched by text as well as by shape id,
    # because the closing slide uses a different id for the same footer.
    for slide in slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            text = shape.text_frame.text.strip()
            if "Final Presentation Template" in text or "Replace all gray" in text:
                set_lines(shape, ["CS4486 Final Presentation"])
            elif shape.shape_id == 5 and "TEMPLATE" in text.upper():
                set_lines(shape, ["CS4486 • PROJECT"])

    # 2. Shortened texts that overflowed.
    for (slide_index, shape_id), lines in SHORT_TEXT.items():
        for shape in slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id == shape_id:
                set_lines(shape, lines)

    # 3. Shrink the small section labels so they stop wrapping in any renderer.
    for slide_index in (2, 3, 4, 5):
        for shape in slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id in (
                    14, 16, 18, 25, 27, 29, 36, 38, 40, 21, 15, 19, 12, 23):
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(7.5)

    # 3b. Normalise the effect lines so the three cards match.
    for slide_index in (4, 5):
        for shape in slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id in (19, 30, 41):
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(11.0)

    # 4. Remove the template's "Best model" marker: its label and the arrow.
    for slide_index in (4, 5):
        for shape in list(slides[slide_index].shapes):
            text = shape.text_frame.text.strip() if shape.has_text_frame else ""
            if text == "Best model":
                shape._element.getparent().remove(shape._element)
                continue
            if text:
                continue
            if 8.3 < shape.left / 914400 < 9.8 and shape.height / 914400 < 0.6:
                shape._element.getparent().remove(shape._element)

    # 4b. Widen the three section labels so HYPOTHESIS / IMPLEMENTATION fit.
    for slide_index in (4, 5):
        for shape in slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id in (14, 16, 18, 25, 27, 29, 36, 38, 40):
                shape.width = shape.width + Inches(0.45)

    # 4c. One body size across all six cards.
    for slide_index in (4, 5):
        for shape in slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id in (15, 17, 26, 28, 37, 39):
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(11.0)

    # 5. Sequential page numbers. Only the small right-hand box qualifies: a wide
    # empty footer box also sits at the same height and must not receive a number.
    for number, slide in enumerate(slides, start=1):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            if abs(shape.top / 914400 - 7.07) > 0.05:
                continue
            if shape.width / 914400 > 1.0:
                current = shape.text_frame.text.strip()
                if current.isdigit():
                    set_lines(shape, [""])
                continue
            set_lines(shape, [f"{number:02d}"])

    # 5b. Title slide: the template's own wordmark.
    for shape in slides[0].shapes:
        if shape.has_text_frame and shape.text_frame.text.strip() == "Template":
            set_lines(shape, ["CS4486"])

    # 6. Blank the placeholder lines that pictures now cover.
    for slide_index, shape_id in ((1, 19), (3, 40)):
        for shape in slides[slide_index].shapes:
            if shape.has_text_frame and shape.shape_id == shape_id:
                set_lines(shape, [""])

    # 7. Move pictures clear of their labels.
    def clear_pictures():
        for slide_index in (1, 2, 3):
            slide = slides[slide_index]
            for shape in list(slide.shapes):
                if shape.shape_type == 13:  # picture
                    shape._element.getparent().remove(shape._element)

    clear_pictures()
    figures = Path(r"C:\Users\16018\cs4486-iceberg\results")

    def place(slide_index, path, left, top, width):
        slides[slide_index].shapes.add_picture(str(path), Inches(left), Inches(top),
                                               width=Inches(width))

    place(1, figures / "eda/representative_images.png", 4.62, 2.10, 4.10)
    place(2, Path(r"C:\Users\16018\cs4486-iceberg\report\fig_eda_panel.png"), 1.02, 1.92, 4.85)
    place(3, figures / "baseline_v1/learning_curves.png", 8.28, 4.32, 3.95)

    # 8. Rebuild the results chart without the outlier first point.
    for shape in list(slides[6].shapes):
        if shape.has_chart:
            shape._element.getparent().remove(shape._element)

    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    data = CategoryChartData()
    data.categories = ["v6", "v7", "v8", "v11", "v13", "v19"]
    data.add_series("Private Log Loss",
                    (0.09106, 0.09027, 0.08944, 0.08784, 0.08752, 0.08708))
    frame = slides[6].shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS,
                                       Inches(0.98), Inches(4.34),
                                       Inches(7.0), Inches(2.1), data)
    chart = frame.chart
    chart.has_title = False
    chart.has_legend = False
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(8.5)
    plot.data_labels.number_format = "0.00000"
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = 2  # above
    chart.value_axis.minimum_scale = 0.0870
    chart.value_axis.maximum_scale = 0.0915
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.tick_labels.font.size = Pt(8)
    chart.value_axis.tick_labels.number_format = "0.001"
    chart.value_axis.tick_labels.number_format_is_linked = False
    chart.category_axis.tick_labels.font.size = Pt(9)

    # 9. Tidy the card C/F badge: restore the background box the earlier cleanup
    # removed, and drop the template's zero-height arrow at the right edge.
    import copy

    from pptx.util import Emu

    for slide_index in (4, 5):
        slide = slides[slide_index]
        by_id = {shape.shape_id: shape for shape in slide.shapes}
        for shape in list(slide.shapes):
            height = shape.height or 0
            if height < Emu(9144) and shape.left > Inches(10):
                shape._element.getparent().remove(shape._element)
        background = by_id.get(31)
        text_shape = by_id.get(43)
        if background is not None and text_shape is not None:
            clone = copy.deepcopy(background._element)
            offset = int(Inches(3.67))
            for node in clone.iter():
                if node.tag.endswith("}off"):
                    node.set("x", str(int(node.get("x")) + offset))
            text_shape._element.getparent().insert(
                list(text_shape._element.getparent()).index(text_shape._element), clone)

    presentation.save(PATH)
    print("layout fixes applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
