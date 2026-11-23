"""Close the two template gaps on the discussion slide, in place."""
from __future__ import annotations

import copy
from pathlib import Path

from pptx import Presentation

TARGET = Path(r"C:\Users\16018\Downloads\CS4486_Iceberg_Final_6min.pptx")
FALLBACK = Path(r"C:\Users\16018\cs4486-iceberg\report\CS4486_Iceberg_Final_6min.pptx")

EDITS = {
    13: ["Angle evidence supplied context: OOF 0.25381 to 0.12109.",
         "Grouped fusion adjusted its trust: 0.11556 to 0.10721.",
         "Diverse students cut shared errors: image OOF 0.1959 to 0.1833."],
    16: ["Stronger augmentation and larger ensembles often hurt.",
         "Cause: lost radar cues and correlated errors.",
         "Next time: change one factor at a time and keep augmentation matched to training."],
    22: ["Two limits dominate: the image model struggles where no exact angle match exists, "
         "and every score was partly chosen by tuning."],
    26: ["Scores partly reflect tuning"],
}


def fill(shape, lines) -> None:
    frame = shape.text_frame
    original = list(frame.paragraphs)
    template = copy.deepcopy(original[0]._p)
    for paragraph in original:
        paragraph._p.getparent().remove(paragraph._p)
    for _ in lines:
        frame._txBody.append(copy.deepcopy(template))
    for paragraph, line in zip(frame.paragraphs, lines):
        runs = paragraph.runs
        if not runs:
            runs = [paragraph.add_run()]
        runs[0].text = line
        for run in list(runs)[1:]:
            run._r.getparent().remove(run._r)


def main() -> int:
    presentation = Presentation(TARGET)
    count = 0
    for shape in presentation.slides[7].shapes:
        if shape.has_text_frame and shape.shape_id in EDITS:
            fill(shape, EDITS[shape.shape_id])
            count += 1
    print(f"boxes changed: {count}")
    try:
        presentation.save(TARGET)
        print(f"saved: {TARGET}")
    except PermissionError:
        presentation.save(FALLBACK)
        print(f"locked; saved copy: {FALLBACK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
