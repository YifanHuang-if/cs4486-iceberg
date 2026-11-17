"""Point the document at the new version names everywhere except the trace column."""
from __future__ import annotations

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

# Longest first so that "v5_best_allrows" is not clipped by "v5_best".
MAPPING = [
    ("v5_mix6_pool50", "version9"),
    ("v5_mix6_poolne", "version10"),
    ("v5_best_allrows", "version14"),
    ("v5_best_c001", "version15"),
    ("v5_best_c005", "version16"),
    ("v5_mix4", "version6"),
    ("v5_mix5", "version7"),
    ("v5_mix6", "version8"),
    ("v5_hp8v2", "version12"),
    ("v5_hp8", "version11"),
    ("v5_best", "version13"),
    ("final_v1_fast", "version5"),
    ("final_fast_s0", "version2"),
    ("fast_s0", "version1"),
    ("v1_s0", "version3"),
    ("v1_fast", "version4"),
]

SCORES = [
    ("version5（final_v1_fast，M5 角度层 v2）", "0.11143", "0.12862"),
    ("version7（v5_mix5）", "0.09027", "0.10739"),
    ("version8（v5_mix6）", "0.08944", "0.10726"),
    ("version9（pool50 诊断）", "0.09029", "0.10630"),
    ("version10（poolne 诊断）", "0.08954", "0.10566"),
    ("version11（v5_hp8，超参调优）", "0.08784", "0.10746"),
    ("version12（v5_hp8v2）", "0.08767", "0.11005"),
    ("version13（v5_best，当前最佳）", "0.08752", "0.10899"),
    ("version14（allrows 诊断，与 13 完全相同）", "0.08752", "0.10899"),
]

COMMANDS = (
    "python -m src.v3.prepare --raw data/raw\n"
    "python work/build_teacher.py v5_teacher6 s101 s202 s404 v5_shape stu1 stu5\n"
    "python -m src.v3.train --tag hp_star --seed 909 --epochs 40 --dropout 0.4 --lr 0.003 "
    "--weight-decay 0.01 --batch 32 --pseudo-threshold 0.25 --teacher v5_teacher6 --pseudo-weight 0.25\n"
    "python -m src.v3.train --tag e40a --seed 1401 --epochs 40 --dropout 0.4 --lr 0.003 "
    "--weight-decay 0.01 --batch 32 --pseudo-threshold 0.25 --teacher v5_teacher6 --pseudo-weight 0.25\n"
    "python -m src.v3.shape_model --tag v5_shape\n"
    "python work/v5_fuse.py --tags v5_shape hp_star hp_b hp_d e40a pt25b stu5 --name fuse_cur\n"
    "python work/v5_predict.py --tags v5_shape hp_star hp_b hp_d pt25a pt25b stu5\n"
    "python work/v5_submit.py --tags v5_shape hp_star hp_b hp_d pt25a pt25b stu5 --name version13\n"
    "python work/validate_submission.py version13\n"
    "python work/verify_submission_provenance.py version13"
)


def rewrite_text(text: str) -> str:
    for old, new in MAPPING:
        text = text.replace(old, new)
    return text


def main() -> int:
    document = Document(PATH)
    version_table = document.tables[1]
    changed = 0

    def touch(paragraph):
        nonlocal changed
        original = paragraph.text
        if not original.strip():
            return
        updated = rewrite_text(original)
        if updated != original and paragraph.runs:
            paragraph.runs[0].text = updated
            for run in paragraph.runs[1:]:
                run.text = ""
            changed += 1

    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("python -m src.v3.prepare"):
            continue
        touch(paragraph)

    for table in document.tables:
        if table is version_table:
            continue
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    touch(paragraph)

    # Rebuild the score table from scratch so its labels and ordering are clean.
    scores = document.tables[7]
    while len(scores.rows) > len(SCORES) + 1:
        scores._tbl.remove(scores.rows[-1]._tr)
    while len(scores.rows) < len(SCORES) + 1:
        scores.add_row()
    for index, (name, private, public) in enumerate(SCORES, start=1):
        cells = scores.rows[index].cells
        for column, text in enumerate((name, private, public)):
            cells[column].text = ""
            run = cells[column].paragraphs[0].add_run(text)
            run.font.size = Pt(8)

    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("python -m src.v3.prepare"):
            for run in list(paragraph.runs)[1:]:
                run._element.getparent().remove(run._element)
            if paragraph.runs:
                paragraph.runs[0].text = COMMANDS
                paragraph.runs[0].font.size = Pt(8.5)
            changed += 1

    document.save(PATH)
    print(f"rewrote {changed} text blocks; score table now {len(scores.rows) - 1} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
