"""Replace the reproduce-command block with the current pipeline."""
from __future__ import annotations

from docx import Document
from docx.shared import Pt

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

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
    "python work/v5_submit.py --tags v5_shape hp_star hp_b hp_d pt25a pt25b stu5 --name v5_best\n"
    "python work/validate_submission.py v5_best\n"
    "python work/verify_submission_provenance.py v5_best"
)


def main() -> int:
    document = Document(PATH)
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("python -m src.v3.prepare"):
            for run in list(paragraph.runs)[1:]:
                run._element.getparent().remove(run._element)
            if paragraph.runs:
                paragraph.runs[0].text = COMMANDS
                paragraph.runs[0].font.name = "Consolas"
                paragraph.runs[0].font.size = Pt(8.5)
            else:
                run = paragraph.add_run(COMMANDS)
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
            document.save(PATH)
            print("replaced reproduce block")
            return 0
    print("block not found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
