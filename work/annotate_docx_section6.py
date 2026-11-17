"""Add a scope note to the ensemble-size section of the iteration document."""
from __future__ import annotations

from docx import Document

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"
NOTE = ("（表中数字为超参调优之前的成员。调参后集成在 8 个成员附近饱和："
        "合理的成员组合落在 0.0994–0.1001，继续增加只会稀释。）")


def main() -> int:
    document = Document(PATH)
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("边际收益从第 4 个模型起跌破"):
            if paragraph.runs:
                paragraph.runs[-1].text = paragraph.runs[-1].text + NOTE
            else:
                paragraph.add_run(NOTE)
            document.save(PATH)
            print("note appended")
            print("new text tail:", paragraph.text[-80:])
            return 0
    print("target paragraph not found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
