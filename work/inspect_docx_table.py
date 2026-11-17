"""Inspect the negative-results table in the iteration document."""
from __future__ import annotations

from docx import Document

PATH = r"C:\Users\16018\cs4486-iceberg\report\CS4486_冰山模型迭代流程总结.docx"

document = Document(PATH)
print("tables:", len(document.tables))
for index, table in enumerate(document.tables):
    header = [cell.text.strip() for cell in table.rows[0].cells]
    print(f"\n--- table {index}: {len(table.rows)} rows, header {header}")
    if header[:2] == ["尝试", "结果"]:
        for row in table.rows[1:]:
            print("   ", row.cells[0].text.strip()[:52], "|", row.cells[1].text.strip()[:52])
