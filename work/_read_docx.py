import io
from docx import Document
d = Document(r"D:\WeChat Files\wxid_q0aexjay5l0022\FileStorage\File\2026-11\Iceberg_V2_Improvement_Summary.docx")
out = []
for p in d.paragraphs:
    t = p.text.strip()
    if t:
        out.append(f"[{p.style.name}] {t}")
out.append("\n===== TABLES =====")
for i, tb in enumerate(d.tables):
    out.append(f"--- table {i} ({len(tb.rows)}x{len(tb.columns)}) ---")
    for r in tb.rows:
        out.append(" | ".join(c.text.strip() for c in r.cells))
io.open(r"C:\Users\16018\cs4486-iceberg\work\_docx_dump.txt","w",encoding="utf-8").write("\n".join(out))
print("ok", sum(len(x) for x in out))
