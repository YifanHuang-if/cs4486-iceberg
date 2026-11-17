"""Undo a UTF-8 -> GBK -> UTF-8 round trip introduced by Get-Content/Set-Content."""
from __future__ import annotations

import sys
from pathlib import Path

path = Path(r"C:\Users\16018\cs4486-iceberg\work\build_iteration_docx.py")
raw = path.read_bytes()
text = raw.decode("utf-8", errors="replace")
print("marker present:", "\u951d" in text or "锝" in text)
for encoding in ("gbk", "gb18030", "cp936", "latin-1"):
    try:
        recovered = text.encode(encoding, errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError) as error:
        print(f"{encoding}: failed ({type(error).__name__})")
        continue
    print(f"{encoding}: round trip ok, length {len(recovered)}")
    if "\u51b0\u5c71" in recovered and "def main" in recovered:
        backup = path.with_suffix(".py.recovered")
        backup.write_text(recovered, encoding="utf-8")
        print(f"  -> wrote {backup}")
        sys.exit(0)
print("no clean recovery")
sys.exit(1)
