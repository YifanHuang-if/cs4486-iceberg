"""Try to reverse the ANSI re-encoding that PowerShell applied to a UTF-8 script."""
from __future__ import annotations

from pathlib import Path

PATH = Path(r"C:\Users\16018\cs4486-iceberg\work\build_slide_companion.py")
raw = PATH.read_bytes()
text = raw.decode("utf-8", errors="replace")

for encoding in ("gbk", "gb18030", "cp936", "latin-1"):
    try:
        recovered = text.encode(encoding, errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError) as error:
        print(f"{encoding:10s} failed: {type(error).__name__}")
        continue
    ok = "\u00b7" in recovered or "CS4486 · PPT" in recovered
    print(f"{encoding:10s} ok, len={len(recovered)}, marker={ok}")
    if ok and "def main" in recovered:
        PATH.write_text(recovered, encoding="utf-8")
        print("recovered in place")
        break
else:
    print("no recovery")
