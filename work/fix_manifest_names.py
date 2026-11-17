"""Point the renamed manifests at their new file names.

The manifests record a "name" field that the provenance tool uses to locate the
CSV. After renaming both, that field has to follow, otherwise verification of the
renamed files fails for a bookkeeping reason rather than a real one.
"""
from __future__ import annotations

import json
from pathlib import Path

SUB = Path(r"C:\Users\16018\cs4486-iceberg\work\submissions")


def main() -> int:
    mapping = json.loads((SUB / "_version_mapping.json").read_text(encoding="utf-8"))
    updated = 0
    for new, old in mapping.items():
        manifest = SUB / f"{new}.json"
        if not manifest.exists():
            continue
        blob = json.loads(manifest.read_text(encoding="utf-8"))
        if blob.get("name") == old:
            blob["name"] = new
            blob["original_name"] = old
            manifest.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
            updated += 1
        provenance = SUB / f"{new}_provenance.json"
        if provenance.exists():
            report = json.loads(provenance.read_text(encoding="utf-8"))
            if report.get("submission") == old:
                report["submission"] = new
                provenance.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"updated {updated} manifests")
    for path in sorted(SUB.glob("version*.json")):
        if path.name.endswith("_provenance.json"):
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        print(f"  {path.stem:10s} name={blob.get('name')} original={blob.get('original_name', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
