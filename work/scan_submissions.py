"""Find every candidate submission CSV and validate it against the sample file.

A valid Kaggle upload for this competition needs a header plus 8,424 rows with
columns id,is_iceberg, ids matching sample_submission.csv, and finite
probabilities inside [0, 1]. Files are ranked by the documented out-of-fold log
loss of the pipeline that produced them, when that is known.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = [ROOT / "work/submissions", ROOT / "outputs", ROOT / "results"]
EXTRA = [Path(r"D:\Desktop\cs4486-iceberg-main (1)\cs4486-iceberg\outputs")]

KNOWN = {
    "v5_mix4": ("v5 ensemble + nested group stacking (4 models)", 0.10307),
    "final_v1_fast": ("final_v1_fast", 0.11897),
    "final_fast_s0": ("final_fast_s0", 0.12109),
    "v1_fast": ("v1_fast", 0.23955),
    "v1_s0": ("v1_s0", 0.24711),
    "fast_s0": ("fast_s0", 0.25006),
    "v3_diverse_ensemble": ("v3_diverse_ensemble", 0.10345),
    "v3_diverse_ensemble_pooled": ("v3_diverse_ensemble (pooled)", 0.10361),
}


def load_sample():
    path = ROOT / "data/raw/sample_submission.csv"
    with io.open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], [row[0] for row in rows[1:]]


def inspect(path, expected_ids):
    issues = []
    with io.open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {"valid": False, "issues": ["empty"]}
    header, body = rows[0], rows[1:]
    if [c.strip().lower() for c in header] != ["id", "is_iceberg"]:
        issues.append(f"header={header}")
    if len(body) != 8424:
        issues.append(f"rows={len(body)}")
    ids = [row[0].strip() for row in body if row]
    if ids != expected_ids:
        if set(ids) == set(expected_ids):
            issues.append("ids present but order differs")
        else:
            issues.append("id set mismatch")
    values = []
    for row in body:
        try:
            values.append(float(row[1]))
        except (IndexError, ValueError):
            issues.append("non-numeric probability")
            break
    if values:
        if any(not (0.0 <= v <= 1.0) for v in values):
            issues.append("probability outside [0,1]")
        stats = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    else:
        stats = {}
    return {"valid": not issues, "issues": issues, "rows": len(body), "bytes": path.stat().st_size,
            **{k: round(v, 6) for k, v in stats.items()}}


def main() -> int:
    _, expected = load_sample()
    candidates = []
    for base in SEARCH + EXTRA:
        if base.exists():
            candidates.extend(base.rglob("*.csv"))
    found = []
    for path in sorted(set(candidates)):
        if path.name.endswith("_oof.csv") or path.name.startswith("submission_manifest"):
            continue
        try:
            with io.open(path, encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
        except (UnicodeDecodeError, StopIteration):
            continue
        if [c.strip().lower() for c in header] != ["id", "is_iceberg"]:
            continue
        report = inspect(path, expected)
        label, oof = KNOWN.get(path.stem, (path.stem, None))
        found.append({"path": str(path), "label": label, "oof": oof, **report})

    valid = [row for row in found if row["valid"]]
    valid.sort(key=lambda row: (row["oof"] is None, row["oof"] if row["oof"] is not None else 9))
    print(f"{len(found)} candidate files, {len(valid)} valid\n")
    for row in valid:
        print(f"OK   oof={row['oof']}  rows={row['rows']}  mean={row.get('mean')}  {row['path']}")
    invalid = [row for row in found if not row["valid"]]
    if invalid:
        print("\ninvalid:")
        for row in invalid:
            print(f"BAD  {row['issues']}  {row['path']}")
    out = ROOT / "work/submissions/_scan.json"
    out.write_text(json.dumps(found, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
