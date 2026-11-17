"""Rename every submission to versionN in chronological order.

Each CSV keeps its manifest and any provenance reports, renamed to match. A
mapping file records the old name so the leaderboard history stays traceable.
The originals are not deleted: they are moved, and the script refuses to
overwrite an existing target.
"""
from __future__ import annotations

import json
from pathlib import Path

SUB = Path(r"C:\Users\16018\cs4486-iceberg\work\submissions")

# Chronological order, which is the order the files were written.
ORDER = [
    "fast_s0", "final_fast_s0", "v1_s0", "v1_fast", "final_v1_fast",
    "v5_mix4", "v5_mix5", "v5_mix6", "v5_mix6_pool50", "v5_mix6_poolne",
    "v5_hp8", "v5_hp8v2", "v5_best", "v5_best_allrows",
    "v5_best_c001", "v5_best_c005",
]

SUFFIXES = [".csv", ".json", ".verify.json", "_provenance.json"]


def main() -> int:
    mapping = {}
    for index, old in enumerate(ORDER, start=1):
        new = f"version{index}"
        mapping[new] = old
        for suffix in SUFFIXES:
            source = SUB / f"{old}{suffix}"
            if not source.exists():
                continue
            target = SUB / f"{new}{suffix}"
            if target.exists():
                raise SystemExit(f"refusing to overwrite {target}")
            source.rename(target)
        print(f"{new:10s} <- {old}")

    (SUB / "_version_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote mapping for {len(mapping)} versions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
