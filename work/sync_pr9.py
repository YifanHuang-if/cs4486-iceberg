"""Copy selected files out of a git branch of another working copy, byte for byte.

Avoids shell pipelines: git writes straight into a file handle, then every file
is verified against the blob hash recorded by git so nothing is silently mangled.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SOURCE = Path(r"D:\Desktop\cs4486-iceberg-main (1)\cs4486-iceberg")
BRANCH = "pr9"
DEST = Path(r"C:\Users\16018\cs4486-iceberg")
PREFIXES = ("src/v3/", "src/v4/", "scripts/")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(SOURCE), *args], capture_output=True, text=True, check=True).stdout


def main() -> int:
    listing = [line for line in git("ls-tree", "-r", "--name-only", BRANCH).splitlines()
               if line.startswith(PREFIXES)]
    copied, failed = 0, []
    for relative in listing:
        target = DEST / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            subprocess.run(["git", "-C", str(SOURCE), "show", f"{BRANCH}:{relative}"],
                           stdout=handle, check=True)
        expected = git("rev-parse", f"{BRANCH}:{relative}").strip()
        actual = subprocess.run(["git", "hash-object", "--no-filters", str(target)],
                                capture_output=True, text=True, check=True).stdout.strip()
        if expected != actual:
            failed.append(relative)
        copied += 1
    print(f"copied {copied} files, {len(failed)} hash mismatches")
    for name in failed:
        print("  MISMATCH", name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
