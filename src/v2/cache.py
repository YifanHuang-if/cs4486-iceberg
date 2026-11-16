"""Build a compact float32 cache of the competition data.

Reads the official JSON arrays without loading a whole document, and writes
train/test image tensors plus ids, labels and incidence angles to ``work/cache``.
Only the source files are read; nothing is modified.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "work" / "cache"


def records(path: Path):
    """Yield one JSON object (bytes) at a time; the records contain no nested braces."""
    read = 1 << 22
    buf = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(read)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(b"}")
            buf = parts.pop()
            for part in parts:
                start = part.find(b"{")
                if start >= 0:
                    yield part[start:] + b"}"
        start = buf.find(b"{")
        if start >= 0:
            yield buf[start:] + b"}"


def band(piece: bytes, name: bytes) -> np.ndarray:
    key = b'"' + name + b'":['
    start = piece.find(key)
    if start < 0:
        start = piece.find(b'"' + name + b'": [')
        if start < 0:
            raise ValueError(f"missing {name!r}")
        start += len(b'"' + name + b'": [')
    else:
        start += len(key)
    end = piece.find(b"]", start)
    return np.fromstring(piece[start:end].decode("ascii"), sep=",", dtype=np.float32)


def scan(path: Path, labelled: bool):
    ids, labels, angles, missing, images = [], [], [], [], []
    started = time.time()
    for index, piece in enumerate(records(path), start=1):
        obj = json.loads(piece)
        images.append(np.stack([band(piece, b"band_1"), band(piece, b"band_2")]))
        ids.append(obj["id"])
        raw = obj.get("inc_angle")
        gap = raw is None or raw == "na"
        missing.append(gap)
        angles.append(np.nan if gap else float(raw))
        if labelled:
            labels.append(int(obj["is_iceberg"]))
        if index % 2000 == 0:
            print(f"  {path.name}: {index} rows ({time.time() - started:.0f}s)", flush=True)
    x = np.stack(images).astype(np.float32).reshape(-1, 2, 75, 75)
    return {
        "ids": np.asarray(ids),
        "images": x,
        "angles": np.asarray(angles, dtype=np.float64),
        "angle_missing": np.asarray(missing, dtype=bool),
        "labels": np.asarray(labels, dtype=np.float32) if labelled else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split, labelled in (("train", True), ("test", False)):
        target = OUT / f"{split}.npz"
        if target.exists() and not args.force:
            print(f"{target} exists; skipping")
            continue
        source = RAW / f"{split}.json"
        print(f"reading {source.name} ({source.stat().st_size / 1e6:.0f} MB)", flush=True)
        data = scan(source, labelled)
        payload = {k: v for k, v in data.items() if v is not None}
        temporary = target.with_suffix(".tmp.npz")
        np.savez(temporary, **payload)
        temporary.replace(target)
        print(f"wrote {target} shape={data['images'].shape}", flush=True)
    (OUT / "cache_info.json").write_text(
        json.dumps(
            {
                "train_rows": int(np.load(OUT / "train.npz")["labels"].size),
                "test_rows": int(np.load(OUT / "test.npz")["ids"].size),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
