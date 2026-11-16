"""Build a label-stratified five-fold split whose folds never share an angle group.

A random KFold lets rows with the same ``inc_angle`` sit on both sides of the
split, which is exactly the leak that makes local scores optimistic. This
generator keeps every exact angle (and every missing-angle row) whole inside one
fold, while balancing the two label counts and the fold sizes.

Writes ``work/splits/folds_angle_grouped.csv`` in the same column layout as the
frozen random split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work" / "cache"
OUT = ROOT / "work" / "splits"


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="folds_angle_grouped.csv")
    args = parser.parse_args()

    train = np.load(CACHE / "train.npz", allow_pickle=False)
    ids, angles, labels = train["ids"], train["angles"], train["labels"].astype(int)

    group_of = np.empty(len(ids), dtype=object)
    counters = 0
    for index, angle in enumerate(angles):
        if not np.isfinite(angle):
            group_of[index] = f"na-{index}"
        else:
            group_of[index] = f"{angle:.6f}"

    groups: dict[str, list[int]] = {}
    for index, key in enumerate(group_of):
        groups.setdefault(key, []).append(index)

    rng = np.random.default_rng(args.seed)
    order = list(groups.values())
    rng.shuffle(order)
    order.sort(key=len, reverse=True)
    fold_sizes = np.zeros(args.folds)
    fold_ships = np.zeros(args.folds)
    fold_icebergs = np.zeros(args.folds)
    total_ships = int((labels == 0).sum())
    total_icebergs = int((labels == 1).sum())
    assignment = np.full(len(ids), -1, dtype=int)
    for members in order:
        ships = int((labels[members] == 0).sum())
        icebergs = len(members) - ships
        # Largest groups first, each into the currently lightest fold; the label
        # terms only break ties so both classes stay balanced.
        target_ship = total_ships / args.folds
        target_iceberg = total_icebergs / args.folds
        balance = (
            np.abs(fold_ships + ships - target_ship)
            + np.abs(fold_icebergs + icebergs - target_iceberg)
        )
        cost = fold_sizes * 1000.0 + balance
        chosen = int(np.argmin(cost))
        assignment[members] = chosen
        fold_sizes[chosen] += len(members)
        fold_ships[chosen] += ships
        fold_icebergs[chosen] += icebergs
    if (assignment < 0).any():
        raise ValueError("some rows were not assigned")

    rows = [[ids[i], int(assignment[i]), int(labels[i]), int(not np.isfinite(angles[i]))] for i in range(len(ids))]
    write_csv(OUT / args.name, ["id", "fold", "label", "angle_missing"], rows)

    overlapping = []
    for fold in range(args.folds):
        inside = set(group_of[assignment == fold])
        outside = set(group_of[assignment != fold])
        overlapping.extend(inside & outside)
    report = {
        "folds": args.folds, "seed": args.seed, "rows": int(len(ids)),
        "groups": int(len(groups)),
        "fold_sizes": fold_sizes.astype(int).tolist(),
        "fold_ships": fold_ships.astype(int).tolist(),
        "fold_icebergs": fold_icebergs.astype(int).tolist(),
        "groups_crossing_folds": len(overlapping),
        "path": str(OUT / args.name),
    }
    (OUT / f"{Path(args.name).stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
