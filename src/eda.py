"""Inspect the competition files without loading the full test JSON into RAM.

Run from the repository root: python -m src.eda --data-dir data/raw
This module does not fit preprocessing, create folds, or train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path

import ijson
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_angle(value, sample_id: str) -> float:
    if value is None or (isinstance(value, str) and value.strip().lower() == "na"):
        return float("nan")
    try:
        angle = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id}: invalid inc_angle {value!r}") from exc
    if not np.isfinite(angle):
        raise ValueError(f"{sample_id}: non-finite inc_angle")
    return angle


def scan_json(path: Path, labeled: bool, examples_per_class: int = 3) -> dict:
    """Stream records; retain metadata and a few illustrative training images."""
    required = {"id", "band_1", "band_2", "inc_angle"}
    if labeled:
        required.add("is_iceberg")
    rows, examples = [], []
    seen_ids, columns = set(), set()
    example_counts = Counter()
    minima, maxima = [float("inf")] * 2, [float("-inf")] * 2
    with path.open("rb") as stream:
        # The competition file is a top-level JSON array.
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        if first != b"[":
            raise ValueError(f"{path.name}: expected a JSON array")
        stream.seek(0)
        for row in ijson.items(stream, "item", use_float=True):
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"{path.name}: missing required fields")
            columns.update(row)
            sample_id = row["id"]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path.name}: invalid sample ID")
            if sample_id in seen_ids:
                raise ValueError(f"{path.name}: duplicate ID {sample_id}")
            seen_ids.add(sample_id)
            label = row.get("is_iceberg")
            if labeled and label not in (0, 1):
                raise ValueError(f"{sample_id}: label must be 0 or 1")
            bands = []
            for j, name in enumerate(("band_1", "band_2")):
                values = np.asarray(row[name], dtype="<f8")
                if values.shape != (5625,):
                    raise ValueError(f"{sample_id}: {name} must contain 5625 values")
                if not np.isfinite(values).all():
                    raise ValueError(f"{sample_id}: {name} contains non-finite values")
                minima[j] = min(minima[j], float(values.min()))
                maxima[j] = max(maxima[j], float(values.max()))
                bands.append(values)
            fingerprint = hashlib.sha256(bands[0].tobytes() + bands[1].tobytes()).hexdigest()
            meta = {"id": sample_id, "inc_angle": parse_angle(row["inc_angle"], sample_id),
                    "image_sha256": fingerprint}
            if labeled:
                meta["is_iceberg"] = int(label)
            rows.append(meta)
            if labeled and example_counts[label] < examples_per_class:
                examples.append({"id": sample_id, "label": int(label),
                                 "image": np.stack(bands).reshape(2, 75, 75)})
                example_counts[label] += 1
    if not rows:
        raise ValueError(f"{path.name}: no records found")
    metadata = pd.DataFrame(rows)
    known = metadata["inc_angle"].dropna()
    sizes = known.value_counts()
    repeated = sizes[sizes > 1]
    summary = {
        "samples": len(metadata), "columns": sorted(columns), "duplicate_ids": 0,
        "band_shape": [75, 75], "all_band_values_finite": True,
        "band_min": minima, "band_max": maxima,
        "missing_angles": int(metadata["inc_angle"].isna().sum()),
        "known_angle_min": float(known.min()) if len(known) else None,
        "known_angle_max": float(known.max()) if len(known) else None,
        "unique_known_angles": len(sizes), "repeated_angle_groups": len(repeated),
        "samples_in_repeated_angle_groups": int(repeated.sum()),
        "exact_duplicate_image_excess_rows": int(metadata["image_sha256"].duplicated().sum()),
    }
    if labeled:
        summary["label_counts"] = {str(k): int(v) for k, v in
                                   metadata["is_iceberg"].value_counts().sort_index().items()}
        summary["missing_angle_label_counts"] = {str(k): int(v) for k, v in
            metadata.loc[metadata["inc_angle"].isna(), "is_iceberg"].value_counts().sort_index().items()}
        grouped = metadata.dropna(subset=["inc_angle"]).groupby("inc_angle")["is_iceberg"]
        class_counts = grouped.nunique().reindex(repeated.index)
        summary["single_class_repeated_angle_groups"] = int((class_counts == 1).sum())
        summary["mixed_class_repeated_angle_groups"] = int((class_counts > 1).sum())
    return {"summary": summary, "metadata": metadata, "examples": examples}


def check_submission(path: Path, test_ids: list[str]) -> dict:
    submission = pd.read_csv(path, dtype={"id": str})
    if submission.columns.tolist() != ["id", "is_iceberg"]:
        raise ValueError("Submission columns must be id,is_iceberg")
    if submission.isna().any().any() or submission["id"].duplicated().any():
        raise ValueError("Submission contains missing values or duplicate IDs")
    probabilities = pd.to_numeric(submission["is_iceberg"], errors="raise")
    if not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
        raise ValueError("Submission probabilities must be finite values in [0, 1]")
    if len(submission) != len(test_ids) or set(submission["id"]) != set(test_ids):
        raise ValueError("Submission IDs do not match the test set")
    return {"rows": len(submission), "columns": submission.columns.tolist(),
            "test_id_set_match": True, "test_id_order_match": submission["id"].tolist() == test_ids,
            "distinct_example_probabilities": sorted(probabilities.unique().tolist())}


def analyze(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    paths = {name: data_dir / name for name in
             ("train.json", "test.json", "sample_submission.csv")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Extract the actual files into the data directory. Missing: " + ", ".join(missing))
    train = scan_json(paths["train.json"], labeled=True)
    test = scan_json(paths["test.json"], labeled=False)
    submission = check_submission(paths["sample_submission.csv"], test["metadata"]["id"].tolist())
    summary = {
        "train": train["summary"], "test": test["summary"], "submission": submission,
        "overlap": {
            "train_test_ids": len(set(train["metadata"]["id"]) & set(test["metadata"]["id"])),
            "exact_image_fingerprints": len(set(train["metadata"]["image_sha256"]) &
                                            set(test["metadata"]["image_sha256"]))},
        "source_files": {name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                         for name, p in paths.items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name)
                                     for name in ("numpy", "pandas", "matplotlib", "ijson")}},
        "scope": "Data inspection only; no fitted preprocessing, folds, training or model scores.",
    }
    return {"summary": summary, "train": train, "test": test}


def plot_overview(bundle: dict, output: Path) -> None:
    train, test = bundle["train"]["metadata"], bundle["test"]["metadata"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15, 9), facecolor="#F7F8FA")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.3], left=.045, right=.97,
                          top=.82, bottom=.16, hspace=.37, wspace=.24)
    fig.text(.045, .94, "Iceberg project: initial data analysis", fontsize=25,
             weight="bold", color="#172B4D")
    fig.text(.045, .895, f"{len(train):,} labeled training samples  |  {len(test):,} unlabeled test samples"
             "  |  Two 75 x 75 radar bands", color="#52637A")
    chosen = [next((x for x in bundle["train"]["examples"] if x["label"] == label), None)
              for label in (0, 1)]
    images = [x["image"] for x in chosen if x is not None]
    lo, hi = np.percentile(np.stack(images), [1, 99])
    for row, example in enumerate(chosen):
        for band in (0, 1):
            ax = fig.add_subplot(gs[row, band])
            if example is not None:
                ax.imshow(example["image"][band], cmap="gray", vmin=lo, vmax=hi)
                ax.set_xlabel(f"ID: {example['id']}", fontsize=9, color="#52637A")
            else:
                ax.text(.5, .5, "No sample in this class", ha="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{'Ship (0)' if row == 0 else 'Iceberg (1)'} - Band {band + 1}", weight="bold")
    ax = fig.add_subplot(gs[0, 2])
    counts = train["is_iceberg"].value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar(["Ship (0)", "Iceberg (1)"], counts, color=["#3B82A0", "#E49C45"], width=.6)
    ax.set_ylim(0, max(1, counts.max()) * 1.25)
    ax.set_title("Training label distribution", weight="bold", loc="left")
    ax.set_ylabel("Samples")
    ax.set_facecolor("#F7F8FA")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + counts.max() * .025,
                f"{count:,}\n({count / len(train):.1%})", ha="center", va="bottom")
    ax = fig.add_subplot(gs[1, 2])
    all_angles = pd.concat([train["inc_angle"], test["inc_angle"]]).dropna()
    if len(all_angles):
        bins = np.linspace(all_angles.min() - .5, all_angles.max() + .5, 30)
        for frame, color, label in [(train, "#3B82A0", "Train (known angles)"),
                                    (test, "#E49C45", "Test")]:
            if frame["inc_angle"].notna().any():
                ax.hist(frame["inc_angle"].dropna(), bins=bins, density=True,
                        histtype="step", linewidth=2, color=color, label=label)
        ax.legend(frameon=False, fontsize=9)
    ax.set_title("Incidence angle distribution", weight="bold", loc="left")
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Density")
    ax.set_facecolor("#F7F8FA")
    missing_train, missing_test = train["inc_angle"].isna().sum(), test["inc_angle"].isna().sum()
    fig.text(.045, .095, f"Missing angles: train {missing_train}/{len(train)} ({missing_train/len(train):.1%}); "
             f"test {missing_test}/{len(test)}. Submission ID sets match the test data.", color="#172B4D")
    fig.text(.045, .058, "Illustrative samples only. Shared 1st-99th percentile display limits affect "
             "visualization only; raw values are unchanged.", fontsize=9, color="#52637A")
    fig.savefig(output, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_results(bundle: dict, results_dir: Path, metadata_dir: Path) -> None:
    results_dir, metadata_dir = Path(results_dir), Path(metadata_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "data_check.json").write_text(
        json.dumps(bundle["summary"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Row-level metadata stay in ignored outputs/, not in the shared repository.
    for split in ("train", "test"):
        bundle[split]["metadata"].to_csv(metadata_dir / f"{split}_metadata.csv", index=False)
    plot_overview(bundle, results_dir / "overview.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/eda"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()
    bundle = analyze(args.data_dir)
    save_results(bundle, args.results_dir, args.metadata_dir)
    print(json.dumps({key: bundle["summary"][key] for key in ("train", "test", "submission", "overlap")}, indent=2))
    print(f"Saved EDA results to {args.results_dir}; local metadata to {args.metadata_dir}")


if __name__ == "__main__":
    main()
