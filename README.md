# CS4486 Iceberg Classification

Competition: https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge

## Setup and exploratory analysis

Use Python 3.12 and open this repository in VS Code. Create and activate a local virtual environment, then run:

```powershell
python -m pip install -r requirements-eda.txt
python src/eda.py
```

Download the competition data after accepting its rules. Place train.json, test.json and sample_submission.csv in data/raw/. Results are written to results/eda/. See docs/EDA.md for findings and preprocessing decisions.

## Shared workflow

A owns steps 4-6: data analysis, validation design and the CNN baseline. B and C prepare subsequent improvement experiments. Coordinate shared interfaces and use the same frozen folds for comparisons. Record configurations, results, AI assistance and actual productive hours. Commit and push completed milestones so teammates can pull them.

Raw data, virtual environments and model weights remain local. The course report and presentation each contribute 50 percent. Confirm the Top-5 evaluation interpretation with the instructor.

## Steps 7-8: improvements and ablations

The V-series extends the current `baseline_v1` with paired image flips, a dropout comparison, incidence-angle fusion and a missing-angle indicator. Six preset configurations share the original five-fold mapping. A predefined improvement threshold triggers a separate three-configuration check with disjoint exact-angle groups.

- [Preset experiment plan](docs/STEPS78_PLAN.md)
- [Environment, training, resume and verification](docs/STEPS78_RUN.md)

Running the summarizer generates `docs/STEPS78_RESULTS.md`, `results/steps78/primary_comparison.csv` and `results/steps78/ablations.csv`. These measured outputs are supplied separately with the experiment bundle.

Use the separate `requirements-steps78.txt` environment for this workflow. Comparisons use a baseline rerun in the same environment; the historical `baseline_v1` score remains separately documented. The companion experiment bundle provides measured results, per-sample predictions and weights for checkpoint replay. The original training script, primary folds and published baseline outputs are preserved.
