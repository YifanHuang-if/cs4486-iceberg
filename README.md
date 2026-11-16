# CS4486 Iceberg Classification

Competition: https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge

## Steps 9-11: work in progress

This is a **working draft, not the final assessed submission**. Error/subgroup analysis, V4 test inference and Kaggle late scoring have been completed. Public Log Loss is **0.21791** and private Log Loss is **0.20895**. The historical fifth-place scores are 0.08805 and 0.08883 respectively, so the historical Top-5 performance target **has not been reached**. No official late-submission rank is claimed.

- [Current report PDF](report/CS4486_Iceberg_Report_Draft.pdf) and [editable LaTeX source](report/CS4486_Iceberg_Report_Draft.tex)
- [Steps 9-11 handoff, reproduction and remaining work](docs/STEPS9_11_HANDOFF.md)
- [Analysis/inference code](src/finalize.py), [focused tests](tests/test_finalize.py) and [continuation dependencies](requirements-final.txt)
- [Final analysis and scoring evidence](results/final/), including [Kaggle score receipt](results/final/kaggle_score_receipt.json)
- [Submitted prediction CSV](outputs/final/submission.csv) and [per-sample error evidence](outputs/final/oof_error_details.csv)

Remaining work includes further model development toward the performance target, team name, member bios/reflections and independent-work details, category-level hours, leader review/signature, and confirmation of the course's treatment of late submissions. Reported total member hours remain 8, 6 and 7 (21 combined). The raw data and model-weight bundle are still required to replay the full workflow; they are not committed here.

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

- [Measured results and limitations](docs/STEPS78_RESULTS.md)
- [Primary comparison](results/steps78/primary_comparison.csv) and [ablation contrasts](results/steps78/ablations.csv)
- [Checkpoint verification](results/steps78/verification.json)

Aggregate results, plots, fold scores, epoch histories and verification metadata are versioned under `results/steps78/`.

Use the separate `requirements-steps78.txt` environment for this workflow. Comparisons use a baseline rerun in the same environment; the historical `baseline_v1` score remains separately documented. The companion experiment bundle provides per-sample predictions, the supplementary grouped fold mapping and weights for checkpoint replay. The original training script, primary folds and published baseline outputs are preserved.
