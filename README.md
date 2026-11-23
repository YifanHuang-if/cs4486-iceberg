# CS4486 Iceberg Classification

## Final version19 source synchronization (2026-11-23)

The final handover's Python source and configuration are now synchronized here,
including `work/v5_submit.py --angle-scale 0.90`. The reported selected submission
has Private Log Loss **0.08708** and Public Log Loss **0.10724**.

- [Final configuration, commands and remaining reproduction limitations](docs/VERSION19_SOURCE_SYNC.md)
- [Version19 settings and source provenance](configs/version19/)
- [Reviewed report and evidence package](https://github.com/YifanHuang-if/cs4486-iceberg/tree/codex/report-reviewed-20261123/report)

This is source/configuration synchronization, not a new training run or a claim of
complete numerical replay. Model weights and missing prediction caches are not
included, and the documented validation isolation issue is not repaired here.
The earlier results below are retained as historical checkpoints.

## Refinement checkpoint (2026-11-16)

The current best CNN ensemble with training-fold-only empirical odds and unlabelled query-group pooling reaches full development five-fold Log Loss **0.13136**, compared with the prior V4 **0.22683** on the same primary split. This is **not an official Kaggle score**; the requested private Score < 0.083 is unverified. The individual local CNN also completed the separate grouped check at **0.19730**. Further architectures are paused and excluded from the best full-fold result.

- [Measured results and limitations](docs/REFINEMENT_RESULTS.md)
- [Plan and adaptive additions](docs/REFINEMENT_PLAN.md)
- [Reproduce, resume and generate predictions](docs/REFINEMENT_RUN.md)
- [All primary comparisons](results/refinement/primary/comparison.csv), [run status](results/refinement/run_status.json) and [verification](results/refinement/verification.json)
- [Current best configuration](results/refinement/current_best.json), [all context studies](results/refinement/studies/comparison.csv) and [second-round plan](docs/REFINEMENT_ROUND2_PLAN.md)

The existing baseline, primary folds and V-series results are preserved.

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

- [Measured results and limitations](docs/STEPS78_RESULTS.md)
- [Primary comparison](results/steps78/primary_comparison.csv) and [ablation contrasts](results/steps78/ablations.csv)
- [Checkpoint verification](results/steps78/verification.json)

Aggregate results, plots, fold scores, epoch histories and verification metadata are versioned under `results/steps78/`.

Use the separate `requirements-steps78.txt` environment for this workflow. Comparisons use a baseline rerun in the same environment; the historical `baseline_v1` score remains separately documented. The companion experiment bundle provides per-sample predictions, the supplementary grouped fold mapping and weights for checkpoint replay. The original training script, primary folds and published baseline outputs are preserved.
