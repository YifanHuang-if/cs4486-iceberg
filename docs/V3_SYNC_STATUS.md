# V3 synchronized work status

Updated: 2026-11-16

## Scope

This file describes the completed work that has been synchronized to the V3 pull-request branch. It is **not** a claim that all files on the experiment computer have been uploaded.

## Available on this branch

- `src/v3/`: data preparation, fold-isolated training, evaluation, nested angle selection, shape model, checkpoint replay, isolation tests and error analysis.
- `scripts/run_v3.ps1`: supervised experiment runner.
- `docs/V3_IMPROVEMENT.md`: rationale, leakage audit, validation boundaries and reproduction commands.
- `results/v3/`: the published model, ensemble, validation-audit, nested-selection and submission-format summaries.

## Verified from published result files

| Candidate | Development five-fold OOF log loss |
| --- | ---: |
| Local-normalization CNN + fold-isolated angle fusion | 0.115420 |
| Global-normalization CNN + fold-isolated angle fusion | 0.115521 |
| Two-CNN equal ensemble + fold-isolated angle fusion | 0.108068 |
| Shape/statistics MLP + fold-isolated angle fusion | 0.113870 |
| Three-family diverse ensemble + fold-isolated angle fusion | **0.103447** |

Source: `results/v3/v3_diverse_ensemble.json`.

All values above are *development cross-validation results*, not official Kaggle leaderboard scores.

## Reported outside the pushed result files — not yet verified here

The active experiment conversation reports a subsequent result of **0.10114**. The corresponding final result JSON and all completed pseudo-label-fold metrics have not been verified in this branch. Do not cite 0.10114 as a reproduced result from this repository until its artifacts are shared and checked.

The target of **Log Loss < 0.083** has not been demonstrated by the published files. There is no official Kaggle score in these files.

## Remaining handoff artifacts

- Final completed pseudo-label experiment evaluation and final ensemble metrics
- Fold predictions, training logs and full reproducibility manifest where missing
- Trained checkpoints and submission CSV from the experiment machine, shared separately when suitable (ordinary Git excludes large checkpoints and raw data)
- Replay and consistency checks after the final artifacts arrive

All source and report files should remain in English. Keep this PR open for review rather than merging incomplete claims into `main`.
