# Training code: steps 4-8

## Setup

Use a separate Python environment for `requirements-train.txt`; the existing EDA requirements record a different environment. The implementation was tested with Python 3.12 and CPU PyTorch 2.9.1.

```powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-train\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv-train\Scripts\python.exe -m unittest discover -s tests -v
```

Place your original competition JSON files in `data/raw/`. Use the environment's Python for these commands from the repository root:

```text
python -m src.audit_data
python -m src.run_experiments --protocol all --threads 2
python -m src.summarize_experiments
python -m src.verify_results --checkpoints
```

The audit preserves the existing primary folds and generates a separate exact-angle-group sensitivity split. Primary and supplementary protocols remain separate. All image normalization and angle imputation/scaling are fitted on each training fold only. Validation is never augmented. The CNN consumes two radar channels; optional metadata branches receive an imputed/scaled angle and, in one variant, its missing indicator.

## Configurations

| ID | Change |
| --- | --- |
| E0_baseline | Image-only CNN |
| E1_augmentation | Add training-only horizontal/vertical flips |
| E2_regularized | Add dropout and weight decay as a package |
| E3_angle | Add the angle branch |
| E4_angle_missing | Add the missing indicator |
| E5_angle_no_aug | Remove flips from E3 |

Use `--protocol primary` to run only primary configurations. To train only the baseline, run `python -m src.train --config configs/experiments/E0_baseline.json`. All configurations share the epoch cap, optimizer, learning rate, batch size, early-stopping rule and fold seeds. The supplementary batch repeats E0, E3 and E4.

Completed folds resume only when recorded data, fold, configuration and source hashes match. Deliberately changed runs should use a new output directory. An interrupted fold restarts from the beginning.

## Results and handoff

The scripts generate local `outputs/experiments/`, `outputs/grouped/`, `results/models/`, `results/comparison/` and `docs/EXPERIMENT_RESULTS.md`. `src.verify_results` can recompute published metrics without raw data/weights once those result files are available; `--checkpoints` additionally replays every saved model with the original training data. Checkpoints retain fitted preprocessing and held-out IDs.

This code-only branch does not include sample-level result artifacts or newly generated data-derived splits. Those completed artifacts are held separately pending upload authorization. Do not infer a performance score or leaderboard position from the presence of a configuration. Report all measured results, including negative comparisons, and consider subgroup performance before final model selection. Validation-based checkpoint and method selection can make CV optimistic.

Raw data and weights remain excluded from Git. Subsequent detailed error analysis and final test inference/submission are not implemented in this change. Students should review the code and disclose the documented AI assistance.
